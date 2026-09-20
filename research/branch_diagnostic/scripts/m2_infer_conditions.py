"""Step 4/5 model side (lawam env): for each qualified state produce
  teacher z_A / z_B (official distill path backend._run_lam_teacher on frames (o_0, o_7^X)),
  u_A / u_B (LAM vision features of o_7^X, same call as h_t),
  B0 (3 noise seeds), B1 (z_X override, 3 seeds), B2 (u_X override, 3 seeds) normalized action chunks + diagnostics.
Nothing here executes in the simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_common as mc  # noqa: E402

NOISE_SEEDS = [101, 202, 303]


def lam_features(backend, img_uint8_hwc: np.ndarray) -> np.ndarray:
    """Same path as LatentWorldPolicyBackend._run_shared_encoding_core for the current image (bf16 autocast)."""
    vid = mc.imagenet_video_tensor([img_uint8_hwc])  # [1,1,3,256,256]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = backend.lam.extract_vision_features(vid[:, 0])  # [1,1,256,768]
    return feats[:, 0].detach().float().cpu().numpy()


def teacher_latent(backend, frames: list[np.ndarray]) -> np.ndarray:
    vid = mc.imagenet_video_tensor(frames)  # [1,T,3,256,256]
    emb = torch.tensor([mc.EMBODIMENT_ID], device="cuda", dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = backend._run_lam_teacher(primary_video=vid, embodiment_id=emb)
    return z.detach().float().cpu().numpy()  # [1,1,32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--states", type=str, default="all")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(run_dir / "reference_manifest.jsonl")]
    states = [m for m in manifest if m.get("qualified")]
    if args.states != "all":
        keep = set(args.states.split(","))
        states = [m for m in states if m["state_id"] in keep]
    out_root = run_dir / "conditions"; out_root.mkdir(exist_ok=True)

    vla, model_config, norm_stats = mc.load_policy()
    backend = vla.policy_backend
    # The deployed policy is cast to bf16 (server --use_bf16). The LAM VAE bottleneck (`lam.vq`) is NOT used by the
    # policy inference path; the official training teacher path runs it with autocast disabled on float32 nodes.
    # Casting only `lam.vq` back to float32 reproduces the training precision for the teacher and leaves the policy
    # path (vision encoder / decoder / flow) untouched.
    backend.lam.vq.float()
    H, D = int(backend.flow.action_horizon), int(backend.flow.config.action_dim)
    noises = {s: mc.make_initial_noise(s, H, D) for s in NOISE_SEEDS}
    np.savez_compressed(out_root / "initial_noise.npz", **{f"seed{s}": n for s, n in noises.items()})
    mc.write_json(out_root / "initial_noise_sha256.json", {f"seed{s}": mc.sha256(n) for s, n in noises.items()})

    index = []
    for st in states:
        sid = st["state_id"]; t0 = time.time()
        sdir = out_root / sid; sdir.mkdir(exist_ok=True)
        o0 = np.load(st["files"]["o0"]); lang = str(o0["lang"][0])
        ex = mc.make_example(o0["primary"], o0["wrist"], o0["state"], lang, norm_stats)
        ref = {b: np.load(st["files"][f"branch_{b}"]) for b in "AB"}
        o7 = {b: ref[b]["frames"][7] for b in "AB"}
        o7_replay = {b: ref[b]["replay_frames"][6] for b in "AB"}
        o8 = {b: ref[b]["frames"][8] for b in "AB"}
        out = {"o0_primary": o0["primary"]}
        # teacher latents + future features
        z = {b: teacher_latent(backend, [o0["primary"], o7[b]]) for b in "AB"}
        u = {b: lam_features(backend, o7[b]) for b in "AB"}
        u_replay = {b: lam_features(backend, o7_replay[b]) for b in "AB"}
        u8 = {b: lam_features(backend, o8[b]) for b in "AB"}
        h_t_direct = lam_features(backend, o0["primary"])
        for b in "AB":
            out[f"z_{b}"] = z[b]; out[f"u_{b}"] = u[b]; out[f"u_{b}_replay"] = u_replay[b]; out[f"u8_{b}"] = u8[b]
        out["h_t_direct"] = h_t_direct
        # optional teacher variant: z from (o_0, o_8) to document sensitivity to the future index
        for b in "AB":
            out[f"z8_{b}"] = teacher_latent(backend, [o0["primary"], o8[b]])
        records = []
        for s in NOISE_SEEDS:
            n = noises[s]
            r0 = vla.predict_action(examples=[ex], initial_noise=n, return_diagnostics=True)
            d0 = r0["diagnostics"]
            out[f"B0_seed{s}_actions"] = np.asarray(r0["normalized_actions"])[0]
            if s == NOISE_SEEDS[0]:
                out["B0_pred_latent"] = np.asarray(d0["pred_latent"]); out["B0_h_t"] = np.asarray(d0["h_t"]); out["B0_h_t1_pred"] = np.asarray(d0["h_t1_pred"])
                out["B0_noise_used_seed101"] = np.asarray(d0["initial_noise"])
            records.append({"condition": "B0", "input_branch": None, "seed": s, "key": f"B0_seed{s}_actions"})
            for b in "AB":
                r1 = vla.predict_action(examples=[ex], initial_noise=n, latent_override=z[b], return_diagnostics=True)
                out[f"B1_{b}_seed{s}_actions"] = np.asarray(r1["normalized_actions"])[0]
                if s == NOISE_SEEDS[0]:
                    out[f"B1_{b}_h_t1_pred_used"] = np.asarray(r1["diagnostics"]["h_t1_pred_used"])
                    assert bool(r1["diagnostics"]["latent_override_applied"])
                records.append({"condition": "B1", "input_branch": b, "seed": s, "key": f"B1_{b}_seed{s}_actions"})
                r2 = vla.predict_action(examples=[ex], initial_noise=n, future_override=u[b], return_diagnostics=True)
                out[f"B2_{b}_seed{s}_actions"] = np.asarray(r2["normalized_actions"])[0]
                if s == NOISE_SEEDS[0]:
                    assert bool(r2["diagnostics"]["future_override_applied"])
                    assert np.allclose(np.asarray(r2["diagnostics"]["future_used"]), u[b].astype(np.asarray(r2["diagnostics"]["future_used"]).dtype), atol=1e-2)
                records.append({"condition": "B2", "input_branch": b, "seed": s, "key": f"B2_{b}_seed{s}_actions"})
        np.savez_compressed(sdir / "model_outputs.npz", **out)
        # small scalar diagnostics
        z0 = out["B0_pred_latent"].reshape(-1)
        def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        diag = {"state_id": sid, "z0_norm": float(np.linalg.norm(z0)), "zA_norm": float(np.linalg.norm(z["A"])), "zB_norm": float(np.linalg.norm(z["B"])),
                "cos_z0_zA": cos(z0, z["A"].reshape(-1)), "cos_z0_zB": cos(z0, z["B"].reshape(-1)), "cos_zA_zB": cos(z["A"].reshape(-1), z["B"].reshape(-1)),
                "l2_zA_zB": float(np.linalg.norm(z["A"] - z["B"])), "l2_z0_zA": float(np.linalg.norm(z0 - z["A"].reshape(-1))), "l2_z0_zB": float(np.linalg.norm(z0 - z["B"].reshape(-1))),
                "l2_z8A_zA": float(np.linalg.norm(out["z8_A"] - z["A"])), "l2_z8B_zB": float(np.linalg.norm(out["z8_B"] - z["B"])),
                "mse_uA_uB": float(((u["A"] - u["B"]) ** 2).mean()), "mse_uA_uA_replay": float(((u["A"] - u_replay["A"]) ** 2).mean()), "mse_uB_uB_replay": float(((u["B"] - u_replay["B"]) ** 2).mean()),
                "mse_ht_uA": float(((h_t_direct - u["A"]) ** 2).mean()), "mse_ht_uB": float(((h_t_direct - u["B"]) ** 2).mean()),
                "mse_ht_direct_vs_B0_h_t": float(((h_t_direct - out["B0_h_t"]) ** 2).mean()),
                "mse_B0pred_uA": float(((out["B0_h_t1_pred"] - u["A"]) ** 2).mean()), "mse_B0pred_uB": float(((out["B0_h_t1_pred"] - u["B"]) ** 2).mean()),
                "mse_B1Apred_uA": float(((out["B1_A_h_t1_pred_used"] - u["A"]) ** 2).mean()), "mse_B1Apred_uB": float(((out["B1_A_h_t1_pred_used"] - u["B"]) ** 2).mean()),
                "mse_B1Bpred_uA": float(((out["B1_B_h_t1_pred_used"] - u["A"]) ** 2).mean()), "mse_B1Bpred_uB": float(((out["B1_B_h_t1_pred_used"] - u["B"]) ** 2).mean()),
                "actions_maxabs_change_B1A_vs_B0_seed101": float(np.max(np.abs(out["B1_A_seed101_actions"][:, :7] - out["B0_seed101_actions"][:, :7]))),
                "actions_maxabs_change_B1B_vs_B0_seed101": float(np.max(np.abs(out["B1_B_seed101_actions"][:, :7] - out["B0_seed101_actions"][:, :7]))),
                "actions_maxabs_change_B2A_vs_B0_seed101": float(np.max(np.abs(out["B2_A_seed101_actions"][:, :7] - out["B0_seed101_actions"][:, :7]))),
                "actions_maxabs_change_B2B_vs_B0_seed101": float(np.max(np.abs(out["B2_B_seed101_actions"][:, :7] - out["B0_seed101_actions"][:, :7]))),
                "actions_maxabs_change_B2A_vs_B2B_seed101": float(np.max(np.abs(out["B2_A_seed101_actions"][:, :7] - out["B2_B_seed101_actions"][:, :7]))),
                "records": records, "wall_sec": time.time() - t0}
        mc.write_json(sdir / "model_diag.json", diag)
        index.append({"state_id": sid, "outputs": str(sdir / "model_outputs.npz"), "diag": str(sdir / "model_diag.json")})
        print(f"{sid}: z0n={diag['z0_norm']:.2f} zAn={diag['zA_norm']:.2f} cos(zA,zB)={diag['cos_zA_zB']:.3f} mse(uA,uB)={diag['mse_uA_uB']:.4f} floor={diag['mse_uA_uA_replay']:.5f} "
              f"B0pred->uA/uB={diag['mse_B0pred_uA']:.4f}/{diag['mse_B0pred_uB']:.4f} dA(B1A)={diag['actions_maxabs_change_B1A_vs_B0_seed101']:.3f} dA(B2B)={diag['actions_maxabs_change_B2B_vs_B0_seed101']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    mc.write_json(out_root / "index.json", {"states": index, "noise_seeds": NOISE_SEEDS, "num_states": len(index), "model_forward_calls_per_state": 3 * 5})


if __name__ == "__main__":
    main()
