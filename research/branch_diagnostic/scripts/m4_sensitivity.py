"""SUPPLEMENTARY (not pre-registered) model-side sensitivity diagnostic, lawam env.

Question: how much do the normalized action chunks move when each conditioning stream is perturbed, holding the
explicit initial noise fixed?  Streams: future tokens (flow head), latent z (LaWM decoder input), VLM context via the
instruction text, and the current image.  Also measures the LaWM decoder's own sensitivity to z.
Outputs: sensitivity.json (per state and aggregated).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_common as mc  # noqa: E402

ALT_INSTRUCTIONS = ["pick up the black bowl next to the plate and place it on the plate", "put the bowl on the stove", "open the top drawer of the cabinet"]


def act_delta(a, b):
    a = np.asarray(a)[..., :7]; b = np.asarray(b)[..., :7]
    return {"maxabs": float(np.max(np.abs(a - b))), "meanabs": float(np.mean(np.abs(a - b))), "xyz_sum_diff_norm": float(np.linalg.norm((a[:, :3] - b[:, :3]).sum(0)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--states", default="t02_ep000,t02_ep001,t02_ep002,t02_ep003,t02_ep004")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = {m["state_id"]: m for m in (json.loads(l) for l in open(run_dir / "reference_manifest.jsonl")) if m.get("qualified")}
    vla, _, norm_stats = mc.load_policy()
    backend = vla.policy_backend
    H, D = int(backend.flow.action_horizon), int(backend.flow.config.action_dim)
    noise = mc.make_initial_noise(101, H, D)
    results = {}
    for sid in args.states.split(","):
        st = manifest[sid]
        o0 = np.load(st["files"]["o0"]); lang = str(o0["lang"][0])
        mo = np.load(run_dir / "conditions" / sid / "model_outputs.npz")
        ex = mc.make_example(o0["primary"], o0["wrist"], o0["state"], lang, norm_stats)
        base = vla.predict_action(examples=[ex], initial_noise=noise, return_diagnostics=True)
        a0 = np.asarray(base["normalized_actions"])[0]; d0 = base["diagnostics"]
        h_t = np.asarray(d0["h_t"]); fut0 = np.asarray(d0["future_used"]); z0 = np.asarray(d0["pred_latent"])
        r = {}
        # --- future-token sensitivity (flow head)
        futs = {"u_A": mo["u_A"], "u_B": mo["u_B"], "h_t(current)": h_t, "zeros": np.zeros_like(fut0), "gauss_matched": np.random.RandomState(0).randn(*fut0.shape).astype(np.float32) * fut0.std() + fut0.mean(),
                "cfg_embedding": backend.flow.cfg_embeddings.detach().float().cpu().numpy(), "u_B_x3_delta": (fut0 + 3.0 * (mo["u_B"] - fut0)).astype(np.float32)}
        r["future"] = {}
        for k, f in futs.items():
            out = vla.predict_action(examples=[ex], initial_noise=noise, future_override=f.astype(np.float32))
            r["future"][k] = {**act_delta(out["normalized_actions"][0], a0), "future_mse_to_pred": float(((f - fut0) ** 2).mean())}
        # --- latent sensitivity (decoder + flow)
        lats = {"z_A": mo["z_A"], "z_B": mo["z_B"], "-z0": -z0, "zeros": np.zeros_like(z0), "z0_x3": 3 * z0, "gauss_unit": np.random.RandomState(1).randn(*z0.shape).astype(np.float32) * float(np.linalg.norm(z0)) / np.sqrt(z0.size)}
        r["latent"] = {}
        for k, z in lats.items():
            out = vla.predict_action(examples=[ex], initial_noise=noise, latent_override=z.astype(np.float32), return_diagnostics=True)
            fu = np.asarray(out["diagnostics"]["h_t1_pred_used"])
            r["latent"][k] = {**act_delta(out["normalized_actions"][0], a0), "decoder_future_mse_to_B0pred": float(((fu - fut0) ** 2).mean()), "decoder_future_mse_to_uA": float(((fu - mo["u_A"]) ** 2).mean()), "decoder_future_mse_to_uB": float(((fu - mo["u_B"]) ** 2).mean())}
        # --- VLM context sensitivity via instruction (everything else fixed)
        r["instruction"] = {}
        for alt in ALT_INSTRUCTIONS:
            ex2 = mc.make_example(o0["primary"], o0["wrist"], o0["state"], alt, norm_stats)
            out = vla.predict_action(examples=[ex2], initial_noise=noise, return_diagnostics=True)
            zz = np.asarray(out["diagnostics"]["pred_latent"])
            r["instruction"][alt] = {**act_delta(out["normalized_actions"][0], a0), "z_l2_change": float(np.linalg.norm(zz - z0)), "future_mse_change": float(((np.asarray(out["diagnostics"]["future_used"]) - fut0) ** 2).mean())}
        # --- current image sensitivity: use the reference o_7^B image as the current observation (state kept)
        refB = np.load(st["files"]["branch_B"]); refA = np.load(st["files"]["branch_A"])
        r["image"] = {}
        for k, img in [("o7_B_as_current", refB["frames"][7]), ("o7_A_as_current", refA["frames"][7])]:
            ex3 = mc.make_example(img, o0["wrist"], o0["state"], lang, norm_stats)
            out = vla.predict_action(examples=[ex3], initial_noise=noise, return_diagnostics=True)
            r["image"][k] = {**act_delta(out["normalized_actions"][0], a0), "z_l2_change": float(np.linalg.norm(np.asarray(out["diagnostics"]["pred_latent"]) - z0))}
        # --- noise sensitivity (reference scale for "what counts as a big change")
        n2 = mc.make_initial_noise(202, H, D)
        out = vla.predict_action(examples=[ex], initial_noise=n2)
        r["noise_seed_change"] = act_delta(out["normalized_actions"][0], a0)
        # --- reference: distance between the reference A and B chunks in normalized space (what a real switch looks like)
        def norm_actions(env_actions):
            s = norm_stats["franka"]["action"]; hi = np.asarray(s["max"]); lo = np.asarray(s["min"])
            a = np.asarray(env_actions)[:, :7].copy(); n = 2 * (a - lo) / (hi - lo) - 1; n[:, 6] = (a[:, 6] > 0).astype(float); return n
        r["reference_A_vs_B_chunks"] = act_delta(norm_actions(refA["actions"]), norm_actions(refB["actions"]))
        r["B0_vs_reference_A"] = act_delta(a0, norm_actions(refA["actions"])); r["B0_vs_reference_B"] = act_delta(a0, norm_actions(refB["actions"]))
        results[sid] = r
        print(sid, "future u_B:", r["future"]["u_B"]["maxabs"], "zeros:", r["future"]["zeros"]["maxabs"], "cfg:", r["future"]["cfg_embedding"]["maxabs"], "| latent z_B:", r["latent"]["z_B"]["maxabs"], "-z0:", r["latent"]["-z0"]["maxabs"],
              "| instr:", [round(v["maxabs"], 3) for v in r["instruction"].values()], "| image o7B:", r["image"]["o7_B_as_current"]["maxabs"], "| noise:", r["noise_seed_change"]["maxabs"], "| refA-refB:", r["reference_A_vs_B_chunks"]["maxabs"], flush=True)
    # aggregate
    agg = {}
    for group in ["future", "latent", "instruction", "image"]:
        agg[group] = {}
        for k in results[next(iter(results))][group]:
            agg[group][k] = {m: float(np.mean([results[s][group][k][m] for s in results])) for m in ["maxabs", "meanabs", "xyz_sum_diff_norm"]}
    agg["noise_seed_change"] = {m: float(np.mean([results[s]["noise_seed_change"][m] for s in results])) for m in ["maxabs", "meanabs", "xyz_sum_diff_norm"]}
    agg["reference_A_vs_B_chunks"] = {m: float(np.mean([results[s]["reference_A_vs_B_chunks"][m] for s in results])) for m in ["maxabs", "meanabs", "xyz_sum_diff_norm"]}
    agg["decoder_sensitivity"] = {k: {m: float(np.mean([results[s]["latent"][k][m] for s in results])) for m in ["decoder_future_mse_to_B0pred", "decoder_future_mse_to_uA", "decoder_future_mse_to_uB"]} for k in results[next(iter(results))]["latent"]}
    mc.write_json(run_dir / "sensitivity.json", {"note": "supplementary, not pre-registered; seed 101 noise fixed", "per_state": results, "aggregate": agg})
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
