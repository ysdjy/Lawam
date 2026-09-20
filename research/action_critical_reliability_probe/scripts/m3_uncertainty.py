"""Step 2 + Step 4 Level 1 (`lawam` env): oracle future-prediction unreliability U and the action
consequence of the realised prediction error.

For every state:
  H_real      = lam.extract_vision_features(o_{t+7})   -- identical encoder / flip / resize / normalisation
                as h_t, applied to the observation the undisturbed rollout actually reached after executing
                7 of the 8 chunk actions (the training future index). Used ONLY as an offline label.
  U_j         = per-token MSE, cosine distance and normalised MSE between H_pred_j and H_real_j
  floors      = the same quantities computed between two renderings of the *same* physical state
                (replay1 vs replay2) and between the online frame and its replay
  A_realFuture= pi(H_real) with the observation, instruction, h_vlm, z, h_t and flow noise unchanged
  E_action    = D(A_pred, A_realFuture)                -- the downstream consequence of the whole error
  per-token error-direction decomposition (protocol amendment A4):
                for each token j, replace ONLY token j by H_real_j and measure the action deviation.
                This is the quantity U x S is supposed to predict.

No training, no weight change, no simulator. H_real never enters a policy input.

Outputs: <run_dir>/uncertainty_records.jsonl, <run_dir>/uncertainty/<state_id>.npz
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
import probe_common as pc  # noqa: E402
import model_common as mc  # noqa: E402
import probe_model as pm  # noqa: E402

NOISE_SEEDS = [101, 202, 303]
BATCH = 32
METRICS = pm.DISTANCE_KEYS


def lam_features(backend, img_uint8_hwc: np.ndarray) -> torch.Tensor:
    """Exactly the h_t path of `_run_shared_encoding_core` applied to one frame."""
    vid = mc.imagenet_video_tensor([img_uint8_hwc])       # [1,1,3,256,256]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = backend.lam.extract_vision_features(vid[:, 0])
    return feats[:, 0]                                     # [1,256,768]


def per_token_u(h_pred: np.ndarray, h_real: np.ndarray) -> dict[str, np.ndarray]:
    diff = h_real - h_pred
    mse = (diff ** 2).mean(axis=-1)
    cos = (h_pred * h_real).sum(-1) / (np.linalg.norm(h_pred, axis=-1) * np.linalg.norm(h_real, axis=-1) + 1e-12)
    return {
        "U_mse": mse,
        "U_cosine_dist": 1.0 - cos,
        "U_l2": np.linalg.norm(diff, axis=-1),
        "U_normalized_mse": mse / np.maximum((h_pred ** 2).mean(axis=-1), 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    ap.add_argument("--states", default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(run_dir / "states_manifest.jsonl")]
    futures = {json.loads(l)["state_id"]: json.loads(l) for l in open(run_dir / "futures_manifest.jsonl")}
    if args.states != "all":
        keep = set(args.states.split(","))
        manifest = [m for m in manifest if m["state_id"] in keep]
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = run_dir / "uncertainty"
    out_dir.mkdir(exist_ok=True)
    rec_path = run_dir / "uncertainty_records.jsonl"
    done = {json.loads(l)["state_id"] for l in open(rec_path)} if rec_path.exists() else set()

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    backend = runner.backend
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noises = {s: pm.make_noise(s, H, Ad, 1) for s in NOISE_SEEDS}

    for m in manifest:
        sid = m["state_id"]
        if sid in done:
            continue
        t0 = time.time()
        d = np.load(m["state_npz"], allow_pickle=True)
        fz = np.load(futures[sid]["futures_npz"])
        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred
        N, D = ctx.n_tokens, ctx.feature_dim

        h_pred = Hp[0].float().cpu().numpy().astype(np.float64)
        h_t = ctx.h_t[0].float().cpu().numpy().astype(np.float64)
        feats = {
            "online_t7": lam_features(backend, d["future_primary"]),
            "replay1_t7": lam_features(backend, fz["replay1_t7"]),
            "replay2_t7": lam_features(backend, fz["replay2_t7"]),
            "online_t8": lam_features(backend, d["future_primary_t8"]),
        }
        f_np = {k: v[0].float().cpu().numpy().astype(np.float64) for k, v in feats.items()}
        h_real = f_np["online_t7"]

        U = per_token_u(h_pred, h_real)
        U_floor_render = per_token_u(f_np["replay1_t7"], f_np["replay2_t7"])      # same physics, 2 renders
        U_floor_online_replay = per_token_u(f_np["online_t7"], f_np["replay1_t7"])
        U_vs_t8 = per_token_u(h_pred, f_np["online_t8"])                          # index-sensitivity check
        U_trivial = per_token_u(h_t, h_real)     # "predict no change" baseline: how much does h_t alone explain

        # ---------------- Step 4 Level 1: action consequence of the whole prediction error
        h_real_t = torch.as_tensor(h_real, device=Hp.device, dtype=Hp.dtype).unsqueeze(0)
        cons = {}
        for s in NOISE_SEEDS:
            stack = torch.cat([Hp, h_real_t], dim=0)
            a = runner.actions(ctx, stack, torch.from_numpy(np.repeat(noises[s], 2, axis=0)))
            cons[f"E_action_seed{s}"] = pm.action_distance(a[1], a[0])
            if s == NOISE_SEEDS[0]:
                a0 = a[0]
                a_real = a[1]

        # ---------------- per-token error-direction decomposition (amendment A4)
        dev = {k: np.zeros(N) for k in METRICS}
        for start in range(0, N, BATCH - 1):
            toks = list(range(start, min(start + BATCH - 1, N)))
            rows = [Hp[0]]
            for j in toks:
                f = Hp[0].clone()
                f[j] = h_real_t[0, j]
                rows.append(f)
            fut = torch.stack(rows, dim=0)
            a = runner.actions(ctx, fut, torch.from_numpy(np.repeat(noises[NOISE_SEEDS[0]], fut.shape[0], axis=0)))
            dd = pm.action_distance_batch(a[1:], a[0])
            for k in METRICS:
                dev[k][toks] = dd[k]

        # signed/vector version for the top tokens: does the sum of single-token effects reproduce the total?
        mi = METRICS.index("l2_norm_all7")
        sum_single = float(dev["l2_norm_all7"].sum())
        total = float(cons[f"E_action_seed{NOISE_SEEDS[0]}"]["l2_norm_all7"])

        np.savez_compressed(
            out_dir / f"{sid}.npz",
            **{f"U_{k}": v.astype(np.float32) for k, v in U.items()},
            **{f"Ufloor_render_{k}": v.astype(np.float32) for k, v in U_floor_render.items()},
            **{f"Ufloor_online_replay_{k}": v.astype(np.float32) for k, v in U_floor_online_replay.items()},
            **{f"Ut8_{k}": v.astype(np.float32) for k, v in U_vs_t8.items()},
            **{f"Utrivial_{k}": v.astype(np.float32) for k, v in U_trivial.items()},
            **{f"errdev_{k}": v.astype(np.float32) for k, v in dev.items()},
            h_real=h_real.astype(np.float16), baseline_action=a0.astype(np.float32),
            action_real_future=a_real.astype(np.float32), metrics=np.asarray(METRICS),
        )

        rec = {
            "state_id": sid, "suite": m["suite"], "task_id": m["task_id"], "phase": m["phase"],
            "episode_idx": m["episode_idx"], "step": m["step"],
            "U_mse_stats": pc.stats(U["U_mse"]),
            "U_cosine_stats": pc.stats(U["U_cosine_dist"]),
            "U_mse_global": float(U["U_mse"].mean()),
            "U_floor_render_mse_global": float(U_floor_render["U_mse"].mean()),
            "U_floor_online_replay_mse_global": float(U_floor_online_replay["U_mse"].mean()),
            "U_vs_t8_mse_global": float(U_vs_t8["U_mse"].mean()),
            "U_trivial_h_t_mse_global": float(U_trivial["U_mse"].mean()),
            "snr_U_over_render_floor": float(U["U_mse"].mean() / max(U_floor_render["U_mse"].mean(), 1e-12)),
            "U_beats_trivial": bool(U["U_mse"].mean() < U_trivial["U_mse"].mean()),
            "E_action": {k: v for k, v in cons.items()},
            "errdev_sum_over_tokens_l2": sum_single,
            "errdev_max_token_l2": float(dev["l2_norm_all7"].max()),
            "errdev_median_token_l2": float(np.median(dev["l2_norm_all7"])),
            "E_action_total_l2": total,
            "superposition_ratio_sum_over_total": float(sum_single / max(total, 1e-30)),
            "wall_sec": round(time.time() - t0, 1),
        }
        pc.append_jsonl(rec_path, rec)
        print(f"{sid}: U={rec['U_mse_global']:.4f} (floor {rec['U_floor_render_mse_global']:.4f}, "
              f"trivial h_t {rec['U_trivial_h_t_mse_global']:.4f}) E_action={total:.4f} "
              f"sum_single={sum_single:.4f} ({rec['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
