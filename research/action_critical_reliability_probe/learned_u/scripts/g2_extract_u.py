"""Step 2 (model side, `lawam` env): extract the Learned-U dataset for the new tasks.

Per state it stores exactly what a deployable predictor is allowed to see, plus the oracle label:

  inputs (available BEFORE the chunk is executed)   h_t [256,768], H_pred [256,768], z [1,32]
  label  (oracle, offline only)                     U_j from H_real = features(o_{t+7})
  floors                                            U between two renderings of the same physical state

The feature encoder and the U definition are IMPORTED from m3_uncertainty.py rather than reimplemented, so
they are literally the same functions the prior round used. The expensive parts of m3 (the 256-token error
decomposition and the consequence sampling) are deliberately not re-run here: they are not needed to learn
U, and Steps 5-7 run on the prior round's holdout-C states where they already exist.

H_real never enters a policy input: the example is built from o_t only, exactly as in m3.

Outputs: <run_dir>/u_dataset/<state_id>.npz and <run_dir>/u_dataset_records.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
import probe_common as pc  # noqa: E402
import model_common as mc  # noqa: E402
import probe_model as pm  # noqa: E402
from m3_uncertainty import lam_features, per_token_u  # noqa: E402  (same functions as the prior round)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(run_dir / "states_manifest.jsonl")]
    futures = {json.loads(l)["state_id"]: json.loads(l) for l in open(run_dir / "futures_manifest.jsonl")}
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = run_dir / "u_dataset"
    out_dir.mkdir(exist_ok=True)
    rec_path = run_dir / "u_dataset_records.jsonl"
    done = {json.loads(l)["state_id"] for l in open(rec_path)} if rec_path.exists() else set()

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    backend = runner.backend

    for m in manifest:
        sid = m["state_id"]
        if sid in done:
            continue
        t0 = time.time()
        d = np.load(m["state_npz"], allow_pickle=True)
        fz = np.load(futures[sid]["futures_npz"])

        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        assert not [k for k in ex if "future" in k.lower() or "real" in k.lower()], "future leaked into the policy input"
        ctx = runner.prepare(ex)

        h_pred = ctx.h_t1_pred[0].float().cpu().numpy().astype(np.float64)
        h_t = ctx.h_t[0].float().cpu().numpy().astype(np.float64)
        z = ctx.z.float().cpu().numpy().astype(np.float32)

        h_real = lam_features(backend, d["future_primary"])[0].float().cpu().numpy().astype(np.float64)
        rep1 = lam_features(backend, fz["replay1_t7"])[0].float().cpu().numpy().astype(np.float64)
        rep2 = lam_features(backend, fz["replay2_t7"])[0].float().cpu().numpy().astype(np.float64)

        U = per_token_u(h_pred, h_real)
        U_floor = per_token_u(rep1, rep2)
        U_trivial = per_token_u(h_t, h_real)

        np.savez_compressed(
            out_dir / f"{sid}.npz",
            h_t=h_t.astype(np.float16), h_t1_pred=h_pred.astype(np.float16), z=z,
            **{f"U_{k}": v.astype(np.float32) for k, v in U.items()},
            **{f"Ufloor_{k}": v.astype(np.float32) for k, v in U_floor.items()},
            **{f"Utrivial_{k}": v.astype(np.float32) for k, v in U_trivial.items()},
        )
        rec = {
            "state_id": sid, "suite": m["suite"], "task_id": m["task_id"], "phase": m["phase"],
            "episode_idx": m["episode_idx"], "step": m["step"], "episode_success": m["episode_success"],
            "instruction": m["instruction"],
            "U_mse_global": float(U["U_mse"].mean()),
            "U_l2_stats": pc.stats(U["U_l2"]),
            "U_floor_mse_global": float(U_floor["U_mse"].mean()),
            "U_trivial_mse_global": float(U_trivial["U_mse"].mean()),
            "snr_U_over_render_floor": float(U["U_mse"].mean() / max(U_floor["U_mse"].mean(), 1e-12)),
            "token_norm_stats": pc.stats(np.linalg.norm(h_pred, axis=-1)),
            "npz": str(out_dir / f"{sid}.npz"),
            "wall_sec": round(time.time() - t0, 2),
        }
        pc.append_jsonl(rec_path, rec)
        print(f"{sid}: U={rec['U_mse_global']:.4f} floor={rec['U_floor_mse_global']:.4f} "
              f"snr={rec['snr_U_over_render_floor']:.1f} trivial={rec['U_trivial_mse_global']:.4f} "
              f"({rec['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
