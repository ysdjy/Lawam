"""G3 pilot step (model side): H_pred, H_real and the two action chunks whose executed difference IS the
future-error treatment.

For every selected state of every condition:
    A_pred       = pi(H_pred)        the policy's own chunk
    A_realFuture = pi(H_real)        H_pred replaced by the encoding of the REAL (and, under families A/B,
                                     equally perturbed) future frame o_{t+7}
at three frozen flow-noise seeds, everything else identical.

The U label can not be misaligned: `future_primary` in the state file is taken from the same perturbed
frame stream the policy consumed, because the perturbation is applied where frames are recorded.

No S, no ranking, no token repair in the pilot. Nothing is trained.
Output: <run_dir>/treatment_chunks/<condition>/<state_id>.npz + treatment_features.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
import probe_common as pc  # noqa: E402
import model_common as mc  # noqa: E402
import probe_model as pm  # noqa: E402
from m3_uncertainty import lam_features, per_token_u  # noqa: E402  (frozen definitions)

NOISE_SEEDS = [101, 202, 303]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    states = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        states = states[: args.limit]
    out_root = run_dir / "treatment_chunks"
    out_root.mkdir(exist_ok=True)
    rec_path = run_dir / "treatment_features.jsonl"
    done = {json.loads(l)["state_key"] for l in open(rec_path)} if rec_path.exists() else set()

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    backend = runner.backend
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noises = {s: pm.make_noise(s, H, Ad, 1) for s in NOISE_SEEDS}

    for st in states:
        key = f"{st['condition']}/{st['state_id']}"
        if key in done:
            continue
        t0 = time.time()
        d = np.load(st["state_npz"], allow_pickle=True)
        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        assert not [k for k in ex if "future" in k.lower()], "future leaked into the policy input"
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred

        h_pred = Hp[0].float().cpu().numpy().astype(np.float64)
        h_t = ctx.h_t[0].float().cpu().numpy().astype(np.float64)
        # the future frame comes from the SAME perturbed stream as the current frame
        h_real_t = lam_features(backend, d["future_primary"])
        h_real = h_real_t[0].float().cpu().numpy().astype(np.float64)

        U = per_token_u(h_pred, h_real)
        D = np.linalg.norm(h_pred - h_t, axis=-1)
        G = np.linalg.norm(h_real - h_t, axis=-1)

        stack = torch.cat([Hp, h_real_t.to(dtype=Hp.dtype)], dim=0)
        out = {"D": D.astype(np.float32), "U_l2": U["U_l2"].astype(np.float32),
               "G": G.astype(np.float32),
               "h_t1_pred": h_pred.astype(np.float16), "h_t": h_t.astype(np.float16),
               "h_real": h_real.astype(np.float16)}
        for s in NOISE_SEEDS:
            a = runner.actions(ctx, stack, torch.from_numpy(np.repeat(noises[s], 2, axis=0)))
            out[f"A_pred_seed{s}"] = a[0].astype(np.float32)
            out[f"A_realFuture_seed{s}"] = a[1].astype(np.float32)
        cdir = out_root / st["condition"]
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cdir / f"{st['state_id']}.npz", **out)

        pc.append_jsonl(rec_path, {
            "state_key": key, "condition": st["condition"], "family": st["family"],
            "severity": st["severity"], "suite": st["suite"], "task_id": st["task_id"],
            "episode_idx": st["episode_idx"], "phase": st["phase"], "step": st["step"],
            "episode_success": st["episode_success"],
            "U_mse_global": float(U["U_mse"].mean()), "U_l2_median": float(np.median(U["U_l2"])),
            "D_median": float(np.median(D)), "G_median": float(np.median(G)),
            "action_maxabs_pred_vs_real": float(np.max(np.abs(
                out["A_pred_seed101"][:, :7] - out["A_realFuture_seed101"][:, :7]))),
            "npz": str(cdir / f"{st['state_id']}.npz"), "wall_sec": round(time.time() - t0, 2)})
        print(f"{key}: U={float(U['U_mse'].mean()):.4f} D={np.median(D):.2f} G={np.median(G):.2f} "
              f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
