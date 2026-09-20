"""Produce the action chunks whose EXECUTION consequences Step 4 Level 2 compares (`lawam` env).

Per state and per flow-noise seed:
  A_pred        = pi(H_pred)                       (what the policy actually does)
  A_realFuture  = pi(H_real)                       (oracle: the future the rollout really reached)
  A_topU        = pi(H_pred with the top-quartile-U tokens replaced by their real values)
  A_topUS       = pi(H_pred with the top-quartile-(U*S) tokens replaced by their real values)

The last two exist to ask a question the correlations cannot answer: if only the tokens that a reliability
side-channel would flag were corrected, how much of the executed consequence would that recover? Same token
budget (64 of 256) for both, so they are directly comparable.

Nothing is trained; H_real is used only as an offline oracle.
Output: <run_dir>/consequence_chunks/<state_id>.npz  (normalized chunks; the simulator side unnormalises)
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
TOP_Q = 0.75
EPS_MID_IDX = 1
METRIC = "l2_norm_all7"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(run_dir / "states_manifest.jsonl")]
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = run_dir / "consequence_chunks"
    out_dir.mkdir(exist_ok=True)

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noises = {s: pm.make_noise(s, H, Ad, 1) for s in NOISE_SEEDS}

    for m in manifest:
        sid = m["state_id"]
        out_path = out_dir / f"{sid}.npz"
        if out_path.exists():
            continue
        t0 = time.time()
        d = np.load(m["state_npz"], allow_pickle=True)
        un = np.load(run_dir / "uncertainty" / f"{sid}.npz")
        sn = np.load(run_dir / "sensitivity" / f"{sid}.npz")
        metrics = [str(x) for x in sn["metrics"]]
        mi = metrics.index(METRIC)

        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred
        N = ctx.n_tokens
        h_real = torch.as_tensor(un["h_real"].astype(np.float32), device=Hp.device, dtype=Hp.dtype)

        U = un["U_U_l2"].astype(np.float64)
        S = sn["S_norm"][EPS_MID_IDX].mean(axis=0)[:, mi].astype(np.float64)
        k = int(round((1 - TOP_Q) * N))
        top_u = np.argsort(-U)[:k]
        top_us = np.argsort(-(U * S))[:k]

        h_topU = Hp[0].clone(); h_topU[top_u] = h_real[top_u]
        h_topUS = Hp[0].clone(); h_topUS[top_us] = h_real[top_us]

        variants = {"A_pred": Hp[0], "A_realFuture": h_real,
                    "A_topU_corrected": h_topU, "A_topUS_corrected": h_topUS}
        out = {"token_budget": np.asarray([k]),
               "top_u_tokens": top_u.astype(np.int32), "top_us_tokens": top_us.astype(np.int32),
               "overlap_topU_topUS": np.asarray([len(set(top_u.tolist()) & set(top_us.tolist()))])}
        stack = torch.stack(list(variants.values()), dim=0)
        for s in NOISE_SEEDS:
            a = runner.actions(ctx, stack, torch.from_numpy(np.repeat(noises[s], stack.shape[0], axis=0)))
            for i, name in enumerate(variants):
                out[f"{name}_seed{s}"] = a[i].astype(np.float32)
        np.savez_compressed(out_path, **out)
        print(f"{sid}: budget={k}/{N} overlap(topU,topUS)={out['overlap_topU_topUS'][0]} "
              f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
