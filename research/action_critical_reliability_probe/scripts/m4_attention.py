"""Attention / relevance baseline (`lawam` env) — the control that decides whether a sensitivity
measurement is needed at all.

For each state, records the cross-attention mass that the action tokens put on each of the 256 H_pred
tokens, in the 4 DiT blocks that read the image half (blocks 2/6/10/14), averaged over heads, action-token
queries and the 10 flow integration steps. Also records, for reference, the attention on the h_t half.

Equivalence guard: the recording processor recomputes attention explicitly instead of using SDPA, so the
script first checks that the action chunk produced with the recorder installed matches the default path.

Outputs: <run_dir>/attention_records.jsonl, <run_dir>/attention/<state_id>.npz
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

NOISE_SEED = 101
EQUIV_TOL = 5e-3   # explicit-softmax vs SDPA differ only by float32 reduction order in bf16 arithmetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    ap.add_argument("--states", default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(run_dir / "states_manifest.jsonl")]
    if args.states != "all":
        keep = set(args.states.split(","))
        manifest = [m for m in manifest if m["state_id"] in keep]
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = run_dir / "attention"
    out_dir.mkdir(exist_ok=True)
    rec_path = run_dir / "attention_records.jsonl"
    done = {json.loads(l)["state_id"] for l in open(rec_path)} if rec_path.exists() else set()

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noise = torch.from_numpy(pm.make_noise(NOISE_SEED, H, Ad, 1))

    for m in manifest:
        sid = m["state_id"]
        if sid in done:
            continue
        t0 = time.time()
        d = np.load(m["state_npz"], allow_pickle=True)
        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        ctx = runner.prepare(ex)
        N = ctx.n_tokens

        a_ref = runner.actions(ctx, ctx.h_t1_pred, noise)[0]
        with pm.record_attention(runner.flow) as rec:
            a_rec = runner.actions(ctx, ctx.h_t1_pred, noise)[0]
            att = rec.stacked()             # [n_blocks*steps, 1, kv_len]
            blocks = list(rec.blocks)
        equiv = float(np.max(np.abs(a_rec - a_ref)))

        att = att[:, 0, :]                                        # [calls, kv_len]
        att_h_t = att[:, :N]
        att_future = att[:, N:2 * N]
        n_calls = att.shape[0]
        steps = n_calls // len(blocks)
        per_block = att_future.reshape(steps, len(blocks), N) if n_calls == steps * len(blocks) else None

        a_fut_mean = att_future.mean(axis=0)                      # [256] mean over blocks and flow steps
        a_ht_mean = att_h_t.mean(axis=0)
        np.savez_compressed(out_dir / f"{sid}.npz",
                            attention_future=a_fut_mean.astype(np.float32),
                            attention_h_t=a_ht_mean.astype(np.float32),
                            attention_future_per_call=att_future.astype(np.float32),
                            blocks=np.asarray(blocks))
        rec_out = {
            "state_id": sid, "suite": m["suite"], "phase": m["phase"], "episode_idx": m["episode_idx"],
            "blocks_recorded": blocks, "n_calls": int(n_calls),
            "equivalence_maxabs_vs_sdpa": equiv,
            "mass_on_future_half": float(att_future.sum(axis=1).mean()),
            "mass_on_h_t_half": float(att_h_t.sum(axis=1).mean()),
            "attention_future_stats": pc.stats(a_fut_mean),
            "attention_future_p95_over_median": float(np.percentile(a_fut_mean, 95) / max(np.median(a_fut_mean), 1e-30)),
            "per_block_mass_on_future": (per_block.sum(axis=2).mean(axis=0).tolist() if per_block is not None else None),
            "wall_sec": round(time.time() - t0, 1),
        }
        pc.append_jsonl(rec_path, rec_out)
        print(f"{sid}: equiv={equiv:.2e} future_mass={rec_out['mass_on_future_half']:.3f} "
              f"h_t_mass={rec_out['mass_on_h_t_half']:.3f} att_p95/med={rec_out['attention_future_p95_over_median']:.2f} "
              f"({rec_out['wall_sec']}s)", flush=True)
        if equiv > EQUIV_TOL:
            print(f"  WARNING: recorder changes the action by {equiv:.3e} (> {EQUIV_TOL})", flush=True)


if __name__ == "__main__":
    main()
