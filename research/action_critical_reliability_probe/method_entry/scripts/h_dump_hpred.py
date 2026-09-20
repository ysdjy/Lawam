"""Dump the LIVE full-precision H_pred for the 120 frozen C2 states (model side, `lawam` env).

Why this exists: the headroom variants B/C/D must be built on the *live* bf16 `H_pred`, with only the 64
tokens chosen by the frozen selector replaced. Rebuilding the whole future from the float16 archive is
forbidden — float16 storage of bf16 features is lossy below the float16 subnormal threshold (bf16 keeps the
fp32 exponent range, fp16 does not), which was measured to shift the action chunk by ~1e-4.

float32 IS lossless for a bf16 value, so this dump *is* the live tensor, stored without loss. Variant A
(baseline) never uses it: it sends no override at all and lets the server use its own live tensor.

Also records, per state, a sha256 of the bf16 bytes so any later reconstruction can be proven identical.
Output: <run_dir>/hpred_fp32/<state_id>.npz  (h_pred float32, plus the frozen picks copied for convenience)
"""
from __future__ import annotations

import argparse
import hashlib
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--g3_run", required=True)
    ap.add_argument("--states", default="all")
    args = ap.parse_args()
    run_dir, g3 = Path(args.run_dir), Path(args.g3_run)
    out_dir = run_dir / "hpred_fp32"
    out_dir.mkdir(parents=True, exist_ok=True)
    states = [json.loads(l) for l in open(g3 / "c2/states_manifest.jsonl")]
    if args.states != "all":
        keep = set(args.states.split(","))
        states = [s for s in states if s["state_id"] in keep]

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    rec_path = run_dir / "hpred_dump_records.jsonl"

    for st in states:
        sid = st["state_id"]
        p = out_dir / f"{sid}.npz"
        if p.exists():
            continue
        t0 = time.time()
        d = np.load(st["state_npz"], allow_pickle=True)
        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred                                   # [1, 256, 768] bf16, the live tensor

        h_fp32 = Hp[0].float().cpu().numpy().astype(np.float32)
        bf16_bytes = Hp[0].cpu().contiguous().view(-1).numpy().tobytes()
        # the archived float16 copy, kept only to quantify the loss that motivated this script
        stored_fp16 = np.load(g3 / "confirm/treatment_chunks" / st["condition"] / f"{sid}.npz")["h_t1_pred"]
        fp16_err = float(np.max(np.abs(h_fp32 - stored_fp16.astype(np.float32))))

        picks = np.load(g3 / "c2/c2_chunks" / f"{sid}.npz")
        np.savez_compressed(
            p, h_pred=h_fp32,
            pick_B_top_D=picks["pick_B_top_D"], pick_D_top_D_times_S=picks["pick_D_top_D_times_S"],
            token_budget=picks["token_budget"])
        pc.append_jsonl(rec_path, {
            "state_id": sid, "condition": st["condition"], "suite": st["suite"],
            "task_id": st["task_id"], "episode_idx": st["episode_idx"],
            "bf16_sha256": hashlib.sha256(bf16_bytes).hexdigest(),
            "fp32_roundtrip_exact": bool(np.array_equal(
                h_fp32.astype(np.float16), stored_fp16)),
            "max_abs_error_of_fp16_archive": fp16_err,
            "wall_sec": round(time.time() - t0, 2)})
        print(f"{sid}: fp16-archive max error {fp16_err:.3e} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
