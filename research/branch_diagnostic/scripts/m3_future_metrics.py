"""Step 6a (lawam env): future-prediction metrics for every execution record.

For each record in episode_records.jsonl (phase filter): extract LAM vision features of the ACTUAL o_7 (and o_8) of that
execution; compare the model's predicted future (B0: h_t1_pred; B1: h_t1_pred_used; B2: the override itself is the
reference u_X, so no LaWM prediction exists) with the actual future and with the reference anchors u_A / u_B.
ROI masks come from projected eef/bowl pixels (offline evaluation only, never given to the policy).
Writes future_metrics.jsonl (one line per record) into the run dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_common as mx  # noqa: E402
import model_common as mc  # noqa: E402
from m2_infer_conditions import lam_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    recs = [json.loads(l) for l in open(run_dir / "episode_records.jsonl")]
    if args.phase:
        recs = [r for r in recs if r["phase"] == args.phase]
    out_path = run_dir / "future_metrics.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["state_id"], r["tag"]) for r in (json.loads(l) for l in open(out_path))}
    vla, _, _ = mc.load_policy()
    backend = vla.policy_backend
    roi = json.load(open(run_dir / "roi_pixels.json")) if (run_dir / "roi_pixels.json").exists() else {}
    cache = {}
    for r in recs:
        sid = r["state_id"]; tag = Path(r["files"]["execution"]).stem
        if (sid, tag) in done:
            continue
        if sid not in cache:
            mo = np.load(run_dir / "conditions" / sid / "model_outputs.npz")
            noise_floor = 0.5 * (float(((mo["u_A"] - mo["u_A_replay"]) ** 2).mean()) + float(((mo["u_B"] - mo["u_B_replay"]) ** 2).mean()))
            masks = {}
            if sid in roi:
                masks = {b: mx.roi_mask_from_pixels(roi[sid][b]["pixels"]) for b in roi[sid]}
            cache[sid] = (mo, noise_floor, masks)
        mo, noise_floor, masks = cache[sid]
        ex = np.load(r["files"]["execution"])
        frames = ex["frames"]
        n = len(frames)
        f7 = lam_features(backend, frames[7]) if n > 7 else None
        f8 = lam_features(backend, frames[8]) if n > 8 else None
        cond, ib = r["condition_group"], r["reference_branch"]
        pred = None
        if cond == "B0":
            pred = mo["B0_h_t1_pred"]
        elif cond == "B1":
            pred = mo[f"B1_{ib}_h_t1_pred_used"]
        elif cond == "B2":
            pred = mo[f"u_{ib}"]  # by construction (no LaWM prediction in B2); reported for completeness, not as a prediction
        u_A, u_B = mo["u_A"], mo["u_B"]
        entry = {"state_id": sid, "tag": tag, "condition_group": cond, "reference_branch": ib, "noise_seed": r["noise_seed"], "executed_branch": r["executed_branch"],
                 "noise_floor_mse": noise_floor, "mse_uA_uB": float(((u_A - u_B) ** 2).mean())}
        if f7 is not None:
            entry["actual7_mse_to_uA"] = mx.feat_mse(f7, u_A); entry["actual7_mse_to_uB"] = mx.feat_mse(f7, u_B)
            lab_act, s_act, _, _ = mx.predicted_label(f7, u_A, u_B, noise_floor)
            entry["actual7_feature_label"] = lab_act; entry["actual7_s_f"] = s_act
        if pred is not None:
            lab, s_f, d_A, d_B = mx.predicted_label(pred, u_A, u_B, noise_floor)
            entry.update({"predicted_branch": lab, "pred_s_f": s_f, "pred_mse_to_uA": d_A, "pred_mse_to_uB": d_B,
                          "pred_mse_to_actual7": None if f7 is None else mx.feat_mse(pred, f7), "pred_mse_to_actual8": None if f8 is None else mx.feat_mse(pred, f8),
                          "pred_mse_to_h_t": mx.feat_mse(pred, mo["B0_h_t"]), "actual7_mse_to_h_t": None if f7 is None else mx.feat_mse(f7, mo["B0_h_t"]),
                          "pred_is_override_not_prediction": cond == "B2"})
            if ib in masks:
                for b in "AB":
                    if b in masks:
                        entry[f"pred_roi{b}_mse_to_u{b}"] = mx.feat_mse(pred, mo[f"u_{b}"], masks[b])
                lab_roi, s_roi, dA_roi, dB_roi = mx.predicted_label(pred, u_A, u_B, noise_floor, mask=masks[ib])
                entry.update({"predicted_branch_roi": lab_roi, "pred_s_f_roi": s_roi})
                if f7 is not None:
                    entry["pred_roi_mse_to_actual7"] = mx.feat_mse(pred, f7, masks[ib])
        else:
            entry["predicted_branch"] = None
        with open(out_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    print("future metrics written:", out_path)


if __name__ == "__main__":
    main()
