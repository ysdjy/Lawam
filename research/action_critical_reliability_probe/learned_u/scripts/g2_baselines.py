"""Step 3: the mandatory no-training baselines for predicting oracle U (numpy only).

These run BEFORE any model is trained, because the prior round measured Spearman(U, token norm) ~ -0.55:
if a simple feature already predicts U, a learned estimator is not warranted (decision G2-A).

  B0  constant (train-split mean of log U)            -> no ranking information by construction
  B1  ||H_pred_j||                                    (sign flipped: U correlates NEGATIVELY with norm)
  B2  ||H_pred_j - h_t_j||
  B3  1 - cos(H_pred_j, h_t_j)
  B4  linear regression on 6 simple per-token statistics, fitted on the TRAIN episodes only

Every metric is computed per state and aggregated with an episode bootstrap.
Outputs: <run_dir>/METRICS_U_baselines.csv, <run_dir>/summary_baselines.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_dataset as gd  # noqa: E402

SPLITS = ["train", "val", "testA", "testB1", "testB2", "testC"]
STAT_KEYS = ["pred_norm", "ht_norm", "change_norm", "cos_dist", "pred_mean", "pred_std"]


def baseline_scores(st: gd.StateData, b4_coef: np.ndarray | None) -> dict[str, np.ndarray]:
    s = st.simple_stats()
    # B0 must carry NO ranking information. A literal constant is wrong here: `argsort` breaks ties by
    # index, which silently turns the "no information" floor into a TOKEN-INDEX predictor (measured at
    # Spearman +0.31, i.e. U really does have positional structure). The honest floor is a random score
    # with a fixed seed, reported alongside the analytic expectation (Spearman 0, recall 64/256 = 0.25).
    import hashlib

    seed = int.from_bytes(hashlib.sha256(st.state_id.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    out = {
        "B0_random": rng.standard_normal(s["pred_norm"].shape),
        "B0x_token_index": np.arange(s["pred_norm"].size, dtype=np.float64),
        "B1_pred_norm_neg": -s["pred_norm"],
        "B2_change_norm": s["change_norm"],
        "B3_cos_dist": s["cos_dist"],
    }
    if b4_coef is not None:
        X = np.stack([s[k] for k in STAT_KEYS], axis=1)
        out["B4_linear_stats"] = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ b4_coef
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    states = gd.load_states(run_dir)
    by_split = {s: [x for x in states if x.split == s] for s in SPLITS}
    print({k: len(v) for k, v in by_split.items()})

    # --- B4 fitted on TRAIN episodes only, target log(U + eps)
    Xtr, ytr = [], []
    for st in by_split["train"]:
        s = st.simple_stats()
        Xtr.append(np.stack([s[k] for k in STAT_KEYS], axis=1))
        ytr.append(np.log(st.U_l2 + gd.LABEL_EPS))
    Xtr = np.concatenate(Xtr, 0); ytr = np.concatenate(ytr, 0)
    A = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    b4_coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)

    rows = []
    for split in SPLITS:
        for st in by_split[split]:
            scores = baseline_scores(st, b4_coef)
            for name, sc in scores.items():
                m = gd.state_metrics(sc, st.U_l2)
                rows.append({"state_id": st.state_id, "split": split, "phase": st.phase,
                             "suite": st.suite, "task_id": st.task_id, "episode": st.episode_key,
                             "predictor": name, **m})

    with open(run_dir / "METRICS_U_baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = {"b4_coefficients": dict(zip(STAT_KEYS + ["intercept"], b4_coef.tolist())),
               "n_states": {k: len(v) for k, v in by_split.items()}, "by_split": {}}
    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        eps = np.asarray([r["episode"] for r in sub])
        entry = {}
        for name in sorted({r["predictor"] for r in sub}):
            sel = [r for r in sub if r["predictor"] == name]
            e = np.asarray([r["episode"] for r in sel])
            entry[name] = {
                m: {"mean": float(np.nanmean([r[m] for r in sel])),
                    "ci": gd.boot_ci(np.asarray([r[m] for r in sel], dtype=float), e)}
                for m in ["spearman", "top64_recall", "auroc_top_quartile", "ndcg64"]
            }
        summary["by_split"][split] = entry
    json.dump(summary, open(run_dir / "summary_baselines.json", "w"), indent=2)

    print("\nB4 coefficients (on log U):",
          {k: round(v, 4) for k, v in summary["b4_coefficients"].items()})
    for split in SPLITS:
        print(f"\n--- {split} (n={len(by_split[split])} states)")
        for name, m in summary["by_split"][split].items():
            sp, rc, au = m["spearman"], m["top64_recall"], m["auroc_top_quartile"]
            print(f"  {name:20s} spearman={sp['mean']:+.3f} CI[{sp['ci'][0]:+.3f},{sp['ci'][1]:+.3f}]  "
                  f"top64_recall={rc['mean']:.3f}  auroc={au['mean']:.3f}")


if __name__ == "__main__":
    main()
