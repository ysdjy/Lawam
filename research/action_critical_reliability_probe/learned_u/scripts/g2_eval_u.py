"""Step 4 evaluation + permutation audit: does the learned U generalise, and does it beat the free baselines?

Everything is computed per STATE and aggregated with a bootstrap over EPISODES. The comparison that decides
G2-A is the PAIRED difference between U_hat and the best no-training baseline on held-out splits.

Also runs the mandatory permutation audit: permuting h_t and H_pred with the same permutation must permute
U_hat identically (the architecture is shared-weight per-token, so this checks the implementation).

Outputs: <run_dir>/METRICS_U.csv, <run_dir>/summary_learned_u.json, figures/g2_learned_u.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_dataset as gd  # noqa: E402
from g2_baselines import STAT_KEYS, baseline_scores  # noqa: E402
from g2_train_u import SEEDS, SharedTokenMLP  # noqa: E402

SPLITS = ["train", "val", "testA", "testB1", "testB2", "testC"]
METRIC_KEYS = ["spearman", "pearson", "top64_recall", "top64_precision", "ndcg64", "auroc_top_quartile"]


def load_models(run_dir: Path, device: str):
    models = []
    for s in SEEDS:
        ck = torch.load(run_dir / "models" / f"u_mlp_seed{s}.pt", map_location=device, weights_only=False)
        m = SharedTokenMLP(ck["in_dim"], ck["hidden"]).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    nz = np.load(run_dir / "models" / "feature_norm.npz")
    return models, nz["mu"], nz["sd"]


def predict(models, mu, sd, feats: np.ndarray, device: str) -> np.ndarray:
    """Seed-averaged prediction in log space -> [256]."""
    x = torch.tensor((feats - mu) / sd, dtype=torch.float32, device=device)
    with torch.no_grad():
        return np.mean([m(x).cpu().numpy() for m in models], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    states = gd.load_states(run_dir)
    models, mu, sd = load_models(run_dir, args.device)

    # ---- B4 refit exactly as in g2_baselines (train split only)
    tr = [s for s in states if s.split == "train"]
    Xtr = np.concatenate([np.stack([s.simple_stats()[k] for k in STAT_KEYS], 1) for s in tr], 0)
    ytr = np.concatenate([np.log(s.U_l2 + gd.LABEL_EPS) for s in tr], 0)
    b4, *_ = np.linalg.lstsq(np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], 1), ytr, rcond=None)

    rows, perm_audit = [], []
    for st in states:
        feats = st.features()
        u_hat = predict(models, mu, sd, feats, args.device)
        preds = {"U_hat_mlp": u_hat, **baseline_scores(st, b4)}
        for name, sc in preds.items():
            rows.append({"state_id": st.state_id, "split": st.split, "phase": st.phase, "suite": st.suite,
                         "task_id": st.task_id, "episode": st.episode_key, "predictor": name,
                         **gd.state_metrics(sc, st.U_l2)})
        # calibration in log space (seed-averaged prediction vs oracle log U)
        rows[-len(preds)]["log_mae"] = float(np.mean(np.abs(u_hat - np.log(st.U_l2 + gd.LABEL_EPS))))
        rows[-len(preds)]["log_rmse"] = float(np.sqrt(np.mean((u_hat - np.log(st.U_l2 + gd.LABEL_EPS)) ** 2)))

        # ---- permutation audit
        rng = np.random.default_rng(7)
        perm = rng.permutation(st.h_t.shape[0])
        st_perm = gd.StateData(st.state_id, st.suite, st.task_id, st.episode, st.phase, st.split,
                               st.h_t[perm], st.h_pred[perm], st.z, st.U_l2[perm], st.U_mse[perm],
                               st.U_floor_mse[perm])
        u_perm = predict(models, mu, sd, st_perm.features(), args.device)
        perm_audit.append(float(np.max(np.abs(u_perm - u_hat[perm]))))

    with open(run_dir / "METRICS_U.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summary = {"n_states": {s: sum(1 for x in states if x.split == s) for s in SPLITS},
               "permutation_audit": {"max_abs_deviation": float(np.max(perm_audit)),
                                     "median": float(np.median(perm_audit)),
                                     "tolerance": 1e-5,
                                     "passed": bool(np.max(perm_audit) < 1e-5),
                                     "note": "shared-weight per-token MLP is permutation-equivariant by construction"},
               "by_split": {}, "paired_vs_baselines": {}, "by_phase": {}}

    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        entry = {}
        for name in sorted({r["predictor"] for r in sub}):
            sel = [r for r in sub if r["predictor"] == name]
            e = np.asarray([r["episode"] for r in sel])
            entry[name] = {m: {"mean": float(np.nanmean([r[m] for r in sel])),
                               "ci": gd.boot_ci(np.asarray([r[m] for r in sel], dtype=float), e)}
                           for m in METRIC_KEYS}
        summary["by_split"][split] = entry

        # paired difference U_hat - baseline, per state, bootstrapped over episodes
        base = {r["state_id"]: r for r in sub if r["predictor"] == "U_hat_mlp"}
        paired = {}
        for name in sorted({r["predictor"] for r in sub} - {"U_hat_mlp"}):
            sel = [r for r in sub if r["predictor"] == name]
            diff = np.asarray([base[r["state_id"]]["spearman"] - r["spearman"] for r in sel])
            e = np.asarray([r["episode"] for r in sel])
            rec_diff = np.asarray([base[r["state_id"]]["top64_recall"] - r["top64_recall"] for r in sel])
            paired[name] = {"spearman_diff_mean": float(np.mean(diff)),
                            "spearman_diff_ci": gd.boot_ci(diff, e),
                            "frac_states_better": float(np.mean(diff > 0)),
                            "top64_recall_diff_mean": float(np.mean(rec_diff)),
                            "top64_recall_diff_ci": gd.boot_ci(rec_diff, e)}
        summary["paired_vs_baselines"][split] = paired

    for phase in sorted({r["phase"] for r in rows}):
        sub = [r for r in rows if r["phase"] == phase and r["split"].startswith("test")]
        summary["by_phase"][phase] = {
            name: float(np.nanmean([r["spearman"] for r in sub if r["predictor"] == name]))
            for name in sorted({r["predictor"] for r in sub})}

    json.dump(summary, open(run_dir / "summary_learned_u.json", "w"), indent=2)

    print(f"permutation audit: max|U_hat(perm) - perm(U_hat)| = {summary['permutation_audit']['max_abs_deviation']:.3e} "
          f"-> {'PASS' if summary['permutation_audit']['passed'] else 'FAIL'}")
    for split in SPLITS:
        print(f"\n--- {split} (n={summary['n_states'][split]})")
        for name, m in summary["by_split"][split].items():
            sp = m["spearman"]; rc = m["top64_recall"]
            print(f"  {name:20s} spearman={sp['mean']:+.4f} CI[{sp['ci'][0]:+.4f},{sp['ci'][1]:+.4f}] "
                  f"recall64={rc['mean']:.3f}")
        for name, p in summary["paired_vs_baselines"][split].items():
            if name.startswith("B0"):
                continue
            print(f"    U_hat - {name:18s} dSpearman={p['spearman_diff_mean']:+.4f} "
                  f"CI[{p['spearman_diff_ci'][0]:+.4f},{p['spearman_diff_ci'][1]:+.4f}] "
                  f"better_in={p['frac_states_better']:.0%}  dRecall={p['top64_recall_diff_mean']:+.3f}")


if __name__ == "__main__":
    main()
