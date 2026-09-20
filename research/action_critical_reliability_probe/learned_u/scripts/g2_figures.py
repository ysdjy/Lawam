"""Figures for the Learned-U round (matplotlib, no model)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_dataset as gd  # noqa: E402

SPLIT_ORDER = ["train", "val", "testA", "testB1", "testB2", "testC"]
SPLIT_LABEL = {"train": "train", "val": "val", "testA": "testA\nunseen resets", "testB1": "testB1\nunseen task",
               "testB2": "testB2\nunseen task+suite", "testC": "testC\nprior tasks"}
PREDICTORS = [("B1_pred_norm_neg", "-||H_pred||", "#999999"),
              ("B2_change_norm", "||H_pred - h_t||  (free)", "#4477aa"),
              ("B4_linear_stats", "linear(6 stats)", "#88ccee"),
              ("U_hat_mlp", "learned MLP", "#cc6677")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    s = json.loads((run_dir / "summary_learned_u.json").read_text())
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # --- (1) Spearman per split
    x = np.arange(len(SPLIT_ORDER))
    w = 0.2
    for i, (key, label, col) in enumerate(PREDICTORS):
        vals = [s["by_split"][sp][key]["spearman"]["mean"] for sp in SPLIT_ORDER]
        cis = [s["by_split"][sp][key]["spearman"]["ci"] for sp in SPLIT_ORDER]
        err = [[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]
        ax[0].bar(x + (i - 1.5) * w, vals, w, yerr=err, label=label, color=col, capsize=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels([SPLIT_LABEL[sp] for sp in SPLIT_ORDER], fontsize=7)
    ax[0].set_ylabel("Spearman with oracle U"); ax[0].set_ylim(0.4, 1.02)
    ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_title("predicting U before execution")
    ax[0].axhline(0.5, color="grey", lw=0.5)

    # --- (2) paired difference learned - free baseline
    diffs = [s["paired_vs_baselines"][sp]["B2_change_norm"]["spearman_diff_mean"] for sp in SPLIT_ORDER]
    cis = [s["paired_vs_baselines"][sp]["B2_change_norm"]["spearman_diff_ci"] for sp in SPLIT_ORDER]
    err = [[v - c[0] for v, c in zip(diffs, cis)], [c[1] - v for v, c in zip(diffs, cis)]]
    colors = ["#cc6677" if d > 0 else "#4477aa" for d in diffs]
    ax[1].bar(x, diffs, 0.6, yerr=err, color=colors, capsize=3)
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels([SPLIT_LABEL[sp] for sp in SPLIT_ORDER], fontsize=7)
    ax[1].set_ylabel("Spearman(learned) - Spearman(free)")
    ax[1].set_title("paired difference, episode-bootstrapped\nred = learned wins, blue = free statistic wins")

    # --- (3) scatter of one held-out-task state
    states = gd.load_states(run_dir, splits=["testB1"])
    st = states[0]
    stats = st.simple_stats()
    ax[2].scatter(stats["change_norm"], st.U_l2, s=6, alpha=0.45, color="#4477aa")
    ax[2].set_xlabel("||H_pred_j - h_t_j||  (free, available before execution)")
    ax[2].set_ylabel("oracle U_j = ||H_pred_j - H_real_j||")
    ax[2].set_title(f"one unseen-task state\n{st.state_id}\nSpearman = {gd.spearman(stats['change_norm'], st.U_l2):.3f}",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "g2_learned_u.png", dpi=150)
    print(f"figure: {fig_dir / 'g2_learned_u.png'}")


if __name__ == "__main__":
    main()
