"""G3 stage C2 analysis — the last mechanism check of Gap Validation.

Only question: under stress, does D x S still stably beat D, S, attention and D x attention, and does it
stay close to the oracle U x S? Every difference is reported as a recovered fraction AND in millimetres,
next to the sampler nuisance measured in the same stressed conditions.

No new formula, proxy, stress or failure predictor is introduced.
Outputs: <run_dir>/TOKEN_SELECTION_RESULTS.csv, EXECUTION_RESULTS.csv, c2_summary.json,
         figures/g3_c2.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/learned_u/scripts"))
import g2_dataset as gd  # noqa: E402

SEL = ["A_random", "B_top_D", "C_top_S", "D_top_D_times_S", "E_top_attention",
       "F_top_D_times_attention", "G_top_U_oracle", "H_top_U_times_S_oracle"]
LABEL = {"A_random": "random", "B_top_D": "D", "C_top_S": "S", "D_top_D_times_S": "D x S",
         "E_top_attention": "attention", "F_top_D_times_attention": "D x attention",
         "G_top_U_oracle": "U (oracle)", "H_top_U_times_S_oracle": "U x S (oracle)"}
# frozen in-distribution reference from the DS round, for the side-by-side comparison
ID_RECOVERY = {"random": 0.156, "D": 0.580, "S": 0.660, "D x S": 0.771,
               "attention": 0.362, "D x attention": 0.578,
               "U (oracle)": 0.544, "U x S (oracle)": 0.742}
ID_DS_MARGIN = 0.148


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--c2_dir", required=True)
    args = ap.parse_args()
    run_dir, c2 = Path(args.run_dir), Path(args.c2_dir)
    recs = [json.loads(l) for l in open(c2 / "c2_execution_records.jsonl")]
    recs.sort(key=lambda r: r["state_id"])
    ep = np.asarray([r["episode"] for r in recs])
    cond = np.asarray([r["condition"] for r in recs])
    suite = np.asarray([r["suite"] for r in recs])
    outcome = np.asarray(["success" if r["episode_success"] else "failure" for r in recs])
    rec = {s: np.asarray([r[f"recovery_{s}"] for r in recs]) for s in SEL}
    resid = {s: np.asarray([r[f"residual_mm_{s}"] for r in recs]) for s in SEL}
    treat = np.asarray([r["eef_treatment_mm"] for r in recs])
    nuis = np.asarray([r["eef_nuisance_mm"] for r in recs])

    with open(run_dir / "EXECUTION_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)

    rows = []
    for s in SEL:
        ci = gd.boot_ci(rec[s], ep)
        rows.append({"selector": LABEL[s], "key": s,
                     "recovery_median": float(np.median(rec[s])), "recovery_mean": float(np.mean(rec[s])),
                     "recovery_ci_low": ci[0], "recovery_ci_high": ci[1],
                     "residual_mm_median": float(np.median(resid[s])),
                     "ID_recovery_DS_round": ID_RECOVERY[LABEL[s]],
                     "recovery_A_occlusion": float(np.median(rec[s][cond == "A_occlusion_medium"])),
                     "recovery_B_shift": float(np.median(rec[s][cond == "B_camera_shift_strong"])),
                     "recovery_success_eps": float(np.median(rec[s][outcome == "success"])),
                     "recovery_failure_eps": float(np.median(rec[s][outcome == "failure"]))})
    with open(run_dir / "TOKEN_SELECTION_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def paired(a: str, b: str, mask=None) -> dict:
        m = np.ones(len(recs), bool) if mask is None else mask
        d = rec[a][m] - rec[b][m]
        mm = resid[b][m] - resid[a][m]
        return {"mean_diff": float(np.mean(d)), "median_diff": float(np.median(d)),
                "ci": gd.boot_ci(d, ep[m]), "frac_states_a_better": float(np.mean(d > 0)),
                "mm_mean": float(np.mean(mm)), "mm_ci": gd.boot_ci(mm, ep[m])}

    primary = {
        "DxS - D": paired("D_top_D_times_S", "B_top_D"),
        "DxS - S": paired("D_top_D_times_S", "C_top_S"),
        "DxS - attention": paired("D_top_D_times_S", "E_top_attention"),
        "DxS - D x attention": paired("D_top_D_times_S", "F_top_D_times_attention"),
        "DxS - UxS (oracle)": paired("D_top_D_times_S", "H_top_U_times_S_oracle"),
        "DxS - random": paired("D_top_D_times_S", "A_random"),
        "UxS - U (oracle margin)": paired("H_top_U_times_S_oracle", "G_top_U_oracle"),
    }
    summary = {
        "n_states": len(recs), "token_budget": recs[0]["token_budget"],
        "subset_rule": "2 frozen stress conditions x 6 tasks x 10 episodes x the approach-phase state",
        "stress_scales_mm": {"treatment_median": float(np.median(treat)),
                             "nuisance_median": float(np.median(nuis)),
                             "treatment_ci": gd.boot_ci(treat, ep),
                             "nuisance_ci": gd.boot_ci(nuis, ep)},
        "recovery": {LABEL[s]: {"median": float(np.median(rec[s])), "mean": float(np.mean(rec[s])),
                                "ci": gd.boot_ci(rec[s], ep),
                                "residual_mm_median": float(np.median(resid[s])),
                                "ID_DS_round": ID_RECOVERY[LABEL[s]]} for s in SEL},
        "primary_comparisons": primary,
        "ID_reference_DS_margin": ID_DS_MARGIN,
        "by_condition": {c: {LABEL[s]: float(np.median(rec[s][cond == c])) for s in SEL}
                         for c in np.unique(cond)},
        "by_suite": {x: {LABEL[s]: float(np.median(rec[s][suite == x])) for s in SEL}
                     for x in np.unique(suite)},
        "by_episode_outcome": {o: {LABEL[s]: float(np.median(rec[s][outcome == o])) for s in SEL}
                               for o in np.unique(outcome)},
        "DS_margin_by_condition": {c: paired("D_top_D_times_S", "B_top_D", cond == c)
                                   for c in np.unique(cond)},
        "DS_margin_by_outcome": {o: paired("D_top_D_times_S", "B_top_D", outcome == o)
                                 for o in np.unique(outcome)},
    }
    json.dump(summary, open(run_dir / "c2_summary.json", "w"), indent=2)

    print(f"n={len(recs)} states, budget {recs[0]['token_budget']}/256 | "
          f"stressed treatment {np.median(treat):.2f} mm, sampler nuisance {np.median(nuis):.2f} mm\n")
    print(f"{'selector':16s} {'recovery':>9s} {'CI':>18s} {'resid mm':>9s} {'ID (DS round)':>14s}")
    for s in SEL:
        v = summary["recovery"][LABEL[s]]
        print(f"  {LABEL[s]:14s} {v['median']:9.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] "
              f"{v['residual_mm_median']:9.3f} {v['ID_DS_round']:14.3f}")
    print("\n--- paired comparisons (episode bootstrap)")
    for k, v in primary.items():
        print(f"  {k:26s} mean={v['mean_diff']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
              f"better_in={v['frac_states_a_better']:.0%}  ({v['mm_mean']:+.3f} mm)")
    print("\n--- D x S - D by condition and by episode outcome")
    for k, v in summary["DS_margin_by_condition"].items():
        print(f"  {k:24s} mean={v['mean_diff']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] ({v['mm_mean']:+.3f} mm)")
    for k, v in summary["DS_margin_by_outcome"].items():
        print(f"  episodes: {k:14s} mean={v['mean_diff']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] ({v['mm_mean']:+.3f} mm)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["A_random", "E_top_attention", "F_top_D_times_attention", "G_top_U_oracle",
                 "B_top_D", "C_top_S", "H_top_U_times_S_oracle", "D_top_D_times_S"]
        cols = ["#999999", "#ddaa33", "#ddaa33", "#44aa99", "#4477aa", "#4477aa", "#44aa99", "#cc6677"]
        fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
        ax[0].boxplot([rec[s] for s in order], tick_labels=[LABEL[s] for s in order], showfliers=False)
        for i, s in enumerate(order):
            ax[0].scatter(np.full(rec[s].size, i + 1) + np.random.uniform(-.12, .12, rec[s].size),
                          rec[s], s=6, alpha=.35, color=cols[i])
            ax[0].plot([i + 1], [ID_RECOVERY[LABEL[s]]], marker="_", ms=18, color="black", lw=2)
        ax[0].set_ylim(-1.0, 1.05); ax[0].axhline(0, color="grey", lw=.6)
        ax[0].tick_params(axis="x", labelsize=7)
        ax[0].set_ylabel("fraction of oracle correction recovered")
        ax[0].set_title(f"under stress, {len(recs)} states, 64/256 tokens\n(black dash = the in-distribution DS-round value)")
        names = ["D x S - D", "D x S -\nD x attention", "D x S -\nS", "D x S -\nU x S (oracle)"]
        keys = ["DxS - D", "DxS - D x attention", "DxS - S", "DxS - UxS (oracle)"]
        vals = [primary[k]["mean_diff"] for k in keys]
        cis = [primary[k]["ci"] for k in keys]
        err = [[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]
        ax[1].bar(range(len(vals)), vals, yerr=err, capsize=4,
                  color=["#cc6677" if v > 0 else "#4477aa" for v in vals])
        ax[1].axhline(0, color="black", lw=1)
        ax[1].axhline(ID_DS_MARGIN, ls="--", color="grey", lw=.9)
        ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels(names, fontsize=8)
        ax[1].set_ylabel("difference in recovered fraction")
        ax[1].set_title("does sensitivity still add under stress?\n(dashed = in-distribution D x S - D margin)")
        fig.tight_layout(); fig.savefig(run_dir / "figures" / "g3_c2.png", dpi=150)
        print(f"\nfigure: {run_dir / 'figures' / 'g3_c2.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[figure skipped] {exc}")


if __name__ == "__main__":
    main()
