"""Step 2 analysis: equal-budget causal comparison of the 8 selectors (numpy only).

Primary quantities (PROTOCOL_DS.yaml:step2_causal.primary_quantities):
    DS_margin        = Recovery(D x S) - Recovery(D)              [must have a 95% episode-bootstrap CI > 0]
    Oracle_margin    = Recovery(U x S) - Recovery(U)
    Margin_retention = DS_margin / Oracle_margin                   [descriptive only]
    attention_check  = Recovery(D x S) - Recovery(D x attention)

Every recovery difference is also expressed in MILLIMETRES of executed end-effector displacement, next to
the two reference scales (flow-sampler nuisance, chunk motion), as the protocol requires.
Outputs: <run_dir>/SELECTION_METRICS.csv, EXECUTION_METRICS.csv, summary_ds.json, figures/ds_selectors.png
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

PRIOR = REPO / "results/action_critical_reliability_probe/acr_20260919_203850"
SELECTORS = ["A_random", "B_top_D", "C_top_S", "D_top_D_times_S", "E_top_U_oracle",
             "F_top_U_times_S", "G_top_attention", "H_top_D_times_attention"]
LABEL = {"A_random": "random", "B_top_D": "D", "C_top_S": "S", "D_top_D_times_S": "D x S",
         "E_top_U_oracle": "U (oracle)", "F_top_U_times_S": "U x S (oracle)",
         "G_top_attention": "attention", "H_top_D_times_attention": "D x attention"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    recs = [json.loads(l) for l in open(run_dir / "ds_execution_records.jsonl")]
    recs.sort(key=lambda r: r["state_id"])
    ep = np.asarray([r["episode"] for r in recs])
    ph = np.asarray([r["phase"] for r in recs])
    su = np.asarray([r["suite"] for r in recs])
    treat = np.asarray([r["eef_treatment_mm"] for r in recs])
    nuis = np.asarray([r["eef_nuisance_mm"] for r in recs])

    rec = {s: np.asarray([r[f"recovery_{s}"] for r in recs]) for s in SELECTORS}
    resid = {s: np.asarray([r[f"residual_mm_{s}"] for r in recs]) for s in SELECTORS}

    with open(run_dir / "EXECUTION_METRICS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)

    rows = []
    for s in SELECTORS:
        rows.append({"selector": LABEL[s], "key": s,
                     "recovery_median": float(np.median(rec[s])),
                     "recovery_mean": float(np.mean(rec[s])),
                     "recovery_ci_low": gd.boot_ci(rec[s], ep)[0],
                     "recovery_ci_high": gd.boot_ci(rec[s], ep)[1],
                     "residual_to_oracle_mm_median": float(np.median(resid[s])),
                     "eef_dev_from_pred_mm_median": float(np.median(
                         [r[f"eef_dev_from_pred_mm_{s}"] for r in recs])),
                     "action_maxabs_median": float(np.median([r[f"action_maxabs_{s}"] for r in recs]))})
    with open(run_dir / "SELECTION_METRICS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    def paired(a: str, b: str) -> dict:
        d = rec[a] - rec[b]
        mm = resid[b] - resid[a]           # positive = a leaves less residual error, in mm
        return {"mean_diff": float(np.mean(d)), "median_diff": float(np.median(d)),
                "ci": gd.boot_ci(d, ep), "frac_states_a_better": float(np.mean(d > 0)),
                "mm_mean": float(np.mean(mm)), "mm_median": float(np.median(mm)),
                "mm_ci": gd.boot_ci(mm, ep)}

    ds_margin = paired("D_top_D_times_S", "B_top_D")
    oracle_margin = paired("F_top_U_times_S", "E_top_U_oracle")
    retention = (ds_margin["mean_diff"] / oracle_margin["mean_diff"]
                 if abs(oracle_margin["mean_diff"]) > 1e-12 else float("nan"))

    summary = {
        "n_states": len(recs), "token_budget": recs[0]["token_budget"],
        "reference_scales_mm": {
            "chunk_motion_round1_median": 84.99,
            "treatment_pi_Hreal_vs_pi_Hpred": {"median": float(np.median(treat)),
                                               "ci": gd.boot_ci(treat, ep)},
            "flow_sampler_nuisance": {"median": float(np.median(nuis)), "ci": gd.boot_ci(nuis, ep)},
        },
        "recovery": {LABEL[s]: {"median": float(np.median(rec[s])), "mean": float(np.mean(rec[s])),
                                "ci": gd.boot_ci(rec[s], ep),
                                "residual_mm_median": float(np.median(resid[s]))} for s in SELECTORS},
        "primary": {
            "DS_margin (DxS - D)": ds_margin,
            "Oracle_margin (UxS - U)": oracle_margin,
            "Margin_retention": retention,
            "attention_check (DxS - Dxattention)": paired("D_top_D_times_S", "H_top_D_times_attention"),
        },
        "other_paired": {
            "DxS - S": paired("D_top_D_times_S", "C_top_S"),
            "DxS - attention": paired("D_top_D_times_S", "G_top_attention"),
            "DxS - random": paired("D_top_D_times_S", "A_random"),
            "DxS - UxS(oracle)": paired("D_top_D_times_S", "F_top_U_times_S"),
            "D - U(oracle)": paired("B_top_D", "E_top_U_oracle"),
            "D - random": paired("B_top_D", "A_random"),
            "D - attention": paired("B_top_D", "G_top_attention"),
        },
        "by_phase": {p: {LABEL[s]: float(np.median(rec[s][ph == p])) for s in SELECTORS}
                     for p in np.unique(ph)},
        "by_suite": {x: {LABEL[s]: float(np.median(rec[s][su == x])) for s in SELECTORS}
                     for x in np.unique(su)},
        "by_phase_margins": {p: {"DS_margin": float(np.mean(rec["D_top_D_times_S"][ph == p]
                                                            - rec["B_top_D"][ph == p])),
                                 "Oracle_margin": float(np.mean(rec["F_top_U_times_S"][ph == p]
                                                                - rec["E_top_U_oracle"][ph == p]))}
                             for p in np.unique(ph)},
        "round1_cross_check": {
            "note": "same quantity recomputed on the same states; round 1 reported median Recovery(U)=0.545, Recovery(UxS)=0.742",
            "recovery_U_median_here": float(np.median(rec["E_top_U_oracle"])),
            "recovery_UxS_median_here": float(np.median(rec["F_top_U_times_S"])),
        },
    }
    json.dump(summary, open(run_dir / "summary_ds.json", "w"), indent=2)

    print(f"n={len(recs)} states, budget {recs[0]['token_budget']}/256")
    print(f"scales: chunk motion 85.0 mm | oracle correction {np.median(treat):.2f} mm | "
          f"sampler nuisance {np.median(nuis):.2f} mm\n")
    print(f"{'selector':16s} {'recovery':>9s} {'CI':>18s} {'residual mm':>12s}")
    for s in SELECTORS:
        v = summary["recovery"][LABEL[s]]
        print(f"  {LABEL[s]:14s} {v['median']:9.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] "
              f"{v['residual_mm_median']:12.3f}")
    print("\n--- primary comparisons (paired, episode bootstrap)")
    for k, v in summary["primary"].items():
        if isinstance(v, dict):
            print(f"  {k:36s} mean={v['mean_diff']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
                  f"better_in={v['frac_states_a_better']:.0%}  ({v['mm_mean']:+.3f} mm)")
        else:
            print(f"  {k:36s} {v:+.3f}")
    print("\n--- other paired comparisons")
    for k, v in summary["other_paired"].items():
        print(f"  {k:22s} mean={v['mean_diff']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
              f"better_in={v['frac_states_a_better']:.0%}  ({v['mm_mean']:+.3f} mm)")
    print("\n--- by phase (median recovery)")
    for p, v in summary["by_phase"].items():
        print(f"  {p:14s} " + "  ".join(f"{k}={x:.3f}" for k, x in v.items()))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        order = ["A_random", "G_top_attention", "H_top_D_times_attention", "C_top_S",
                 "B_top_D", "D_top_D_times_S", "E_top_U_oracle", "F_top_U_times_S"]
        cols = ["#999999", "#ddaa33", "#ddaa33", "#4477aa", "#4477aa", "#cc6677", "#44aa99", "#44aa99"]
        ax[0].boxplot([rec[s] for s in order], tick_labels=[LABEL[s] for s in order], showfliers=False)
        for i, s in enumerate(order):
            ax[0].scatter(np.full(rec[s].size, i + 1) + np.random.uniform(-.12, .12, rec[s].size),
                          rec[s], s=5, alpha=.3, color=cols[i])
        ax[0].set_ylabel("fraction of oracle correction recovered")
        # A few states have strongly negative recovery (the repaired chunk ends further from the oracle
        # than the unrepaired one). They are kept in every statistic; the axis is clipped only so the bulk
        # is readable, and the number of clipped points is annotated rather than hidden.
        lo = -1.0
        n_clipped = int(sum(int((rec[s] < lo).sum()) for s in order))
        ax[0].set_ylim(lo, 1.05)
        ax[0].text(0.02, 0.03, f"{n_clipped} points below {lo:.0f} not shown (kept in all statistics)",
                   transform=ax[0].transAxes, fontsize=6.5, color="#555555")
        ax[0].set_title(f"equal budget {recs[0]['token_budget']}/256 tokens repaired, {len(recs)} states")
        ax[0].tick_params(axis="x", labelsize=7); ax[0].axhline(0, color="grey", lw=.6)
        names = ["D x S  -  D", "U x S  -  U\n(oracle)", "D x S  -  D x attention"]
        vals = [summary["primary"]["DS_margin (DxS - D)"]["mean_diff"],
                summary["primary"]["Oracle_margin (UxS - U)"]["mean_diff"],
                summary["primary"]["attention_check (DxS - Dxattention)"]["mean_diff"]]
        cis = [summary["primary"]["DS_margin (DxS - D)"]["ci"],
               summary["primary"]["Oracle_margin (UxS - U)"]["ci"],
               summary["primary"]["attention_check (DxS - Dxattention)"]["ci"]]
        err = [[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]
        ax[1].bar(range(3), vals, yerr=err, capsize=4,
                  color=["#cc6677" if v > 0 else "#4477aa" for v in vals])
        ax[1].axhline(0, color="black", lw=1)
        ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names, fontsize=8)
        ax[1].set_ylabel("difference in recovered fraction")
        ax[1].set_title("does sensitivity add to the deployable signal?\n(mean, episode-bootstrap CI)")
        fig.tight_layout(); fig.savefig(run_dir / "figures" / "ds_selectors.png", dpi=150)
        print(f"\nfigure: {run_dir / 'figures' / 'ds_selectors.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[figure skipped] {exc}")


if __name__ == "__main__":
    main()
