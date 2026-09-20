"""G3 pilot analysis and PRE-REGISTERED severity selection.

Produces, per family x severity: baseline task success, future-error treatment (mm), sampler nuisance (mm),
treatment/nuisance, and the failure breakdown — then applies the severity-selection rule of
PROTOCOL_G3.yaml literally:

    primary  : lightest severity with 20% < success < 90%
    tie-break: lightest
    fallback : max treatment/nuisance subject to success >= 20%
    NO_EFFECT      : every severity keeps success ~100% AND treatment stays near the ID 0.9 mm
    TOO_DESTRUCTIVE: mild already below 20% success

The rule reads ONLY success, treatment and nuisance. No D, S, D x S or selector quantity exists in this
script, by construction.
Outputs: <run_dir>/PILOT_RESULTS.csv, pilot_summary.json, figures/g3_pilot.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/learned_u/scripts"))
import g2_dataset as gd  # noqa: E402  (episode bootstrap)

ID_TREATMENT_MM = 0.904          # in-distribution reference, round 1 / DS
ID_NUISANCE_MM = 0.474
SEV_ORDER = ["mild", "medium", "strong"]
SUCCESS_LOW, SUCCESS_HIGH = 0.20, 0.90
NO_EFFECT_SUCCESS = 0.95
NO_EFFECT_TREATMENT_FACTOR = 1.5     # "stays near the ID value"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    # ---- rollout-level success, from the episode metadata (all episodes, successes and failures)
    eps = []
    for cond_dir in sorted((run_dir / "rollouts").iterdir()):
        for task_dir in sorted(cond_dir.iterdir()):
            for ep_dir in sorted(task_dir.glob("ep*")):
                p = ep_dir / "episode.json"
                if p.exists():
                    eps.append(json.loads(p.read_text()))
    exe = [json.loads(l) for l in open(run_dir / "treatment_execution.jsonl")]

    by_cond_eps = defaultdict(list)
    for e in eps:
        by_cond_eps[e["condition"]].append(e)
    by_cond_exe = defaultdict(list)
    for r in exe:
        by_cond_exe[r["condition"]].append(r)

    rows = []
    for cond in sorted(by_cond_eps):
        E = by_cond_eps[cond]
        valid = [e for e in E if e.get("valid", True)]
        invalid = [e for e in E if not e.get("valid", True)]
        succ = [bool(e["success"]) for e in valid if e["success"] is not None]
        X = by_cond_exe.get(cond, [])
        t = np.asarray([r["eef_treatment_mm"] for r in X]) if X else np.asarray([np.nan])
        n = np.asarray([r["eef_nuisance_mm"] for r in X]) if X else np.asarray([np.nan])
        ep_key = np.asarray([r["episode"] for r in X]) if X else np.asarray(["-"])
        fam = E[0]["family"]
        rows.append({
            "condition": cond, "family": fam, "severity": E[0]["severity"],
            "n_episodes": len(E), "n_valid": len(valid), "n_invalid": len(invalid),
            "success_rate": float(np.mean(succ)) if succ else float("nan"),
            "n_success": int(np.sum(succ)), "n_fail": int(len(succ) - np.sum(succ)),
            "n_states": len(X),
            "treatment_mm_median": float(np.median(t)), "treatment_mm_mean": float(np.mean(t)),
            "treatment_ci_low": gd.boot_ci(t, ep_key)[0] if X else float("nan"),
            "treatment_ci_high": gd.boot_ci(t, ep_key)[1] if X else float("nan"),
            "nuisance_mm_median": float(np.median(n)),
            "treatment_over_nuisance_median": float(np.median(t / np.maximum(n, 1e-9))),
            "treatment_vs_ID": float(np.median(t) / ID_TREATMENT_MM),
            "obj_treatment_mm_median": float(np.median([r["obj_treatment_mm"] for r in X])) if X else float("nan"),
            "chunk_motion_mm_median": float(np.median([r["chunk_motion_mm"] for r in X])) if X else float("nan"),
            "U_mse_median": float(np.median([r["U_mse_global"] for r in X])) if X else float("nan"),
            "D_median": float(np.median([r["D_median"] for r in X])) if X else float("nan"),
            "G_median": float(np.median([r["G_median"] for r in X])) if X else float("nan"),
            "timeout_failures": int(sum(1 for e in valid if e["success"] is False
                                        and e.get("num_actions", 0) >= 250)),
            "early_failures": int(sum(1 for e in valid if e["success"] is False
                                      and 0 < e.get("num_actions", 0) < 250)),
            "infrastructure_errors": int(sum(1 for e in E if e.get("error"))),
        })

    with open(run_dir / "PILOT_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- pre-registered severity selection (reads only success / treatment / nuisance)
    by_family = defaultdict(dict)
    for r in rows:
        if r["family"] != "ID":
            by_family[r["family"]][r["severity"]] = r
    id_row = next(r for r in rows if r["family"] == "ID")

    selection = {}
    for fam, sevs in by_family.items():
        ordered = [sevs[s] for s in SEV_ORDER if s in sevs]
        qualifying = [r for r in ordered if SUCCESS_LOW < r["success_rate"] < SUCCESS_HIGH]
        all_near_perfect = all(r["success_rate"] >= NO_EFFECT_SUCCESS for r in ordered)
        all_id_treatment = all(r["treatment_mm_median"] <= NO_EFFECT_TREATMENT_FACTOR * ID_TREATMENT_MM
                               for r in ordered)
        mild_collapsed = ordered and ordered[0]["success_rate"] < SUCCESS_LOW
        if all_near_perfect and all_id_treatment:
            verdict, chosen, why = "NO_EFFECT", None, "every severity keeps success >=95% and treatment near the ID value"
        elif mild_collapsed:
            verdict, chosen, why = "TOO_DESTRUCTIVE", None, "the mildest severity already gives success <20%"
        elif qualifying:
            verdict, chosen = "VALID", qualifying[0]["severity"]
            why = f"lightest severity with {SUCCESS_LOW:.0%} < success < {SUCCESS_HIGH:.0%}"
        else:
            eligible = [r for r in ordered if r["success_rate"] >= SUCCESS_LOW]
            if eligible:
                best = max(eligible, key=lambda r: r["treatment_over_nuisance_median"])
                verdict, chosen = "VALID", best["severity"]
                why = "fallback: max treatment/nuisance subject to success >= 20%"
            else:
                verdict, chosen, why = "TOO_DESTRUCTIVE", None, "no severity keeps success >= 20%"
        selection[fam] = {"verdict": verdict, "chosen_severity": chosen, "rule_applied": why,
                          "per_severity": {s: {k: sevs[s][k] for k in
                                               ["success_rate", "treatment_mm_median", "nuisance_mm_median",
                                                "treatment_over_nuisance_median", "treatment_vs_ID",
                                                "n_valid", "n_invalid", "n_states"]}
                                           for s in SEV_ORDER if s in sevs}}

    summary = {
        "ID_reference": {"success_rate": id_row["success_rate"],
                         "treatment_mm_median": id_row["treatment_mm_median"],
                         "nuisance_mm_median": id_row["nuisance_mm_median"],
                         "treatment_over_nuisance": id_row["treatment_over_nuisance_median"],
                         "round1_reference_mm": {"treatment": ID_TREATMENT_MM, "nuisance": ID_NUISANCE_MM}},
        "selection_rule_inputs": ["baseline success", "future treatment (mm)", "sampler nuisance (mm)"],
        "selection_rule_forbidden_inputs": ["D", "S", "D x S", "any selector recovery"],
        "selection": selection,
        "rows": rows,
    }
    json.dump(summary, open(run_dir / "pilot_summary.json", "w"), indent=2)

    print(f"ID reference: success {id_row['success_rate']:.0%}  treatment {id_row['treatment_mm_median']:.2f} mm  "
          f"nuisance {id_row['nuisance_mm_median']:.2f} mm  ratio {id_row['treatment_over_nuisance_median']:.2f}")
    print(f"(round-1 in-distribution reference: treatment {ID_TREATMENT_MM} mm, nuisance {ID_NUISANCE_MM} mm)\n")
    hdr = f"{'condition':26s} {'succ':>6s} {'n_ok/inv':>9s} {'treat mm':>9s} {'nuis mm':>8s} {'t/n':>6s} {'vs ID':>6s} {'fail(t/e)':>10s}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["family"], SEV_ORDER.index(x["severity"]) if x["severity"] in SEV_ORDER else -1)):
        print(f"{r['condition']:26s} {r['success_rate']:6.0%} {r['n_valid']:4d}/{r['n_invalid']:<4d} "
              f"{r['treatment_mm_median']:9.2f} {r['nuisance_mm_median']:8.2f} "
              f"{r['treatment_over_nuisance_median']:6.2f} {r['treatment_vs_ID']:6.2f} "
              f"{r['timeout_failures']:4d}/{r['early_failures']:<4d}")
    print("\n--- pre-registered severity selection")
    for fam, s in selection.items():
        print(f"  {fam:18s} {s['verdict']:16s} chosen={s['chosen_severity']}   ({s['rule_applied']})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = [f for f in SEV_ORDER and by_family]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for i, fam in enumerate(sorted(by_family)):
            sv = [by_family[fam][s] for s in SEV_ORDER if s in by_family[fam]]
            x = range(len(sv))
            ax[0].plot(x, [r["success_rate"] for r in sv], "o-", label=fam)
            ax[1].plot(x, [r["treatment_mm_median"] for r in sv], "o-", label=fam)
        for a, ylab, ref in ((ax[0], "baseline task success", None),
                             (ax[1], "future-error treatment (mm)", ID_TREATMENT_MM)):
            a.set_xticks(range(3)); a.set_xticklabels(SEV_ORDER); a.legend(fontsize=7); a.set_ylabel(ylab)
        ax[0].axhspan(SUCCESS_LOW, SUCCESS_HIGH, color="#cccccc", alpha=.4)
        ax[0].axhline(id_row["success_rate"], ls="--", c="k", lw=.8)
        ax[1].axhline(ID_TREATMENT_MM, ls="--", c="k", lw=.8, label="ID treatment")
        ax[1].axhline(ID_NUISANCE_MM, ls=":", c="r", lw=.8, label="sampler nuisance")
        ax[1].legend(fontsize=7)
        ax[0].set_title("severity vs success (shaded = pre-registered 20-90% window)")
        ax[1].set_title("severity vs future-error consequence")
        fig.tight_layout(); fig.savefig(run_dir / "figures" / "g3_pilot.png", dpi=150)
        print(f"\nfigure: {run_dir / 'figures' / 'g3_pilot.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[figure skipped] {exc}")


if __name__ == "__main__":
    main()
