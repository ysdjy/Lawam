"""Headroom analysis — paired, endpoint-restricted, no invented metrics.

Primary endpoint   final task success (0/1), the simulator's own signal
Secondary endpoint remaining steps to success, counted from the intervention state, successes only

The four variants of a snapshot are PAIRED. Binary contrasts use the paired difference, an exact McNemar
test and an episode-level paired bootstrap CI; step contrasts use a paired bootstrap over episodes. The
statistical unit is the snapshot/episode, never the action step.

Required case counts:
  case1 rescue              baseline fail  -> D x S success
  case2 damage              baseline succeed -> D x S fail
  case3 oracle headroom     baseline fail  -> full oracle success
  case4 beyond one chunk    baseline fail  -> full oracle also fail

Outputs: <run_dir>/HEADROOM_RESULTS.csv, RESCUE_CASES.csv, SUMMARY_HEADROOM.json, figures/h_headroom.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/learned_u/scripts"))
import g2_dataset as gd  # noqa: E402  (episode bootstrap)

VARIANTS = ["A_baseline", "B_top_D", "C_top_DS", "D_full_oracle"]
LABEL = {"A_baseline": "baseline", "B_top_D": "top-D repair",
         "C_top_DS": "top-(D x S) repair", "D_full_oracle": "full oracle"}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (b = a-only wins, c = b-only wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def paired_boot(diff: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    return gd.boot_ci(diff, groups, n_boot=n_boot, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--schedule", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    recs = [json.loads(l) for l in open(run_dir / "continuation_records.jsonl")]
    main_recs = [r for r in recs if r["schedule"] == args.schedule]

    by_state: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in main_recs:
        by_state[r["state_id"]][r["variant"]] = r
    states = [s for s, v in by_state.items() if len(v) == 4]
    assert states, "no complete state found"

    rows = []
    for sid in sorted(states):
        v = by_state[sid]
        a = v["A_baseline"]
        row = {"state_id": sid, "condition": a["condition"], "family": a["family"], "suite": a["suite"],
               "task_id": a["task_id"], "episode_idx": a["episode_idx"], "episode": a["episode"],
               "source_episode_success": a["source_episode_success"],
               "intervention_step": a["intervention_step"], "step_budget": a["step_budget"]}
        for k in VARIANTS:
            row[f"success_{k}"] = int(v[k]["success"])
            row[f"steps_{k}"] = v[k]["steps_used"]
            row[f"rts_{k}"] = v[k]["remaining_steps_to_success"]
        row["case1_rescue_DxS"] = int(row["success_A_baseline"] == 0 and row["success_C_top_DS"] == 1)
        row["case2_damage_DxS"] = int(row["success_A_baseline"] == 1 and row["success_C_top_DS"] == 0)
        row["case3_oracle_rescue"] = int(row["success_A_baseline"] == 0 and row["success_D_full_oracle"] == 1)
        row["case4_oracle_also_fail"] = int(row["success_A_baseline"] == 0 and row["success_D_full_oracle"] == 0)
        row["rescue_topD"] = int(row["success_A_baseline"] == 0 and row["success_B_top_D"] == 1)
        row["damage_topD"] = int(row["success_A_baseline"] == 1 and row["success_B_top_D"] == 0)
        row["damage_oracle"] = int(row["success_A_baseline"] == 1 and row["success_D_full_oracle"] == 0)
        rows.append(row)

    with open(run_dir / "HEADROOM_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    interesting = [r for r in rows if r["case1_rescue_DxS"] or r["case2_damage_DxS"]
                   or r["case3_oracle_rescue"] or r["rescue_topD"] or r["damage_topD"]
                   or r["damage_oracle"]]
    with open(run_dir / "RESCUE_CASES.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(interesting)

    ep = np.asarray([r["episode"] for r in rows])
    succ = {k: np.asarray([r[f"success_{k}"] for r in rows], dtype=float) for k in VARIANTS}

    def contrast(a: str, b: str) -> dict:
        d = succ[a] - succ[b]
        wins = int(np.sum(d > 0)); losses = int(np.sum(d < 0))
        return {"success_rate_a": float(succ[a].mean()), "success_rate_b": float(succ[b].mean()),
                "paired_diff": float(d.mean()), "ci": paired_boot(d, ep),
                "n_a_only": wins, "n_b_only": losses,
                "mcnemar_p": mcnemar_exact(wins, losses)}

    contrasts = {
        "DxS - baseline": contrast("C_top_DS", "A_baseline"),
        "DxS - topD": contrast("C_top_DS", "B_top_D"),
        "full_oracle - baseline": contrast("D_full_oracle", "A_baseline"),
        "topD - baseline": contrast("B_top_D", "A_baseline"),
        "full_oracle - DxS": contrast("D_full_oracle", "C_top_DS"),
    }

    # secondary endpoint: remaining steps to success, on states where BOTH variants succeeded
    def steps_contrast(a: str, b: str) -> dict:
        m = [(r[f"rts_{a}"], r[f"rts_{b}"], r["episode"]) for r in rows
             if r[f"rts_{a}"] is not None and r[f"rts_{b}"] is not None]
        if not m:
            return {"n_pairs": 0}
        d = np.asarray([x[0] - x[1] for x in m], dtype=float)
        g = np.asarray([x[2] for x in m])
        return {"n_pairs": len(m), "mean_diff_steps": float(d.mean()),
                "median_diff_steps": float(np.median(d)), "ci": paired_boot(d, g),
                "frac_a_faster": float(np.mean(d < 0))}

    steps = {k: steps_contrast(*k.split(" - ")[::-1]) if False else None for k in []}
    steps = {
        "DxS vs baseline": steps_contrast("C_top_DS", "A_baseline"),
        "DxS vs topD": steps_contrast("C_top_DS", "B_top_D"),
        "full_oracle vs baseline": steps_contrast("D_full_oracle", "A_baseline"),
    }

    def breakdown(keyf) -> dict:
        out = defaultdict(dict)
        for r in rows:
            out[str(keyf(r))].setdefault("_n", 0)
            out[str(keyf(r))]["_n"] += 1
        for k in out:
            sel = [r for r in rows if str(keyf(r)) == k]
            for v in VARIANTS:
                out[k][LABEL[v]] = float(np.mean([r[f"success_{v}"] for r in sel]))
        return dict(out)

    summary = {
        "n_states": len(rows), "schedule": args.schedule,
        "success_rate": {LABEL[k]: float(succ[k].mean()) for k in VARIANTS},
        "n_success": {LABEL[k]: int(succ[k].sum()) for k in VARIANTS},
        "contrasts_primary": contrasts,
        "secondary_remaining_steps": steps,
        "case_counts": {
            "case1_rescue_DxS": int(sum(r["case1_rescue_DxS"] for r in rows)),
            "case2_damage_DxS": int(sum(r["case2_damage_DxS"] for r in rows)),
            "case3_oracle_rescue": int(sum(r["case3_oracle_rescue"] for r in rows)),
            "case4_oracle_also_fail": int(sum(r["case4_oracle_also_fail"] for r in rows)),
            "rescue_topD": int(sum(r["rescue_topD"] for r in rows)),
            "damage_topD": int(sum(r["damage_topD"] for r in rows)),
            "damage_oracle": int(sum(r["damage_oracle"] for r in rows)),
            "n_baseline_fail": int(np.sum(succ["A_baseline"] == 0)),
        },
        "by_condition": breakdown(lambda r: r["condition"]),
        "by_suite": breakdown(lambda r: r["suite"]),
        "by_task": breakdown(lambda r: f"{r['suite']}_t{r['task_id']}"),
        "by_source_episode_outcome": breakdown(
            lambda r: "source_success" if r["source_episode_success"] else "source_failure"),
    }

    # robustness subset, if extra schedules exist
    extra = sorted({r["schedule"] for r in recs} - {args.schedule})
    if extra:
        sub = defaultdict(dict)
        for r in recs:
            if r["schedule"] in extra:
                sub[(r["state_id"], r["schedule"])][r["variant"]] = r
        comp = [v for v in sub.values() if len(v) == 4]
        srows = [{k: int(v[k]["success"]) for k in VARIANTS} | {"episode": v["A_baseline"]["episode"]}
                 for v in comp]
        if srows:
            g = np.asarray([r["episode"] for r in srows])
            s = {k: np.asarray([r[k] for r in srows], dtype=float) for k in VARIANTS}
            summary["robustness_subset"] = {
                "schedules": extra, "n_state_schedule_pairs": len(srows),
                "success_rate": {LABEL[k]: float(s[k].mean()) for k in VARIANTS},
                "DxS_minus_baseline": {"paired_diff": float((s["C_top_DS"] - s["A_baseline"]).mean()),
                                       "ci": paired_boot(s["C_top_DS"] - s["A_baseline"], g)},
                "oracle_minus_baseline": {"paired_diff": float((s["D_full_oracle"] - s["A_baseline"]).mean()),
                                          "ci": paired_boot(s["D_full_oracle"] - s["A_baseline"], g)},
            }

    json.dump(summary, open(run_dir / "SUMMARY_HEADROOM.json", "w"), indent=2)

    print(f"n_states={len(rows)} (schedule {args.schedule})\n")
    print("--- primary endpoint: final task success")
    for k in VARIANTS:
        print(f"  {LABEL[k]:22s} {summary['n_success'][LABEL[k]]:3d}/{len(rows)}  = {summary['success_rate'][LABEL[k]]:.3f}")
    print("\n--- paired contrasts (McNemar exact; CI = episode paired bootstrap)")
    for k, v in contrasts.items():
        print(f"  {k:24s} diff={v['paired_diff']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
              f"a_only={v['n_a_only']} b_only={v['n_b_only']} p={v['mcnemar_p']:.4f}")
    print("\n--- case counts")
    for k, v in summary["case_counts"].items():
        print(f"  {k:26s} {v}")
    print("\n--- secondary endpoint: remaining steps to success (successes in both variants)")
    for k, v in steps.items():
        if v.get("n_pairs"):
            print(f"  {k:26s} n={v['n_pairs']:3d} mean={v['mean_diff_steps']:+.2f} "
                  f"median={v['median_diff_steps']:+.1f} CI[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] "
                  f"faster_in={v['frac_a_faster']:.0%}")
        else:
            print(f"  {k:26s} no paired successes")
    print("\n--- success by condition / source outcome")
    for name in ["by_condition", "by_source_episode_outcome"]:
        for k, v in summary[name].items():
            print(f"  {k:24s} n={v['_n']:3d} " + "  ".join(f"{LABEL[x]}={v[LABEL[x]]:.3f}" for x in VARIANTS))
    if "robustness_subset" in summary:
        rs = summary["robustness_subset"]
        print(f"\n--- robustness subset (schedules {rs['schedules']}, {rs['n_state_schedule_pairs']} pairs)")
        for k, val in rs["success_rate"].items():
            print(f"  {k:22s} {val:.3f}")
        print(f"  DxS - baseline  diff={rs['DxS_minus_baseline']['paired_diff']:+.4f} "
              f"CI{[round(x,4) for x in rs['DxS_minus_baseline']['ci']]}")
        print(f"  oracle - baseline diff={rs['oracle_minus_baseline']['paired_diff']:+.4f} "
              f"CI{[round(x,4) for x in rs['oracle_minus_baseline']['ci']]}")


if __name__ == "__main__":
    main()


def make_figure(run_dir: Path) -> None:  # noqa: D103  (called from the shell after the analysis)
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = _json.loads((run_dir / "SUMMARY_HEADROOM.json").read_text())
    names = ["baseline", "top-D repair", "top-(D x S) repair", "full oracle"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].bar(range(4), [s["success_rate"][n] for n in names],
              color=["#999999", "#4477aa", "#cc6677", "#44aa99"])
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["baseline", "top-D", "top-(DxS)", "full oracle"], fontsize=8)
    ax[0].set_ylim(0, 1.0); ax[0].set_ylabel("final task success")
    ax[0].axhline(s["success_rate"]["baseline"], ls="--", c="k", lw=.8)
    ax[0].set_title(f"primary endpoint, {s['n_states']} states\n(one-chunk oracle intervention, then the unmodified policy)")
    keys = ["DxS - baseline", "topD - baseline", "full_oracle - baseline"]
    vals = [s["contrasts_primary"][k]["paired_diff"] for k in keys]
    cis = [s["contrasts_primary"][k]["ci"] for k in keys]
    err = [[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]
    ax[1].bar(range(3), vals, yerr=err, capsize=4,
              color=["#cc6677" if v > 0 else "#4477aa" for v in vals])
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["D x S\n- baseline", "top-D\n- baseline", "full oracle\n- baseline"], fontsize=8)
    ax[1].set_ylabel("paired difference in success rate")
    ax[1].set_title("paired contrasts (episode bootstrap CI)\nall intervals contain zero")
    c = s["case_counts"]
    labels = ["rescue\n(base fail ->\nDxS ok)", "damage\n(base ok ->\nDxS fail)",
              "oracle rescue", "oracle damage", "base fail &\noracle fails too"]
    counts = [c["case1_rescue_DxS"], c["case2_damage_DxS"], c["case3_oracle_rescue"],
              c["damage_oracle"], c["case4_oracle_also_fail"]]
    ax[2].bar(range(5), counts, color=["#44aa99", "#cc6677", "#44aa99", "#cc6677", "#999999"])
    ax[2].set_xticks(range(5)); ax[2].set_xticklabels(labels, fontsize=7)
    ax[2].set_ylabel("number of states")
    ax[2].set_title(f"case counts ({c['n_baseline_fail']} baseline failures in total)")
    fig.tight_layout(); fig.savefig(run_dir / "figures" / "h_headroom.png", dpi=150)
    print(f"figure: {run_dir / 'figures' / 'h_headroom.png'}")
