"""Step 6b: aggregate records -> metrics_by_state.csv, confusion matrices, summary.json, figures (libero310 env: numpy+matplotlib)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

LABELS = ["A", "B", "OTHER", "FAIL", "UNKNOWN"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def majority(labels):
    c = Counter(labels)
    top, n = c.most_common(1)[0]
    return top if n > len(labels) / 2 else "MIXED"


def boot_ci(values_by_state, n_boot=2000, seed=0):
    """values_by_state: list of per-state rates. Paired bootstrap over states."""
    v = np.asarray(values_by_state, dtype=np.float64)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_diff_ci(a_by_state, b_by_state, n_boot=2000, seed=0):
    a = np.asarray(a_by_state, dtype=np.float64); b = np.asarray(b_by_state, dtype=np.float64)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), size=(n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase", default="confirm")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    recs = [r for r in load_jsonl(run_dir / "episode_records.jsonl") if args.phase == "all" or r["phase"] == args.phase]
    fut = {(f["state_id"], f["tag"]): f for f in load_jsonl(run_dir / "future_metrics.jsonl")}
    manifest = {m["state_id"]: m for m in load_jsonl(run_dir / "reference_manifest.jsonl")}
    states = sorted({r["state_id"] for r in recs})
    out_dir = run_dir / ("analysis_" + args.phase); out_dir.mkdir(exist_ok=True)

    # join
    rows = []
    for r in recs:
        tag = Path(r["files"]["execution"]).stem
        f = fut.get((r["state_id"], tag), {})
        rows.append({**r, "tag": tag, "predicted_branch": f.get("predicted_branch"), "pred_s_f": f.get("pred_s_f"), "predicted_branch_roi": f.get("predicted_branch_roi"),
                     "pred_mse_to_uA": f.get("pred_mse_to_uA"), "pred_mse_to_uB": f.get("pred_mse_to_uB"), "pred_mse_to_actual7": f.get("pred_mse_to_actual7"),
                     "pred_mse_to_actual8": f.get("pred_mse_to_actual8"), "pred_mse_to_h_t": f.get("pred_mse_to_h_t"), "actual7_mse_to_h_t": f.get("actual7_mse_to_h_t"),
                     "actual7_feature_label": f.get("actual7_feature_label"), "actual7_s_f": f.get("actual7_s_f"), "noise_floor_mse": f.get("noise_floor_mse"), "mse_uA_uB": f.get("mse_uA_uB"),
                     "pred_roi_mse_to_actual7": f.get("pred_roi_mse_to_actual7"), "pred_s_f_roi": f.get("pred_s_f_roi")})

    # ---------------- confusion matrices (B1, B2: input branch -> predicted / executed); B0 marginals; R sanity
    conf = {}
    for cond in ["B1", "B2"]:
        for kind in ["predicted_branch", "executed_branch"]:
            m = {ib: Counter() for ib in "AB"}
            for r in rows:
                if r["condition_group"] == cond:
                    m[r["reference_branch"]][str(r.get(kind))] += 1
            conf[f"{cond}__input_to_{kind}"] = {ib: {l: m[ib].get(l, 0) for l in LABELS + ["None"]} for ib in "AB"}
    conf["B0__predicted_marginal"] = dict(Counter(str(r.get("predicted_branch")) for r in rows if r["condition_group"] == "B0"))
    conf["B0__executed_marginal"] = dict(Counter(str(r.get("executed_branch")) for r in rows if r["condition_group"] == "B0"))
    conf["R__input_to_executed"] = {ib: dict(Counter(str(r["executed_branch"]) for r in rows if r["condition_group"] == "R" and r["reference_branch"] == ib)) for ib in "AB"}
    conf["R__full_task_success"] = {ib: dict(Counter(str(r["full_task_success"]) for r in rows if r["condition_group"] == "R" and r["reference_branch"] == ib)) for ib in "AB"}

    # ---------------- per-state metrics
    per_state = []
    for sid in states:
        rs = [r for r in rows if r["state_id"] == sid]
        d = {"state_id": sid, "snapshot_step": manifest[sid]["snapshot_step"], "sep_t7": manifest[sid]["sep_t7"]}
        R = {ib: next(r for r in rs if r["condition_group"] == "R" and r["reference_branch"] == ib) for ib in "AB"}
        d["R_exec_A"] = R["A"]["executed_branch"]; d["R_exec_B"] = R["B"]["executed_branch"]
        d["R_success_A"] = R["A"]["full_task_success"]; d["R_success_B"] = R["B"]["full_task_success"]
        d["R_replay_dist_A_m"] = R["A"]["dist_to_pA_t7"]; d["R_replay_dist_B_m"] = R["B"]["dist_to_pB_t7"]
        b0 = [r for r in rs if r["condition_group"] == "B0"]
        d["B0_exec_labels"] = "/".join(str(r["executed_branch"]) for r in b0); d["B0_exec_majority"] = majority([r["executed_branch"] for r in b0])
        d["B0_pred_label"] = b0[0].get("predicted_branch") if b0 else None; d["B0_pred_s_f"] = b0[0].get("pred_s_f") if b0 else None
        d["B0_success_rate"] = float(np.mean([bool(r["full_task_success"]) for r in b0])) if b0 else None
        d["B0_mean_s_e"] = float(np.nanmean([r["s_e"] for r in b0])) if b0 else None
        for cond in ["B1", "B2"]:
            for ib in "AB":
                cs = [r for r in rs if r["condition_group"] == cond and r["reference_branch"] == ib]
                d[f"{cond}_{ib}_exec_labels"] = "/".join(str(r["executed_branch"]) for r in cs)
                d[f"{cond}_{ib}_follow_rate"] = float(np.mean([r["executed_branch"] == ib for r in cs])) if cs else None
                d[f"{cond}_{ib}_mean_s_e"] = float(np.nanmean([r["s_e"] for r in cs])) if cs else None
                d[f"{cond}_{ib}_pred_label"] = cs[0].get("predicted_branch") if cs else None
                d[f"{cond}_{ib}_pred_s_f"] = cs[0].get("pred_s_f") if cs else None
                d[f"{cond}_{ib}_pred_mse_actual7"] = float(np.nanmean([r["pred_mse_to_actual7"] for r in cs if r.get("pred_mse_to_actual7") is not None])) if cs else None
                d[f"{cond}_{ib}_full_chain_rate"] = float(np.mean([(r.get("predicted_branch") == ib) and (r["executed_branch"] == ib) for r in cs])) if cs else None
                d[f"{cond}_{ib}_success_rate"] = float(np.mean([bool(r["full_task_success"]) for r in cs])) if cs else None
                d[f"{cond}_{ib}_n_fail"] = sum(r["short_horizon_status"] == "FAIL" for r in cs)
            fa, fb = d[f"{cond}_A_follow_rate"], d[f"{cond}_B_follow_rate"]
            d[f"{cond}_both_branches_kept"] = (fa is not None and fb is not None and fa > 0.5 and fb > 0.5)
            d[f"{cond}_exec_switch_score"] = None if fa is None else float(d[f"{cond}_A_mean_s_e"] - d[f"{cond}_B_mean_s_e"])  # >0 means executed side follows input
        per_state.append(d)
    keys = list(per_state[0].keys()) if per_state else []
    with open(out_dir / "metrics_by_state.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_state)

    # ---------------- summary
    def rate(cond, ib, key):
        v = [d[f"{cond}_{ib}_{key}"] for d in per_state if d.get(f"{cond}_{ib}_{key}") is not None]
        return {"mean": float(np.mean(v)) if v else None, "ci95_paired_bootstrap_states": boot_ci(v), "n_states": len(v)}
    summary = {"phase": args.phase, "n_states": len(states), "n_records": len(rows), "n_infra_failures": sum(r["short_horizon_status"] == "FAIL" for r in rows),
               "counts_by_condition": dict(Counter(r["condition_group"] for r in rows)),
               "R": {"exec_A_correct": sum(d["R_exec_A"] == "A" for d in per_state), "exec_B_correct": sum(d["R_exec_B"] == "B" for d in per_state),
                     "success_A": sum(bool(d["R_success_A"]) for d in per_state), "success_B": sum(bool(d["R_success_B"]) for d in per_state),
                     "replay_dist_max_m": float(max(max(d["R_replay_dist_A_m"], d["R_replay_dist_B_m"]) for d in per_state))},
               "B0": {"exec_majority_counts": dict(Counter(d["B0_exec_majority"] for d in per_state)), "pred_label_counts": dict(Counter(str(d["B0_pred_label"]) for d in per_state)),
                      "success_rate_mean": float(np.mean([d["B0_success_rate"] for d in per_state if d["B0_success_rate"] is not None]))}}
    for cond in ["B1", "B2"]:
        summary[cond] = {ib: {"follow_rate": rate(cond, ib, "follow_rate"), "full_chain_rate": rate(cond, ib, "full_chain_rate"), "success_rate": rate(cond, ib, "success_rate"),
                              "pred_label_counts": dict(Counter(str(d[f"{cond}_{ib}_pred_label"]) for d in per_state)), "n_fail": int(sum(d[f"{cond}_{ib}_n_fail"] for d in per_state))} for ib in "AB"}
        summary[cond]["both_branches_kept_states"] = int(sum(d[f"{cond}_both_branches_kept"] for d in per_state))
        summary[cond]["exec_switch_score_mean_s_e_A_minus_B"] = {"mean": float(np.mean([d[f"{cond}_exec_switch_score"] for d in per_state])), "ci95": boot_ci([d[f"{cond}_exec_switch_score"] for d in per_state])}
    summary["B2_minus_B1_followB_rate_diff_ci95"] = boot_diff_ci([d["B2_B_follow_rate"] for d in per_state], [d["B1_B_follow_rate"] for d in per_state])
    # future prediction errors
    fp = {}
    for cond in ["B0", "B1"]:
        cs = [r for r in rows if r["condition_group"] == cond and r.get("pred_mse_to_actual7") is not None]
        if cond == "B0":
            fp[cond] = {"pred_mse_to_actual7": float(np.mean([r["pred_mse_to_actual7"] for r in cs])), "pred_mse_to_uA": float(np.mean([r["pred_mse_to_uA"] for r in cs])),
                        "pred_mse_to_uB": float(np.mean([r["pred_mse_to_uB"] for r in cs])), "pred_mse_to_h_t": float(np.mean([r["pred_mse_to_h_t"] for r in cs])),
                        "actual7_mse_to_h_t": float(np.mean([r["actual7_mse_to_h_t"] for r in cs])), "mean_s_f": float(np.mean([r["pred_s_f"] for r in cs if r["pred_s_f"] is not None and np.isfinite(r["pred_s_f"])]))}
        else:
            fp[cond] = {}
            for ib in "AB":
                cb = [r for r in cs if r["reference_branch"] == ib]
                fp[cond][ib] = {"pred_mse_to_correct_anchor": float(np.mean([r[f"pred_mse_to_u{ib}"] for r in cb])), "pred_mse_to_wrong_anchor": float(np.mean([r["pred_mse_to_u" + ("B" if ib == "A" else "A")] for r in cb])),
                                "pred_mse_to_actual7": float(np.mean([r["pred_mse_to_actual7"] for r in cb])), "pred_roi_mse_to_actual7": float(np.nanmean([r["pred_roi_mse_to_actual7"] for r in cb if r.get("pred_roi_mse_to_actual7") is not None])),
                                "mean_s_f": float(np.mean([r["pred_s_f"] for r in cb])), "mean_s_f_roi": float(np.nanmean([r["pred_s_f_roi"] for r in cb if r.get("pred_s_f_roi") is not None])),
                                "pred_mse_to_h_t": float(np.mean([r["pred_mse_to_h_t"] for r in cb]))}
    ra = [r for r in rows if r["condition_group"] == "R"]
    fp["reference_anchor_separation"] = {"mse_uA_uB_mean": float(np.mean([r["mse_uA_uB"] for r in ra if r.get("mse_uA_uB")])), "noise_floor_mse_mean": float(np.mean([r["noise_floor_mse"] for r in ra if r.get("noise_floor_mse")])),
                                          "actual7_feature_label_counts_R": {ib: dict(Counter(str(r.get("actual7_feature_label")) for r in ra if r["reference_branch"] == ib)) for ib in "AB"}}
    summary["future_prediction"] = fp
    summary["confusion"] = conf
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "confusion_matrices.json", "w") as f:
        json.dump(conf, f, indent=2)

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = run_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, cond in zip(axes, ["B0", "B1", "B2"]):
        for ib, col in [("A", "tab:blue"), ("B", "tab:red"), (None, "tab:gray")]:
            xs = [r["s_e"] for r in rows if r["condition_group"] == cond and r["reference_branch"] == ib and np.isfinite(r["s_e"])]
            if xs:
                ax.hist(xs, bins=np.linspace(-1.2, 1.2, 25), alpha=0.6, color=col, label=f"input {ib}" if ib else "no input")
        ax.axvline(0.25, ls="--", c="k"); ax.axvline(-0.25, ls="--", c="k")
        ax.set_title(f"{cond}: executed position score s_e at t+7 (A=+1, B=-1)"); ax.set_xlabel("s_e"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / f"{args.phase}_executed_s_e_hist.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, cond in zip(axes, ["B0", "B1"]):
        for ib, col in [("A", "tab:blue"), ("B", "tab:red"), (None, "tab:gray")]:
            xs = [r["pred_s_f"] for r in rows if r["condition_group"] == cond and r["reference_branch"] == ib and r.get("pred_s_f") is not None and np.isfinite(r["pred_s_f"])]
            if xs:
                ax.hist(xs, bins=np.linspace(-0.6, 0.6, 25), alpha=0.6, color=col, label=f"input {ib}" if ib else "no input")
        ax.axvline(0.05, ls="--", c="k"); ax.axvline(-0.05, ls="--", c="k")
        ax.set_title(f"{cond}: predicted-future anchor score s_f (A>0, B<0)"); ax.set_xlabel("s_f"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / f"{args.phase}_predicted_s_f_hist.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = []; vals = []; errs = []
    for cond in ["B1", "B2"]:
        for ib in "AB":
            r_ = summary[cond][ib]["follow_rate"]; labels.append(f"{cond} input {ib}"); vals.append(r_["mean"]); ci = r_["ci95_paired_bootstrap_states"]; errs.append([r_["mean"] - ci[0], ci[1] - r_["mean"]])
    ax.bar(labels, vals, yerr=np.array(errs).T, capsize=4, color=["tab:blue", "tab:red", "tab:blue", "tab:red"]); ax.set_ylim(0, 1.05); ax.set_ylabel("executed-branch follow rate (per-state mean, 95% CI)")
    fig.tight_layout(); fig.savefig(fig_dir / f"{args.phase}_follow_rates.png", dpi=130); plt.close(fig)
    print(json.dumps({k: summary[k] for k in ["n_states", "n_records", "n_infra_failures", "R", "B0"]}, indent=1))
    for cond in ["B1", "B2"]:
        print(cond, {ib: {"follow": summary[cond][ib]["follow_rate"]["mean"], "chain": summary[cond][ib]["full_chain_rate"]["mean"], "pred": summary[cond][ib]["pred_label_counts"]} for ib in "AB"}, "both_kept", summary[cond]["both_branches_kept_states"])
    print("future:", json.dumps(fp, indent=1))
    print("confusion:", json.dumps(conf, indent=1))


if __name__ == "__main__":
    main()
