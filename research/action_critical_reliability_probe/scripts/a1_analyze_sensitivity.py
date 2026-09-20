"""Step-1 analysis / AUDIT 1 decision (runs anywhere with numpy; no model, no simulator).

Answers exactly the pre-registered AUDIT 1 questions:
  1. Are the actions reproducible with the same H and the same flow noise?     (in-batch floor)
  2. Is S clearly above the numerical floor?
  3. Do different tokens have clearly different S?                              (p95/median, Gini)
  4. Does S change with the execution phase?
  5. Is the sensitivity ranking stable across epsilon and across directions?    (Spearman)
  6. Would Case A (S ~ 0) or Case B (uniform S) trigger?

and puts every number next to the reference scales that make it interpretable (flow-noise nuisance,
zeroing the future, replacing the future by h_t).

Outputs: <run_dir>/metrics_sensitivity.csv, <run_dir>/summary_step1.json, figures/step1_*.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402

EPS_MID_IDX = 1
METRIC = "l2_norm_all7"


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Resample whole episodes (groups), not individual states."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        vals = np.concatenate([values[groups == g] for g in pick])
        means.append(vals.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    recs = [json.loads(l) for l in open(run_dir / "sensitivity_records.jsonl")]
    sens_dir = run_dir / "sensitivity"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    rows = []
    per_state_S: dict[str, np.ndarray] = {}
    for r in recs:
        sid = r["state_id"]
        npz = np.load(sens_dir / f"{sid}.npz")
        metrics = [str(x) for x in npz["metrics"]]
        mi = metrics.index(METRIC)
        S_norm = npz["S_norm"]                    # [eps, dirs, tokens, metrics]
        S_raw = npz["S"]
        s_mid = S_norm[EPS_MID_IDX].mean(axis=0)[:, mi]
        per_state_S[sid] = s_mid
        ctrl = r["controls"]
        noise_scale = float(np.mean([ctrl[k][METRIC] for k in ctrl if k.startswith("flow_noise_seed")]))
        rows.append({
            "state_id": sid, "suite": r["suite"], "task_id": r["task_id"], "phase": r["phase"],
            "episode_idx": r["episode_idx"], "step": r["step"],
            "S_median": float(np.median(s_mid)), "S_p95": float(np.percentile(s_mid, 95)),
            "S_max": float(s_mid.max()), "S_min": float(s_mid.min()), "S_mean": float(s_mid.mean()),
            "S_p95_over_median": float(np.percentile(s_mid, 95) / max(np.median(s_mid), 1e-30)),
            "S_max_over_median": float(s_mid.max() / max(np.median(s_mid), 1e-30)),
            "S_gini": gini(s_mid),
            "raw_dev_median_eps_mid": float(np.median(S_raw[EPS_MID_IDX].mean(axis=0)[:, mi])),
            "raw_dev_max_eps_mid": float(S_raw[EPS_MID_IDX].mean(axis=0)[:, mi].max()),
            "raw_dev_max_eps_large": float(S_raw[2].mean(axis=0)[:, mi].max()),
            "raw_dev_max_mm_eps_mid": float(
                S_raw[EPS_MID_IDX].mean(axis=0)[:, metrics.index("translation_mm_equiv")].max()),
            "sum_tokens_raw_dev_eps_mid": float(S_raw[EPS_MID_IDX].mean(axis=0)[:, mi].sum()),
            "rank_stab_eps_small_mid": r["rank_stability_across_eps"]["small_vs_mid"],
            "rank_stab_eps_mid_large": r["rank_stability_across_eps"]["mid_vs_large"],
            "rank_stab_dirs_mean": float(np.mean(list(r["rank_stability_across_directions_eps_mid"].values()))),
            "linearity_mid_over_small": r["linearity_ratio_mid_over_small"],
            "linearity_large_over_mid": r["linearity_ratio_large_over_mid"],
            "ctrl_zeros": ctrl["future_all_zeros"][METRIC],
            "ctrl_replaced_by_h_t": ctrl["future_replaced_by_h_t"][METRIC],
            "ctrl_scaled_105": ctrl["future_scaled_1.05"][METRIC],
            "ctrl_permuted": ctrl["future_token_permuted"][METRIC],
            "ctrl_all_tokens_eps_mid": ctrl["all_tokens_perturbed_eps_mid"][METRIC],
            "ctrl_repeat_identical": ctrl["repeat_identical"][METRIC],
            "ctrl_flow_noise": noise_scale,
            "zeros_over_noise": ctrl["future_all_zeros"][METRIC] / max(noise_scale, 1e-30),
            "inbatch_baseline_vs_B1": r["inbatch_baseline_vs_B1_maxabs"],
            "delta_norm_rel_err": r["effective_delta_norm_rel_error_max"],
            "baseline_action_absmean": r["baseline_action_absmean_first7"],
        })

    with open(run_dir / "metrics_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    arr = {k: np.asarray([r[k] for r in rows]) for k in rows[0] if isinstance(rows[0][k], (int, float))}
    phases = np.asarray([r["phase"] for r in rows])
    suites = np.asarray([r["suite"] for r in rows])
    episodes = np.asarray([f"{r['suite']}_{r['episode_idx']}" for r in rows])

    def by(group: np.ndarray, key: str) -> dict:
        return {str(g): pc.stats(arr[key][group == g]) for g in np.unique(group)}

    # cross-state stability of the token ranking (is the same token important in different states?)
    sids = [r["state_id"] for r in rows]
    cross = {}
    for suite in np.unique(suites):
        idx = [i for i, s in enumerate(suites) if s == suite]
        vals = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                vals.append(spearman(per_state_S[sids[idx[a]]], per_state_S[sids[idx[b]]]))
        cross[str(suite)] = pc.stats(np.asarray(vals)) if vals else None

    # ---- pre-registered decision rules
    case_a_token = bool(arr["raw_dev_max_eps_mid"].max() < 1e-3)
    case_a_global = bool(np.median(arr["ctrl_zeros"]) < np.median(arr["ctrl_flow_noise"]))
    case_b = bool(np.median(arr["S_p95_over_median"]) < 2.0
                  or np.median(arr["rank_stab_eps_small_mid"]) < 0.3
                  or np.median(arr["rank_stab_dirs_mean"]) < 0.3)
    ci_zeros = bootstrap_ci(arr["ctrl_zeros"], episodes)
    ci_noise = bootstrap_ci(arr["ctrl_flow_noise"], episodes)
    ci_p95 = bootstrap_ci(arr["S_p95_over_median"], episodes)

    summary = {
        "n_states": len(rows),
        "n_states_by_suite_phase": {f"{s}/{p}": int(((suites == s) & (phases == p)).sum())
                                    for s in np.unique(suites) for p in np.unique(phases)},
        "floors": {
            "inbatch_baseline_floor": "0.0 by construction (verified in phase0_batch_noise.json)",
            "inbatch_baseline_vs_B1_maxabs_max": float(arr["inbatch_baseline_vs_B1"].max()),
            "ctrl_repeat_identical_max": float(arr["ctrl_repeat_identical"].max()),
            "ctrl_permuted_max": float(arr["ctrl_permuted"].max()),
            "effective_delta_norm_rel_error_max": float(arr["delta_norm_rel_err"].max()),
        },
        "magnitudes_l2_norm_all7": {
            "single_token_eps_mid_median_over_states": pc.stats(arr["raw_dev_median_eps_mid"]),
            "single_token_eps_mid_max_over_states": pc.stats(arr["raw_dev_max_eps_mid"]),
            "single_token_eps_large_max_over_states": pc.stats(arr["raw_dev_max_eps_large"]),
            "sum_over_tokens_eps_mid": pc.stats(arr["sum_tokens_raw_dev_eps_mid"]),
            "all_tokens_perturbed_eps_mid": pc.stats(arr["ctrl_all_tokens_eps_mid"]),
            "future_zeroed": pc.stats(arr["ctrl_zeros"]),
            "future_replaced_by_h_t": pc.stats(arr["ctrl_replaced_by_h_t"]),
            "future_scaled_1.05": pc.stats(arr["ctrl_scaled_105"]),
            "flow_noise_seed_change": pc.stats(arr["ctrl_flow_noise"]),
            "baseline_action_absmean": pc.stats(arr["baseline_action_absmean"]),
        },
        "translation_mm_equivalent": {
            "single_token_eps_mid_max": pc.stats(arr["raw_dev_max_mm_eps_mid"]),
        },
        "nonuniformity": {
            "S_p95_over_median": pc.stats(arr["S_p95_over_median"]),
            "S_max_over_median": pc.stats(arr["S_max_over_median"]),
            "S_gini": pc.stats(arr["S_gini"]),
        },
        "rank_stability": {
            "across_eps_small_mid": pc.stats(arr["rank_stab_eps_small_mid"]),
            "across_eps_mid_large": pc.stats(arr["rank_stab_eps_mid_large"]),
            "across_directions": pc.stats(arr["rank_stab_dirs_mean"]),
            "across_states_within_suite": cross,
        },
        "linearity": {
            "eps_mid_over_small_expected_3.0": pc.stats(arr["linearity_mid_over_small"]),
            "eps_large_over_mid_expected_3.33": pc.stats(arr["linearity_large_over_mid"]),
        },
        "by_phase": {k: by(phases, k) for k in
                     ["S_median", "S_p95_over_median", "ctrl_zeros", "ctrl_flow_noise", "zeros_over_noise",
                      "raw_dev_max_eps_mid"]},
        "by_suite": {k: by(suites, k) for k in
                     ["S_median", "S_p95_over_median", "ctrl_zeros", "zeros_over_noise"]},
        "bootstrap_ci_episode_resampled": {
            "future_zeroed_mean": ci_zeros,
            "flow_noise_mean": ci_noise,
            "S_p95_over_median_mean": ci_p95,
        },
        "decision_rules": {
            "case_A_trigger_token_part(max single-token deviation < 1e-3)": case_a_token,
            "case_A_trigger_global_part(median zeroed-future < median flow-noise)": case_a_global,
            "case_A_triggered(both)": bool(case_a_token and case_a_global),
            "case_B_triggered(uniform or rank-unstable)": case_b,
            "proceed_to_step2": bool(not (case_a_token and case_a_global) and not case_b),
        },
    }
    pc.write_json(run_dir / "summary_step1.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_states", "floors", "decision_rules")}, indent=1))
    print("\nmagnitudes (l2_norm_all7, median over states):")
    for k, v in summary["magnitudes_l2_norm_all7"].items():
        print(f"  {k:42s} median={v['median']:.3e}  [{v['min']:.2e}, {v['max']:.2e}]")
    print("\nby phase (median over states):")
    for k, v in summary["by_phase"].items():
        print(f"  {k:22s} " + "  ".join(f"{p}={v[p]['median']:.4g}" for p in sorted(v)))

    # ---------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        all_s = np.concatenate([per_state_S[s] for s in sids])
        axes[0].hist(np.log10(np.maximum(all_s, 1e-12)), bins=60, color="#4477aa")
        axes[0].set_xlabel("log10 S (per token, eps_mid, l2)"); axes[0].set_ylabel("count")
        axes[0].set_title(f"token sensitivity distribution\n{len(sids)} states x 256 tokens")
        for ph in np.unique(phases):
            vals = np.concatenate([per_state_S[s] for s, p in zip(sids, phases) if p == ph])
            axes[1].hist(np.log10(np.maximum(vals, 1e-12)), bins=50, alpha=0.5, label=ph, density=True)
        axes[1].legend(); axes[1].set_xlabel("log10 S"); axes[1].set_title("by execution phase")
        names = ["single token\n(eps mid, max)", "all tokens\nperturbed", "flow noise\nseed", "future\nscaled 1.05",
                 "future\nzeroed", "future ->\nh_t", "|action|\nmean"]
        vals = [np.median(arr[k]) for k in ["raw_dev_max_eps_mid", "ctrl_all_tokens_eps_mid", "ctrl_flow_noise",
                                            "ctrl_scaled_105", "ctrl_zeros", "ctrl_replaced_by_h_t",
                                            "baseline_action_absmean"]]
        axes[2].bar(range(len(vals)), vals, color=["#cc6677"] * 2 + ["#999999"] * 2 + ["#4477aa"] * 2 + ["#000000"])
        axes[2].set_yscale("log"); axes[2].set_xticks(range(len(vals)))
        axes[2].set_xticklabels(names, fontsize=7); axes[2].set_ylabel("action change (l2, normalized units)")
        axes[2].set_title("magnitudes in context (median over states)")
        fig.tight_layout(); fig.savefig(fig_dir / "step1_sensitivity_overview.png", dpi=150)
        print(f"\nfigure: {fig_dir / 'step1_sensitivity_overview.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[figure skipped] {exc}")


if __name__ == "__main__":
    main()
