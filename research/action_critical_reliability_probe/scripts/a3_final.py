"""Final integration: environment-level consequences + the decision inputs for the judgement tree.

Adds to the Step-3/4 correlational analysis the *causal* comparison that the correlations cannot make:
with the same token budget (64 of 256), does correcting the tokens selected by U x S recover more of the
oracle correction than correcting the tokens selected by U alone (or by attention x U)?

All CIs are paired bootstraps resampling EPISODES.
Outputs: <run_dir>/summary.json (merged), <run_dir>/metrics.csv, figures/step4_consequences.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402


def boot_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0, stat=np.mean):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        vals = np.concatenate([values[groups == g] for g in pick])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out.append(stat(vals))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cons = [json.loads(l) for l in open(run_dir / "consequence_records.jsonl")]
    cons.sort(key=lambda r: r["state_id"])
    step34 = json.loads((run_dir / "summary_step34.json").read_text())
    step1 = json.loads((run_dir / "summary_step1.json").read_text())

    ep = np.asarray([f"{c['suite']}_{c['episode_idx']}" for c in cons])
    ph = np.asarray([c["phase"] for c in cons])
    mm = lambda k: np.asarray([c[k] for c in cons]) * 1000.0  # noqa: E731

    treat = mm("eef_treatment_m")
    nuis = mm("eef_nuisance_m")
    res_u = mm("eef_topU_residual_to_real_m")
    res_us = mm("eef_topUS_residual_to_real_m")
    motion = mm("eef_displacement_of_chunk_m")

    # fraction of the oracle correction recovered by each equal-budget selection
    rec_u = 1.0 - res_u / np.maximum(treat, 1e-9)
    rec_us = 1.0 - res_us / np.maximum(treat, 1e-9)
    diff = rec_us - rec_u

    env = {
        "n_states": len(cons),
        "token_budget": cons[0]["token_budget"],
        "overlap_topU_topUS_tokens": float(np.mean([c["overlap_topU_topUS"] for c in cons])),
        "eef_displacement_of_chunk_mm": pc.stats(motion),
        "treatment_eef_mm(pi(H_real) vs pi(H_pred))": pc.stats(treat),
        "nuisance_eef_mm(flow noise seed)": pc.stats(nuis),
        "treatment_over_nuisance": pc.stats(treat / np.maximum(nuis, 1e-9)),
        "treatment_as_fraction_of_chunk_motion": pc.stats(treat / motion),
        "treatment_mean_ci": boot_ci(treat, ep),
        "nuisance_mean_ci": boot_ci(nuis, ep),
        "object_treatment_mm": pc.stats(mm("obj_treatment_m")),
        "object_nuisance_mm": pc.stats(mm("obj_nuisance_m")),
        "gripper_command_changed_states": int(sum(c["gripper_cmd_differs"] for c in cons)),
        "recovered_fraction_topU": pc.stats(rec_u),
        "recovered_fraction_topUS": pc.stats(rec_us),
        "recovered_fraction_topU_mean_ci": boot_ci(rec_u, ep),
        "recovered_fraction_topUS_mean_ci": boot_ci(rec_us, ep),
        "paired_diff_topUS_minus_topU": {
            "mean": float(np.mean(diff)), "median": float(np.median(diff)),
            "ci": boot_ci(diff, ep),
            "frac_states_topUS_better": float(np.mean(diff > 0)),
        },
        "by_phase": {p: {
            "treatment_mm": float(np.median(treat[ph == p])),
            "nuisance_mm": float(np.median(nuis[ph == p])),
            "ratio": float(np.median(treat[ph == p]) / np.median(nuis[ph == p])),
            "recovered_topU": float(np.median(rec_u[ph == p])),
            "recovered_topUS": float(np.median(rec_us[ph == p])),
        } for p in np.unique(ph)},
    }

    merged = {"step1_sensitivity": step1, "step3_step4": step34, "step4_environment": env}
    pc.write_json(run_dir / "summary.json", merged)

    print("--- Step 4 Level 2: executed consequences (mm of end-effector displacement)")
    for k in ["eef_displacement_of_chunk_mm", "treatment_eef_mm(pi(H_real) vs pi(H_pred))",
              "nuisance_eef_mm(flow noise seed)", "treatment_over_nuisance",
              "treatment_as_fraction_of_chunk_motion", "object_treatment_mm", "object_nuisance_mm"]:
        v = env[k]
        print(f"  {k:46s} median={v['median']:.4g}  p25={v['p25']:.4g}  p75={v['p75']:.4g}  max={v['max']:.4g}")
    print(f"  treatment mean CI (episode bootstrap): {[round(x,3) for x in env['treatment_mean_ci']]} mm")
    print(f"  nuisance  mean CI                    : {[round(x,3) for x in env['nuisance_mean_ci']]} mm")
    print(f"  gripper command changed in {env['gripper_command_changed_states']} / {env['n_states']} states")
    print("\n--- equal-budget causal comparison (correct 64 of 256 tokens with the real value)")
    print(f"  recovered fraction, top-U   : median={env['recovered_fraction_topU']['median']:.3f} "
          f"mean CI={[round(x,3) for x in env['recovered_fraction_topU_mean_ci']]}")
    print(f"  recovered fraction, top-U*S : median={env['recovered_fraction_topUS']['median']:.3f} "
          f"mean CI={[round(x,3) for x in env['recovered_fraction_topUS_mean_ci']]}")
    d = env["paired_diff_topUS_minus_topU"]
    print(f"  paired difference (U*S - U) : mean={d['mean']:+.3f} CI={[round(x,3) for x in d['ci']]} "
          f"better in {d['frac_states_topUS_better']:.0%} of states")
    print(f"  token overlap between the two selections: {env['overlap_topU_topUS_tokens']:.1f} / {env['token_budget']}")
    print("\n--- by phase")
    for p, v in env["by_phase"].items():
        print(f"  {p:14s} treatment={v['treatment_mm']:.2f} mm  nuisance={v['nuisance_mm']:.2f} mm  "
              f"ratio={v['ratio']:.2f}  recovered U={v['recovered_topU']:.2f} U*S={v['recovered_topUS']:.2f}")

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].boxplot([treat, nuis], labels=["prediction error\n(pi(H_real) vs pi(H_pred))", "sampler noise\n(different seed)"])
        ax[0].set_ylabel("executed eef displacement difference (mm)")
        ax[0].set_title(f"consequence vs nuisance\n(chunk itself moves {np.median(motion):.0f} mm)")
        ax[1].boxplot([rec_u, rec_us], labels=["top-U tokens", "top-(U x S) tokens"])
        ax[1].set_ylabel("fraction of oracle correction recovered")
        ax[1].set_title(f"equal budget: {cons[0]['token_budget']}/256 tokens corrected")
        ax[1].axhline(0, color="grey", lw=0.8)
        names = ["U only", "attention", "attention x U", "S only", "attention x S", "U x S", "R linear"]
        keys = ["U_only_l2", "attention_only", "attention_x_U_l2", "S_only", "attention_x_S",
                "R_mul_U_l2_S", "R_linear_heldout"]
        vals = [step34["step4_token_level"][k]["spearman_mean"] for k in keys]
        cis = [step34["step4_token_level"][k]["spearman_ci"] for k in keys]
        ax[2].barh(range(len(vals)), vals, xerr=[[v - c[0] for v, c in zip(vals, cis)],
                                                 [c[1] - v for v, c in zip(vals, cis)]],
                   color=["#999999", "#ddaa33", "#ddaa33", "#4477aa", "#ddaa33", "#cc6677", "#cc6677"])
        ax[2].set_yticks(range(len(vals))); ax[2].set_yticklabels(names, fontsize=8)
        ax[2].set_xlabel("Spearman with the realised per-token action deviation")
        ax[2].set_title("token-level predictors (mean over 90 states)")
        fig.tight_layout(); fig.savefig(run_dir / "figures" / "step4_consequences.png", dpi=150)
        print(f"\nfigure: {run_dir / 'figures' / 'step4_consequences.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[figure skipped] {exc}")


if __name__ == "__main__":
    main()
