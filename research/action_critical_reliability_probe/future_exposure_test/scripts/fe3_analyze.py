"""FE analysis — the 2x2, the interaction, and the pre-registered decision rule.

Endpoint (locked): the Training Exposure x Future Quality INTERACTION on held-out adaptation
samples, not the absolute level of either checkpoint. Any prior exposure of the release
checkpoint to these demonstrations is common to both arms and cancels here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common as fc  # noqa: E402

BOOT = 2000
THRESH = {
    "material_flow_gain_relative": 0.02,
    "material_interaction_relative": 0.01,
    "deployment_noninferiority_margin_relative": 0.02,
}


def episode_bootstrap(values: np.ndarray, episodes: np.ndarray, salt: str,
                      reps: int = BOOT) -> tuple[float, float, float]:
    """Mean and a 95% CI resampling whole EPISODES, the unit used in every prior round."""
    uniq = np.unique(episodes)
    by_ep = {e: values[episodes == e] for e in uniq}
    rng = np.random.default_rng(fc.seed_from("FE2026-boot", salt) % (2 ** 32))
    means = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means[r] = np.concatenate([by_ep[e] for e in pick]).mean()
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_rows(d: Path) -> list[dict]:
    rows = []
    for p in sorted(d.glob("OFFLINE_FLOW_METRICS_*.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--primary-split", default="test_seen_task")
    ap.add_argument("--tag", default="best", choices=["best", "final", "u300"],
                    help="amendment FE-A6: the checkpoint the stopping rule selected is primary")
    args = ap.parse_args()

    rows = [r for r in load_rows(Path(args.metrics_dir)) if r.get("tag", "best") == args.tag]
    if not rows:
        raise SystemExit(f"no OFFLINE_FLOW_METRICS_*.csv with tag={args.tag} under {args.metrics_dir}")

    # key -> arm -> {P, G}
    table: dict = defaultdict(dict)
    for r in rows:
        key = (r["split"], int(r["episode"]), int(r["start"]), int(r["seed"]))
        table[key][r["arm"]] = (float(r["flow_loss_P"]), float(r["flow_loss_G"]))

    summary: dict = {"checkpoint_tag": args.tag, "thresholds": THRESH,
                     "bootstrap_replicates": BOOT, "by_seed": {}, "pooled": {}}

    seeds = sorted({int(r["seed"]) for r in rows})
    splits = sorted({r["split"] for r in rows})

    pooled_acc: dict = defaultdict(lambda: defaultdict(list))
    for split in splits:
        for seed in seeds:
            keys = [k for k in table if k[0] == split and k[3] == seed
                    and "control" in table[k] and "future_exposed" in table[k]]
            if not keys:
                continue
            keys.sort()
            eps = np.array([k[1] for k in keys])
            CP = np.array([table[k]["control"][0] for k in keys])
            CG = np.array([table[k]["control"][1] for k in keys])
            FP = np.array([table[k]["future_exposed"][0] for k in keys])
            FG = np.array([table[k]["future_exposed"][1] for k in keys])

            d_C = CP - CG
            d_F = FP - FG
            inter = d_F - d_C
            deploy = FP - CP

            # Amendment FE-A4: mandatory alongside the interaction. F_G_minus_C_P separates
            # "the channel carries usable information" from "the head learned to trust its
            # future slot"; delta_C is reported with its sign so an interaction inflated by the
            # Control arm's out-of-distribution penalty is visible rather than hidden.
            fg_minus_cp = FG - CP

            cell = {}
            for name, v in (("C_P", CP), ("C_G", CG), ("F_P", FP), ("F_G", FG),
                            ("delta_C", d_C), ("delta_F", d_F),
                            ("interaction", inter), ("deployment_cost", deploy),
                            ("F_G_minus_C_P", fg_minus_cp)):
                m, lo, hi = episode_bootstrap(v, eps, f"{split}|{seed}|{name}")
                cell[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
                pooled_acc[split][name].append((v, eps, seed))

            cell["n_samples"] = len(keys)
            cell["n_episodes"] = int(len(np.unique(eps)))
            cell["delta_F_relative_to_F_P"] = cell["delta_F"]["mean"] / cell["F_P"]["mean"]
            cell["delta_C_relative_to_C_P"] = cell["delta_C"]["mean"] / cell["C_P"]["mean"]
            cell["interaction_relative_to_C_P"] = cell["interaction"]["mean"] / cell["C_P"]["mean"]
            rel = deploy / CP
            m, lo, hi = episode_bootstrap(rel, eps, f"{split}|{seed}|deploy_rel")
            cell["deployment_cost_relative"] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
            summary["by_seed"].setdefault(split, {})[str(seed)] = cell

    # pooled over seeds (secondary summary only)
    for split in splits:
        if split not in summary["by_seed"]:
            continue
        cell = {}
        for name in ("C_P", "C_G", "F_P", "F_G", "delta_C", "delta_F", "interaction",
                     "deployment_cost", "F_G_minus_C_P"):
            vs = np.concatenate([v for v, _, _ in pooled_acc[split][name]])
            es = np.concatenate([e for _, e, _ in pooled_acc[split][name]])
            m, lo, hi = episode_bootstrap(vs, es, f"pooled|{split}|{name}")
            cell[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
        cell["delta_F_relative_to_F_P"] = cell["delta_F"]["mean"] / cell["F_P"]["mean"]
        cell["interaction_relative_to_C_P"] = cell["interaction"]["mean"] / cell["C_P"]["mean"]
        summary["pooled"][split] = cell

    # ------------------------------------------------------------------ decision rule
    ps = args.primary_split
    verdict, reasons = "FE-FAIL", []
    if ps in summary["by_seed"] and ps in summary["pooled"]:
        per_seed = summary["by_seed"][ps]
        pool = summary["pooled"][ps]
        c_a = all(v["delta_F"]["mean"] > 0 for v in per_seed.values())
        c_b = pool["delta_F"]["ci_lo"] > 0
        c_c = pool["delta_F_relative_to_F_P"] >= THRESH["material_flow_gain_relative"]
        c_d = (pool["interaction"]["ci_lo"] > 0
               and pool["interaction_relative_to_C_P"] >= THRESH["material_interaction_relative"])
        dep = [v["deployment_cost_relative"] for v in per_seed.values()]
        dep_hi = max(v["ci_hi"] for v in dep)
        dep_lo = min(v["ci_lo"] for v in dep)
        c_f = dep_hi < THRESH["deployment_noninferiority_margin_relative"]
        reasons = {
            "a_delta_F_positive_in_all_seeds": bool(c_a),
            "b_pooled_delta_F_lower_bound_above_zero": bool(c_b),
            "c_delta_F_at_least_2pct_of_F_P": bool(c_c),
            "d_interaction_positive_and_at_least_1pct_of_C_P": bool(c_d),
            "f_deployment_upper_bound_below_2pct": bool(c_f),
            "deployment_cost_relative_ci": {"lo": dep_lo, "hi": dep_hi},
            "note_e_action_metrics": "checked separately against OFFLINE_ACTION_METRICS_*.csv",
        }
        core = c_a and c_b and c_c and c_d
        if core and c_f:
            verdict = "FE-PASS"
        elif core and dep_lo > THRESH["deployment_noninferiority_margin_relative"]:
            verdict = "FE-TRADEOFF"
        else:
            verdict = "FE-FAIL"
    summary["primary_split"] = ps
    summary["decision_checks"] = reasons
    if ps in summary["pooled"]:
        pool = summary["pooled"][ps]
        summary["FE_A4_mandatory_context"] = {
            "F_G_minus_C_P": pool["F_G_minus_C_P"],
            "delta_C": pool["delta_C"],
            "channel_beats_status_quo": bool(pool["F_G_minus_C_P"]["ci_hi"] < 0),
            "interaction_may_be_inflated_by_control_OOD": bool(pool["delta_C"]["mean"] < 0),
            "reading_rule": "If the interaction is positive while F_G - C_P >= 0, exposure taught "
                            "the head to condition on its future slot WITHOUT beating the status "
                            "quo; the verdict must not be read as evidence that the future "
                            "channel carries usable action information.",
        }
    summary["verdict_flow_endpoint_only"] = verdict
    summary["verdict_note"] = ("Condition (e), the secondary action metrics, is evaluated "
                               "alongside this and folded into the reported verdict in "
                               "REPORT_FE.md. This field reports the flow endpoint alone.")

    # ------------------------------------------------------------------ secondary: action errors
    # Decision condition (e): the action metrics must point the same way as the flow endpoint.
    # Reported in millimetres beside the round-1 reference scales, never as a bare percentage.
    act_rows = []
    for p in sorted(Path(args.metrics_dir).glob("OFFLINE_ACTION_METRICS_*.csv")):
        with open(p) as f:
            act_rows.extend([r for r in csv.DictReader(f) if r.get("tag", "best") == args.tag])
    if act_rows:
        atab: dict = defaultdict(dict)
        for r in act_rows:
            atab[(r["split"], int(r["episode"]), int(r["start"]), int(r["seed"]))][r["arm"]] = r
        action_summary: dict = {}
        for metric in ("l2_norm_all7", "translation_mm_equiv"):
            for split in splits:
                for seed in seeds:
                    keys = [k for k in atab if k[0] == split and k[3] == seed
                            and "control" in atab[k] and "future_exposed" in atab[k]]
                    if not keys:
                        continue
                    keys.sort()
                    eps = np.array([k[1] for k in keys])
                    CP = np.array([float(atab[k]["control"][f"{metric}_P"]) for k in keys])
                    CG = np.array([float(atab[k]["control"][f"{metric}_G"]) for k in keys])
                    FP = np.array([float(atab[k]["future_exposed"][f"{metric}_P"]) for k in keys])
                    FG = np.array([float(atab[k]["future_exposed"][f"{metric}_G"]) for k in keys])
                    cell = {}
                    for name, v in (("C_P", CP), ("C_G", CG), ("F_P", FP), ("F_G", FG),
                                    ("delta_C", CP - CG), ("delta_F", FP - FG),
                                    ("interaction", (FP - FG) - (CP - CG)),
                                    ("deployment_cost", FP - CP),
                                    ("F_G_minus_C_P", FG - CP)):
                        m, lo, hi = episode_bootstrap(v, eps, f"act|{metric}|{split}|{seed}|{name}")
                        cell[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
                    cell["n_samples"] = len(keys)
                    action_summary.setdefault(metric, {}).setdefault(split, {})[str(seed)] = cell
        summary["action_metrics"] = action_summary
        summary["action_metric_reference_scales_mm"] = {
            "in_distribution_chunk_motion": 85.0,
            "future_error_treatment_round1": 0.904,
            "flow_sampler_nuisance_round1": 0.474,
            "note": "translation_mm_equiv is on this scale; every action effect size must be read "
                    "against it, never reported as a bare percentage",
        }
        if ps in action_summary.get("l2_norm_all7", {}):
            dF = [v["delta_F"]["mean"] for v in action_summary["l2_norm_all7"][ps].values()]
            summary["decision_checks"]["e_action_metrics_same_direction"] = bool(
                all(x > 0 for x in dF)) if dF else None
    else:
        summary["action_metrics"] = None

    fc.write_json(Path(args.out), summary)
    print(json.dumps(summary, indent=2)[:8000])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
