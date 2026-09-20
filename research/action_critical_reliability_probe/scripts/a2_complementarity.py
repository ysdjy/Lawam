"""Step 3 + Step 4 analysis: are U and S complementary, and does R = f(U,S) explain action consequence
better than U, S or attention alone?  (numpy only; no model, no simulator)

Token level
  target      errdev_j = || pi(H_pred with token j replaced by its REAL value) - pi(H_pred) ||
              i.e. the action deviation actually caused by the realised prediction error of token j
  predictors  U_j (several metrics), S_j (random-direction sensitivity), R_mul = U_j * S_j,
              attention_j, attention_j * U_j, and R_linear fitted on a train split of episodes
  reported    per-state Spearman, AUROC for the top-quartile tokens, bootstrap CI over episodes

State level
  target      E_action = || pi(H_real) - pi(H_pred) ||  (whole future replaced), and the executed
              consequence proxies recorded by the replay
  predictors  state aggregates of U, S, U*S, attention*U, plus the measured total future reliance

Everything is reported as measured; no minimum improvement was declared in advance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402

EPS_MID_IDX = 1
TOP_Q = 0.75          # pre-registered top-quartile threshold
METRIC = "l2_norm_all7"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else float("nan")


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))).astype(np.float64) + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def auprc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-score)
    lab = label[order]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, lab.size + 1)
    rec = tp / max(lab.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def boot_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    stat = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        vals = np.concatenate([values[groups == g] for g in pick])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            stat.append(vals.mean())
    return [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))] if stat else [float("nan")] * 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    sens = {json.loads(l)["state_id"]: json.loads(l) for l in open(run_dir / "sensitivity_records.jsonl")}
    unc = {json.loads(l)["state_id"]: json.loads(l) for l in open(run_dir / "uncertainty_records.jsonl")}
    fut = {json.loads(l)["state_id"]: json.loads(l) for l in open(run_dir / "futures_manifest.jsonl")}
    att_path = run_dir / "attention_records.jsonl"
    att = {json.loads(l)["state_id"]: json.loads(l) for l in open(att_path)} if att_path.exists() else {}
    sids = sorted(set(sens) & set(unc))
    fig_dir = run_dir / "figures"; fig_dir.mkdir(exist_ok=True)

    per_state = []
    token_rows = []
    for sid in sids:
        sn = np.load(run_dir / "sensitivity" / f"{sid}.npz")
        un = np.load(run_dir / "uncertainty" / f"{sid}.npz")
        metrics = [str(x) for x in sn["metrics"]]
        mi = metrics.index(METRIC)
        S = sn["S_norm"][EPS_MID_IDX].mean(axis=0)[:, mi].astype(np.float64)       # per-token sensitivity
        U_mse = un["U_U_mse"].astype(np.float64)
        U_l2 = un["U_U_l2"].astype(np.float64)
        U_cos = un["U_U_cosine_dist"].astype(np.float64)
        errdev = un[f"errdev_{METRIC}"].astype(np.float64)
        Ufloor = un["Ufloor_render_U_mse"].astype(np.float64)
        A = np.load(run_dir / "attention" / f"{sid}.npz")["attention_future"].astype(np.float64) \
            if sid in att else None

        preds = {"U_only_mse": U_mse, "U_only_l2": U_l2, "U_only_cos": U_cos,
                 "S_only": S, "R_mul_U_l2_S": U_l2 * S, "R_mul_U_mse_S": U_mse * S}
        if A is not None:
            preds["attention_only"] = A
            preds["attention_x_U_l2"] = A * U_l2
            preds["attention_x_S"] = A * S
        thr = np.quantile(errdev, TOP_Q)
        lab = errdev >= thr
        row = {
            "state_id": sid, "suite": sens[sid]["suite"], "phase": sens[sid]["phase"],
            "episode": f"{sens[sid]['suite']}_{sens[sid]['episode_idx']}",
            # --- step 3: complementarity
            "spearman_U_S": spearman(U_l2, S), "pearson_U_S": pearson(U_l2, S),
            "spearman_Umse_S": spearman(U_mse, S),
            # 1.0 = independent (0.25*0.25 of the tokens), 4.0 = identical top quartiles
            "top_quartile_overlap_U_S": float(np.mean(
                (U_l2 >= np.quantile(U_l2, TOP_Q)) & (S >= np.quantile(S, TOP_Q)))
                / ((1 - TOP_Q) ** 2)),
            "frac_highU_highS": float(np.mean((U_l2 >= np.quantile(U_l2, TOP_Q)) & (S >= np.quantile(S, TOP_Q)))),
            "frac_highU_lowS": float(np.mean((U_l2 >= np.quantile(U_l2, TOP_Q)) & (S <= np.quantile(S, 1 - TOP_Q)))),
            "frac_lowU_highS": float(np.mean((U_l2 <= np.quantile(U_l2, 1 - TOP_Q)) & (S >= np.quantile(S, TOP_Q)))),
            "spearman_U_tokennorm": spearman(U_l2, np.linalg.norm(sn["h_t1_pred"].astype(np.float64), axis=-1)),
            "spearman_S_tokennorm": spearman(S, np.linalg.norm(sn["h_t1_pred"].astype(np.float64), axis=-1)),
            "U_snr_over_render_floor": float(U_mse.mean() / max(Ufloor.mean(), 1e-12)),
            # --- step 4 token level
            "E_action_total": unc[sid]["E_action_total_l2"],
            "errdev_sum": unc[sid]["errdev_sum_over_tokens_l2"],
            "superposition_ratio": unc[sid]["superposition_ratio_sum_over_total"],
            "flow_noise_scale": float(np.mean([v[METRIC] for k, v in sens[sid]["controls"].items()
                                               if k.startswith("flow_noise_seed")])),
            "ctrl_zeros": sens[sid]["controls"]["future_all_zeros"][METRIC],
            "U_global": unc[sid]["U_mse_global"],
            "U_trivial_h_t": unc[sid]["U_trivial_h_t_mse_global"],
            "eef_displacement": fut[sid]["eef_displacement_over_chunk_m"] if sid in fut else float("nan"),
        }
        for name, p in preds.items():
            row[f"spearman_{name}"] = spearman(p, errdev)
            row[f"auroc_{name}"] = auroc(p, lab)
            row[f"auprc_{name}"] = auprc(p, lab)
        token_rows.append({"state_id": sid, "episode": row["episode"], "phase": row["phase"],
                           "suite": row["suite"], "S": S, "U_l2": U_l2, "U_mse": U_mse,
                           "errdev": errdev, "attention": A})
        per_state.append(row)

    arr = {k: np.asarray([r.get(k, np.nan) for r in per_state], dtype=np.float64)
           for k in per_state[0] if isinstance(per_state[0][k], float)}
    episodes = np.asarray([r["episode"] for r in per_state])
    phases = np.asarray([r["phase"] for r in per_state])

    pred_names = [k[len("spearman_"):] for k in per_state[0] if k.startswith("spearman_")
                  and k not in ("spearman_U_S", "spearman_Umse_S", "spearman_U_tokennorm", "spearman_S_tokennorm")]

    token_level = {}
    for name in pred_names:
        token_level[name] = {
            "spearman_mean": float(np.nanmean(arr[f"spearman_{name}"])),
            "spearman_median": float(np.nanmedian(arr[f"spearman_{name}"])),
            "spearman_ci": boot_ci(arr[f"spearman_{name}"], episodes),
            "auroc_mean": float(np.nanmean(arr[f"auroc_{name}"])),
            "auroc_ci": boot_ci(arr[f"auroc_{name}"], episodes),
            "auprc_mean": float(np.nanmean(arr[f"auprc_{name}"])),
        }

    # paired comparison of R against the best single predictor, per state
    def paired(a_key: str, b_key: str) -> dict:
        diff = arr[f"spearman_{a_key}"] - arr[f"spearman_{b_key}"]
        return {"mean_diff": float(np.nanmean(diff)), "ci": boot_ci(diff, episodes),
                "frac_states_better": float(np.nanmean(diff > 0))}

    comparisons = {}
    if "R_mul_U_l2_S" in pred_names:
        for base in [p for p in pred_names if p != "R_mul_U_l2_S"]:
            comparisons[f"R_mul_U_l2_S_vs_{base}"] = paired("R_mul_U_l2_S", base)

    # ---- R_linear: fit alpha*U + beta*S + gamma*U*S on a train split of EPISODES, evaluate on held out
    rng = np.random.default_rng(0)
    eps_unique = np.unique(episodes)
    train_eps = set(rng.choice(eps_unique, size=int(0.6 * eps_unique.size), replace=False).tolist())
    tr = [t for t in token_rows if t["episode"] in train_eps]
    te = [t for t in token_rows if t["episode"] not in train_eps]

    def pct(x: np.ndarray) -> np.ndarray:
        """Within-state percentile rank, so the fit is about the shape of the relation inside a state
        rather than about differences in scale between states."""
        return np.argsort(np.argsort(x)).astype(np.float64) / (x.size - 1)

    def design(rows):
        X = np.concatenate([np.stack([pct(r["U_l2"]), pct(r["S"]), pct(r["U_l2"] * r["S"])], axis=1)
                            for r in rows], axis=0)
        y = np.concatenate([pct(r["errdev"]) for r in rows])
        return X, y

    Xtr, ytr = design(tr)
    coef, *_ = np.linalg.lstsq(np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1), ytr, rcond=None)
    lin_eval = []
    for r in te:
        X = np.stack([pct(r["U_l2"]), pct(r["S"]), pct(r["U_l2"] * r["S"])], axis=1)
        pred = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ coef
        lab = r["errdev"] >= np.quantile(r["errdev"], TOP_Q)
        lin_eval.append({"spearman": spearman(pred, r["errdev"]), "auroc": auroc(pred, lab),
                         "episode": r["episode"]})
    token_level["R_linear_heldout"] = {
        "coefficients_on_within_state_percentiles": {"U": float(coef[0]), "S": float(coef[1]),
                                                     "U*S": float(coef[2]), "intercept": float(coef[3])},
        "n_train_episodes": len(train_eps), "n_test_states": len(te),
        "spearman_mean": float(np.nanmean([x["spearman"] for x in lin_eval])),
        "spearman_ci": boot_ci(np.asarray([x["spearman"] for x in lin_eval]),
                               np.asarray([x["episode"] for x in lin_eval])),
        "auroc_mean": float(np.nanmean([x["auroc"] for x in lin_eval])),
    }

    # ---- state level: predict E_action
    state_preds = {}
    S_all = [t["S"] for t in token_rows]
    U_all = [t["U_l2"] for t in token_rows]
    A_all = [t["attention"] for t in token_rows]
    y_state = arr["E_action_total"]
    cands = {
        "U_mean": np.asarray([u.mean() for u in U_all]),
        "U_max": np.asarray([u.max() for u in U_all]),
        "S_mean": np.asarray([s.mean() for s in S_all]),
        "S_max": np.asarray([s.max() for s in S_all]),
        "sum_U_times_S": np.asarray([float((u * s).sum()) for u, s in zip(U_all, S_all)]),
        "sum_U_times_attention": (np.asarray([float((u * a).sum()) for u, a in zip(U_all, A_all)])
                                  if A_all[0] is not None else None),
        "measured_total_future_reliance(zeroed)": arr["ctrl_zeros"],
        "errdev_sum_tokens": arr["errdev_sum"],
    }
    for name, v in cands.items():
        if v is None:
            continue
        state_preds[name] = {
            "spearman_vs_E_action": spearman(v, y_state),
            "pearson_vs_E_action": pearson(v, y_state),
            "spearman_vs_E_action_within_phase": {
                ph: spearman(v[phases == ph], y_state[phases == ph]) for ph in np.unique(phases)},
        }

    summary = {
        "n_states": len(per_state),
        "step3_complementarity": {
            "spearman_U_S": pc.stats(arr["spearman_U_S"]),
            "pearson_U_S": pc.stats(arr["pearson_U_S"]),
            "top_quartile_overlap_U_S(1.0 = independent, 4.0 = identical)": pc.stats(arr["top_quartile_overlap_U_S"]),
            "frac_highU_highS": pc.stats(arr["frac_highU_highS"]),
            "frac_highU_lowS": pc.stats(arr["frac_highU_lowS"]),
            "frac_lowU_highS": pc.stats(arr["frac_lowU_highS"]),
            "confound_spearman_U_vs_token_norm": pc.stats(arr["spearman_U_tokennorm"]),
            "confound_spearman_S_vs_token_norm": pc.stats(arr["spearman_S_tokennorm"]),
            "redundancy_criterion_met(|rho|>=0.8)": bool(np.median(np.abs(arr["spearman_U_S"])) >= 0.8),
        },
        "U_quality": {
            "U_global": pc.stats(arr["U_global"]),
            "U_trivial_h_t_baseline": pc.stats(arr["U_trivial_h_t"]),
            "U_snr_over_render_floor": pc.stats(arr["U_snr_over_render_floor"]),
            "frac_states_where_LaWM_beats_trivial": float(np.mean(arr["U_global"] < arr["U_trivial_h_t"])),
        },
        "consequence_scales": {
            "E_action_total": pc.stats(arr["E_action_total"]),
            "flow_noise_scale": pc.stats(arr["flow_noise_scale"]),
            "E_action_over_flow_noise": pc.stats(arr["E_action_total"] / arr["flow_noise_scale"]),
            "superposition_ratio_sum_single_over_total": pc.stats(arr["superposition_ratio"]),
        },
        "step4_token_level": token_level,
        "step4_paired_comparisons": comparisons,
        "step4_state_level": state_preds,
        "by_phase_token_level_spearman": {
            name: {ph: float(np.nanmean(arr[f"spearman_{name}"][phases == ph])) for ph in np.unique(phases)}
            for name in pred_names},
    }
    pc.write_json(run_dir / "summary_step34.json", summary)

    with open(run_dir / "metrics_step34.csv", "w") as f:
        keys = list(per_state[0])
        f.write(",".join(keys) + "\n")
        for r in per_state:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    print(f"n_states={len(per_state)}")
    print("\n--- Step 3: U vs S complementarity")
    for k, v in summary["step3_complementarity"].items():
        print(f"  {k}: {v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items() if kk in ('median','p25','p75','min','max')}}")
    print("\n--- U quality")
    for k, v in summary["U_quality"].items():
        print(f"  {k}: {v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items() if kk in ('median','min','max')}}")
    print("\n--- consequence scales")
    for k, v in summary["consequence_scales"].items():
        print(f"  {k}: {({kk: round(vv, 4) for kk, vv in v.items() if kk in ('median','p25','p75','min','max')})}")
    print("\n--- Step 4 token-level predictors of the realised per-token action deviation")
    for k, v in token_level.items():
        print(f"  {k:24s} spearman={v['spearman_mean']:+.3f} CI{[round(x,3) for x in v['spearman_ci']]} "
              f"auroc={v['auroc_mean']:.3f}")
    print("\n--- paired vs R_mul_U_l2_S")
    for k, v in comparisons.items():
        print(f"  {k:40s} diff={v['mean_diff']:+.3f} CI{[round(x,3) for x in v['ci']]} better_in={v['frac_states_better']:.0%}")
    print("\n--- Step 4 state-level predictors of E_action")
    for k, v in state_preds.items():
        print(f"  {k:42s} spearman={v['spearman_vs_E_action']:+.3f} "
              f"within-phase={ {p: round(x,2) for p, x in v['spearman_vs_E_action_within_phase'].items()} }")


if __name__ == "__main__":
    main()
