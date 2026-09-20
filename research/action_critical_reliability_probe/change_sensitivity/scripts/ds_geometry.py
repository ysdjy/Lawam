"""Step 1: why does D = ||H_pred - h_t|| predict U = ||H_pred - H_real||?  (numpy only, no model)

Decomposes the oracle error through the identity
    U^2 = D^2 + G^2 - 2*D*G*cos(theta)
with  D = ||Delta_pred||,  G = ||Delta_real||,  Delta_pred = H_pred - h_t,  Delta_real = H_real - h_t,
and reports, per state and per token, how much of Spearman(D, U) is length geometry rather than a
LaWAM-specific effect, using two counterfactuals that keep the magnitudes and destroy only the pairing:
    B4a  cos(theta) permuted across tokens within the state
    B4b  cos(theta) = 0 (the high-dimensional random-direction expectation)

Also fits U ~ f(D) per state (isotonic and log-log linear) and asks whether the frozen S explains the
residual that D cannot.

All inputs are read from the frozen round-1 run; nothing is recomputed there and S is never re-derived.
Outputs: <run_dir>/GEOMETRY_ANALYSIS.csv, <run_dir>/summary_geometry.json
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
import g2_dataset as gd  # noqa: E402  (spearman / pearson / boot_ci / frozen-S loaders)

PRIOR = REPO / "results/action_critical_reliability_probe/acr_20260919_203850"
EPS = 1e-12


def isotonic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators isotonic regression of y on x (increasing). Analysis only."""
    order = np.argsort(x)
    ys = y[order].astype(np.float64).copy()
    w = np.ones_like(ys)
    # PAVA
    i = 0
    lvl, lvl_w = [], []
    for val, wt in zip(ys, w):
        lvl.append(val); lvl_w.append(wt)
        while len(lvl) > 1 and lvl[-2] > lvl[-1]:
            v2, w2 = lvl.pop(), lvl_w.pop()
            v1, w1 = lvl.pop(), lvl_w.pop()
            lvl.append((v1 * w1 + v2 * w2) / (w1 + w2)); lvl_w.append(w1 + w2)
    fitted_sorted = np.concatenate([[v] * int(round(wt)) for v, wt in zip(lvl, lvl_w)])
    out = np.empty_like(ys)
    out[:] = fitted_sorted[: ys.size]
    res = np.empty_like(y, dtype=np.float64)
    res[order] = out
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(PRIOR / "states_manifest.jsonl")]

    rows = []
    consistency = []
    pooled = {k: [] for k in ["D", "G", "U", "cos", "S", "resid_iso", "resid_lin", "errdev"]}
    for m in manifest:
        sid = m["state_id"]
        s = np.load(PRIOR / "sensitivity" / f"{sid}.npz")
        u = np.load(PRIOR / "uncertainty" / f"{sid}.npz")
        h_t = s["h_t"].astype(np.float64)
        h_pred = s["h_t1_pred"].astype(np.float64)
        h_real = u["h_real"].astype(np.float64)
        U_stored = u["U_U_l2"].astype(np.float64)
        S = gd.load_S(sid)
        att = gd.load_attention(sid)
        errdev = u["errdev_l2_norm_all7"].astype(np.float64)

        d_pred = h_pred - h_t
        d_real = h_real - h_t
        D = np.linalg.norm(d_pred, axis=-1)
        G = np.linalg.norm(d_real, axis=-1)
        U = np.linalg.norm(h_pred - h_real, axis=-1)
        cos = (d_pred * d_real).sum(-1) / (D * G + EPS)

        # --- float16 consistency check (registered in PROTOCOL_DS.yaml:data.precision_note)
        consistency.append(float(np.max(np.abs(U - U_stored)) / max(np.median(U_stored), EPS)))

        # --- counterfactuals: keep D and G, destroy only the direction pairing
        rng = np.random.default_rng(int.from_bytes(sid.encode()[:8].ljust(8, b"0"), "big") % (2 ** 31))
        cos_perm = cos[rng.permutation(cos.size)]
        U_b4a = np.sqrt(np.maximum(D ** 2 + G ** 2 - 2 * D * G * cos_perm, 0.0))
        U_b4b = np.sqrt(D ** 2 + G ** 2)

        # --- residual of U after what D explains (per state, analysis only)
        logD, logU = np.log(D + EPS), np.log(U + EPS)
        a, b = np.polyfit(logD, logU, 1)
        resid_lin = U - np.exp(a * logD + b)
        resid_iso = U - isotonic(D, U)

        sp = gd.spearman
        rows.append({
            "state_id": sid, "suite": m["suite"], "task_id": m["task_id"], "phase": m["phase"],
            "episode": f"{m['suite']}_t{m['task_id']}_ep{m['episode_idx']}",
            # magnitudes
            "D_median": float(np.median(D)), "G_median": float(np.median(G)),
            "U_median": float(np.median(U)), "cos_median": float(np.median(cos)),
            "D_over_G_median": float(np.median(D / (G + EPS))),
            "G_over_htnorm": float(np.median(G) / float(np.median(np.linalg.norm(h_t, axis=-1)))),
            "frac_tokens_G_lt_D": float(np.mean(G < D)),
            # core correlations
            "sp_D_U": sp(D, U), "sp_G_U": sp(G, U), "sp_D_G": sp(D, G), "sp_cos_U": sp(cos, U),
            "sp_absDG_U": sp(np.abs(D - G), U), "sp_1mcos_U": sp(1 - cos, U),
            "pe_D_U": gd.pearson(D, U),
            # counterfactuals
            "sp_D_U_b4a_cosperm": sp(D, U_b4a), "sp_D_U_b4b_cos0": sp(D, U_b4b),
            "sp_G_U_b4a": sp(G, U_b4a),
            # does direction mismatch grow with predicted change?
            "sp_D_1mcos": sp(D, 1 - cos),
            "sp_D_relerr": sp(D, U / (G + EPS)),
            # residual analysis
            "sp_S_resid_iso": sp(S, resid_iso), "sp_S_resid_lin": sp(S, resid_lin),
            "sp_S_U": sp(S, U), "sp_S_D": sp(S, D),
            "sp_resid_iso_errdev": sp(resid_iso, errdev),
            "sp_D_errdev": sp(D, errdev), "sp_U_errdev": sp(U, errdev), "sp_S_errdev": sp(S, errdev),
            "sp_DS_errdev": sp(D * S, errdev), "sp_US_errdev": sp(U * S, errdev),
            "sp_Datt_errdev": sp(D * att, errdev), "sp_att_errdev": sp(att, errdev),
            "u_float16_rel_error": consistency[-1],
        })
        for k, v in [("D", D), ("G", G), ("U", U), ("cos", cos), ("S", S),
                     ("resid_iso", resid_iso), ("resid_lin", resid_lin), ("errdev", errdev)]:
            pooled[k].append(v)

    with open(run_dir / "GEOMETRY_ANALYSIS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    arr = {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0]
           if isinstance(rows[0][k], float)}
    eps_key = np.asarray([r["episode"] for r in rows])
    phases = np.asarray([r["phase"] for r in rows])
    suites = np.asarray([r["suite"] for r in rows])

    def agg(k):
        return {"median": float(np.median(arr[k])), "mean": float(np.mean(arr[k])),
                "ci": gd.boot_ci(arr[k], eps_key),
                "min": float(arr[k].min()), "max": float(arr[k].max())}

    summary = {
        "n_states": len(rows),
        "float16_consistency": {"max_rel_error_U": float(np.max(consistency)),
                                "passed": bool(np.max(consistency) < 1e-2)},
        "magnitudes": {k: agg(k) for k in ["D_median", "G_median", "U_median", "cos_median",
                                           "D_over_G_median", "G_over_htnorm", "frac_tokens_G_lt_D"]},
        "correlations": {k: agg(k) for k in ["sp_D_U", "sp_G_U", "sp_D_G", "sp_cos_U", "sp_absDG_U",
                                             "sp_1mcos_U", "pe_D_U", "sp_D_1mcos", "sp_D_relerr"]},
        "counterfactuals": {k: agg(k) for k in ["sp_D_U_b4a_cosperm", "sp_D_U_b4b_cos0", "sp_G_U_b4a"]},
        "residual_analysis": {k: agg(k) for k in ["sp_S_resid_iso", "sp_S_resid_lin", "sp_S_U", "sp_S_D",
                                                  "sp_resid_iso_errdev"]},
        "errdev_predictors": {k: agg(k) for k in ["sp_D_errdev", "sp_U_errdev", "sp_S_errdev",
                                                  "sp_DS_errdev", "sp_US_errdev", "sp_Datt_errdev",
                                                  "sp_att_errdev"]},
        "by_phase": {p: {k: float(np.median(arr[k][phases == p]))
                         for k in ["sp_D_U", "sp_G_U", "D_over_G_median", "cos_median", "sp_D_U_b4a_cosperm",
                                   "sp_S_resid_iso"]} for p in np.unique(phases)},
        "by_suite": {s: {k: float(np.median(arr[k][suites == s]))
                         for k in ["sp_D_U", "sp_G_U", "D_over_G_median", "cos_median"]}
                     for s in np.unique(suites)},
    }
    json.dump(summary, open(run_dir / "summary_geometry.json", "w"), indent=2)

    p = lambda k, d: f"{d[k]['median']:+.4f} [{d[k]['ci'][0]:+.4f},{d[k]['ci'][1]:+.4f}]"  # noqa: E731
    print(f"float16 consistency: max relative error of recomputed U = "
          f"{summary['float16_consistency']['max_rel_error_U']:.2e} -> "
          f"{'PASS' if summary['float16_consistency']['passed'] else 'FAIL'}\n")
    print("--- magnitudes (median over 90 states)")
    for k in summary["magnitudes"]:
        print(f"  {k:22s} {p(k, summary['magnitudes'])}")
    print("\n--- correlations with U")
    for k in summary["correlations"]:
        print(f"  {k:22s} {p(k, summary['correlations'])}")
    print("\n--- counterfactuals (keep D and G, destroy the direction pairing)")
    for k in summary["counterfactuals"]:
        print(f"  {k:22s} {p(k, summary['counterfactuals'])}")
    print("\n--- residual analysis (does S explain what D cannot?)")
    for k in summary["residual_analysis"]:
        print(f"  {k:22s} {p(k, summary['residual_analysis'])}")
    print("\n--- token-level predictors of the realised per-token action deviation")
    for k in summary["errdev_predictors"]:
        print(f"  {k:22s} {p(k, summary['errdev_predictors'])}")
    print("\n--- by phase")
    for ph, v in summary["by_phase"].items():
        print(f"  {ph:14s} " + "  ".join(f"{k}={x:+.3f}" for k, x in v.items()))


if __name__ == "__main__":
    main()
