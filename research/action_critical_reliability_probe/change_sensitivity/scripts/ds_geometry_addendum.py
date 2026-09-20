"""Step 1 addendum (protocol amendment DS-A1): resolve an apparent contradiction.

`ds_geometry.py` finds median U (7.09) > median G (2.82): for the MEDIAN token, LaWAM's predicted future is
further from the real future than simply predicting "no change" would be. Round 1 reported the opposite
summary statistic — LaWAM beats the no-change baseline in 83 % of states — computed as a MEAN over tokens
of per-element MSE.

Both can be true only if the error distribution is strongly skewed across tokens. This script checks that
directly, conditioning on how much the token actually changed (G), so that "does LaWAM help?" is answered
per token regime instead of by one aggregate.

Read-only; no model, no S recomputation.
Outputs: <run_dir>/GEOMETRY_ADDENDUM.csv, appended block in summary_geometry.json
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
EPS = 1e-12
QUARTILE_EDGES = [0.25, 0.5, 0.75]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(PRIOR / "states_manifest.jsonl")]

    rows = []
    for m in manifest:
        sid = m["state_id"]
        s = np.load(PRIOR / "sensitivity" / f"{sid}.npz")
        u = np.load(PRIOR / "uncertainty" / f"{sid}.npz")
        h_t = s["h_t"].astype(np.float64)
        h_pred = s["h_t1_pred"].astype(np.float64)
        h_real = u["h_real"].astype(np.float64)
        S = gd.load_S(sid)

        D = np.linalg.norm(h_pred - h_t, axis=-1)
        G = np.linalg.norm(h_real - h_t, axis=-1)
        U = np.linalg.norm(h_pred - h_real, axis=-1)

        # per-token: does predicting the change beat predicting no change?
        better = U < G
        # per-element MSE aggregates, i.e. the round-1 statistic, for direct comparison
        mse_pred = float(((h_pred - h_real) ** 2).mean())
        mse_triv = float(((h_t - h_real) ** 2).mean())

        # condition on how much the token actually moved
        q = np.quantile(G, QUARTILE_EDGES)
        bins = np.digitize(G, q)                       # 0..3, 0 = most static tokens
        row = {"state_id": sid, "suite": m["suite"], "phase": m["phase"],
               "episode": f"{m['suite']}_t{m['task_id']}_ep{m['episode_idx']}",
               "frac_tokens_lawm_better_than_nochange": float(np.mean(better)),
               "mse_pred_global": mse_pred, "mse_trivial_global": mse_triv,
               "lawm_better_on_mse": bool(mse_pred < mse_triv),
               "U_median": float(np.median(U)), "G_median": float(np.median(G)),
               "D_median": float(np.median(D)),
               "sum_sq_err_pred": float((U ** 2).sum()), "sum_sq_err_trivial": float((G ** 2).sum())}
        for b in range(4):
            sel = bins == b
            row[f"q{b}_G_median"] = float(np.median(G[sel]))
            row[f"q{b}_U_median"] = float(np.median(U[sel]))
            row[f"q{b}_D_median"] = float(np.median(D[sel]))
            row[f"q{b}_frac_better"] = float(np.mean(better[sel]))
            row[f"q{b}_S_median"] = float(np.median(S[sel]))
        rows.append(row)

    with open(run_dir / "GEOMETRY_ADDENDUM.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    ep = np.asarray([r["episode"] for r in rows])
    arr = {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0]
           if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)}
    block = {
        "frac_states_lawm_better_on_global_mse": float(np.mean([r["lawm_better_on_mse"] for r in rows])),
        "frac_tokens_lawm_better_than_nochange": {
            "median": float(np.median(arr["frac_tokens_lawm_better_than_nochange"])),
            "ci": gd.boot_ci(arr["frac_tokens_lawm_better_than_nochange"], ep)},
        "total_squared_error_ratio_pred_over_trivial": {
            "median": float(np.median(arr["sum_sq_err_pred"] / arr["sum_sq_err_trivial"]))},
        "by_real_change_quartile": {
            f"q{b}": {"G_median": float(np.median(arr[f"q{b}_G_median"])),
                      "U_median": float(np.median(arr[f"q{b}_U_median"])),
                      "D_median": float(np.median(arr[f"q{b}_D_median"])),
                      "frac_tokens_lawm_better": float(np.median(arr[f"q{b}_frac_better"])),
                      "S_median": float(np.median(arr[f"q{b}_S_median"]))}
            for b in range(4)},
    }
    summ_path = run_dir / "summary_geometry.json"
    summ = json.loads(summ_path.read_text())
    summ["addendum_lawm_vs_nochange"] = block
    json.dump(summ, open(summ_path, "w"), indent=2)

    print(f"states where LaWAM beats no-change on global per-element MSE (round-1 statistic): "
          f"{block['frac_states_lawm_better_on_global_mse']:.1%}")
    fb = block["frac_tokens_lawm_better_than_nochange"]
    print(f"fraction of TOKENS where U < G (LaWAM better than no-change): "
          f"median {fb['median']:.3f} CI[{fb['ci'][0]:.3f},{fb['ci'][1]:.3f}]")
    print(f"total squared error ratio (pred / trivial), median over states: "
          f"{block['total_squared_error_ratio_pred_over_trivial']['median']:.3f}\n")
    print("conditioned on how much the token ACTUALLY changed (quartiles of G):")
    print(f"  {'quartile':10s} {'G':>8s} {'D':>8s} {'U':>8s} {'frac U<G':>9s} {'S':>10s}")
    for b in range(4):
        v = block["by_real_change_quartile"][f"q{b}"]
        print(f"  q{b} {'(static)' if b == 0 else '(moving)' if b == 3 else '        '} "
              f"{v['G_median']:8.3f} {v['D_median']:8.3f} {v['U_median']:8.3f} "
              f"{v['frac_tokens_lawm_better']:9.3f} {v['S_median']:10.3e}")


if __name__ == "__main__":
    main()
