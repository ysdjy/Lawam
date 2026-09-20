"""G3 Confirm stage C1 — practical significance only.

Compares ID vs A (occlusion medium) vs B (camera shift strong) on the confirm tasks:
    task success, U (DIAGNOSTIC ONLY), future treatment (mm), sampler nuisance (mm),
    treatment/nuisance, action deviation, executed eef and object consequence.

Evidence rule (PROTOCOL_G3.yaml amendment G3-A4): the grey occluder inflates U inside the occluded region
by construction, so **the rise in U is not admissible as primary evidence**. The primary evidence is the
executed consequence of pi(H_pred) vs pi(H_real) at fixed flow noise from the identical snapshot. U is
reported split into occluded-token and non-occluded-token parts.

This script computes NO S, NO D x S, NO selector and NO token repair.
Outputs: <run_dir>/CONFIRM_RESULTS.csv, confirm_c1_summary.json, figures/g3_confirm_c1.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/learned_u/scripts"))
import g2_dataset as gd  # noqa: E402  (episode bootstrap)

ID_TREAT_MM, ID_NUIS_MM = 0.904, 0.474          # round-1 / DS in-distribution reference
OCC_SIZE = 80                                   # frozen confirm severity for family A
COND_ORDER = ["ID", "A_occlusion_medium", "B_camera_shift_strong"]


def occlusion_mask(anchor_rc, size: int = OCC_SIZE) -> np.ndarray:
    """Which of the 256 tokens (16x16 grid of 16 px patches) lie inside the grey rectangle."""
    r, c = anchor_rc
    half = size // 2
    r = int(np.clip(r, half, 256 - half - 1)); c = int(np.clip(c, half, 256 - half - 1))
    grid = np.zeros((16, 16), dtype=bool)
    grid[(r - half) // 16:(r + half) // 16, (c - half) // 16:(c + half) // 16] = True
    return grid.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    conf = run_dir / "confirm"

    episodes = [json.loads(Path(p).read_text()) for p in glob.glob(str(conf / "rollouts/*/*/ep*/episode.json"))]
    exe = [json.loads(l) for l in open(conf / "treatment_execution.jsonl")]
    man = {json.loads(l)["state_id"]: json.loads(l) for l in open(conf / "states_manifest.jsonl")}
    ep_meta = {(e["condition"], e["suite"], e["task_id"], e["episode_idx"]): e for e in episodes}

    # ---- per-state U split into occluded / non-occluded tokens (diagnostic only)
    for r in exe:
        m = man[r["state_id"]]
        z = np.load(conf / "treatment_chunks" / r["condition"] / f"{r['state_id']}.npz")
        U = z["U_l2"].astype(np.float64)
        r["U_l2_mean_all"] = float(U.mean())
        e = ep_meta[(r["condition"], r["suite"], r["task_id"], r["episode_idx"])]
        pert = e.get("perturbation")
        if r["family"] == "A_occlusion" and pert and pert.get("anchor_rc"):
            mask = occlusion_mask(tuple(pert["anchor_rc"]))
            r["U_l2_occluded"] = float(U[mask].mean())
            r["U_l2_non_occluded"] = float(U[~mask].mean())
            r["n_occluded_tokens"] = int(mask.sum())
        else:
            r["U_l2_occluded"] = float("nan")
            r["U_l2_non_occluded"] = float(U.mean())
            r["n_occluded_tokens"] = 0

    with open(run_dir / "CONFIRM_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in exe for k in r}))
        w.writeheader(); w.writerows(exe)

    by_cond = defaultdict(list)
    for r in exe:
        by_cond[r["condition"]].append(r)
    ep_by_cond = defaultdict(list)
    for e in episodes:
        ep_by_cond[e["condition"]].append(e)

    def agg(cond: str) -> dict:
        X = by_cond[cond]
        E = [e for e in ep_by_cond[cond] if e.get("valid", True)]
        succ = [bool(e["success"]) for e in E if e["success"] is not None]
        k = np.asarray([r["episode"] for r in X])
        g = lambda key: np.asarray([r[key] for r in X], dtype=float)  # noqa: E731
        t, n = g("eef_treatment_mm"), g("eef_nuisance_mm")
        return {
            "n_episodes": len(E), "n_states": len(X),
            "success_rate": float(np.mean(succ)), "n_fail": int(len(succ) - np.sum(succ)),
            "treatment_mm_median": float(np.median(t)), "treatment_mm_mean": float(np.mean(t)),
            "treatment_mm_ci": gd.boot_ci(t, k),
            "nuisance_mm_median": float(np.median(n)), "nuisance_mm_ci": gd.boot_ci(n, k),
            "treatment_over_nuisance_median": float(np.median(t / np.maximum(n, 1e-9))),
            "treatment_over_nuisance_ci": gd.boot_ci(t / np.maximum(n, 1e-9), k),
            "treatment_vs_ID_round1": float(np.median(t) / ID_TREAT_MM),
            "action_maxabs_median": float(np.median(g("action_maxabs_pred_vs_real"))),
            "obj_treatment_mm_median": float(np.median(g("obj_treatment_mm"))),
            "chunk_motion_mm_median": float(np.median(g("chunk_motion_mm"))),
            "U_mean_all_median": float(np.median(g("U_l2_mean_all"))),
            "U_occluded_median": float(np.nanmedian(g("U_l2_occluded"))) if cond.startswith("A_") else None,
            "U_non_occluded_median": float(np.median(g("U_l2_non_occluded"))),
            "D_median": float(np.median(g("D_median"))), "G_median": float(np.median(g("G_median"))),
        }

    summary = {"conditions": {c: agg(c) for c in COND_ORDER if c in by_cond},
               "id_reference_round1_mm": {"treatment": ID_TREAT_MM, "nuisance": ID_NUIS_MM},
               "evidence_rule": "U is diagnostic only; the primary evidence is the executed consequence",
               "not_computed_in_C1": ["S", "D x S", "any selector", "token repair"]}

    # paired comparison against the matched ID controls, resampling episodes
    base = by_cond["ID"]
    for c in COND_ORDER[1:]:
        if c not in by_cond:
            continue
        X = by_cond[c]
        kk = np.asarray([r["episode"] for r in X])
        t_c = np.asarray([r["eef_treatment_mm"] for r in X])
        t_id = np.asarray([r["eef_treatment_mm"] for r in base])
        k_id = np.asarray([r["episode"] for r in base])
        summary["conditions"][c]["treatment_minus_ID_mm"] = float(np.mean(t_c) - np.mean(t_id))
        summary["conditions"][c]["treatment_ratio_to_ID_confirm"] = float(np.median(t_c) / np.median(t_id))
        summary["conditions"][c]["bootstrap_id_treatment_ci"] = gd.boot_ci(t_id, k_id)
        summary["conditions"][c]["bootstrap_cond_treatment_ci"] = gd.boot_ci(t_c, kk)

    # per suite / per task / per phase, and success vs failure episodes
    def breakdown(field: str, keyf) -> dict:
        out = {}
        for c in COND_ORDER:
            if c not in by_cond:
                continue
            d = defaultdict(list)
            for r in by_cond[c]:
                d[keyf(r)].append(r[field])
            out[c] = {str(k): float(np.median(v)) for k, v in sorted(d.items())}
        return out

    summary["treatment_by_suite"] = breakdown("eef_treatment_mm", lambda r: r["suite"])
    summary["treatment_by_task"] = breakdown("eef_treatment_mm", lambda r: f"{r['suite']}_t{r['task_id']}")
    summary["treatment_by_phase"] = breakdown("eef_treatment_mm", lambda r: r["phase"])
    summary["treatment_by_episode_outcome"] = breakdown(
        "eef_treatment_mm", lambda r: "success" if r["episode_success"] else "failure")
    summary["nuisance_by_episode_outcome"] = breakdown(
        "eef_nuisance_mm", lambda r: "success" if r["episode_success"] else "failure")

    json.dump(summary, open(run_dir / "confirm_c1_summary.json", "w"), indent=2)

    print(f"round-1 ID reference: treatment {ID_TREAT_MM} mm, nuisance {ID_NUIS_MM} mm\n")
    hdr = f"{'condition':24s} {'succ':>6s} {'eps':>4s} {'states':>7s} {'treat mm':>9s} {'CI':>16s} {'nuis mm':>8s} {'t/n':>6s} {'xID':>5s}"
    print(hdr); print("-" * len(hdr))
    for c in COND_ORDER:
        if c not in summary["conditions"]:
            continue
        a = summary["conditions"][c]
        print(f"{c:24s} {a['success_rate']:6.0%} {a['n_episodes']:4d} {a['n_states']:7d} "
              f"{a['treatment_mm_median']:9.2f} [{a['treatment_mm_ci'][0]:.2f},{a['treatment_mm_ci'][1]:.2f}] "
              f"{a['nuisance_mm_median']:8.2f} {a['treatment_over_nuisance_median']:6.2f} "
              f"{a['treatment_vs_ID_round1']:5.2f}")
    print("\n--- U diagnostic (NOT primary evidence; occluded tokens are inflated by construction)")
    for c in COND_ORDER:
        if c not in summary["conditions"]:
            continue
        a = summary["conditions"][c]
        occ = f"{a['U_occluded_median']:.2f}" if a["U_occluded_median"] is not None else "   -"
        print(f"  {c:24s} U_all={a['U_mean_all_median']:6.2f}  U_occluded={occ:>6s}  "
              f"U_non_occluded={a['U_non_occluded_median']:6.2f}  D={a['D_median']:5.2f} G={a['G_median']:5.2f}")
    print("\n--- treatment (mm) by suite / phase / episode outcome")
    for name in ["treatment_by_suite", "treatment_by_phase", "treatment_by_episode_outcome"]:
        print(f"  {name}:")
        for c in COND_ORDER:
            if c in summary[name]:
                print(f"    {c:24s} " + "  ".join(f"{k}={v:.2f}" for k, v in summary[name][c].items()))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        conds = [c for c in COND_ORDER if c in by_cond]
        lbl = ["ID", "occlusion\n80 px", "camera shift\n24 px"][: len(conds)]
        t = [np.asarray([r["eef_treatment_mm"] for r in by_cond[c]]) for c in conds]
        n = [np.asarray([r["eef_nuisance_mm"] for r in by_cond[c]]) for c in conds]
        ax[0].boxplot(t, tick_labels=lbl, showfliers=False)
        ax[0].boxplot(n, tick_labels=lbl, showfliers=False,
                      boxprops=dict(color="red"), medianprops=dict(color="red"),
                      whiskerprops=dict(color="red"), capprops=dict(color="red"))
        ax[0].set_ylabel("mm"); ax[0].set_title("black = future-error treatment\nred = sampler nuisance")
        ax[0].axhline(ID_TREAT_MM, ls="--", c="grey", lw=.8)
        ax[1].bar(range(len(conds)), [summary["conditions"][c]["treatment_over_nuisance_median"] for c in conds],
                  color=["#999999", "#cc6677", "#4477aa"][: len(conds)])
        ax[1].set_xticks(range(len(conds))); ax[1].set_xticklabels(lbl, fontsize=8)
        ax[1].axhline(1.84, ls="--", c="k", lw=.8)
        ax[1].set_ylabel("treatment / nuisance"); ax[1].set_title("amplification vs the sampler's own noise")
        sr = [summary["conditions"][c]["success_rate"] for c in conds]
        ax[2].bar(range(len(conds)), sr, color=["#999999", "#cc6677", "#4477aa"][: len(conds)])
        ax[2].set_xticks(range(len(conds))); ax[2].set_xticklabels(lbl, fontsize=8)
        ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("task success")
        ax[2].set_title("did the task get harder without collapsing?")
        fig.tight_layout(); fig.savefig(run_dir / "figures" / "g3_confirm_c1.png", dpi=150)
        print(f"\nfigure: {run_dir / 'figures' / 'g3_confirm_c1.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[figure skipped] {exc}")


if __name__ == "__main__":
    main()
