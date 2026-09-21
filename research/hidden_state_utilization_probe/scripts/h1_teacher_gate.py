"""HSU AUDIT 1 — teacher gate: 5 NOMINAL + 5 HIGH episodes, >= 4/5 each, and action legality.

No model, no training, no adapter. Usage: python h1_teacher_gate.py <run_id> [n_episodes]
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsu_common as hc  # noqa: E402
import sim_common as sc  # noqa: E402

RUN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = hc.OUT_ROOT / RUN
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    suite, task, env, desc = hc.make_task_env()
    inits = suite.get_task_init_states(hc.TASK_ID)
    rows = []
    for level in ("NOMINAL", "HIGH"):
        for k in range(N):
            t0 = time.time()
            obs, tgt, mu = hc.reset_with_level(env, inits[k], level)
            r = hc.run_teacher_episode(env, tgt)
            tgt.restore_nominal()
            rows.append({"level": level, "init": k, "mu_eff": round(mu, 4),
                         "c": round(hc.context_value(mu), 6), "success": r["success"],
                         "steps": r["steps"], "progress_mm": round(r["progress_m"] * 1000, 2),
                         "dist_end_mm": round(r["dist_end"] * 1000, 2),
                         "contact_steps": r["contact_steps"],
                         "max_abs_action": round(r["max_abs_action"], 4),
                          "sec": round(time.time() - t0, 1)})
            print(json.dumps(rows[-1]), flush=True)
    env.close()

    with open(OUT / "TEACHER_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep = {"run_id": RUN, "n_per_level": N,
           "success": {lv: sum(r["success"] for r in rows if r["level"] == lv) for lv in ("NOMINAL", "HIGH")},
           "max_abs_action_overall": max(r["max_abs_action"] for r in rows),
           "actions_within_dataset_range": bool(max(r["max_abs_action"] for r in rows) <= hc.ACTION_LIMIT + 1e-9),
           "mu_eff": {lv: sorted({r["mu_eff"] for r in rows if r["level"] == lv}) for lv in ("NOMINAL", "HIGH")}}
    rep["GATE_PASS"] = bool(rep["success"]["NOMINAL"] >= 0.8 * N and rep["success"]["HIGH"] >= 0.8 * N
                            and rep["actions_within_dataset_range"])
    sc.write_json(OUT / "TEACHER_GATE.json", rep)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
