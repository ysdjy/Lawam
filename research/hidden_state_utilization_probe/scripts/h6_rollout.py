"""Closed-loop paired pilot (simulator side, `libero310` env).

Conditions are run against `hsu_server.py`: ORIGINAL (arm none), C_P (control adapter, context
forced to 0 inside the server), H_CORRECT and H_WRONG (hidden adapter; the client sends the true or
the swapped context). Reset states and the pinned flow-noise schedule are identical across every
condition, so outcomes are paired.

Usage: python h6_rollout.py <run_id> <condition> <port>
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
import pcp_common as pcp  # noqa: E402

RUN, COND, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3])
OUT = hc.OUT_ROOT / RUN
EPISODES = {"NOMINAL": list(range(30, 40)), "HIGH": list(range(30, 50))}
C_HIGH = float(np.log(2.0))


class HSUClientShim:
    """Adds the pinned flow noise and the oracle context scalar to each query."""

    def __init__(self, inner):
        self.inner, self.initial_noise, self.hsu_c = inner, None, None

    def predict_action(self, q: dict) -> dict:
        q = dict(q)
        if self.initial_noise is not None:
            q["initial_noise"] = np.asarray(self.initial_noise, np.float32)
        if self.hsu_c is not None:
            q["hsu_c"] = float(self.hsu_c)
        return self.inner.predict_action(q)

    def close(self):
        return self.inner.close()


def main() -> None:
    rows_path = OUT / "ROLLOUT_RESULTS.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {(r["condition"], r["level"], int(r["init"])) for r in csv.DictReader(f)}

    suite, task, env, desc = hc.make_task_env()
    inits = suite.get_task_init_states(hc.TASK_ID)
    client = sc.ModelClient(policy_ckpt_path=str(hc.CKPT), port=PORT)
    shim = HSUClientShim(client.client)
    client.client = shim

    rows = []
    for level, ks in EPISODES.items():
        for k in ks:
            if (COND, level, k) in done:
                continue
            t0 = time.time()
            obs, tgt, mu = hc.reset_with_level(env, inits[k], level)
            c_true = hc.context_value(mu)
            c_sent = {"ORIGINAL": c_true, "C_P": c_true, "H_CORRECT": c_true,
                      "H_WRONG": (C_HIGH if abs(c_true) < 1e-9 else 0.0)}[COND]
            client.reset(task_description=desc)
            p0 = tgt.plate_pose()[0][:2].copy()
            d0 = float(np.linalg.norm(p0 - tgt.goal_xy()))
            qi, step, success, contact = 0, 0, False, 0
            acts, eef = [], []
            while step < hc.MAX_STEPS:
                if client._get_slot_state(0).needs_query():
                    shim.initial_noise = hc.noise(f"rollout|{level}|{k}", qi)
                    shim.hsu_c = c_sent
                    qi += 1
                a = pcp.env_action_from_raw(client.step(example=sc.policy_example(obs, desc)[0],
                                                        step=step)["raw_action"])
                shim.initial_noise = None
                acts.append(a.copy())
                eef.append(sc.eef_pos(env).copy())
                obs, _, _, _ = env.step(a.tolist())
                hits, _ = tgt.contacts()
                contact += int(hits > 0)
                step += 1
                if env.check_success():
                    success = True
                    break
            p1 = tgt.plate_pose()[0][:2]
            A = np.asarray(acts)
            row = {"condition": COND, "level": level, "init": k, "mu_eff": round(mu, 4),
                   "c_sent": round(c_sent, 6), "c_true": round(c_true, 6),
                   "success": success, "steps": step,
                   "progress_mm": round((d0 - float(np.linalg.norm(p1 - tgt.goal_xy()))) * 1000, 2),
                   "dist_end_mm": round(float(np.linalg.norm(p1 - tgt.goal_xy())) * 1000, 2),
                   "contact_steps": contact,
                   "mean_abs_trans": round(float(np.abs(A[:, :3]).mean()), 5),
                   "mean_abs_rot": round(float(np.abs(A[:, 3:6]).mean()), 5),
                   "frac_gripper_close": round(float((A[:, 6] > 0).mean()), 4),
                   "eef_path_m": round(float(np.abs(np.diff(np.asarray(eef), axis=0)).sum()), 4),
                   "mean_eef_z": round(float(np.asarray(eef)[:, 2].mean()), 4),
                   "sec": round(time.time() - t0, 1)}
            rows.append(row)
            tgt.restore_nominal()
            write_header = not rows_path.exists()
            with open(rows_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            print(json.dumps({kk: row[kk] for kk in ("condition", "level", "init", "success",
                                                     "steps", "progress_mm", "mean_abs_trans")}),
                  flush=True)
    env.close()
    client.close()
    for level in EPISODES:
        r = [x for x in rows if x["level"] == level]
        if r:
            print(f"{COND} {level}: {sum(x['success'] for x in r)}/{len(r)} success", flush=True)


if __name__ == "__main__":
    main()
