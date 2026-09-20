"""AUDIT 0 (simulation side, `libero310` env).

Confirms that the simulation half of the prior infrastructure still works before any new experiment:
  * the LIBERO suites we may draw tasks from still load, with their task lists,
  * an env builds / resets / renders with the official adapter,
  * the full snapshot -> steps -> restore -> replay path is still bit-accurate (this is what makes
    "same state, different intervention" comparisons legitimate later).

Output: <run_dir>/audit0_sim.json.  No policy is loaded here, no experiment is run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402
import sim_common as sc  # noqa: E402  (reused verbatim from research/branch_diagnostic)

SUITES = ["libero_spatial", "libero_object", "libero_goal"]
RESTORE_TASK = ("libero_spatial", 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    args = ap.parse_args()
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import mujoco
    import robosuite

    audit: dict = {
        "run_id": out_dir.name,
        "env": pc.env_versions() | {"mujoco": mujoco.__version__, "robosuite": robosuite.__version__},
        "libero_home": str(sc.LIBERO_HOME),
        "constants": {"NUM_STEPS_WAIT": sc.NUM_STEPS_WAIT, "CHUNK_LEN": sc.CHUNK_LEN, "FUTURE_IDX": sc.FUTURE_IDX,
                      "LIBERO_ENV_RESOLUTION": sc.LIBERO_ENV_RESOLUTION},
        "suites": {},
    }
    for name in SUITES:
        suite = sc.load_suite(name)
        tasks = []
        for i in range(suite.n_tasks):
            t = suite.get_task(i)
            tasks.append({"id": i, "name": t.name, "language": t.language,
                          "n_init_states": int(len(suite.get_task_init_states(i)))})
        audit["suites"][name] = {"n_tasks": suite.n_tasks, "tasks": tasks}
        print(f"{name}: {suite.n_tasks} tasks", flush=True)

    # --- snapshot / restore bit-accuracy re-check on one task
    suite_name, task_id = RESTORE_TASK
    suite = sc.load_suite(suite_name)
    task = suite.get_task(task_id)
    env, task_description = sc.make_env(task, seed=0)
    t0 = time.time()
    obs = sc.reset_to_init_state(env, suite.get_task_init_states(task_id)[0])
    for _ in range(sc.NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(sc.LIBERO_DUMMY_ACTION)
    # drive a few non-trivial steps so the controller carries internal state
    probe_actions = [np.array([0.2, -0.1, -0.3, 0.0, 0.0, 0.0, -1.0], dtype=np.float64) for _ in range(6)]
    for a in probe_actions:
        obs, _, _, _ = env.step(a)

    snap = sc.take_snapshot(env)
    img_a = sc.flipped_agentview(obs)
    seq = [np.array([0.3, 0.25, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float64) for _ in range(8)]
    traj_1 = []
    for a in seq:
        obs, _, _, _ = env.step(a)
        traj_1.append((sc.eef_pos(env).copy(), sc.raw_env(env).sim.data.qpos.copy()))

    obs_restored = sc.restore_snapshot(env, snap)
    img_b = sc.flipped_agentview(obs_restored)
    traj_2 = []
    for a in seq:
        obs, _, _, _ = env.step(a)
        traj_2.append((sc.eef_pos(env).copy(), sc.raw_env(env).sim.data.qpos.copy()))

    eef_diff = max(float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(traj_1, traj_2))
    qpos_diff = max(float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(traj_1, traj_2))
    audit["restore_check"] = {
        "suite": suite_name, "task_id": task_id, "task_description": task_description,
        "replay_len": len(seq),
        "eef_maxabs_diff_after_restore": eef_diff,
        "qpos_maxabs_diff_after_restore": qpos_diff,
        "snapshot_digest": snap.digest(),
        "rerender_same_state_maxabs_pixel_diff": float(np.max(np.abs(img_a.astype(np.int32) - img_b.astype(np.int32)))),
        "wall_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(audit["restore_check"], indent=1), flush=True)
    env.close()

    pc.write_json(out_dir / "audit0_sim.json", audit)


if __name__ == "__main__":
    main()
