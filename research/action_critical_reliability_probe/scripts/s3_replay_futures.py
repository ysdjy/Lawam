"""Replay the baseline chunk from each state's snapshot (sim side, `libero310` env).

Purpose (all of it is needed before U can be trusted):
  * verify that restoring the snapshot and re-executing the recorded chunk reproduces the rollout exactly
    (physics check: eef/object at t+7 must match the online record),
  * produce a SECOND rendering of o_{t+7} so that the render-noise floor of U can be measured in feature
    space (the same state rendered twice differs slightly under EGL),
  * record the environment-level quantities at t+7 (eef pose, object pose) that Step 4 Level 2 will use.

No policy is queried here: the actions replayed are exactly the ones the policy already executed online.

Writes <run_dir>/futures/<state_id>.npz and <run_dir>/futures_manifest.jsonl.
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
import sim_common as sc  # noqa: E402

FUTURE_IDX = 7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    ap.add_argument("--states", default="all")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(run_dir / "states_manifest.jsonl")]
    if args.states != "all":
        keep = set(args.states.split(","))
        manifest = [m for m in manifest if m["state_id"] in keep]
    out_dir = run_dir / "futures"
    out_dir.mkdir(exist_ok=True)
    out_manifest = run_dir / "futures_manifest.jsonl"
    if out_manifest.exists():
        out_manifest.unlink()

    manifest.sort(key=lambda m: (m["suite"], m["task_id"], m["episode_idx"], m["step"]))
    env = None
    current_task = None
    for m in manifest:
        key = (m["suite"], m["task_id"])
        if key != current_task:
            if env is not None:
                env.close()
            suite = sc.load_suite(m["suite"])
            task = suite.get_task(m["task_id"])
            env, desc = sc.make_env(task, 0)
            sc.reset_to_init_state(env, suite.get_task_init_states(m["task_id"])[m["episode_idx"]])
            current_task = key
        t0 = time.time()
        snap = sc.SimSnapshot.from_npz(np.load(m["snapshot"]))
        chunk = np.asarray(m["executed_chunk"], dtype=np.float64)
        target = m["target_object"]

        runs = []
        for rep in range(2):
            sc.restore_snapshot(env, snap)
            frames, eefs, objs = [], [], []
            obs = None
            for a in chunk:
                obs, _, _, _ = env.step(a.tolist())
                frames.append(sc.flipped_agentview(obs))
                eefs.append(sc.eef_pos(env).copy())
                objs.append(sc.body_pos(env, target).copy())
            runs.append({"frames": np.stack(frames), "eef": np.stack(eefs), "obj": np.stack(objs)})

        # physics reproduction check against the online record (frame index t+7 == chunk step 7 => runs[...][6]
        # is after 7 executed actions, i.e. the observation the training future frame corresponds to)
        eef_online_t7 = np.asarray(m["eef_pos_t7"], dtype=np.float64)
        obj_online_t7 = np.asarray(m["obj_pos_t7"], dtype=np.float64)
        eef_err = float(np.linalg.norm(runs[0]["eef"][FUTURE_IDX - 1] - eef_online_t7))
        obj_err = float(np.linalg.norm(runs[0]["obj"][FUTURE_IDX - 1] - obj_online_t7))

        online = np.load(m["state_npz"], allow_pickle=True)
        px_online_vs_replay = float(np.mean(np.abs(
            runs[0]["frames"][FUTURE_IDX - 1].astype(np.int32) - online["future_primary"].astype(np.int32))))
        px_replay_vs_replay = float(np.mean(np.abs(
            runs[0]["frames"][FUTURE_IDX - 1].astype(np.int32) - runs[1]["frames"][FUTURE_IDX - 1].astype(np.int32))))

        np.savez_compressed(
            out_dir / f"{m['state_id']}.npz",
            replay1_t7=runs[0]["frames"][FUTURE_IDX - 1], replay2_t7=runs[1]["frames"][FUTURE_IDX - 1],
            replay1_t8=runs[0]["frames"][-1],
            eef=runs[0]["eef"], obj=runs[0]["obj"],
            eef_rep2=runs[1]["eef"], obj_rep2=runs[1]["obj"],
        )
        rec = {
            "state_id": m["state_id"], "suite": m["suite"], "task_id": m["task_id"],
            "episode_idx": m["episode_idx"], "step": m["step"], "phase": m["phase"],
            "eef_reproduction_error_m": eef_err, "obj_reproduction_error_m": obj_err,
            "eef_repeat_difference_m": float(np.linalg.norm(runs[0]["eef"][FUTURE_IDX - 1]
                                                            - runs[1]["eef"][FUTURE_IDX - 1])),
            "pixel_meanabs_online_vs_replay": px_online_vs_replay,
            "pixel_meanabs_replay_vs_replay": px_replay_vs_replay,
            "eef_t7": runs[0]["eef"][FUTURE_IDX - 1].tolist(),
            "obj_t7": runs[0]["obj"][FUTURE_IDX - 1].tolist(),
            "eef_displacement_over_chunk_m": float(np.linalg.norm(
                runs[0]["eef"][FUTURE_IDX - 1] - np.asarray(m["eef_pos_t"], dtype=np.float64))),
            "futures_npz": str(out_dir / f"{m['state_id']}.npz"),
            "wall_sec": round(time.time() - t0, 2),
        }
        pc.append_jsonl(out_manifest, rec)
        print(f"{m['state_id']}: eef_err={eef_err:.2e} m obj_err={obj_err:.2e} m "
              f"px(online-replay)={px_online_vs_replay:.3f} px(replay-replay)={px_replay_vs_replay:.3f} "
              f"({rec['wall_sec']}s)", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
