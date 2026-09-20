"""Online rollouts of the UNMODIFIED policy (sim side, `libero310` env; needs the websocket model server).

Generalisation of `research/branch_diagnostic/scripts/s1_collect_candidates.py` to any LIBERO suite/task.
Records every step (agentview + wrist frames, state, eef/object poses, executed action) and a full
simulator snapshot at every chunk boundary (policy query step). No intervention, no perturbation.

The recorded frames are what later supplies BOTH the probe input o_t and the real future o_{t+7}, since the
official client executes all 8 chunk actions before querying again.
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
from examples.LIBERO.eval_files.libero_eval_core import _binarize_gripper_open, invert_gripper_action  # noqa: E402

CKPT = pc.REPO_ROOT / "results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
MAX_STEPS = 250


def obs_record_bodies_only(env, obs, objects: list[str]):
    """`sim_common.obs_record` restricted to the obj_of_interest entries that resolve to a rigid body.

    Some LIBERO tasks (e.g. libero_goal t5 "push the plate to the front of the stove") list a goal REGION
    among obj_of_interest; it has no body and `body_pos` raises. Rather than duplicating the record format
    or editing the shared read-only helper, narrow `obj_of_interest` for the duration of the call so that
    `sim_common.obs_record` stays the single source of truth, then restore it immediately.
    """
    e = sc.raw_env(env)
    original = e.obj_of_interest
    try:
        e.obj_of_interest = objects
        return sc.obs_record(env, obs)
    finally:
        e.obj_of_interest = original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--episodes", type=str, default="0-9")
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--seed", type=int, default=0)
    # libero_goal scenes fail on a second env.reset() inside the same process (LIBERO object-registry
    # state); rebuilding the env per episode avoids it. Env construction is deterministic given the seed,
    # so this changes nothing about the recorded data.
    ap.add_argument("--fresh_env_per_episode", action="store_true")
    args = ap.parse_args()
    lo, hi = [int(x) for x in args.episodes.split("-")]
    run_dir = Path(args.run_dir)
    out_root = run_dir / "rollouts" / f"{args.suite}_t{args.task_id:02d}"
    out_root.mkdir(parents=True, exist_ok=True)

    suite = sc.load_suite(args.suite)
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    client = sc.ModelClient(policy_ckpt_path=str(CKPT), port=args.port)
    assert client.horizon_sec == 0.4 and client.action_hz == 20.0
    assert client._expected_action_chunk_length(20.0) == sc.CHUNK_LEN
    env, desc = sc.make_env(task, args.seed)
    e = sc.raw_env(env)
    # Some LIBERO tasks list a goal REGION (a site, e.g. "main_table_stove_front_region") among
    # obj_of_interest; those have no rigid body and cannot be logged with body_pos. Keep only the
    # entries that resolve to a body, and record which were skipped. For tasks whose obj_of_interest
    # are all bodies this is a no-op (verified against the previously collected tasks).
    objects, skipped_objects = [], []
    for n in list(e.obj_of_interest):
        try:
            sc.body_pos(env, n)
            objects.append(n)
        except KeyError:
            skipped_objects.append(n)
    if not objects:
        raise RuntimeError(f"no obj_of_interest of {task.name} resolves to a body: {list(e.obj_of_interest)}")
    if skipped_objects:
        print(f"[s1_rollouts] non-body obj_of_interest skipped: {skipped_objects}", flush=True)

    for ep in range(lo, hi + 1):
        ep_dir = out_root / f"ep{ep:03d}"
        (ep_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if args.fresh_env_per_episode and ep > lo:
            env.close()
            env, desc = sc.make_env(task, args.seed)
            e = sc.raw_env(env)
        client.reset(task_description=desc)
        obs = sc.reset_to_init_state(env, init_states[ep])
        obj0 = {n: sc.body_pos(env, n).copy() for n in objects}
        frames, wrists, states, recs, actions, boundaries = [], [], [], [], [], []
        success, step, error = False, 0, None
        try:
            while step < MAX_STEPS:
                ex, img, wrist = sc.policy_example(obs, desc)
                is_boundary = client._get_slot_state(0).needs_query()
                if is_boundary:
                    snap = sc.take_snapshot(env)
                    np.savez_compressed(ep_dir / "snapshots" / f"step{step:04d}.npz", **snap.to_npz_dict())
                resp = client.step(example=ex, step=step)
                raw = resp["raw_action"]
                wv = np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)
                rot = np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)
                grip = invert_gripper_action(_binarize_gripper_open(
                    np.asarray(raw["open_gripper"], dtype=np.float32).reshape(-1)))
                delta = np.concatenate([wv, rot, grip]).astype(np.float64)
                rec = obs_record_bodies_only(env, obs, objects)
                rec.update({
                    "step": step,
                    "is_boundary": bool(is_boundary),
                    "obj_disp": {n: float(np.linalg.norm(sc.body_pos(env, n) - obj0[n])) for n in objects},
                })
                if is_boundary:
                    rec["raw_chunk_unnormalized"] = np.asarray(client._get_slot_state(0).raw_actions).tolist()
                    boundaries.append(step)
                frames.append(img.copy()); wrists.append(wrist.copy())
                states.append(ex["state"][0].copy()); recs.append(rec); actions.append(delta)
                obs, _, done, _ = env.step(delta.tolist())
                step += 1
                if done:
                    success = True
                    break
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
        ex, img, wrist = sc.policy_example(obs, desc)
        frames.append(img.copy()); wrists.append(wrist.copy()); states.append(ex["state"][0].copy())
        recs.append({**obs_record_bodies_only(env, obs, objects), "step": step, "is_boundary": False, "final": True})
        np.savez_compressed(ep_dir / "steps.npz", primary=np.stack(frames), wrist=np.stack(wrists),
                            state=np.stack(states), actions=np.asarray(actions))
        with open(ep_dir / "records.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, default=sc._json_default) + "\n")
        pc.write_json(ep_dir / "episode.json", {
            "suite": args.suite, "task_id": args.task_id, "task_name": task.name, "instruction": desc,
            "episode_idx": ep, "init_state_idx": ep, "env_seed": args.seed, "success": success,
            "num_actions": len(actions), "chunk_boundaries": boundaries, "error": error,
            "objects_of_interest": objects, "object_init_pos": {n: v.tolist() for n, v in obj0.items()},
            "non_body_obj_of_interest_skipped": skipped_objects,
            "wall_sec": time.time() - t0,
        })
        print(f"{args.suite} t{args.task_id} ep{ep}: success={success} steps={len(actions)} "
              f"boundaries={len(boundaries)} err={error} ({time.time()-t0:.1f}s)", flush=True)
    env.close()
    client.close()


if __name__ == "__main__":
    main()
