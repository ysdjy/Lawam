"""G3 pilot/confirm rollouts under a frozen stress condition (sim side, needs the websocket model server).

Identical to the validated `s1_rollouts.py` except that a pre-registered perturbation is applied to the
observation stream (families A/B) or to the initial simulator state (family C), persistently, from the
first step of the episode.

Because families A and B perturb the frame at the moment it is recorded AND at the moment it is sent to
the policy, `o_t` and the future frame `o_{t+7}` share the same perturbation by construction — there is no
way for the U label to come from a clean future.

All episodes are kept, successful or not. Family-C episodes that fail a validity check are marked invalid
and recorded; they are never replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402
import sim_common as sc  # noqa: E402
import g3_perturb as gp  # noqa: E402
from examples.LIBERO.eval_files.libero_eval_core import _binarize_gripper_open, invert_gripper_action  # noqa: E402

CKPT = REPO / "results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
MAX_STEPS = 250


def obs_record_bodies_only(env, obs, objects: list[str]):
    e = sc.raw_env(env)
    original = e.obj_of_interest
    try:
        e.obj_of_interest = objects
        return sc.obs_record(env, obs)
    finally:
        e.obj_of_interest = original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--episodes", default="0-4")
    ap.add_argument("--family", required=True, choices=list(gp.SEVERITIES) + ["ID"])
    ap.add_argument("--severity", default="none")
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh_env_per_episode", action="store_true")
    args = ap.parse_args()
    lo, hi = [int(x) for x in args.episodes.split("-")]
    run_dir = Path(args.run_dir)
    cond = "ID" if args.family == "ID" else f"{args.family}_{args.severity}"
    out_root = run_dir / "rollouts" / cond / f"{args.suite}_t{args.task_id:02d}"
    out_root.mkdir(parents=True, exist_ok=True)

    suite = sc.load_suite(args.suite)
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    client = sc.ModelClient(policy_ckpt_path=str(CKPT), port=args.port)
    env, desc = sc.make_env(task, args.seed)
    e = sc.raw_env(env)
    objects, skipped = [], []
    for n in list(e.obj_of_interest):
        try:
            sc.body_pos(env, n); objects.append(n)
        except KeyError:
            skipped.append(n)
    target = objects[0]

    # bounding box of the target object's position over all init states (for family-C check 2)
    init_xy = []
    for st in init_states:
        sc.reset_to_init_state(env, st)
        init_xy.append(sc.body_pos(env, target)[:2].copy())
    init_xy = np.stack(init_xy)

    for ep in range(lo, hi + 1):
        ep_dir = out_root / f"ep{ep:03d}"
        (ep_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if args.fresh_env_per_episode and ep > lo:
            env.close(); env, desc = sc.make_env(task, args.seed); e = sc.raw_env(env)
        client.reset(task_description=desc)
        obs = sc.reset_to_init_state(env, init_states[ep])

        shift_rec = None
        if args.family == "C_object_shift":
            shift_rec = gp.apply_object_shift(env, target, gp.SEVERITIES[args.family][args.severity], init_xy)
            obs = e._get_observations()
            if not shift_rec["valid"]:
                pc.write_json(ep_dir / "episode.json", {
                    "suite": args.suite, "task_id": args.task_id, "task_name": task.name,
                    "instruction": desc, "episode_idx": ep, "condition": cond, "family": args.family,
                    "severity": args.severity, "valid": False, "object_shift": shift_rec,
                    "success": None, "num_actions": 0, "chunk_boundaries": [],
                    "objects_of_interest": objects, "error": None,
                    "note": "episode marked INVALID by a pre-registered family-C validity check; recorded, not replaced"})
                print(f"{cond} {args.suite} t{args.task_id} ep{ep}: INVALID ({shift_rec.get('reason')})", flush=True)
                continue

        pert = None
        if args.family in ("A_occlusion", "B_camera_shift"):
            anchor, src = (gp.occlusion_anchor(env, target) if args.family == "A_occlusion" else (None, ""))
            pert = gp.VisualPerturbation(args.family, args.severity, anchor, src)

        obj0 = {n: sc.body_pos(env, n).copy() for n in objects}
        frames, wrists, states, recs, actions, boundaries = [], [], [], [], [], []
        success, step, error = False, 0, None
        try:
            while step < MAX_STEPS:
                ex, img, wrist = sc.policy_example(obs, desc)
                if pert is not None:                      # perturb ONCE, then use it everywhere
                    img = pert.apply(img)
                    ex["primary_image"] = [img]
                is_boundary = client._get_slot_state(0).needs_query()
                if is_boundary:
                    np.savez_compressed(ep_dir / "snapshots" / f"step{step:04d}.npz",
                                        **sc.take_snapshot(env).to_npz_dict())
                resp = client.step(example=ex, step=step)
                raw = resp["raw_action"]
                delta = np.concatenate([
                    np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1),
                    np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1),
                    invert_gripper_action(_binarize_gripper_open(
                        np.asarray(raw["open_gripper"], dtype=np.float32).reshape(-1)))]).astype(np.float64)
                rec = obs_record_bodies_only(env, obs, objects)
                rec.update({"step": step, "is_boundary": bool(is_boundary),
                            "obj_disp": {n: float(np.linalg.norm(sc.body_pos(env, n) - obj0[n])) for n in objects}})
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
        if pert is not None:
            img = pert.apply(img)
        frames.append(img.copy()); wrists.append(wrist.copy()); states.append(ex["state"][0].copy())
        recs.append({**obs_record_bodies_only(env, obs, objects), "step": step, "is_boundary": False, "final": True})
        np.savez_compressed(ep_dir / "steps.npz", primary=np.stack(frames), wrist=np.stack(wrists),
                            state=np.stack(states), actions=np.asarray(actions))
        with open(ep_dir / "records.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, default=sc._json_default) + "\n")
        pc.write_json(ep_dir / "episode.json", {
            "suite": args.suite, "task_id": args.task_id, "task_name": task.name, "instruction": desc,
            "episode_idx": ep, "init_state_idx": ep, "env_seed": args.seed, "condition": cond,
            "family": args.family, "severity": args.severity, "valid": True,
            "perturbation": (None if pert is None else
                             {"size": pert.size(), "anchor_rc": pert.anchor_rc, "anchor_source": pert.anchor_source}),
            "object_shift": shift_rec,
            "success": success, "num_actions": len(actions), "chunk_boundaries": boundaries, "error": error,
            "objects_of_interest": objects, "object_init_pos": {n: v.tolist() for n, v in obj0.items()},
            "non_body_obj_of_interest_skipped": skipped, "wall_sec": time.time() - t0})
        print(f"{cond} {args.suite} t{args.task_id} ep{ep}: success={success} steps={len(actions)} "
              f"err={error} ({time.time()-t0:.1f}s)", flush=True)
    env.close(); client.close()


if __name__ == "__main__":
    main()
