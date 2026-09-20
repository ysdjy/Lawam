"""Step 3a: run the ORIGINAL policy online (official ModelClient + websocket server) from distinct reset episodes,
recording every step, plus full snapshots at every chunk boundary (policy query step). No model intervention here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_common as sc  # noqa: E402
from examples.LIBERO.eval_files.libero_eval_core import _binarize_gripper_open, invert_gripper_action  # noqa: E402

CKPT = sc.REPO_ROOT / "results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
MAX_STEPS = 250  # libero_spatial canonical max steps (libero_benchmark_adapters._get_canonical_max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--episodes", type=str, default="0-9")
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lo, hi = [int(x) for x in args.episodes.split("-")]
    run_dir = Path(args.run_dir)
    out_root = run_dir / "candidates" / f"task{args.task_id:02d}"
    out_root.mkdir(parents=True, exist_ok=True)

    suite = sc.load_suite()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    client = sc.ModelClient(policy_ckpt_path=str(CKPT), port=args.port)
    assert client.horizon_sec == 0.4 and client.action_hz == 20.0 and client._expected_action_chunk_length(20.0) == sc.CHUNK_LEN
    env, desc = sc.make_env(task, args.seed)
    e = sc.raw_env(env)
    bowl_name, plate_name = e.obj_of_interest[0], e.obj_of_interest[1]

    for ep in range(lo, hi + 1):
        ep_dir = out_root / f"ep{ep:03d}"
        ep_dir.mkdir(exist_ok=True)
        (ep_dir / "snapshots").mkdir(exist_ok=True)
        t0 = time.time()
        client.reset(task_description=desc)
        obs = sc.reset_to_init_state(env, init_states[ep])
        bowl0 = sc.body_pos(env, bowl_name)
        frames, wrists, states, recs, actions, boundaries = [], [], [], [], [], []
        success = False
        step = 0
        error = None
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
                grip = invert_gripper_action(_binarize_gripper_open(np.asarray(raw["open_gripper"], dtype=np.float32).reshape(-1)))
                delta = np.concatenate([wv, rot, grip]).astype(np.float64)
                rec = sc.obs_record(env, obs)
                rec.update({"step": step, "is_boundary": bool(is_boundary), "bowl_disp": float(np.linalg.norm(sc.body_pos(env, bowl_name) - bowl0))})
                if is_boundary:
                    rec["raw_chunk_unnormalized"] = np.asarray(client._get_slot_state(0).raw_actions).tolist()
                    boundaries.append(step)
                frames.append(img.copy()); wrists.append(wrist.copy()); states.append(ex["state"][0].copy()); recs.append(rec); actions.append(delta)
                obs, _, done, _ = env.step(delta.tolist())
                step += 1
                if done:
                    success = True
                    break
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
        # final obs
        ex, img, wrist = sc.policy_example(obs, desc)
        frames.append(img.copy()); wrists.append(wrist.copy()); states.append(ex["state"][0].copy()); recs.append({**sc.obs_record(env, obs), "step": step, "is_boundary": False, "final": True})
        np.savez_compressed(ep_dir / "steps.npz", primary=np.stack(frames), wrist=np.stack(wrists), state=np.stack(states), actions=np.asarray(actions))
        with open(ep_dir / "records.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, default=sc._json_default) + "\n")
        meta = {"task_id": args.task_id, "task_name": task.name, "instruction": desc, "episode_idx": ep, "init_state_idx": ep, "env_seed": args.seed,
                "success": success, "num_actions": len(actions), "chunk_boundaries": boundaries, "error": error, "bowl_name": bowl_name, "plate_name": plate_name,
                "bowl_init_pos": bowl0.tolist(), "plate_init_pos": sc.body_pos(env, plate_name).tolist(), "wall_sec": time.time() - t0}
        sc.write_json(ep_dir / "episode.json", meta)
        print(f"task{args.task_id} ep{ep}: success={success} steps={len(actions)} boundaries={len(boundaries)} err={error} ({time.time()-t0:.1f}s)", flush=True)
    env.close()
    client.close()


if __name__ == "__main__":
    main()
