"""Step 4/5 simulator side (libero310): execute R / B0 / B1 / B2 chunks from the restored snapshot.

For every execution: restore full snapshot, clear any cache (no client is involved: actions are an explicit 8-step list),
execute the 8 env actions without re-query, record o_1..o_8 (+ wrist, eef, bowl, gripper), label the executed branch
(pre-registered rule), then run the scripted continuation of the labelled branch (A/B only) for full_task_success.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_common as mx  # noqa: E402
import scripted as sp  # noqa: E402
import sim_common as sc  # noqa: E402

CKPT_DIR = sc.REPO_ROOT / "results/Checkpoints/libero/lawam_libero_sft_release"
BRANCH_DIRS = {"A": np.array([0.0, 1.0]), "B": np.array([0.0, -1.0])}  # identical to s2_reference_branches.BRANCHES


def execute_chunk(env, snap, env_actions, desc, bowl_name):
    obs = sc.restore_snapshot(env, snap)
    rec = sp.Recorder(env, desc); rec.log(obs)
    bowl0 = sc.body_pos(env, bowl_name)
    err = None; n_exec = 0; done_flag = False
    try:
        for a in env_actions:
            obs, done = sp.step_env(env, a, rec)
            n_exec += 1
            if done:
                done_flag = True
                break
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    bowl_disp = float(max(np.linalg.norm(np.array(r[f"obj_{bowl_name}"]) - bowl0) for r in rec.records))
    return rec, n_exec, err, done_flag, bowl_disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase", required=True, choices=["pilot", "confirm"])
    ap.add_argument("--states", type=str, required=True, help="comma-separated state ids")
    ap.add_argument("--task_id", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = {m["state_id"]: m for m in (json.loads(l) for l in open(run_dir / "reference_manifest.jsonl")) if m.get("qualified")}
    state_ids = args.states.split(",")
    norm_stats = sc.load_norm_stats(CKPT_DIR)
    exec_root = run_dir / "executions"; exec_root.mkdir(exist_ok=True)
    vid_root = run_dir / "videos" / args.phase; vid_root.mkdir(parents=True, exist_ok=True)
    episodes_path = run_dir / "episode_records.jsonl"
    failures_path = run_dir / "failures.jsonl"

    suite = sc.load_suite(); task = suite.get_task(args.task_id); init_states = suite.get_task_init_states(args.task_id)
    env, desc = sc.make_env(task, args.seed); e = sc.raw_env(env)
    env.reset(); env.set_init_state(init_states[0])
    bowl_name, plate_name = e.obj_of_interest[0], e.obj_of_interest[1]
    n_core = 0; n_cont = 0
    for sid in state_ids:
        st = manifest[sid]; t0 = time.time()
        snap = sc.SimSnapshot.from_npz(np.load(st["files"]["snapshot"]))
        ref = {b: np.load(st["files"][f"branch_{b}"]) for b in "AB"}
        p_A, p_B = ref["A"]["eef"][7], ref["B"]["eef"][7]
        mo = np.load(run_dir / "conditions" / sid / "model_outputs.npz")
        diag = json.load(open(run_dir / "conditions" / sid / "model_diag.json"))
        sdir = exec_root / sid; sdir.mkdir(exist_ok=True)
        plan = [{"condition": "R", "input_branch": b, "seed": None, "env_actions": ref[b]["actions"], "normalized": None, "key": f"ref_{b}"} for b in "AB"]
        for r in diag["records"]:
            normalized = mo[r["key"]]  # [8, 32]
            plan.append({**r, "normalized": normalized, "env_actions": sc.normalized_to_env_action(normalized[:, :7], norm_stats)})
        for item in plan:
            cond, ib, seed = item["condition"], item["input_branch"], item["seed"]
            tag = f"{cond}" + (f"_{ib}" if ib else "") + (f"_s{seed}" if seed is not None else "")
            rec, n_exec, err, done_flag, bowl_disp = execute_chunk(env, snap, item["env_actions"], desc, bowl_name)
            n_core += 1
            eef = np.array([r["eef_pos"] for r in rec.records]); bowl = np.array([r[f"obj_{bowl_name}"] for r in rec.records])
            failed = err is not None or n_exec < sc.CHUNK_LEN
            p7 = eef[7] if len(eef) > 7 else eef[-1]
            label, s_e, perp = mx.executed_label(p7, p_A, p_B, bowl_disp, failed=failed)
            n_chunk_frames = len(rec.frames)
            full_success = None; cont_steps = None
            if label in ("A", "B"):
                ok, cont_steps = sp.continuation(env, bowl_name, plate_name, BRANCH_DIRS[label], rec)
                full_success = bool(ok); n_cont += 1
            np.savez_compressed(sdir / f"{tag}.npz", frames=np.stack(rec.frames[:n_chunk_frames]), wrist=np.stack(rec.wrist[:n_chunk_frames]), states=np.stack(rec.states[:n_chunk_frames]),
                                eef=eef, bowl=bowl, env_actions=np.asarray(item["env_actions"]), normalized_actions=np.asarray(item["normalized"]) if item["normalized"] is not None else np.zeros(0),
                                gripper_qpos=np.array([r["gripper_qpos"] for r in rec.records]))
            imageio.mimwrite(vid_root / f"{sid}_{tag}.mp4", rec.frames, fps=20)
            entry = {"run_id": run_dir.name, "phase": args.phase, "state_id": sid, "task": st["task_name"], "instruction": st["instruction"], "snapshot_id": st["snapshot_digest"],
                     "snapshot_step": st["snapshot_step"], "condition_group": cond, "reference_branch": ib, "noise_seed": seed,
                     "requested_horizon": sc.CHUNK_LEN, "executed_steps": n_exec, "terminated_early": bool(done_flag), "error": err,
                     "executed_branch": label, "s_e": s_e, "perp_m": perp, "eef_t0": eef[0].tolist(), "eef_t7": p7.tolist(), "eef_t8": eef[-1].tolist(),
                     "dist_to_pA_t7": float(np.linalg.norm(p7 - p_A)), "dist_to_pB_t7": float(np.linalg.norm(p7 - p_B)), "bowl_disp_chunk": bowl_disp,
                     "gripper_closed_in_chunk": bool(np.any(np.asarray(item["env_actions"])[:, 6] > 0)),
                     "short_horizon_status": "FAIL" if failed else ("OK" if label in ("A", "B") else "OTHER"),
                     "full_task_success": full_success, "continuation_steps": cont_steps, "continuation_branch": label if label in ("A", "B") else None,
                     "files": {"execution": str(sdir / f"{tag}.npz"), "video": str(vid_root / f"{sid}_{tag}.mp4"), "model_outputs": str(run_dir / "conditions" / sid / "model_outputs.npz"),
                               "model_output_key": item.get("key"), "reference": st["files"], "initial_noise": str(run_dir / "conditions" / "initial_noise.npz")}}
            sc.append_jsonl(episodes_path, entry)
            if failed:
                sc.append_jsonl(failures_path, {"state_id": sid, "tag": tag, "error": err, "executed_steps": n_exec, "kind": "execution_failure"})
        print(f"{sid}: done {len(plan)} executions ({time.time()-t0:.0f}s)", flush=True)
    sc.append_jsonl(run_dir / "execution_counts.jsonl", {"script": "s3_execute_conditions", "phase": args.phase, "states": len(state_ids), "core_chunk_executions": n_core, "continuations": n_cont})
    env.close()


if __name__ == "__main__":
    main()
