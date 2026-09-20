"""Step 3b: pre-registered snapshot selection + two-branch reference construction (libero310, no model).

Snapshot rule (pre-registered in configs/protocol.yaml):
  earliest policy-query step k of the ORIGINAL rollout such that
    gripper open (qpos[0] > 0.035, no close action yet), bowl undisplaced (< 2 mm),
    eef above bowl: horizontal dist <= 0.08 m and 0.10 <= (z_eef - z_bowl) <= 0.24 m.
Branches: A = pre-grasp above the +y rim, B = pre-grasp above the -y rim (fingers close along y).
Reference chunk = 8 scripted OSC actions toward the branch hover point; continuation = fixed scripted grasp/place.
Qualification (all must hold): both branches succeed the full task; chunk determinism on replay; bowl untouched
during chunk; ||eef_A(t7) - eef_B(t7)|| >= 0.03 m.
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
import scripted as sp  # noqa: E402
import sim_common as sc  # noqa: E402

RULE = {"grip_qpos_min": 0.035, "bowl_disp_max": 0.002, "dxy_max": 0.08, "dz_min": 0.10, "dz_max": 0.24}
QUAL = {"min_sep_t7": 0.03, "chunk_bowl_disp_max": 0.005, "replay_det_max": 1e-6}
BRANCHES = {"A": np.array([0.0, 1.0]), "B": np.array([0.0, -1.0])}


def select_snapshot(ep_dir: Path):
    meta = json.load(open(ep_dir / "episode.json"))
    recs = [json.loads(l) for l in open(ep_dir / "records.jsonl")]
    acts = np.load(ep_dir / "steps.npz")["actions"]
    first_close = next((i for i, a in enumerate(acts) if a[6] > 0), len(acts))
    evals = []
    chosen = None
    for r in recs:
        if not r.get("is_boundary"):
            continue
        k = r["step"]
        eef = np.array(r["eef_pos"]); bowl = np.array(r[f"obj_{meta['bowl_name']}"])
        dxy = float(np.linalg.norm((eef - bowl)[:2])); dz = float(eef[2] - bowl[2])
        ok = (r["gripper_qpos"][0] > RULE["grip_qpos_min"] and k < first_close and r["bowl_disp"] < RULE["bowl_disp_max"]
              and dxy <= RULE["dxy_max"] and RULE["dz_min"] <= dz <= RULE["dz_max"])
        evals.append({"step": k, "dxy": dxy, "dz": dz, "grip_qpos": r["gripper_qpos"][0], "bowl_disp": r["bowl_disp"], "passes_rule": bool(ok)})
        if ok and chosen is None:
            chosen = k
    return meta, evals, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task_id", type=int, default=2)
    ap.add_argument("--episodes", type=str, default="0-9")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lo, hi = [int(x) for x in args.episodes.split("-")]
    run_dir = Path(args.run_dir)
    cand_root = run_dir / "candidates" / f"task{args.task_id:02d}"
    ref_root = run_dir / "reference"; ref_root.mkdir(exist_ok=True)
    vid_root = run_dir / "videos" / "reference"; vid_root.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "reference_manifest.jsonl"

    suite = sc.load_suite(); task = suite.get_task(args.task_id); init_states = suite.get_task_init_states(args.task_id)
    env, desc = sc.make_env(task, args.seed); e = sc.raw_env(env)
    env.reset(); env.set_init_state(init_states[0])
    exec_count = 0
    for ep in range(lo, hi + 1):
        ep_dir = cand_root / f"ep{ep:03d}"
        if not (ep_dir / "episode.json").exists():
            print(f"ep{ep}: missing candidate data, skip"); continue
        t0 = time.time()
        meta, evals, k = select_snapshot(ep_dir)
        state_id = f"t{args.task_id:02d}_ep{ep:03d}"
        entry = {"state_id": state_id, "task_id": args.task_id, "task_name": meta["task_name"], "instruction": meta["instruction"], "episode_idx": ep,
                 "init_state_idx": meta["init_state_idx"], "env_seed": args.seed, "boundary_evals": evals, "snapshot_step": k, "rule": RULE, "qualification_rule": QUAL}
        if k is None:
            entry.update({"qualified": False, "exclusion_reason": "no boundary satisfies snapshot rule"})
            sc.append_jsonl(manifest, entry); print(f"{state_id}: no snapshot"); continue
        snap = sc.SimSnapshot.from_npz(np.load(ep_dir / "snapshots" / f"step{k:04d}.npz"))
        steps = np.load(ep_dir / "steps.npz")
        o0_primary, o0_wrist, o0_state = steps["primary"][k], steps["wrist"][k], steps["state"][k]
        sdir = ref_root / state_id; sdir.mkdir(exist_ok=True)
        np.savez_compressed(sdir / "snapshot.npz", **snap.to_npz_dict())
        np.savez_compressed(sdir / "o0.npz", primary=o0_primary, wrist=o0_wrist, state=o0_state, lang=np.array([desc]))
        bowl_name, plate_name = meta["bowl_name"], meta["plate_name"]
        branch_res = {}
        for bname, bdir in BRANCHES.items():
            obs = sc.restore_snapshot(env, snap)
            bowl = sc.body_pos(env, bowl_name); target = sp.hover_point(bowl, bdir)
            rec = sp.Recorder(env, desc); rec.log(obs)  # rec.frames[0] is the *restored* render (o0 online image kept separately)
            chunk_actions = sp.branch_chunk_actions(env, target, rec)
            exec_count += 1
            eef_chunk = np.array([r["eef_pos"] for r in rec.records]); bowl_chunk = np.array([r[f"obj_{bowl_name}"] for r in rec.records])
            chunk_bowl_disp = float(np.max(np.linalg.norm(bowl_chunk - bowl_chunk[0], axis=1)))
            n_chunk_frames = len(rec.frames)
            success, n_cont = sp.continuation(env, bowl_name, plate_name, bdir, rec)
            # replay determinism of the recorded chunk (open-loop)
            sc.restore_snapshot(env, snap); rec2 = sp.Recorder(env, desc)
            for a in chunk_actions:
                sp.step_env(env, a, rec2)
            exec_count += 1
            det = float(np.max(np.abs(np.array([r["eef_pos"] for r in rec2.records]) - eef_chunk[1:])))
            img_det = int(np.max(np.abs(np.stack(rec2.frames).astype(int) - np.stack(rec.frames[1:n_chunk_frames]).astype(int))))
            np.savez_compressed(sdir / f"branch_{bname}_chunk.npz", actions=chunk_actions, frames=np.stack(rec.frames[:n_chunk_frames]), wrist=np.stack(rec.wrist[:n_chunk_frames]),
                                states=np.stack(rec.states[:n_chunk_frames]), eef=eef_chunk, bowl=bowl_chunk, replay_frames=np.stack(rec2.frames), target=target)
            with open(sdir / f"branch_{bname}_full_records.jsonl", "w") as f:
                for r, a in zip(rec.records, [None] + rec.actions):
                    f.write(json.dumps({**r, "action": None if a is None else a.tolist()}, default=sc._json_default) + "\n")
            imageio.mimwrite(vid_root / f"{state_id}_ref_{bname}.mp4", rec.frames, fps=20)
            branch_res[bname] = {"full_task_success": bool(success), "continuation_steps": int(n_cont), "chunk_bowl_disp": chunk_bowl_disp, "replay_eef_maxdiff": det,
                                 "replay_image_maxdiff": img_det, "eef_t0": eef_chunk[0].tolist(), "eef_t7": eef_chunk[7].tolist(), "eef_t8": eef_chunk[8].tolist(),
                                 "target_hover": target.tolist(), "chunk_action_absmax": float(np.max(np.abs(chunk_actions[:, :6])))}
        sep7 = float(np.linalg.norm(np.array(branch_res["A"]["eef_t7"]) - np.array(branch_res["B"]["eef_t7"])))
        sep8 = float(np.linalg.norm(np.array(branch_res["A"]["eef_t8"]) - np.array(branch_res["B"]["eef_t8"])))
        reasons = []
        for b in "AB":
            if not branch_res[b]["full_task_success"]: reasons.append(f"branch {b} continuation failed")
            if branch_res[b]["chunk_bowl_disp"] > QUAL["chunk_bowl_disp_max"]: reasons.append(f"branch {b} bowl moved during chunk")
            if branch_res[b]["replay_eef_maxdiff"] > QUAL["replay_det_max"]: reasons.append(f"branch {b} replay not deterministic")
        if sep7 < QUAL["min_sep_t7"]: reasons.append("branches not separated at t7")
        entry.update({"branches": branch_res, "sep_t7": sep7, "sep_t8": sep8, "qualified": len(reasons) == 0, "exclusion_reason": "; ".join(reasons) or None,
                      "snapshot_digest": snap.digest(), "files": {"snapshot": str(sdir / "snapshot.npz"), "o0": str(sdir / "o0.npz"), "branch_A": str(sdir / "branch_A_chunk.npz"), "branch_B": str(sdir / "branch_B_chunk.npz")},
                      "wall_sec": time.time() - t0})
        sc.append_jsonl(manifest, entry)
        print(f"{state_id}: step{k} qualified={entry['qualified']} sep7={sep7:.3f} A={branch_res['A']['full_task_success']} B={branch_res['B']['full_task_success']} det={branch_res['A']['replay_eef_maxdiff']:.1e} {reasons} ({time.time()-t0:.0f}s)", flush=True)
    sc.append_jsonl(run_dir / "execution_counts.jsonl", {"script": "s2_reference_branches", "episodes": args.episodes, "sim_executions": exec_count, "note": "2 branches x (chunk+continuation, chunk replay) per state"})
    env.close()


if __name__ == "__main__":
    main()
