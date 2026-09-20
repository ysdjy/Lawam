"""Phase-0 check D: state restore consistency (libero310 env, no model needed).

Also measures: eef displacement per control step for saturated actions (used to design branches),
bowl/plate geometry, and produces a projection-calibration image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_common as sc  # noqa: E402


def run_actions(env, actions):
    traj = []
    imgs = []
    obs = None
    for a in actions:
        obs, _, done, _ = env.step(list(map(float, a)))
        e = sc.raw_env(env)
        traj.append(np.concatenate([e.sim.data.qpos.copy(), e.sim.data.qvel.copy(), sc.eef_pos(env), sc.body_pos(env, e.obj_of_interest[0])]))
        imgs.append(sc.flipped_agentview(obs).copy())
    return np.asarray(traj), np.asarray(imgs), obs


def max_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task_id", type=int, default=2)
    ap.add_argument("--init_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = sc.load_suite()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    env, desc = sc.make_env(task, args.seed)
    e = sc.raw_env(env)
    print("task:", desc, "| obj_of_interest:", e.obj_of_interest)
    print("control_timestep", e.control_timestep, "model_timestep", e.model_timestep, "substeps", int(e.control_timestep / e.model_timestep))

    obs0 = sc.reset_to_init_state(env, init_states[args.init_idx])
    # prefix: fixed pseudo-random moderate actions including rotation deltas (exercise goal_ori state)
    rng = np.random.RandomState(123)
    prefix = np.clip(rng.randn(12, 7) * 0.4, -0.9, 0.9)
    prefix[:, 6] = -1.0
    prefix[6:, 3:6] = 0.0  # last 6 have zero rotation -> goal_ori persists from step 5 (controller state)
    _, _, obs_s = run_actions(env, prefix)
    snap = sc.take_snapshot(env)
    img_s = sc.flipped_agentview(obs_s).copy()
    rec_s = sc.obs_record(env, obs_s)

    # test action sequence X: translation + zero rotation + gripper toggling
    X = np.clip(rng.randn(16, 7) * 0.5, -0.9, 0.9)
    X[:, 3:6] = 0.0
    X[:, 6] = -1.0
    X[8:, 6] = 1.0
    T1, I1, _ = run_actions(env, X)

    results = {}
    # (a) full restore, 3 repeats
    diffs_full = []
    for k in range(3):
        obs_r = sc.restore_snapshot(env, snap)
        img_r = sc.flipped_agentview(obs_r)
        obs_img_diff = int(np.max(np.abs(img_r.astype(int) - img_s.astype(int))))
        rec_r = sc.obs_record(env, obs_r)
        Tk, Ik, _ = run_actions(env, X)
        diffs_full.append({
            "restored_obs_image_maxdiff": obs_img_diff,
            "restored_eef_maxdiff": max_diff(rec_r["eef_pos"], rec_s["eef_pos"]),
            "traj_maxdiff_qpos_qvel": max_diff(Tk[:, : e.sim.model.nq + e.sim.model.nv], T1[:, : e.sim.model.nq + e.sim.model.nv]),
            "traj_maxdiff_eef": max_diff(Tk[:, -6:-3], T1[:, -6:-3]),
            "traj_maxdiff_obj": max_diff(Tk[:, -3:], T1[:, -3:]),
            "image_maxdiff_final": int(np.max(np.abs(Ik[-1].astype(int) - I1[-1].astype(int)))),
            "image_maxdiff_all": int(np.max(np.abs(Ik.astype(int) - I1.astype(int)))),
        })
    results["full_restore"] = diffs_full

    # (b) naive restore: only qpos/qvel/time via set_init_state (what the official eval uses for init)
    env.set_init_state(snap.sim_state)
    Tn, In, _ = run_actions(env, X)
    results["naive_restore_qpos_qvel_only"] = {
        "traj_maxdiff_qpos_qvel": max_diff(Tn[:, : e.sim.model.nq + e.sim.model.nv], T1[:, : e.sim.model.nq + e.sim.model.nv]),
        "traj_maxdiff_eef": max_diff(Tn[:, -6:-3], T1[:, -6:-3]),
        "traj_maxdiff_obj": max_diff(Tn[:, -3:], T1[:, -3:]),
        "image_maxdiff_all": int(np.max(np.abs(In.astype(int) - I1.astype(int)))),
    }

    # (c) reset + init + replay prefix (alternative restore path)
    sc.reset_to_init_state(env, init_states[args.init_idx])
    _, _, obs_s2 = run_actions(env, prefix)
    snap2 = sc.take_snapshot(env)
    Tr, Ir, _ = run_actions(env, X)
    results["reset_and_replay_prefix"] = {
        "snapshot_state_maxdiff": max_diff(snap2.sim_state, snap.sim_state),
        "snapshot_goal_ori_maxdiff": max_diff(snap2.goal_ori, snap.goal_ori),
        "traj_maxdiff_qpos_qvel": max_diff(Tr[:, : e.sim.model.nq + e.sim.model.nv], T1[:, : e.sim.model.nq + e.sim.model.nv]),
        "traj_maxdiff_eef": max_diff(Tr[:, -6:-3], T1[:, -6:-3]),
        "traj_maxdiff_obj": max_diff(Tr[:, -3:], T1[:, -3:]),
        "image_maxdiff_all": int(np.max(np.abs(Ir.astype(int) - I1.astype(int)))),
    }

    # (d) restore into a *fresh* env instance (as used when snapshots are loaded from disk in later scripts)
    env2, _ = sc.make_env(task, args.seed)
    env2.reset()
    env2.set_init_state(init_states[args.init_idx])
    obs_r2 = sc.restore_snapshot(env2, sc.SimSnapshot.from_npz(snap.to_npz_dict()))
    e2 = sc.raw_env(env2)
    Tf, If, _ = run_actions(env2, X)
    results["fresh_env_restore_from_npz"] = {
        "restored_obs_image_maxdiff": int(np.max(np.abs(sc.flipped_agentview(obs_r2).astype(int) - img_s.astype(int)))),
        "traj_maxdiff_qpos_qvel": max_diff(Tf[:, : e2.sim.model.nq + e2.sim.model.nv], T1[:, : e.sim.model.nq + e.sim.model.nv]),
        "traj_maxdiff_eef": max_diff(Tf[:, -6:-3], T1[:, -6:-3]),
        "traj_maxdiff_obj": max_diff(Tf[:, -3:], T1[:, -3:]),
        "image_maxdiff_all": int(np.max(np.abs(If.astype(int) - I1.astype(int)))),
    }
    env2.close()

    # (e) displacement calibration: from snapshot, constant action of +0.9 on x / y / z for 7 steps
    calib = {}
    for axis, name in [(0, "x"), (1, "y"), (2, "z")]:
        sc.restore_snapshot(env, snap)
        p0 = sc.eef_pos(env)
        a = np.zeros((8, 7)); a[:, axis] = 0.9; a[:, 6] = -1
        per_step = []
        for i in range(8):
            env.step(list(a[i]))
            per_step.append(float((sc.eef_pos(env) - p0)[axis]))
        calib[name] = per_step
    results["eef_displacement_for_action_0.9_after_k_steps"] = calib

    # (f) geometry: bowl / plate positions + bowl geom sizes; projection calibration image
    sc.restore_snapshot(env, snap)
    geo = {}
    for name in e.obj_of_interest:
        bid = e.obj_body_id[name]
        geom_ids = [g for g in range(e.sim.model.ngeom) if e.sim.model.geom_bodyid[g] == bid]
        sizes = [e.sim.model.geom_size[g].tolist() for g in geom_ids]
        rb = [float(e.sim.model.geom_rbound[g]) for g in geom_ids]
        geo[name] = {"pos": sc.body_pos(env, name).tolist(), "n_geoms": len(geom_ids), "max_rbound": max(rb) if rb else None, "geom_sizes_first5": sizes[:5]}
    geo["eef_pos"] = sc.eef_pos(env).tolist()
    geo["eef_quat"] = sc.eef_quat(env).tolist()
    results["geometry_at_snapshot"] = geo
    pts = np.stack([sc.eef_pos(env)] + [sc.body_pos(env, n) for n in e.obj_of_interest])
    pix = sc.world_to_model_pixels(env, pts)
    img = img_s.copy()
    for (r, c), col in zip(pix, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        r, c = int(r), int(c)
        img[max(0, r - 3): r + 4, max(0, c - 3): c + 4] = col
    imageio.imwrite(out_dir / "projection_check.png", img)
    results["projection_pixels_rowcol"] = {"eef": pix[0].tolist(), **{n: pix[i + 1].tolist() for i, n in enumerate(e.obj_of_interest)}}
    results["task"] = desc
    results["nq_nv"] = [int(e.sim.model.nq), int(e.sim.model.nv)]
    sc.write_json(out_dir / "restore_test.json", results)
    np.savez_compressed(out_dir / "restore_test_snapshot.npz", **snap.to_npz_dict(), prefix=prefix, X=X, img_s=img_s)
    print(sc.json.dumps(results, indent=1, default=sc._json_default))
    env.close()


if __name__ == "__main__":
    main()
