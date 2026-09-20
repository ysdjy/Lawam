"""Step 4 Level 2 (sim side, `libero310` env): execute the chunks and measure real consequences.

From each state's exact snapshot, execute:
  A_pred / A_realFuture / A_topU_corrected / A_topUS_corrected, each for 3 flow-noise seeds,
using the official unnormalise + gripper post-processing, and record where the end-effector and the target
object end up after the chunk.

Two reference scales come out of the same table:
  * treatment  = A_realFuture vs A_pred at the SAME noise seed   (consequence of the prediction error)
  * nuisance   = A_pred at seed a vs A_pred at seed b            (consequence of the sampler's own noise)

No policy is queried here and nothing is trained.
Output: <run_dir>/consequence_records.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402
import sim_common as sc  # noqa: E402

NOISE_SEEDS = [101, 202, 303]
VARIANTS = ["A_pred", "A_realFuture", "A_topU_corrected", "A_topUS_corrected"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(run_dir / "states_manifest.jsonl")]
    if args.limit:
        manifest = manifest[: args.limit]
    manifest.sort(key=lambda m: (m["suite"], m["task_id"], m["episode_idx"], m["step"]))
    # sim_common.load_norm_stats already returns stats["franka"]["action"]
    action_stats = sc.load_norm_stats(pc.REPO_ROOT / "results/Checkpoints/libero/lawam_libero_sft_release")
    rec_path = run_dir / "consequence_records.jsonl"
    done = {json.loads(l)["state_id"] for l in open(rec_path)} if rec_path.exists() else set()

    env = None
    current = None
    for m in manifest:
        sid = m["state_id"]
        if sid in done:
            continue
        key = (m["suite"], m["task_id"])
        if key != current:
            if env is not None:
                env.close()
            suite = sc.load_suite(m["suite"])
            task = suite.get_task(m["task_id"])
            env, _ = sc.make_env(task, 0)
            sc.reset_to_init_state(env, suite.get_task_init_states(m["task_id"])[m["episode_idx"]])
            current = key
        t0 = time.time()
        snap = sc.SimSnapshot.from_npz(np.load(m["snapshot"]))
        chunks = np.load(run_dir / "consequence_chunks" / f"{sid}.npz")
        target = m["target_object"]

        res: dict[str, dict] = {}
        for variant, seed in itertools.product(VARIANTS, NOISE_SEEDS):
            env_actions = sc.normalized_to_env_action(chunks[f"{variant}_seed{seed}"], action_stats)
            sc.restore_snapshot(env, snap)
            eefs, objs, grips = [], [], []
            for a in env_actions:
                obs, _, _, _ = env.step(np.asarray(a, dtype=np.float64).tolist())
                eefs.append(sc.eef_pos(env).copy())
                objs.append(sc.body_pos(env, target).copy())
                grips.append(np.asarray(sc.gripper_qpos(env, obs), dtype=np.float64).copy())
            res[f"{variant}_seed{seed}"] = {
                "eef_final": np.stack(eefs)[-1], "obj_final": np.stack(objs)[-1],
                "eef_t7": np.stack(eefs)[6], "obj_t7": np.stack(objs)[6],
                "grip_final": np.stack(grips)[-1],
                "gripper_cmd": env_actions[:, 6].tolist(),
            }

        def dist(a: str, b: str, field: str = "eef_final") -> float:
            return float(np.linalg.norm(res[a][field] - res[b][field]))

        treat = [dist(f"A_realFuture_seed{s}", f"A_pred_seed{s}") for s in NOISE_SEEDS]
        nuis = [dist(f"A_pred_seed{a}", f"A_pred_seed{b}") for a, b in itertools.combinations(NOISE_SEEDS, 2)]
        topu = [dist(f"A_topU_corrected_seed{s}", f"A_pred_seed{s}") for s in NOISE_SEEDS]
        topus = [dist(f"A_topUS_corrected_seed{s}", f"A_pred_seed{s}") for s in NOISE_SEEDS]
        # how much of the full oracle correction does each partial correction recover?
        rec_u = [dist(f"A_topU_corrected_seed{s}", f"A_realFuture_seed{s}") for s in NOISE_SEEDS]
        rec_us = [dist(f"A_topUS_corrected_seed{s}", f"A_realFuture_seed{s}") for s in NOISE_SEEDS]

        out = {
            "state_id": sid, "suite": m["suite"], "phase": m["phase"], "episode_idx": m["episode_idx"],
            "step": m["step"], "token_budget": int(chunks["token_budget"][0]),
            "overlap_topU_topUS": int(chunks["overlap_topU_topUS"][0]),
            "eef_treatment_m": float(np.mean(treat)), "eef_treatment_all": treat,
            "eef_nuisance_m": float(np.mean(nuis)), "eef_nuisance_all": nuis,
            "eef_topU_vs_pred_m": float(np.mean(topu)), "eef_topUS_vs_pred_m": float(np.mean(topus)),
            "eef_topU_residual_to_real_m": float(np.mean(rec_u)),
            "eef_topUS_residual_to_real_m": float(np.mean(rec_us)),
            "obj_treatment_m": float(np.mean([dist(f"A_realFuture_seed{s}", f"A_pred_seed{s}", "obj_final")
                                              for s in NOISE_SEEDS])),
            "obj_nuisance_m": float(np.mean([dist(f"A_pred_seed{a}", f"A_pred_seed{b}", "obj_final")
                                             for a, b in itertools.combinations(NOISE_SEEDS, 2)])),
            "gripper_cmd_differs": bool(any(
                res[f"A_realFuture_seed{s}"]["gripper_cmd"] != res[f"A_pred_seed{s}"]["gripper_cmd"]
                for s in NOISE_SEEDS)),
            "eef_displacement_of_chunk_m": float(np.linalg.norm(
                res["A_pred_seed101"]["eef_final"] - np.asarray(m["eef_pos_t"], dtype=np.float64))),
            "wall_sec": round(time.time() - t0, 2),
        }
        pc.append_jsonl(rec_path, out)
        print(f"{sid}: treatment={out['eef_treatment_m']*1000:.2f} mm  nuisance={out['eef_nuisance_m']*1000:.2f} mm  "
              f"topU={out['eef_topU_vs_pred_m']*1000:.2f}  topUS={out['eef_topUS_vs_pred_m']*1000:.2f}  "
              f"chunk_motion={out['eef_displacement_of_chunk_m']*1000:.0f} mm ({out['wall_sec']}s)", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
