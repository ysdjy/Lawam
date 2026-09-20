"""G3 pilot step (sim side): execute pi(H_pred) and pi(H_real) to measure the FUTURE-ERROR TREATMENT,
and re-execute pi(H_pred) at different flow-noise seeds to measure the SAMPLER NUISANCE.

Both quantities are executed end-effector displacements in millimetres, from the exact snapshot of the
state, with the identical controller and post-processing used in every previous round:

    treatment = || eef(pi(H_real)) - eef(pi(H_pred)) ||    at the same noise seed, averaged over seeds
    nuisance  = || eef(pi(H_pred)) - eef(pi(H_pred)) ||    across seed pairs

The visual perturbation does not enter here: it acts through the observations that produced the chunks,
and executing an action chunk needs no observation. For family C the perturbation is in the snapshot itself.

No token repair, no selectors, no S in the pilot.
Output: <run_dir>/treatment_execution.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
import probe_common as pc  # noqa: E402
import sim_common as sc  # noqa: E402

NOISE_SEEDS = [101, 202, 303]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    states = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        states = states[: args.limit]
    states.sort(key=lambda s: (s["suite"], s["task_id"], s["condition"], s["episode_idx"], s["step"]))
    action_stats = sc.load_norm_stats(REPO / "results/Checkpoints/libero/lawam_libero_sft_release")
    rec_path = run_dir / "treatment_execution.jsonl"
    done = {json.loads(l)["state_key"] for l in open(rec_path)} if rec_path.exists() else set()

    env, current = None, None
    for st in states:
        key = f"{st['condition']}/{st['state_id']}"
        if key in done:
            continue
        chunk_p = run_dir / "treatment_chunks" / st["condition"] / f"{st['state_id']}.npz"
        if not chunk_p.exists():
            continue
        task_key = (st["suite"], st["task_id"])
        if task_key != current:
            if env is not None:
                env.close()
            suite = sc.load_suite(st["suite"])
            env, _ = sc.make_env(suite.get_task(st["task_id"]), 0)
            sc.reset_to_init_state(env, suite.get_task_init_states(st["task_id"])[st["episode_idx"]])
            current = task_key
        t0 = time.time()
        snap = sc.SimSnapshot.from_npz(np.load(st["snapshot"]))
        chunks = np.load(chunk_p)
        target = st["target_object"]

        res = {}
        for variant, seed in itertools.product(["A_pred", "A_realFuture"], NOISE_SEEDS):
            env_actions = sc.normalized_to_env_action(chunks[f"{variant}_seed{seed}"], action_stats)
            sc.restore_snapshot(env, snap)
            eefs, objs = [], []
            for a in env_actions:
                obs, _, _, _ = env.step(np.asarray(a, dtype=np.float64).tolist())
                eefs.append(sc.eef_pos(env).copy())
                objs.append(sc.body_pos(env, target).copy())
            res[f"{variant}_seed{seed}"] = {"eef": np.stack(eefs)[-1], "obj": np.stack(objs)[-1],
                                            "eef_path": np.stack(eefs)}

        def dist(a, b, field="eef"):
            return float(np.linalg.norm(res[a][field] - res[b][field])) * 1000.0

        treat = [dist(f"A_realFuture_seed{s}", f"A_pred_seed{s}") for s in NOISE_SEEDS]
        nuis = [dist(f"A_pred_seed{a}", f"A_pred_seed{b}") for a, b in itertools.combinations(NOISE_SEEDS, 2)]
        obj_treat = [dist(f"A_realFuture_seed{s}", f"A_pred_seed{s}", "obj") for s in NOISE_SEEDS]
        obj_nuis = [dist(f"A_pred_seed{a}", f"A_pred_seed{b}", "obj")
                    for a, b in itertools.combinations(NOISE_SEEDS, 2)]
        motion = float(np.linalg.norm(res["A_pred_seed101"]["eef"]
                                      - np.asarray(st["eef_pos_t"], dtype=np.float64))) * 1000.0

        out = {"state_key": key, "state_id": st["state_id"], "condition": st["condition"],
               "family": st["family"], "severity": st["severity"], "suite": st["suite"],
               "task_id": st["task_id"], "episode_idx": st["episode_idx"], "phase": st["phase"],
               "step": st["step"], "episode_success": st["episode_success"],
               "episode": f"{st['condition']}_{st['suite']}_t{st['task_id']}_ep{st['episode_idx']}",
               "eef_treatment_mm": float(np.mean(treat)), "eef_nuisance_mm": float(np.mean(nuis)),
               "treatment_over_nuisance": float(np.mean(treat) / max(np.mean(nuis), 1e-9)),
               "obj_treatment_mm": float(np.mean(obj_treat)), "obj_nuisance_mm": float(np.mean(obj_nuis)),
               "chunk_motion_mm": motion,
               "action_maxabs_pred_vs_real": float(np.max(np.abs(
                   chunks["A_pred_seed101"][:, :7] - chunks["A_realFuture_seed101"][:, :7]))),
               "U_mse_global": float((np.asarray(chunks["U_l2"], dtype=np.float64) ** 2).mean() / 768.0),
               "U_l2_median": float(np.median(chunks["U_l2"])),
               "D_median": float(np.median(chunks["D"])), "G_median": float(np.median(chunks["G"])),
               "wall_sec": round(time.time() - t0, 2)}
        pc.append_jsonl(rec_path, out)
        print(f"{key}: treat={out['eef_treatment_mm']:.2f}mm nuis={out['eef_nuisance_mm']:.2f}mm "
              f"ratio={out['treatment_over_nuisance']:.2f} motion={motion:.0f}mm", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
