"""G3 stage C2 (sim side): execute every selector's chunk under the same stress and measure recovery.

Same protocol as the DS round: from each state's exact snapshot, 10 variants x 3 flow-noise seeds, using
the official unnormalise + gripper post-processing. Recovery is the fraction of the ORACLE CORRECTION of
the executed end-effector position that a selector's 64-token repair achieves:

    recovery = 1 - ||eef(selector) - eef(pi(H_real))|| / ||eef(pi(H_pred)) - eef(pi(H_real))||

Output: <c2>/c2_execution_records.jsonl
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
SELECTORS = ["A_random", "B_top_D", "C_top_S", "D_top_D_times_S", "E_top_attention",
             "F_top_D_times_attention", "G_top_U_oracle", "H_top_U_times_S_oracle"]
VARIANTS = ["A_pred", "A_realFuture"] + SELECTORS
EPS = 1e-12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    c2 = Path(args.c2_dir)
    states = [json.loads(l) for l in open(c2 / "states_manifest.jsonl")]
    if args.limit:
        states = states[: args.limit]
    states.sort(key=lambda s: (s["suite"], s["task_id"], s["condition"], s["episode_idx"]))
    action_stats = sc.load_norm_stats(REPO / "results/Checkpoints/libero/lawam_libero_sft_release")
    rec_path = c2 / "c2_execution_records.jsonl"
    done = {json.loads(l)["state_id"] for l in open(rec_path)} if rec_path.exists() else set()

    env, current = None, None
    for st in states:
        sid = st["state_id"]
        if sid in done:
            continue
        chunk_p = c2 / "c2_chunks" / f"{sid}.npz"
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
        for variant, seed in itertools.product(VARIANTS, NOISE_SEEDS):
            env_actions = sc.normalized_to_env_action(chunks[f"{variant}_seed{seed}"], action_stats)
            sc.restore_snapshot(env, snap)
            eefs, objs = [], []
            for a in env_actions:
                obs, _, _, _ = env.step(np.asarray(a, dtype=np.float64).tolist())
                eefs.append(sc.eef_pos(env).copy())
                objs.append(sc.body_pos(env, target).copy())
            res[f"{variant}_seed{seed}"] = {"eef": np.stack(eefs)[-1], "obj": np.stack(objs)[-1],
                                            "act": chunks[f"{variant}_seed{seed}"]}

        def dist(a, b, field="eef"):
            return float(np.linalg.norm(res[a][field] - res[b][field])) * 1000.0

        treat = {s: dist(f"A_realFuture_seed{s}", f"A_pred_seed{s}") for s in NOISE_SEEDS}
        out = {"state_id": sid, "condition": st["condition"], "family": st["family"],
               "suite": st["suite"], "task_id": st["task_id"], "episode_idx": st["episode_idx"],
               "phase": st["phase"], "episode_success": st["episode_success"],
               "episode": f"{st['condition']}_{st['suite']}_t{st['task_id']}_ep{st['episode_idx']}",
               "token_budget": int(chunks["token_budget"][0]),
               "eef_treatment_mm": float(np.mean(list(treat.values()))),
               "eef_nuisance_mm": float(np.mean([dist(f"A_pred_seed{a}", f"A_pred_seed{b}")
                                                 for a, b in itertools.combinations(NOISE_SEEDS, 2)])),
               "wall_sec": 0.0}
        for sel in SELECTORS:
            rec, resid, dev = [], [], []
            for s in NOISE_SEEDS:
                r = dist(f"{sel}_seed{s}", f"A_realFuture_seed{s}")
                rec.append(1.0 - r / max(treat[s], EPS))
                resid.append(r)
                dev.append(dist(f"{sel}_seed{s}", f"A_pred_seed{s}"))
            out[f"recovery_{sel}"] = float(np.mean(rec))
            out[f"residual_mm_{sel}"] = float(np.mean(resid))
            out[f"eef_dev_from_pred_mm_{sel}"] = float(np.mean(dev))
            out[f"obj_dev_from_pred_mm_{sel}"] = float(np.mean(
                [dist(f"{sel}_seed{s}", f"A_pred_seed{s}", "obj") for s in NOISE_SEEDS]))
        out["wall_sec"] = round(time.time() - t0, 2)
        pc.append_jsonl(rec_path, out)
        print(f"{sid}: treat={out['eef_treatment_mm']:.2f}mm nuis={out['eef_nuisance_mm']:.2f}mm | "
              f"D={out['recovery_B_top_D']:.3f} S={out['recovery_C_top_S']:.3f} "
              f"DxS={out['recovery_D_top_D_times_S']:.3f} Dxatt={out['recovery_F_top_D_times_attention']:.3f} "
              f"UxS={out['recovery_H_top_U_times_S_oracle']:.3f} ({out['wall_sec']}s)", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
