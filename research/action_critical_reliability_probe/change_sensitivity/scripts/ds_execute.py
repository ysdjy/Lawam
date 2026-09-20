"""Step 2b (sim side, `libero310` env): execute every selector's chunk and measure real consequences.

From each frozen state's exact snapshot, executes 10 chunk variants x 3 flow-noise seeds = 30 executions
per state (2,700 in total), using the official unnormalise + gripper post-processing.

Per selector the reported quantity is the fraction of the ORACLE CORRECTION recovered:
    recovery = 1 - ||eef(selector) - eef(pi(H_real))|| / ||eef(pi(H_pred)) - eef(pi(H_real))||
so 0 = no better than the unrepaired policy, 1 = as good as repairing all 256 tokens.

The two reference scales from round 1 travel with every number: flow-sampler nuisance and chunk motion.
No policy is queried here; nothing is trained.
Output: <run_dir>/ds_execution_records.jsonl
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

PRIOR = REPO / "results/action_critical_reliability_probe/acr_20260919_203850"
NOISE_SEEDS = [101, 202, 303]
SELECTORS = ["A_random", "B_top_D", "C_top_S", "D_top_D_times_S", "E_top_U_oracle",
             "F_top_U_times_S", "G_top_attention", "H_top_D_times_attention"]
VARIANTS = ["A_pred", "A_realFuture"] + SELECTORS
EPS = 1e-12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(PRIOR / "states_manifest.jsonl")]
    if args.limit:
        manifest = manifest[: args.limit]
    manifest.sort(key=lambda m: (m["suite"], m["task_id"], m["episode_idx"], m["step"]))
    action_stats = sc.load_norm_stats(REPO / "results/Checkpoints/libero/lawam_libero_sft_release")
    rec_path = run_dir / "ds_execution_records.jsonl"
    done = {json.loads(l)["state_id"] for l in open(rec_path)} if rec_path.exists() else set()

    env, current = None, None
    for m in manifest:
        sid = m["state_id"]
        if sid in done:
            continue
        key = (m["suite"], m["task_id"])
        if key != current:
            if env is not None:
                env.close()
            suite = sc.load_suite(m["suite"])
            env, _ = sc.make_env(suite.get_task(m["task_id"]), 0)
            sc.reset_to_init_state(env, suite.get_task_init_states(m["task_id"])[m["episode_idx"]])
            current = key
        t0 = time.time()
        snap = sc.SimSnapshot.from_npz(np.load(m["snapshot"]))
        chunks = np.load(run_dir / "ds_chunks" / f"{sid}.npz")
        target = m["target_object"]

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
            return float(np.linalg.norm(res[a][field] - res[b][field]))

        out = {"state_id": sid, "suite": m["suite"], "task_id": m["task_id"], "phase": m["phase"],
               "episode": f"{m['suite']}_t{m['task_id']}_ep{m['episode_idx']}",
               "token_budget": int(chunks["token_budget"][0])}

        # the oracle correction this round is scored against, and the nuisance scale, per seed
        treat = {s: dist(f"A_realFuture_seed{s}", f"A_pred_seed{s}") for s in NOISE_SEEDS}
        out["eef_treatment_mm"] = float(np.mean(list(treat.values()))) * 1000
        out["eef_nuisance_mm"] = float(np.mean([dist(f"A_pred_seed{a}", f"A_pred_seed{b}")
                                                for a, b in itertools.combinations(NOISE_SEEDS, 2)])) * 1000
        out["obj_treatment_mm"] = float(np.mean([dist(f"A_realFuture_seed{s}", f"A_pred_seed{s}", "obj")
                                                 for s in NOISE_SEEDS])) * 1000

        for sel in SELECTORS:
            rec, resid, dev, act = [], [], [], []
            for s in NOISE_SEEDS:
                r = dist(f"{sel}_seed{s}", f"A_realFuture_seed{s}")      # residual to the full oracle
                t = treat[s]
                rec.append(1.0 - r / max(t, EPS))
                resid.append(r * 1000)
                dev.append(dist(f"{sel}_seed{s}", f"A_pred_seed{s}") * 1000)
                act.append(float(np.max(np.abs(res[f"{sel}_seed{s}"]["act"][:, :7]
                                               - res[f"A_pred_seed{s}"]["act"][:, :7]))))
            out[f"recovery_{sel}"] = float(np.mean(rec))
            out[f"residual_mm_{sel}"] = float(np.mean(resid))
            out[f"eef_dev_from_pred_mm_{sel}"] = float(np.mean(dev))
            out[f"action_maxabs_{sel}"] = float(np.mean(act))
            out[f"obj_dev_from_pred_mm_{sel}"] = float(np.mean(
                [dist(f"{sel}_seed{s}", f"A_pred_seed{s}", "obj") for s in NOISE_SEEDS])) * 1000
        out["wall_sec"] = round(time.time() - t0, 2)
        pc.append_jsonl(rec_path, out)
        print(f"{sid}: treat={out['eef_treatment_mm']:.2f}mm nuis={out['eef_nuisance_mm']:.2f}mm | "
              f"D={out['recovery_B_top_D']:.3f} DxS={out['recovery_D_top_D_times_S']:.3f} "
              f"U={out['recovery_E_top_U_oracle']:.3f} UxS={out['recovery_F_top_U_times_S']:.3f} "
              f"Dxatt={out['recovery_H_top_D_times_attention']:.3f} rnd={out['recovery_A_random']:.3f} "
              f"({out['wall_sec']}s)", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
