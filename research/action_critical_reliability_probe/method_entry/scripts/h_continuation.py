"""The headroom test: intervene on ONE chunk, then hand control back to the unmodified LaWAM policy.

For each frozen C2 state and each of the four variants:
  1. restore the exact simulator snapshot (every variant starts independently; variants are never chained),
  2. issue the FIRST policy query with the stored observation, the schedule's first noise tensor and the
     variant's `future_override` (variant A sends none, so the server uses its own live H_pred),
  3. execute that chunk,
  4. continue with the UNMODIFIED policy — no repair, no H_real, no override — until task success or the
     original episode budget (250 steps counted from the episode start) is exhausted.

The stress perturbation persists for the whole continuation, with the episode's own frozen occlusion anchor.
Every policy query carries an explicit noise tensor from the state's schedule, and the four variants of a
state share that schedule, so outcome differences are attributable to the first-chunk intervention.

Nothing is trained; no repository file is modified.
Output: <run_dir>/continuation_records.jsonl
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
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/practical_significance/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402
import sim_common as sc  # noqa: E402
import g3_perturb as gp  # noqa: E402
from examples.LIBERO.eval_files.libero_eval_core import _binarize_gripper_open, invert_gripper_action  # noqa: E402
from h_feasibility import NoiseScheduledClient, make_noise  # noqa: E402
from h_audit1_equivalence import build_variants, noise_schedule  # noqa: E402

CKPT = REPO / "results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
MAX_STEPS = 250                      # identical to the source rollouts
VARIANTS = ["A_baseline", "B_top_D", "C_top_DS", "D_full_oracle"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--g3_run", required=True)
    ap.add_argument("--schedules", default="0")
    ap.add_argument("--states", default="all", help="'all' or a comma-separated list of state ids")
    ap.add_argument("--port", type=int, default=10093)
    args = ap.parse_args()
    run_dir, g3 = Path(args.run_dir), Path(args.g3_run)
    schedules = [int(x) for x in args.schedules.split(",")]
    states = [json.loads(l) for l in open(g3 / "c2/states_manifest.jsonl")]
    if args.states != "all":
        keep = set(args.states.split(","))
        states = [s for s in states if s["state_id"] in keep]
    states.sort(key=lambda s: (s["suite"], s["task_id"], s["condition"], s["episode_idx"]))

    # the episode's frozen perturbation parameters (occlusion anchor / shift size)
    ep_meta = {}
    for p in (g3 / "confirm/rollouts").glob("*/*/ep*/episode.json"):
        m = json.loads(p.read_text())
        ep_meta[(m["condition"], m["suite"], m["task_id"], m["episode_idx"])] = m

    rec_path = run_dir / "continuation_records.jsonl"
    done = {(r["state_id"], r["variant"], r["schedule"])
            for r in map(json.loads, open(rec_path))} if rec_path.exists() else set()

    client = sc.ModelClient(policy_ckpt_path=str(CKPT), port=args.port)
    shim = NoiseScheduledClient(client.client)
    client.client = shim

    env, cur = None, None
    for st in states:
        sid = st["state_id"]
        d = np.load(st["state_npz"], allow_pickle=True)
        hp = np.load(run_dir / "hpred_fp32" / f"{sid}.npz")
        tz = np.load(g3 / "confirm/treatment_chunks" / st["condition"] / f"{sid}.npz")
        variants = build_variants(hp, tz)
        meta = ep_meta[(st["condition"], st["suite"], st["task_id"], st["episode_idx"])]
        pert = None
        if meta["family"] in ("A_occlusion", "B_camera_shift"):
            pv = meta["perturbation"]
            pert = gp.VisualPerturbation(meta["family"], meta["severity"],
                                         tuple(pv["anchor_rc"]) if pv.get("anchor_rc") else None,
                                         pv.get("anchor_source", ""))
        budget = MAX_STEPS - int(st["step"])
        target = st["target_object"]

        key = (st["suite"], st["task_id"])
        if key != cur:
            if env is not None:
                env.close()
            suite = sc.load_suite(st["suite"])
            env, desc = sc.make_env(suite.get_task(st["task_id"]), 0)
            sc.reset_to_init_state(env, suite.get_task_init_states(st["task_id"])[st["episode_idx"]])
            cur = key
        snap = sc.SimSnapshot.from_npz(np.load(st["snapshot"]))
        lang = str(d["lang"][0])

        for sched_id in schedules:
            sched = noise_schedule(sid, sched_id)
            for variant in VARIANTS:
                if (sid, variant, sched_id) in done:
                    continue
                t0 = time.time()
                sc.restore_snapshot(env, snap)
                client.reset(task_description=lang)
                shim.initial_noise, shim.future_override = None, None
                obs, success, steps, qi = None, False, 0, 0
                first_ex = {"primary_image": [np.asarray(d["primary"])],
                            "wrist_image": [np.asarray(d["wrist"])], "lang": lang,
                            "state": np.asarray(d["state"])[None, :],
                            "embodiment_id": 25, "action_hz": 20.0}
                err = None
                try:
                    while steps < budget:
                        if client._get_slot_state(0).needs_query():
                            shim.initial_noise = sched[min(qi, len(sched) - 1)]
                            # ONLY the first query carries the repaired future
                            shim.future_override = variants[variant] if qi == 0 else None
                            qi += 1
                        if steps == 0:
                            ex = first_ex
                        else:
                            ex, img, wrist = sc.policy_example(obs, lang)
                            if pert is not None:
                                ex["primary_image"] = [pert.apply(img)]
                        resp = client.step(example=ex, step=steps)
                        raw = resp["raw_action"]
                        delta = np.concatenate([
                            np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1),
                            np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1),
                            invert_gripper_action(_binarize_gripper_open(
                                np.asarray(raw["open_gripper"], dtype=np.float32).reshape(-1)))]).astype(np.float64)
                        obs, _, done_flag, _ = env.step(delta.tolist())
                        steps += 1
                        if done_flag:
                            success = True
                            break
                except Exception as exc:  # noqa: BLE001
                    err = repr(exc)
                pc.append_jsonl(rec_path, {
                    "state_id": sid, "variant": variant, "schedule": sched_id,
                    "condition": st["condition"], "family": meta["family"], "suite": st["suite"],
                    "task_id": st["task_id"], "episode_idx": st["episode_idx"],
                    "episode": f"{st['condition']}_{st['suite']}_t{st['task_id']}_ep{st['episode_idx']}",
                    "source_episode_success": st["episode_success"],
                    "intervention_step": int(st["step"]), "step_budget": int(budget),
                    "success": bool(success), "steps_used": int(steps),
                    "remaining_steps_to_success": int(steps) if success else None,
                    "n_queries": int(qi), "error": err,
                    "eef_final": sc.eef_pos(env).tolist(), "obj_final": sc.body_pos(env, target).tolist(),
                    "wall_sec": round(time.time() - t0, 2)})
                print(f"{sid} sched{sched_id} {variant:14s}: success={success} steps={steps}/{budget} "
                      f"q={qi} ({time.time()-t0:.1f}s)", flush=True)
    if env is not None:
        env.close()
    client.close()


if __name__ == "__main__":
    main()
