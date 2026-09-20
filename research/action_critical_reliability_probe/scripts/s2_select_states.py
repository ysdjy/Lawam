"""Select probe states from the recorded rollouts and label their execution phase (pure numpy/json).

One state = one policy query step (chunk boundary) of one episode. Phases follow `protocol.yaml`
(step1_sensitivity.pilot_states.phase_definition) and are derived from the ACTUAL executed trajectory:

  approach      MEDIAN boundary among those where the gripper has not been commanded closed yet and the
                eef is >6 cm from the target object (amendment A5: "first" always selected step 0, which is
                the same pre-motion scene in every episode of a task; the median gives mid-approach states
                that actually differ across episodes. Selection never looks at U or S.)
  pre_grasp     last boundary before the first commanded gripper close
  manipulation  first boundary where the target object has been lifted >2 cm above its initial height

A state is only usable if the real future frame o_{t+7} exists in the recording (the policy executed the
whole chunk), which is what Step 2 will need as the U label.

Writes `<run_dir>/states_manifest.jsonl` and `<run_dir>/states/<state_id>.npz`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402

FUTURE_IDX = 7          # H_pred targets o_{t+7} (training video delta index)
CHUNK_LEN = 8
GRIPPER_CLOSE_SIGN = 1.0   # LIBERO: +1 closes
APPROACH_MIN_DIST = 0.06   # m
LIFT_THRESHOLD = 0.02      # m


def episode_states(ep_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((ep_dir / "episode.json").read_text())
    recs = [json.loads(line) for line in open(ep_dir / "records.jsonl")]
    data = np.load(ep_dir / "steps.npz")
    actions = data["actions"]
    target = meta["objects_of_interest"][0]
    obj_key = f"obj_{target}"
    obj_z0 = meta["object_init_pos"][target][2]

    closes = np.nonzero(np.asarray(actions)[:, 6] > 0.5 * GRIPPER_CLOSE_SIGN)[0]
    first_close = int(closes[0]) if closes.size else None

    boundaries = [b for b in meta["chunk_boundaries"] if b + FUTURE_IDX < len(actions)]
    info = []
    for b in boundaries:
        r = recs[b]
        eef = np.asarray(r["eef_pos"], dtype=np.float64)
        obj = np.asarray(r[obj_key], dtype=np.float64)
        info.append({
            "step": b,
            "dist_eef_obj": float(np.linalg.norm(eef - obj)),
            "obj_lift": float(obj[2] - obj_z0),
            "gripper_closed_before": bool(first_close is not None and b > first_close),
        })

    phases: dict[str, int | None] = {"approach": None, "pre_grasp": None, "manipulation": None}
    approach_candidates = [it["step"] for it in info
                           if not it["gripper_closed_before"]
                           and (first_close is None or it["step"] <= first_close)
                           and it["dist_eef_obj"] > APPROACH_MIN_DIST]
    if approach_candidates:
        phases["approach"] = int(approach_candidates[len(approach_candidates) // 2])
    if first_close is not None:
        before = [it["step"] for it in info if it["step"] <= first_close]
        phases["pre_grasp"] = max(before) if before else None
    for it in info:
        if phases["manipulation"] is None and it["obj_lift"] > LIFT_THRESHOLD:
            phases["manipulation"] = it["step"]

    out = []
    for phase, step in phases.items():
        if step is None:
            continue
        if any(o["step"] == step for o in out):      # one state per step; keep the earliest phase label
            continue
        it = next(i for i in info if i["step"] == step)
        out.append({
            "suite": meta["suite"], "task_id": meta["task_id"], "episode_idx": meta["episode_idx"],
            "instruction": meta["instruction"], "step": step, "phase": phase,
            "dist_eef_obj": it["dist_eef_obj"], "obj_lift": it["obj_lift"],
            "first_close_step": first_close, "episode_success": bool(meta["success"]),
            "num_actions": int(len(actions)), "target_object": target,
            "eef_pos_t": recs[step]["eef_pos"], "eef_pos_t7": recs[step + FUTURE_IDX]["eef_pos"],
            "obj_pos_t": recs[step][obj_key], "obj_pos_t7": recs[step + FUTURE_IDX][obj_key],
            "executed_chunk": np.asarray(actions[step:step + CHUNK_LEN]).tolist(),
            "snapshot": str(ep_dir / "snapshots" / f"step{step:04d}.npz"),
            "rollout_steps_npz": str(ep_dir / "steps.npz"),
        })
    return meta, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    ap.add_argument("--rollouts", nargs="*", default=None, help="subdirectory names under <run_dir>/rollouts")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    roll_root = run_dir / "rollouts"
    names = args.rollouts or sorted(d.name for d in roll_root.iterdir()
                                    if d.is_dir() and any(d.glob("ep*/episode.json")))
    states_dir = run_dir / "states"
    states_dir.mkdir(exist_ok=True)
    manifest_path = run_dir / "states_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    total = 0
    for name in names:
        for ep_dir in sorted((roll_root / name).glob("ep*")):
            if not (ep_dir / "episode.json").exists():
                continue
            meta, states = episode_states(ep_dir)
            data = np.load(ep_dir / "steps.npz")
            for st in states:
                sid = f"{name}_ep{st['episode_idx']:03d}_s{st['step']:04d}_{st['phase']}"
                st["state_id"] = sid
                step = st["step"]
                np.savez_compressed(
                    states_dir / f"{sid}.npz",
                    primary=data["primary"][step], wrist=data["wrist"][step],
                    state=data["state"][step],
                    future_primary=data["primary"][step + FUTURE_IDX],
                    future_wrist=data["wrist"][step + FUTURE_IDX],
                    future_primary_t8=data["primary"][step + CHUNK_LEN],
                    lang=np.asarray([st["instruction"]]),
                )
                st["state_npz"] = str(states_dir / f"{sid}.npz")
                pc.append_jsonl(manifest_path, st)
                total += 1
            print(f"{name} ep{meta['episode_idx']:03d}: {[s['phase'] + '@' + str(s['step']) for s in states]}",
                  flush=True)
    print(f"total states: {total}")


if __name__ == "__main__":
    main()
