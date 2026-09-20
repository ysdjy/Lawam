"""G3 state selection: apply the FROZEN phase rule of s2_select_states.py to every stress condition.

Reuses `episode_states()` from the round-1 selector verbatim, so phases are defined exactly as before, and
writes one manifest across all conditions. Episodes marked invalid by a family-C validity check are skipped
here but remain recorded in their `episode.json`.

A state is usable only if the real future frame o_{t+7} exists in the recording — under stress many
episodes are short or fail, so this is checked per episode rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
import probe_common as pc  # noqa: E402
from s2_select_states import CHUNK_LEN, FUTURE_IDX, episode_states  # noqa: E402  (frozen phase rule)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    roll_root = run_dir / "rollouts"
    states_dir = run_dir / "states"
    states_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else run_dir / "states_manifest.jsonl"
    if out_path.exists():
        out_path.unlink()

    counts, invalid, no_state = Counter(), Counter(), Counter()
    for cond_dir in sorted(roll_root.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        for task_dir in sorted(cond_dir.iterdir()):
            for ep_dir in sorted(task_dir.glob("ep*")):
                meta_p = ep_dir / "episode.json"
                if not meta_p.exists():
                    continue
                meta = json.loads(meta_p.read_text())
                if not meta.get("valid", True):
                    invalid[cond] += 1
                    continue
                if not (ep_dir / "steps.npz").exists():
                    no_state[cond] += 1
                    continue
                _, sts = episode_states(ep_dir)
                if not sts:
                    no_state[cond] += 1
                    continue
                data = np.load(ep_dir / "steps.npz")
                for st in sts:
                    step = st["step"]
                    sid = f"{cond}_{meta['suite']}_t{meta['task_id']:02d}_ep{meta['episode_idx']:03d}_s{step:04d}_{st['phase']}"
                    np.savez_compressed(
                        states_dir / f"{sid}.npz",
                        primary=data["primary"][step], wrist=data["wrist"][step],
                        state=data["state"][step],
                        future_primary=data["primary"][step + FUTURE_IDX],
                        future_primary_t8=data["primary"][step + CHUNK_LEN],
                        lang=np.asarray([st["instruction"]]))
                    st.update({"state_id": sid, "condition": cond, "family": meta["family"],
                               "severity": meta["severity"],
                               "state_npz": str(states_dir / f"{sid}.npz")})
                    pc.append_jsonl(out_path, st)
                    counts[cond] += 1
    print("states per condition:")
    for c in sorted(counts):
        print(f"  {c:24s} {counts[c]:4d} states   (invalid episodes {invalid[c]}, episodes with no usable state {no_state[c]})")
    print(f"total {sum(counts.values())}")


if __name__ == "__main__":
    main()
