"""Step 1 (materialise the pre-registered splits). Pure bookkeeping; no model, no data inspection.

Applies PROTOCOL_G2.yaml:splits mechanically:
  Split A  4 training-pool tasks, per task: episodes 0-5 train, 6-7 val, 8-9 test (episode-disjoint)
  Split B1 libero_spatial t9  -> test only (unseen task, seen suite)
  Split B2 libero_goal    t5  -> test only (unseen task and unseen suite)
  Split C  the prior run's 3 tasks -> test only (unseen, and already carry frozen S)

The episode is the minimum independent unit; tokens and states are never split randomly.
Writes <run_dir>/DATA_SPLIT.md and <run_dir>/splits.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
import probe_common as pc  # noqa: E402

PRIOR_RUN = REPO / "results/action_critical_reliability_probe/acr_20260919_203850"
TRAIN_POOL = [("libero_spatial", 0), ("libero_spatial", 7), ("libero_object", 3), ("libero_object", 6)]
B1 = [("libero_spatial", 9)]
B2 = [("libero_goal", 5)]
TRAIN_EPISODES, VAL_EPISODES, TEST_EPISODES = set(range(0, 6)), {6, 7}, {8, 9}
N_TOKENS = 256


def assign(suite: str, task_id: int, episode: int) -> str:
    key = (suite, task_id)
    if key in TRAIN_POOL:
        if episode in TRAIN_EPISODES:
            return "train"
        if episode in VAL_EPISODES:
            return "val"
        return "testA"
    if key in B1:
        return "testB1"
    if key in B2:
        return "testB2"
    raise KeyError(f"unregistered task {key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    new = [json.loads(l) for l in open(run_dir / "states_manifest.jsonl")]
    prior = [json.loads(l) for l in open(PRIOR_RUN / "states_manifest.jsonl")]

    rows = []
    for s in new:
        rows.append({"suite": s["suite"], "task_id": s["task_id"], "episode": s["episode_idx"],
                     "phase": s["phase"], "state_id": s["state_id"], "step": s["step"],
                     "episode_success": s["episode_success"],
                     "split": assign(s["suite"], s["task_id"], s["episode_idx"]), "source": "g2"})
    for s in prior:
        rows.append({"suite": s["suite"], "task_id": s["task_id"], "episode": s["episode_idx"],
                     "phase": s["phase"], "state_id": s["state_id"], "step": s["step"],
                     "episode_success": s["episode_success"],
                     "split": "testC", "source": "prior_run"})

    # leakage assertions -------------------------------------------------------
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add((r["suite"], r["task_id"], r["episode"]))
    train_eps = by_split["train"]
    checks = {
        "episode_disjoint_train_val": len(train_eps & by_split["val"]) == 0,
        "episode_disjoint_train_testA": len(train_eps & by_split["testA"]) == 0,
        "episode_disjoint_val_testA": len(by_split["val"] & by_split["testA"]) == 0,
        "no_state_id_in_two_splits": len({r["state_id"] for r in rows}) == len(rows),
    }
    train_tasks = {(s, t) for s, t, _ in train_eps}
    for name in ("testB1", "testB2", "testC"):
        checks[f"task_disjoint_train_{name}"] = len(train_tasks & {(s, t) for s, t, _ in by_split[name]}) == 0
    assert all(checks.values()), checks

    counts = Counter(r["split"] for r in rows)
    ep_counts = {k: len(v) for k, v in by_split.items()}
    payload = {
        "train_pool_tasks": [list(t) for t in TRAIN_POOL],
        "episode_rule": {"train": sorted(TRAIN_EPISODES), "val": sorted(VAL_EPISODES), "test": sorted(TEST_EPISODES)},
        "n_states": dict(counts), "n_episodes": ep_counts,
        "n_tokens": {k: v * N_TOKENS for k, v in counts.items()},
        "leakage_checks": checks, "rows": rows,
    }
    pc.write_json(run_dir / "splits.json", payload)

    lines = [
        "# DATA_SPLIT.md — Learned-U (G2)", "",
        "Generated mechanically from `PROTOCOL_G2.yaml:splits`. The **episode** is the minimum independent",
        "unit; tokens and states are never split randomly. Once written this file is immutable: no split may",
        "be changed after any result is seen.", "",
        "| split | meaning | states | episodes | tokens |", "|---|---|---|---|---|",
    ]
    meaning = {
        "train": "training-pool tasks, episodes 0-5",
        "val": "training-pool tasks, episodes 6-7 (model selection / early stopping)",
        "testA": "training-pool tasks, episodes 8-9 — unseen resets of a SEEN task",
        "testB1": "libero_spatial t9 — unseen TASK, seen suite",
        "testB2": "libero_goal t5 — unseen task AND unseen suite (non-prehensile push)",
        "testC": "prior run's 3 tasks — unseen, and already carry frozen S / attention / consequences",
    }
    for k in ["train", "val", "testA", "testB1", "testB2", "testC"]:
        lines.append(f"| `{k}` | {meaning[k]} | {counts[k]} | {ep_counts[k]} | {counts[k] * N_TOKENS:,} |")
    lines += ["", "## Leakage checks", ""]
    for k, v in checks.items():
        lines.append(f"- `{k}`: **{'PASS' if v else 'FAIL'}**")
    lines += ["", "## Full listing", "",
              "| suite | task | episode | phase | split | success | state_id |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["split"], x["suite"], x["task_id"], x["episode"], x["step"])):
        lines.append(f"| {r['suite']} | {r['task_id']} | {r['episode']} | {r['phase']} | {r['split']} | "
                     f"{r['episode_success']} | `{r['state_id']}` |")
    (run_dir / "DATA_SPLIT.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"states": dict(counts), "episodes": ep_counts, "checks": checks}, indent=1))


if __name__ == "__main__":
    main()
