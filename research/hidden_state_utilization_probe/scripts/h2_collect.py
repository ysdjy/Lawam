"""Collect the teacher dataset: 30 NOMINAL + 30 HIGH episodes (init 0-29), episode-split.

Per policy-query boundary (every 8 env steps) one sample is stored: the observation the policy
would see, the proprio state, the instruction, and the NEXT 8 teacher actions, in env space and in
the checkpoint's normalized action space. Images are kept so the encoding cache can be rebuilt.

Usage: python h2_collect.py <run_id> [n_episodes_per_level]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsu_common as hc  # noqa: E402
import sim_common as sc  # noqa: E402

RUN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUT = hc.OUT_ROOT / RUN
EP_DIR = OUT / "episodes"
EP_DIR.mkdir(parents=True, exist_ok=True)
SPLIT = {"train": list(range(0, 20)), "val": list(range(20, 25)), "test": list(range(25, 30))}


def main() -> None:
    stats = hc.load_norm_stats()
    suite, task, env, desc = hc.make_task_env()
    inits = suite.get_task_init_states(hc.TASK_ID)
    meta = []
    for level in ("NOMINAL", "HIGH"):
        for k in range(N):
            f = EP_DIR / f"{level}_init{k:02d}.npz"
            if f.exists():
                meta.append(json.loads((EP_DIR / f"{level}_init{k:02d}.json").read_text()))
                continue
            t0 = time.time()
            obs, tgt, mu = hc.reset_with_level(env, inits[k], level)
            r = hc.run_teacher_episode(env, tgt, record=True)
            tgt.restore_nominal()

            acts = np.asarray(r["actions"], np.float32)              # [T, 7] env space
            samples = []
            for rec in r["obs_log"]:
                s = rec["step"]
                chunk = acts[s:s + hc.CHUNK]
                if chunk.shape[0] < hc.CHUNK:
                    break                                             # drop a truncated tail chunk
                o = rec["obs"]
                ex, _, _ = sc.policy_example(o, desc)
                samples.append({
                    "step": s,
                    "primary": np.asarray(ex["primary_image"][0], np.uint8),
                    "wrist": np.asarray(ex["wrist_image"][0], np.uint8),
                    "state": np.asarray(ex["state"], np.float32).reshape(-1),
                    "env_chunk": chunk,
                    "norm_chunk": hc.env_action_to_normalized(chunk, stats),
                })
            if not samples:
                print(f"{level} init{k}: no samples", flush=True)
                continue
            np.savez_compressed(
                f,
                primary=np.stack([s["primary"] for s in samples]),
                wrist=np.stack([s["wrist"] for s in samples]),
                state=np.stack([s["state"] for s in samples]),
                env_chunk=np.stack([s["env_chunk"] for s in samples]),
                norm_chunk=np.stack([s["norm_chunk"] for s in samples]),
                step=np.array([s["step"] for s in samples], np.int32),
            )
            m = {"level": level, "init": k, "mu_eff": float(mu), "c": hc.context_value(mu),
                 "success": r["success"], "steps": r["steps"], "n_samples": len(samples),
                 "progress_mm": round(r["progress_m"] * 1000, 2), "contact_steps": r["contact_steps"],
                 "max_abs_action": round(r["max_abs_action"], 4), "lang": desc,
                 "split": next(s for s, v in SPLIT.items() if k in v), "file": str(f),
                 "sec": round(time.time() - t0, 1)}
            (EP_DIR / f"{level}_init{k:02d}.json").write_text(json.dumps(m))
            meta.append(m)
            print(json.dumps({kk: m[kk] for kk in ("level", "init", "success", "steps", "n_samples",
                                                   "progress_mm", "split")}), flush=True)
    env.close()

    split = {"policy": "episode-level; no chunk of an episode is shared between splits",
             "episodes": SPLIT, "n_per_level": N,
             "success_rate": {lv: sum(m["success"] for m in meta if m["level"] == lv) / max(1, len([m for m in meta if m["level"] == lv])) for lv in ("NOMINAL", "HIGH")},
             "n_samples": {lv: sum(m["n_samples"] for m in meta if m["level"] == lv) for lv in ("NOMINAL", "HIGH")},
             "n_samples_by_split": {s: sum(m["n_samples"] for m in meta if m["split"] == s) for s in SPLIT},
             "episodes_meta": meta}
    sc.write_json(OUT / "SPLIT.json", split)
    print(json.dumps({k: v for k, v in split.items() if k != "episodes_meta"}, indent=2))


if __name__ == "__main__":
    main()
