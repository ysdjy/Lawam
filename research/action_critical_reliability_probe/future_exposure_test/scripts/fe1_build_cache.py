"""FE Stage A driver — select the pre-registered sample sets and build the frozen-encoding cache.

For each split the loader is restricted to THAT split's episodes before any sample is fetched, so
the disjointness in SPLIT_MANIFEST.json is enforced mechanically rather than trusted. Writes
CACHE_MANIFEST.json, which every later script reads as the single source of addresses.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_cache as fca  # noqa: E402
import fe_common as fc  # noqa: E402

N_TRAIN = 8000
N_VAL = 512
EVAL_PER_EPISODE = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test_seen_task", "test_heldout_task"])
    ap.add_argument("--limit", type=int, default=0, help="debug: cap samples per split")
    args = ap.parse_args()

    man = fc.load_split_manifest()
    episodes = {k: sorted(int(e) for e in v) for k, v in man["episodes"].items()}

    vla, backend = fc.load_backend(use_bf16=False)
    fc.freeze_all_but_flow(backend)
    backend.eval()  # Stage A is pure encoding: the whole model is in eval mode
    cfg = fc.load_train_cfg()
    ds_all = fc.build_dataset(cfg, dataset_statistics_override=None)
    collator = fc.build_collator(cfg, training=True)
    leaf = fc._inner_datasets(ds_all)[0]

    manifest_path = fca.CACHE_ROOT / "CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("_meta", {})
    manifest["_meta"].update({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split_manifest_sha256": fc.SPLIT_MANIFEST_SHA256,
        "checkpoint": str(fc.CKPT_FILE.relative_to(fc.REPO_ROOT)),
        "use_bf16_cast": False,
        "n_train": N_TRAIN, "n_val": N_VAL, "eval_per_episode": EVAL_PER_EPISODE,
        "cached_keys": list(fca.CACHED_KEYS),
    })

    for split in args.splits:
        eps = episodes[split]
        restrict = fc.restrict_to_episodes(ds_all, eps)
        lens = fc.episode_lengths(leaf)
        assert sorted(lens) == eps, f"{split}: restricted episode set does not match the manifest"

        if split == "train":
            addr = fca.select_train_like(lens, N_TRAIN, "FE2026-train")
        elif split == "validation":
            addr = fca.select_train_like(lens, N_VAL, "FE2026-val")
        else:
            addr = fca.select_evaluation(lens, EVAL_PER_EPISODE)
        if args.limit:
            addr = addr[:args.limit]

        print(f"\n=== {split}: {len(eps)} episodes, {restrict['total_active_steps']} steps, "
              f"{len(addr)} samples ===", flush=True)
        rep = fca.build_cache(backend, ds_all, leaf, split, addr, collator)
        print(json.dumps(rep, indent=2), flush=True)

        manifest[split] = {
            "episodes": eps,
            "n_episodes": len(eps),
            "active_steps": restrict["total_active_steps"],
            "n_samples": len(addr),
            "n_unique_episodes_in_addresses": len({a[0] for a in addr}),
            "addresses": [[int(a), int(b)] for a, b in addr],
            "build_report": rep,
        }
        fc.write_json(manifest_path, manifest)

    # Disjointness of the realised address sets, as built.
    keys = [k for k in args.splits if k in manifest]
    ep_sets = {k: {a[0] for a in manifest[k]["addresses"]} for k in keys}
    manifest["_meta"]["realised_pairwise_episode_overlap"] = {
        f"{a}&{b}": len(ep_sets[a] & ep_sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]
    }
    fc.write_json(manifest_path, manifest)
    print("\n" + json.dumps(manifest["_meta"], indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
