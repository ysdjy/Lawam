"""FE Stage A — the arm-independent frozen-encoding cache, and the pre-registered sample sets.

Everything except the Flow Head is frozen and in eval mode and augmentation is off, so the
shared encoding is a deterministic function of the (episode, start) address. Computing it once
is an exact optimisation, not an approximation: both arms, both seeds and every epoch then read
literally the same bytes, which is what makes the fairness guarantee mechanical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common as fc  # noqa: E402

CACHE_ROOT = fc.REPO_ROOT / "results" / "action_critical_reliability_probe" / "fe_cache"

CACHED_KEYS = (
    "h_vlm", "h_t", "h_t1_pred", "h_t1_gt",
    "actions", "actions_mask", "attention_mask", "action_hz", "embodiment_id",
)


# ------------------------------------------------------------------ pre-registered sample sets
def eligible_starts(length: int) -> int:
    """Number of episode-local starts with a real future frame and an unpadded action chunk."""
    return max(0, int(length) - 7)


def select_train_like(lens: dict[int, int], n: int, salt: str) -> list[tuple[int, int]]:
    """PROTOCOL sampling: rank each episode's eligible starts by SHA256(salt|ep|start), then take
    round-robin across episodes in episode-id order until `n` samples are collected."""
    order: dict[int, list[int]] = {}
    for ep in sorted(lens):
        starts = list(range(eligible_starts(lens[ep])))
        starts.sort(key=lambda s: fc.sha256_text(salt, str(ep), str(s)))
        order[ep] = starts
    out: list[tuple[int, int]] = []
    rank = 0
    eps = sorted(lens)
    while len(out) < n:
        progressed = False
        for ep in eps:
            if rank < len(order[ep]):
                out.append((ep, order[ep][rank]))
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            raise RuntimeError(f"only {len(out)} eligible samples available, need {n}")
        rank += 1
    return out


def select_evaluation(lens: dict[int, int], per_episode: int = 4) -> list[tuple[int, int]]:
    """PROTOCOL evaluation starts: `per_episode` evenly spaced eligible starts, deduplicated."""
    out: list[tuple[int, int]] = []
    for ep in sorted(lens):
        last = eligible_starts(lens[ep]) - 1
        if last < 0:
            continue
        idx = sorted({int(round(k * last / (per_episode - 1))) for k in range(per_episode)})
        out.extend((ep, s) for s in idx)
    return out


# ------------------------------------------------------------------ cache build / load
def sample_path(split: str, episode: int, start: int) -> Path:
    return CACHE_ROOT / split / f"{int(episode):05d}" / f"{int(start):04d}.pt"


def build_cache(backend, mixture, leaf, split: str, addresses: list[tuple[int, int]],
                collator, batch_size: int = 1, log_every: int = 200) -> dict:
    """Encode every address once and write it to disk. Idempotent: existing files are skipped.

    batch_size MUST stay 1. The collator right-pads `input_ids` to the batch maximum, and the
    VLM sequence length varies with the task instruction (measured 206-219 tokens). Encoding two
    samples together would therefore make a cached sample's `h_vlm` depend on which other sample
    happened to share its build batch. At batch_size=1 every sample is stored at its own true
    length; `collate_cached` re-creates the official right padding at use time."""
    import time

    todo = [a for a in addresses if not sample_path(split, *a).exists()]
    report = {"split": split, "requested": len(addresses), "already_present": len(addresses) - len(todo)}
    t0 = time.time()
    written = 0
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        samples = [fc.get_sample(mixture, leaf, ep, st) for ep, st in chunk]
        batch = collator(samples)
        batch = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            prepared = backend._prepare_train_batch(batch=batch)
            shared = backend._run_shared_encoding_train(
                prepared_batch=prepared, source="fe_cache.build_cache", lam_features_with_no_grad=True
            )
        src = {
            "h_vlm": shared.h_vlm, "h_t": shared.h_t,
            "h_t1_pred": shared.h_t1_pred, "h_t1_gt": shared.h_t1_gt,
            "actions": prepared["actions"], "actions_mask": prepared["actions_mask"],
            "attention_mask": (prepared["attention_mask"] == 1),
            "action_hz": prepared["action_hz"], "embodiment_id": prepared["embodiment_id"],
        }
        for j, (ep, st) in enumerate(chunk):
            rec = {k: src[k][j].detach().to("cpu").clone() for k in CACHED_KEYS}
            rec["episode"] = int(ep)
            rec["start"] = int(st)
            rec["split"] = split
            p = sample_path(split, ep, st)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(rec, p)
            written += 1
        if log_every and (i // batch_size) % log_every == 0:
            done = i + len(chunk)
            rate = done / max(1e-6, time.time() - t0)
            print(f"[{split}] {done}/{len(todo)}  {rate:.1f} samples/s  "
                  f"eta {int((len(todo)-done)/max(rate,1e-6))}s", flush=True)
    report["written"] = written
    report["seconds"] = round(time.time() - t0, 1)
    return report


def load_sample(split: str, episode: int, start: int) -> dict:
    return torch.load(sample_path(split, episode, start), map_location="cpu", weights_only=False)


def collate_cached(records: list[dict], device: str = "cuda") -> dict:
    """Stack cached per-sample tensors into a batch, on device.

    `h_vlm` and `attention_mask` are stored at each sample's own VLM sequence length and are
    right-padded here to the batch maximum -- zeros for `h_vlm`, False for `attention_mask` --
    which is exactly the layout the official collator produces (verified: padding is right-side
    and marked by attention_mask==0).
    """
    out = {}
    smax = max(int(r["h_vlm"].shape[0]) for r in records)
    for k in CACHED_KEYS:
        vals = [r[k] for r in records]
        if k == "h_vlm":
            padded = []
            for v in vals:
                pad = smax - int(v.shape[0])
                padded.append(torch.nn.functional.pad(v, (0, 0, 0, pad)) if pad else v)
            out[k] = torch.stack(padded, dim=0).to(device, non_blocking=True)
        elif k == "attention_mask":
            padded = []
            for v in vals:
                pad = smax - int(v.shape[0])
                padded.append(torch.nn.functional.pad(v, (0, pad), value=False) if pad else v)
            out[k] = torch.stack(padded, dim=0).to(device, non_blocking=True)
        elif vals[0].ndim == 0:
            out[k] = torch.stack(vals).to(device, non_blocking=True)
        else:
            out[k] = torch.stack(vals, dim=0).to(device, non_blocking=True)
    out["episode"] = [int(r["episode"]) for r in records]
    out["start"] = [int(r["start"]) for r in records]
    out["vlm_seq_lens"] = [int(r["h_vlm"].shape[0]) for r in records]
    return out
