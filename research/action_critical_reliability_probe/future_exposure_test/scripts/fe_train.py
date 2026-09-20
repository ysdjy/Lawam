"""FE Stage B — Flow-Head-only training, one arm, one seed.

Pre-registered in PROTOCOL_FE.yaml. The ONLY difference between the two arms is `p_pred`
(1.0 for Control, 0.5 for Future-Exposed). Everything else -- initialisation, cache bytes,
sample order, optimizer, LR schedule, flow time/noise per row, validation set and RNG, and the
early-stopping rule -- is identical by construction and verified by FE AUDIT 1.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_cache as fca  # noqa: E402
import fe_common as fc  # noqa: E402
import fe_flow as ff  # noqa: E402

ARMS = {"control": 1.0, "future_exposed": 0.5}

# Pre-registered optimizer settings, taken verbatim from starVLA/config/training/train_libero.yaml
LR = 1e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 1e-8
GRAD_CLIP = 1.0
MIN_LR = 5e-7
WARMUP_UPDATES = 100
BATCH_SIZE = 2
ACCUM = 4
SNAPSHOT_EVERY = 300


def lr_at(update: int, max_updates: int) -> float:
    """Linear warmup then cosine to MIN_LR, the official shape at this round's budget."""
    if update < WARMUP_UPDATES:
        return LR * (update + 1) / WARMUP_UPDATES
    t = (update - WARMUP_UPDATES) / max(1, max_updates - WARMUP_UPDATES)
    t = min(1.0, max(0.0, t))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * t))


def permuted(addresses: list[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    """Training order depends on the SEED ONLY, never on the arm."""
    g = torch.Generator().manual_seed(fc.seed_from("FE2026-order", str(seed)) % (2 ** 31))
    perm = torch.randperm(len(addresses), generator=g).tolist()
    return [addresses[i] for i in perm]


@torch.no_grad()
def validate(backend, val_addr, p_pred: float, device: str = "cuda") -> dict:
    """Primary metric: mean flow loss under the arm's OWN future distribution, with a frozen
    validation RNG and a frozen deterministic mask. Also logs the H_pred-only and H_gt-only
    losses and their paired difference -- transparency, not endpoints."""
    was_training = backend.flow.training
    backend.flow.eval()
    n_rep = int(backend.model_cfg.repeated_diffusion_steps)
    tot = {"own": 0.0, "pred": 0.0, "gt": 0.0}
    n_batches = 0
    gt_rows = 0
    total_rows = 0
    for i in range(0, len(val_addr), BATCH_SIZE):
        chunk = val_addr[i:i + BATCH_SIZE]
        recs = [fca.load_sample("validation", ep, st) for ep, st in chunk]
        cb = fca.collate_cached(recs, device=device)
        fseed = fc.seed_from("FE2026-valflow", str(i)) % (2 ** 31)
        mask = ff.deterministic_mask(chunk, p_pred, n_rep, device=device)
        l_own, pm = ff.flow_loss(backend, cb, p_pred=p_pred, flow_seed=fseed, forced_mask=mask)
        l_pred, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=fseed, h_t1_star_override="pred")
        l_gt, _ = ff.flow_loss(backend, cb, p_pred=0.0, flow_seed=fseed, h_t1_star_override="gt")
        tot["own"] += float(l_own); tot["pred"] += float(l_pred); tot["gt"] += float(l_gt)
        gt_rows += int((~pm).sum()); total_rows += int(pm.numel())
        n_batches += 1
    if was_training:
        backend.flow.train()
    return {
        "val_loss_own": tot["own"] / n_batches,
        "val_loss_Hpred": tot["pred"] / n_batches,
        "val_loss_Hgt": tot["gt"] / n_batches,
        "val_paired_delta_pred_minus_gt": (tot["pred"] - tot["gt"]) / n_batches,
        "val_gt_row_fraction": gt_rows / max(1, total_rows),
        "val_batches": n_batches,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-updates", type=int, default=1500)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta-rel", type=float, default=1e-4)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="engineering check only; results are excluded from the round")
    args = ap.parse_args()

    determinism = fc.enable_determinism()  # amendment FE-A5
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"TRAIN_LOG_{args.arm.upper()}_seed{args.seed}.jsonl"
    p_pred = ARMS[args.arm]

    manifest = json.loads((fca.CACHE_ROOT / "CACHE_MANIFEST.json").read_text())
    train_addr = [tuple(a) for a in manifest["train"]["addresses"]]
    val_addr = [tuple(a) for a in manifest["validation"]["addresses"]]

    vla, backend = fc.load_backend(use_bf16=False)
    freeze = fc.freeze_all_but_flow(backend)
    if not freeze["trainable_equals_flow"] or freeze["non_flow_trainable_names"]:
        raise RuntimeError(f"freeze plan violated: {freeze}")
    fc.set_train_modes(backend)
    frozen_before = {g: fc.module_state_hash(getattr(backend, g)) for g in fc.FROZEN_GROUPS}
    flow_before = fc.module_state_hash(backend.flow)

    # Required artefact: the exact realised configuration of this arm.
    (out / f"TRAIN_CONFIG_{args.arm.upper()}.yaml").write_text(
        "# Generated by fe_train.py. The ONLY field that differs between the two arms is p_pred.\n"
        f"arm: {args.arm}\n"
        f"p_pred: {p_pred}                       # 1.0 = 100% H_pred; 0.5 = fixed 50/50\n"
        f"detach_future_feature: true\n"
        f"trainable_module: policy_backend.flow\n"
        f"initial_checkpoint: {fc.CKPT_FILE.relative_to(fc.REPO_ROOT)}\n"
        f"normalization: results/Checkpoints/libero/lawam_libero_sft_release/dataset_statistics.json\n"
        f"use_bf16_cast: false                   # training precision contract (amendment FE-A1)\n"
        f"max_optimizer_updates: {args.max_updates}\n"
        f"validation_every_updates: {args.val_every}\n"
        f"early_stop_patience_validations: {args.patience}\n"
        f"min_delta_relative: {args.min_delta_rel}\n"
        f"optimizer: AdamW\nlearning_rate: {LR}\nbetas: {list(BETAS)}\neps: {EPS}\n"
        f"weight_decay: {WEIGHT_DECAY}\ngradient_clipping: {GRAD_CLIP}\n"
        f"lr_schedule: linear warmup {WARMUP_UPDATES} updates then cosine to {MIN_LR}\n"
        f"per_device_batch_size: {BATCH_SIZE}\ngradient_accumulation_steps: {ACCUM}\n"
        f"samples_per_update: {BATCH_SIZE * ACCUM}\n"
        f"repeated_diffusion_steps: {int(backend.model_cfg.repeated_diffusion_steps)}\n"
        f"n_train_samples: {len(train_addr)}\nn_validation_samples: {len(val_addr)}\n"
        f"seed: {args.seed}\nsmoke: {bool(args.smoke)}\n"
    )

    # The encoder is no longer needed: Stage A is cached. Free it so the optimizer has the GPU.
    for g in fc.FROZEN_GROUPS:
        mod = getattr(backend, g)
        if not torch.is_tensor(mod):
            mod.to("cpu")
    torch.cuda.empty_cache()

    params = [p for p in backend.flow.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
    mask_gen = torch.Generator(device="cuda")
    mask_gen.manual_seed(fc.seed_from("FE2026-mask", str(args.seed)) % (2 ** 31))

    order = permuted(train_addr, args.seed)
    cursor = 0
    best = math.inf
    best_update = -1
    stale = 0
    stopped = None
    gt_rows_total = 0
    rows_total = 0

    fc.append_jsonl(log_path, {
        "event": "start", "arm": args.arm, "p_pred": p_pred, "seed": args.seed,
        "smoke": bool(args.smoke), "max_updates": args.max_updates,
        "val_every": args.val_every, "patience": args.patience,
        "min_delta_rel": args.min_delta_rel,
        "n_train_samples": len(train_addr), "n_val_samples": len(val_addr),
        "batch_size": BATCH_SIZE, "accum": ACCUM,
        "repeated_diffusion_steps": int(backend.model_cfg.repeated_diffusion_steps),
        "freeze": freeze, "flow_hash_before": flow_before, "frozen_hashes_before": frozen_before,
        "determinism": determinism,
        "lr": LR, "betas": list(BETAS), "eps": EPS, "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP, "warmup_updates": WARMUP_UPDATES, "min_lr": MIN_LR,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    t_start = time.time()
    for update in range(args.max_updates):
        lr = lr_at(update, args.max_updates)
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        micro_losses = []
        for micro in range(ACCUM):
            chunk = []
            for _ in range(BATCH_SIZE):
                if cursor >= len(order):
                    order = permuted(train_addr, args.seed * 1000 + (cursor // len(train_addr)))
                    cursor = 0
                chunk.append(order[cursor]); cursor += 1
            recs = [fca.load_sample("train", ep, st) for ep, st in chunk]
            cb = fca.collate_cached(recs)
            fseed = fc.seed_from("FE2026-flow", str(args.seed), str(update), str(micro)) % (2 ** 31)
            loss, pm = ff.flow_loss(backend, cb, p_pred=p_pred, flow_seed=fseed, mask_gen=mask_gen)
            (loss / ACCUM).backward()
            micro_losses.append(float(loss))
            gt_rows_total += int((~pm).sum()); rows_total += int(pm.numel())
        gnorm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()

        if (update + 1) % 10 == 0 or update == 0:
            fc.append_jsonl(log_path, {
                "event": "train", "update": update + 1, "lr": lr,
                "loss": sum(micro_losses) / len(micro_losses),
                "grad_norm": float(gnorm),
                "gt_row_fraction_so_far": gt_rows_total / max(1, rows_total),
                "elapsed_s": round(time.time() - t_start, 1),
            })

        if (update + 1) % args.val_every == 0 or (update + 1) == args.max_updates:
            v = validate(backend, val_addr, p_pred)
            improved = v["val_loss_own"] < best * (1.0 - args.min_delta_rel)
            if improved:
                best = v["val_loss_own"]; best_update = update + 1; stale = 0
                torch.save(backend.flow.state_dict(), out / f"flow_{args.arm}_seed{args.seed}_best.pt")
            else:
                stale += 1
            fc.append_jsonl(log_path, {
                "event": "validate", "update": update + 1, **v,
                "improved": bool(improved), "best": best, "best_update": best_update,
                "stale_validations": stale,
                "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                "elapsed_s": round(time.time() - t_start, 1),
            })
            if stale >= args.patience:
                stopped = "early_stop_patience"
                break

        if (update + 1) % SNAPSHOT_EVERY == 0:
            torch.save(backend.flow.state_dict(),
                       out / f"flow_{args.arm}_seed{args.seed}_u{update + 1}.pt")

    final_update = update + 1
    torch.save(backend.flow.state_dict(), out / f"flow_{args.arm}_seed{args.seed}_final.pt")

    # Hashing reads through .to("cpu"), so the frozen modules are hashed where they already are.
    frozen_after = {g: fc.module_state_hash(getattr(backend, g)) for g in fc.FROZEN_GROUPS}
    flow_after = fc.module_state_hash(backend.flow)

    fc.append_jsonl(log_path, {
        "event": "end", "stopped": stopped or "budget_exhausted",
        "final_update": final_update, "best_update": best_update, "best_val_loss_own": best,
        "realised_gt_row_fraction": gt_rows_total / max(1, rows_total),
        "target_gt_row_fraction": 1.0 - p_pred,
        "frozen_hashes_after": frozen_after,
        "frozen_unchanged": frozen_after == frozen_before,
        "flow_hash_after": flow_after, "flow_changed": flow_after != flow_before,
        "wall_seconds": round(time.time() - t_start, 1),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(json.dumps({
        "arm": args.arm, "seed": args.seed, "final_update": final_update,
        "best_update": best_update, "best_val_loss_own": best,
        "realised_gt_row_fraction": gt_rows_total / max(1, rows_total),
        "frozen_unchanged": frozen_after == frozen_before,
        "flow_changed": flow_after != flow_before,
        "wall_seconds": round(time.time() - t_start, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
