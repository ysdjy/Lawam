"""FE evaluation — the 2x2 on held-out adaptation samples.

For every evaluation sample the flow time tau, the flow noise epsilon and the resulting noisy
action are IDENTICAL across the four cells; only `h_t1_star` changes. That removes the
flow-sampler nuisance from the paired contrast by construction, which matters here because
round 1 measured that nuisance at 0.474 mm against a 0.904 mm treatment.

The eval-mode future selector in the model always returns H_pred, so the four cells can only be
produced by calling the Flow Head with an explicit `h_t1_star`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_cache as fca  # noqa: E402
import fe_common as fc  # noqa: E402
import fe_flow as ff  # noqa: E402

from starVLA.model.framework.vlas.lawam import _cuda_autocast  # noqa: E402

R_REPEATS = 8
ACTION_NOISE_SEEDS = [101, 202, 303]
MM_PER_NORMALIZED_UNIT_PER_STEP = 46.875  # round-1 convention, unchanged
TRANS, ROT, GRIP = slice(0, 3), slice(3, 6), 6


def action_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Errors of a predicted [T,>=7] normalized chunk against the ground-truth chunk."""
    d = np.asarray(pred, dtype=np.float64)[:, :7] - np.asarray(gt, dtype=np.float64)[:, :7]
    per_axis_mm = np.abs(d[:, TRANS]).sum(axis=0) * MM_PER_NORMALIZED_UNIT_PER_STEP
    return {
        "l2_norm_all7": float(np.sqrt((d ** 2).mean())),
        "translation_l2": float(np.sqrt((d[:, TRANS] ** 2).mean())),
        "rotation_l2": float(np.sqrt((d[:, ROT] ** 2).mean())),
        "gripper_absdiff": float(np.abs(d[:, GRIP]).max()),
        "translation_mm_equiv": float(np.linalg.norm(per_axis_mm)),
    }


@torch.no_grad()
def sample_actions(backend, cb: dict, condition: str, noise: torch.Tensor) -> np.ndarray:
    """`noise` may carry K rows: the K action-noise seeds are sampled in ONE batch. Both
    conditions use the identical batch shape, so the pairing is unaffected and every cell sees
    the same reduction order."""
    k = int(noise.shape[0])
    h_t1 = (cb["h_t1_pred"] if condition == "pred" else cb["h_t1_gt"]).expand(k, -1, -1)
    b = k
    state = torch.zeros(b, int(backend.flow.config.state_dim), device=h_t1.device, dtype=torch.float32)
    with _cuda_autocast(ff.FLOW_STAGE_DTYPE):
        a = backend.flow.sample_actions_cfg(
            h_t=cb["h_t"].expand(k, -1, -1), h_t1_star=h_t1,
            h_vlm=cb["h_vlm"].expand(k, -1, -1),
            state=state, state_mask=torch.zeros_like(state, dtype=torch.bool),
            action_hz=cb["action_hz"].expand(k), embodiment_id=cb["embodiment_id"].expand(k),
            cfg_scale=float(backend.flow.config.cfg_guidance_scale),
            num_inference_steps=int(backend.flow.config.num_inference_steps),
            attention_mask=cb["attention_mask"].expand(k, -1), return_padded=False,
            initial_noise=noise,
        )
    return a.detach().float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained Flow-Head state_dict (.pt)")
    ap.add_argument("--arm", required=True, choices=["control", "future_exposed"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="best", choices=["best", "final", "u300"],
                    help="amendment FE-A6: 'best' is the primary evaluation, 'final' secondary; "
                         "'u300' realises the pre-registered equal-update comparison")
    ap.add_argument("--splits", nargs="*", default=["test_seen_task", "test_heldout_task"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-actions", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    determinism = fc.enable_determinism()  # amendment FE-A5
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((fca.CACHE_ROOT / "CACHE_MANIFEST.json").read_text())

    vla, backend = fc.load_backend(use_bf16=False)
    fc.freeze_all_but_flow(backend)
    sd = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    incompatible = backend.flow.load_state_dict(sd, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Flow-Head checkpoint does not match the runtime head: {incompatible}")
    backend.eval(); backend.flow.eval()
    for g in fc.FROZEN_GROUPS:
        m = getattr(backend, g)
        if not torch.is_tensor(m):
            m.to("cpu")
    torch.cuda.empty_cache()
    loaded_hash = fc.module_state_hash(backend.flow)

    flow_rows, action_rows = [], []
    t0 = time.time()
    for split in args.splits:
        addr = [tuple(a) for a in manifest[split]["addresses"]]
        if args.limit:
            addr = addr[:args.limit]
        t_split = time.time()   # per-split, so the printed rate is the CURRENT rate
        for i, (ep, st) in enumerate(addr):
            rec = fca.load_sample(split, ep, st)
            cb = fca.collate_cached([rec])
            fseed = fc.seed_from("FE2026-evalflow", split, str(ep), str(st)) % (2 ** 31)
            row = {"arm": args.arm, "seed": args.seed, "tag": args.tag, "split": split,
                   "episode": ep, "start": st, "flow_seed": fseed}
            for cell, cond in (("P", "pred"), ("G", "gt")):
                l, _ = ff.flow_loss(backend, cb, p_pred=1.0, flow_seed=fseed,
                                    h_t1_star_override=cond, repeat_steps=R_REPEATS)
                row[f"flow_loss_{cell}"] = float(l)
            row["flow_delta_P_minus_G"] = row["flow_loss_P"] - row["flow_loss_G"]
            flow_rows.append(row)

            if not args.skip_actions:
                n_valid = int(cb["actions_mask"][0].any(dim=-1).sum())
                gt_chunk = cb["actions"][0, :n_valid].float().cpu().numpy()
                # The K action-noise tensors are built once and reused for BOTH conditions, so
                # the two cells are paired on initial noise exactly as the protocol requires.
                noises = []
                for ns in ACTION_NOISE_SEEDS:
                    g = torch.Generator(device="cuda")
                    g.manual_seed(fc.seed_from("FE2026-actnoise", str(ns), str(ep), str(st)) % (2 ** 31))
                    noises.append(torch.randn(cb["actions"].shape[1:], generator=g,
                                              device="cuda", dtype=torch.float32))
                noise_k = torch.stack(noises, dim=0)
                best = {}
                for cond, cell in (("pred", "P"), ("gt", "G")):
                    acts = sample_actions(backend, cb, cond, noise_k)
                    errs = [action_errors(acts[j][:n_valid], gt_chunk) for j in range(len(noises))]
                    k = min(range(len(errs)), key=lambda j: errs[j]["l2_norm_all7"])
                    for m, v in errs[k].items():
                        best[f"{m}_{cell}"] = v
                    best[f"best_of_k_index_{cell}"] = k
                action_rows.append({"arm": args.arm, "seed": args.seed, "tag": args.tag,
                                    "split": split,
                                    "episode": ep, "start": st, "n_valid_steps": n_valid, **best})
            if (i + 1) % 100 == 0:
                print(f"[{split}] {i+1}/{len(addr)}  "
                      f"{(i+1)/max(1e-6, time.time()-t_split):.2f}/s", flush=True)

    tag = f"{args.arm}_seed{args.seed}_{args.tag}"
    fp = out / f"OFFLINE_FLOW_METRICS_{tag}.csv"
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flow_rows[0].keys())); w.writeheader(); w.writerows(flow_rows)
    print(f"wrote {fp}  ({len(flow_rows)} rows)")
    if action_rows:
        ap_ = out / f"OFFLINE_ACTION_METRICS_{tag}.csv"
        with open(ap_, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(action_rows[0].keys())); w.writeheader(); w.writerows(action_rows)
        print(f"wrote {ap_}  ({len(action_rows)} rows)")

    fc.write_json(out / f"EVAL_META_{tag}.json", {
        "checkpoint": args.checkpoint, "flow_state_hash_after_load": loaded_hash,
        "determinism": determinism, "flow_mode": "eval (dropout off, deployment path)",
        "R_repeats": R_REPEATS, "action_noise_seeds": ACTION_NOISE_SEEDS,
        "splits": args.splits, "n_flow_rows": len(flow_rows), "n_action_rows": len(action_rows),
        "seconds": round(time.time() - t0, 1),
        "mm_per_normalized_unit_per_step": MM_PER_NORMALIZED_UNIT_PER_STEP,
    })


if __name__ == "__main__":
    main()
