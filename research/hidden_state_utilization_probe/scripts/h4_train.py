"""Train ONE adaptation arm (control or hidden) on the teacher chunks.

The two arms are run as two invocations of this script with the same seed, so every matched item
(batch order, flow noise/time draws, initialisation, optimiser, schedule) is identical and the ONLY
difference is the scalar fed to the context projector: 0 for control, c = log(mu_eff/0.95) for
hidden.

Usage: python h4_train.py <run_id> control|hidden
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsu_model as hm  # noqa: E402

# importable from other scripts: the run id may come from the environment instead of argv
RUN = os.environ.get("HSU_RUN") or (sys.argv[1] if len(sys.argv) > 1 else "")
ARM = sys.argv[2] if len(sys.argv) > 2 else "hidden"
OUT = hm.REPO / "results" / "hidden_state_utilization_probe" / RUN
CACHE = OUT / "cache"
SEED, LR, BATCH, MAX_UPDATES, VAL_EVERY = 0, 1e-3, 8, 2000, 100
SPLIT = {"train": range(0, 20), "val": range(20, 25), "test": range(25, 30)}


def load_split(split: str) -> list[dict]:
    keep = set(SPLIT[split])
    out = []
    for f in sorted(CACHE.glob("*.npz")):
        level, init = f.stem.split("_init")
        if int(init) not in keep:
            continue
        # np.load on an .npz decompresses a whole member on every access. Keep one array per
        # member and let per-sample slices share it, rather than retaining a full copy per sample.
        with np.load(f) as d:
            arrays = {k: d[k] for k in ("h_t", "h_pred", "h_vlm", "attn", "norm_chunk", "c")}
        for i in range(arrays["h_t"].shape[0]):
            out.append({"h_t": arrays["h_t"][i], "h_pred": arrays["h_pred"][i],
                        "h_vlm": arrays["h_vlm"][i], "attn": arrays["attn"][i],
                        "norm_chunk": arrays["norm_chunk"][i],
                        "c": float(arrays["c"][i]), "level": level, "episode": f.stem, "idx": i})
    return out


def to_batch(samples, flow, dev, dt, wrong: bool = False, shuffled_c=None):
    h_t = torch.as_tensor(np.stack([s["h_t"] for s in samples])).to(dev, dt)
    h_pred = torch.as_tensor(np.stack([s["h_pred"] for s in samples])).to(dev, dt)
    h_vlm = torch.as_tensor(np.stack([s["h_vlm"] for s in samples])).to(dev, dt)
    attn = torch.as_tensor(np.stack([s["attn"] for s in samples])).to(dev)
    acts, masks = zip(*[hm.build_actions_target(s["norm_chunk"], int(flow.action_horizon))
                        for s in samples])
    actions = torch.as_tensor(np.stack(acts)).to(dev, dt)
    amask = torch.as_tensor(np.stack(masks)).to(dev, dt)
    cs = [s["c"] for s in samples]
    if wrong:                     # swap the two physics levels
        cs = [0.0 if abs(c) > 1e-9 else float(np.log(2.0)) for c in cs]
    if shuffled_c is not None:
        cs = list(shuffled_c)
    c = torch.tensor(cs, device=dev, dtype=torch.float32).unsqueeze(1)
    hz = torch.full((len(samples),), 20.0, device=dev)
    emb = torch.full((len(samples),), 25, device=dev, dtype=torch.long)
    return h_t, h_pred, h_vlm, attn, actions, amask, c, hz, emb


def main() -> None:
    audit = json.loads((OUT / "AUDIT0_FORWARD.json").read_text())
    if not audit["READY_TO_TRAIN"] or ARM not in {"control", "hidden"}:
        raise RuntimeError("Pre-training audit failed or adaptation arm is invalid")
    if audit["trainable"]["CONTROL"]["total_trainable"] != audit["trainable"]["HIDDEN"]["total_trainable"]:
        raise RuntimeError("Adaptation arms have different trainable parameter counts")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    vla, backend = hm.load_backend(use_bf16=True)
    backend.eval()
    flow = backend.flow
    hm.freeze_base(backend)
    lora_rep = hm.install_lora(flow, hm.LORA_RANK)
    torch.manual_seed(SEED)                        # identical init for both arms
    arm = hm.ContextArm(flow, control=(ARM == "control")).to(next(flow.parameters()).device)
    dev = next(flow.parameters()).device
    dt = next(flow.parameters()).dtype
    for m in flow.modules():
        if isinstance(m, hm.LoRALinear):
            m.A.to(torch.float32)
            m.B.to(torch.float32)
    arm.projector.to(torch.float32)

    params = list(arm.projector.parameters()) + hm.lora_parameters(flow)
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

    train, val = load_split("train"), load_split("val")
    mini = val[:8]                                  # frozen context-sensitivity mini-set
    rng = np.random.default_rng(SEED)
    log = OUT / "TRAIN_LOG.jsonl"
    best = {"val": float("inf"), "update": -1}
    t0 = time.time()

    def val_loss() -> float:
        arm.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), BATCH):
                b = val[i:i + BATCH]
                h_t, h_pred, h_vlm, attn, actions, amask, c, hz, emb = to_batch(b, flow, dev, dt)
                hv, am = arm.condition(h_vlm, attn, c)
                loss = hm.flow_loss(flow, h_t, h_pred, hv, am, actions, amask, hz, emb,
                                    flow_seed=12345 + i, repeat=1)
                tot += float(loss) * len(b)
                n += len(b)
        arm.train()
        return tot / max(n, 1)

    def context_sensitivity() -> float:
        """maxabs(action(correct) - action(wrong)) on the frozen mini-set. Diagnostic only."""
        arm.eval()  # DiT uses dropout in train mode; the paired action probe must be deterministic.
        with torch.no_grad():
            h_t, h_pred, h_vlm, attn, _, _, c, hz, emb = to_batch(mini, flow, dev, dt)
            _, _, _, _, _, _, c_w, _, _ = to_batch(mini, flow, dev, dt, wrong=True)
            noise = torch.randn(len(mini), int(flow.action_horizon), int(flow.config.action_dim),
                                generator=torch.Generator().manual_seed(7)).numpy()
            hv, am = arm.condition(h_vlm, attn, c)
            a1 = hm.flow_sample(flow, h_t, h_pred, hv, am, hz, emb, noise).float().cpu().numpy()
            hv, am = arm.condition(h_vlm, attn, c_w)
            a2 = hm.flow_sample(flow, h_t, h_pred, hv, am, hz, emb, noise).float().cpu().numpy()
        arm.train()
        return float(np.abs(a1[:, :, :7] - a2[:, :, :7]).max())

    arm.train()
    order = rng.permutation(len(train))
    pos = 0
    for update in range(MAX_UPDATES):
        if pos + BATCH > len(order):
            order, pos = rng.permutation(len(train)), 0
        batch = [train[i] for i in order[pos:pos + BATCH]]
        pos += BATCH
        h_t, h_pred, h_vlm, attn, actions, amask, c, hz, emb = to_batch(batch, flow, dev, dt)
        hv, am = arm.condition(h_vlm, attn, c)
        loss = hm.flow_loss(flow, h_t, h_pred, hv, am, actions, amask, hz, emb,
                            flow_seed=1000 + update, repeat=2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if update % 25 == 0 or update == MAX_UPDATES - 1:
            rec = {"arm": ARM, "update": update, "loss": float(loss), "grad_norm": float(gn),
                   "alpha": float(arm.projector.alpha.detach()), "sec": round(time.time() - t0, 1)}
            if update % VAL_EVERY == 0 or update == MAX_UPDATES - 1:
                rec["val_loss"] = val_loss()
                rec["context_sensitivity"] = context_sensitivity()
                if rec["val_loss"] < best["val"]:
                    best = {"val": rec["val_loss"], "update": update}
                    torch.save({"projector": arm.projector.state_dict(),
                                "lora": {n: p.detach().cpu() for n, p in flow.named_parameters()
                                         if ".A.weight" in n or ".B.weight" in n},
                                "arm": ARM, "update": update, "val_loss": rec["val_loss"],
                                "trainable_params": n_train, "lora_report": lora_rep},
                               OUT / f"adapter_{ARM}.pt")
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

    summary = {"arm": ARM, "trainable_params": n_train, "projector_params": sum(p.numel() for p in arm.projector.parameters()),
               "lora_params": sum(p.numel() for p in hm.lora_parameters(flow)),
               "n_train_samples": len(train), "n_val_samples": len(val),
               "best_val_loss": best["val"], "best_update": best["update"],
               "final_alpha": float(arm.projector.alpha.detach()),
               "seconds": round(time.time() - t0, 1)}
    (OUT / f"TRAIN_SUMMARY_{ARM}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
