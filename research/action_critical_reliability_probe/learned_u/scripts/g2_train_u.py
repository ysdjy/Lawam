"""Step 4: train the minimal shared-token U predictor (torch, CPU/GPU; LaWAM is never loaded).

Architecture (fixed in PROTOCOL_G2.yaml, v1, no escalation without a new audit entry):
    x_j = [h_t_j, H_pred_j, H_pred_j - h_t_j, z]   (2336 dims, no token index, no position)
    Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> GELU -> Linear -> scalar
Target  log(U_l2 + 1e-6).  Loss  Huber.  Optimiser AdamW.  Early stopping on validation Spearman.
Shared weights across all 256 tokens, applied independently => permutation-equivariant BY CONSTRUCTION
(the property is nevertheless measured, to catch an implementation mistake rather than to trust the design).

LaWAM is not imported here and no LaWAM parameter exists in this process; the checkpoint file's
size/mtime are recorded before and after training as evidence it was never written.

Outputs: <run_dir>/models/u_mlp_seed<k>.pt, <run_dir>/TRAIN_LOG.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_dataset as gd  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
CKPT = REPO / "results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
SEEDS = [0, 1, 2]
HIDDEN = 512
MAX_EPOCHS = 200
PATIENCE = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_STATES = 8          # batching is by STATE, so tokens of one state never split across batches


class SharedTokenMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [..., in_dim] -> [...]
        return self.net(x).squeeze(-1)


def stack_split(states: list[gd.StateData]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([s.features() for s in states], axis=0)              # [S, 256, 2336]
    y = np.stack([np.log(s.U_l2 + gd.LABEL_EPS) for s in states], 0)  # [S, 256]
    return X, y.astype(np.float32)


def evaluate(model: nn.Module, X: torch.Tensor, states: list[gd.StateData]) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
    return float(np.mean([gd.spearman(pred[i], states[i].U_l2) for i in range(len(states))]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    (run_dir / "models").mkdir(exist_ok=True)
    log_path = run_dir / "TRAIN_LOG.jsonl"

    ckpt_before = {"size": CKPT.stat().st_size, "mtime": CKPT.stat().st_mtime}
    states = gd.load_states(run_dir)
    tr = [s for s in states if s.split == "train"]
    va = [s for s in states if s.split == "val"]
    print(f"train {len(tr)} states / {len(tr)*256} tokens; val {len(va)} states")

    Xtr, ytr = stack_split(tr)
    Xva, yva = stack_split(va)
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    np.savez(run_dir / "models" / "feature_norm.npz", mu=mu, sd=sd)
    Xtr_t = torch.tensor((Xtr - mu) / sd, device=args.device)
    Xva_t = torch.tensor((Xva - mu) / sd, device=args.device)
    ytr_t = torch.tensor(ytr, device=args.device)

    loss_fn = nn.HuberLoss(delta=1.0)
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SharedTokenMLP(Xtr.shape[-1]).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best, best_epoch, best_state, bad = -np.inf, -1, None, 0
        t0 = time.time()
        for epoch in range(MAX_EPOCHS):
            model.train()
            order = torch.randperm(Xtr_t.shape[0], device=args.device)
            tot = 0.0
            for i in range(0, len(order), BATCH_STATES):
                idx = order[i:i + BATCH_STATES]
                opt.zero_grad()
                out = model(Xtr_t[idx])
                loss = loss_fn(out, ytr_t[idx])
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            val_sp = evaluate(model, Xva_t, va)
            train_sp = evaluate(model, Xtr_t, tr)
            with open(log_path, "a") as f:
                f.write(json.dumps({"seed": seed, "epoch": epoch, "train_huber": tot / len(order),
                                    "train_spearman": train_sp, "val_spearman": val_sp}) + "\n")
            if val_sp > best + 1e-5:
                best, best_epoch, bad = val_sp, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
            if epoch % 20 == 0:
                print(f"  seed{seed} ep{epoch:3d} huber={tot/len(order):.4f} "
                      f"train_sp={train_sp:.4f} val_sp={val_sp:.4f}", flush=True)
        torch.save({"state_dict": best_state, "in_dim": Xtr.shape[-1], "hidden": HIDDEN,
                    "best_val_spearman": best, "best_epoch": best_epoch, "seed": seed,
                    "n_params": n_params},
                   run_dir / "models" / f"u_mlp_seed{seed}.pt")
        print(f"seed{seed}: best val Spearman {best:.4f} at epoch {best_epoch} "
              f"({n_params/1e6:.2f} M params, {time.time()-t0:.0f}s)", flush=True)

    ckpt_after = {"size": CKPT.stat().st_size, "mtime": CKPT.stat().st_mtime}
    with open(log_path, "a") as f:
        f.write(json.dumps({"event": "lawam_checkpoint_untouched",
                            "before": ckpt_before, "after": ckpt_after,
                            "identical": ckpt_before == ckpt_after,
                            "note": "LaWAM is never imported or loaded by this script"}) + "\n")
    print("LaWAM checkpoint untouched:", ckpt_before == ckpt_after)


if __name__ == "__main__":
    main()
