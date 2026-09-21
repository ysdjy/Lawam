"""Model-side pieces for the HSU round (runs in the `lawam` env).

Everything here is additive and lives outside the repository's model code: the Flow Head already
accepts `h_vlm` and `attention_mask` as call arguments, so the oracle context token is appended at
the CALL SITE, and the LoRA is installed by wrapping existing `nn.Linear` modules in place. No
starVLA file is modified.

Layout of the flow conditioning, with the pre-registered context position:
    encoder_hidden_states = [ H_t (256) ; H_future (256) ; enc_vlm(h_vlm) (S) ; enc_vlm(e_c) (1) ]
AlternateVLDiT derives its `vlm_mask` from the length of `cond_vlm`, so the appended token lands in
the VLM branch that the text-attending cross-attention blocks read.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path("/home/zbh/Downloads/IsaacLab/Lawam_paper/LaWAM")
CKPT_DIR = REPO / "results/Checkpoints/libero/lawam_libero_sft_release"
CKPT = CKPT_DIR / "final_model" / "pytorch_model.pt"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

LORA_RANK = 4
CTX_HIDDEN = 128
MU_NOMINAL = 0.95


# ----------------------------------------------------------------------------- base model
def load_backend(use_bf16: bool = True):
    from deployment.model_server.server_policy import load_policy_from_checkpoint

    vla = load_policy_from_checkpoint(str(CKPT), use_bf16=use_bf16, device="cuda")
    return vla, vla.policy_backend


def state_norm_stats() -> dict:
    return json.loads((CKPT_DIR / "dataset_statistics.json").read_text())["franka"]["state"]


def normalize_state(state: np.ndarray, stats: dict) -> np.ndarray:
    """Identical formula to ModelClient.normalize_state (the eval client applies it before sending)."""
    s = np.asarray(state, np.float32)
    if s.ndim == 1:
        s = s[None, :]
    hi = np.asarray(stats["max"], np.float32)
    lo = np.asarray(stats["min"], np.float32)
    s = s[..., : hi.shape[0]]
    out = s.copy()
    den = hi - lo
    ok = np.abs(den) > 1e-12
    out[..., ok] = (out[..., ok] - lo[ok]) / den[ok] * 2.0 - 1.0
    out[..., ~ok] = 0.0
    return np.clip(out, -1.0, 1.0)


def sha_module(m) -> str:
    h = hashlib.sha256()
    sd = m.state_dict() if hasattr(m, "state_dict") else {"": m}
    for k in sorted(sd):
        t = sd[k]
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- context projector
class ContextTokenProjector(nn.Module):
    """c -> e_c in R^2048, gated by a learnable scalar.

    Zero-init note (amendment HSU-A4): the protocol first asked for alpha = 0 AND a zero second
    Linear. That configuration is DEAD -- with both zero, d e_c/d alpha = 0 and d e_c/d W2 = 0, so
    nothing can ever train. Only `alpha` is zero-initialised here: the pathway still contributes
    EXACTLY zero at initialisation (so the zero-init equivalence check is exact), while the
    gradient w.r.t. alpha is non-zero. Identical in both arms.
    """

    def __init__(self, dim: int = 2048, hidden: int = CTX_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, c: torch.Tensor) -> torch.Tensor:      # c: [B, 1] -> [B, dim]
        c = c.to(dtype=self.net[0].weight.dtype)
        return self.alpha * self.net(c)


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """base(x) + B(A(x)), with B zero-initialised so the residual starts at exactly zero."""

    def __init__(self, base: nn.Linear, rank: int = LORA_RANK):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / rank)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.B(self.A(x.to(self.A.weight.dtype)))
        return out + lora.to(out.dtype)


CROSS_ATTENTION_TARGETS = ("to_q", "to_k", "to_v", "to_out")


def install_lora(flow, rank: int = LORA_RANK) -> dict:
    """Wrap q/k/v/out of the cross-attending DiT blocks only.

    AlternateVLDiT alternates: even block index = cross-attention over
    [H_t ; H_future ; cond_vlm ; e_c]; odd index = self-attention over the action tokens. Only the
    even blocks can read the context token, so only they are adapted. The action encoder/decoder,
    the FFNs, the time encoder and the output projections are untouched.
    """
    report = {"rank": rank, "wrapped": [], "params": 0}
    blocks = flow.DiT.transformer_blocks
    for idx, block in enumerate(blocks):
        if idx % 2 != 0:
            continue                                   # self-attention block: not a context reader
        attn = block.attn1
        for name in CROSS_ATTENTION_TARGETS:
            if name == "to_out":
                base = attn.to_out[0]
                lora = LoRALinear(base, rank)
                attn.to_out[0] = lora
            else:
                base = getattr(attn, name)
                lora = LoRALinear(base, rank)
                setattr(attn, name, lora)
            report["wrapped"].append(f"DiT.transformer_blocks.{idx}.attn1.{name}")
            report["params"] += sum(p.numel() for p in (lora.A.weight, lora.B.weight))
    report["n_modules"] = len(report["wrapped"])
    report["n_cross_blocks"] = len(blocks) // 2 + len(blocks) % 2
    report["n_blocks"] = len(blocks)
    return report


def lora_parameters(flow):
    return [p for m in flow.modules() if isinstance(m, LoRALinear) for p in (m.A.weight, m.B.weight)]


# ----------------------------------------------------------------------------- conditioning
def append_context(h_vlm: torch.Tensor, attention_mask: torch.Tensor, e_c: torch.Tensor):
    """[H_t ; H_future ; h_vlm ; e_c] -- the pre-registered position."""
    h = torch.cat([h_vlm, e_c.unsqueeze(1).to(h_vlm.dtype)], dim=1)
    m = torch.cat([attention_mask,
                   torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype,
                              device=attention_mask.device)], dim=1)
    return h, m


class ContextArm(nn.Module):
    """One adaptation arm: a context projector plus the Flow-Head LoRA (installed on the shared
    frozen flow head). `control=True` forces the scalar input to 0 for every sample."""

    def __init__(self, flow, control: bool, dim: int = 2048):
        super().__init__()
        self.flow, self.control = flow, control
        self.projector = ContextTokenProjector(dim)

    def context_token(self, c: torch.Tensor) -> torch.Tensor:
        if self.control:
            c = torch.zeros_like(c)
        return self.projector(c)

    def condition(self, h_vlm, attention_mask, c):
        return append_context(h_vlm, attention_mask, self.context_token(c))


def freeze_base(backend) -> None:
    for p in backend.parameters():
        p.requires_grad_(False)


def trainable_report(arm: ContextArm) -> dict:
    proj = sum(p.numel() for p in arm.projector.parameters())
    lora = sum(p.numel() for p in lora_parameters(arm.flow))
    return {"projector_params": proj, "lora_params": lora, "total_trainable": proj + lora}


# ----------------------------------------------------------------------------- flow calls
def flow_loss(flow, h_t, h_pred, h_vlm, attn, actions, actions_mask, action_hz, embodiment_id,
              *, flow_seed: int, repeat: int = 1):
    """The original LaWAM flow-matching loss, with `use_state=False` zeros as the model expects."""
    from starVLA.model.framework.vlas.lawam import _cuda_autocast

    def rep(t):
        return t if repeat == 1 else t.repeat(repeat, *([1] * (t.ndim - 1)))

    h_t, h_pred, h_vlm, attn = rep(h_t), rep(h_pred), rep(h_vlm), rep(attn)
    actions, actions_mask = rep(actions), rep(actions_mask)
    action_hz, embodiment_id = action_hz.repeat(repeat), embodiment_id.repeat(repeat)
    b = h_t.shape[0]
    state = torch.zeros(b, int(flow.config.state_dim), device=h_t.device, dtype=torch.float32)
    torch.manual_seed(int(flow_seed))
    with _cuda_autocast(torch.float32):
        return flow(h_t=h_t, h_t1_star=h_pred, h_vlm=h_vlm, state=state, actions=actions,
                    action_hz=action_hz, embodiment_id=embodiment_id,
                    state_mask=torch.zeros_like(state, dtype=torch.bool),
                    actions_mask=actions_mask, attention_mask=attn)


@torch.inference_mode()
def flow_sample(flow, h_t, h_pred, h_vlm, attn, action_hz, embodiment_id, initial_noise):
    from starVLA.model.framework.vlas.lawam import _cuda_autocast

    b = h_t.shape[0]
    state = torch.zeros(b, int(flow.config.state_dim), device=h_t.device, dtype=torch.float32)
    with _cuda_autocast(torch.float32):
        return flow.sample_actions_cfg(
            h_t=h_t, h_t1_star=h_pred, h_vlm=h_vlm, state=state,
            state_mask=torch.zeros_like(state, dtype=torch.bool),
            action_hz=action_hz, embodiment_id=embodiment_id,
            cfg_scale=float(flow.config.cfg_guidance_scale),
            num_inference_steps=int(flow.config.num_inference_steps),
            attention_mask=attn, return_padded=False,
            initial_noise=torch.as_tensor(initial_noise).to(h_t.device))


def build_actions_target(norm_chunk: np.ndarray, action_horizon: int, action_dim: int = 32,
                         n_valid_dims: int = 7):
    """[T,32] teacher chunk -> padded [action_horizon, 32] actions and mask.

    The head checks that the number of valid action tokens equals floor(horizon_sec * hz) = 8, so
    exactly the first 8 tokens are marked valid, on the 7 embodiment dims.
    """
    a = np.zeros((action_horizon, action_dim), np.float32)
    m = np.zeros((action_horizon, action_dim), np.float32)
    t = int(norm_chunk.shape[0])
    a[:t] = norm_chunk
    m[:t, :n_valid_dims] = 1.0
    return a, m
