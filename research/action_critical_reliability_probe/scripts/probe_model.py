"""Shared-encoding + batched flow-head replay (model side, `lawam` env).

Why this exists: a token-level sensitivity sweep needs thousands of action samplings per state, but the
perturbation only touches `h_t1_star`, which enters *after* the VLM and the LAM decoder. So the shared
encoding (VLM -> z -> LaWM -> H_pred, plus h_vlm / h_t) is computed once per state and only
`ConditionalFlowMatchingHead.sample_actions_cfg` is re-run, batched over perturbations.

This adds NO model code: it re-uses `backend._run_shared_encoding_infer` and `backend.flow.sample_actions_cfg`
exactly as `LatentWorldPolicyBackend.predict_action` calls them. `m1_equivalence.py` verifies that the
resulting actions equal the ones from the untouched `vla.predict_action` path.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402  (puts branch_diagnostic/scripts on sys.path)
import model_common as mc  # noqa: E402

from starVLA.model.framework.vlas.lawam import _cuda_autocast  # noqa: E402

FLOW_STAGE_DTYPE = torch.float32  # identical to predict_action


@dataclass
class StateContext:
    """Everything the flow head needs for one fixed observation."""
    batch: dict
    h_t: torch.Tensor          # [1, 256, 768]
    h_vlm: torch.Tensor        # [1, S, 2048]
    h_t1_pred: torch.Tensor    # [1, 256, 768]  == H_pred
    z: torch.Tensor            # [1, 1, 32]
    attn_flow: torch.Tensor    # [1, S] bool
    guidance_scale: float
    num_inference_steps: int

    @property
    def n_tokens(self) -> int:
        return int(self.h_t1_pred.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.h_t1_pred.shape[2])


class ProbeRunner:
    def __init__(self, vla) -> None:
        self.vla = vla
        self.backend = vla.policy_backend
        self.flow = self.backend.flow
        self.builder = vla.policy_runner.infer_batch_builder

    # ------------------------------------------------------------------ shared encoding (once per state)
    @torch.inference_mode()
    def prepare(self, example: dict[str, Any]) -> StateContext:
        batch = self.builder.build_infer_batch([example])
        shared = self.backend._run_shared_encoding_infer(
            prepared_batch=batch,
            source="acr_probe.ProbeRunner.prepare",
            lam_features_with_no_grad=False,
        )
        return StateContext(
            batch=batch,
            h_t=shared.h_t,
            h_vlm=shared.h_vlm,
            h_t1_pred=shared.h_t1_pred,
            z=shared.pred_action_emb,
            attn_flow=(batch["attention_mask"] == 1),
            guidance_scale=float(self.flow.config.cfg_guidance_scale),
            num_inference_steps=int(self.flow.config.num_inference_steps),
        )

    # ------------------------------------------------------------------ batched flow sampling
    @torch.inference_mode()
    def actions(
        self,
        ctx: StateContext,
        future: torch.Tensor,
        noise: torch.Tensor,
    ) -> np.ndarray:
        """future: [B, 256, 768]; noise: [B, action_horizon, action_dim]. Returns [B, 8, 32] float32 numpy."""
        b = int(future.shape[0])
        if int(noise.shape[0]) != b:
            raise ValueError(f"noise batch {tuple(noise.shape)} does not match future batch {b}.")
        future = future.to(device=ctx.h_t1_pred.device, dtype=ctx.h_t1_pred.dtype)

        def rep(t: torch.Tensor) -> torch.Tensor:
            return t if int(t.shape[0]) == b else t.expand(b, *t.shape[1:])

        with _cuda_autocast(FLOW_STAGE_DTYPE):
            actions = self.flow.sample_actions_cfg(
                h_t=rep(ctx.h_t),
                h_t1_star=future,
                h_vlm=rep(ctx.h_vlm),
                state=rep(ctx.batch["state"]),
                state_mask=rep(ctx.batch["state_mask"]),
                action_hz=rep(ctx.batch["action_hz"]),
                embodiment_id=rep(ctx.batch["embodiment_id"]),
                cfg_scale=ctx.guidance_scale,
                num_inference_steps=ctx.num_inference_steps,
                attention_mask=rep(ctx.attn_flow),
                return_padded=False,
                initial_noise=noise,
            )
        return actions.detach().float().cpu().numpy()

    def cast_future(self, ctx: StateContext, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(arr) if not torch.is_tensor(arr) else arr
        return t.to(device=ctx.h_t1_pred.device, dtype=ctx.h_t1_pred.dtype)


# ---------------------------------------------------------------------------- action distance metrics
TRANS = slice(0, 3)
ROT = slice(3, 6)
GRIP = 6
MM_PER_NORMALIZED_UNIT_PER_STEP = 46.875  # (max-min)/2 = 0.9375 (normalized->raw) * 0.05 m/unit * 1000


def action_distance(a: np.ndarray, a0: np.ndarray) -> dict[str, float]:
    """Distances between two [T, >=7] normalized action chunks (only the 7 embodiment dims are used)."""
    d = np.asarray(a, dtype=np.float64)[:, :7] - np.asarray(a0, dtype=np.float64)[:, :7]
    per_axis_mm = np.abs(d[:, TRANS]).sum(axis=0) * MM_PER_NORMALIZED_UNIT_PER_STEP
    return {
        "l2_norm_all7": float(np.sqrt((d ** 2).mean())),
        "maxabs_all7": float(np.abs(d).max()),
        "translation_l2": float(np.sqrt((d[:, TRANS] ** 2).mean())),
        "rotation_l2": float(np.sqrt((d[:, ROT] ** 2).mean())),
        "gripper_absdiff": float(np.abs(d[:, GRIP]).max()),
        "translation_mm_equiv": float(np.linalg.norm(per_axis_mm)),
    }


DISTANCE_KEYS = ["l2_norm_all7", "maxabs_all7", "translation_l2", "rotation_l2", "gripper_absdiff",
                 "translation_mm_equiv"]


def action_distance_batch(a: np.ndarray, a0: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised `action_distance` for a batch [B, T, >=7] against a single [T, >=7] baseline."""
    d = np.asarray(a, dtype=np.float64)[:, :, :7] - np.asarray(a0, dtype=np.float64)[None, :, :7]
    per_axis_mm = np.abs(d[:, :, TRANS]).sum(axis=1) * MM_PER_NORMALIZED_UNIT_PER_STEP
    return {
        "l2_norm_all7": np.sqrt((d ** 2).mean(axis=(1, 2))),
        "maxabs_all7": np.abs(d).max(axis=(1, 2)),
        "translation_l2": np.sqrt((d[:, :, TRANS] ** 2).mean(axis=(1, 2))),
        "rotation_l2": np.sqrt((d[:, :, ROT] ** 2).mean(axis=(1, 2))),
        "gripper_absdiff": np.abs(d[:, :, GRIP]).max(axis=1),
        "translation_mm_equiv": np.linalg.norm(per_axis_mm, axis=1),
    }


def unit_directions(state_id: str, token: int, n_dirs: int, dim: int) -> np.ndarray:
    """Deterministic unit directions: seed derived from (state_id, token) only, so the k-th direction of a
    token is the same in every run and for every epsilon."""
    import hashlib

    digest = hashlib.sha256(f"{state_id}|{int(token)}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big")  # stable across processes (unlike builtin hash on str)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n_dirs, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_noise(seed: int, horizon: int, dim: int, batch: int = 1) -> np.ndarray:
    return mc.make_initial_noise(seed, horizon, dim, batch=batch)


# ---------------------------------------------------------------------------- attention baseline
class _RecordingAttnProcessor:
    """Replica of diffusers `AttnProcessor2_0` that materialises the attention weights.

    Used only inside `record_attention(...)`, which swaps it in, runs one forward and swaps it back out.
    The model files are untouched; `m4_attention.py` checks that the actions produced with the recorder
    installed match the default SDPA path.
    """

    def __init__(self, store: list) -> None:
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        residual = hidden_states
        is_cross = encoder_hidden_states is not None
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
        if attention_mask is not None:
            # `Attention.prepare_attention_mask` does NOT convert a boolean mask into an additive one; SDPA
            # interprets bool as keep/drop. Adding it numerically (as an earlier version of this recorder
            # did) silently lets the block attend to the masked-out half of the condition sequence.
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask.float()
        probs = torch.softmax(scores, dim=-1)                      # [B, heads, q_len, kv_len]
        if is_cross:
            # average over heads and query positions -> one weight per condition token
            self.store.append(probs.mean(dim=(1, 2)).detach().cpu())
        hidden_states = torch.matmul(probs.to(value.dtype), value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class record_attention:
    """Context manager: record cross-attention weights of the image-reading DiT blocks.

    `blocks` defaults to the blocks that attend to the image half (h_t + H_pred); with
    `attend_text_every_n_blocks=2` and 16 layers those are 2, 6, 10, 14.
    """

    def __init__(self, flow_head, blocks: list[int] | None = None) -> None:
        self.flow = flow_head
        cfg = flow_head.config
        every = int(cfg.attend_text_every_n_blocks)
        self.blocks = blocks if blocks is not None else [
            i for i in range(int(cfg.num_layers)) if i % 2 == 0 and i % (2 * every) != 0
        ]
        self.store: list = []
        self._saved: dict = {}

    def __enter__(self) -> "record_attention":
        for idx in self.blocks:
            attn = self.flow.DiT.transformer_blocks[idx].attn1
            self._saved[idx] = attn.processor
            attn.set_processor(_RecordingAttnProcessor(self.store))
        return self

    def __exit__(self, *exc) -> None:
        for idx, proc in self._saved.items():
            self.flow.DiT.transformer_blocks[idx].attn1.set_processor(proc)
        self._saved.clear()

    def stacked(self) -> np.ndarray:
        """[n_calls, B, kv_len] -> numpy; n_calls = len(blocks) * num_inference_steps."""
        return torch.stack(self.store, dim=0).float().numpy()
