"""FE Stage B — the flow stage, replicating `LaWAM.forward()`'s flow block exactly.

ONE implementation is used by training, validation and the 2x2 evaluation, so the three can
never drift apart. It reproduces the official ordering: repeat by `repeated_diffusion_steps`
FIRST, then draw the per-row future-condition mask over the REPEATED batch, exactly as
`_build_flow_future_condition` does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common as fc  # noqa: E402

from starVLA.model.framework.vlas.lawam import _cuda_autocast  # noqa: E402

FLOW_STAGE_DTYPE = torch.float32  # identical to LaWAM.forward()


def _rep(t: torch.Tensor, n: int) -> torch.Tensor:
    return t.repeat(n, *([1] * (t.ndim - 1)))


def build_future_condition(
    h_t1_pred: torch.Tensor,
    h_t1_gt: torch.Tensor,
    p_pred: float,
    mask_gen: torch.Generator | None,
    forced_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cond_future, pred_mask). Mirrors `_build_flow_future_condition` with
    detach_future_feature=True, which is the release setting for both arms.

    The mask draw happens even when p_pred == 1.0 so the Control arm consumes the mask
    generator identically; the generator is separate from the flow RNG stream, so a draw in one
    arm can never shift the other arm's flow time or noise.
    """
    bsz = h_t1_pred.shape[0]
    if forced_mask is not None:
        pred_mask = forced_mask.view(bsz, 1, 1).to(h_t1_pred.device)
        if mask_gen is not None:
            torch.rand(bsz, 1, 1, generator=mask_gen, device=mask_gen.device)  # keep streams aligned
    else:
        u = torch.rand(bsz, 1, 1, generator=mask_gen, device=mask_gen.device)
        pred_mask = (u < float(p_pred)).to(h_t1_pred.device)
    cond = torch.where(pred_mask, h_t1_pred, h_t1_gt)
    return cond.detach(), pred_mask.view(bsz)


def flow_loss(
    backend,
    cb: dict,
    *,
    p_pred: float,
    flow_seed: int,
    mask_gen: torch.Generator | None = None,
    forced_mask: torch.Tensor | None = None,
    h_t1_star_override: str | None = None,
    repeat_steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow loss on one cached batch. Returns (loss, pred_mask_over_repeated_batch).

    `h_t1_star_override` in {"pred", "gt"} bypasses the sampling entirely and is what the 2x2
    evaluation uses: the eval-mode selector in the model always returns H_pred, so the four
    cells can only be produced by passing `h_t1_star` explicitly.

    `flow_seed` pins the Flow Head's INTERNAL draws (`sample_noise`, `sample_time`), which take
    no generator argument. Seeding immediately before the call gives row r of a given step the
    same (tau, epsilon) in both arms and in every cell of the 2x2.
    """
    n = int(backend.model_cfg.repeated_diffusion_steps) if repeat_steps is None else int(repeat_steps)

    h_t = _rep(cb["h_t"], n)
    h_t1_pred = _rep(cb["h_t1_pred"], n)
    h_t1_gt = _rep(cb["h_t1_gt"], n)
    h_vlm = _rep(cb["h_vlm"], n)
    actions = _rep(cb["actions"], n)
    actions_mask = _rep(cb["actions_mask"], n)
    attn = _rep(cb["attention_mask"], n)
    action_hz = cb["action_hz"].repeat(n)
    embodiment = cb["embodiment_id"].repeat(n)
    bsz = h_t.shape[0]

    if h_t1_star_override is not None:
        cond = (h_t1_pred if h_t1_star_override == "pred" else h_t1_gt).detach()
        pred_mask = torch.full((bsz,), h_t1_star_override == "pred", dtype=torch.bool, device=h_t.device)
    else:
        cond, pred_mask = build_future_condition(h_t1_pred, h_t1_gt, p_pred, mask_gen, forced_mask)

    # `use_state=False`, so the state tensors are unused by the head; pass correctly-shaped zeros.
    state = torch.zeros(bsz, int(backend.flow.config.state_dim), device=h_t.device, dtype=torch.float32)
    state_mask = torch.zeros_like(state, dtype=torch.bool)

    torch.manual_seed(int(flow_seed))  # seeds CPU and CUDA: covers sample_noise and beta_dist
    with _cuda_autocast(FLOW_STAGE_DTYPE):
        loss = backend.flow(
            h_t=h_t, h_t1_star=cond, h_vlm=h_vlm,
            state=state, actions=actions, action_hz=action_hz,
            embodiment_id=embodiment, state_mask=state_mask,
            actions_mask=actions_mask, attention_mask=attn,
        )
    return loss, pred_mask


def deterministic_mask(addresses: list[tuple[int, int]], p_pred: float, n_repeat: int,
                       device: str = "cuda") -> torch.Tensor:
    """Frozen validation mask: reproducible from the address alone, identical across arms,
    seeds and validation points. Shape [len(addresses) * n_repeat]."""
    vals = []
    for r in range(n_repeat):
        for ep, st in addresses:
            u = int(fc.sha256_text("FE2026-valmask", str(ep), str(st), str(r))[:8], 16) / 0xFFFFFFFF
            vals.append(u < float(p_pred))
    return torch.tensor(vals, dtype=torch.bool, device=device)
