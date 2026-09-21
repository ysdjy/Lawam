"""HSU AUDIT 0 / 2 / forward audit — must pass BEFORE any training step.

  AUDIT 0  base checkpoint hash, module parameter counts, and the trainable-parameter budget of the
           two arms (they must match EXACTLY).
  FORWARD  with H_t, H_future, h_vlm, the flow noise and the timestep all held fixed, changing ONLY
           the context scalar must change the K/V tensors of the context token inside the adapted
           cross-attention layers, and the token must not be masked away.
  AUDIT 2  zero-init equivalence: Original vs Control@init vs Hidden@init produce the same action
           for the same pinned noise.

Usage: python h3_audit.py <run_id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsu_model as hm  # noqa: E402

RUN = sys.argv[1]
OUT = hm.REPO / "results" / "hidden_state_utilization_probe" / RUN
C_HIGH = float(np.log(1.9 / 0.95))


def main() -> None:
    rep = {"run_id": RUN, "checkpoint": str(hm.CKPT)}
    vla, backend = hm.load_backend(use_bf16=True)
    backend.eval()
    flow = backend.flow

    # ---------------------------------------------------------------- AUDIT 0
    rep["audit0"] = {
        "state_dict_sha256_backend": hm.sha_module(backend),
        "params_total": sum(p.numel() for p in backend.parameters()),
        "params_flow": sum(p.numel() for p in flow.parameters()),
        "params_vlm": sum(p.numel() for p in backend.vlm.parameters()) if hasattr(backend, "vlm") else None,
        "dtype": str(next(flow.parameters()).dtype),
        "n_dit_blocks": len(flow.DiT.transformer_blocks),
        "cfg_guidance_scale": float(flow.config.cfg_guidance_scale),
        "num_inference_steps": int(flow.config.num_inference_steps),
        "action_horizon": int(flow.action_horizon),
        "repeated_diffusion_steps": int(backend.model_cfg.repeated_diffusion_steps),
    }

    # take one cached sample for the audits
    f = sorted((OUT / "cache").glob("HIGH_*.npz"))[0]
    d = np.load(f)
    dev = next(flow.parameters()).device
    dt = next(flow.parameters()).dtype
    h_t = torch.as_tensor(d["h_t"][0:1]).to(dev, dt)
    h_pred = torch.as_tensor(d["h_pred"][0:1]).to(dev, dt)
    h_vlm = torch.as_tensor(d["h_vlm"][0:1]).to(dev, dt)
    attn = torch.as_tensor(d["attn"][0:1]).to(dev)
    hz = torch.tensor([20.0], device=dev)
    emb = torch.tensor([25], device=dev)
    noise = np.asarray(torch.randn(1, int(flow.action_horizon), int(flow.config.action_dim),
                                   generator=torch.Generator().manual_seed(0)).numpy(), np.float32)

    # ORIGINAL action (no context token, no LoRA installed yet)
    a_orig = hm.flow_sample(flow, h_t, h_pred, h_vlm, attn, hz, emb, noise).float().cpu().numpy()

    # ---------------------------------------------------------------- install LoRA + arms
    hm.freeze_base(backend)
    lora_rep = hm.install_lora(flow, hm.LORA_RANK)
    arm_h = hm.ContextArm(flow, control=False).to(dev)
    arm_c = hm.ContextArm(flow, control=True).to(dev)
    rep["lora"] = {k: v for k, v in lora_rep.items() if k != "wrapped"}
    rep["lora"]["example_modules"] = lora_rep["wrapped"][:6]
    rep["trainable"] = {"HIDDEN": hm.trainable_report(arm_h), "CONTROL": hm.trainable_report(arm_c)}
    rep["trainable"]["params_match"] = (rep["trainable"]["HIDDEN"]["total_trainable"]
                                        == rep["trainable"]["CONTROL"]["total_trainable"])
    rep["trainable"]["under_5M"] = rep["trainable"]["HIDDEN"]["total_trainable"] < 5_000_000
    rep["trainable"]["base_frozen_params"] = sum(p.numel() for p in backend.parameters() if not p.requires_grad)
    rep["trainable"]["base_trainable_leak"] = [n for n, p in backend.named_parameters()
                                               if p.requires_grad and ".A." not in n and ".B." not in n][:5]

    # ---------------------------------------------------------------- FORWARD AUDIT
    # capture the K/V of the LAST encoder token (= the context token) in an adapted cross block
    captured = {}

    def hook(name):
        def fn(mod, inp, out):
            captured[name] = out.detach().float().cpu().numpy()
        return fn

    blk = flow.DiT.transformer_blocks[0].attn1
    handles = [blk.to_k.register_forward_hook(hook("k")), blk.to_v.register_forward_hook(hook("v"))]

    def run(arm, c_value):
        c = torch.tensor([[c_value]], device=dev, dtype=torch.float32)
        hv, am = arm.condition(h_vlm, attn, c)
        act = hm.flow_sample(flow, h_t, h_pred, hv, am, hz, emb, noise)
        return act.float().cpu().numpy(), {k: v.copy() for k, v in captured.items()}, hv, am

    # with alpha = 0 the token is present but identical; temporarily open the gate to prove the
    # token is READ by the adapted layer (the gate is restored to 0 immediately afterwards).
    with torch.no_grad():
        arm_h.projector.alpha.fill_(1.0)
    a_lo, cap_lo, hv_lo, am_lo = run(arm_h, 0.0)
    a_hi, cap_hi, cap_hv, cap_am = run(arm_h, C_HIGH)
    with torch.no_grad():
        arm_h.projector.alpha.fill_(0.0)
    for h in handles:
        h.remove()

    n_enc = int(h_t.shape[1] + h_pred.shape[1] + hv_lo.shape[1])
    ctx_pos = n_enc - 1
    rep["forward_audit"] = {
        "encoder_tokens": n_enc,
        "context_token_index": ctx_pos,
        "context_token_in_attention_mask": bool(cap_am[0, -1].item()),
        "vlm_block_len_with_context": int(hv_lo.shape[1]),
        "k_delta_at_context_token": float(np.abs(cap_hi["k"][0, ctx_pos] - cap_lo["k"][0, ctx_pos]).max()),
        "v_delta_at_context_token": float(np.abs(cap_hi["v"][0, ctx_pos] - cap_lo["v"][0, ctx_pos]).max()),
        "k_delta_at_other_tokens": float(np.abs(cap_hi["k"][0, :ctx_pos] - cap_lo["k"][0, :ctx_pos]).max()),
        "action_delta_gate_open": float(np.abs(a_hi[:, :, :7] - a_lo[:, :, :7]).max()),
        "adapted_layer_is_lora": isinstance(blk.to_k, hm.LoRALinear),
    }
    fa = rep["forward_audit"]
    rep["forward_audit"]["PASS"] = bool(fa["context_token_in_attention_mask"]
                                        and fa["k_delta_at_context_token"] > 0
                                        and fa["v_delta_at_context_token"] > 0
                                        and fa["k_delta_at_other_tokens"] == 0
                                        and fa["adapted_layer_is_lora"])

    # ---------------------------------------------------------------- AUDIT 2 zero-init equivalence
    a_h0, _, _, _ = run(arm_h, C_HIGH)          # alpha back to 0, LoRA B = 0
    a_c0, _, _, _ = run(arm_c, C_HIGH)          # control forces c = 0 as well
    # the flow sampler's own nuisance on the SAME state: a second noise draw
    noise2 = np.asarray(torch.randn(1, int(flow.action_horizon), int(flow.config.action_dim),
                                    generator=torch.Generator().manual_seed(1)).numpy(), np.float32)
    a_orig2 = hm.flow_sample(flow, h_t, h_pred, h_vlm, attn, hz, emb, noise2).float().cpu().numpy()
    sampler_nuisance = float(np.abs(a_orig2[:, :, :7] - a_orig[:, :, :7]).max())

    rep["audit2_zero_init"] = {
        "maxabs_hidden_vs_original": float(np.abs(a_h0[:, :, :7] - a_orig[:, :, :7]).max()),
        "maxabs_control_vs_original": float(np.abs(a_c0[:, :, :7] - a_orig[:, :, :7]).max()),
        "maxabs_hidden_vs_control": float(np.abs(a_h0[:, :, :7] - a_c0[:, :, :7]).max()),
        "flow_sampler_nuisance_same_state": sampler_nuisance,
        "note": ("same pinned flow noise; the context token is present but gated to exactly zero. "
                 "enc_vlm carries a BIAS, so even a zero-valued extra token contributes enc_vlm(0)=b "
                 "and slightly redistributes attention over the VLM block -- adding a token is never "
                 "bitwise free. Amendment HSU-A5 criterion: the two arms must be identical at init, "
                 "and the deviation from ORIGINAL must be <= 10% of the flow-sampler nuisance."),
    }
    z = rep["audit2_zero_init"]
    rep["audit2_zero_init"]["arms_identical_at_init"] = bool(z["maxabs_hidden_vs_control"] == 0.0)
    rep["audit2_zero_init"]["deviation_vs_nuisance_ratio"] = (
        max(z["maxabs_hidden_vs_original"], z["maxabs_control_vs_original"]) / max(sampler_nuisance, 1e-12))
    rep["audit2_zero_init"]["PASS_literal_1e-5"] = bool(
        max(z["maxabs_hidden_vs_original"], z["maxabs_control_vs_original"]) <= 1e-5)
    rep["audit2_zero_init"]["PASS"] = bool(rep["audit2_zero_init"]["arms_identical_at_init"]
                                           and rep["audit2_zero_init"]["deviation_vs_nuisance_ratio"] <= 0.10)

    rep["READY_TO_TRAIN"] = bool(rep["forward_audit"]["PASS"] and rep["audit2_zero_init"]["PASS"]
                                 and rep["trainable"]["params_match"] and rep["trainable"]["under_5M"])
    (OUT / "AUDIT0_FORWARD.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
