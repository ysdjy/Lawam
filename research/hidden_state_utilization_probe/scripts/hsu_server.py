"""Context-aware inference server for the closed-loop pilot (runs in the `lawam` env).

Speaks the SAME websocket protocol as deployment.model_server.server_policy, so the ordinary
LIBERO `ModelClient` drives it unchanged. Two extra optional fields per query:
    initial_noise : pinned flow noise (already supported upstream)
    hsu_c         : the oracle context scalar for this observation

The inference path replicates `LatentWorldPolicyBackend.predict_action` exactly
(`_run_shared_encoding_infer` -> `flow.sample_actions_cfg`) with the context token appended to
h_vlm, which is the pre-registered position. Nothing in the repository is modified.

Usage: python hsu_server.py --port 10096 --arm none|control|hidden [--adapter path]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsu_model as hm  # noqa: E402

from deployment.model_server.server_policy import build_policy_server_metadata  # noqa: E402
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer  # noqa: E402


class HSUPolicy:
    def __init__(self, arm_name: str, adapter: Path | None):
        self.vla, self.backend = hm.load_backend(use_bf16=True)
        self.backend.eval()
        self.flow = self.backend.flow
        self.builder = self.vla.policy_runner.infer_batch_builder
        hm.freeze_base(self.backend)
        hm.install_lora(self.flow, hm.LORA_RANK)
        dev = next(self.flow.parameters()).device
        self.arm = hm.ContextArm(self.flow, control=(arm_name == "control")).to(dev)
        self.arm.projector.to(torch.float32)
        for m in self.flow.modules():
            if isinstance(m, hm.LoRALinear):
                m.A.to(torch.float32)
                m.B.to(torch.float32)
        self.arm_name = arm_name
        if adapter is None:
            with torch.no_grad():                      # exactly the base policy
                self.arm.projector.alpha.zero_()
                for m in self.flow.modules():
                    if isinstance(m, hm.LoRALinear):
                        m.B.weight.zero_()
            logging.info("HSU server: ORIGINAL LaWAM (no adapter)")
        else:
            ck = torch.load(adapter, map_location="cpu")
            self.arm.projector.load_state_dict(ck["projector"])
            named = dict(self.flow.named_parameters())
            with torch.no_grad():
                for n, v in ck["lora"].items():
                    named[n].copy_(v.to(named[n].dtype).to(named[n].device))
            logging.info("HSU server: arm=%s adapter=%s (update %s, val %.5f)",
                         arm_name, adapter, ck["update"], ck["val_loss"])

    @torch.inference_mode()
    def predict_action(self, **msg):
        examples = msg["examples"]
        initial_noise = msg.get("initial_noise", None)
        c_val = msg.get("hsu_c", None)
        batch = self.builder.build_infer_batch(examples)
        shared = self.backend._run_shared_encoding_infer(
            prepared_batch=batch, source="hsu_server", lam_features_with_no_grad=False)
        attn = (batch["attention_mask"] == 1)
        dev = shared.h_t.device
        b = shared.h_t.shape[0]
        if self.arm_name == "none":
            h_vlm, attn_c = shared.h_vlm, attn
        else:
            c = torch.tensor([[float(c_val if c_val is not None else 0.0)]] * b,
                             device=dev, dtype=torch.float32)
            h_vlm, attn_c = self.arm.condition(shared.h_vlm, attn, c)
        if initial_noise is None:
            noise = None
        else:
            noise = torch.as_tensor(np.asarray(initial_noise, np.float32)).to(dev)
        from starVLA.model.framework.vlas.lawam import _cuda_autocast
        state = torch.zeros(b, int(self.flow.config.state_dim), device=dev, dtype=torch.float32)
        with _cuda_autocast(torch.float32):
            actions = self.flow.sample_actions_cfg(
                h_t=shared.h_t, h_t1_star=shared.h_t1_pred, h_vlm=h_vlm, state=state,
                state_mask=torch.zeros_like(state, dtype=torch.bool),
                action_hz=batch["action_hz"], embodiment_id=batch["embodiment_id"],
                cfg_scale=float(self.flow.config.cfg_guidance_scale),
                num_inference_steps=int(self.flow.config.num_inference_steps),
                attention_mask=attn_c, return_padded=False, initial_noise=noise)
        return {"normalized_actions": actions.detach().float().cpu().numpy()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=10096)
    ap.add_argument("--arm", default="none", choices=["none", "control", "hidden"])
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = HSUPolicy(args.arm, Path(args.adapter) if args.adapter else None)
    # identical metadata to the stock server (the eval client verifies ckpt_path + framework_name)
    meta = build_policy_server_metadata(policy.vla, ckpt_path=str(hm.CKPT),
                                        server_type="starvla_websocket", env="libero",
                                        supported_eval_envs=["libero"],
                                        extra_metadata={"hsu_arm": args.arm})
    WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port,
                          idle_timeout=-1, metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
