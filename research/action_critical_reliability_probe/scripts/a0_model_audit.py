"""AUDIT 0 (model side, `lawam` env): what exactly is the interface we are about to probe?

Records, from the *loaded* checkpoint rather than from the config file alone:
  * flow-head / future-prediction configuration actually in effect,
  * real shapes of pred_latent (z), h_t, h_t1_pred, initial noise, action chunk,
  * which DiT blocks consume the future tokens and how (attention-baseline feasibility),
  * numerical repeatability floor of the action chunk (fixed noise, repeated inference),
  * equivalence of the pre-existing diagnostic hooks with the default path (regression),
  * scale statistics of h_t1_pred needed later to choose perturbation magnitudes.

No perturbation experiment here, no scientific claim. Output: <run_dir>/audit0_model.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402
import model_common as mc  # noqa: E402  (reused verbatim from research/branch_diagnostic)

BD_RUN = pc.REPO_ROOT / "results/branch_diagnostic/bd_20260914_093902"
# Three observations from *different* episodes and different execution steps of the prior run's
# online rollouts. They are used only to read out shapes / floors, not for any S or U statistic.
AUDIT_OBS = [("ep000", 8), ("ep011", 24), ("ep019", 56)]


def load_obs(ep: str, step: int):
    d = np.load(BD_RUN / f"candidates/task02/{ep}/steps.npz")
    meta = json.loads((BD_RUN / f"candidates/task02/{ep}/episode.json").read_text())
    return {
        "ep": ep,
        "step": step,
        "primary": d["primary"][step],
        "wrist": d["wrist"][step],
        "state": d["state"][step],
        "lang": meta["instruction"],
        "num_actions": int(meta["num_actions"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    args = ap.parse_args()
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    vla, model_config, norm_stats = mc.load_policy()
    backend = vla.policy_backend
    flow = backend.flow
    cfg = flow.config
    load_sec = time.time() - t0

    am = model_config["framework"]["action_model"]
    audit: dict = {
        "run_id": out_dir.name,
        "git_commit": pc.git("rev-parse", "HEAD"),
        "git_status_short": pc.git("status", "--short"),
        "checkpoint": str(mc.DEFAULT_CKPT),
        "checkpoint_size_bytes": mc.DEFAULT_CKPT.stat().st_size,
        "checkpoint_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mc.DEFAULT_CKPT.stat().st_mtime)),
        "policy_load_sec": round(load_sec, 1),
        "env": pc.env_versions(),
        "gpu": torch.cuda.get_device_name(0),
        "config_effective": {
            "future_prediction": bool(am.get("future_prediction")),
            "detach_future_feature": bool(am.get("detach_future_feature")),
            "enable_flow_h_t1_scheduled_sampling": bool(am.get("enable_flow_h_t1_scheduled_sampling")),
            "flow_h_t1_pred_prob_now": float(backend._flow_h_t1_pred_prob()),
            "action_horizon_cfg": int(am.get("action_horizon")),
            "flow.action_horizon": int(flow.action_horizon),
            "horizon_sec": float(cfg.horizon_sec),
            "action_hz_client_default": float(mc.ACTION_HZ),
            "effective_chunk_len": int(np.floor(float(cfg.horizon_sec) * mc.ACTION_HZ)),
            "action_dim": int(cfg.action_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "vision_dim": int(cfg.vision_dim),
            "num_vision_tokens": int(cfg.num_vision_tokens),
            "num_target_vision_tokens": int(cfg.num_target_vision_tokens),
            "flow_future_query_tokens": None if flow.future_tokens is None else int(flow.future_tokens.weight.shape[0]),
            "use_state": bool(cfg.use_state),
            "cfg_guidance_scale": float(cfg.cfg_guidance_scale),
            "num_inference_steps": int(cfg.num_inference_steps),
            "num_layers": int(cfg.num_layers),
            "attention_heads": int(cfg.attention_heads),
            "use_alternate_vldit": bool(cfg.use_alternate_vldit),
            "attend_text_every_n_blocks": int(cfg.attend_text_every_n_blocks),
            "num_action_queries": int(am.get("num_action_queries")),
            "lam_ckpt_path": am.get("lam_ckpt_path"),
        },
        "model_dtype": str(flow._compute_dtype()),
    }

    # --- which DiT blocks see which condition stream (attention-baseline feasibility)
    n_layers = int(cfg.num_layers)
    every = int(cfg.attend_text_every_n_blocks)
    blocks = []
    for idx in range(n_layers):
        if idx % 2 == 1:
            kind = "self_attn(action tokens)"
        elif idx % (2 * every) == 0:
            kind = "cross_attn(VLM tokens)"
        else:
            kind = "cross_attn(image tokens: h_t + h_t1_pred)"
        blocks.append({"idx": idx, "kind": kind, "processor": type(flow.DiT.transformer_blocks[idx].attn1.processor).__name__})
    audit["dit_blocks"] = blocks
    audit["future_reading_blocks"] = [b["idx"] for b in blocks if b["kind"].startswith("cross_attn(image")]

    # --- shapes and floors on real observations
    H, Ad = int(flow.action_horizon), int(cfg.action_dim)
    noise = mc.make_initial_noise(1234, H, Ad)
    per_obs = []
    for ep, step in AUDIT_OBS:
        o = load_obs(ep, step)
        ex = mc.make_example(o["primary"], o["wrist"], o["state"], o["lang"], norm_stats)

        r = vla.predict_action(examples=[ex], initial_noise=noise, return_diagnostics=True)
        d = r["diagnostics"]
        a0 = np.asarray(r["normalized_actions"])
        h_pred = np.asarray(d["h_t1_pred"])
        h_t = np.asarray(d["h_t"])
        z = np.asarray(d["pred_latent"])

        # repeatability with identical inputs and identical noise
        reps = [np.asarray(vla.predict_action(examples=[ex], initial_noise=noise)["normalized_actions"]) for _ in range(4)]
        repeat_floor = max(float(np.max(np.abs(reps[i] - reps[j]))) for i in range(len(reps)) for j in range(i + 1, len(reps)))
        repeat_floor_vs_first = float(np.max(np.abs(reps[0] - a0)))

        # hook no-op regression: overriding the future with itself must reproduce the default path
        r_self = vla.predict_action(examples=[ex], initial_noise=noise, future_override=torch.from_numpy(h_pred))
        self_override_diff = float(np.max(np.abs(np.asarray(r_self["normalized_actions"]) - a0)))
        r_selfz = vla.predict_action(examples=[ex], initial_noise=noise, latent_override=torch.from_numpy(z))
        self_latent_diff = float(np.max(np.abs(np.asarray(r_selfz["normalized_actions"]) - a0)))

        # default path (own RNG) vs explicit-noise path: different noise -> only documents the scale of
        # action variation caused by the flow noise itself
        mc.seed_all(0)
        a_def = np.asarray(vla.predict_action(examples=[ex])["normalized_actions"])
        noise_b = mc.make_initial_noise(4321, H, Ad)
        a_noise_b = np.asarray(vla.predict_action(examples=[ex], initial_noise=noise_b)["normalized_actions"])

        tok_norm = np.linalg.norm(h_pred[0], axis=-1)
        tok_norm_ht = np.linalg.norm(h_t[0], axis=-1)
        per_obs.append({
            "obs": f"{ep}@step{step}",
            "instruction": o["lang"],
            "shapes": {
                "pred_latent_z": list(z.shape),
                "h_t": list(h_t.shape),
                "h_t1_pred": list(h_pred.shape),
                "initial_noise_used": list(np.asarray(d["initial_noise"]).shape),
                "normalized_actions": list(a0.shape),
                "h_vlm": list(d["h_vlm_shape"]),
                "h_vlm_dtype": d["h_vlm_dtype"],
            },
            "action_repeat_floor_maxabs": repeat_floor,
            "action_repeat_floor_vs_diagnostics_call": repeat_floor_vs_first,
            "self_future_override_maxabs_diff": self_override_diff,
            "self_latent_override_maxabs_diff": self_latent_diff,
            "action_maxabs_change_other_noise_seed": float(np.max(np.abs(a_noise_b[..., :7] - a0[..., :7]))),
            "action_maxabs_change_default_rng_vs_fixed_noise": float(np.max(np.abs(a_def[..., :7] - a0[..., :7]))),
            "action_abs_mean_first7dims": float(np.mean(np.abs(a0[..., :7]))),
            "h_t1_pred_token_norm": pc.stats(tok_norm),
            "h_t_token_norm": pc.stats(tok_norm_ht),
            "h_t1_pred_elementwise_std": float(h_pred.std()),
            "h_t1_pred_perdim_std_over_tokens": pc.stats(h_pred[0].std(axis=0)),
            "mse_h_t1_pred_vs_h_t": float(((h_pred - h_t) ** 2).mean()),
            "cos_h_t1_pred_vs_h_t_per_token": pc.stats(
                (h_pred[0] * h_t[0]).sum(-1) / (np.linalg.norm(h_pred[0], axis=-1) * np.linalg.norm(h_t[0], axis=-1) + 1e-12)
            ),
        })
        print(f"{ep}@{step}: h_t1_pred{list(h_pred.shape)} actions{list(a0.shape)} "
              f"floor={repeat_floor:.3g} selfovr={self_override_diff:.3g} tok_norm~{tok_norm.mean():.2f}", flush=True)

    audit["observations"] = per_obs
    audit["notes"] = [
        "All numbers come from the loaded checkpoint, not from config.yaml alone.",
        "The four diagnostic hooks (latent_override / future_override / initial_noise / return_diagnostics) "
        "pre-exist in the working tree from run bd_20260914_093902; this audit re-verifies their no-op behaviour.",
        "num_target_vision_tokens<=0 means the flow head has NO learnable future query tokens in hidden_states; "
        "h_t1_pred reaches the action head only as cross-attention keys/values inside the image-token half.",
    ]
    pc.write_json(out_dir / "audit0_model.json", audit)
    print(json.dumps({k: audit[k] for k in ("config_effective", "future_reading_blocks")}, indent=1))


if __name__ == "__main__":
    main()
