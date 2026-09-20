"""Phase-0 model-side checks (lawam env).

--mode baseline : run on the *unpatched* code path (standard predict_action + return_intermediates). Saves outputs.
--mode patched  : run on patched code: (A) default path must equal baseline, (B) self-override equivalence,
                  (C) repeat consistency with fixed noise, plus dtype/shape/finite validation tests.
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
import model_common as mc  # noqa: E402

SEEDS = [0, 1, 2]


def run_default(vla, ex, seed):
    mc.seed_all(seed)
    out = vla.predict_action(examples=[ex], return_intermediates=True)
    return {"actions": np.asarray(out["normalized_actions"]), "h_t": np.asarray(out["intermediates"]["h_t"]), "h_t1_pred": np.asarray(out["intermediates"]["h_t1_pred"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--mode", choices=["baseline", "patched"], required=True)
    args = ap.parse_args()
    out_dir = Path(args.run_dir) / "phase0"
    obs = np.load(out_dir / "phase0_obs.npz")
    lang = str(obs["lang"][0])

    t0 = time.time()
    vla, model_config, norm_stats = mc.load_policy()
    print(f"policy loaded in {time.time()-t0:.1f}s; cuda mem {torch.cuda.memory_allocated()/1e9:.2f} GB")
    examples = [mc.make_example(obs["primary"][i], obs["wrist"][i], obs["state"][i][0], lang, norm_stats) for i in range(len(obs["tags"]))]

    if args.mode == "baseline":
        res = {}
        for i, ex in enumerate(examples):
            for s in SEEDS:
                r = run_default(vla, ex, s)
                for k, v in r.items():
                    res[f"obs{i}_seed{s}_{k}"] = v
            # repeat seed 0 three times (nondeterminism floor of the pristine path)
            for rep in range(3):
                r = run_default(vla, ex, 0)
                res[f"obs{i}_seed0_rep{rep}_actions"] = r["actions"]
                res[f"obs{i}_seed0_rep{rep}_h_t1_pred"] = r["h_t1_pred"]
        np.savez_compressed(out_dir / "model_baseline_pristine.npz", **res)
        rep_diffs = [float(np.max(np.abs(res[f"obs{i}_seed0_rep{a}_actions"] - res[f"obs{i}_seed0_rep{b}_actions"]))) for i in range(len(examples)) for a in range(3) for b in range(a + 1, 3)]
        rep_diffs_h = [float(np.max(np.abs(res[f"obs{i}_seed0_rep{a}_h_t1_pred"] - res[f"obs{i}_seed0_rep{b}_h_t1_pred"]))) for i in range(len(examples)) for a in range(3) for b in range(a + 1, 3)]
        seed_diff = float(np.max(np.abs(res["obs0_seed0_actions"] - res["obs0_seed1_actions"])))
        summary = {"actions_shape": list(res["obs0_seed0_actions"].shape), "h_t_shape": list(res["obs0_seed0_h_t"].shape), "h_t1_pred_shape": list(res["obs0_seed0_h_t1_pred"].shape),
                   "pristine_repeat_maxabs_actions": rep_diffs, "pristine_repeat_maxabs_h_t1_pred": rep_diffs_h, "seed0_vs_seed1_maxabs_actions": seed_diff,
                   "h_t_equals_h_t1_gt_note": "at inference primary_image is a single frame, so backend h_t1_gt == h_t (not a real future)"}
        mc.write_json(out_dir / "model_baseline_pristine_summary.json", summary)
        print(json.dumps(summary, indent=1))
        return

    # ---------------- patched mode
    base = np.load(out_dir / "model_baseline_pristine.npz")
    checks = {}
    backend = vla.policy_backend
    action_horizon = int(backend.flow.action_horizon)
    action_dim = int(backend.flow.config.action_dim)
    code_dim = int(backend.lam.code_dim)

    # (A) default path equals pristine baseline (same seeds, same RNG consumption)
    a_diffs = []
    for i, ex in enumerate(examples):
        for s in SEEDS:
            r = run_default(vla, ex, s)
            a_diffs.append({"obs": i, "seed": s,
                            "actions_maxabs": float(np.max(np.abs(r["actions"] - base[f"obs{i}_seed{s}_actions"]))),
                            "h_t_maxabs": float(np.max(np.abs(r["h_t"] - base[f"obs{i}_seed{s}_h_t"]))),
                            "h_t1_pred_maxabs": float(np.max(np.abs(r["h_t1_pred"] - base[f"obs{i}_seed{s}_h_t1_pred"])))})
    checks["A_default_path_vs_pristine"] = a_diffs

    # (B) self-override equivalence, with explicit initial noise so the comparison is exact
    b_res = []
    for i, ex in enumerate(examples):
        for s in SEEDS:
            noise = mc.make_initial_noise(1000 + s, action_horizon, action_dim)
            mc.seed_all(s)
            ref = vla.predict_action(examples=[ex], initial_noise=noise, return_diagnostics=True)
            d = ref["diagnostics"]
            z = np.asarray(d["pred_latent"]); fut = np.asarray(d["future_used"]); noise_used = np.asarray(d["initial_noise"])
            mc.seed_all(s)
            r_z = vla.predict_action(examples=[ex], initial_noise=noise, latent_override=z, return_diagnostics=True)
            mc.seed_all(s)
            r_f = vla.predict_action(examples=[ex], initial_noise=noise, future_override=fut, return_diagnostics=True)
            mc.seed_all(s)
            r_rep = vla.predict_action(examples=[ex], initial_noise=noise, return_diagnostics=True)
            b_res.append({"obs": i, "seed": s,
                          "noise_roundtrip_maxabs": float(np.max(np.abs(noise_used - noise))),
                          "latent_self_override_actions_maxabs": float(np.max(np.abs(np.asarray(r_z["normalized_actions"]) - np.asarray(ref["normalized_actions"])))),
                          "latent_self_override_future_maxabs": float(np.max(np.abs(np.asarray(r_z["diagnostics"]["future_used"]) - fut))),
                          "future_self_override_actions_maxabs": float(np.max(np.abs(np.asarray(r_f["normalized_actions"]) - np.asarray(ref["normalized_actions"])))),
                          "repeat_fixed_noise_actions_maxabs": float(np.max(np.abs(np.asarray(r_rep["normalized_actions"]) - np.asarray(ref["normalized_actions"])))),
                          "repeat_fixed_noise_h_t1_pred_maxabs": float(np.max(np.abs(np.asarray(r_rep["diagnostics"]["h_t1_pred"]) - np.asarray(d["h_t1_pred"])))),
                          "z_shape": list(z.shape), "z_norm": float(np.linalg.norm(z)), "future_shape": list(fut.shape),
                          "actions_absmax": float(np.max(np.abs(np.asarray(ref["normalized_actions"]))))})
    checks["B_self_override_and_repeat"] = b_res

    # (B2) explicit-noise path vs seeded default path: with the same seed the default path draws torch.randn on cuda;
    # we reproduce that draw and pass it explicitly -> must match the default path exactly.
    b2 = []
    for i, ex in enumerate(examples[:1]):
        for s in SEEDS:
            mc.seed_all(s)
            model_dtype = backend.flow._compute_dtype()
            drawn = torch.randn((1, action_horizon, action_dim), dtype=model_dtype, device="cuda").float().cpu().numpy()
            mc.seed_all(s)
            r_e = vla.predict_action(examples=[ex], initial_noise=drawn)
            b2.append({"obs": i, "seed": s, "explicit_noise_vs_seeded_default_actions_maxabs": float(np.max(np.abs(np.asarray(r_e["normalized_actions"]) - base[f"obs{i}_seed{s}_actions"])))})
    checks["B2_explicit_noise_reproduces_seeded_default"] = b2

    # (C) repeat consistency (3 repeats, fixed noise) already in B; add 5 more repeats on obs0 seed0
    noise = mc.make_initial_noise(1000, action_horizon, action_dim)
    outs = []
    for rep in range(5):
        outs.append(np.asarray(vla.predict_action(examples=[examples[0]], initial_noise=noise)["normalized_actions"]))
    checks["C_repeat_5x_fixed_noise_maxabs"] = float(max(np.max(np.abs(outs[a] - outs[b])) for a in range(5) for b in range(a + 1, 5)))

    # (D) validation: wrong shape / non-finite / dtype
    val = {}
    for name, kwargs in [("latent_wrong_shape", {"latent_override": np.zeros((1, 2, code_dim), np.float32)}),
                         ("latent_nan", {"latent_override": np.full((1, 1, code_dim), np.nan, np.float32)}),
                         ("future_wrong_tokens", {"future_override": np.zeros((1, 100, 768), np.float32)}),
                         ("noise_wrong_shape", {"initial_noise": np.zeros((1, 8, action_dim), np.float32)})]:
        try:
            vla.predict_action(examples=[examples[0]], **kwargs)
            val[name] = "NO_ERROR (BAD)"
        except Exception as e:  # noqa: BLE001
            val[name] = f"raised {type(e).__name__}: {str(e)[:120]}"
    checks["D_validation"] = val

    # (E) sanity that overrides actually change outputs when different (not a no-op)
    mc.seed_all(0)
    ref = vla.predict_action(examples=[examples[0]], initial_noise=noise, return_diagnostics=True)
    z = np.asarray(ref["diagnostics"]["pred_latent"])
    r_neg = vla.predict_action(examples=[examples[0]], initial_noise=noise, latent_override=-z, return_diagnostics=True)
    checks["E_override_effect"] = {"neg_latent_actions_maxabs_change": float(np.max(np.abs(np.asarray(r_neg["normalized_actions"]) - np.asarray(ref["normalized_actions"])))),
                                   "neg_latent_future_maxabs_change": float(np.max(np.abs(np.asarray(r_neg["diagnostics"]["future_used"]) - np.asarray(ref["diagnostics"]["future_used"]))))}
    checks["config"] = {"action_horizon": action_horizon, "action_dim": action_dim, "code_dim": code_dim, "horizon_sec": float(backend.flow.config.horizon_sec),
                        "num_inference_steps": int(backend.flow.config.num_inference_steps), "cfg_guidance_scale": float(backend.flow.config.cfg_guidance_scale),
                        "use_state": bool(backend.flow.config.use_state), "future_prediction": bool(backend.model_cfg.future_prediction),
                        "flow_param_dtype": str(backend.flow._compute_dtype()), "num_action_queries": int(backend.num_action_queries),
                        "lam_vq_type": str(backend.lam.vq_type), "lam_num_frames": int(backend.lam.encoder.num_frames), "vision_tokens": int(backend.lam.encoder.grid_height * backend.lam.encoder.grid_width),
                        "vision_dim": int(backend.lam.input_dim), "cuda_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
    mc.write_json(out_dir / "model_patched_checks.json", checks)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
