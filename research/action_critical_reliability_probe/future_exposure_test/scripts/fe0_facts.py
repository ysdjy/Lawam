"""FE AUDIT 0 — collect every fact the frozen PROTOCOL_FE.yaml depends on.

Nothing here trains, and nothing here is a scientific result: it verifies the checkpoint anchor,
the freeze plan, the split, the normalization source, the loader's episode subsetting, the exact
time alignment of (h_t, H_gt, A_gt), and the engineering sizes/throughput that the cached design
needs. Writes PREFLIGHT_FE_AUDIT0.json into the FE run directory.

Run under the `lawam` env.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common as fc  # noqa: E402

OUT = fc.RUN_DIR / "PREFLIGHT_FE_AUDIT0.json"


def main() -> None:
    rep: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": "FE AUDIT 0"}

    # ---------------------------------------------------------------- 1. source anchors
    import subprocess

    rep["git"] = {
        "branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=fc.REPO_ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=fc.REPO_ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "dirty_tracked_files": subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                              cwd=fc.REPO_ROOT, capture_output=True, text=True).stdout.strip(),
    }

    stats_path = fc.CKPT_DIR / "dataset_statistics.json"
    stats = json.loads(stats_path.read_text())
    rep["normalization"] = {
        "path": str(stats_path.relative_to(fc.REPO_ROOT)),
        "sha256": fc.sha256_file(stats_path),
        "embodiments": list(stats.keys()),
        "num_trajectories": stats["franka"]["num_trajectories"],
        "num_transitions": stats["franka"]["num_transitions"],
        "action_mask": stats["franka"]["action"]["mask"],
        "source_is_release_checkpoint": True,
        "recomputed": False,
    }

    # ---------------------------------------------------------------- 2. split manifest
    man = fc.load_split_manifest()
    sets = {k: set(int(e) for e in v) for k, v in man["episodes"].items()}
    keys = sorted(sets)
    rep["split"] = {
        "manifest_sha256": fc.SPLIT_MANIFEST_SHA256,
        "counts": {k: len(v) for k, v in sets.items()},
        "total": sum(len(v) for v in sets.values()),
        "pairwise_overlap": {f"{a}&{b}": len(sets[a] & sets[b])
                             for i, a in enumerate(keys) for b in keys[i + 1:]},
        "union_min": min(min(v) for v in sets.values()),
        "union_max": max(max(v) for v in sets.values()),
        "union_size": len(set().union(*sets.values())),
    }

    # ---------------------------------------------------------------- 3. model + freeze plan
    # NOTE: previous rounds loaded with use_bf16=True, the DEPLOYMENT cast. This round trains, so
    # it uses the configuration the official trainer uses: fp32 weights plus the model's own
    # _cuda_autocast(bf16) stages. The deployment cast additionally breaks the LAM teacher, whose
    # VQ runs in fp32 (`RuntimeError: expected scalar type Float but found BFloat16` in
    # _compute_distill_loss), which is further evidence it is not the training configuration.
    t0 = time.time()
    vla, backend = fc.load_backend(use_bf16=False)
    rep["load_seconds"] = round(time.time() - t0, 1)
    rep["precision_mode"] = {
        "use_bf16_cast": False,
        "reason": "official training configuration: fp32 weights + _cuda_autocast per stage",
        "differs_from_previous_rounds": True,
    }

    sd = backend.state_dict()
    import hashlib
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        h.update(k.encode()); h.update(b"\0")
        h.update(str(t.dtype).encode()); h.update(b"\0")
        h.update(str(tuple(t.shape)).encode()); h.update(b"\0")
        h.update(t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    rep["checkpoint"] = {
        "file": str(fc.CKPT_FILE.relative_to(fc.REPO_ROOT)),
        "runtime_state_dict_keys": len(sd),
        "runtime_total_params": sum(p.numel() for p in backend.parameters()),
    }

    freeze = fc.freeze_all_but_flow(backend)
    rep["freeze_plan"] = freeze
    fc.set_train_modes(backend)
    rep["module_modes"] = {
        "backend_training": bool(backend.training),
        "flow_training": bool(backend.flow.training),
        "vlm_training": bool(backend.vlm.training),
        "lam_training": bool(backend.lam.training),
    }
    rep["frozen_module_hashes"] = {
        g: fc.module_state_hash(getattr(backend, g)) for g in fc.FROZEN_GROUPS if hasattr(backend, g)
    }
    def _numel(obj):
        return int(obj.numel()) if torch.is_tensor(obj) else sum(p.numel() for p in obj.parameters())

    rep["frozen_module_params"] = {
        g: _numel(getattr(backend, g)) for g in fc.FROZEN_GROUPS if hasattr(backend, g)
    }
    rep["flow_initial_hash"] = fc.module_state_hash(backend.flow)

    # ---------------------------------------------------------------- 4. config facts read from the model
    fcfg = backend.flow.config
    mcfg = backend.model_cfg
    rep["flow_config"] = {
        k: getattr(fcfg, k) for k in (
            "action_dim", "state_dim", "use_state", "vlm_dim", "hidden_dim", "attention_heads",
            "num_layers", "cfg_drop_prob", "cfg_guidance_scale", "num_inference_steps",
            "num_timestep_buckets", "horizon_sec", "noise_beta_alpha", "noise_beta_beta",
            "noise_s", "token_independent_noise", "use_action_positional_embeddings",
        ) if hasattr(fcfg, k)
    }
    rep["model_config"] = {
        k: getattr(mcfg, k) for k in (
            "detach_future_feature", "enable_flow_h_t1_scheduled_sampling",
            "flow_h_t1_pred_prob_start", "flow_h_t1_pred_prob_end", "flow_h_t1_pred_ramp_steps",
            "flow_only_mode", "future_prediction", "enable_loss_distill",
            "lam_encoder_distill_weight", "perceptual_weight", "repeated_diffusion_steps",
        ) if hasattr(mcfg, k)
    }
    rep["eval_future_selector_returns_H_pred"] = bool(
        backend._build_flow_future_condition(
            h_t1_pred=torch.zeros(1, 1, 1), h_t1_gt=torch.ones(1, 1, 1)
        ).sum().item() == 0.0
    )

    # ---------------------------------------------------------------- 5. dataset + episode subsetting
    cfg = fc.load_train_cfg()
    rep["train_config_dataset"] = {
        k: cfg.datasets.vla_data.get(k) for k in (
            "data_root_dir", "data_mix", "image_resolution", "num_frames", "sec_chunk",
            "enable_primary_video_aug", "enable_primary_random_resized_crop", "train_split_all",
            "video_backend",
        )
    }
    t0 = time.time()
    ds_all = fc.build_dataset(cfg, dataset_statistics_override=None)
    rep["dataset_build_seconds"] = round(time.time() - t0, 1)
    leaves = fc._inner_datasets(ds_all)
    rep["dataset"] = {
        "n_leaves": len(leaves),
        "leaf_names": [getattr(d, "dataset_name", None) for d in leaves],
        "total_steps_mode_all": int(sum(int(d._subset_total_steps) for d in leaves)),
        "total_episodes_mode_all": int(sum(len(d._active_episode_indices) for d in leaves)),
    }

    train_eps = sorted(sets["train"])
    rep["restrict_train"] = fc.restrict_to_episodes(ds_all, train_eps)

    # time alignment + eligibility, from the restricted loader
    leaf = fc._inner_datasets(ds_all)[0]
    lens = fc.episode_lengths(leaf)
    eligible = {e: max(0, L - 7) for e, L in lens.items()}   # starts 0 .. L-8 inclusive
    rep["eligibility"] = {
        "n_episodes": len(lens),
        "total_steps": int(sum(lens.values())),
        "total_eligible_starts": int(sum(eligible.values())),
        "min_episode_length": int(min(lens.values())),
        "max_episode_length": int(max(lens.values())),
        "episodes_with_zero_eligible_starts": [int(e) for e, n in eligible.items() if n == 0],
    }

    # ---------------------------------------------------------------- 6. one real sample: shapes + alignment
    collator = fc.build_collator(cfg, training=True)
    rep["collator"] = {
        "class": type(collator).__name__,
        "training": bool(getattr(collator, "training", None)),
        "enable_primary_video_aug": bool(getattr(collator, "enable_primary_video_aug", None)),
        "enable_primary_random_resized_crop": bool(getattr(collator, "enable_primary_random_resized_crop", None)),
        "act_queries": int(getattr(collator, "act_queries", -1)),
        "flow_queries": int(getattr(collator, "flow_queries", -1)),
    }
    rep["video_modality"] = {
        "keys": list(leaf.modality_keys.get("video", [])),
        "random_single_non_wrist_view": bool(getattr(ds_all, "random_single_non_wrist_view", False)),
        "video_delta_indices": [int(x) for x in getattr(leaf, "video_delta_indices", [])] or None,
    }

    ep0 = int(train_eps[0])
    start0 = 5
    s_a = fc.get_sample(ds_all, leaf, ep0, start0)
    s_b = fc.get_sample(ds_all, leaf, ep0, start0 + 1)
    rep["sample_probe"] = {
        "episode": ep0, "start": start0,
        "local_index": int(fc.episode_start_to_local_index(leaf, ep0, start0)),
        "keys": sorted(s_a.keys()),
        "primary_videos_shape": list(s_a["primary_videos"].shape),
        "wrist_images_shape": list(s_a["wrist_images"].shape),
        "action_shape": list(s_a["action"].shape),
        "state_shape": list(s_a["state"].shape),
        "lang": str(s_a["lang"])[:120],
    }

    # Determinism of the sample pipeline (the cached design in the protocol depends on it).
    s_a2 = fc.get_sample(ds_all, leaf, ep0, start0)
    rep["sample_determinism"] = {
        k: bool(torch.equal(s_a[k], s_a2[k]))
        for k in ("primary_videos", "wrist_images", "action", "state")
    }

    # Time alignment. primary_videos is [view, time, C, H, W] with time = num_frames = 2 and
    # video_delta_indices [0, 7]: frame 0 is o_t, frame 1 is o_{t+7}. The decisive check is that
    # the FUTURE frame of start s is the very same image as the CURRENT frame of start s+7.
    checks = {"future_of_s_is_current_of_s_plus_7": [], "two_frames_differ": [],
              "action_chunk_matches_next_starts": []}
    for ep in train_eps[:8]:
        L = lens[int(ep)]
        for s in (0, 5, min(40, L - 16)):
            if s < 0 or s + 7 > L - 8:
                continue
            a = fc.get_sample(ds_all, leaf, int(ep), int(s))
            c = fc.get_sample(ds_all, leaf, int(ep), int(s + 7))
            checks["future_of_s_is_current_of_s_plus_7"].append(
                bool(torch.equal(a["primary_videos"][:, 1], c["primary_videos"][:, 0])))
            checks["two_frames_differ"].append(
                bool(not torch.equal(a["primary_videos"][:, 0], a["primary_videos"][:, 1])))
            # A_gt of start s, steps 1..7, must equal A_gt of start s+1, steps 0..6.
            b1 = fc.get_sample(ds_all, leaf, int(ep), int(s + 1))
            checks["action_chunk_matches_next_starts"].append(
                bool(torch.equal(a["action"][1:8], b1["action"][0:7])))
    rep["time_alignment"] = {
        "video_time_axis": 1,
        "frames_per_sample": int(s_a["primary_videos"].shape[1]),
        "n_checks": len(checks["future_of_s_is_current_of_s_plus_7"]),
        "future_of_s_is_current_of_s_plus_7__all_true": all(checks["future_of_s_is_current_of_s_plus_7"]),
        "two_frames_differ__all_true": all(checks["two_frames_differ"]),
        "action_chunk_matches_next_starts__all_true": all(checks["action_chunk_matches_next_starts"]),
        "meaning": "H_gt is built from frame s+7 of the SAME episode and A_gt is the action chunk "
                   "of the SAME time window. Unlike the Headroom round's H_real, this future is "
                   "the consistent counterfactual of the supervised actions by construction.",
    }

    batch = collator([s_a, s_b])
    rep["collated_batch_shapes"] = {
        k: (list(v.shape) + [str(v.dtype)]) if torch.is_tensor(v) else str(type(v))
        for k, v in batch.items()
    }

    # ---------------------------------------------------------------- 7. shared encoding: shapes, bytes, time
    batch_cuda = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        prepared = backend._prepare_train_batch(batch=batch_cuda)
        t0 = time.time()
        shared = backend._run_shared_encoding_train(
            prepared_batch=prepared, source="fe0_facts", lam_features_with_no_grad=True
        )
        torch.cuda.synchronize()
        enc_s = time.time() - t0

    def _desc(t):
        return {"shape": list(t.shape), "dtype": str(t.dtype),
                "bytes_per_sample": int(t.numel() // t.shape[0] * t.element_size())}

    cached = {
        "h_vlm": shared.h_vlm, "h_t": shared.h_t,
        "h_t1_pred": shared.h_t1_pred, "h_t1_gt": shared.h_t1_gt,
        "actions": prepared["actions"], "actions_mask": prepared["actions_mask"],
    }
    rep["shared_encoding"] = {k: _desc(v) for k, v in cached.items()}
    rep["shared_encoding"]["attention_mask"] = _desc(prepared["attention_mask"])
    per_sample_bytes = sum(rep["shared_encoding"][k]["bytes_per_sample"] for k in rep["shared_encoding"])
    rep["cache_sizing"] = {
        "bytes_per_sample": per_sample_bytes,
        "mib_per_sample": round(per_sample_bytes / 1024 ** 2, 2),
        "projected_gib_8000_train": round(per_sample_bytes * 8000 / 1024 ** 3, 1),
        "projected_gib_512_val": round(per_sample_bytes * 512 / 1024 ** 3, 1),
        "projected_gib_1380_eval": round(per_sample_bytes * 1380 / 1024 ** 3, 1),
        "ceiling_gib": 100,
        "encode_seconds_per_batch_of_2": round(enc_s, 3),
    }

    # H_pred vs H_gt separation, purely descriptive (no arm, no training)
    d_pred_gt = (shared.h_t1_pred.float() - shared.h_t1_gt.float()).norm(dim=-1).mean().item()
    d_pred_ht = (shared.h_t1_pred.float() - shared.h_t.float()).norm(dim=-1).mean().item()
    d_gt_ht = (shared.h_t1_gt.float() - shared.h_t.float()).norm(dim=-1).mean().item()
    rep["future_feature_separation_2_samples"] = {
        "mean_token_norm_Hpred_minus_Hgt": round(d_pred_gt, 4),
        "mean_token_norm_Hpred_minus_ht": round(d_pred_ht, 4),
        "mean_token_norm_Hgt_minus_ht": round(d_gt_ht, 4),
        "note": "two samples only; descriptive sanity check that H_gt is not H_pred and not h_t",
    }

    # ---------------------------------------------------------------- 8. distill/perceptual carry no flow grad
    for p in backend.flow.parameters():
        p.grad = None
    out = backend.forward(batch=batch_cuda)
    rep["official_forward_losses"] = {k: float(v.detach().float().item()) for k, v in out.items()}
    aux = out["loss_perceptual"] + out["loss_distill"]
    flow_params = [p for p in backend.flow.parameters() if p.requires_grad]
    if aux.requires_grad:
        grads = torch.autograd.grad(aux, flow_params, allow_unused=True, retain_graph=False)
        n_non_none = sum(1 for g in grads if g is not None)
    else:
        n_non_none = 0
    rep["aux_loss_grad_wrt_flow"] = {
        "aux_requires_grad": bool(aux.requires_grad),
        "n_flow_params_with_non_none_grad": int(n_non_none),
        "n_flow_params": len(flow_params),
        "gate_passed": n_non_none == 0,
        "meaning": "loss_perceptual + loss_distill do not depend on any Flow parameter, so "
                   "back-propagating loss_flow alone is exact for the only trainable module",
    }

    rep["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
    }

    fc.write_json(OUT, rep)
    print(json.dumps(rep, indent=2, default=str)[:12000])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
