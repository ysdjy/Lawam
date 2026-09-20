"""G2 AUDIT 0 (model side, `lawam` env): verify the prior run's baseline is still reproducible.

Checks, against the FROZEN prior run `acr_20260919_203850` (read-only):
  A. LaWAM weights unchanged -> sha256 of the full state_dict (this hash is the reference that every
     later training log must match, proving no LaWAM parameter was touched).
  B. The policy path still reproduces the prior run bit-exactly: for sampled prior states, recompute
     H_pred / h_t / the baseline action chunk and compare with the stored arrays.
  C. A sampled subset of per-token S values is recomputed with the *unchanged* m2 settings and compared
     with the stored S.
  D. Permutation invariance of the action head w.r.t. future tokens still holds.
  E. H_real is confirmed to be absent from every policy input (structural check: the example dict passed
     to the model is built only from o_t fields).

Writes <run_dir>/g2_audit0_model.json. Exits 1 if any hard check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
import probe_common as pc  # noqa: E402
import model_common as mc  # noqa: E402
import probe_model as pm  # noqa: E402

PRIOR_RUN = REPO / "results/action_critical_reliability_probe/acr_20260919_203850"
SAMPLE_STATES = [
    "libero_spatial_t02_ep000_s0024_approach",
    "libero_object_t00_ep004_s0048_pre_grasp",
    "libero_goal_t08_ep007_s0056_manipulation",
]
SAMPLE_TOKENS = [0, 37, 128, 255]
EPS_MID = 0.812
EPS_MID_IDX = 1
N_DIRS = 3
NOISE_SEED = 101
M2_BATCH = 32          # identical to m2_sensitivity.BATCH; the batch layout must match bit-for-bit


def state_dict_hash(module) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        t = v.detach()
        h.update(np.ascontiguousarray(t.float().cpu().numpy()).tobytes())
        n += t.numel()
    return h.hexdigest(), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {json.loads(l)["state_id"]: json.loads(l)
                for l in open(PRIOR_RUN / "states_manifest.jsonl")}
    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    backend = runner.backend
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noise = torch.from_numpy(pm.make_noise(NOISE_SEED, H, Ad, 1))

    t0 = time.time()
    full_hash, n_params = state_dict_hash(backend)
    flow_hash, _ = state_dict_hash(runner.flow)
    print(f"LaWAM state_dict sha256 = {full_hash} ({n_params} params, {time.time()-t0:.0f}s)", flush=True)

    failures: list[str] = []
    per_state = []
    for sid in SAMPLE_STATES:
        m = manifest[sid]
        d = np.load(m["state_npz"], allow_pickle=True)
        stored = np.load(PRIOR_RUN / "sensitivity" / f"{sid}.npz")
        metrics = [str(x) for x in stored["metrics"]]
        mi = metrics.index("l2_norm_all7")

        # --- E: the example is built from o_t only (no future field can reach it)
        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        forbidden = [k for k in ex if "future" in k.lower() or "real" in k.lower()]
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred

        # --- B: features and baseline action reproduce the stored arrays
        h_pred_now = Hp[0].float().cpu().numpy().astype(np.float16)
        h_t_now = ctx.h_t[0].float().cpu().numpy().astype(np.float16)
        d_hpred = float(np.max(np.abs(h_pred_now.astype(np.float32) - stored["h_t1_pred"].astype(np.float32))))
        d_ht = float(np.max(np.abs(h_t_now.astype(np.float32) - stored["h_t"].astype(np.float32))))
        a0 = runner.actions(ctx, Hp, noise)[0]
        d_a0 = float(np.max(np.abs(a0 - stored["baseline_action"])))

        # --- C: recompute S for the FIRST m2 batch, replicating m2_sensitivity's batch composition
        # exactly (batch of 32 = 1 in-batch baseline + 31 consecutive tokens, one epsilon, one direction).
        # Reproducing the batch layout matters: AUDIT 0b established that the same input evaluated at a
        # different batch size shifts by ~5e-7, which is not small next to S values of ~1e-5.
        k = 0
        toks = list(range(0, M2_BATCH - 1))
        rows = [Hp[0]]
        for j in toks:
            f = Hp[0].clone()
            delta = torch.as_tensor(EPS_MID * pm.unit_directions(sid, j, N_DIRS, ctx.feature_dim)[k],
                                    device=Hp.device, dtype=torch.float32)
            f[j] = (f[j].float() + delta).to(f.dtype)
            rows.append(f)
        fut = torch.stack(rows, dim=0)
        a = runner.actions(ctx, fut, torch.from_numpy(np.repeat(pm.make_noise(NOISE_SEED, H, Ad, 1),
                                                                fut.shape[0], axis=0)))
        dd = pm.action_distance_batch(a[1:], a[0])["l2_norm_all7"]
        stored_S = stored["S"][EPS_MID_IDX, k, toks, mi]
        d_S = float(np.max(np.abs(dd - stored_S)))

        # --- D: permutation invariance of the action head over future tokens
        g = torch.Generator().manual_seed(11)
        perm = torch.randperm(ctx.n_tokens, generator=g).to(Hp.device)
        a_perm = runner.actions(ctx, Hp[0][perm].unsqueeze(0), noise)[0]
        d_perm = float(np.max(np.abs(a_perm - a0)))

        rec = {
            "state_id": sid,
            "example_keys": sorted(ex.keys()),
            "forbidden_future_keys_in_example": forbidden,
            "h_t1_pred_maxabs_vs_stored": d_hpred,
            "h_t_maxabs_vs_stored": d_ht,
            "baseline_action_maxabs_vs_stored": d_a0,
            "sampled_S_maxabs_vs_stored": d_S,
            "future_token_permutation_action_maxabs": d_perm,
        }
        per_state.append(rec)
        if d_hpred > 0.0 or d_ht > 0.0:
            failures.append(f"{sid}: features differ from stored (h_pred {d_hpred}, h_t {d_ht})")
        if d_a0 > 0.0:
            failures.append(f"{sid}: baseline action differs from stored by {d_a0}")
        if d_S > 1e-9:
            failures.append(f"{sid}: recomputed S differs from stored by {d_S}")
        if d_perm > 1e-6:
            failures.append(f"{sid}: future-token permutation changes the action by {d_perm}")
        if forbidden:
            failures.append(f"{sid}: policy example contains future-like keys {forbidden}")
        print(f"{sid}: dH={d_hpred:.3g} dh_t={d_ht:.3g} dA0={d_a0:.3g} dS={d_S:.3g} perm={d_perm:.3g}",
              flush=True)

    payload = {
        "run_id": out_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prior_run": str(PRIOR_RUN),
        "lawam_state_dict_sha256": full_hash,
        "lawam_n_params": n_params,
        "flow_head_state_dict_sha256": flow_hash,
        "checkpoint": str(mc.DEFAULT_CKPT),
        "checkpoint_size_bytes": mc.DEFAULT_CKPT.stat().st_size,
        "checkpoint_mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                          time.localtime(mc.DEFAULT_CKPT.stat().st_mtime)),
        "per_state": per_state,
        "failures": failures,
        "passed": not failures,
    }
    pc.write_json(out_dir / "g2_audit0_model.json", payload)
    print(json.dumps({"passed": payload["passed"], "failures": failures}, indent=1))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
