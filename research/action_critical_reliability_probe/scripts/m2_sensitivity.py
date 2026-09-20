"""Step 1: Downstream Action Sensitivity S, per future token (`lawam` env).

For every selected state and every one of the 256 future tokens:
    h_j' = h_j + eps * u_k     (u_k a unit vector in R^768, 3 fixed directions, 3 pre-registered eps)
    S_j  = D(A_j, A_0) / ||delta_effective_j||
with the observation, instruction, h_vlm, z, h_t, CFG scale, inference steps and the flow initial noise all
held fixed, and A_0 computed as a row of the SAME batch (amendment A1: in-batch baseline has a 0.0 floor).

Also measured per state, as controls / reference scales:
  * repeat rows (numerical floor)
  * random perturbation of the same total norm spread over all tokens
  * random token permutation of H_pred
  * global scaling, all-zero future, future replaced by h_t
  * a different flow-noise seed (the nuisance scale)

Nothing here trains, changes weights or touches the simulator.

Outputs: <run_dir>/sensitivity_records.jsonl (summaries) and <run_dir>/sensitivity/<state_id>.npz (raw).
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
import model_common as mc  # noqa: E402
import probe_model as pm  # noqa: E402

EPS = [0.271, 0.812, 2.706]        # protocol: 1% / 3% / 10% of the mean H_pred token norm (27.06)
N_DIRS = 3
NOISE_SEEDS = [101, 202, 303]
BATCH = 32                          # rows per forward: 1 in-batch baseline + 31 perturbations
METRICS = pm.DISTANCE_KEYS


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    ap.add_argument("--states", default="all", help="comma-separated state_ids, or 'all'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    manifest = [json.loads(l) for l in open(run_dir / "states_manifest.jsonl")]
    if args.states != "all":
        keep = set(args.states.split(","))
        manifest = [m for m in manifest if m["state_id"] in keep]
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = run_dir / "sensitivity"
    out_dir.mkdir(exist_ok=True)
    rec_path = run_dir / "sensitivity_records.jsonl"

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noises = {s: pm.make_noise(s, H, Ad, 1) for s in NOISE_SEEDS}

    done = set()
    if rec_path.exists():
        done = {json.loads(l)["state_id"] for l in open(rec_path)}

    for m in manifest:
        sid = m["state_id"]
        if sid in done:
            print(f"{sid}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        d = np.load(m["state_npz"], allow_pickle=True)
        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred                      # [1, 256, 768] bf16 on cuda
        N, D = ctx.n_tokens, ctx.feature_dim
        noise_primary = torch.from_numpy(noises[NOISE_SEEDS[0]])

        # ---------------- baseline (B=1, also used for cross-check) and reference scales
        a0_single = runner.actions(ctx, Hp, noise_primary)[0]

        # ---------------- per-token sweep
        S = np.zeros((len(EPS), N_DIRS, N, len(METRICS)), dtype=np.float64)
        delta_norms = np.zeros((len(EPS), N_DIRS, N), dtype=np.float64)
        inbatch_base_consistency = []
        dirs_all = np.stack([pm.unit_directions(sid, j, N_DIRS, D) for j in range(N)], axis=0)  # [N, K, D]

        for ei, eps in enumerate(EPS):
            for k in range(N_DIRS):
                for start in range(0, N, args.batch - 1):
                    toks = list(range(start, min(start + args.batch - 1, N)))
                    rows = [Hp[0]]
                    for j in toks:
                        f = Hp[0].clone()
                        delta = torch.as_tensor(eps * dirs_all[j, k], device=Hp.device, dtype=torch.float32)
                        f[j] = (f[j].float() + delta).to(f.dtype)
                        rows.append(f)
                        delta_norms[ei, k, j] = float(torch.linalg.norm(f[j].float() - Hp[0, j].float()))
                    fut = torch.stack(rows, dim=0)
                    a = runner.actions(ctx, fut, torch.from_numpy(
                        np.repeat(noises[NOISE_SEEDS[0]], fut.shape[0], axis=0)))
                    dist = pm.action_distance_batch(a[1:], a[0])
                    for mi, key in enumerate(METRICS):
                        S[ei, k, toks, mi] = dist[key]
                    inbatch_base_consistency.append(float(np.max(np.abs(a[0] - a0_single))))

        # normalise by the effective delta norm
        S_norm = S / delta_norms[..., None]

        # ---------------- controls / reference scales, measured in one batch with an in-batch baseline
        rng = np.random.default_rng(int.from_bytes(sid.encode()[:4].ljust(4, b"0"), "big"))
        all_tok_dir = rng.standard_normal((N, D))
        all_tok_dir /= np.linalg.norm(all_tok_dir, axis=1, keepdims=True)
        perm = torch.as_tensor(rng.permutation(N), device=Hp.device)
        ctrl_rows = {
            "repeat_identical": Hp[0],
            "future_all_zeros": torch.zeros_like(Hp[0]),
            "future_replaced_by_h_t": ctx.h_t[0],
            "future_scaled_1.05": (Hp[0].float() * 1.05).to(Hp.dtype),
            "future_scaled_0.95": (Hp[0].float() * 0.95).to(Hp.dtype),
            "future_token_permuted": Hp[0][perm],
            "all_tokens_perturbed_eps_mid": (Hp[0].float() + torch.as_tensor(
                EPS[1] * all_tok_dir, device=Hp.device, dtype=torch.float32)).to(Hp.dtype),
            "all_tokens_perturbed_eps_mid_over_sqrtN": (Hp[0].float() + torch.as_tensor(
                EPS[1] / np.sqrt(N) * all_tok_dir, device=Hp.device, dtype=torch.float32)).to(Hp.dtype),
        }
        names = list(ctrl_rows)
        stack = torch.stack([Hp[0]] + [ctrl_rows[n] for n in names], dim=0)
        a_ctrl = runner.actions(ctx, stack, torch.from_numpy(np.repeat(noises[NOISE_SEEDS[0]], stack.shape[0], axis=0)))
        dist_ctrl = pm.action_distance_batch(a_ctrl[1:], a_ctrl[0])
        controls = {n: {k: float(v[i]) for k, v in dist_ctrl.items()} for i, n in enumerate(names)}
        for s in NOISE_SEEDS[1:]:
            a_n = runner.actions(ctx, Hp, torch.from_numpy(noises[s]))[0]
            controls[f"flow_noise_seed_{s}"] = pm.action_distance(a_n, a0_single)

        # ---------------- summaries
        mi_l2 = METRICS.index("l2_norm_all7")
        mid = EPS.index(0.812)
        s_mid = S_norm[mid].mean(axis=0)[:, mi_l2]                  # [N] mean over directions
        raw_mid = S[mid].mean(axis=0)[:, mi_l2]
        summary = {
            "state_id": sid, "suite": m["suite"], "task_id": m["task_id"], "phase": m["phase"],
            "episode_idx": m["episode_idx"], "step": m["step"], "instruction": m["instruction"],
            "n_tokens": N, "feature_dim": D, "epsilons": EPS, "n_directions": N_DIRS,
            "inbatch_baseline_vs_B1_maxabs": float(np.max(inbatch_base_consistency)),
            "effective_delta_norm_rel_error_max": float(np.max(np.abs(
                delta_norms / np.asarray(EPS)[:, None, None] - 1.0))),
            "baseline_action_absmean_first7": float(np.mean(np.abs(a0_single[:, :7]))),
            "S_stats_eps_mid_l2": pc.stats(s_mid),
            "raw_deviation_stats_eps_mid_l2": pc.stats(raw_mid),
            "raw_deviation_stats_eps_mid_mm": pc.stats(S[mid].mean(axis=0)[:, METRICS.index("translation_mm_equiv")]),
            "raw_deviation_stats_eps_large_l2": pc.stats(S[2].mean(axis=0)[:, mi_l2]),
            "p95_over_median_eps_mid": float(np.percentile(s_mid, 95) / max(np.median(s_mid), 1e-30)),
            "max_over_median_eps_mid": float(np.max(s_mid) / max(np.median(s_mid), 1e-30)),
            "top10_tokens_eps_mid": np.argsort(-s_mid)[:10].tolist(),
            "rank_stability_across_eps": {
                "small_vs_mid": spearman(S_norm[0].mean(axis=0)[:, mi_l2], s_mid),
                "mid_vs_large": spearman(s_mid, S_norm[2].mean(axis=0)[:, mi_l2]),
                "small_vs_large": spearman(S_norm[0].mean(axis=0)[:, mi_l2], S_norm[2].mean(axis=0)[:, mi_l2]),
            },
            "rank_stability_across_directions_eps_mid": {
                f"d{i}_vs_d{j}": spearman(S_norm[mid, i, :, mi_l2], S_norm[mid, j, :, mi_l2])
                for i in range(N_DIRS) for j in range(i + 1, N_DIRS)
            },
            "linearity_ratio_mid_over_small": float(np.median(
                S[mid].mean(axis=0)[:, mi_l2] / np.maximum(S[0].mean(axis=0)[:, mi_l2], 1e-30))),
            "linearity_ratio_large_over_mid": float(np.median(
                S[2].mean(axis=0)[:, mi_l2] / np.maximum(S[mid].mean(axis=0)[:, mi_l2], 1e-30))),
            "controls": controls,
            "wall_sec": round(time.time() - t0, 1),
        }
        np.savez_compressed(
            out_dir / f"{sid}.npz",
            S=S.astype(np.float32), S_norm=S_norm.astype(np.float32),
            delta_norms=delta_norms.astype(np.float32),
            metrics=np.asarray(METRICS), epsilons=np.asarray(EPS),
            baseline_action=a0_single.astype(np.float32),
            h_t1_pred=Hp[0].float().cpu().numpy().astype(np.float16),
            h_t=ctx.h_t[0].float().cpu().numpy().astype(np.float16),
            z=ctx.z.float().cpu().numpy().astype(np.float32),
        )
        pc.append_jsonl(rec_path, summary)
        print(f"{sid}: S_mid median={summary['S_stats_eps_mid_l2']['median']:.3e} "
              f"p95/med={summary['p95_over_median_eps_mid']:.1f} "
              f"raw_mid_max={summary['raw_deviation_stats_eps_mid_l2']['max']:.2e} "
              f"zeros={controls['future_all_zeros']['l2_norm_all7']:.4f} "
              f"noise={controls['flow_noise_seed_202']['l2_norm_all7']:.4f} "
              f"perm={controls['future_token_permuted']['l2_norm_all7']:.1e} ({summary['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
