"""Batch-noise characterisation for the sensitivity sweep (`lawam` env).

`m1_equivalence.py` showed that (a) the fast path is bit-exact at B=1, but (b) running the same input in a
batch changes the result by ~5e-7, while (c) a single-token perturbation at eps=0.812 moves the action chunk
by only ~1e-5. A 20x signal/noise ratio is too thin to leave unmeasured, so before any S number is produced
this script answers:

  Q1  Do identical rows inside one batch give identical results?  (row-to-row spread, B = 1/2/8/32/64)
  Q2  Does the result of a row depend on its position in the batch?
  Q3  Does it depend on the batch size?
  Q4  Does the presence of perturbed rows leak into unperturbed rows?
  Q5  How large is the perturbation signal at each pre-registered epsilon, for a sample of tokens,
      compared with those floors?

Output: <run_dir>/phase0_batch_noise.json. No scientific claim; this only fixes the measurement protocol.
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

BD_RUN = pc.REPO_ROOT / "results/branch_diagnostic/bd_20260914_093902"
OBS = [("ep000", 8), ("ep011", 24), ("ep019", 56)]
BATCH_SIZES = [1, 2, 8, 32, 64]
EPS = [0.271, 0.812, 2.706]
SAMPLE_TOKENS = [0, 37, 96, 128, 200, 255]


def load_obs(ep: str, step: int) -> dict:
    d = np.load(BD_RUN / f"candidates/task02/{ep}/steps.npz")
    meta = json.loads((BD_RUN / f"candidates/task02/{ep}/episode.json").read_text())
    return {"primary": d["primary"][step], "wrist": d["wrist"][step], "state": d["state"][step],
            "lang": meta["instruction"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(pc.run_dir()))
    args = ap.parse_args()
    out_dir = Path(args.run_dir)

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noise1 = pm.make_noise(101, H, Ad, 1)

    out: dict = {"run_id": out_dir.name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "states": []}
    for ep, step in OBS:
        sid = f"t02_{ep}@{step}"
        o = load_obs(ep, step)
        ex = mc.make_example(o["primary"], o["wrist"], o["state"], o["lang"], norm_stats)
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred
        rec: dict = {"state": sid}

        # ---- Q1/Q3: identical rows, several batch sizes
        per_bs = {}
        a_single = None
        for bs in BATCH_SIZES:
            fut = Hp.expand(bs, -1, -1).contiguous()
            a = runner.actions(ctx, fut, torch.from_numpy(np.repeat(noise1, bs, axis=0)))
            if bs == 1:
                a_single = a[0]
            row_spread = float(np.max(np.abs(a - a[0][None]))) if bs > 1 else 0.0
            per_bs[f"B{bs}"] = {
                "row_to_row_maxabs": row_spread,
                "vs_B1_maxabs": float(np.max(np.abs(a - a_single[None]))),
            }
        rec["Q1_Q3_identical_rows"] = per_bs

        # ---- Q2/Q4: one perturbed row placed at different positions of a B=32 batch;
        #             all other rows are the unperturbed future.
        bs = 32
        eps_probe = 0.812
        dirs = pm.unit_directions(sid, SAMPLE_TOKENS[1], 1, ctx.feature_dim)
        delta = torch.as_tensor(eps_probe * dirs[0], device=Hp.device, dtype=torch.float32)
        leak = {}
        for pos in (0, 1, 16, 31):
            fut = Hp.expand(bs, -1, -1).clone()
            fut[pos, SAMPLE_TOKENS[1]] = (fut[pos, SAMPLE_TOKENS[1]].float() + delta).to(fut.dtype)
            a = runner.actions(ctx, fut, torch.from_numpy(np.repeat(noise1, bs, axis=0)))
            others = np.delete(a, pos, axis=0)
            leak[f"pos{pos}"] = {
                "unperturbed_rows_row_to_row_maxabs": float(np.max(np.abs(others - others[0][None]))),
                "unperturbed_rows_vs_B1_maxabs": float(np.max(np.abs(others - a_single[None]))),
                "perturbed_row_change_vs_inbatch_baseline": float(np.max(np.abs(a[pos] - others[0]))),
                "perturbed_row_change_vs_B1": float(np.max(np.abs(a[pos] - a_single))),
            }
        rec["Q2_Q4_position_and_leak"] = leak

        # ---- Q5: signal magnitude per epsilon for a sample of tokens (one batch per epsilon,
        #          row 0 = unperturbed in-batch baseline)
        sig = {}
        for eps in EPS:
            rows = [Hp[0]]
            eff_norms = []
            for tok in SAMPLE_TOKENS:
                d = torch.as_tensor(eps * pm.unit_directions(sid, tok, 1, ctx.feature_dim)[0],
                                    device=Hp.device, dtype=torch.float32)
                f = Hp[0].clone()
                f[tok] = (f[tok].float() + d).to(f.dtype)
                rows.append(f)
                eff_norms.append(float(torch.linalg.norm((f[tok].float() - Hp[0, tok].float()))))
            fut = torch.stack(rows, dim=0)
            a = runner.actions(ctx, fut, torch.from_numpy(np.repeat(noise1, fut.shape[0], axis=0)))
            dist = pm.action_distance_batch(a[1:], a[0])
            sig[f"eps{eps}"] = {
                "batch_size": int(fut.shape[0]),
                "tokens": SAMPLE_TOKENS,
                "effective_delta_norm": eff_norms,
                "l2_norm_all7": dist["l2_norm_all7"].tolist(),
                "maxabs_all7": dist["maxabs_all7"].tolist(),
                "translation_mm_equiv": dist["translation_mm_equiv"].tolist(),
                "inbatch_baseline_vs_B1_maxabs": float(np.max(np.abs(a[0] - a_single))),
            }
        rec["Q5_signal_vs_epsilon"] = sig

        # ---- reference scales measured the same way: whole-future interventions
        refs = {}
        zeros = torch.zeros_like(Hp[0])
        scaled = Hp[0].float() * 1.05
        perm = Hp[0][torch.randperm(ctx.n_tokens, generator=torch.Generator().manual_seed(7))]
        all_tok = Hp[0].float() + torch.as_tensor(
            np.random.default_rng(3).standard_normal((ctx.n_tokens, ctx.feature_dim)) /
            np.sqrt(ctx.feature_dim) * 0.812, device=Hp.device, dtype=torch.float32)
        stack = torch.stack([Hp[0], zeros, scaled.to(Hp.dtype), perm, all_tok.to(Hp.dtype), ctx.h_t[0]], dim=0)
        a = runner.actions(ctx, stack, torch.from_numpy(np.repeat(noise1, stack.shape[0], axis=0)))
        dist = pm.action_distance_batch(a[1:], a[0])
        for i, name in enumerate(["future_all_zeros", "future_scaled_1.05", "future_token_permuted",
                                  "all_tokens_perturbed_eps0.812", "future_replaced_by_h_t"]):
            refs[name] = {k: float(v[i]) for k, v in dist.items()}
        # flow-noise nuisance, measured on this state
        a_n2 = runner.actions(ctx, Hp, torch.from_numpy(pm.make_noise(202, H, Ad, 1)))[0]
        refs["different_flow_noise_seed"] = pm.action_distance(a_n2, a_single)
        rec["reference_scales"] = refs

        out["states"].append(rec)
        print(f"{sid}: identical-row spread B32={per_bs['B32']['row_to_row_maxabs']:.3g}; "
              f"eps0.812 tokens maxabs={np.max(sig['eps0.812']['maxabs_all7']):.3g}; "
              f"zeros={refs['future_all_zeros']['maxabs_all7']:.3g}; "
              f"noise-seed={refs['different_flow_noise_seed']['maxabs_all7']:.3g}", flush=True)

    pc.write_json(out_dir / "phase0_batch_noise.json", out)


if __name__ == "__main__":
    main()
