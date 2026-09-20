"""Step 2a (model side, `lawam` env): action chunks for the 8 pre-registered equal-budget selectors.

For each of the 90 frozen holdout-C states, each selector picks 64 of 256 future tokens and those tokens
have `H_pred_j` replaced by `H_real_j` (an OFFLINE EVALUATION INSTRUMENT for ranking quality, never a
deployment proposal). The resulting future is fed to the flow head with everything else held identical.

Selectors (PROTOCOL_DS.yaml:step2_causal.selectors):
    A random (seeded)            B top-D                 C top-S                 D top-(D x S)
    E top-U (oracle)             F top-(U x S) (oracle)  G top-attention         H top-(D x attention)
plus the two references already produced in round 1: A_pred (no repair) and A_realFuture (full repair),
which are regenerated here so that all variants come from one batch and share the exact reduction order.

Nothing is trained. S and attention are READ from the frozen round-1 run and never recomputed.
Output: <run_dir>/ds_chunks/<state_id>.npz
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
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/learned_u/scripts"))
import probe_common as pc  # noqa: E402
import model_common as mc  # noqa: E402
import probe_model as pm  # noqa: E402
import g2_dataset as gd  # noqa: E402

PRIOR = REPO / "results/action_critical_reliability_probe/acr_20260919_203850"
NOISE_SEEDS = [101, 202, 303]          # frozen, never changed
BUDGET = 64                            # of 256, frozen, never changed
EPS = 1e-12


def selectors(D: np.ndarray, S: np.ndarray, U: np.ndarray, att: np.ndarray, sid: str) -> dict[str, np.ndarray]:
    seed = int.from_bytes(hashlib.sha256(f"ds_random|{sid}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return {
        "A_random": rng.standard_normal(D.shape),
        "B_top_D": D,
        "C_top_S": S,
        "D_top_D_times_S": D * S,
        "E_top_U_oracle": U,
        "F_top_U_times_S": U * S,
        "G_top_attention": att,
        "H_top_D_times_attention": D * att,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "ds_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in open(PRIOR / "states_manifest.jsonl")]
    if args.limit:
        manifest = manifest[: args.limit]

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noises = {s: pm.make_noise(s, H, Ad, 1) for s in NOISE_SEEDS}

    for m in manifest:
        sid = m["state_id"]
        out_path = out_dir / f"{sid}.npz"
        if out_path.exists():
            continue
        t0 = time.time()
        d = np.load(m["state_npz"], allow_pickle=True)
        s_npz = np.load(PRIOR / "sensitivity" / f"{sid}.npz")
        u_npz = np.load(PRIOR / "uncertainty" / f"{sid}.npz")

        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred
        N = ctx.n_tokens

        # verify the frozen features reproduce exactly before using them (cheap regression guard)
        assert np.array_equal(Hp[0].float().cpu().numpy().astype(np.float16), s_npz["h_t1_pred"]), sid

        h_t = s_npz["h_t"].astype(np.float64)
        h_pred = s_npz["h_t1_pred"].astype(np.float64)
        h_real_np = u_npz["h_real"]
        h_real = torch.as_tensor(h_real_np.astype(np.float32), device=Hp.device, dtype=Hp.dtype)

        D = np.linalg.norm(h_pred - h_t, axis=-1)                 # deployable
        U = u_npz["U_U_l2"].astype(np.float64)                    # oracle
        S = gd.load_S(sid)                                        # frozen
        att = gd.load_attention(sid)                              # frozen

        scores = selectors(D, S, U, att, sid)
        variants: dict[str, torch.Tensor] = {"A_pred": Hp[0], "A_realFuture": h_real}
        picks: dict[str, np.ndarray] = {}
        for name, sc in scores.items():
            idx = np.argsort(-sc)[:BUDGET]
            picks[name] = idx.astype(np.int32)
            f = Hp[0].clone()
            f[torch.as_tensor(idx, device=Hp.device)] = h_real[torch.as_tensor(idx, device=Hp.device)]
            variants[name] = f

        stack = torch.stack(list(variants.values()), dim=0)
        out: dict[str, np.ndarray] = {f"pick_{k}": v for k, v in picks.items()}
        out["token_budget"] = np.asarray([BUDGET])
        out["D"] = D.astype(np.float32)
        out["S"] = S.astype(np.float32)
        out["U"] = U.astype(np.float32)
        out["attention"] = att.astype(np.float32)
        for s in NOISE_SEEDS:
            a = runner.actions(ctx, stack, torch.from_numpy(np.repeat(noises[s], stack.shape[0], axis=0)))
            for i, name in enumerate(variants):
                out[f"{name}_seed{s}"] = a[i].astype(np.float32)
        np.savez_compressed(out_path, **out)

        ov = {k: len(set(picks[k].tolist()) & set(picks["F_top_U_times_S"].tolist())) for k in picks}
        print(f"{sid}: overlap with top-(UxS): " +
              " ".join(f"{k.split('_')[0]}={ov[k]}" for k in picks) + f" ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
