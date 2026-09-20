"""G3 stage C2 (model side): action chunks for the frozen selectors on the stressed states.

Selectors are exactly the frozen set — no new formula, no new proxy:
    A_random (pre-existing floor)  B_top_D  C_top_S  D_top_D_times_S
    E_top_attention  F_top_D_times_attention  G_top_U_oracle  H_top_U_times_S_oracle
Budget 64 of 256. Selected tokens get H_pred_j -> H_real_j, an OFFLINE evaluation instrument for ranking
quality only.

Everything is read, never recomputed here:
    S          c2/sensitivity/<sid>.npz          (fresh sweep, round-1 parameters)
    D, U, H_real  confirm/treatment_chunks/<cond>/<sid>.npz
    attention  c2/attention/<sid>.npz            (frozen recorder, equivalence-guarded)
Output: <c2>/c2_chunks/<state_id>.npz
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

NOISE_SEEDS = [101, 202, 303]
BUDGET = 64
EPS_MID_IDX = 1
S_METRIC = "l2_norm_all7"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2_dir", required=True)
    ap.add_argument("--confirm_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    c2 = Path(args.c2_dir)
    confirm = Path(args.confirm_dir)
    out_dir = c2 / "c2_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    states = [json.loads(l) for l in open(c2 / "states_manifest.jsonl")]
    if args.limit:
        states = states[: args.limit]

    vla, model_config, norm_stats = mc.load_policy()
    runner = pm.ProbeRunner(vla)
    H, Ad = int(runner.flow.action_horizon), int(runner.flow.config.action_dim)
    noises = {s: pm.make_noise(s, H, Ad, 1) for s in NOISE_SEEDS}

    for st in states:
        sid = st["state_id"]
        out_path = out_dir / f"{sid}.npz"
        if out_path.exists():
            continue
        t0 = time.time()
        d = np.load(st["state_npz"], allow_pickle=True)
        s_npz = np.load(c2 / "sensitivity" / f"{sid}.npz")
        t_npz = np.load(confirm / "treatment_chunks" / st["condition"] / f"{sid}.npz")
        a_npz = np.load(c2 / "attention" / f"{sid}.npz")

        ex = mc.make_example(d["primary"], d["wrist"], d["state"], str(d["lang"][0]), norm_stats)
        ctx = runner.prepare(ex)
        Hp = ctx.h_t1_pred
        # regression guard: the freshly computed future must equal both frozen copies
        assert np.array_equal(Hp[0].float().cpu().numpy().astype(np.float16), s_npz["h_t1_pred"]), sid
        assert np.array_equal(s_npz["h_t1_pred"], t_npz["h_t1_pred"]), sid

        metrics = [str(x) for x in s_npz["metrics"]]
        S = s_npz["S_norm"][EPS_MID_IDX].mean(axis=0)[:, metrics.index(S_METRIC)].astype(np.float64)
        D = t_npz["D"].astype(np.float64)
        U = t_npz["U_l2"].astype(np.float64)
        att = a_npz["attention_future"].astype(np.float64)
        h_real = torch.as_tensor(t_npz["h_real"].astype(np.float32), device=Hp.device, dtype=Hp.dtype)

        seed = int.from_bytes(hashlib.sha256(f"g3c2_random|{sid}".encode()).digest()[:8], "big")
        scores = {
            "A_random": np.random.default_rng(seed).standard_normal(D.shape),
            "B_top_D": D, "C_top_S": S, "D_top_D_times_S": D * S,
            "E_top_attention": att, "F_top_D_times_attention": D * att,
            "G_top_U_oracle": U, "H_top_U_times_S_oracle": U * S,
        }
        variants: dict[str, torch.Tensor] = {"A_pred": Hp[0], "A_realFuture": h_real}
        out: dict[str, np.ndarray] = {"D": D.astype(np.float32), "S": S.astype(np.float32),
                                      "U": U.astype(np.float32), "attention": att.astype(np.float32),
                                      "token_budget": np.asarray([BUDGET])}
        for name, sc_ in scores.items():
            idx = np.argsort(-sc_)[:BUDGET]
            out[f"pick_{name}"] = idx.astype(np.int32)
            f = Hp[0].clone()
            ti = torch.as_tensor(idx, device=Hp.device)
            f[ti] = h_real[ti]
            variants[name] = f

        stack = torch.stack(list(variants.values()), dim=0)
        for s in NOISE_SEEDS:
            a = runner.actions(ctx, stack, torch.from_numpy(np.repeat(noises[s], stack.shape[0], axis=0)))
            for i, name in enumerate(variants):
                out[f"{name}_seed{s}"] = a[i].astype(np.float32)
        np.savez_compressed(out_path, **out)
        ov = len(set(out["pick_D_top_D_times_S"].tolist()) & set(out["pick_H_top_U_times_S_oracle"].tolist()))
        print(f"{sid}: overlap(DxS, UxS)={ov}/{BUDGET} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
