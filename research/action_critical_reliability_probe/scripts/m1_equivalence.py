"""Phase-0 equivalence / regression tests for the probe machinery (`lawam` env).

Checks, on real observations, before any sensitivity measurement:
  E1  shared-encoding + batched flow replay (B=1)  ==  untouched `vla.predict_action`
  E2  batched replay (B=2,8,32)                    ~=  B=1 replay, per row (see note)
  E3  repeated identical replay                    ==  0 difference (numerical floor)
  E4  unperturbed rows of a mixed batch            ==  the in-batch baseline row (no leak between rows)

NOTE (protocol amendment, see AUDIT_LOG "AUDIT 0b"): `m1b_batch_noise.py` established that identical rows
inside one batch produce *bit-identical* actions (row-to-row spread exactly 0.0, independent of row position
and batch size), while the same input evaluated at a *different batch size* shifts by ~5e-7 (cuBLAS/SDPA
reduction order). Since a single-token perturbation only moves the action chunk by ~1e-6..1e-4, every
sensitivity measurement must use a baseline row inside the same batch; E2's cross-batch-size difference is
therefore recorded as a documented property with tolerance 1e-5, and E4 compares against the in-batch
baseline with tolerance 0.0.
  E5  bf16 storage of the future: the effective perturbation actually applied to the tensor is measured,
      so that S is normalised by the delta that the model really saw, not by the requested epsilon
  E6  validation still rejects wrong shapes / non-finite overrides

Output: <run_dir>/phase0_equivalence.json. Exit code 1 if any hard check fails.
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
import probe_common as pc  # noqa: E402  (must come first: it puts branch_diagnostic/scripts on sys.path)
import model_common as mc  # noqa: E402
import probe_model as pm  # noqa: E402

BD_RUN = pc.REPO_ROOT / "results/branch_diagnostic/bd_20260914_093902"
OBS = [("ep000", 8), ("ep011", 24), ("ep019", 56)]
BATCHES = [2, 8, 32]
TOL_EXACT = 0.0
TOL_BATCH = 1e-5   # cross-batch-size reduction-order offset; in-batch comparisons stay at TOL_EXACT


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
    H = int(runner.flow.action_horizon)
    Ad = int(runner.flow.config.action_dim)

    results: list[dict] = []
    failures: list[str] = []
    for ep, step in OBS:
        o = load_obs(ep, step)
        ex = mc.make_example(o["primary"], o["wrist"], o["state"], o["lang"], norm_stats)
        noise1 = pm.make_noise(101, H, Ad, 1)
        sid = f"t02_{ep}@{step}"

        # reference: the untouched public path
        ref = vla.predict_action(examples=[ex], initial_noise=noise1, return_diagnostics=True)
        a_ref = np.asarray(ref["normalized_actions"])[0]
        h_pred_ref = np.asarray(ref["diagnostics"]["h_t1_pred"])

        ctx = runner.prepare(ex)
        h_pred = ctx.h_t1_pred
        # E1
        a_fast = runner.actions(ctx, h_pred, torch.from_numpy(noise1))[0]
        e1 = float(np.max(np.abs(a_fast - a_ref)))
        # shared encoding must also reproduce the public diagnostics
        e1_feat = float(np.max(np.abs(h_pred.detach().float().cpu().numpy() - h_pred_ref)))

        # E2: batched rows must equal the single-row result
        e2 = {}
        for b in BATCHES:
            fut = h_pred.expand(b, -1, -1).contiguous()
            noise_b = np.repeat(noise1, b, axis=0)
            a_b = runner.actions(ctx, fut, torch.from_numpy(noise_b))
            e2[f"B{b}"] = float(np.max(np.abs(a_b - a_fast[None])))

        # E3: repeat floor of the fast path
        reps = [runner.actions(ctx, h_pred, torch.from_numpy(noise1))[0] for _ in range(3)]
        e3 = max(float(np.max(np.abs(reps[i] - reps[j]))) for i in range(3) for j in range(i + 1, 3))

        # E4: in a batch mixing unperturbed and perturbed rows, the unperturbed rows must remain identical
        # to each other (the in-batch baseline). Row 0 and row 2 are unperturbed; rows 1 and 3 are perturbed.
        eps = 0.812
        dirs = pm.unit_directions(sid, 37, 2, ctx.feature_dim)
        fut_mix = h_pred.expand(4, -1, -1).clone()
        for row, k in ((1, 0), (3, 1)):
            fut_mix[row, 37] = (fut_mix[row, 37].float() + torch.as_tensor(
                eps * dirs[k], device=fut_mix.device, dtype=torch.float32)).to(fut_mix.dtype)
        a_mix = runner.actions(ctx, fut_mix, torch.from_numpy(np.repeat(noise1, 4, axis=0)))
        e4 = float(np.max(np.abs(a_mix[0] - a_mix[2])))            # in-batch baseline consistency
        e4_vs_b1 = float(np.max(np.abs(a_mix[0] - a_fast)))        # documented cross-batch-size offset
        e4_perturbed_change = [float(np.max(np.abs(a_mix[r] - a_mix[0]))) for r in (1, 3)]

        # E5: what perturbation did the (bf16) tensor actually receive?
        eff = []
        for eps_val in (0.271, 0.812, 2.706):
            d0 = torch.as_tensor(eps_val * dirs[0], device=h_pred.device, dtype=torch.float32)
            perturbed = (h_pred[0, 0].float() + d0).to(h_pred.dtype)        # what the model stores
            delta_eff = (perturbed.float() - h_pred[0, 0].float())
            eff.append({
                "eps_requested": eps_val,
                "eps_effective_norm": float(torch.linalg.norm(delta_eff)),
                "relative_error": float(abs(torch.linalg.norm(delta_eff).item() - eps_val) / eps_val),
            })

        results.append({
            "state": sid,
            "E1_fastpath_vs_predict_action_maxabs": e1,
            "E1_shared_h_t1_pred_maxabs": e1_feat,
            "E2_batched_vs_single_maxabs": e2,
            "E3_repeat_floor_maxabs": e3,
            "E4_inbatch_baseline_rows_agree_maxabs": e4,
            "E4_inbatch_baseline_vs_B1_maxabs": e4_vs_b1,
            "E4_perturbed_rows_change_vs_inbatch_baseline": e4_perturbed_change,
            "E5_bf16_effective_delta": eff,
            "future_dtype": str(h_pred.dtype),
            "h_t1_pred_shape": list(h_pred.shape),
        })
        if e1 > TOL_EXACT:
            failures.append(f"{sid}: E1 fast path differs from predict_action by {e1}")
        if e1_feat > TOL_EXACT:
            failures.append(f"{sid}: E1 shared h_t1_pred differs by {e1_feat}")
        if max(e2.values()) > TOL_BATCH:
            failures.append(f"{sid}: E2 batching changes the result by {max(e2.values())}")
        if e3 > TOL_EXACT:
            failures.append(f"{sid}: E3 repeat floor is {e3}, not 0")
        if e4 > TOL_EXACT:
            failures.append(f"{sid}: E4 in-batch baseline rows disagree by {e4}")
        print(f"{sid}: E1={e1:.3g} E2={max(e2.values()):.3g} E3={e3:.3g} E4={e4:.3g} "
              f"(perturbed rows moved {e4_perturbed_change})", flush=True)

    # E6: validation still rejects malformed overrides (on the public path)
    o = load_obs(*OBS[0])
    ex = mc.make_example(o["primary"], o["wrist"], o["state"], o["lang"], norm_stats)
    e6 = {}
    for name, kwargs in {
        "wrong_future_shape": {"future_override": torch.zeros(1, 128, 768)},
        "nonfinite_future": {"future_override": torch.full((1, 256, 768), float("nan"))},
        "wrong_noise_shape": {"initial_noise": np.zeros((1, 7, 32), dtype=np.float32)},
    }.items():
        try:
            vla.predict_action(examples=[ex], **kwargs)
            e6[name] = "NOT RAISED"
            failures.append(f"E6 {name}: no exception raised")
        except (ValueError, TypeError) as exc:
            e6[name] = f"{type(exc).__name__}: {str(exc)[:90]}"

    payload = {
        "run_id": out_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tolerances": {"exact": TOL_EXACT, "batch": TOL_BATCH},
        "per_state": results,
        "E6_validation": e6,
        "failures": failures,
        "passed": not failures,
    }
    pc.write_json(out_dir / "phase0_equivalence.json", payload)
    print(json.dumps({"passed": payload["passed"], "failures": failures}, indent=1))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
