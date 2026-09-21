"""Offline gate: teacher-action prediction on the TEST episodes under all five conditions.

  ORIGINAL    unmodified LaWAM (no context token; LoRA residual zeroed)
  C_P         control arm, context input 0
  H_CORRECT   hidden arm, true physics context
  H_WRONG     hidden arm, swapped context (NOMINAL <-> HIGH)
  H_SHUFFLED  hidden arm, episode-level shuffled context (marginal preserved, pairing destroyed)

Usage: python h5_offline.py <run_id>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsu_model as hm  # noqa: E402
import h4_train as h4  # noqa: E402

RUN = sys.argv[1]
OUT = hm.REPO / "results" / "hidden_state_utilization_probe" / RUN
MM_PER_UNIT = 46.875      # (max-min)/2 * 0.05 m/unit * 1000 -> mm per normalized unit per step
BATCH = 8


def load_adapter(arm_module, flow, path: Path | None):
    """Load a trained arm, or reset to the exact base policy when path is None."""
    if path is None:
        with torch.no_grad():
            arm_module.projector.alpha.zero_()
            for m in flow.modules():
                if isinstance(m, hm.LoRALinear):
                    m.B.weight.zero_()
        return {"loaded": "ORIGINAL (LoRA B = 0, alpha = 0)"}
    ck = torch.load(path, map_location="cpu")
    arm_module.projector.load_state_dict(ck["projector"])
    named = dict(flow.named_parameters())
    with torch.no_grad():
        for n, v in ck["lora"].items():
            named[n].copy_(v.to(named[n].dtype).to(named[n].device))
    return {"loaded": str(path), "arm": ck["arm"], "update": ck["update"],
            "val_loss": ck["val_loss"], "trainable_params": ck["trainable_params"]}


def main() -> None:
    torch.manual_seed(0)
    vla, backend = hm.load_backend(use_bf16=True)
    backend.eval()
    flow = backend.flow
    hm.freeze_base(backend)
    hm.install_lora(flow, hm.LORA_RANK)
    dev = next(flow.parameters()).device
    dt = next(flow.parameters()).dtype
    arm = hm.ContextArm(flow, control=False).to(dev)
    arm.projector.to(torch.float32)
    for m in flow.modules():
        if isinstance(m, hm.LoRALinear):
            m.A.to(torch.float32)
            m.B.to(torch.float32)

    test = h4.load_split("test")
    episodes = sorted({s["episode"] for s in test})
    rng = np.random.default_rng(123)
    ep_c = {e: (0.0 if e.startswith("NOMINAL") else float(np.log(2.0))) for e in episodes}
    original_vals = np.array([ep_c[e] for e in episodes])
    # With just two levels and ten test episodes, one random permutation can leave almost every
    # context correct. Fix the seed and require a meaningful episode-level mismatch before scoring.
    while True:
        shuffled_vals = rng.permutation(original_vals)
        if np.count_nonzero(shuffled_vals != original_vals) >= 6:
            break
    ep_shuffled = dict(zip(episodes, shuffled_vals))

    conditions = [
        ("ORIGINAL", None, "correct"),
        ("C_P", OUT / "adapter_control.pt", "zero"),
        ("H_CORRECT", OUT / "adapter_hidden.pt", "correct"),
        ("H_WRONG", OUT / "adapter_hidden.pt", "wrong"),
        ("H_SHUFFLED", OUT / "adapter_hidden.pt", "shuffled"),
    ]
    rows, meta = [], {}
    for name, path, ctx_mode in conditions:
        meta[name] = load_adapter(arm, flow, path)
        arm.control = (ctx_mode == "zero")
        for i in range(0, len(test), BATCH):
            b = test[i:i + BATCH]
            h_t, h_pred, h_vlm, attn, actions, amask, c, hz, emb = h4.to_batch(b, flow, dev, dt)
            if ctx_mode == "wrong":
                c = torch.tensor([[0.0 if abs(s["c"]) > 1e-9 else float(np.log(2.0))] for s in b],
                                 device=dev, dtype=torch.float32)
            elif ctx_mode == "shuffled":
                c = torch.tensor([[ep_shuffled[s["episode"]]] for s in b], device=dev, dtype=torch.float32)
            noise = torch.randn(len(b), int(flow.action_horizon), int(flow.config.action_dim),
                                generator=torch.Generator().manual_seed(5000 + i)).numpy()
            hv, am = (h_vlm, attn) if name == "ORIGINAL" else arm.condition(h_vlm, attn, c)
            pred = hm.flow_sample(flow, h_t, h_pred, hv, am, hz, emb, noise).float().cpu().numpy()
            tgt = np.stack([s["norm_chunk"] for s in b])
            n = min(pred.shape[1], tgt.shape[1])
            d = pred[:, :n, :7] - tgt[:, :n, :7]
            for j, s in enumerate(b):
                rows.append({"condition": name, "level": s["level"], "episode": s["episode"],
                             "idx": s["idx"], "c_used": float(c[j, 0]),
                             "l2_per_element": float(np.sqrt((d[j] ** 2).mean())),
                             "l1_translation_mm": float(np.abs(d[j][:, :3]).sum() * MM_PER_UNIT / 3),
                             "maxabs": float(np.abs(d[j]).max()),
                             "gripper_absdiff": float(np.abs(d[j][:, 6]).max())})
        print(f"{name}: {meta[name]}", flush=True)

    with open(OUT / "OFFLINE_METRICS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import statistics as st
    summ = {}
    for name, _, _ in conditions:
        r = [x for x in rows if x["condition"] == name]
        summ[name] = {"all": round(st.mean(x["l2_per_element"] for x in r), 6),
                      "mm_all": round(st.mean(x["l1_translation_mm"] for x in r), 3)}
        for lv in ("NOMINAL", "HIGH"):
            rr = [x for x in r if x["level"] == lv]
            summ[name][lv] = round(st.mean(x["l2_per_element"] for x in rr), 6)
            summ[name][f"mm_{lv}"] = round(st.mean(x["l1_translation_mm"] for x in rr), 3)
    gate = {
        "H_CORRECT_vs_C_P": round(summ["C_P"]["all"] - summ["H_CORRECT"]["all"], 6),
        "H_CORRECT_vs_C_P_HIGH": round(summ["C_P"]["HIGH"] - summ["H_CORRECT"]["HIGH"], 6),
        "H_CORRECT_vs_H_WRONG": round(summ["H_WRONG"]["all"] - summ["H_CORRECT"]["all"], 6),
        "H_CORRECT_vs_H_WRONG_HIGH": round(summ["H_WRONG"]["HIGH"] - summ["H_CORRECT"]["HIGH"], 6),
        "H_CORRECT_vs_H_SHUFFLED": round(summ["H_SHUFFLED"]["all"] - summ["H_CORRECT"]["all"], 6),
        "H_CORRECT_vs_ORIGINAL": round(summ["ORIGINAL"]["all"] - summ["H_CORRECT"]["all"], 6),
        "C_P_vs_ORIGINAL": round(summ["ORIGINAL"]["all"] - summ["C_P"]["all"], 6),
    }
    out = {"run_id": RUN, "n_test_samples": len(test), "episodes": episodes,
           "shuffled_map": {k: float(v) for k, v in ep_shuffled.items()},
           "adapters": meta, "mean_error": summ, "gate_deltas_positive_favours_correct": gate}
    (OUT / "OFFLINE_SUMMARY.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"mean_error": summ, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
