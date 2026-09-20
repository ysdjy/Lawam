# LONG_TERM_PLAN.md — Action-Critical Reliability (PLAN ONLY, nothing here was implemented)

Status: **not started, and not yet authorised by the evidence.** `REPORT.md` §12 records SUPPORTED for the
mechanism but a future-utilisation budget (≈0.9 mm of executed motion, ≈1.9× sampler noise) too small to
justify building anything. The gating experiments are `REPORT.md` §13 items 1–3. This file exists so that
the design is written down, not so that it is executed.

---

## 1. What the probe licenses and what it does not

Licensed by measurement (90 states, 3 tasks, 10 independent resets each):

- U (future-prediction unreliability) and S (downstream action sensitivity) are **independent** within a
  state (Spearman ≈ 0.02) and both are real, above-floor measurements.
- **U × S** is the best available token ranking for "which part of the predicted future actually matters to
  the action": Spearman 0.877 vs 0.850 (S), 0.238 (U), 0.281 (attention); and in a causal equal-budget
  correction it recovers 74 % of the oracle effect versus 55 % for U.
- Attention mass is **not** a proxy for sensitivity (42 % of image-half attention sits on future tokens that
  barely affect the action), so an attention-based side-channel is not the cheap substitute it appears to be.

Not licensed:

- That any of this improves control. The entire prediction error is worth ~1 % of the chunk's motion.
- That a *learned* U preserves the advantage (everything here used an oracle future).
- That U×S identifies risky **states** — it does not; mean U is as good at that.
- Anything about OOD, occlusion, distractors, or failure prediction: the sample contains no failures.

## 2. Gating experiments (must pass before any module is built)

| # | Experiment | Decision rule |
|---|---|---|
| G1 | Re-run this exact protocol on a checkpoint trained with `enable_flow_h_t1_scheduled_sampling=true` | treatment/nuisance ratio must rise materially above the ≈1.9 measured here; otherwise stop — there is nothing to gate |
| G2 | Replace oracle U with a learnable estimator (LaWM decoder ensemble / MC-dropout disagreement, or a head on (h_t, z)); repeat the equal-budget causal test | the paired advantage over top-U selection must survive with a learned U |
| G3 | Add a token-norm-only selector and failure-containing states (perturbed observations, distractors) | U×S must beat the norm-only baseline, and must show signal against actual failures |

If G1 fails, the correct research question becomes **Future Utilization / Conditioning Effectiveness** (why
the action head barely uses a 256-token future channel it attends to 42 % of the time), not reliability.

## 3. Architecture sketch (only if G1–G3 pass)

```
LaWM ──► H_future [256, 768]
            │
            ├─► Reliability Estimator ──────► U_hat  [256]      (learned, no oracle at test time)
            ├─► Sensitivity / Risk Encoder ─► S_hat  [256]      (amortised; finite differences are too slow online)
            │        └─► R = f(U_hat, S_hat), start with the plain product (the fitted linear
            │            combination did not beat it in the probe)
            ▼
     Risk Side Tokens  ──► Gated Adapter / Cross-Attention ──► Flow Action Head
```

Constraints that follow directly from the probe's measurements:

- **Content, not position.** The action head consumes H_future as an *unordered bag* (permutation changes
  the action by ~0). A side-channel must therefore modulate token *content* (e.g. gating values/keys) — a
  positional mask or an index-based embedding carries no information.
- **The gate must be able to do nothing.** With a per-token influence of ~1e-4, an unconstrained side-channel
  can easily dominate the signal it is supposed to modulate; the default must be identity.
- **Respect the phase structure.** Future use concentrates in approach (10× the sampler noise) and nearly
  vanishes at pre-grasp (1.7×); a global always-on gate would be mistuned for two thirds of the trajectory.
- **Compare against sampler noise, always.** Any reported improvement must be shown against the ≈0.47 mm
  run-to-run variation of the flow sampler, which is half the size of the whole effect being modulated.

## 4. Comparison set for that phase

1. original LaWAM, 2. attention-only gating, 3. U-only, 4. S-only, 5. U×attention, 6. U×S,
7. learned risk head, 8. NoiseGate-style gating, 9. future-free baseline (the probe shows this is only
≈1.4e-2 away from the full model in action units — it is a serious baseline, not a strawman).

Evaluation axes: OOD, occlusion, noisy observation, dynamic distractors, contact phase, unseen
object/layout — each reported against the sampler-noise floor, with episode-level bootstrap CIs.

## 5. Re-usable assets from this round

`research/action_critical_reliability_probe/` — `probe_model.py` (batched flow replay proven bit-identical
to `predict_action`, attention recorder with an equivalence guard), `m2_sensitivity.py` (token sweep),
`m3_uncertainty.py` (oracle U + error decomposition), `m5_consequence_chunks.py` / `s4_execute_consequences.py`
(equal-budget causal test), and the pre-registration + audit discipline in `configs/protocol.yaml` and
`AUDIT_LOG.md`.
