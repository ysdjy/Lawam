# NEXT_STAGE.md

**Gap Validation is complete. The next stage is Method Design.**

Method Design is **not started**. This file records what the four rounds hand over and what must not be
re-litigated.

---

## 1. What is settled, and must not be re-opened

| question | answer | round |
|---|---|---|
| Is downstream action sensitivity `S` real and non-uniform? | yes; attention cannot replace it | Round 1 |
| Does an oracle reliability signal help locate action-critical future tokens? | yes, `U × S` > `U` | Round 1 |
| Must `U` be learned? | **no** — a learned estimator beats the free statistic only on tasks it trained on | G2 (G2-A) |
| Why does `D = ‖H_pred − h_t‖` predict `U`? | length geometry, not reliability; LaWAM over-predicts change on a mostly static token population | DS (DS-C) |
| Is `D` an uncertainty measure? | **no** — it is *predicted change magnitude*; the name is frozen | DS |
| Does the deployable `D × S` match the oracle `U × S`? | yes, in-distribution and under stress | DS, G3 |
| Does difficulty make the future channel matter more? | yes, roughly double | G3 |
| Does `D × S` survive difficulty? | yes, against `D`, `S`, attention and `D × attention` | G3 |

Closed lines: new U, learned U, new uncertainty estimators, new explanations of D, new S definitions, new
formulas (`D²S`, `DS²`, …), new proxies, new attention variants, new stress families, state-level failure
predictors.

## 2. The numbers Method Design inherits

| quantity | in-distribution | under stress |
|---|---|---|
| chunk motion | 85.0 mm | ~85 mm |
| whole future-error consequence | 0.90 mm | 1.87 / 1.70 mm (C1); 2.17 mm (C2 subset) |
| flow-sampler nuisance | 0.47 mm | 0.51 – 0.61 mm |
| treatment / nuisance | 1.77 | 3.15 – 3.57 |
| recovery: random / attention / D×att / U / D / S / U×S / **D×S** | 0.156 / 0.362 / 0.578 / 0.544 / 0.580 / 0.660 / 0.742 / **0.771** | 0.209 / 0.432 / 0.701 / 0.629 / 0.736 / 0.787 / 0.821 / **0.822** |
| `D × S` − `D` | +0.148 → **+0.131 mm** | +0.122 → **+0.231 mm** |

## 3. The three constraints that travel with the handover

1. **Headroom is unverified.** Nothing shows that repairing high-`D × S` tokens changes task success or
   progress; the one-chunk headroom test was not run. **This should be the first experiment of Method
   Design**, before any module is designed: repair a chunk, continue with the unmodified policy under the
   same stress, compare outcomes. If full-oracle repair moves nothing, a side-channel cannot either.
2. **`S` is offline.** ~2,300 forward passes (~86 s) per state. No `S` predictor has been trained, on
   purpose, so that a failure could never be attributed to two changed variables at once. An online or
   amortised `S` is a prerequisite for any runtime mechanism, and is itself an open problem.
3. **Deployment has no `H_real`.** Every recovery number in four rounds was measured with an *oracle
   repair* — selected tokens replaced by their true values. At deployment no true value exists, so a flag
   must be used for something else entirely (gating, abstention, re-planning, weighting). None of those is
   tested.

## 4. Design constraints already measured

- **Content, not position.** The action head consumes the 256 future tokens as an unordered bag
  (permutation changes the action by ~4e-7). No positional mechanism can work.
- **Identity must be the default.** A single future token is worth ~1e-4 in normalized action units; a gate
  that can misfire will dominate the signal it modulates.
- **Phase dependence.** In-distribution, future use concentrates in the approach phase
  (zeroed-future/nuisance ≈ 10) and nearly vanishes at pre-grasp (≈ 1.7).
- **The millimetre discipline.** Report every claim against the sampler's own noise, never as a bare
  percentage. Under stress the mechanism is worth +0.231 mm against 0.51 mm of noise on an 85 mm chunk.

## 5. Known negatives to state honestly in any write-up

- Family C (object/layout shift) was tested at three severities and does **not** increase future-prediction
  difficulty; it also produced 18/45 physically invalid episodes.
- Part of the occlusion-driven U increase is an artefact of the uniform grey patch (U 14.31 inside vs 8.67
  outside vs 7.63 ID); the camera-shift family carries the result without that artefact.
- Per-state treatment does not separate successful from failed episodes.
- Round 1's `REPORT.md` §7 carries five stale figures; `ERRATA.md` corrects them and `summary.json` is
  authoritative.

## 6. Re-usable assets

`research/action_critical_reliability_probe/` — `scripts/` (round 1: sensitivity sweep, oracle U,
attention recorder with an SDPA equivalence guard, consequence execution), `learned_u/` (episode/task
splits, bootstrap metrics), `change_sensitivity/` (geometry decomposition, 8-selector equal-budget causal
test), `practical_significance/` (stress families with physical-validity gates, pilot severity selection,
two-stage confirm). Together they reproduce everything from rollout to executed consequence without
retraining anything.
