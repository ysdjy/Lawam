# REPORT_G3.md — Practical Significance Validation (the final Gap-Validation gate)

run `acr_g3_practical_20260920_134247` · 2026-09-20 · branch `research/action-critical-reliability-probe` @ `ec7dd4b`
LaWAM `state_dict` sha256 `37a53b8c…b523`, verified unchanged · **nothing was trained in this round**

---

## 1. Why G3 Is the Final Gap-Validation Gate

Three earlier rounds settled the *mechanism* and left one question open. Round 1 showed that oracle
`U × S` locates action-critical future tokens better than `U` alone; G2 showed a learned `U` is unnecessary
because the free statistic `D = ‖H_pred − h_t‖` predicts it; the DS round showed that `D × S` matches the
oracle `U × S` at equal token budget. But every one of those results lived inside a tiny budget of
influence: in-distribution, the *entire* future-prediction error was worth **0.90 mm** of executed
end-effector motion against **0.47 mm** of flow-sampler noise on an **85 mm** chunk, and `D × S` beat `D` by
only **+0.131 mm**. Unless difficulty changes that, no amount of ranking quality matters. G3 asks exactly
that, and nothing else.

## 2. Frozen Findings from Previous Rounds

| round | frozen result | not reopened here |
|---|---|---|
| Round 1 | `S` is real and highly non-uniform; attention cannot replace it; oracle `U × S` > `U` | new S definitions |
| G2 (Case G2-A) | a learned `U` beats the free statistic only on tasks it trained on | learned U, new uncertainty estimators |
| DS (Case DS-C) | `D → U` is length geometry, not reliability; `D` is *predicted change magnitude*; `D × S` (0.771) ≈ oracle `U × S` (0.742) | new explanations of D, new formulas, new proxies |

## 3. Practical Question

> In genuinely difficult, failure-producing conditions, does LaWAM's future prediction error grow large
> enough to matter for control, and does `D × S` still locate the action-relevant future information better
> than simple alternatives?

## 4. Stress Protocol

Three families × three severities, all frozen in `PROTOCOL_G3.yaml` **before the first rollout** and
grounded in the token geometry (256×256 image, 16×16 grid of 16 px patches = one patch per future token):

| family | mild | medium | strong |
|---|---|---|---|
| A primary-camera occlusion (grey 128 rectangle, wrist untouched, centred on the target object's projection at the first query and then frozen) | 48 px = 9 tokens | 80 px = 25 tokens | 112 px = 49 tokens |
| B camera-like image shift (fixed diagonal, edge padding, wrist untouched) | 6 px | 12 px | 24 px |
| C object/layout shift (target object xy at init, three physical-validity checks) | 2 cm | 4 cm | 6 cm |

Two definitions that make the measurement honest: the perturbation is applied **where frames are
recorded**, so `o_t` and `o_{t+7}` share it by construction and the U label cannot be misaligned; and
`H_real` is the encoding of the **equally perturbed** `o_{t+7}`, because the perturbation is part of the
observation process.

## 5. Pilot Severity Selection

150 rollout attempts (3 suites × 1 task × 5 resets × 10 conditions); 132 executed, 18 invalidated by the
family-C physical gate and **recorded, never replaced**; 393 states; 2,358 executed chunks.

| condition | success | treatment mm | nuisance mm | t/n | vs ID | U |
|---|---|---|---|---|---|---|
| ID | 100 % | 0.83 | 0.46 | 1.84 | 0.91 | 0.087 |
| A mild | 100 % | 1.06 | 0.44 | 2.34 | 1.17 | 0.126 |
| **A medium** | **87 %** | **2.04** | 0.53 | **3.69** | 2.26 | 0.134 |
| A strong | 93 % | 2.16 | 0.70 | 3.26 | 2.39 | 0.161 |
| B mild | 100 % | 0.88 | 0.46 | 2.04 | 0.97 | 0.115 |
| B medium | 93 % | 1.09 | 0.47 | 2.05 | 1.21 | 0.147 |
| **B strong** | 93 % | **1.57** | 0.50 | **3.22** | 1.73 | 0.201 |
| C mild / medium / strong | 100 / 100 / 88 % | 0.84 / 1.16 / 1.04 | ~0.5 | ~2 | ~1.2 | 0.093–0.097 |

The ID condition reproduced the frozen reference on fresh episodes (0.83 vs 0.904 mm; 0.46 vs 0.474).
Severities were chosen by the pre-registered rule inside a script that has **no access to any D, S or
`D × S` quantity**: A → medium (lightest in the 20–90 % success window), B → strong (fallback: max t/n with
success ≥ 20 %), C → strong.

**Family C: tested but not informative for the target mechanism.** It does not increase future-prediction
difficulty at all — U stays at 0.093–0.097 against an ID value of 0.087, and `D` is unchanged — because
moving an object to another valid position does not make the *future* harder to predict. It also produced
18 of 45 physically invalid episodes, including **all 15 `libero_object` episodes** (`contact:floor`). It
was excluded from the confirm stage on those grounds (amendment G3-A1), not repaired, and not replaced by a
new family. Its complete pilot results are retained above and in `PILOT_RESULTS.csv`.

## 6. Baseline Success / Failure Distribution

Confirm C1: 3 suites × 2 tasks × 10 independent resets × {ID, A, B} = **180 rollouts**.
Success 95 % (ID), 78 % (occlusion), 68 % (shift). **All 35 failures are 250-step timeouts**; zero crashes,
zero infrastructure errors, no episode excluded. The stress bites unevenly across tasks (shift: `goal t5`
1/10, `object t6` 5/10, but `spatial t2` and `t7` still 10/10) — a difficulty gradient, not a collapse.

## 7. Future Prediction Error under Stress

| condition | U all tokens | U occluded | U non-occluded | D | G |
|---|---|---|---|---|---|
| ID | 7.63 | — | 7.63 | 7.02 | 2.73 |
| A occlusion 80 px | 9.15 | **14.31** | 8.67 | 7.30 | 2.56 |
| B camera shift 24 px | **11.15** | — | 11.15 | **10.13** | 2.91 |

The grey rectangle inflates U inside itself by construction, so **U was barred from being primary
evidence** (amendment G3-A4). Outside the rectangle U still rises 13.6 % above ID, and family B — which has
no occluder at all — shows the largest latent degradation. No new occluder was designed in response.

## 8. Future Error → Action Consequence

The primary evidence: executed end-effector displacement between π(H_real) and π(H_pred) at fixed flow
noise from identical snapshots, 60 episodes per condition, 477 states, 2,862 executed chunks.

| condition | success | treatment mm (median / mean) | mean CI | nuisance mm | treatment / nuisance |
|---|---|---|---|---|---|
| ID | 95 % | 0.82 / 1.00 | [0.90, 1.12] | 0.44 | **1.77** |
| A occlusion | 78 % | **1.87 / 2.23** | [2.01, 2.45] | 0.61 | **3.15** |
| B camera shift | 68 % | **1.70 / 2.47** | [2.16, 2.81] | 0.49 | **3.57** |

The mean CIs of the stressed conditions do not overlap the ID CI. The pilot's amplification reproduced on
six tasks and new resets.

## 9. Effect Size vs Sampler Noise

| scale | in-distribution | under stress |
|---|---|---|
| chunk motion | 85.0 mm | ~85 mm |
| whole future-error consequence | 0.90 mm | **1.87 / 1.70 mm** (C1), 2.17 mm on the C2 subset |
| flow-sampler nuisance | 0.47 mm | 0.51–0.61 mm |
| treatment / nuisance | 1.77 | **3.15 – 3.57** |
| `D × S` advantage over `D` | +0.131 mm (28 % of nuisance) | **+0.231 mm (45 % of nuisance)** |

The problem became measurably more important — roughly double — and the mechanism's benefit roughly
doubled with it. Both remain small in absolute terms: the future channel governs ~2 % of the chunk's
motion, and the `D × S` advantage is still below the sampler's own run-to-run variation.

## 10. D / S / D×S / Attention Comparison

Stage C2, on a subset frozen before any ranking number existed: 2 stress conditions × 6 tasks × 10 episodes
× the approach-phase state = **exactly 120 states** (0 substitutions, 32 from failed episodes), a fresh
S sweep with round-1 parameters (~276,000 perturbed samplings), attention on all 120, **3,600 executed
chunks**, budget 64/256.

| selector | recovery under stress | residual mm | in-distribution (DS round) |
|---|---|---|---|
| random | 0.209 | 1.472 | 0.156 |
| attention | 0.432 | 1.189 | 0.362 |
| U (oracle) | 0.629 | 0.851 | 0.544 |
| D × attention | 0.701 | 0.635 | 0.578 |
| **D** | 0.736 | 0.589 | 0.580 |
| S | 0.787 | 0.419 | 0.660 |
| U × S (oracle) | 0.821 | 0.390 | 0.742 |
| **D × S** | **0.822** | **0.371** | 0.771 |

Every selector recovers more under stress, and the ordering is unchanged.

## 11. Equal-Budget Causal Repair

| comparison | mean | 95 % CI | better in | in mm |
|---|---|---|---|---|
| **D × S − D** | **+0.1218** | **[+0.0838, +0.1618]** | 74 % | **+0.231** |
| D × S − S | +0.0369 | [+0.0138, +0.0601] | 64 % | +0.097 |
| D × S − attention | +0.3722 | [+0.3188, +0.4233] | 92 % | +1.180 |
| **D × S − D × attention** | **+0.1253** | **[+0.0932, +0.1586]** | 82 % | **+0.349** |
| D × S − U × S (oracle) | +0.0093 | [−0.0110, +0.0295] | 52 % | +0.009 |
| U × S − U (oracle margin) | +0.2054 | [+0.1618, +0.2515] | 84 % | +0.501 |

All four required margins have CIs excluding zero; the difference from the **oracle** has a CI containing
zero. Consistent across both stresses (+0.121 occlusion, +0.123 shift) and all three suites. Margin
retention relative to the oracle margin is 0.593 under stress (0.852 in-distribution) — `D × S` keeps a
smaller share of a larger gap.

## 12. Failure / Critical-State Analysis

Two findings, one positive and one negative, both reported as measured.

**Positive:** the `D × S` advantage is **larger in episodes that actually failed** — +0.152 [+0.077, +0.233]
versus +0.111 [+0.066, +0.158] in successful ones. Where the robot failed, correctly ranking the future
tokens mattered more.

**Negative:** there is **no clean relationship between per-state treatment and episode failure**. In ID,
failed episodes have *higher* treatment (1.27 vs 0.80 mm); under occlusion *lower* (1.53 vs 1.95); under
shift equal (1.67 vs 1.71). Per protocol this auxiliary analysis does not rewrite the research question,
and no state-level failure predictor was built.

## 13. Optional One-Chunk Headroom Test

**Not run** — it was outside the scope set for stage C2. Consequently this round shows that `D × S` *ranks*
action-critical future tokens correctly under stress and that the ranking is worth +0.231 mm of executed
trajectory; it does **not** show that repairing those tokens would change task success or progress.
Task-level headroom remains undemonstrated and is the first thing Method Design must confront.

## 14. Alternative Explanations

1. *The grey patch drives everything.* No — family B has no occluder and shows a larger latent degradation
   (U +46 %) with the same doubled treatment. U was barred from being primary evidence regardless.
2. *Vision was destroyed and the policy simply broke.* No — success stays at 78 %/68 %, all failures are
   timeouts, no crashes, and the matched ID control on the same six tasks is 95 %.
3. *The chunks just move further.* No — chunk motion is comparable and the nuisance, measured on the same
   chunks with only the seed changed, barely moves while the treatment doubles.
4. *One task carries the result.* No — the treatment rises in every one of the six tasks for A and in five
   of six for B; the C2 margins hold in both stresses and all three suites.
5. *S alone, D alone, or attention would do.* No — `D × S` beats each with a CI excluding zero.
6. *Everything is easier under stress, so ranking is trivial.* Recovery does rise for every selector, but
   random still reaches only 0.209 and the gaps widen in millimetres.

## 15. Final Decision

**G3-PASS, qualified on criterion 5.**

Criteria 1–4 of the pre-registered G3-PASS definition are met decisively: the future-error influence is
clearly above in-distribution under two independent stresses (0.82 → 1.87/1.70 mm, t/n 1.77 → 3.15/3.57,
non-overlapping CIs); it is not drowned by the sampler nuisance; `D × S` still stably beats `D`, `S`,
attention and `D × attention` in failure-rich states; and the equal-budget causal repair still gives a
stable gain (0.822 vs 0.736, +0.231 mm, 74 % of states).

Criterion 5 — *some real task-progress / failure evidence linked to the mechanism* — is **only partially
met**: the mechanism's advantage is significantly larger in episodes that actually failed, but no
task-level headroom was demonstrated because the one-chunk headroom test was not run, and per-state
treatment does not separate successes from failures.

## 16. Stop-or-Proceed Recommendation

**Gap Validation is complete. The next stage is Method Design.**

It is not started here, and three constraints should travel with it:

1. **Headroom first.** The one-chunk headroom test (repair, then continue with the unmodified policy under
   the same stress, and compare task success/progress) is the cheapest way to learn whether any of this can
   change outcomes. If full-oracle repair moves nothing, a side-channel cannot either.
2. **`S` is still offline.** It costs ~2,300 forward passes (~86 s) per state. `D × S` is not yet an online
   quantity, and no `S` predictor has been trained — deliberately, so that a failure could never be
   attributed to two changed variables at once.
3. **Keep the millimetre discipline.** Every claim in four rounds has been reported against the flow
   sampler's own noise. Under stress the mechanism is worth +0.231 mm against 0.51 mm of sampler noise on
   an 85 mm chunk. That ratio, not the recovered-fraction percentages, is what a method has to beat.

---

### Deliverables
`PROTOCOL_G3.yaml` (with amendments G3-A1…A5), `AUDIT_G3.md` (AUDIT 0/1/2/3 + FINAL), `STATUS_G3.md`,
`PILOT_RESULTS.csv`, `pilot_summary.json`, `CONFIRM_RESULTS.csv`, `confirm_c1_summary.json`,
`TOKEN_SELECTION_RESULTS.csv`, `EXECUTION_RESULTS.csv`, `c2_summary.json`, `treatment_*.jsonl` (pilot and
confirm), `c2/c2_execution_records.jsonl`, `figures/{g3_pilot,g3_confirm_c1,g3_c2}.png`, `NEXT_STAGE.md`.
Code: `research/action_critical_reliability_probe/practical_significance/`.
