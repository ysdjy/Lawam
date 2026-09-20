# LONG_TERM_PLAN_DS.md — what the DS round changes (PLAN ONLY, nothing implemented)

Status: **no method is proposed and no name is coined.** Case DS-C was reached on the causal criterion, but
the effect lives below the policy's own sampling noise, so nothing here authorises building anything. The
next gate is G3, which was **not** entered.

---

## 1. What three rounds have established, and what they have removed

| once assumed | now measured |
|---|---|
| a *Reliability Estimator* module must be learned | **no**: a learned U beats a free statistic only on tasks it trained on (G2, Case G2-A) |
| `D = ‖H_pred − h_t‖` predicts U because bigger changes are harder to predict | **no**: destroying all directional information leaves Spearman(D,U) equal or higher (0.923 → 0.907 / 0.973). It is length geometry: LaWAM over-predicts change ≈2× on a mostly static token population, so on ~75 % of tokens the error *is* the predicted change |
| D might be a deployable *uncertainty* | **no**: it is a predicted-change magnitude, and on the static majority that prediction is largely spurious. The name stays |
| the oracle `U × S` advantage needs an oracle | **no**: the deployable `D × S` recovers 0.771 vs the oracle's 0.742, retains 85 % of the oracle margin, and the difference from `U × S` has a CI containing 0 |
| attention might substitute for sensitivity | **no**: `D × S` − `D × attention` = +0.419 [+0.152, +0.882], better in 82 % of states |
| S might explain what D misses | **no**: Spearman(S, residual of U after D) = −0.076 [−0.113, −0.021] |

What survives as a *candidate observation*: predicted future-change magnitude combined with downstream
action sensitivity ranks action-critical future tokens as well as an oracle error signal does. That is an
observation about ranking under an oracle repair instrument — not a mechanism, not a method.

## 2. The constraint that governs everything

| scale | value |
|---|---|
| chunk motion | 85.0 mm |
| whole future-prediction error's executed consequence | 0.90 mm |
| flow-sampler nuisance (different seed) | 0.47 mm |
| advantage of `D × S` over `D` | **+0.131 mm** |

Three rounds have now measured the same ceiling from three directions. Until a condition is found where the
future channel governs materially more than ~1 mm, every ranking improvement inside it is academic.

## 3. Gates, in order, none of them started

| gate | question | why it is next |
|---|---|---|
| **G3** | under pre-registered perturbations (occlusion, distractors, camera perturbation, unseen layout, object perturbation): does static-token dominance disappear (G → D), does `D` remain a good proxy for `U`, and does `D × S` predict **real action error and task failure** rather than oracle-repair recovery? | it attacks both the effect-size ceiling and the strongest limitation of the current instrument |
| G4 | can `S` be approximated online at all (amortised sensitivity, one-shot gradient proxy)? | `S` still costs ~2,300 forwards per state; `D × S` is not an online quantity |
| G1 | does a checkpoint trained with `enable_flow_h_t1_scheduled_sampling=true` give the future channel more influence? | unchanged since round 1; still the decisive test of whether any of this can matter for control |

**Sequencing rule:** G3 before G4. If G3 shows the future channel still governs ≈1 mm under difficulty, an
online `S` would be an expensive answer to a question that does not matter.

## 4. Design constraints any future mechanism must respect

1. **Content, not position** — the action head consumes the 256 future tokens as an unordered bag
   (permutation changes the action by ~4e-7).
2. **Identity, not repair** — at deployment there is no true `H_real` to substitute. A flag must be used
   for something else: gating, abstention, re-planning, or weighting — none of which is tested.
3. **Below-noise budget** — any gate must be compared against the sampler's 0.47 mm variation, not against
   zero.
4. **Terminology** — `D` is predicted change magnitude. Renaming it to uncertainty/reliability would assert
   what §4 of `REPORT_DS.md` specifically failed to find.
5. **Scene dependence** — every number here comes from trajectories where 75 % of tokens barely move.

## 5. Re-usable assets

`research/action_critical_reliability_probe/change_sensitivity/`: `ds_geometry.py` (the U = f(D, G, cos θ)
decomposition with direction-destroying counterfactuals), `ds_geometry_addendum.py` (conditional analysis by
real-change quartile), `ds_chunks.py` (8-selector equal-budget chunk generation on frozen S/attention),
`ds_execute.py` (2,700-chunk execution with recovery/residual in millimetres), `ds_analyze.py` (paired
episode-bootstrap comparisons). Together with the round-1 and G2 assets these now reproduce the whole
pipeline from rollout to executed consequence without retraining anything.
