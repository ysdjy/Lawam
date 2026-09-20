# REPORT_DS.md — predicted change × downstream action sensitivity

run `acr_change_sensitivity_20260920_124748` · 2026-09-20 · branch `research/action-critical-reliability-probe` @ `b733f07`
prior runs read-only: `acr_20260919_203850` (round 1), `acr_learned_u_20260920_120110` (G2)
LaWAM `state_dict` sha256 `37a53b8c…b523`, verified unchanged · **nothing was trained in this round**

---

## 1. Research Question

**Q1.** Why does `D_j = ‖H_pred_j − h_t_j‖` — a quantity available *before* execution — predict the oracle
error `U_j = ‖H_pred_j − H_real_j‖` at Spearman ≈ 0.92? Is it (A) largely a geometric consequence, or (B) a
model phenomenon meaning "larger predicted change is genuinely harder to predict"?

**Q2.** Without using `H_real` at selection time, does `D × S` select action-relevant future tokens better
than `D`, `S`, attention and `D × attention` — and how much of the oracle `U × S` advantage does it retain?

## 2. Why D Emerged from G2

G2 set out to learn a deployable `U`. It found instead that a zero-parameter statistic,
`‖H_pred − h_t‖`, predicts oracle `U` at ρ ≈ 0.87–0.945 on every split including an unseen task and an
unseen suite, while a 1.46 M-parameter MLP beat it only on tasks it had trained on (Case G2-A). That made
two questions urgent: *why* does this work, and does the free proxy inherit the causal advantage that round
1 measured for the oracle `U × S`.

## 3. Definitions of U / D / G / S

| symbol | definition | nature |
|---|---|---|
| `U_j` | `‖H_pred_j − H_real_j‖` | oracle prediction **error**; needs the future; offline only |
| `D_j` | `‖H_pred_j − h_t_j‖` | **predicted change magnitude**, available before execution. *Not* uncertainty/confidence/reliability — see §5 |
| `G_j` | `‖H_real_j − h_t_j‖` | real change magnitude; oracle, used only to explain D→U |
| `θ_j` | angle between `Δ_pred` and `Δ_real` | — |
| `S_j` | frozen finite-difference downstream action sensitivity of round 1 | read-only; never recomputed here |

Identity used throughout: `U² = D² + G² − 2·D·G·cos θ`.

## 4. Geometry of D → U

Medians over 90 states / 23,040 tokens: **D = 7.19, G = 2.82, U = 7.09, D/G = 2.00, cos θ = 0.28**, and 80 %
of tokens have `G < D`. **U ≈ D**: the error is almost exactly the size of the whole predicted change.

Counterfactuals keep both magnitude distributions and destroy only the pairing of directions:

| | Spearman(D, U) |
|---|---|
| real data | **+0.923** [+0.910, +0.928] |
| cos θ permuted across tokens | **+0.907** [+0.891, +0.909] |
| cos θ = 0 (orthogonal) | **+0.973** [+0.969, +0.974] |

Removing every directional relationship leaves the correlation **equal or higher**. The relation is length
geometry, not a LaWAM-specific signal.

The mechanism, conditioned on how much each token actually moved:

| quartile of G | G | D | U | fraction with U < G | S |
|---|---|---|---|---|---|
| q0 (most static) | 1.64 | 6.68 | **6.71** | **0.000** | 2.4e-6 |
| q1 | 2.27 | 6.17 | 6.13 | 0.000 | 6.2e-6 |
| q2 | 3.58 | 5.94 | 6.08 | 0.031 | 1.1e-5 |
| q3 (most moving) | 18.42 | 17.40 | **9.64** | **0.883** | 1.9e-5 |

For three quarters of the tokens the scene barely changes while LaWAM predicts a change of ≈6–6.7, so the
error *is* the predicted change. Only on the moving quartile does the model genuinely help.

This also resolves an apparent contradiction with round 1. Round 1's statistic (global per-element MSE)
reproduces exactly — LaWAM beats "predict no change" in **83.3 %** of *states*, halving total squared error
(ratio 0.556) — but only **23.2 %** [22.0, 25.8] of individual *tokens* have `U < G`. The aggregate win is
earned entirely on the moving quartile, whose squared errors dominate the sum.

**Q1 verdict: Hypothesis A SUPPORTED, Hypothesis B NOT SUPPORTED.** Indeed the evidence runs against B:
Spearman(D, U/G) = **−0.277** and Spearman(D, 1 − cos θ) = **−0.510** — where LaWAM predicts a large change,
its *relative* error is lower and its direction agreement better.

## 5. Is D Really Reliability?

**No, and the term must not be used.** `D` measures how much the model *claims* a token will change, and on
~75 % of tokens that claim is largely spurious. It predicts `U` because on those tokens the claim is the
error. Its usefulness (§7–8) is real, but it is the usefulness of a **predicted-change** signal, not of an
uncertainty estimate. Separately, `S` does **not** rescue the concept: after removing what `D` explains, the
correlation between `S` and the residual is **−0.076** [−0.113, −0.021], and the residual's correlation with
the realised per-token action deviation is **−0.033** [−0.049, +0.038].

## 6. Phase / Task Breakdown

Geometry is stable: Spearman(D,U) = 0.905 / 0.924 / 0.924 for approach / manipulation / pre-grasp, with the
permuted-direction counterfactual at 0.876 / 0.909 / 0.915. `D/G` rises from 1.59 (approach) to 2.21
(pre-grasp) — the over-prediction of change is worst where the scene is most static.

## 7. Equal-Budget Token Selection

90 frozen states, **64 of 256 tokens** repaired (selected tokens get their true `H_real_j` — an offline
evaluation instrument, never a deployment proposal), 3 flow-noise seeds, identical snapshot, observation,
instruction, `h_vlm`, `h_t`, controller and post-processing. **2,700 executed chunks.**

| selector | recovery (median) | residual to full oracle | uses the future? |
|---|---|---|---|
| random | 0.156 | 0.732 mm | no |
| attention | 0.362 | 0.516 mm | no |
| `D × attention` | 0.578 | 0.383 mm | no |
| `U` (oracle) | 0.544 | 0.395 mm | **yes** |
| `D` | 0.580 | 0.346 mm | no |
| `S` | 0.660 | 0.278 mm | no |
| `U × S` (oracle) | 0.742 | 0.213 mm | **yes** |
| **`D × S`** | **0.771** | **0.184 mm** | **no** |

Cross-round validation: recomputed with an independent script, Recovery(U) = **0.5443** and
Recovery(U×S) = **0.7420** against round 1's 0.545 / 0.742.

## 8. D vs D×S vs U vs U×S

| comparison | mean | 95 % CI (episode bootstrap) | states better | in mm |
|---|---|---|---|---|
| **DS_margin = R(D×S) − R(D)** | **+0.1480** | **[+0.0939, +0.1982]** | 81 % | +0.131 |
| Oracle_margin = R(U×S) − R(U) | +0.1737 | [+0.1146, +0.2297] | 73 % | +0.130 |
| **Margin_retention** | **0.852** | — | — | — |
| R(D×S) − R(S) | +0.1218 | [+0.0594, +0.2028] | 66 % | +0.115 |
| R(D×S) − R(U×S) oracle | +0.0212 | [−0.0106, +0.0536] | 58 % | +0.041 |
| R(D) − R(U) oracle | +0.0469 | [+0.0045, +0.0941] | 54 % | +0.039 |

The pre-registered requirement is met: **the CI of DS_margin excludes 0**. The deployable `D × S` is
statistically **indistinguishable from the oracle `U × S`**, and even `D` alone slightly beats the oracle
`U` at choosing what to repair. A plausible reason follows from §4 — because `U ≈ D` on static tokens but
`U ≪ D` on moving ones, ranking by `D` tilts toward moving tokens, which carry ~8× higher `S`. This was not
tested directly and is offered as a hypothesis.

Consistency across phases (DS_margin +0.144 / +0.113 / +0.188 for approach / manipulation / pre-grasp) and
suites (`D × S` ≥ `U × S` in all three: goal 0.790 vs 0.721, object 0.794 vs 0.792, spatial 0.708 vs 0.685).

## 9. D×Attention Baseline

The mandatory "why not just attention?" control is answered: **`D × S` − `D × attention` = +0.419
[+0.152, +0.882]**, better in 82 % of states. `D × attention` also has a very unstable mean (CI
[−0.200, +0.552]), and attention alone (0.362) trails `D` alone (0.580).

Stability differs sharply, which the medians alone hide. Counting states with *negative* recovery — the
repaired chunk ending further from the oracle than the unrepaired one — `D × attention` has **7 of 90**
(minimum **−18.9**), `D` has 5 (minimum −0.18), and **`D × S` has 1** (minimum −0.09). The two extreme
`D × attention` states are what widen its CI; they are retained in every statistic and are clipped only
from the figure's axis, which annotates the omission. Attention mass is not a substitute
for measured downstream sensitivity — consistent with round 1, where the action head put 42 % of its
image-half attention on future tokens that barely affect the action.

## 10. Executed Consequences

All numbers above are executed end-effector displacements from exact simulator snapshots, not action-space
norms. The residual-to-oracle column of §7 is the physical quantity: repairing the 64 tokens chosen by
`D × S` leaves the executed trajectory **0.184 mm** from the fully repaired oracle, versus 0.346 mm for `D`
and 0.213 mm for the oracle `U × S`. Object-pose effects remain at the ~0.02 mm level, as in round 1.

## 11. Effect Size vs Sampler Noise

Measured on the same states:

| scale | value |
|---|---|
| chunk motion | **85.0 mm** |
| whole oracle correction (π(H_real) vs π(H_pred)) | **0.90 mm** |
| flow-sampler nuisance (different seed) | **0.47 mm** |
| **advantage of `D × S` over `D`** | **+0.131 mm** |

The improvement is about **28 % of the policy's own sampling noise** and **0.15 % of the chunk's motion**.
The 20 %-scale gains in §8 are gains inside a budget of influence smaller than the sampler's own variation.

## 12. Alternative Explanations

1. **Pure geometry** — established for Q1 (§4), and it does *not* dissolve Q2: the selection advantage is
   measured causally, by executing chunks, not by correlation.
2. **`S` does the work and `D` is decoration** — refuted: `D × S` beats `S` alone by +0.122 [+0.059, +0.203].
3. **Attention suffices** — refuted (§9).
4. **Repair-instrument artefact** — the recovery metric rewards removing large errors, and `D` ranks large
   predicted changes; since `U ≈ D` on most tokens, `D` is nearly optimal for that metric *by construction*.
   This is a genuine limitation of the instrument, not of the statistics: it means §7–8 measure *ranking
   quality under oracle repair*, which is not the same as usefulness at deployment (§14).
5. **Static-token dominance** — 75 % of tokens barely move; a different scene distribution (motion, clutter,
   OOD) could change every number here. Untested.

## 13. Final Decision: DS-A / B / C / D

**Case DS-C.** `D` approximates `U` well; `D × S` stably beats `D` (CI excludes 0), `S`, attention and
`D × attention`; it retains 85 % of the oracle margin and is statistically indistinguishable from `U × S`.
DS-A is not triggered (its geometric clause holds, its causal clause is refuted), DS-B is not triggered
(attention is insufficient), DS-D is not triggered (no reliability estimator is needed).

Per `PROTOCOL_DS.yaml:decision.DS-C`, the result is stated as: *the evidence supports a simpler candidate
mechanism — the joint use of predicted future-change magnitude and downstream action sensitivity may
identify action-critical future information in advance.* **No formal name is coined, and no method is
proposed.**

## 14. What This Result Does NOT Prove

1. That `D` measures reliability or uncertainty — §4–5 specifically refute that reading.
2. That flagging a token helps at deployment. Recovery is measured by replacing tokens with their **true**
   values; at deployment no true value exists, and nothing here shows what a high-`D × S` flag would be
   *used for*.
3. That `S` is obtainable online — it still costs ~2,300 forward passes per state. `D × S` is not yet an
   online quantity, and no `S` predictor was trained (deliberately, so that a failure could not be
   attributed to two changed variables at once).
4. That any of this improves success rate, robustness or safety — no such quantity was measured.
5. That it matters at the observed scale — +0.131 mm against 0.47 mm of sampler noise (§11).
6. That it generalises beyond successful, in-distribution trajectories of three tasks on one checkpoint
   trained with `enable_flow_h_t1_scheduled_sampling=false`.

## 15. Recommended Next Gate

**G3 — failure / OOD enrichment**, not entered this round. Under pre-registered perturbations (occlusion,
visual distractors, camera perturbation, unseen layout, object perturbation) the scene stops being mostly
static, `G` should approach `D`, and the future channel may govern more than ~1 mm. The questions that
matter there:

1. Does the static-token dominance that drives §4 disappear, and does `D` remain a good proxy for `U` when
   it does?
2. Does `D × S` predict **real action error and task failure**, not just oracle-repair recovery?
3. Does the future channel's influence exceed the sampler nuisance under difficulty — the precondition,
   still unmet, for any of this to matter for control?

Two supporting gates remain: **G4** (can `S` be approximated online at all) and **G1** (a checkpoint trained
with scheduled sampling, still the decisive test of whether the future channel can carry more influence).

---

### Deliverables
`PROTOCOL_DS.yaml`, `AUDIT_DS.md` (DS AUDIT 0/1/2 + FINAL), `GEOMETRY_ANALYSIS.csv`,
`GEOMETRY_ADDENDUM.csv`, `summary_geometry.json`, `SELECTION_METRICS.csv`, `EXECUTION_METRICS.csv`,
`ds_execution_records.jsonl`, `summary_ds.json`, `figures/ds_selectors.png`, `g2_audit0_model.json`,
`code_changes.patch`, `LONG_TERM_PLAN_DS.md`, `ds_chunks/` (raw per-state selector chunks).
Code: `research/action_critical_reliability_probe/change_sensitivity/`.
