# REPORT_G2.md — Learned-U (G2)

run `acr_learned_u_20260920_120110` · 2026-09-20 · branch `research/action-critical-reliability-probe` @ `4f9c641`
prior run `acr_20260919_203850` (read-only) · LaWAM `state_dict` sha256 `37a53b8c…b523`, unchanged

No LaWM / action-head / Qwen / LAM training, no S predictor, no side-channel, no RL, no OOD experiment.
The only thing trained is a 1.46 M-parameter diagnostic MLP.

---

## 1. Why Learned U Is Necessary

The prior round defined `U_j = ‖H_pred_j − H_real_j‖`, where `H_real` is the encoding of the observation the
robot reaches **after** executing the chunk. At deployment that quantity does not exist. So the prior
round's result — that oracle `U × S` selects action-relevant future errors better than U, S or attention —
was a statement about an instrument, not about a method. If U cannot be obtained before execution, the
Action-Critical Reliability line cannot become a method at all. G2 tests exactly that, and nothing else.

## 2. Exact Short-Term Question

> Can per-token U be predicted from information that already exists before the chunk is executed
> (`h_t_j`, `H_pred_j`, `H_pred_j − h_t_j`, `z`), and does a learned estimator do so **better than free
> statistics of those same tensors**, on unseen episodes *and* unseen tasks?

## 3. Data Collection

6 tasks fixed in `PROTOCOL_G2.yaml` before any collection, chosen only for "simulator loads / baseline
LaWAM runs / type diversity / no known asset error": `libero_spatial` t0, t7, t9; `libero_object` t3, t6;
`libero_goal` t5. 10 independent reset episodes each (60 rollouts of the unmodified policy through the
official server), all executed chunks recorded, **failures kept** (1 of 60: goal t5 ep9, 250-step timeout).

States are chunk boundaries labelled by the frozen phase rule: 160 states. The target was ~180; the
shortfall is because `libero_goal t5` is a non-prehensile **push** — it has no `pre_grasp` or `manipulation`
phase, and the missing phases were recorded as missing rather than substituted.

Collection reused `s1_rollouts.py`, `s2_select_states.py`, `s3_replay_futures.py`; U extraction imports
`lam_features()` and `per_token_u()` directly from `m3_uncertainty.py`, so the encoder and the U definition
are literally the prior round's objects. Snapshot replay reproduces the online end-effector/object positions
exactly for 158/160 states (2 contact-rich `spatial t7` pre-grasp states differ by 0.19 / 0.56 mm; the U
label uses the online frame, so U is unaffected).

New-task U statistics match the prior round closely: U median **0.0875** (prior 0.0867), render-floor SNR
**16.2×** (16.0×), LaWM beats "predict no change" in **82.5 %** of states (83.3 %).

## 4. Episode/Task Split

Episode is the minimum independent unit; tokens and states are never split randomly (`DATA_SPLIT.md`,
`splits.json`, all leakage assertions PASS).

| split | meaning | states | episodes | tokens |
|---|---|---|---|---|
| train | 4 training-pool tasks, episodes 0–5 | 72 | 24 | 18,432 |
| val | same tasks, episodes 6–7 | 24 | 8 | 6,144 |
| testA | same tasks, episodes 8–9 — **unseen resets** | 24 | 8 | 6,144 |
| testB1 | `libero_spatial t9` — **unseen task**, seen suite | 30 | 10 | 7,680 |
| testB2 | `libero_goal t5` — **unseen task + unseen suite**, push skill | 10 | 10 | 2,560 |
| testC | prior round's 3 tasks — unseen, and already carry frozen S | 90 | 30 | 23,040 |

## 5. Oracle-U Definition

Unchanged from the prior round: `H_real = lam.extract_vision_features(o_{t+7})` with the identical flip,
resize, ImageNet normalisation and bf16 autocast as `h_t`; `U_j = ‖H_pred_j − H_real_j‖`. `H_real` never
enters a policy input — asserted at runtime during extraction — and appears in the dataset only as the
scalar label.

## 6. Simple Baselines

Mandatory, run before any training (`summary_baselines.json`). Spearman with oracle U:

| predictor | train | testA | testB1 | testB2 | testC |
|---|---|---|---|---|---|
| `B0_random` (floor) | −0.000 | +0.015 | +0.010 | −0.004 | −0.003 |
| `B0x_token_index` (control) | −0.310 | −0.315 | −0.377 | −0.095 | −0.289 |
| `B1 = −‖H_pred_j‖` (token norm) | 0.606 | 0.614 | 0.681 | 0.631 | 0.548 |
| **`B2 = ‖H_pred_j − h_t_j‖`** | **0.929** | **0.931** | **0.945** | **0.873** | **0.919** |
| `B3 = 1 − cos(H_pred_j, h_t_j)` | 0.930 | 0.931 | 0.945 | 0.874 | 0.919 |
| `B4 = linear(6 stats)`, fitted on train | 0.926 | 0.928 | 0.936 | 0.875 | 0.915 |

Two things matter here. First, **a zero-parameter statistic predicts U at ρ ≈ 0.87–0.945 everywhere**,
including an unseen suite — where LaWM predicts a large change from the present, prediction error
concentrates. Second, the **token-norm** baseline the prior round flagged as a confound is clearly weaker
(0.55–0.68), so U is *not* just feature magnitude.

## 7. Learned-U Architecture

`[h_t_j, H_pred_j, H_pred_j − h_t_j, z]` → `Linear(2336→512) → LN → GELU → Linear(512→512) → LN → GELU →
Linear(512→1)`, shared across tokens, 1.46 M parameters, target `log(U+1e-6)`, Huber loss, AdamW, early
stopping on validation Spearman, 3 seeds averaged. No token index, no positional id, no ranking loss.
Details and limitations: `MODEL_CARD_U.md`.

## 8. Learned-U Generalization

| split | learned MLP | free `B2` | paired Δ (MLP − B2), episode bootstrap | MLP better in |
|---|---|---|---|---|
| train | 0.987 | 0.929 | +0.058 [+0.049, +0.068] | 100 % |
| val | 0.960 | 0.929 | — | — |
| **testA** unseen resets | **0.959** | 0.931 | **+0.028 [+0.020, +0.039]** | **100 %** |
| **testB1** unseen task | **0.901** | 0.945 | **−0.044 [−0.050, −0.037]** | 3 % |
| **testB2** unseen task+suite | **0.840** | 0.873 | **−0.034 [−0.058, −0.011]** | 20 % |
| **testC** prior tasks | **0.909** | 0.919 | **−0.009 [−0.026, +0.006]** | 48 % |

The learned model wins on held-out **episodes** and loses or ties on every held-out **task**, with
confidence intervals excluding zero in the losing direction on both B1 and B2 splits. Had only the episode
holdout been run, the opposite conclusion would have been reported — which is why both were pre-registered.

Ranking-head metrics are more favourable to the MLP than Spearman: on testC it reaches recall@64 **0.821**
vs 0.762, NDCG@64 **0.932** vs 0.875, AUROC **0.960** vs 0.941. So it orders the most-unreliable tokens
somewhat better while being marginally worse over the full ranking. This nuance is recorded, but it does not
change the verdict: it is not a clear win on held-out tasks, and the pre-registered criterion is "clearly
beats the simple baselines".

Phase breakdown over all test splits is flat (MLP 0.908 / 0.913 / 0.913 vs B2 0.917 / 0.931 / 0.922 for
approach / pre-grasp / manipulation) — the transfer failure is not phase-specific.

## 9. Norm Confound Analysis

Answered, and in the project's favour: `B1` (pure `‖H_pred_j‖`) reaches only 0.55–0.68, far below `B2`'s
0.92–0.945, so the predictability of U is **not** explained by token magnitude. The prior round's observed
ρ(U, token-norm) ≈ −0.55 is real but is a weak part of the story; the dominant signal is the *predicted
change*, `‖H_pred − h_t‖`.

## 10. Permutation Equivariance

**PASS, exactly 0.0** on all 250 states: permuting `h_t` and `H_pred` permutes `U_hat` identically. This is
architectural (shared per-token weights, no index feature), and the audit exists to catch an implementation
mistake. It is not a formality: a **token-index-only** predictor reaches Spearman −0.31 (train) / −0.29
(testC), i.e. U genuinely carries positional structure that a careless model could have exploited.

## 11. Learned U + Offline S

**Not run.** `PROTOCOL_G2.yaml` gates Step 5 on the learned-U result, and the result is negative
(§15). No `U_hat × S` number exists for this run and none was estimated.

## 12. Equal-Budget Causal Test

**Not run**, for the same reason (`METRICS_SELECTION.csv` records this explicitly). The prior round's
equal-budget machinery and its frozen S on the 90 testC states remain available and untouched should the
question be re-opened with the free proxy instead of a learned estimator.

## 13. Seen vs Unseen Task

This is the axis on which the round turns. Seen task / unseen resets: learned 0.959 vs free 0.931 — learned
wins in 100 % of states. Unseen task: 0.901 vs 0.945, 0.840 vs 0.873, 0.909 vs 0.919 — learned loses or
ties. The learned advantage is task-bound; the free statistic is task-general.

## 14. Limitations

1. One checkpoint, one horizon (0.4 s), simulation only, 10 tasks' worth of data at most.
2. Only the v1 feature set and a shared-token MLP were tried. A different architecture (e.g. one with
   cross-token context) was **not** attempted — the protocol requires a new audit before escalating, and
   the free baseline's strength makes that escalation hard to justify on present evidence.
3. The oracle label is built from a single rendered future frame; SNR over the render floor is ~16×, but
   2 of 160 states have contact-rich replays whose floor is overestimated.
4. `testB2` is approach-only (10 states) because its task has no grasp phase; conclusions from it are
   correspondingly weak.
5. Almost all episodes succeed (1 failure in 60), so nothing here speaks to failure or OOD conditions.
6. U remains defined against a *latent* future; it is not a task-level risk measure.

## 15. Decision: G2-A / B / C / D

**G2-A: SIMPLE BASELINE SUFFICIENT.**

The learned estimator does not clearly beat the free statistics on held-out tasks — it is worse on testB1
(−0.044 [−0.050, −0.037]) and testB2 (−0.034 [−0.058, −0.011]) and ties on testC (−0.009 [−0.026, +0.006]).
Per the pre-registered decision tree, a complex reliability estimator is **not warranted** and the
learned-U line stops.

**The deployability question G2 was created to answer nevertheless comes out positive**, and this must be
read alongside the verdict: U — a quantity defined against a future that only exists after execution — is
predictable *before* execution at Spearman ≈ 0.92, across unseen tasks and an unseen suite, by a statistic
that costs one subtraction and one norm and requires no training. What G2 refutes is the need to **learn**
reliability, not the existence of a pre-execution reliability signal.

## 16. What Is Allowed Next

Nothing automatically. The pre-registered stop is in force: no side-channel, no S predictor, no OOD/G3
experiment, no further training.

One well-posed question survives and was already anticipated by the protocol's selector list
(`top_change_norm`): whether the **free** proxy `‖H_pred − h_t‖`, combined with the frozen S, retains the
oracle `U × S` advantage in the equal-budget causal execution test on the 90 testC states. It would need one
logged amendment (adding `top_change_norm × S` to the selector list) and consumes only existing artefacts.
It is **not started without an explicit decision**, because the gate it sits behind returned negative.

## 17. What Is Still Not Demonstrated

- That any reliability signal — learned or free — improves robot success, robustness, or safety.
- That `U × S` survives with a deployable U (not tested this round).
- That S can be obtained online at all; S still costs thousands of finite-difference forwards, and no S
  predictor was trained, by design.
- That the mechanism matters outside the tiny headroom measured previously (the whole future-prediction
  error was worth ≈0.9 mm of executed motion against ≈0.47 mm of sampler noise).
- Anything about OOD, occlusion, distractors, or failure prediction.

---

### Deliverables
`PROTOCOL_G2.yaml`, `AUDIT_G2.md` (G2 AUDIT 0/1/2), `DATA_SPLIT.md`, `splits.json`, `MODEL_CARD_U.md`,
`TRAIN_LOG.jsonl`, `METRICS_U.csv`, `METRICS_U_baselines.csv`, `METRICS_SELECTION.csv` (NOT_RUN marker),
`summary_baselines.json`, `summary_learned_u.json`, `g2_audit0_model.json`, `u_dataset_records.jsonl`,
`states_manifest.jsonl`, `futures_manifest.jsonl`, `figures/g2_learned_u.png`, `models/`,
`code_changes.patch`, `LONG_TERM_PLAN_G2.md`.
Code: `research/action_critical_reliability_probe/learned_u/`.
