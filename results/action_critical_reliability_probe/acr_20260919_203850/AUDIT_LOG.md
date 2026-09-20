# AUDIT_LOG.md — Action-Critical Reliability probe (`acr_20260919_203850`)

Append-only. Every stage adds one entry. Entries are written before the next stage starts.

---

## AUDIT 0 — environment, checkpoint and interface audit (2026-09-19/20)

**Scope of what was done.** Read-only inspection of the repository, checkpoint, both conda environments and
the LaWAM code path; two audit scripts that load the real model and the real simulator and read out shapes,
floors and scale statistics. No perturbation, no new experiment, no model change, no training, no git
mutation (no reset/checkout/clean/pull/push), no checkpoint write.

**Artifacts.** `environment_audit.md`, `audit0_model.json`, `audit0_sim.json`, `protocol.yaml`
(pre-registration), `code_changes.patch` (the pre-existing dirty diff, unchanged by this round).
Scripts: `research/action_critical_reliability_probe/scripts/{probe_common,a0_model_audit,a0_sim_audit}.py`.

**Input data scale / what actually ran.** 3 real LIBERO observations (episodes ep000/ep011/ep019 of
libero_spatial task 2, steps 8/24/56, reused from the prior run's online rollouts) × {1 diagnostics
forward + 4 repeat forwards + 2 self-override forwards + 2 alternative-noise forwards} = 27 policy
forwards. 1 simulator env built, 1 snapshot/restore/replay cycle of 8 steps, 3 suites enumerated
(30 tasks total). Excluded data: none. Infrastructure failures: none.

### 1. Which hypothesis was tested here?
None. AUDIT 0 is infrastructure only: it establishes what the interface *is* (N, D, chunk length, time
alignment, where H_pred enters the action head, what the numerical floors are), so that later measurements
can be attributed to the model rather than to plumbing.

### 2. Was the original research question changed?
No. The question stays as fixed in `protocol.yaml:research_question.fixed_text`. Nothing about U, S or R
was measured or concluded here.

### 3. Was an unnecessary new module added?
No. Zero lines of model code were changed this round. The four diagnostic hooks present in the working tree
(`latent_override`, `future_override`, `initial_noise`, `return_diagnostics`) pre-date this round (from run
`bd_20260914_093902`); they were re-verified as exact no-ops rather than re-implemented. Step 1 can be built
entirely on top of `future_override`; only the Step-4 attention baseline will require a new (default-off)
extraction hook.

### 4. Could today's numbers be explained by an engineering bug, randomness or time misalignment?
The numbers recorded are floors and shapes, and they are mutually consistent:
- action chunk repeatability with identical inputs and identical initial noise is **exactly 0.0** across
  4 repeats on 3 observations — the policy path is deterministic once the flow noise is fixed;
- `future_override = H_pred` and `latent_override = z` reproduce the default path to **0.0**, so the hooks
  do not perturb anything by themselves;
- simulator snapshot → 8 steps → restore → replay reproduces eef and qpos to **0.0**;
- shapes read from the loaded model ([1,256,768] future, [1,8,32] actions, [1,214,2048] h_vlm) match what
  the code path predicts.
Time alignment was re-derived from the training data sampler (`num_frames=2`, `sec_chunk=0.4` → indices
[0,7]) and from the client execution pattern (all 8 chunk steps executed before the next query): H_pred
corresponds to `o_{t+7}`. The known 2 ms render-lag / EGL anti-aliasing issue is carried over from the prior
run together with its mitigation (model inputs always come from the saved online frame).

### 5. Is there a simpler explanation for anything observed?
The only "observation" with content is that `h_t` token norms are *exactly* constant (27.713 = √768). The
simple explanation is the LayerNorm at the end of the DINOv3 feature extractor, not anything about the
model's behaviour. This is recorded because it fixes the natural scale for perturbations, and because it
means per-token feature *norm* cannot be a confound on the h_t side (it can still be one on the H_pred
side, whose norms vary by ±0.4).

### 6. Is the next step supported by the current evidence?
Yes, and only the next step. The audit shows the Step-1 experiment is *mechanically possible and
well-posed*: the future enters the action head at an identifiable place (cross-attention blocks 2/6/10/14,
256 keys), can be perturbed token-wise without touching anything else, the action sampler is exactly
repeatable, and the perturbation scale can be calibrated against measured feature statistics. It does not
support any statement about S, U or R.

One carry-over that must be stated explicitly: run `bd_20260914_093902` already found, on this same
checkpoint, that replacing the *entire* future with a genuinely different real future moved the normalized
action chunk by ≈0.034 max-abs (≈3 mm of executed eef displacement, versus ≈70 mm needed for a plan change).
That is prior evidence pointing toward weak future utilisation, i.e. toward Case A. It is the reason Step 1
is the gate, and it is a reason to expect a negative result — it is *not* itself a result of this round, and
it must not be used to skip Step 1.

### 7. Most conservative conclusion if we stopped now.
"The LaWAM LIBERO SFT checkpoint `lawam_libero_sft_release` exposes a 256×768 predicted future latent that
reaches the flow action head only as cross-attention keys in 4 of 16 DiT blocks; the inference path is
bit-exactly repeatable under fixed flow noise, and the diagnostic hooks needed to intervene on that latent
exist and are exact no-ops when disabled." No claim about whether the action head *uses* that latent.

### 8. What would overturn the current direction?
- Step 1 finding single-token action changes at or below the (zero) numerical floor in physically
  meaningless magnitude, together with a global-perturbation change below the flow-noise nuisance scale
  → Case A, pivot to Future Utilization.
- Step 1 finding S essentially uniform across tokens (p95/median < 2) or rank-unstable across ε /
  directions → Case B, token-level reliability not worth building.
- Later: |Spearman(U,S)| ≥ 0.8 → Case C.
All three triggers are numerically fixed in `protocol.yaml` before any run.

### Open engineering items carried into Step 1
1. A batched flow-only fast path is needed for 2,304 perturbations per state; it must be proven equal to the
   full `predict_action` path (`step1_sensitivity.controls.D_batch_equivalence`, tolerance 1e-6, otherwise
   the fast path is abandoned).
2. Attention extraction hook does not exist yet (needed only at Step 4).
3. Disk: 24 GB free. Per-token raw action chunks will not be stored in full; summary metrics + a small raw
   subset only.
4. libero_object rollouts do not exist yet and must be collected with the official server/client path.

**Status: AUDIT 0 passed. Proceeding to Step 1 (gate: does the action head read the future at all?).**

---

## AUDIT 0b — measurement-protocol amendment before any S number (2026-09-20)

**Why this entry exists.** The batched fast path needed for the sweep turned out to have a float-noise
property that could have contaminated every S value, so the measurement protocol was fixed *before* any
sensitivity data was produced.

**What ran.** `m1_equivalence.py` (3 real observations, ~40 policy forwards) and `m1b_batch_noise.py`
(3 observations × {5 batch sizes, 4 row positions, 3 ε × 6 tokens, 6 whole-future controls}).
Artifacts: `phase0_equivalence.json`, `phase0_batch_noise.json`. Excluded data: none. Failures: none.

**Findings that changed the protocol.**
1. The shared-encoding + batched-flow fast path reproduces the untouched `vla.predict_action` **exactly**
   (max-abs 0.0) at B=1, and repeated identical calls differ by 0.0.
2. Rows with identical inputs inside one batch are **bit-identical** (spread 0.0) regardless of row position
   and batch size — but the *same* input evaluated at a *different batch size* shifts by ~5e-7.
3. A single-token perturbation at ε_mid moves the action chunk by only ~1e-6…2e-4, i.e. 2–400× the
   cross-batch-size offset. → Amendment **A1**: the baseline A₀ is always a row of the same batch
   (floor 0.0); cross-batch-size tolerance in the regression test is 1e-5, not 0.
4. The future tensor is bf16; the realised perturbation norm matches the requested ε to <5e-7 relative, and
   S is normalised by the realised norm. → Amendment **A2**.
5. **Structural finding**: permuting the 256 future tokens changes the action by ~0 (4.5e-8 max over 90
   states, 0.0 exactly in the first tests). The DiT adds no positional embedding to condition tokens, so the
   action head consumes H_pred as an *unordered bag*. → Amendment **A3**; token-level S can only ever
   reflect token content, never token position.
6. Amendment **A4** (declared before U existed): add the error-direction decomposition to Step 4.
7. Amendment **A5** (state selection, before any S measurement): the "approach" phase originally selected
   the *first* qualifying chunk boundary, which was step 0 — the same pre-motion scene in every episode.
   Changed to the *median* qualifying boundary. Selection never inspects U or S.

**Infrastructure faults found and fixed.** (a) `libero_goal` tasks fail on a second `env.reset()` in the
same process — worked around with `--fresh_env_per_episode`; (b) `libero_goal` task 0 has a genuinely
missing asset region (`wooden_cabinet_1_middle_region`) in this LIBERO install and was replaced by task 8
("put the bowl on the plate"); (c) building many envs in one process corrupts LIBERO's object registry and
produces misleading cascading errors — an earlier "all goal tasks are broken" reading was **wrong** and is
retracted here: in a fresh process the goal tasks build normally.

**Conservative conclusion at this point.** Only that the measurement apparatus is sound and that the action
head treats the future as an unordered bag of 256 tokens.

---

## AUDIT 1 — Step 1: does the action head read the future, and is the reading token-dependent? (2026-09-20)

**Hypothesis under test.** Only the gate: is Downstream Action Sensitivity S measurable above the numerical
floor, and does it differ across future tokens? (Nothing about U, nothing about R.)

**Input scale / what actually ran.** 90 states = 3 suites × 1 task each (libero_spatial t2, libero_object t0,
libero_goal t8) × 10 independent reset episodes × 3 execution phases (approach / pre_grasp / manipulation),
all from successful undisturbed rollouts (30/30 episodes succeeded). Per state: 3 ε × 3 directions × 256
tokens = 2,304 perturbed samplings + 9 whole-future controls + 2 extra flow-noise seeds ≈ 2,315 action
samplings → **≈ 208,000 action samplings in total**, 80.3 s/state. Excluded states: none. Infrastructure
failures: none. Artifacts: `sensitivity_records.jsonl` (90), `sensitivity/*.npz`, `metrics_sensitivity.csv`,
`summary_step1.json`, `figures/step1_sensitivity_overview.png`.

### 1. Which hypothesis did this verify?
Only the Step-1 gate. Answers to the pre-registered AUDIT-1 questions:

1. *Reproducible with the same H and the same ξ?* Yes — in-batch repeat difference **0.0**, and the
   `repeat_identical` control is 0.0 on all 90 states.
2. *Is S above the floor?* Yes, by a wide margin in relative terms: the median token's action deviation at
   ε_mid is 6.9e-6 and the most sensitive token's is 1.5e-4 (median over states), against a floor of 0.0.
3. *Do tokens differ?* Yes: p95/median = **9.26** (median over states, 5.6–16.4; bootstrap CI of the mean
   over episodes [9.08, 10.06]), max/median = 22.8, Gini 0.64. Case B (< 2) is far from triggering.
4. *Does it change with phase?* The *shape* changes mildly (p95/median 12.1 approach, 8.4 manipulation,
   7.7 pre-grasp), but the aggregate future dependence changes a lot — see §5.
5. *Rank stability?* Spearman across ε: 0.998 (small↔mid) and 0.996 (mid↔large); across random directions
   0.901; across different states of the same suite only 0.53–0.58. So within a state the ranking is
   essentially exact, and it is partly (not wholly) state-specific.
6. *Case A (S≈0)?* Not triggered: its token half is true (max single-token deviation 1.5e-4 < 1e-3) but its
   global half is false (zeroing the future changes actions by 1.4e-2 median, 3.1× the flow-noise nuisance).

### 2. Was the research question changed?
No. Step 1 was run exactly as pre-registered; the only changes are the logged amendments A1–A5, all made
before any S value existed and none of which touches the question.

### 3. Was an unnecessary module added?
No model code was changed at all. The sweep runs on the pre-existing `future_override` hook plus a
research-side batched replay of the flow head that is proven bit-identical to `predict_action`.

### 4. Could the result be an engineering artefact, randomness, or time misalignment?
- Floor: 0.0 (in-batch), so a 6.9e-6 deviation is not sampler noise.
- The realised perturbation norm equals the requested ε to 4.5e-7 relative.
- Linearity check: deviations scale by ×2.95 for a ×3.0 increase in ε and ×3.50 for ×3.33 — the
  perturbations are in the local, near-linear regime, so they are not OOD-driven artefacts.
- Token permutation ≈ 0 rules out any positional/index bug in how the perturbed tensor is passed.
- No time alignment is involved in Step 1 (no future observation is used yet).

### 5. Is there a simpler explanation?
Two, and both are partly true and are reported as such rather than argued away:
- **Per-token effects are physically negligible.** The most sensitive single token at ε_mid moves the
  commanded end-effector trajectory by ≈0.11 mm (median over states; max 0.60 mm), while the flow sampler's
  own noise moves the action by 4.5e-3 (≈30× more) and the mean action magnitude is 0.23. Token-level
  sensitivity exists and is strongly non-uniform, but it is small compared with the policy's own stochasticity.
- **Most of the future dependence is an aggregate, phase-dependent effect.** Zeroing the whole future
  changes the action by 1.4e-2 (median; up to 0.13), replacing it by h_t by 1.1e-2 — but broken down by
  phase, the ratio "zeroed future / flow noise" is **10.2 in approach, 3.3 in manipulation, 1.7 in
  pre-grasp**. The future channel is used mainly while approaching, and is close to irrelevant at the
  pre-grasp boundary. A single "does LaWAM use the future" number would have hidden this.
- Also: the sum of the 256 single-token deviations (4.1e-3) is ~10× the deviation from perturbing all 256
  tokens at once (3.8e-4), i.e. individual token effects largely cancel. Any token-level score therefore
  cannot be assumed to add up to a state-level effect.

### 6. Is the next step supported?
Yes, for Step 2 (U) only. S is measurable, extremely rank-stable within a state, and non-uniform by ~an
order of magnitude, so asking whether U and S are complementary is a well-posed question. The next step does
not assume that the per-token magnitudes are practically important — that question is deferred to Step 4,
where U×S must be shown to predict an actual action consequence.

### 7. Most conservative conclusion if we stopped now.
"On this checkpoint, the flow action head does use the predicted future latent as a whole (removing it
changes the action chunk by ≈3× the sampler's own noise, concentrated in the approach phase), and its
sensitivity to individual future tokens is highly non-uniform (p95/median ≈ 9) and stable in rank within a
state. However, a single future token is worth ≈1e-4 in normalized action units (≈0.1 mm of commanded
motion), which is ~30× smaller than the flow sampler's own run-to-run variation."

### 8. What would overturn the current direction now?
- Step 2: U indistinguishable from the render-noise floor, or U dominated by an uninteresting global term.
- Step 3: |Spearman(U,S)| ≥ 0.8 within states (Case C).
- Step 4: R = U×S not better than U or S alone at predicting the *realised* per-token action deviation and
  the state-level E_action, or an attention baseline matching it.
- A finding that the state-level E_action is itself below the flow-noise nuisance would make the whole
  quantity practically irrelevant regardless of correlations.

**Status: Case A NOT triggered, Case B NOT triggered → proceeding to Step 2 (U).**

---

## AUDIT 2 — Step 2: is the oracle U a real measurement of future-prediction error? (2026-09-20)

**Hypothesis under test.** Only whether U is measurable and means what it claims — not whether it predicts
anything.

**What actually ran.** All 90 states. Simulator side: each state's chunk replayed **twice** from its exact
snapshot (180 replays, 1,440 env steps) to obtain a second rendering of the same physical future. Model
side: 4 feature extractions per state (online o_{t+7}, replay-1 o_{t+7}, replay-2 o_{t+7}, online o_{t+8}),
3 whole-future consequence samplings × 3 noise seeds, and a 256-token error-direction decomposition
(≈266 samplings/state) → ≈24,000 samplings, 10 s/state. Exclusions: none. Failures: none.
Artifacts: `futures_manifest.jsonl`, `uncertainty_records.jsonl`, `uncertainty/*.npz`.

### 1. Is the time alignment exact?
Yes, and it was verified physically rather than assumed. `H_pred` targets `o_{t+7}` (training sampler
indices [0,7]); the label is the frame the *undisturbed rollout* recorded at step t+7, i.e. after 7 executed
chunk actions. Re-executing the recorded chunk from the snapshot reproduces the end-effector and object
positions at that index to **≤4.7e-7 m** (median 0.0), so the label frame is the state the model was
supposed to predict. Using `o_{t+8}` instead changes U by only 0.087 → 0.090 (median), so the conclusion is
not sensitive to an off-by-one in the future index.

### 2. Are camera / preprocessing identical?
Yes: `H_real` uses the same `lam.extract_vision_features` call, the same 180° flip, the same 256×256, the
same ImageNet normalisation and the same bf16 autocast as `h_t` — it is literally the same helper the
prior run validated, applied to a different frame.

### 3. Is `H_real` really a future encoding and not the current frame again?
Yes. The "predict no change" baseline (encode `h_t` and call it the future) gives MSE **0.139** median,
while LaWM's prediction gives **0.087**; they are different tensors and LaWM beats the trivial baseline in
**75/90 states (83 %)**. Note the phase structure: trivial baseline 0.175 (approach) / 0.098 (pre-grasp) /
0.161 (manipulation) versus LaWM 0.087 / 0.083 / 0.097 — at the pre-grasp boundary the scene barely changes,
so LaWM's margin over "predict nothing" is small there.

### 4. What is the noise floor of U?
Two floors were measured. Same physics rendered twice: MSE **0.0041** median → the U signal is **16×** the
render floor (p25 12.4, worst state 1.9×). Online frame vs its replay: 0.014 median — larger, because
robosuite renders one 2 ms substep behind, a known artefact of the prior run. Both floors are well below
U = 0.087, but the worst-case state (SNR 1.9) is flagged: U is not equally trustworthy everywhere.

### 5. Is U affected by simulator stochasticity?
No. Repeated execution from the same snapshot gives **identical** end-effector positions (repeat difference
0.0 m in all 90 states); the only variation is rendering.

### 6. Does U have stable spatial structure?
U varies by token and correlates negatively with the predicted token's own norm (Spearman ≈ −0.55, see
AUDIT 3) — a confound that is reported rather than removed.

### Conservative conclusion at this point
"The oracle U is a real, well-aligned measurement of per-token future-prediction error, about 16× above the
rendering floor, and LaWAM's future prediction is better than a no-change baseline in 83 % of states —
though barely so at pre-grasp boundaries." Nothing yet about its usefulness.

---

## AUDIT 3 — Step 3: are U and S complementary or redundant? (2026-09-20)

**Hypothesis under test.** Whether U and S carry different information (Case C check). No claim about
consequence yet.

**What ran.** Pure analysis over the 90 states already collected (90 × 256 = 23,040 token pairs).
Artifacts: `summary_step34.json`, `metrics_step34.csv`.

### 1. Are U and S complementary?
Yes, strikingly so: within-state Spearman(U, S) has **median 0.024** (p25 −0.035, p75 0.091, full range
−0.20 … 0.19) and Pearson median 0.045. The pre-registered redundancy criterion (|ρ| ≥ 0.8) is **not** met
in any state. Top-quartile overlap is 1.25× chance (1.0 = independent, 4.0 = identical).

### 2. Do high-U/high-S tokens actually exist?
Yes, and all four quadrants are populated: high-U∧high-S 7.8 % of tokens (chance 6.25 %), high-U∧low-S
4.7 %, low-U∧high-S 8.2 %. So "unreliable but irrelevant" and "reliable but critical" tokens are both
common — which is exactly the premise the research question needs.

### 3. Is this confined to one task or phase?
No. The near-zero U–S correlation holds in all three suites and all three phases (per-phase medians in
`summary_step34.json:by_phase_token_level_spearman` and `metrics_step34.csv`).

### 4. Could this be an artefact of feature norm?
Partly, and it is reported as a limitation: **both** U and S correlate negatively with the predicted
token's L2 norm (Spearman ≈ **−0.55** for U, **−0.32** for S). A shared confound of that size would tend to
make U and S *positively* correlated, yet their observed correlation is ≈0 — so the independence is not
manufactured by the norm. But it does mean neither quantity is "pure": part of both is explained by how
large the token is.

### 5. Is it stable across metrics and epsilons?
Yes. U computed as per-token MSE, L2 or cosine distance gives the same picture (Spearman with the target
within 0.002 of each other), and S's token ranking is stable across ε (ρ ≈ 0.996–0.998) and across random
perturbation directions (ρ ≈ 0.90).

### Conservative conclusion at this point
"U and S are empirically independent within a state (|ρ| ≈ 0.02), both partly explained by token norm, and
all four U/S quadrants are populated. Case C (redundancy) does not trigger."

---

## AUDIT 4 — Step 4: does R explain action / execution consequence better than U, S or attention? (2026-09-20)

**Hypothesis under test.** The central one: is the *joint* quantity more informative than its parts and
than an attention baseline — and does it matter in the environment?

**What ran.** (a) Token level, all 90 states: correlations of 10 predictors against the realised per-token
action deviation. (b) Attention baseline: cross-attention weights of the 4 image-reading DiT blocks over
10 flow steps, 90 states, with an equivalence guard. (c) Environment level: 90 states × 4 chunk variants ×
3 noise seeds = **1,080 executed chunks** (8,640 control steps) from exact snapshots.
Artifacts: `attention_records.jsonl`, `consequence_records.jsonl`, `summary.json`, `figures/step4_*.png`.

### 1. Is R really more informative than U and S alone?
Split by level, because the answer differs:
- **Token ranking (within a state).** Spearman with the realised per-token deviation: U alone **0.238**
  [0.225, 0.251], attention alone **0.281** [0.246, 0.318], attention×U **0.375**, S alone **0.850**,
  attention×S **0.802**, **U×S 0.877** [0.869, 0.886], fitted R_linear (held-out episodes) 0.876. The paired
  gain of U×S over S alone is **+0.027** [0.026, 0.029], positive in **100 %** of states; over U alone
  +0.640; over attention alone +0.596. AUROC for the top-quartile tokens: 0.642 (U), 0.665 (attention),
  0.907 (S), **0.927 (U×S)**.
- **State-level risk.** R does **not** win: Spearman with E_action is 0.402 for mean U, 0.364 for Σ(U·S),
  0.226 for mean S. Predicting *which state* is risky is a different problem, and U alone is at least as
  good there.
- **Causal, equal budget (the strongest test).** Correcting 64 of 256 tokens with their true values:
  top-U selection recovers **55 %** of the oracle correction of the executed end-effector position; top-(U×S)
  selection recovers **74 %**; paired difference **+0.173** [0.116, 0.227], better in **72 %** of states, with
  only 28 of 64 tokens shared between the two selections. This is an intervention, not a correlation.

### 2. Is the improvement confined to one task or phase?
No. The equal-budget advantage holds in every phase (recovered U→U×S: 0.67→0.84 approach, 0.61→0.71
manipulation, 0.37→0.67 pre-grasp) and the token-level advantage is positive in 100 % of states across all
three suites.

### 3. Is it stronger in contact / manipulation phases?
Not in the way one might expect. The *absolute* consequence is largest in approach (1.22 mm vs 0.78 mm at
pre-grasp), and the *relative* benefit of U×S over U is largest at pre-grasp (0.37→0.67), where the future
is least used overall. The phase story is about where the future matters at all (approach), not about contact.

### 4. Is it just feature magnitude?
Partly-shared confound, quantified in AUDIT 3 (U ~ −0.55, S ~ −0.32 with token norm). Since the two are
nevertheless uncorrelated with each other, the product is not reducible to norm; but a norm-only baseline
was not run, and that is listed as a limitation.

### 5. Is it just that S already encodes the action error?
**This is the main threat to the token-level result and it is real.** The token-level target
errdev_j = ‖π(H_pred with token j ← real) − π(H_pred)‖ is, to first order, S_j(error direction)·‖error_j‖,
so a large part of S's 0.850 correlation is near-definitional. This is precisely why the decision rests on
(a) the *causal* equal-budget experiment, where the token sets differ and the outcome is executed motion,
and (b) the state-level analysis, where R does not win. The token-level number alone must not be read as
evidence of a new mechanism.

### 6. Does R still win against an attention baseline?
Yes, clearly at token level (+0.596 over attention alone, +0.502 over attention×U, +0.075 over attention×S,
each positive in 100 % of states). Notably **attention mass does not track sensitivity at all**: the action
head puts 42 % of its image-half attention on the future tokens, yet perturbing them changes almost nothing,
and attention is far flatter across tokens (p95/median ≈ 2.9) than sensitivity (≈ 9.3). "Why not just use
attention" now has a measured answer.

### 7. What is the size of the thing being predicted?
This is the finding that governs the final judgement. The **entire** realised future-prediction error is
worth a median **0.90 mm** [mean CI 0.95, 1.22] of executed end-effector displacement, on chunks that move
**85 mm** (≈1.1 %), while the flow sampler's own seed change is worth **0.47 mm** [0.49, 0.72]. Object pose
effects are ≈0.02 mm and the gripper command changes in 3 of 90 states. So R ranks correctly inside a very
small budget of influence.

### Engineering fault found and fixed during this step (stop-rule applied)
The first attention recorder disagreed with the deployed SDPA path by 0.05–0.09 in action units — ten times
larger than the effect under study. Cause: `Attention.prepare_attention_mask` returns a **boolean** mask,
which the recorder added numerically instead of using to mask, so it attended to the VLM half as well. All
attention numbers were discarded, the recorder was fixed, and equivalence is now **2.4e-7** on all 90 states.
No attention result from the faulty version survives in any deliverable.

---

## FINAL AUDIT (2026-09-20)

**Total executed work.** 30 rollout episodes (3 suites × 1 task × 10 independent resets, 30/30 successful);
90 probe states (3 phases × 10 episodes × 3 tasks); ≈208,000 action samplings for S; ≈24,000 for U and the
error decomposition; 90 attention recordings; 1,080 executed consequence chunks; 180 verification replays.
Excluded data: **none** — no state was dropped after its U or S was seen. Infrastructure failures: none in
the final pipeline; three were found and fixed before any result was used (batch-noise baseline, attention
mask, libero_goal env reset), and one reading was retracted (goal-suite availability).

**Did the research question change?** No. `protocol.yaml:research_question.fixed_text` is the question that
was answered, with amendments A1–A5 all logged before the corresponding measurement.

**Was anything trained or modified?** No. The checkpoint is untouched; `git status` shows exactly the same
four modified files as at the start (the pre-existing diagnostic hooks), and `code_changes.patch` is
byte-identical to the one from run `bd_20260914_093902`. Everything new lives in
`research/action_critical_reliability_probe/`.

**What would still overturn the conclusion?**
1. A checkpoint trained with `enable_flow_h_t1_scheduled_sampling=true` (this one never saw a real future as
   conditioning) could have a much larger future-utilisation budget, changing the magnitude verdict.
2. A norm-only token baseline could absorb part of the U×S advantage.
3. Out-of-distribution or perturbed observations were not tested; the whole probe is on successful,
   in-distribution rollouts of three tasks, and no failure cases exist in the sample to test risk prediction.
4. The equal-budget experiment uses an *oracle* correction; a learned U would be weaker.

**Final decision:** see `REPORT.md` §12 — **SUPPORTED (mechanism), with the magnitude on this checkpoint too
small to justify the long-term goal without first enlarging the future-utilisation budget.**
