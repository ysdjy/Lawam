# Action-Critical Reliability probe — REPORT

run_id `acr_20260919_203850` · 2026-09-19/20 · commit `7d27b96` (working tree unchanged by this round)
checkpoint `results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt`

Direction-assessment round. No training, no new loss, no architecture change, no checkpoint modification.
Everything below is measured on one LaWAM LIBERO SFT checkpoint in simulation.

---

## 1. Research Question

Fixed before any measurement (`protocol.yaml:research_question.fixed_text`):

> In LaWAM's predicted future latent H_pred, (i) does the downstream action head respond to different future
> tokens differently (Downstream Action Sensitivity **S**), (ii) is the future-prediction unreliability **U**
> complementary to S, and (iii) does a joint quantity **R = f(U,S)** explain downstream action error /
> execution risk better than U or S alone?

Explicitly *not* studied this round: what each latent dimension means, cross-stage latent alignment,
decoding latents to human semantics, where attention should look, multi-candidate action generation.

## 2. Short-Term Goal vs Long-Term Goal

**Short term (this round):** decide whether the phenomenon exists — is S non-uniform, is U complementary,
does R predict consequence better than U / S / attention. Allowed conclusions: SUPPORTED / NOT SUPPORTED /
INSUFFICIENT EVIDENCE / PIVOT TO FUTURE UTILIZATION.

**Long term (not started):** an Action-Critical Reliability side-channel injected at the future→action-head
interface. Written up in `LONG_TERM_PLAN.md`; nothing from it was implemented or trained.

## 3. Exact LaWAM Interface Audited

Read from the loaded checkpoint, not from config (`environment_audit.md`, `audit0_model.json`):

| item | value |
|---|---|
| H_pred = `h_t1_pred` | **[1, 256, 768]** → N = 256 tokens, D = 768 |
| h_t | [1, 256, 768], DINOv3 penultimate + LayerNorm (every token norm = √768 = 27.713) |
| h_vlm | [1, 214, 2048] bf16 |
| action chunk | [1, 8, 32]; 7 dims used; `floor(horizon_sec 0.4 × action_hz 20) = 8` executed steps |
| future supervision index | `o_{t+7}` (`num_frames=2`, `sec_chunk=0.4` → [0,7]) |
| flow sampler | 10 Euler steps, CFG scale 1.0 (inactive), initial noise `randn` |
| `use_state` | false — state never reaches the action head |
| `num_target_vision_tokens` | −1 → **no learnable future query tokens**; H_pred enters only as cross-attention keys |
| blocks that read the image half | **2, 6, 10, 14** of 16 (the other even blocks read VLM tokens; odd blocks are self-attention) |
| `enable_flow_h_t1_scheduled_sampling` | false → the action head was trained on the *predicted* future only, never on a real one |

Two structural facts established by measurement:

1. **The action head consumes H_pred as an unordered bag.** Permuting the 256 future tokens changes the
   action chunk by ≤4.5e-8 (0.0 in most states): the DiT adds no positional embedding to condition tokens.
   Token-level quantities can therefore only act through token *content*, never position.
2. **The inference path is bit-exact** given the flow noise: repeated identical calls differ by 0.0, and
   identical rows inside one batch are bit-identical (different batch *sizes* differ by ~5e-7, which is why
   every baseline in this study lives inside the same batch as its perturbations).

## 4. Experimental Protocol

Pre-registered in `configs/protocol.yaml` (copied here) before any S/U number existed; amendments A1–A5 are
listed there with reasons and repeated in `AUDIT_LOG.md`.

- **States.** 3 LIBERO suites × 1 task each — `libero_spatial` t2 (bowl→plate), `libero_object` t0 (soup→
  basket), `libero_goal` t8 (bowl→plate, kitchen scene) — × 10 **independent reset episodes** × 3 execution
  phases = **90 states**. Phases are derived from the executed trajectory: *approach* (median chunk boundary
  with gripper open and eef > 6 cm from the target), *pre_grasp* (last boundary before the first commanded
  close), *manipulation* (first boundary after the object is lifted > 2 cm). All 30 rollouts succeeded.
- **S.** For each of the 256 tokens: `h_j' = h_j + ε·u`, 3 unit directions × 3 ε = 1 %, 3 %, 10 % of the mean
  token norm (0.271 / 0.812 / 2.706). A real semantic change of a token is ≈10–13 in L2, so even the largest
  ε is ~21 % of one. `S_j = D(A_j, A_0)/‖δ_effective‖`, everything else (observation, instruction, h_vlm, z,
  h_t, CFG, steps, flow noise) held fixed. ≈208,000 action samplings.
- **U.** Oracle only: `H_real = lam.extract_vision_features(o_{t+7})` from the undisturbed rollout, same
  encoder/flip/resize/normalisation as h_t. Per-token MSE / cosine / L2. H_real never enters a policy input.
- **Consequence.** Level 1: `E_action = D(π(H_pred), π(H_real))` at identical flow noise. Level 2: 90 states
  × 4 chunk variants × 3 noise seeds = **1,080 executed chunks** from exact simulator snapshots.
- **Controls.** repeat rows; random equal-norm perturbation of all tokens; token permutation; future scaled
  ±5 %; future zeroed; future replaced by h_t; different flow-noise seeds; attention baseline.

## 5. Future Utilization / S Results

| quantity (action units, l2 over the chunk) | median over 90 states |
|---|---|
| numerical floor (repeat, in-batch) | **0.0** |
| token permutation | 4.5e-8 |
| single token at ε_mid, median token | 6.9e-6 |
| single token at ε_mid, most sensitive token | 1.5e-4 (≈ **0.11 mm** of commanded motion) |
| all 256 tokens perturbed at ε_mid | 3.8e-4 |
| future scaled ×1.05 | 2.5e-3 |
| **flow-noise seed change (nuisance)** | **4.5e-3** |
| future replaced by h_t | 1.1e-2 |
| future zeroed | 1.4e-2 |
| mean \|action\| | 0.225 |

- **S is measurable and strongly non-uniform**: p95/median = **9.26** (bootstrap CI of the mean over
  episodes [9.08, 10.06]), max/median 22.8, Gini 0.64.
- **Ranking is extremely stable within a state**: Spearman across ε = 0.998 / 0.996; across random
  directions = 0.901. Across different states of the same suite only 0.53–0.58 — important tokens are
  partly shared, partly state-specific.
- **Local regime confirmed**: deviations scale ×2.95 for a ×3.0 increase in ε (and ×3.50 for ×3.33), so the
  perturbations are not OOD artefacts.
- **Future use is phase-dependent**: zeroing the future / flow-noise = **10.2 (approach)**, 3.3
  (manipulation), **1.7 (pre-grasp)**. The future channel matters mainly while approaching and is close to
  irrelevant at the pre-grasp boundary.
- **Single-token effects largely cancel**: the sum of 256 single-token deviations is ~10× the effect of
  perturbing all 256 at once.

## 6. U Results

| quantity (per-element feature MSE) | median |
|---|---|
| U (H_pred vs H_real at o_{t+7}) | **0.087** |
| render floor (same physics rendered twice) | 0.0041 → **SNR 16×** (worst state 1.9×) |
| online-vs-replay floor (1 substep render lag) | 0.014 |
| trivial "no change" baseline (h_t vs H_real) | 0.139 — LaWM beats it in **83 %** of states |
| U against o_{t+8} instead of o_{t+7} | 0.090 (conclusion not sensitive to the index) |

Physics reproduction from snapshots is exact (≤4.7e-7 m), and repeated execution gives identical
end-effector positions, so U carries no simulator stochasticity. Per phase, LaWM's margin over the trivial
baseline is large in approach (0.087 vs 0.175) and small at pre-grasp (0.083 vs 0.098).

## 7. U–S Complementarity

- Within-state **Spearman(U, S) = 0.024** (p25 −0.035, p75 0.091, range −0.20…0.19); Pearson 0.045.
  The pre-registered redundancy criterion (|ρ| ≥ 0.8) is met in **0 of 90** states.
- Top-quartile overlap 1.25× chance. All four quadrants populated: high-U∧high-S 7.8 %, high-U∧low-S 4.7 %,
  low-U∧high-S 8.2 % (chance 6.25 %).
- **Confound reported**: both correlate with the token's own norm (U ρ ≈ −0.55, S ρ ≈ −0.32). Since they are
  uncorrelated with *each other*, the independence is not produced by that shared confound — but neither
  quantity is "pure".

## 8. U vs S vs R vs Attention

Token-level target = the action deviation actually caused by replacing token *j* with its true value
(mean Spearman over 90 states; CI = episode bootstrap):

| predictor | Spearman | CI | AUROC (top quartile) |
|---|---|---|---|
| U only | 0.238 | [0.225, 0.251] | 0.642 |
| attention only | 0.281 | [0.246, 0.318] | 0.665 |
| attention × U | 0.375 | [0.342, 0.407] | 0.727 |
| S only | 0.850 | [0.841, 0.859] | 0.907 |
| attention × S | 0.802 | [0.789, 0.815] | 0.884 |
| **U × S** | **0.877** | [0.869, 0.886] | **0.927** |
| R_linear (αU+βS+γUS, fitted on 60 % of episodes, evaluated held-out) | 0.876 | [0.868, 0.885] | 0.923 |

Paired: U×S beats S alone by +0.027 [0.026, 0.029] in **100 %** of states, U alone by +0.640, attention alone
by +0.596, attention×S by +0.075 (100 %). The learned linear combination does not beat the plain product.

**Attention ≠ sensitivity, measured.** The action head places **42 %** of its image-half attention mass on
the future tokens (consistently across blocks 2/6/10/14), yet perturbing those tokens barely moves the
action; attention is also much flatter across tokens (p95/median ≈ 2.9) than sensitivity (≈ 9.3).

**State level, R does not win.** Predicting *which state* has a large E_action: mean U 0.402, Σ(U·S) 0.364,
mean S 0.226, Σ(U·attention) 0.251. Token ranking and state-level risk are different problems.

## 9. Environment-Level Consequences

1,080 executed chunks from exact snapshots (`consequence_records.jsonl`):

| quantity | median | mean CI |
|---|---|---|
| motion of the chunk itself | 85.0 mm | — |
| **treatment**: π(H_real) vs π(H_pred), same noise | **0.90 mm** (1.1 % of the motion) | [0.95, 1.22] |
| **nuisance**: π(H_pred) at a different noise seed | **0.47 mm** | [0.49, 0.72] |
| treatment / nuisance | 1.90 | — |
| target-object displacement difference | 0.021 mm | — |
| gripper command changed | 3 of 90 states | — |

**Equal-budget causal test** — correct 64 of 256 tokens with their true values and execute:

| selection | fraction of the oracle correction recovered |
|---|---|
| top-U tokens | 0.55 (mean CI [0.44, 0.58]) |
| **top-(U×S) tokens** | **0.74** (mean CI [0.63, 0.73]) |
| paired difference | **+0.173** [0.116, 0.227], better in **72 %** of states |

The two selections share only 28 of 64 tokens, and the advantage holds in every phase (approach 0.67→0.84,
manipulation 0.61→0.71, pre-grasp 0.37→0.67).

## 10. Confounders and Limitations

1. **The token-level target is partly definitional.** errdev_j ≈ S_j(error direction)·‖error_j‖, so much of
   S's 0.850 correlation is near-tautological. The decision therefore rests on the equal-budget *causal*
   experiment and the state-level analysis, not on that number.
2. **Magnitude.** The entire realised prediction error is worth ≈0.9 mm of executed motion on an 85 mm
   chunk — only ~1.9× the sampler's own noise. Everything R ranks lives inside that small budget.
3. **This checkpoint never saw a real future as conditioning** (`scheduled_sampling=false`,
   `detach_future_feature=true`), so feeding H_real is an out-of-training-distribution condition, and the
   small future-utilisation budget may be a property of *this* training recipe rather than of LaWAM.
4. **Oracle U.** A learned reliability estimator would be strictly weaker than this oracle.
5. **No failures in the sample.** All 30 rollouts succeeded, so "execution risk" could only be measured as
   displacement, never as failure prediction; AUROC against real failures was not possible.
6. **Norm confound** (§7); a norm-only baseline was not run.
7. **Scope.** 3 tasks × 1 checkpoint × 0.4 s horizon, in-distribution, no OOD / occlusion / distractor tests.
8. **Bag-of-tokens.** Any future mechanism must act through content, not token position (§3).

## 11. Evidence For / Against the Hypothesis

**For.**
- S exists, is ~9× non-uniform across tokens, and its ranking is near-deterministic within a state.
- U is a real measurement, 16× above its noise floor, and beats a no-change baseline in 83 % of states.
- U and S are empirically independent (ρ ≈ 0.02) with all four quadrants populated — the premise of the
  research question holds.
- U×S ranks action-relevant tokens better than U, S, attention, attention×U and attention×S, in 100 % of
  states, and better than a fitted linear combination.
- In a causal equal-budget intervention measured by executed motion, U×S recovers 74 % of the oracle
  correction versus 55 % for U (+0.173 [0.116, 0.227]).

**Against.**
- The whole quantity being apportioned is ~0.9 mm of executed displacement, ≈1.9× the flow sampler's own
  noise and ≈1 % of the chunk's motion.
- Single future tokens are worth ~1e-4 in action units (~0.1 mm); token effects largely cancel.
- At the state level, U alone predicts E_action as well as U×S — R does not identify risky *states*.
- Most of the token-level margin of S is near-definitional.
- Future use is concentrated in the approach phase and is almost absent at pre-grasp, the phase where
  control errors matter most for grasping.

## 12. Final Decision

**SUPPORTED — for the mechanism, at token level, on this checkpoint; with the magnitude too small to
justify the long-term goal as-is.**

Concretely: the phenomenon the research question posits is real and reproducible — prediction error and
downstream sensitivity are independent sources of information, their product is the better predictor of
which parts of the predicted future actually matter to the action, and it beats the obvious attention
baseline both correlationally and in a causal equal-budget intervention. What is *not* established is that
acting on this would improve control: on this checkpoint the entire future-prediction error moves the
executed trajectory by about 0.9 mm against 0.47 mm of sampler noise, so a side-channel built on it has
almost no headroom to work with.

This is **not** Case A (S is measurable, the future is used), **not** Case B (S is strongly non-uniform and
rank-stable), **not** Case C (U and S are independent and their product beats both). It matches Case D on
the informational criteria, while failing the practical premise behind the long-term goal.

## 13. Recommended Next Step

In order, and none of them is "build the side-channel":

1. **Measure the future-utilisation budget as a function of training recipe, not architecture.** The single
   most decisive experiment is to obtain or fine-tune a checkpoint with
   `enable_flow_h_t1_scheduled_sampling=true` (action head trained with real futures mixed in) and re-run
   *this exact protocol*. If the treatment/nuisance ratio rises from ≈1.9 to something substantial, the
   long-term goal becomes worth pursuing; if it does not, Action-Critical Reliability has no room to act
   regardless of how well U×S ranks tokens.
2. **Test whether the ranking survives a learned U.** Replace the oracle U with a cheap predictor
   (e.g. LaWM decoder ensemble disagreement or a small head on h_t/z) and repeat the equal-budget causal
   test. If a learnable U loses most of the +0.173, the mechanism is not deployable.
3. **Add the missing baselines and failure cases**: a token-norm-only selector, and states drawn from
   *failed* episodes (deliberately perturbed observations / distractors) so that "execution risk" can be
   measured as failure, not only as millimetres.
4. Only if 1–3 come out positive, proceed to `LONG_TERM_PLAN.md`.

## Deliverables

`environment_audit.md`, `protocol.yaml`, `AUDIT_LOG.md` (AUDIT 0, 0b, 1, 2, 3, 4, FINAL),
`phase0_equivalence.json`, `phase0_batch_noise.json`, `audit0_model.json`, `audit0_sim.json`,
`states_manifest.jsonl` (90), `futures_manifest.jsonl` (90), `sensitivity_records.jsonl` (90),
`uncertainty_records.jsonl` (90), `attention_records.jsonl` (90), `consequence_records.jsonl` (90),
`metrics.csv` (90 × 97), `metrics_sensitivity.csv`, `metrics_step34.csv`, `summary.json`,
`summary_step1.json`, `summary_step34.json`, `figures/step1_sensitivity_overview.png`,
`figures/step4_consequences.png`, `code_changes.patch` (unchanged from the previous run — no model code was
modified this round), `LONG_TERM_PLAN.md`, raw arrays under `sensitivity/`, `uncertainty/`, `attention/`,
`consequence_chunks/`, `futures/`, `states/`, `rollouts/`.
Code: `research/action_critical_reliability_probe/` (see `README.md` for the reproduction order).
