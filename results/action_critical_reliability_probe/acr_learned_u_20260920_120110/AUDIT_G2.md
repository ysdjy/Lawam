# AUDIT_G2.md — Learned-U (G2) probe, run `acr_learned_u_20260920_120110`

Append-only. The prior run `acr_20260919_203850` is **read-only** throughout.

---

## G2 AUDIT 0 — repository, prior-run and code baseline (2026-09-20)

**The single hypothesis under test here.** None — this is a baseline verification. It establishes that the
frozen artefacts of the prior round are intact and that the policy path still reproduces them bit-exactly,
so that anything measured later is attributable to the new Learned-U work.

**What actually ran.** Read-only inspection of git state and the prior run's 31 artefacts; a numerical
cross-check of `summary.json` against every quantitative claim in `REPORT.md`; a code-level audit of
`m3_uncertainty.py` and `m2_sensitivity.py`; and a live model check (`learned_u/scripts/g2_audit0.py`) on
3 prior states drawn from the 3 different suites — 1 shared-encoding forward each, 1 baseline sampling,
32-row S batch, 1 permutation sampling, plus a full `state_dict` hash. Excluded data: none.
Artifact: `g2_audit0_model.json`.

### 1. Branch / commit / working tree
- branch **`research/action-critical-reliability-probe`**, HEAD **`4f9c641`**, parent `7d27b96` (= local
  `main` = upstream RLinf/LaWAM). Working tree clean, nothing staged.
- **Model diff carried by this branch vs upstream `main`: 4 files, +113/−9 lines** —
  `starVLA/model/framework/latent_world/runtime/{output_mapper,runner}.py`,
  `starVLA/model/framework/vlas/{flowmatching_expert,lawam}.py`. These are the default-off diagnostic
  hooks (`latent_override`, `future_override`, `initial_noise`, `return_diagnostics`) introduced by the
  earlier `branch_diagnostic` round.
  **Correction of a possible misreading:** the statement "the previous round changed no model code" refers
  only to `acr_20260919_203850`, which added none. It does **not** mean this branch is model-identical to
  upstream. Any G2 statement about "no model change" must be phrased against this branch's baseline.

### 2. Prior-run completeness
31 artefacts present; `sensitivity_records.jsonl`, `uncertainty_records.jsonl` and
`consequence_records.jsonl` have **90 lines each**; `summary.json` reports `n_states = 90` in all three
sections (step1 / step3_step4 / step4_environment).

### 3. `summary.json` vs `REPORT.md` — one defect found
Checked 11 headline figures. Eight agree exactly. **Five numbers in `REPORT.md` §7 (U–S complementarity)
were taken from an intermediate analysis run over 59 states, not from the final 90-state `summary.json`:**

| §7 claim | final `summary.json` (90 states) |
|---|---|
| Spearman(U,S) median **0.024** | **0.053** |
| p25 −0.035 | −0.015 |
| p75 0.091 | 0.095 |
| Pearson(U,S) **0.045** | **0.037** |
| low-U∧high-S **8.2 %** | **7.4 %** |
| S vs token-norm ρ ≈ **−0.32** | **−0.348** |

Agreeing: top-quartile overlap 1.25, high-U∧high-S 7.8 %, high-U∧low-S 4.7 %, U vs token-norm −0.55,
range −0.20…0.19, and every figure in §5, §6, §8, §9.

**Assessment.** The defect is a reporting error, not a data error: `summary.json`, `metrics*.csv` and the
raw records are the final 90-state products and are internally consistent. No conclusion changes —
Spearman(U,S) of 0.053 is as far from the pre-registered redundancy threshold (0.8) as 0.024 is, so
"U and S are empirically independent" stands, and the Case-C trigger remains untripped.
**Per the standing instruction not to modify prior-round outputs, `REPORT.md` was left untouched.**
For all G2 work, **`summary.json` is the authoritative source** and the §7 figures are not used.
A correction note in the prior run requires the user's explicit authorisation.

### 4. Is `H_real` still only an offline label?
Verified at code level and at runtime.
- In `m3_uncertainty.py`, the policy input is built at line 94 by
  `mc.make_example(d["primary"], d["wrist"], d["state"], d["lang"])` — current observation only. Every
  other use of `h_real` is (a) the U metrics, (b) the offline oracle token replacement, (c) the saved label.
- At runtime the example dict for all 3 sampled states contains keys
  `['action_hz','embodiment_id','lang','primary_image','state','wrist_image']` and **no future/real key**.

### 5. Is the S computation still frozen?
`m2_sensitivity.py` constants unchanged: `EPS = [0.271, 0.812, 2.706]`, `N_DIRS = 3`,
`NOISE_SEEDS = [101, 202, 303]` with **seed 101 used for every perturbation**, `BATCH = 32`,
`METRICS = pm.DISTANCE_KEYS`. `m5_consequence_chunks.py` uses the same seeds, `TOP_Q = 0.75`,
`EPS_MID_IDX = 1`, `METRIC = "l2_norm_all7"`.

### 6. Does the policy still reproduce the prior run?
On 3 states from 3 suites, against the stored arrays:

| check | result |
|---|---|
| `h_t1_pred` vs stored | **0.0** |
| `h_t` vs stored | **0.0** |
| baseline action chunk vs stored | **0.0** |
| S of the first 31 tokens (ε_mid, dir 0), **same 32-row batch layout as m2** | ≤ **4.3e-12** (float32 storage rounding) |
| future-token permutation → action change | 3.6e-7 – 4.8e-7 |

A first attempt compared S using a 13-row batch and disagreed by ~2e-8. That was **not** drift: AUDIT 0b of
the prior run had already established a ~5e-7 cross-batch-size reduction-order offset. The check was
corrected by replicating m2's batch composition rather than by relaxing the tolerance.

Permutation invariance holds at the float-reduction level (~4e-7, versus real effects of 1e-3…1e-2 and a
mean action magnitude of 0.225): **the action head still reads the future as an unordered bag**, which
constrains the Learned-U design (content-based, no token index).

### 7. Are the weights unchanged?
`LatentWorldPolicyBackend.state_dict()` **sha256 = `37a53b8c799c39725a18900eeaa687d1d2cebc24fca28d8e1f2881ca2870b523`**
over 2,555,179,360 parameters; flow head sha256 recorded separately in `g2_audit0_model.json`.
Checkpoint file 7,174,214,645 B, mtime 2026-09-14 01:24:55 — unchanged.
**This hash is the reference for every later training log: it must be identical before and after training
the U predictor.**

### 8. Could anything here be explained more simply / is there leakage?
No measurement was made that could leak: nothing was trained, no future information entered any policy
input, and no data split exists yet. The only judgement call was the S-comparison tolerance, resolved by
matching the batch layout instead of loosening the threshold.

### 9. Most conservative conclusion, and what would stop the project here
"The prior run's artefacts are complete and bit-reproducible by the current code and weights; one
reporting defect in `REPORT.md` §7 is recorded and does not affect any conclusion." If the weights hash or
the stored-array reproduction had failed, G2 would have been suspended until the baseline was restored.

**Status: G2 AUDIT 0 PASSED (with the §7 reporting defect recorded). Proceeding to Step 1 —
pre-registration of the U-only dataset and the episode/task splits.**

---

## G2 AUDIT 1 — new U-only dataset and split leakage (2026-09-20)

**The single hypothesis under test.** None yet. This entry certifies the dataset and the splits, so that
the later "U_hat generalises" claim cannot be an artefact of how the data were divided.

**What actually ran.** 6 pre-registered tasks × 10 independent reset episodes = **60 online rollouts** with
the unmodified policy through the official websocket server; state selection with the frozen phase rule;
160 snapshot replays ×2 for the render floor (2,560 env steps); U extraction on 160 states.
Artifacts: `rollouts/`, `states_manifest.jsonl` (160), `futures_manifest.jsonl` (160),
`u_dataset/` + `u_dataset_records.jsonl` (160), `splits.json`, `DATA_SPLIT.md`.
Excluded data: **none** — the one failed episode (libero_goal t5 ep9, 250-step timeout) is kept, as
pre-registered.

### 1. Token leakage?
No. Nothing is ever split at token level. The unit assigned to a split is the episode; all 256 tokens of a
state, and all states of an episode, always land in the same split. `no_state_id_in_two_splits` PASS.

### 2. Episode leakage?
No. `episode_disjoint_train_val`, `episode_disjoint_train_testA`, `episode_disjoint_val_testA` all PASS.
Train = episodes 0–5, val = 6–7, testA = 8–9 of the 4 training-pool tasks, by the rule fixed in
`PROTOCOL_G2.yaml` before collection.

### 3. Task leakage?
No, at three increasing levels of difficulty. `task_disjoint_train_testB1/B2/C` all PASS:
- **testB1** `libero_spatial t9` — unseen task, *seen* suite;
- **testB2** `libero_goal t5` — unseen task *and* unseen suite, and a non-prehensile *push* skill;
- **testC** the prior run's 3 tasks — unseen, and already carrying frozen S / attention / consequences.

### 4. Was any split changed because some task was hard to predict?
No. Tasks and splits were fixed in `PROTOCOL_G2.yaml` (frozen copy in the run directory, written at
12:08:49) before a single U value of the new tasks existed, and the U statistics were only looked at
afterwards. The one substitution decision that arose is documented in §7 below and was *not* made on the
basis of U.

### 5. Sizes

| split | states | episodes | tokens |
|---|---|---|---|
| train | 72 | 24 | 18,432 |
| val | 24 | 8 | 6,144 |
| testA (unseen resets, seen task) | 24 | 8 | 6,144 |
| testB1 (unseen task, seen suite) | 30 | 10 | 7,680 |
| testB2 (unseen task + suite) | 10 | 10 | 2,560 |
| testC (prior run, frozen S) | 90 | 30 | 23,040 |

160 new states + 90 prior = 250 states in total. The target was ~180 new states; the realised number is
160 because **libero_goal t5 is a pushing task with no grasp**, so `pre_grasp` and `manipulation` do not
exist for it and were recorded as missing rather than substituted (pre-registered policy). testB2 is
therefore approach-only, and every claim about it is qualified accordingly.

### 6. Is the new U comparable to the prior round?
Yes, which matters because Steps 5–7 evaluate on testC. New tasks vs prior run: U median **0.0875** vs
0.0867; render-floor SNR **16.2×** vs 16.0×; LaWM beats the "predict no change" baseline in **82.5 %** vs
83.3 % of states. Per-task U medians span 0.064 (object t6) – 0.0996 (spatial t9), so there is real
between-task variation to generalise across.

### 7. Engineering deviations, and why they are not scientific choices
- **Instrumentation fix (logged, verified).** `libero_goal t5` lists a goal *region*
  (`main_table_stove_front_region`) in `obj_of_interest`; it has no rigid body, so `body_pos` raised.
  Two changes were made **in this round's `s1_rollouts.py` only**: non-body entries are filtered out (and
  recorded in `episode.json:non_body_obj_of_interest_skipped`), and per-step records go through a wrapper
  that narrows `obj_of_interest` for the duration of the call so the shared read-only helper
  `sim_common.obs_record` remains the single source of truth. **`sim_common.py` was not modified.**
  Verified as a recording-level no-op on an already-collected task (`libero_spatial t2 ep0`): identical
  `objects_of_interest`, empty skip list, same record structure. (The rollout trajectory itself differs
  between collections because the policy server samples flow noise unseeded — unrelated to this change and
  true of the prior round as well.)
  This kept the *pre-registered* task instead of invoking the fallback list, which would have swapped a
  task for the convenience of my own logging code.
- **Physics reproduction.** 158/160 states reproduce the online end-effector and object positions from the
  snapshot exactly (0.0); **2 states** — both `libero_spatial t07` pre-grasp, a contact-rich "bowl on the
  stove" configuration — differ by 0.19 mm and 0.56 mm, while two replays of each agree exactly (0.0).
  **This does not touch U**: the U label is the *online* frame `o_{t+7}`, and the replay frames are used
  only to estimate the render floor, which for those two states is therefore slightly overestimated. Both
  states are kept and flagged.

### 8. Any oracle / future information in the inputs?
No. `g2_extract_u.py` asserts at runtime that the example dict contains no future-like key, and stores as
*inputs* only `h_t`, `H_pred` and `z` — all computed from `o_t`. `H_real` is stored nowhere in the input
arrays; it survives only through the scalar label `U_j` and the floor arrays.

### 9. Is the next step still necessary, and what would stop the project?
Yes: with leakage-free splits at three generalisation levels, the question "is U predictable before
execution?" is now well posed. Step 3 (mandatory simple baselines) comes first, precisely so that a learned
model is never credited with something `‖H_pred‖` or `‖H_pred − h_t‖` already explains. If the MLP fails to
beat those baselines on testA, the answer is **G2-A** and the side-channel line stops.

**Status: G2 AUDIT 1 PASSED. Proceeding to Step 3 (simple baselines, no training).**

---

## G2 AUDIT 2 — simple baselines vs the learned U predictor (2026-09-20)

**The single hypothesis under test.** Can per-token U be predicted from pre-execution information, and does
a learned estimator do so better than free statistics of `H_pred` and `h_t`?

**What actually ran.** 5 no-training baselines + a 1.46 M-parameter shared-token MLP (3 seeds, 1–2 s each,
early-stopped on validation Spearman) evaluated on 250 states / 64,000 tokens across 6 splits; a
permutation audit on every state. Artifacts: `METRICS_U_baselines.csv`, `summary_baselines.json`,
`METRICS_U.csv`, `summary_learned_u.json`, `TRAIN_LOG.jsonl`, `models/u_mlp_seed{0,1,2}.pt`.
Excluded data: none.

### 1. How do the simple baselines perform?
Far better than anticipated. **`B2 = ‖H_pred_j − h_t_j‖` (and its twin `B3 = 1 − cos`) reach Spearman
0.87–0.945 with oracle U on *every* split, with no training at all**, including the unseen suite:

| split | B1 ‖H_pred‖ | **B2 ‖H_pred − h_t‖** | B4 linear(6 stats) |
|---|---|---|---|
| train | 0.606 | 0.929 | 0.926 |
| testA | 0.614 | 0.931 | 0.928 |
| testB1 | 0.681 | **0.945** | 0.936 |
| testB2 | 0.631 | 0.873 | 0.875 |
| testC | 0.548 | 0.919 | 0.915 |

The mechanism is not mysterious: where LaWM predicts a large change from the present, the future really is
far from the present and prediction error concentrates; where it predicts "no change" (background), error is
small. The mandatory token-norm baseline B1 is clearly weaker (0.55–0.68) — so U is **not** merely feature
magnitude, which answers one of the prior round's open confounds.

### 2. Does the MLP really beat the token-norm / change baselines?
**Only on tasks it was trained on.** Paired per-state differences, episode-bootstrapped:

| split | U_hat | B2 | U_hat − B2 (ΔSpearman) | U_hat better in |
|---|---|---|---|---|
| train | 0.987 | 0.929 | +0.058 [+0.049, +0.068] | 100 % |
| **testA** (unseen resets, seen task) | 0.959 | 0.931 | **+0.028 [+0.020, +0.039]** | 100 % |
| **testB1** (unseen task, seen suite) | 0.901 | 0.945 | **−0.044 [−0.050, −0.037]** | 3 % |
| **testB2** (unseen task + suite) | 0.840 | 0.873 | **−0.034 [−0.058, −0.011]** | 20 % |
| **testC** (prior round's 3 tasks) | 0.909 | 0.919 | **−0.009 [−0.026, +0.006]** | 48 % |

So the learned estimator wins on held-out *episodes* and loses or ties on every held-out *task*, with CIs
excluding zero in the losing direction on B1 and B2. On the ranking metrics the picture is mixed rather than
uniformly bad: on testC, U_hat has higher top-64 recall (0.821 vs 0.762), NDCG@64 (0.932 vs 0.875) and
AUROC (0.960 vs 0.941) than B2 even though its Spearman is marginally lower — the MLP orders the *head* of
the distribution better while being slightly worse over the whole ranking.

### 3. Is the held-out episode split meaningful, and the held-out task split?
Both are meaningful and they disagree, which is exactly why both were pre-registered. Episode holdout alone
(testA) would have supported "the MLP beats the free baseline"; task holdout shows that advantage is
task-bound. Phase breakdown over all test splits is flat (approach 0.908 vs 0.917, pre-grasp 0.913 vs 0.931,
manipulation 0.913 vs 0.922) — the failure to transfer is not phase-specific.

### 4. Permutation equivariance?
**PASS, exactly 0.0** on all 250 states: permuting `h_t` and `H_pred` permutes `U_hat` identically. This is
architectural (shared weights applied per token, no token index in the features) and the test confirms the
implementation rather than the design. It matters because a measured control shows U *does* carry positional
structure: a pure **token-index** predictor reaches Spearman −0.31 (train) / −0.29 (testC). A model with
access to position could have exploited that; this one provably cannot.

### 5. Overfitting?
Yes, mildly and visibly: train Spearman 0.987 vs validation 0.959, and the gap widens on unseen tasks
(0.84–0.91). Early stopping on validation Spearman was in force, and three seeds agree to ±0.001, so this is
a generalisation limit of the feature set, not an optimisation artefact.

### 6. Did the predictor use any future information?
No. Inputs are `[h_t_j, H_pred_j, H_pred_j − h_t_j, z]`, all computed from `o_t`; `g2_extract_u.py` asserts
at extraction time that no future-like key can reach the policy example, and `H_real` appears in the dataset
only inside the scalar label. The training script never imports or loads LaWAM, and the checkpoint's
size/mtime are identical before and after training (`TRAIN_LOG.jsonl:lawam_checkpoint_untouched`).

### 7. Was anything changed after seeing results?
One instrument was corrected **before** any conclusion was drawn: the original `B0_constant` baseline
assigned every token the same score, and `argsort` broke the ties by index — silently turning the
"no information" floor into a token-index predictor (Spearman +0.31, recall 0.119 < chance). It was replaced
by a seeded random score, which now returns the analytic expectation exactly (Spearman −0.000, recall 0.248,
AUROC 0.500), and the index ordering was kept as an explicit, separately reported control `B0x_token_index`.
No split, threshold or task selection was changed at any point.

### 8. Is there a simpler explanation for the headline result?
Yes, and it *is* the headline result: **the simplest available statistic already predicts U almost as well
as anything else, and better than the learned model outside the training tasks.** The learned model's extra
capacity buys task-specific structure that does not transfer.

### 9. Verdict against the pre-registered decision tree
This is **Case G2-A (SIMPLE BASELINE SUFFICIENT)**: `U_hat` does not clearly beat
`‖H_pred − h_t‖` on held-out tasks — it loses on testB1/testB2 with CIs excluding zero and ties on testC.
Per `PROTOCOL_G2.yaml:decision`, a complex reliability estimator is **not** warranted, and the learned-U
main line stops here.

**A finding that is not a failure, and must not be buried:** the *deployability* question G2 was created to
answer comes out **positive**. U — defined against a future that only exists after execution — is
predictable before execution with Spearman ≈ 0.92 by a statistic that costs nothing, requires no training,
and transfers across tasks and suites. What is refuted is the need to *learn* it.

### 10. What would stop, and what could still be asked
The pre-registered stop applies to the learned estimator. It leaves one well-posed question that the
protocol's own selector list already anticipated (`top_change_norm`): whether the deployable free proxy,
combined with the frozen S, retains the oracle U×S advantage in the equal-budget causal execution test.
Running it requires one logged amendment (adding `top_change_norm x S` to the selector list) and consumes
only existing holdout-C artefacts. **It is not started without the user's decision**, because
`PROTOCOL_G2.yaml` gates Steps 5–7 on the learned-U result, and that result is negative.

**Status: G2 AUDIT 2 — Case G2-A. Learned-U line stopped. Steps 5–7 held pending a user decision.**

---

## FINAL G2 AUDIT (2026-09-20)

**Total executed work.** 60 online rollouts (6 pre-registered tasks × 10 independent resets, 59 successes /
1 kept failure); 160 probe states + 320 snapshot replays; U extraction on 160 states; 5 no-training
baselines and a 1.46 M-parameter MLP (3 seeds) evaluated on **250 states / 64,000 tokens** across 6 splits;
250 permutation audits. Excluded data: **none**. Infrastructure faults: two found and fixed before any
conclusion (the `obj_of_interest` region crash, the tie-broken-by-index `B0` floor); both are recorded with
their verification in AUDIT 1 §7 and AUDIT 2 §7.

**Did the question change?** No. `PROTOCOL_G2.yaml:question.fixed_text` is the question answered. Task
selection, splits, inputs, architecture, baselines and decision rules were all frozen before the
corresponding measurement; the only change made after seeing numbers was the correction of a broken
*control* (B0), which moved it to its analytic expectation and made the comparison stricter, not looser.

**Was anything trained that should not have been?** No. The only trained object is the diagnostic MLP.
LaWAM was never imported by the training script; the checkpoint's size and mtime are identical before and
after (`TRAIN_LOG.jsonl`), and its `state_dict` sha256 `37a53b8c…b523` was verified in AUDIT 0. No S
predictor, no side-channel, no gate, no RL, no OOD experiment.

**Was any oracle future information used as an input?** No — asserted at runtime in `g2_extract_u.py` and
re-checked in AUDIT 0 §4. `H_real` exists in this round only inside the scalar label and the floor arrays.

**Was any split or threshold changed because of results?** No. `DATA_SPLIT.md` was generated mechanically
from the frozen protocol with seven leakage assertions, all PASS, before training.

**Verdict: Case G2-A — SIMPLE BASELINE SUFFICIENT.** The learned estimator beats the free statistic
`‖H_pred − h_t‖` only on tasks it was trained on (testA +0.028 [+0.020, +0.039], 100 % of states) and loses
or ties on every held-out task (testB1 −0.044 [−0.050, −0.037]; testB2 −0.034 [−0.058, −0.011];
testC −0.009 [−0.026, +0.006]). A complex reliability estimator is not warranted.

**The part of the result that is positive and must not be buried:** the deployability question G2 was built
to answer is answered **yes**. Per-token U, defined against a future that only exists after execution, is
predictable before execution at Spearman ≈ 0.92 — across unseen tasks and an unseen suite — by a
zero-parameter statistic. G2 refutes the need to *learn* reliability, not the existence of a pre-execution
reliability signal.

**What would overturn this.**
1. A feature set or architecture that transfers across tasks better than `‖H_pred − h_t‖` — but escalation
   requires a new audit entry and a reason why more capacity should transfer better than no parameters.
2. Evidence that rank correlation is the wrong target and that calibrated magnitude (where the MLP may do
   better) is what a downstream mechanism needs.
3. OOD/failure conditions, where the free statistic might degrade while a learned one holds up — untested,
   and the subject of gate G3.

**Steps 5–7 (S re-introduction, `U_hat × S`, equal-budget causal execution) were NOT run**, because
`PROTOCOL_G2.yaml` gates them on a positive learned-U result. `METRICS_SELECTION.csv` records this
explicitly rather than containing placeholder numbers. The one question that survives — whether the *free*
proxy preserves the oracle `U × S` advantage — is described in `REPORT_G2.md` §16 and
`LONG_TERM_PLAN_G2.md` §2 and awaits an explicit decision.

**Status: G2 complete. Decision G2-A. No further experiment started.**
