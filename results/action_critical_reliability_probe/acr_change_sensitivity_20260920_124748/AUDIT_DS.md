# AUDIT_DS.md — predicted-change × sensitivity (DS) round, `acr_change_sensitivity_20260920_124748`

Append-only. Both prior runs (`acr_20260919_203850`, `acr_learned_u_20260920_120110`) are **read-only**.

---

## DS AUDIT 0 — prior artefacts, definitions and baseline integrity (2026-09-20)

**The single hypothesis under test.** None — this certifies that every quantity this round reads is the
frozen quantity it claims to be, so that Q1 and Q2 are answered on round-1 data rather than on anything
recomputed here.

**What actually ran.** Read-only inspection of git state and of both prior run directories; a file-level
reconciliation of all frozen products against the 90-state manifest; shape/dtype readout of the arrays this
round consumes; a re-run of `g2_audit0.py` (3 states, full `state_dict` hash). No rollouts, no S, no
attention, no training. Excluded data: none. Artifacts: `PROTOCOL_DS.yaml` (frozen before any statistic),
`g2_audit0_model.json`.

### 1. Branch / commit / working tree
`research/action-critical-reliability-probe` @ **`b733f07`** ("research(G2): …Case G2-A"), parent `4f9c641`,
grandparent `7d27b96` (= local `main` = upstream RLinf/LaWAM). Working tree **clean**. The branch carries
the 4-file, +113/−9 diagnostic-hook diff against upstream that AUDIT 0 of the G2 round documented; this
round adds no model code.

### 2. Are the 90 holdout-C states complete?
Yes, and every derived product matches the manifest **exactly**:

| product | count | identical to the 90-state manifest? |
|---|---|---|
| `sensitivity/*.npz` | 90 | yes |
| `uncertainty/*.npz` | 90 | yes |
| `attention/*.npz` | 90 | yes |
| `consequence_chunks/*.npz` | 90 | yes |
| `futures/*.npz` | 90 | yes |
| `consequence_records.jsonl` | 90 | yes |
| `uncertainty_records.jsonl` / `sensitivity_records.jsonl` / `attention_records.jsonl` | 90 each | yes |

**One discrepancy found and resolved: `states/` holds 120 `.npz`, not 90.** The 30 extra files are exactly
the `*_s0000_approach.npz` produced by the *first* state-selection pass, before round-1 amendment A5
changed the approach phase from "first qualifying boundary" (which always landed on step 0) to "median
qualifying boundary". The manifest was rewritten; the superseded files were never deleted.
Verified consequences: every one of the 90 manifest entries resolves to an existing file (0 missing), and
**none of the 30 orphans appears in any downstream product** (each keyed set equals the manifest exactly,
orphans present = 0 in all five directories and all four record files). They are stale files on disk, not
data in any analysis. They are left in place (prior runs are read-only) and flagged here.

### 3. What this round reads, and its provenance

| symbol | file / key | shape | dtype |
|---|---|---|---|
| `h_t` | `sensitivity/<s>.npz:h_t` | [256, 768] | float16 |
| `H_pred` | `sensitivity/<s>.npz:h_t1_pred` | [256, 768] | float16 |
| `H_real` | `uncertainty/<s>.npz:h_real` | [256, 768] | float16 |
| `U` (authoritative) | `uncertainty/<s>.npz:U_U_l2` | [256] | float32, computed in float64 before storage |
| `S` (frozen) | `sensitivity/<s>.npz:S_norm[1].mean(dirs)[:, 'l2_norm_all7']` | [3,3,256,6] → [256] | float32 |
| attention | `attention/<s>.npz:attention_future`, blocks [2, 6, 10, 14] | [256] | float32 |
| ε grid | `sensitivity/<s>.npz:epsilons` = [0.271, 0.812, 2.706] | — | — |

**S is read, never recomputed** — `PROTOCOL_DS.yaml:symbols.S` states the rule, and no script in
`change_sensitivity/` calls `m2_sensitivity.py`.

**Precision caveat, registered before use:** the three feature tensors are stored as float16 whereas the
authoritative `U` was computed in float64 before storage. D, G and cos θ will therefore be computed from
float16 inputs. Step 1 must first check that U recomputed from these float16 arrays agrees with the stored
U; if it does not, the geometry analysis is suspended (`PROTOCOL_DS.yaml:data.precision_note`).

### 4. Are the causal-experiment settings the same as round 1?
Yes, and they are frozen: token budget **64 of 256** (`consequence_chunks/<s>.npz:token_budget = 64`),
flow-noise seeds **[101, 202, 303]** (all three chunk variants stored per seed), execution from the exact
simulator snapshot named in the manifest, and the round-1 definition of "oracle correction" — the executed
end-effector displacement between π(H_real) and π(H_pred) at the same seed. None of these may change this
round, for any reason.

### 5. Round-1 causal reference this round must be compared against
From `acr_20260919_203850/summary.json:step4_environment` (median over 90 states unless stated):

| quantity | value |
|---|---|
| chunk motion | **85.0 mm** |
| treatment: π(H_real) vs π(H_pred) | **0.904 mm** |
| nuisance: different flow-noise seed | **0.474 mm** |
| Recovery(top-U) | **0.545** |
| Recovery(top-U×S) | **0.742** |
| Oracle margin (U×S − U), paired | median **0.147**, mean +0.173 [+0.116, +0.227] |

### 6. Are the weights unchanged?
Yes. `state_dict` sha256 **`37a53b8c…b523`** over 2,555,179,360 parameters — identical to the value recorded
in the G2 round and to `PROTOCOL_DS.yaml`. Checkpoint 7,174,214,645 B, mtime 2026-09-14 01:24:55. The
re-run also reproduced stored `h_t1_pred` / `h_t` / baseline actions at **0.0** and the frozen S at ≤4.3e-12,
and re-confirmed future-token permutation invariance (~4e-7).

### 7. Does the round-1 ERRATA affect this round?
No. `ERRATA.md` corrects five §7 figures of round 1's `REPORT.md` that were taken from a 59-state
intermediate analysis (e.g. Spearman(U,S) 0.024 → **0.053**). This round does not use any §7 figure; it
reads the raw arrays and `summary.json`, which were always the final 90-state products. The corrected
value strengthens nothing and weakens nothing here — U and S remain effectively independent, which is why
`D × S` is worth testing at all.

### 8. Use of oracle information in this round, declared in advance
- `H_real`, `G`, `cos θ`, `|D − G|` and `U` are **oracle** and are used **only** (a) to explain the D→U
  relation in Step 1 and (b) as evaluation instruments in Step 2 (the token repair and the reference
  selectors E/F).
- The quantity under test, `D`, and the candidate `D × S`, use **no** future information at selection time.
- No oracle quantity enters any selector other than the explicitly labelled oracle references.

### 9. Is there a simpler explanation for anything so far, and what would stop the round?
Nothing has been measured yet. The known risk, already named in `PROTOCOL_DS.yaml`, is that the D→U
relation is a length-geometry artefact rather than a model property; Step 1's counterfactual baselines
B4a/B4b exist precisely to quantify that. If the float16 consistency check in §3 fails, or if any frozen
product had disagreed with the manifest, the round would have stopped here.

**Status: DS AUDIT 0 PASSED (with the 30 stale `states/` files recorded as a disk-level artefact that never
entered an analysis). Proceeding to Step 1 — geometry of D → U.**

---

## DS AUDIT 1 — geometry of D → U (2026-09-20)

**The single hypothesis under test.** Q1 only: is the strong D→U relation (A) a largely geometric
consequence, or (B) a model phenomenon meaning "larger predicted change is genuinely harder to predict"?
Nothing about selection or causal effect is claimed here.

**What actually ran.** Read-only analysis of the 90 frozen holdout-C states (23,040 tokens): the
decomposition `U² = D² + G² − 2DG·cos θ`, six per-state correlations, three counterfactuals, a per-state
residual fit (isotonic + log-log linear), and a conditional breakdown by how much each token actually
moved. No model, no simulator, no S recomputation. Artifacts: `GEOMETRY_ANALYSIS.csv`,
`GEOMETRY_ADDENDUM.csv`, `summary_geometry.json`. Excluded data: none.

**Protocol amendment DS-A1 (logged before the conclusion was drawn).** `ds_geometry_addendum.py` was added
to resolve an apparent contradiction between this round's median statistics and round 1's reported
"LaWM beats the no-change baseline in 83 % of states". The amendment adds only a conditional re-analysis of
existing arrays; it changes no definition, no split and no threshold.

### 1. Did the data change? Was any oracle used, and for what?
No data changed; everything is read from the frozen run. `H_real`, `G`, `cos θ` and `U` are oracle and are
used **only** to explain the D→U relation. `D` itself uses no future information. The float16 consistency
check registered in `PROTOCOL_DS.yaml` passes: U recomputed from the stored float16 tensors matches the
authoritative float64-derived U to a maximum relative error of 6.5e-4.

### 2. Magnitudes (median over 90 states)

| quantity | value |
|---|---|
| D — predicted change | **7.19** |
| G — real change | **2.82** |
| U — prediction error | **7.09** |
| D/G | **2.00** |
| cos θ | **0.28** |
| G / ‖h_t‖ | 0.10 |
| fraction of tokens with G < D | **0.80** |

**U ≈ D.** The error is almost exactly the size of the entire predicted change, and LaWM predicts about
twice as much change as actually occurs, in a direction only weakly aligned with the real one.

### 3. Is the D→U correlation geometric or a model phenomenon?
**Geometric.** The counterfactuals keep both magnitude distributions and destroy only the pairing between
predicted and real directions:

| | Spearman(D, U) |
|---|---|
| real data | **+0.923** [+0.910, +0.928] |
| B4a: cos θ permuted across tokens | **+0.907** [+0.891, +0.909] |
| B4b: cos θ = 0 (orthogonal directions) | **+0.973** [+0.969, +0.974] |

Destroying every directional relationship between prediction and reality leaves the correlation **equal or
higher**. Whatever LaWAM-specific structure exists in the direction of the predicted change contributes
nothing to — in fact slightly detracts from — the ability of D to predict U.

### 4. Why, mechanistically? (the conditional analysis)
Conditioned on how much each token actually moved (quartiles of G, medians over states):

| quartile | G | D | U | fraction with U < G | S |
|---|---|---|---|---|---|
| q0 (most static) | 1.64 | 6.68 | **6.71** | **0.000** | 2.4e-6 |
| q1 | 2.27 | 6.17 | 6.13 | 0.000 | 6.2e-6 |
| q2 | 3.58 | 5.94 | 6.08 | 0.031 | 1.1e-5 |
| q3 (most moving) | 18.42 | 17.40 | **9.64** | **0.883** | 1.9e-5 |

For three quarters of the tokens the world barely changes (G ≈ 1.6–3.6) while LaWM predicts a change of
≈6–6.7, so **the error simply *is* the predicted change** (q0: U 6.71 vs D 6.68). That identity, holding
over most of the token population, is what produces ρ ≈ 0.92. Only on the moving quartile does the model
genuinely help, cutting the error from 18.4 to 9.6.

### 5. Does this contradict round 1's "LaWM beats no-change in 83 % of states"?
No — both are correct and they measure different things, which the addendum makes explicit. Round 1's
statistic (global per-element MSE) reproduces **exactly: 83.3 %**, and total squared error is roughly halved
(ratio 0.556). But only **23.2 %** [22.0, 25.8] of individual *tokens* have U < G. The aggregate win is
earned entirely on the moving quartile, whose squared errors dominate the sum (G² ≈ 339 vs ≈ 2.7 for the
static quartile). Round 1's statement stands; it just must not be read as "LaWM predicts most tokens better
than doing nothing".

### 6. Is there evidence for "larger change is harder to predict"?
**No — the evidence points the other way.** Spearman(D, U/G), the relative error, is **−0.277**, and
Spearman(D, 1 − cos θ) is **−0.510**: where LaWM predicts a large change, its relative error is *lower* and
its direction agreement is *better*. The absolute error grows with D only because the predicted change
itself is what is mostly wrong on the static majority.

### 7. Should D still be called "predicted change magnitude" rather than uncertainty?
**Yes, emphatically, and the distinction now has content.** D is "how much the model claims this token will
change", and on ~75 % of tokens that claim is largely spurious. Calling it uncertainty, confidence or
reliability would assert a property this analysis specifically fails to find. The naming rule in
`PROTOCOL_DS.yaml:symbols.D` stays in force.

### 8. Does S explain what D cannot?
**No.** After removing what D explains (per-state isotonic and log-log linear fits), the correlation between
S and the residual is **−0.076** [−0.113, −0.021] (isotonic) and −0.167 (linear), and the residual's
correlation with the realised per-token action deviation is **−0.033** [−0.049, +0.038]. S carries no
information about the part of the prediction error that D misses. (S is, separately, correlated with G: it
is ~8× larger on moving tokens than on static ones.)

### 9. Phase / suite dependence
Stable. Spearman(D,U) is 0.905 / 0.924 / 0.924 for approach / manipulation / pre-grasp, with the
counterfactual B4a at 0.876 / 0.909 / 0.915 — the geometric account holds in every phase. D/G rises from
1.59 (approach) to 2.21 (pre-grasp), i.e. the over-prediction of change is worst where the scene is most
static.

### 10. Verdict on Q1
**Hypothesis A: SUPPORTED. Hypothesis B: NOT SUPPORTED.**
The D→U relation is primarily a length-geometry consequence of LaWAM over-predicting change on a
predominantly static token population, not evidence that larger predicted change is harder to predict.

### 11. What would overturn this, and is Step 2 still warranted?
It would be overturned by a counterfactual that *reduces* the correlation when direction information is
destroyed (the opposite of what was measured), or by a token population in which G is comparable to D
(e.g. OOD or high-motion conditions — untested, and the subject of the later G3 gate).

**Step 2 remains warranted and its meaning is now sharper.** Q1's answer does not by itself decide the
project: the causal question is whether ranking tokens by `D × S` repairs the executed trajectory as well as
`U × S` does, and `U ≈ D` on most tokens actually predicts that it might. Case DS-A requires *both* the
geometric finding *and* a failure of `D × S` to beat `D`; only the first half is established. Step 2 is run
exactly as pre-registered, with the 64/256 budget, the frozen S and the frozen noise seeds.

**Status: DS AUDIT 1 complete — Q1 answered (geometry). Proceeding to Step 2.**

---

## DS AUDIT 2 — equal-budget causal selection (2026-09-20)

**The single hypothesis under test.** Q2 only: at an identical 64/256 token budget, does `D × S` repair the
executed trajectory better than `D` alone, than `S`, than attention and than `D × attention` — and how much
of the oracle `U × S` advantage does it retain? No claim about deployment or robustness is made.

**What actually ran.** 90 frozen holdout-C states × 10 chunk variants × 3 flow-noise seeds = **2,700
executed chunks** (21,600 control steps) from the exact snapshots, plus the chunk generation (90 × 10 × 3
flow samplings). S, attention, snapshots, seeds and the 64-token budget are all the frozen round-1 objects;
none was recomputed or re-tuned. Excluded data: none. Infrastructure failures: none. Artifacts:
`ds_chunks/`, `ds_execution_records.jsonl`, `SELECTION_METRICS.csv`, `EXECUTION_METRICS.csv`,
`summary_ds.json`, `figures/ds_selectors.png`.

### 1. Did the data change? Any oracle at selection time?
No. `D` is computed from `H_pred` and `h_t` only. Oracle information enters in exactly two declared places:
the reference selectors E (`top-U`) and F (`top-U×S`), and the *repair* itself (selected tokens get their
true `H_real_j`), which is an offline evaluation instrument for ranking quality — not a deployment step.
A regression guard in `ds_chunks.py` asserts that the freshly computed `H_pred` equals the stored one
bit-for-bit before any selector is built.

### 2. Cross-round reproduction
**Exact.** Recomputed on the same states with an independent script, Recovery(U) = **0.5443** and
Recovery(U×S) = **0.7420**, against round 1's reported 0.545 and 0.742; per-state spot checks matched to
three decimals on treatment (mm) and recovery. Round 1's causal pipeline is reproduced, not merely re-used.

### 3. Results (90 states, budget 64/256, median recovery; CI = episode bootstrap of the mean)

| selector | recovery | residual to full oracle | deployable? |
|---|---|---|---|
| random | 0.156 | 0.732 mm | — |
| attention | 0.362 | 0.516 mm | yes |
| D × attention | 0.578 | 0.383 mm | yes |
| **U (oracle)** | 0.544 | 0.395 mm | no |
| **D** | **0.580** | 0.346 mm | **yes** |
| S | 0.660 | 0.278 mm | offline only |
| **U × S (oracle)** | 0.742 | 0.213 mm | no |
| **D × S** | **0.771** | **0.184 mm** | **yes (S offline)** |

### 4. The pre-registered primary comparisons

| quantity | mean | 95 % CI | states better | in mm |
|---|---|---|---|---|
| **DS_margin = R(D×S) − R(D)** | **+0.1480** | **[+0.0939, +0.1982]** | 81 % | +0.131 mm |
| Oracle_margin = R(U×S) − R(U) | +0.1737 | [+0.1146, +0.2297] | 73 % | +0.130 mm |
| **Margin_retention** | **0.852** | — | — | — |
| **attention_check = R(D×S) − R(D×attention)** | **+0.4193** | **[+0.1520, +0.8823]** | 82 % | +0.234 mm |
| R(D×S) − R(S) | +0.1218 | [+0.0594, +0.2028] | 66 % | +0.115 mm |
| R(D×S) − R(U×S) (oracle) | +0.0212 | [−0.0106, +0.0536] | 58 % | +0.041 mm |

**The pre-registered requirement is met: the 95 % episode-bootstrap CI of DS_margin excludes 0.** The
deployable `D × S` is statistically indistinguishable from the oracle `U × S` (CI of the difference contains
0) and retains 85 % of the oracle margin.

### 5. Is attention enough?
No. `D × attention` recovers 0.578 with a very wide CI ([−0.200, +0.552] on the mean), and `D × S` beats it
by +0.419 [+0.152, +0.882] in 82 % of states. Attention alone (0.362) is far behind `D` alone (0.580). The
mandatory "why not just attention" baseline is answered: attention mass does not substitute for measured
downstream sensitivity.

### 6. Does it hold across phases and suites?
Yes, everywhere. DS_margin by phase: approach +0.144, manipulation +0.113, pre-grasp +0.188 (oracle margins
+0.176 / +0.147 / +0.199). By suite, `D × S` ≥ `U × S`: goal 0.790 vs 0.721, object 0.794 vs 0.792,
spatial 0.708 vs 0.685. No phase or suite carries the result alone.

### 7. Is there a simpler explanation? An unexpected observation, reported as measured
`D` alone (0.580) slightly **beats the oracle `U`** (0.544) at choosing which tokens to repair
(+0.047 [+0.005, +0.094]). The likely reason follows from DS AUDIT 1: because LaWAM over-predicts change on
static tokens, `U ≈ D` there, while on the moving quartile `U` (9.6) is much smaller than `D` (17.4).
Ranking by `D` therefore tilts toward moving tokens — which carry ~8× higher `S` — so `D` is already a
slightly better *action-relevance* proxy than the true prediction error is. This is consistent with the
token-overlap counts (top-D shares 25–34 of 64 tokens with top-U×S, top-U only 24–27), but the round did
not run a dedicated test of this explanation, and it is offered as a hypothesis, not a finding.

### 8. Effect size against the sampler nuisance — the decisive caveat
Measured on the same states: the whole oracle correction is **0.90 mm** of executed end-effector
displacement, the flow sampler's own seed-to-seed variation is **0.47 mm**, and the chunk itself moves
**85.0 mm**. The advantage of `D × S` over `D` is **+0.131 mm** — about **28 % of the sampler nuisance** and
**0.15 % of the chunk motion**. Every fraction above must be read with this: the ranking question is settled
*inside a budget of influence that is smaller than the policy's own sampling noise*.

### 9. Verdict against the pre-registered decision tree
**Case DS-C**, on all four required clauses: `D` approximates `U` well; `D × S` stably beats `D`
(CI excludes 0), `S`, `attention` and `D × attention`; and it retains 85 % of the oracle margin while being
statistically indistinguishable from `U × S`. Case DS-A is **not** triggered: its first clause (the relation
is geometric) holds, but its second clause (`D × S` fails to beat `D`) is refuted. Case DS-B is not
triggered (attention is clearly insufficient). Case DS-D is not triggered (no learned reliability estimator
is needed; the deployable proxy matches the oracle).

Per `PROTOCOL_DS.yaml:decision.DS-C`, **no formal name is coined for the mechanism this round.**

### 10. What would overturn this, and what stops the project?
- The effect size: if a later round shows the whole future channel still governs ~1 mm of motion under
  harder conditions, the ranking advantage remains practically irrelevant regardless of its statistics.
- The repair instrument: recovery is measured by replacing tokens with their *true* values. Nothing here
  shows that anything useful can be done with a token flagged as high-`D × S` at deployment, where no true
  value exists.
- `S` is still offline: it costs ~2,300 forward passes per state, so `D × S` is not yet an online quantity.
- Step 1's geometry: `D` is a predicted-change magnitude, not a reliability measure, so the mechanism's name
  and interpretation must not drift toward "uncertainty".

**Status: DS AUDIT 2 complete — Case DS-C on the causal criterion, with the effect-size caveat in §8.**

---

## FINAL DS AUDIT (2026-09-20)

**Total executed work.** Read-only re-analysis of the 90 frozen holdout-C states (23,040 tokens) for the
geometry of D→U, including three counterfactuals and a conditional breakdown; chunk generation for 8
selectors + 2 references × 3 noise seeds (2,700 flow samplings); **2,700 executed chunks** (21,600 control
steps) from exact snapshots. Excluded data: **none**. Infrastructure failures: none. Nothing was trained,
no rollout was collected, S / attention / snapshots / seeds / budget are the frozen round-1 objects, and
the LaWAM `state_dict` sha256 `37a53b8c…b523` was verified before the round.

**Did the question change?** No. `PROTOCOL_DS.yaml:question` was frozen before any statistic. One
append-only amendment was logged (DS-A1: a conditional re-analysis added to resolve an apparent
contradiction with round 1's aggregate statistic); it changed no definition, split, threshold or budget.

**Two findings, and they must be read together.**

1. **Q1 — the D→U relation is geometry, not reliability.** Destroying every directional relationship
   between predicted and real change leaves Spearman(D, U) unchanged or higher (0.923 → 0.907 with cos θ
   permuted, → 0.973 with cos θ = 0). The mechanism is that LaWAM predicts ≈2× more change than occurs on a
   predominantly static token population, so for ~75 % of tokens the error simply *is* the predicted change
   (static quartile: U 6.71 vs D 6.68). Relative error and direction agreement both *improve* with D. There
   is **no evidence** that "larger predicted change is harder to predict". `D` therefore stays named
   *predicted change magnitude*; calling it uncertainty, confidence or reliability is not supported.

2. **Q2 — yet the deployable combination works, and matches the oracle.** At an identical 64/256 budget,
   `D × S` recovers **0.771** of the oracle correction versus 0.580 for `D`, 0.660 for `S`, 0.578 for
   `D × attention`, 0.362 for attention and 0.156 for random — and versus **0.742 for the oracle `U × S`**.
   DS_margin = **+0.148 [+0.0939, +0.1982]** (81 % of states), margin retention **0.852**, and
   `D × S` − `U × S` = +0.021 [−0.011, +0.054], i.e. statistically indistinguishable from the oracle.
   The round-1 numbers were reproduced exactly (Recovery(U) 0.5443, Recovery(U×S) 0.7420).

**Case DS-C.** A learned U is not needed and an oracle U is not needed either: the free predicted-change
magnitude, multiplied by the measured downstream sensitivity, selects action-critical future tokens as well
as the oracle does. Per the protocol, **no formal name is coined**.

**The caveat that governs everything above.** The entire quantity being apportioned is **0.90 mm** of
executed end-effector displacement, against a flow-sampler nuisance of **0.47 mm** and a chunk motion of
**85.0 mm**. `D × S` improves on `D` by **+0.131 mm** — roughly a quarter of the policy's own sampling
noise. The selection question is settled; the practical significance is not, and this round does not
establish any.

**What is still not demonstrated** (carried into `REPORT_DS.md` §14): that flagging a token at deployment
helps when no true value exists to repair it with; that `S` can be obtained online at all; that anything
here improves success, robustness or safety; that the result survives outside successful, in-distribution
trajectories of three tasks on one checkpoint.

**Next gate, not entered this round:** G3 — failure / OOD enrichment (occlusion, distractors, camera
perturbation, unseen layout), where the future channel may carry more than 1 mm and where `D × S` can be
tested against real action error and task failure rather than against an oracle repair.

**Status: DS round complete. Case DS-C with the effect-size caveat. No further experiment started.**
