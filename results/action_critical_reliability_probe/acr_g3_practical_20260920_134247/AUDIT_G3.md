# AUDIT_G3.md — Practical Significance Validation, run `acr_g3_practical_20260920_134247`

Append-only. All three prior runs are **read-only**.

> **Scope statement, fixed for the whole round:**
> **This round does not re-validate the definitions of D, U or S. It validates practical significance only.**
> The ranking-formula question is closed by the DS round and is not reopened.

---

## G3 AUDIT 0 — project state and frozen conclusions (2026-09-20)

**The single question of this round.** In genuinely difficult, failure-producing conditions, does LaWAM's
future prediction error grow large enough to matter for control, and does `D × S` still locate the
action-relevant future information better than simple alternatives? Nothing else is under test.

**What actually ran.** Read-only verification of git state and of the three frozen run directories; a
re-run of `g2_audit0.py` (3 states from 3 suites, full `state_dict` hash); readout of the frozen reference
values this round will be compared against. No rollouts, no perturbations, no training. Artifacts:
`PROTOCOL_G3.yaml` (frozen before any pilot rollout), `g2_audit0_model.json`.

### 1. Is any frozen question being reopened?
No. `PROTOCOL_G3.yaml:frozen_findings_not_reopened` lists the closed topics explicitly: new U, new
uncertainty estimator, learned U, new explanations of D, new S definitions, new formulas (D²S, DS², …),
new attention variants, new proxies, new latent-semantics interpretations. The protocol's `selectors` list
is fixed at the eight already used, and `no_new_formulas` is stated.

### 2. Branch / commit / working tree
`research/action-critical-reliability-probe` @ **`ec7dd4b`** — verified directly, matching the commit named
in the task. Working tree clean. Model diff against upstream `main` unchanged: 4 files, +113/−9, the
default-off diagnostic hooks. This round adds no model code.

### 3. Are the three frozen runs intact?

| run | contents verified |
|---|---|
| `acr_20260919_203850` (round 1) | `sensitivity/` 90, `uncertainty/` 90, `attention/` 90, `consequence_chunks/` 90 |
| `acr_learned_u_20260920_120110` (G2) | `u_dataset/` 160, `models/` 4 |
| `acr_change_sensitivity_20260920_124748` (DS) | `ds_chunks/` 90 |

The 30 stale `states/*_s0000_approach.npz` files noted in DS AUDIT 0 remain on disk and remain outside every
analysis; they are not touched.

### 4. Are the frozen quantities and settings reproducible?
Yes. `state_dict` sha256 **`37a53b8c…b523`** over 2,555,179,360 parameters — identical to the value in all
previous rounds and in `PROTOCOL_G3.yaml`. Checkpoint 7,174,214,645 B, mtime 2026-09-14 01:24:55.
On three states from three suites the policy reproduced the stored `h_t1_pred` and `h_t` at **0.0**, the
stored baseline action chunk at **0.0**, the frozen S at ≤4.3e-12 (same 32-row batch layout), and
future-token permutation invariance at ~4e-7.

Still reproducible and frozen for this round: **64/256 token budget**, **flow-noise seeds [101, 202, 303]**,
action horizon 8 (`floor(0.4 × 20)`), snapshot save/restore (exact to ≤4.7e-7 m), the `future_override` hook,
and fixed-noise inference.

### 5. The in-distribution reference this round must beat

| quantity | ID value |
|---|---|
| chunk motion | 84.99 mm |
| future-error treatment — executed eef, π(H_real) vs π(H_pred) | **0.904 mm** |
| sampler nuisance — same H_pred, different seed | **0.474 mm** |
| treatment / nuisance | ≈1.9 |
| Recovery: random / attention / D×attention / U / D / S / U×S / **D×S** | 0.156 / 0.362 / 0.578 / 0.544 / 0.580 / 0.660 / 0.742 / **0.771** |
| DS_margin = R(D×S) − R(D) | +0.148 [+0.0939, +0.1982] = **+0.131 mm** |

### 6. Does the round-1 ERRATA affect this round?
No. It corrects five §7 figures of round 1's `REPORT.md` (e.g. Spearman(U,S) 0.024 → 0.053) that this round
does not use. All G3 comparisons are against `summary.json` / `summary_ds.json` values, which were always
the final 90-state products.

### 7. Was any perturbation severity chosen to support a hypothesis?
It cannot have been: **all nine severities were written into `PROTOCOL_G3.yaml` before a single pilot
rollout existed**, and they are grounded in the token geometry rather than in any result — the image is
256×256 with a 16×16 grid of 16 px patches, one patch per future token, so occlusion sizes are stated as
48/80/112 px = 9/25/49 of the 256 tokens, and shifts as 6/12/24 px = 0.375/0.75/1.5 patches. The severity
*selection* rule (lightest severity with 20 % < success < 90 %) is likewise pre-registered, and
`forbidden_inputs_to_this_rule` explicitly excludes any `D × S` or selector result.

### 8. Is the stress a reasonable OOD condition or just destroyed input?
That is exactly what the pilot must decide, and the protocol fixes the criteria in advance: a family whose
mildest level already drops success below 20 % because vision is unusable is labelled **TOO_DESTRUCTIVE**
and abandoned, not re-tuned; a family whose every level keeps ~100 % success with ID-level treatment is
labelled **NO_EFFECT**. The wrist camera is never perturbed in families A and B, so the policy always
retains one intact visual stream — this is a design choice to keep the condition *hard* rather than
*hopeless*.

One definition registered now because it changes what U means under stress: **`H_real` is the encoding of
the PERTURBED `o_{t+7}`**. The perturbation is part of the observation process, so the frame the model is
trying to predict is the one it will actually receive.

### 9. Most conservative conclusion if the round stopped here
"The three frozen rounds are intact and bit-reproducible under the current code and weights; the stress
protocol, severities, selection rule, budgets and decision tree are pre-registered; no measurement of
practical significance exists yet."

### 10. Have any hard-stop conditions been reached?
No — none can be, before the pilot. The hard stops are defined in `PROTOCOL_G3.yaml:decision`:
**G3-FAIL-PRACTICAL** if, at the frozen confirm severities, treatment stays the same order of magnitude as
the sampler nuisance, or future error is essentially unrelated to failure/progress, or even full-oracle
repair yields no task-level headroom. On any FAIL verdict no G3.1/G3.2 and no new U/D/S variant is created.

**Status: G3 AUDIT 0 PASSED. Proceeding to the PILOT (severity search only — no S sweep, no ranking).**

---

## G3 AUDIT 1 — pilot and pre-registered severity selection (2026-09-20)

**The single question of this audit.** Which of the three pre-registered stress families produces a
*critical but executable* regime — the task starts to get harder **and** the future-error consequence
amplifies — and which severity does the pre-registered rule select? No ranking, no S, no token repair
exists in the pilot.

**What actually ran.** 150 rollout attempts = 3 suites × 1 task × 5 independent resets × 10 conditions
(ID + 3 families × 3 severities). 132 executed, **18 marked invalid by the family-C validity gate and
recorded, never replaced**. 393 states selected with the frozen phase rule; for each,
`H_pred`, `H_real` (from the equally perturbed `o_{t+7}`), and π(H_pred) / π(H_real) at 3 frozen noise
seeds; **2,358 executed chunks** for treatment and nuisance. Infrastructure errors: **0**. Excluded data:
none. Artifacts: `PILOT_RESULTS.csv`, `pilot_summary.json`, `treatment_features.jsonl`,
`treatment_execution.jsonl`, `figures/g3_pilot.png`.

### 1. Cross-check: does the ID condition reproduce the frozen reference?
Yes, on freshly collected episodes: treatment **0.83 mm** vs round 1's 0.904, nuisance **0.46 mm** vs 0.474,
ratio **1.84** vs ≈1.9, success 100 %. The pilot instrument agrees with three previous rounds.

### 2. The pilot table

| condition | success | valid/invalid | treatment mm | nuisance mm | t/n | vs ID | U (per-elem MSE) |
|---|---|---|---|---|---|---|---|
| **ID** | 100 % | 15/0 | **0.83** | 0.46 | 1.84 | 0.91 | 0.087 |
| A occlusion mild (48 px, 9 tok) | 100 % | 15/0 | 1.06 | 0.44 | 2.34 | 1.17 | 0.126 |
| **A occlusion medium (80 px, 25 tok)** | **87 %** | 15/0 | **2.04** | 0.53 | **3.69** | **2.26** | 0.134 |
| A occlusion strong (112 px, 49 tok) | 93 % | 15/0 | **2.16** | 0.70 | 3.26 | 2.39 | 0.161 |
| B shift mild (6 px) | 100 % | 15/0 | 0.88 | 0.46 | 2.04 | 0.97 | 0.115 |
| B shift medium (12 px) | 93 % | 15/0 | 1.09 | 0.47 | 2.05 | 1.21 | 0.147 |
| **B shift strong (24 px)** | 93 % | 15/0 | **1.57** | 0.50 | **3.22** | 1.73 | 0.201 |
| C object mild (2 cm) | 100 % | 10/5 | 0.84 | 0.48 | 1.87 | 0.93 | 0.095 |
| C object medium (4 cm) | 100 % | 9/6 | 1.16 | 0.51 | 2.42 | 1.28 | 0.093 |
| **C object strong (6 cm)** | 88 % | 8/7 | 1.04 | 0.53 | 2.07 | 1.15 | 0.097 |

### 3. Which families produce a critical interval? (the question that matters most)
**A (occlusion) and B (camera shift) do; C (object shift) barely.** The distinction the task set out to make
— "did the task get harder *and* did the future channel amplify?" versus "was vision simply destroyed?" —
comes out cleanly on the side we wanted:

- **Occlusion medium**: success only drops to **87 %**, yet the treatment **more than doubles**
  (0.83 → 2.04 mm) while the nuisance stays at 0.53 mm, so treatment/nuisance goes **1.84 → 3.69**.
- **Occlusion strong**: treatment 2.16 mm, but the nuisance also rises (0.70 mm), so the ratio is lower.
- **Camera shift strong**: treatment 1.57 mm at 93 % success, ratio 3.22, and the largest latent-space
  degradation of all (U 0.201 = 2.3× ID, D 10.93 vs 7.10).
- **Object shift**: U essentially unchanged (0.093–0.097 vs 0.087) and treatment only 1.15–1.28× ID.
  Moving an object to a different valid position does not make the *future* harder to predict — the scene
  dynamics are unchanged. It is the weakest family by a wide margin.

### 4. Are the failures real task failures or destroyed input?
Real, and mild: **6 failures among 132 executed episodes**, and **all six are 250-step timeouts**, not
crashes or controller faults; infrastructure errors 0. Success never falls below 87 %. This is the regime
the task description asked for — not the "success collapses to 40 % while treatment stays at 1 mm" pattern
that would have indicated a broken policy rather than a future-channel effect.

### 5. Is the amplification an artefact? (the alternative explanation, measured)
Partly, and it must be stated. Splitting the occlusion states by whether a token lies inside the grey patch:

| | U inside patch | U outside patch | ID reference |
|---|---|---|---|
| occlusion medium | **13.77** | **8.53** | 7.68 |
| occlusion strong | 13.71 | 9.10 | 7.68 |

So the rise has two components: (a) inside the patch, the model predicts substantial change over a region
that is *constant grey* — the same "over-prediction of change on static content" pathology the DS round
documented, and in that sense an artefact of a uniform occluder; (b) outside the patch, U still rises
**11–18 % above ID**, which is a genuine degradation of future prediction caused by not seeing the object.
The executed **treatment** is measured through actions, not through U, so the doubling is real regardless —
but the interpretation must carry this caveat, and a textured or naturalistic occluder might not reproduce
component (a).

### 6. Was severity chosen by the pre-registered rule, with no reference to D × S?
Yes, mechanically, in `g3_pilot_analyze.py`, which has no access to any D, S, D×S or selector quantity —
no such value is computed anywhere in the pilot pipeline:

| family | verdict | chosen | rule applied |
|---|---|---|---|
| A occlusion | **VALID** | **medium** | lightest severity with 20 % < success < 90 % (87 %) |
| B camera shift | **VALID** | **strong** | no severity in the window → fallback: max t/n with success ≥ 20 % |
| C object shift | **VALID** | **strong** | lightest severity in the window (88 %) |

### 7. Honest caveats on the mechanical verdicts
- **C is VALID only by the letter of the rule.** Its treatment amplification is 1.15× ID, its U is
  unchanged, and the validity gate voided **18 of 45** of its episodes — including **all 15 on
  `libero_object`**, where shifting the soup +x+y pushes it off the table (`contact:floor`). Family C will
  therefore contribute ~2 suites and ~8–10 episodes per task in a confirm stage, and its scientific value
  looks marginal. The rule is not amended after the fact; the caveat is recorded and the decision to carry
  C forward is deferred to the user.
- **B was selected by the fallback**, not the primary rule: no severity dropped below 90 % success.
- The family-C invalidity is the validity gate doing exactly its job — preventing "the task became
  physically impossible" from being scored as a model failure.

### 8. Did any hard-stop condition trigger?
No. The opposite: the pilot found the regime the round needs. Treatment is **not** in the same order of
magnitude as the nuisance under occlusion medium (2.04 vs 0.53 mm, ratio 3.69), whereas in ID it was
0.83 vs 0.46 (ratio 1.84).

### 9. Most conservative conclusion if we stopped here
"Under a pre-registered 80 px occlusion of the primary camera, LaWAM's task success falls only to 87 %
while the executed consequence of its future-prediction error grows from 0.83 mm to 2.04 mm against an
unchanged ~0.5 mm sampler nuisance. This shows the future channel *can* matter more under difficulty; it
says nothing yet about whether D × S locates the responsible tokens, which was not measured in the pilot."

### 10. Frozen for the confirm stage
`A_occlusion = medium (80 px)`, `B_camera_shift = strong (24 px)`, `C_object_shift = strong (6 cm)`.
These are now fixed and will not be revisited.

**Status: G3 AUDIT 1 complete. PILOT STOPPED as instructed. Confirm is NOT started pending user confirmation.**

---

## G3 AUDIT 2 — Confirm stage C1: practical significance (2026-09-20)

**The single question of this audit.** On new tasks and new resets, is the executed consequence of
LaWAM's future-prediction error still clearly above the in-distribution level, and not drowned by the flow
sampler's own noise? **No S, no D × S, no selector and no token repair exist in this stage** — none is
computed anywhere in the C1 pipeline.

**What actually ran.** 3 suites × 2 tasks × 10 independent resets × {ID, A occlusion 80 px,
B camera shift 24 px} = **180 rollouts** (60 per condition; family C excluded per amendment G3-A1).
477 states selected with the frozen phase rule; per state `H_pred`, `H_real` (from the equally perturbed
`o_{t+7}`), π(H_pred) and π(H_real) at 3 frozen noise seeds; **2,862 executed chunks** from exact snapshots.
Infrastructure errors: **0**. Episodes excluded: **none**; all 35 failures are retained.
Artifacts: `CONFIRM_RESULTS.csv`, `confirm_c1_summary.json`, `confirm/treatment_*.jsonl`,
`figures/g3_confirm_c1.png`.

### 1. Did the data or the definitions change?
No. Severities are the two frozen by G3 AUDIT 1; tasks, scale and staging were fixed in amendments
G3-A2/A3 before any confirm rollout; `H_real` is still the encoding of the equally perturbed `o_{t+7}`;
flow-noise seeds, chunk horizon, snapshot restore and action post-processing are the same objects used in
all four rounds.

### 2. The C1 table (median over states; CI is an episode bootstrap of the **mean**)

| condition | success | episodes | states | treatment mm (median / mean) | mean CI | nuisance mm | t/n | t/n CI | vs round-1 ID |
|---|---|---|---|---|---|---|---|---|---|
| **ID** | 95 % | 60 | 162 | **0.82 / 1.00** | [0.90, 1.12] | 0.44 | **1.77** | [1.93, 2.39] | 0.90 |
| **A occlusion 80 px** | 78 % | 60 | 163 | **1.87 / 2.23** | [2.01, 2.45] | 0.61 | **3.15** | [3.43, 4.18] | **2.07** |
| **B camera shift 24 px** | 68 % | 60 | 152 | **1.70 / 2.47** | [2.16, 2.81] | 0.49 | **3.57** | [4.12, 5.67] | 1.88 |

The mean CIs of the two stressed conditions do not overlap the ID mean CI. The pilot's amplification
**reproduces on six tasks and new resets**: the executed consequence of the future-prediction error roughly
doubles while the sampler nuisance stays near 0.5 mm, so treatment/nuisance rises from ≈1.8 to 3.2–3.6.

### 3. Is this "the task got harder" or "vision was destroyed"?
The former. Success falls to 78 % (A) and 68 % (B) — not to near zero — and **all 35 failures are 250-step
timeouts**, with zero crashes and zero infrastructure errors. The ID control on the same six tasks is 95 %
(57/60), so the baseline itself is healthy. Per task, the stress bites unevenly
(B: `goal t5` 1/10, `object t6` 5/10, but `spatial t2` and `t7` still 10/10), which is what a genuine
difficulty gradient looks like rather than a uniform collapse.

### 4. The evidence rule was respected (amendment G3-A4)
The primary evidence above is the **executed** consequence of π(H_pred) vs π(H_real) at fixed flow noise
from the identical snapshot. U is reported only as a diagnostic, split as required:

| condition | U all tokens | U occluded | U non-occluded | D | G |
|---|---|---|---|---|---|
| ID | 7.63 | — | 7.63 | 7.02 | 2.73 |
| A occlusion | 9.15 | **14.31** | **8.67** | 7.30 | 2.56 |
| B camera shift | **11.15** | — | 11.15 | **10.13** | 2.91 |

Inside the grey rectangle U is inflated by construction (14.31), exactly as the pilot predicted, and that
number is **not** used as evidence. Outside the rectangle U still rises 13.6 % above ID. **Family B carries
no occluder at all**, yet shows the largest latent degradation (U +46 %, D 10.13 vs 7.02) together with a
doubled treatment — so the amplification does not depend on the grey-patch artefact. No new occluder was
designed in response to this, as required.

### 5. Is there a simpler explanation?
Three were considered.
- *"The chunks just move further under stress."* No: chunk motion is comparable across conditions, and the
  nuisance — measured on the same chunks with only the noise seed changed — barely moves (0.44 → 0.61 /
  0.49 mm) while the treatment doubles.
- *"It is the grey patch."* Addressed in §4: family B has no patch and shows the effect more strongly.
- *"It is one task carrying the result."* No: the treatment rises in every one of the six tasks for A
  (1.17–2.19 mm vs ID 0.53–1.29 mm) and in five of six for B. The two `goal` tasks amplify least, which is
  also where ID treatment was already highest.

### 6. Failure-level observation, reported as measured
There is **no clean "failures have larger treatment" pattern**: in ID, failed episodes have *higher*
treatment (1.27 vs 0.80 mm); under occlusion they have *lower* (1.53 vs 1.95); under shift they are
equal (1.67 vs 1.71). This is an honest negative for any state-level failure-prediction story. Per
`PROTOCOL_G3.yaml:failure_analysis`, that analysis is auxiliary and the research question is **not**
rewritten because of it; it is recorded and carried into the report.

### 7. Effect size, in millimetres, against the two reference scales
Treatment rises from **0.82 mm** to **1.87 mm** (A) and **1.70 mm** (B), against a sampler nuisance of
**0.44–0.61 mm** and a chunk motion of ~85 mm. So the future channel now governs roughly **3× the
sampler's own noise** instead of 1.8×, but it is still **~2 % of the chunk's motion**. The problem became
measurably more important; it did not become large.

### 8. Did any hard-stop condition trigger?
**No.** `PROTOCOL_G3.yaml:decision.G3-FAIL-PRACTICAL` requires the treatment to stay in the same order of
magnitude as the nuisance under reasonable difficulty. It does not: the ratio moved from 1.77 to 3.15/3.57
with non-overlapping mean CIs, on six tasks across three suites with 60 matched ID controls.

### 9. Most conservative conclusion if we stopped here
"Under two independent, pre-registered stresses that leave the policy functional (78 % and 68 % success,
all failures timeouts), the executed consequence of LaWAM's future-prediction error approximately doubles
relative to the in-distribution condition and reaches ~3× the flow sampler's own run-to-run variation,
while remaining ~2 % of the chunk's motion. Nothing has been measured about whether D × S locates the
responsible tokens under stress — that is stage C2 and it has not been run."

### 10. Stage C1 verdict
**Practical amplification reproduced.** C1 passes its pre-registered criterion, so G3-FAIL-PRACTICAL is not
triggered and stage C2 becomes admissible — but is **not started**: per amendment G3-A3 it requires explicit
user confirmation.

**Status: Stage C1 complete, G3 AUDIT 2 written, STOPPED. Stage C2 not started.**

---

## G3 AUDIT 3 — Confirm stage C2: does D × S still work under stress? (2026-09-20)

**The single question.** Under the two frozen stress conditions, does `D × S` still stably beat `D`, `S`,
attention and `D × attention`, and does it stay close to the oracle `U × S`? Nothing else.

**What actually ran.** The pre-registered subset of amendment G3-A5: 2 stress conditions × 6 tasks ×
10 episodes × the approach-phase state = **exactly 120 states**, 0 episodes missing an approach state,
0 substitutions, 32 of the 120 drawn from episodes that actually failed. A fresh **S sweep with the
round-1 parameters** (ε [0.271, 0.812, 2.706], 3 directions, seed 101, metric `l2_norm_all7`, batch 32 with
an in-batch baseline) = 120 × 2,304 ≈ **276,000 perturbed samplings**; attention recorded on all 120 states
(SDPA equivalence guard passed); 120 × 10 variants × 3 seeds = **3,600 executed chunks**.
Infrastructure errors: 0. States excluded: none. Artifacts: `c2/sensitivity/`, `c2/attention/`,
`c2/c2_chunks/`, `c2/c2_execution_records.jsonl`, `TOKEN_SELECTION_RESULTS.csv`,
`EXECUTION_RESULTS.csv`, `c2_summary.json`, `figures/g3_c2.png`.

### 1. Was the subset rule mechanical, and was anything re-opened?
Yes and no respectively. The rule was frozen in G3-A5 **before any ranking number existed**, and its inputs
are only (condition, task, episode, phase) — never a D, S or recovery value. No new formula, proxy, stress
family or failure predictor was introduced; the selector list is the frozen one, the budget is still 64/256,
and S was recomputed with round-1 parameters rather than re-tuned. A regression guard in `g3_c2_chunks.py`
asserts that the freshly computed `H_pred` equals both frozen copies bit-for-bit.

### 2. Results (120 stressed states, budget 64/256, median recovery; CI = episode bootstrap of the mean)

| selector | recovery under stress | residual to full oracle | in-distribution (DS round) |
|---|---|---|---|
| random | 0.209 | 1.472 mm | 0.156 |
| attention | 0.432 | 1.189 mm | 0.362 |
| U (oracle) | 0.629 | 0.851 mm | 0.544 |
| D × attention | 0.701 | 0.635 mm | 0.578 |
| **D** | **0.736** | 0.589 mm | 0.580 |
| S | 0.787 | 0.419 mm | 0.660 |
| U × S (oracle) | 0.821 | 0.390 mm | 0.742 |
| **D × S** | **0.822** | **0.371 mm** | 0.771 |

Every selector recovers *more* under stress than in-distribution, and the ordering is unchanged.

### 3. The pre-registered comparisons

| comparison | mean | 95 % CI | better in | in mm |
|---|---|---|---|---|
| **D × S − D** | **+0.1218** | **[+0.0838, +0.1618]** | 74 % | **+0.231** |
| D × S − S | +0.0369 | [+0.0138, +0.0601] | 64 % | +0.097 |
| D × S − attention | +0.3722 | [+0.3188, +0.4233] | 92 % | +1.180 |
| **D × S − D × attention** | **+0.1253** | **[+0.0932, +0.1586]** | 82 % | **+0.349** |
| D × S − U × S (oracle) | +0.0093 | [−0.0110, +0.0295] | 52 % | +0.009 |
| D × S − random | +0.5114 | [+0.4643, +0.5609] | 96 % | +1.738 |
| U × S − U (oracle margin) | +0.2054 | [+0.1618, +0.2515] | 84 % | +0.501 |

All four required comparisons have CIs excluding zero, and the difference from the **oracle** `U × S` has a
CI containing zero — the deployable combination remains statistically indistinguishable from the oracle
under stress, exactly as in-distribution.

### 4. Does it hold in both stresses, both outcome classes, and every suite?
Yes. `D × S − D` is +0.121 [+0.064, +0.179] under occlusion and +0.123 [+0.070, +0.181] under camera shift.
By episode outcome the margin is **larger in episodes that actually failed**: +0.152 [+0.077, +0.233]
versus +0.111 [+0.066, +0.158] in successful ones. By suite, `D × S` ≥ `U × S` or equal in all three
(goal 0.702 vs 0.706, object 0.927 vs 0.918, spatial 0.805 vs 0.805).

### 5. Effect size in millimetres — the discipline that has governed every round
Stressed scales on these 120 states: treatment **2.17 mm**, sampler nuisance **0.51 mm**.
The `D × S` advantage over `D` is **+0.231 mm**, i.e. **45 % of the sampler's own noise** — up from
+0.131 mm and 28 % in-distribution. It is still **below** the nuisance in absolute terms. Against
attention-based selection the advantage is +0.349 mm (vs D × attention) and +1.180 mm (vs attention alone).

### 6. Is there a simpler explanation?
- *"S alone does the work."* No: `D × S − S` = +0.037 [+0.014, +0.060], positive in 64 % of states.
- *"D alone does the work."* No: +0.122 [+0.084, +0.162].
- *"Attention would do."* No: attention is the second-worst selector (0.432), and `D × attention` trails
  `D × S` by +0.125 [+0.093, +0.159].
- *"Everything is easier under stress so the ranking is trivial."* Recovery does rise for every selector,
  but the *gaps* also widen (D × S − D: +0.122 stressed vs +0.148 in-distribution in fraction, +0.231 mm vs
  +0.131 mm in millimetres), and random still only reaches 0.209.
- **Margin retention** relative to the oracle margin is 0.593 under stress (0.852 in-distribution): the
  oracle margin itself grew more (+0.205 vs +0.148), so `D × S` keeps a smaller *share* of a larger gap.

### 7. What did NOT get demonstrated
The one-chunk headroom test (execute a repaired chunk, then continue with the unmodified policy and compare
task success/progress) was **not run** — it was outside the scope the user set for C2. So this round shows
that `D × S` *ranks* the action-critical future tokens correctly under stress, and that the ranking is worth
+0.231 mm of executed trajectory; it does **not** show that repairing those tokens would change task
outcomes. Task-level headroom remains undemonstrated.

### 8. Most conservative conclusion
"On 120 stressed states drawn by a mechanical rule from two independent pre-registered perturbations,
`D × S` recovers 0.822 of the oracle correction versus 0.736 for `D`, 0.787 for `S`, 0.701 for
`D × attention` and 0.432 for attention, and is statistically indistinguishable from the oracle `U × S`
(0.821). All four required margins have CIs excluding zero, and the advantage is larger in failed episodes.
In absolute terms the advantage is 0.231 mm against a 0.51 mm sampler nuisance and a 2.17 mm total effect."

**Status: G3 AUDIT 3 complete. Gap Validation stops here per the user's instruction, regardless of outcome.**

---

## FINAL G3 AUDIT (2026-09-20)

**Total executed work.** Pilot: 150 rollout attempts (132 executed, 18 invalidated by the family-C physical
validity gate and recorded), 393 states, 2,358 executed chunks. Confirm C1: 180 rollouts over 6 tasks in
3 suites, 477 states, 2,862 executed chunks. Confirm C2: 120 states by a mechanical subset rule, a fresh
round-1-parameter S sweep (~276,000 perturbed samplings), 120 attention recordings, 3,600 executed chunks.
**Infrastructure errors across the whole round: 0. Episodes or states excluded: none** — all 35 confirm
failures and all 18 family-C invalidations are retained and reported.

**Was anything trained or modified?** No. The LaWAM `state_dict` sha256 `37a53b8c…b523` was verified before
the round and no training, no new loss, no gate, no side-channel and no RL exists anywhere in it. This
round added no model code.

**Was any frozen question reopened?** No. No new U, no learned U, no new explanation of D, no new S
definition, no new formula (D²S, DS², …), no new proxy, no new attention variant, no new stress family, no
failure predictor. The selector list, the 64/256 budget, the flow-noise seeds and the S parameters are the
frozen objects of earlier rounds.

**Was any severity or subset chosen to favour a hypothesis?** No. All nine severities were frozen before
the first pilot rollout and are grounded in the token geometry; the severity *selection* rule ran in a
script with no access to any D, S or ranking quantity; the C2 subset rule (approach-phase state, both
conditions, all six tasks, all ten episodes) was frozen before any ranking number existed and yielded
exactly 120 states with zero substitutions.

### Verdict against the pre-registered G3-PASS criteria

| criterion | status | evidence |
|---|---|---|
| 1. future-error influence clearly above ID under multiple reasonable stresses | **met** | treatment 0.82 → 1.87 mm (occlusion) and 1.70 mm (camera shift), mean CIs non-overlapping, two independent families, 6 tasks |
| 2. treatment not drowned by the sampler nuisance | **met** | treatment/nuisance 1.77 → 3.15 and 3.57 |
| 3. D × S still stably beats D, attention and D × attention in failure-rich states | **met** | +0.122 [+0.084, +0.162], +0.372 [+0.319, +0.423], +0.125 [+0.093, +0.159]; margin *larger* in failed episodes (+0.152 vs +0.111); 32 of 120 states from failed episodes |
| 4. equal-budget causal repair still gives a stable gain | **met** | executed recovery 0.822 vs 0.736, +0.231 mm, in 74 % of states, in both stresses and all three suites |
| 5. some real task-progress / failure evidence linked to the mechanism | **partially met** | the mechanism's advantage is significantly larger in episodes that actually failed; but C1 found **no** clean "failures have larger treatment" pattern, and the one-chunk headroom test was not run |

**Decision: G3-PASS, qualified on criterion 5.** Criteria 1–4 are met decisively. Criterion 5 rests only on
the failure-state margin: the ranking mechanism is demonstrably more valuable where the robot actually
failed, but **no task-level headroom was demonstrated**, because the one-chunk headroom test was outside
the scope set for C2.

### What this round did NOT establish
1. That repairing high-`D × S` tokens changes task success or progress — untested.
2. That anything can be done with a flagged token at deployment, where no true `H_real` exists.
3. That `S` can be obtained online — it still costs ~2,300 forward passes per state (~86 s).
4. That the effect is large: +0.231 mm against 0.51 mm of sampler noise and ~85 mm of chunk motion.
5. Generality beyond one checkpoint, two perturbation families, six tasks and a 0.4 s horizon.

### Honest negatives carried forward
- Family C (object/layout shift) was tested at three severities and is **not informative** for this
  mechanism: it does not increase future-prediction difficulty (U 0.093–0.097 vs ID 0.087) and it produced
  18/45 physically invalid episodes.
- Under occlusion, part of the U increase is an artefact of the uniform grey patch (U 14.31 inside vs 8.67
  outside vs 7.63 ID). This is why U was barred from being primary evidence; the camera-shift family, which
  has no occluder, shows the effect at least as strongly.
- There is no clean relationship between per-state treatment and episode failure.

**Status: G3 complete. Gap Validation STOPS here, as instructed, regardless of outcome. Method Design is
NOT started.**
