# AUDIT_HEADROOM.md — Method-Design Entry: one-chunk task-level headroom test

run `acr_headroom_20260920_194346`. Append-only. The four earlier runs are **read-only**.

> **Scope, fixed for the whole round:** this round does **not** re-validate that D × S selects the right
> tokens — G3 settled that. It asks only whether correcting that information for a single chunk, then
> handing control back to the unmodified LaWAM policy, changes the **final task outcome**.

---

## H AUDIT 0 — G3 assets, frozen ranking and the checkpoint (2026-09-20)

**The single question of this audit.** None — this certifies that every quantity the headroom test reuses
is the frozen G3 quantity, and that the interface can run the experiment as specified.

**What actually ran.** Read-only inspection of git state and the G3 run; a feasibility probe of the
inference interface (`h_feasibility.py`, ~8 queries); a full-precision dump of `H_pred` for all 120 states
(`h_dump_hpred.py`). No simulator rollouts, no training.

### 1. Branch / commit / working tree
`research/action-critical-reliability-probe` @ **`60a9ae2`** — verified directly, not taken from the
prompt. Working tree clean. This round adds no model code and modifies no repository file.

### 2. Are the 120 C2 states and their frozen products complete?
Yes: `c2/states_manifest.jsonl` 120 lines, `c2/sensitivity/` 120, `c2/attention/` 120, `c2/c2_chunks/` 120.
Every snapshot named in the manifest exists. `token_budget = 64`, and the frozen selector picks
`pick_B_top_D` and `pick_D_top_D_times_S` are 64 tokens each — **S is read, never recomputed**, and the
rankings are reused verbatim rather than re-derived.

### 3. Is the checkpoint unchanged?
`state_dict` sha256 **`37a53b8c…b523`**, identical to every previous round (re-verified by `g2_audit0.py`
at the start of G3 and unchanged since; no process in this round loads a different checkpoint).

### 4. Can the interface do what the protocol requires? (the gating engineering question)
Yes, and it was checked before anything was promised. The protocol demanded that the *entire* continuation
noise sequence be fixable, and that the round stop if it is not. Wrapping the websocket transport
research-side — **no repository file is patched** — lets every policy query carry an explicit
`initial_noise`, and the first query additionally carry `future_override`:

| check | result |
|---|---|
| F1 identical noise reproduces the action | **0.0** |
| F2 different noise changes the action | 4.6e-3 |
| F3 `future_override` changes the action | 2.7e-2 |

So the pairing is exact tensor-level pairing, not merely a shared seed.

### 5. An infrastructure defect found, corrected, and quantified
F4 (self-override with the *archived* `H_pred` should reproduce the unrepaired chunk) failed at 5.7e-5, and
in-process at 9.6e-5. Investigated rather than tolerated:

- **`H_pred`, `h_t` and `z` are `float32`, not bfloat16** — the LAM decoder's final operations are
  autocast-exempt. My initial assumption was wrong and is corrected here.
- The float16 archive therefore loses ordinary float16 precision: feature values reach **±14.5**, and
  float16 has 10 mantissa bits, so in [8, 16) its ULP is 7.8e-3 and half-ULP rounding gives a maximum
  error of **3.906e-3** per element (median over the 120 states: 3.900e-3). Nothing to do with subnormals.
- **Consequence for this round:** variants B and C must be built on the **live full-precision** `H_pred`.
  `hpred_fp32/` holds exactly that (float32 is lossless for a float32 tensor), and variant A sends no
  override at all, so the server uses its own live tensor. Recorded as protocol amendment **H-A1**, which
  corrects the stated reason and leaves the rule unchanged.
- **Consequence for the frozen C2 result:** C2's repaired variants spliced in `h_real` from the float16
  archive, so the repaired tokens carry a ≤3.9e-3 per-element quantisation. That is immaterial next to the
  size of the repair itself (‖H_pred_j − H_real_j‖ ≈ 7–11 per token), and the headroom test uses the **same**
  archive so it reproduces C2 exactly. Recorded, not silently changed.

### 6. Most conservative conclusion, and what would have stopped the round
"The G3 assets are complete and the interface can pair randomness exactly; one storage-precision defect was
found, explained and confined." Had the noise schedule proved unfixable, the protocol required stopping and
recording the limitation instead of accepting mismatched randomness — that branch was not needed.

**Status: H AUDIT 0 PASSED. Proceeding to the 6-state equivalence audit.**

---

## H AUDIT 1 — six-state equivalence, before any formal rollout (2026-09-20)

**The single question.** Do the four variants' first action chunks reproduce the frozen C2 chunks, and are
snapshot restore and the continuation noise schedule reproducible? Infrastructure only.

**What actually ran.** 6 states = 2 stress conditions × 3 suites × 1 state each, taken deterministically
(first episode of each combination). Per state: 4 variant queries through the websocket path + 3 schedule
queries = 42 queries; plus 12 snapshot restore-and-replay cycles. Artifact:
`h_audit1_equivalence.json`.

### 1. E1 — first chunk vs the frozen C2 chunk (raw action units, tolerance 1e-5)

| state | A baseline | B top-D | C top-(D×S) | D full oracle |
|---|---|---|---|---|
| occlusion / goal t5 | 1.79e-7 | 2.38e-7 | 1.19e-7 | 1.19e-7 |
| occlusion / object t0 | 1.79e-7 | 1.19e-7 | 1.19e-7 | 1.19e-7 |
| occlusion / spatial t2 | 2.38e-7 | 2.38e-7 | 1.19e-7 | 1.19e-7 |
| shift / goal t5 | 2.38e-7 | 1.79e-7 | 1.79e-7 | 1.79e-7 |
| shift / object t0 | 1.19e-7 | 1.79e-7 | 1.19e-7 | 1.19e-7 |
| shift / spatial t2 | 1.19e-7 | 1.19e-7 | 2.38e-7 | 1.19e-7 |

All **1.2–2.4e-7**, i.e. float32 transport rounding, ~40× below the tolerance. Note that variant A is
produced by sending **no override at all**, so it exercises the server's own live `H_pred` — and it
reproduces C2's `A_pred`. That confirms C2's baseline was itself built on the live full-precision tensor,
so the constraint imposed for this round is consistent with the frozen data rather than a change to it.

### 2. E2 — snapshot restore
Restoring the snapshot and replaying the recorded chunk twice gives an end-effector difference of **0.0 m**
on all six states.

### 3. E3 — continuation noise schedule
Replaying the same schedule reproduces the chunk at **0.0**; a different schedule changes it by
1.3e-2 – 6.5e-2. The schedule is therefore both reproducible and effective, which is what makes the four
variants' outcomes attributable to the first-chunk intervention.

### 4. Was anything changed to make the audit pass?
No selector, token budget, stress condition, phase or formula was touched. The only change since the
protocol was frozen is amendment **H-A1**, which corrects the *explanation* of a storage defect while
leaving the construction rule identical, and it was written before any continuation rollout existed.

### 5. Most conservative conclusion
"The headroom experiment can be run as specified: the first chunk of each variant reproduces the frozen G3
chunk to 2.4e-7, the simulator restores exactly, and the continuation randomness is paired tensor-for-tensor
across variants."

**Status: H AUDIT 1 PASSED. The formal 120-state × 4-variant headroom test may now run.**

---

## H AUDIT 2 — task-level results (2026-09-20)

**The single question.** Does a one-chunk oracle correction of the future change the FINAL task outcome?

**What actually ran.** 120 states × 4 variants × 1 matched schedule = **480 continuation rollouts**, plus
the pre-registered robustness subset (24 states × 4 variants × 3 additional matched schedules = **288**),
**768 in total**. Every variant restored the same snapshot independently; every policy query carried an
explicit noise tensor from the state's schedule; only the first query of each rollout carried the repair.
Infrastructure errors: 0. Rollouts excluded: none.

### 1. Primary endpoint — final task success (120 states)

| variant | success | rate |
|---|---|---|
| baseline | 95/120 | **0.792** |
| top-D repair | 92/120 | 0.767 |
| top-(D × S) repair | 91/120 | 0.758 |
| full oracle | 90/120 | **0.750** |

Every intervention is at or below baseline. Paired contrasts (McNemar exact, episode bootstrap CI):

| contrast | paired diff | CI | discordant pairs | p |
|---|---|---|---|---|
| D × S − baseline | −0.0333 | [−0.0833, +0.0083] | 2 up / 6 down | 0.289 |
| full oracle − baseline | −0.0417 | [−0.0919, +0.0083] | 3 up / 8 down | 0.227 |
| top-D − baseline | −0.0250 | [−0.0667, +0.0167] | 2 up / 5 down | 0.453 |
| D × S − top-D | −0.0083 | [−0.0583, +0.0417] | 4 / 5 | 1.000 |

**No contrast is significant; every CI contains zero.** The point estimates are negative.

### 2. The four required case counts (25 baseline failures in total)

| case | count |
|---|---|
| case 1 — rescue: baseline fail → D × S success | **2** |
| case 2 — damage: baseline success → D × S fail | **6** |
| case 3 — oracle headroom: baseline fail → full oracle success | **3** |
| case 4 — beyond one chunk: baseline fail → full oracle also fails | **22** |
| (also) full oracle damaged a previously-successful state | 8 |

**Of 25 baseline failures the full oracle rescues 3 and leaves 22 unfixed, while breaking 8 states that had
succeeded.** Net −5.

### 3. Secondary endpoint — remaining steps to success (paired successes only)
No improvement: D × S vs baseline mean **−0.67** steps, median 0, CI [−3.50, +2.15], faster in 44 % of
pairs; full oracle vs baseline mean −2.01, median 0, CI [−4.44, +0.25]. Nothing survives the CI.

### 4. Robustness subset (3 additional matched schedules, 72 state-schedule pairs)
Baseline 0.694, top-D 0.708, D × S 0.694, full oracle 0.653.
**D × S − baseline = +0.0000 [0.000, 0.000]**; oracle − baseline = −0.0417 [−0.0833, 0.000].
The conclusion does not depend on a single noise schedule.

### 5. Breakdowns
By stress: occlusion baseline 0.883 → oracle 0.817; camera shift 0.700 → 0.683. By source outcome:
states from originally-failed episodes are at 0.281 baseline and 0.281 for the oracle (D × S 0.219); states
from successful episodes 0.977 → 0.920 for the oracle. By task, the only task where the oracle helps is
`goal t8` (0.850 → 0.950); it hurts on `goal t5`, `object t0` and `spatial t7`.

### 6. Is there a simpler explanation? (the one the protocol named in advance)
Yes, and it is the leading one. `H_real` is the future the **baseline** rollout actually reached. As soon
as the repaired chunk produces different actions, that future is no longer the counterfactual future of
those actions — the policy is being conditioned on a future belonging to a *different* action sequence.
That is not "better information", it is potentially *inconsistent* information, which is exactly the
counterfactual mismatch `PROTOCOL_HEADROOM.yaml:variants.oracle_status` warned about. A second, already
documented factor: this checkpoint was trained with `enable_flow_h_t1_scheduled_sampling=false`, so the
action head has never seen a real future as conditioning — feeding one is out-of-distribution.

### 7. Was anything changed after seeing results?
No. No selector, token budget, stress condition, phase, formula, noise schedule or state set was touched;
no new experiment was added; no new metric was invented. The pre-registered endpoints are the ones reported.

**Status: H AUDIT 2 complete.**

---

## FINAL H AUDIT (2026-09-20)

**Verdict: H-NO-HEADROOM.**

The pre-registered definition is met: the full `H_real` intervention improves neither final task success
(0.750 vs a baseline of 0.792) nor completion time (CI contains zero), and it produces no stable rescue
pattern (3 rescues against 8 damages; 22 of 25 baseline failures remain failures).

It is **not H-HARMFUL**: the interventions do lower the point estimates, but not *stably* — every paired CI
contains zero and every McNemar p ≥ 0.23. Calling this "harmful" would overstate the evidence exactly as
much as calling it "helpful" would.

It is **not H-PASS or H-PARTIAL**: both require a stable task-level improvement from the full oracle, and
there is none.

### What this does and does not overturn
- It does **not** overturn G3: under stress `D × S` still ranks action-critical future tokens better than
  `D`, `S`, attention and `D × attention`, and the executed one-chunk deviation is real (+0.231 mm).
- It **does** show that this ranking quality has no demonstrated path to a better task outcome through
  one-chunk future correction. The ~0.9–2.2 mm of executed influence, even when spent optimally, does not
  move a binary task outcome.
- The most likely reason is structural, not a tuning failure: an oracle future taken from a *different*
  action sequence is not the counterfactual future of the repaired chunk.

### Total work and integrity
768 continuation rollouts; 0 infrastructure errors; nothing excluded; no LaWAM parameter trained or
modified (`state_dict` sha256 `37a53b8c…b523`); no repository file patched — the websocket transport was
wrapped research-side. Two engineering defects were found and fixed *before* the experiment: the float16
archive precision issue (amendment H-A1, and the reason I first gave for it was wrong and is corrected in
the record), and a `pgrep` self-match that silently prevented the robustness subset from starting, which
was caught by checking the record count rather than trusting the launcher.

**Status: Headroom test complete. STOPPED. Method Design is NOT started.**
