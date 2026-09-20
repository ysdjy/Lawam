# REPORT_FE — Future Exposure Test

**Verdict: FE-FAIL.**

Training the Flow Action Head on matched ground-truth future features does **not** make a real
future useful to it. After an arm was trained with 49.9% of its rows conditioned on the true
`H_gt`, handing it the true future instead of the predicted one changes its action by
**0.03–0.10 mm** — smaller than the 0.474 mm flow-sampler nuisance measured in round 1, and
0.04–0.12% of the 85.0 mm the gripper actually moves in a chunk.

Pre-registered in `PROTOCOL_FE.yaml` (frozen before any data existed; seven append-only
amendments, all timestamped before the quantity they touch was measured). Gates in `AUDIT_FE.md`.

---

## 1. The question, and why it was worth asking

The Headroom round ended at **H-NO-HEADROOM**: repairing the action-critical future information
for one chunk did not change task success. `METHOD_HANDOFF.md` named the single unresolved
question — is the wall the **intervention** (a one-chunk oracle future is an inconsistent
counterfactual) or the **channel** (this head, trained with scheduled sampling off, simply cannot
use a better future)?

This round tests the channel by training the channel to use it. It also removes the Headroom
round's structural defect: there, `H_real` was the future the *baseline rollout* reached, so once
a repaired chunk acted differently that future stopped being its counterfactual. Here `h_t`,
`H_gt` and `A_gt` come from one trajectory and one time window by construction — verified by 24
checks over 8 episodes × 3 starts, all passing (AUDIT 0 Q0.5).

## 2. Design

Two arms, identical in every respect except the future-condition sampling:

| | Control (C) | Future-Exposed (F) |
| --- | --- | --- |
| future condition | 100% `H_pred` | fixed 50% `H_pred` / 50% matched `H_gt` |
| realised `H_gt` row fraction | **0.0000** / **0.0000** (seeds 0/1) | **0.4935** / **0.4986** |

Both initialise from `lawam_libero_sft_release`; only `policy_backend.flow` is trainable
(306,405,632 parameters, `non_flow_trainable_names` empty); five frozen module hashes verified
identical before and after every run. Budget 1,500 optimizer updates, validation every 100,
early stop after 4 non-improving validations — the same rule, unrelaxed, for both arms.

**Naming honesty.** This is an *adaptation-held-out* test. It does **not** claim the release
checkpoint never saw these demonstrations — it almost certainly did, since the released SFT
recipe used `train_split_all: true`. The guarantee is narrower and exact: no evaluation episode
is used by **this round's** adaptation, in either arm. Prior exposure is common to both arms and
cancels in the interaction, which is why the interaction is the endpoint.

**Splits** (all six pairwise episode overlaps = 0): train 1,181 episodes / 8,000 samples;
validation 167 / 512; `test_seen_task` 167 / 668 (**primary**); `test_heldout_task` 178 / 712
(auxiliary).

## 3. What happened during training

Under the pre-registered learning rate (1e-4, the official flow LR), three of four runs made the
model **worse** and early-stopped:

| run | stop | final | best | realised `H_gt` rows |
| --- | --- | ---: | ---: | ---: |
| Control seed 0 | early stop | 600 | 200 | 0.0000 |
| Control seed 1 | early stop | 500 | 100 | 0.0000 |
| Future-Exposed seed 0 | early stop | 600 | 200 | 0.4935 |
| Future-Exposed seed 1 | budget exhausted | 1500 | **1500** | 0.4986 |

Future-Exposed seed 1 improved monotonically from update 500 to 1500 (0.0339 → 0.0204). That
looks like the exposure working. **It is not.** Its own training log shows `H_pred`-only and
`H_gt`-only validation loss falling *together*, with the paired gap `L(H_pred) − L(H_gt)` staying
at **+3e-6** at update 1500 — 0.015% of the loss. This seed simply found a productive
optimisation trajectory; it gained nothing from the future.

The learning rate was **not** changed. That would be result-driven tuning, which this round
forbids. It is recorded as a limitation (amendment FE-A6) and revisited in §7.

## 4. Primary endpoint — the 2×2 on `test_seen_task`

Matched flow loss, R = 8 repetitions per sample with τ, ε and the noisy action **identical across
all four cells**; only `h_t1_star` changes. Episode-level bootstrap, 2,000 replicates.

Pooled over seeds:

| quantity | mean | 95% CI |
| --- | ---: | --- |
| C-P | +0.028536 | [+0.026080, +0.031070] |
| C-G | +0.028492 | [+0.026160, +0.031007] |
| F-P | +0.022031 | [+0.020035, +0.024154] |
| F-G | +0.021920 | [+0.019916, +0.024048] |
| **Δ_F = F-P − F-G** | **+0.000111** | **[−0.000056, +0.000350]** — contains zero |
| **interaction** | **+0.000066** | **[−0.000006, +0.000142]** — contains zero |
| Δ_C = C-P − C-G | +0.000045 | [−0.000139, +0.000310] — contains zero |

Δ_F is **0.50%** of F-P against a 2% materiality threshold. The interaction is **0.23%** of C-P
against a 1% threshold.

Per seed, the picture is the same and cleaner:

| | seed 0 (arms matched at ~200 updates) | seed 1 (C 100 vs F 1500 updates) |
| --- | --- | --- |
| Δ_F | +0.000098, CI [−0.000097, +0.000363] | +0.000123, CI [−0.000040, +0.000347] |
| interaction | +0.000052, CI [−0.000054, +0.000194] | +0.000080, CI [−0.000107, +0.000257] |

Both CIs contain zero in both seeds.

## 5. The mandatory FE-A4 context — and why it matters here

Amendment FE-A4, written before the cache had finished building, required two quantities to be
reported *beside* the interaction, because a positive interaction can arise from
training-distribution familiarity alone. Both are now decisive.

**F-G − C-P = −0.006616** [−0.007363, −0.005851]: the Future-Exposed checkpoint, given a real
future, does beat the status-quo model. But **F-P − C-P = −0.006505** — it beats it by almost
exactly as much *without* the real future. Of the 0.006616 advantage, **0.006505 (98.3%) is the
checkpoint simply being better**, and only 0.000111 (1.7%) comes from having the true future.

In millimetres on seed 1, where the gap is largest: F-G − C-P = **−9.33 mm**, F-P − C-P =
**−9.29 mm**. Of a 9.33 mm advantage, **9.29 mm is training length and 0.03 mm is the future.**
And seed 1's Future-Exposed arm ran 1,500 updates against Control's 100 — so even that 9.29 mm is
an artefact of asymmetric stopping, not of exposure.

On seed 0, where both arms stopped at the same point, **F-G − C-P = +0.121 mm, CI
[−0.382, +0.648]** — nothing at all.

**The one place a threshold is met, and why it does not count.** On the *auxiliary*
`test_heldout_task` split, seed 1, the interaction is +0.000478 with CI [+0.000308, +0.000648]
excluding zero, 1.52% of C-P. Decomposed: Δ_C = **−0.000361**, CI [−0.000647, −0.000024],
excluding zero — the *Control* arm is actively **hurt** by a real future, which is the
out-of-distribution penalty documented since round 1. Δ_F meanwhile is +0.000117 with a CI
containing zero. So that "significant interaction" is driven by Control being damaged, not by
Future-Exposed gaining. This is exactly the failure mode FE-A4 was pre-registered to catch, on an
auxiliary split, in one seed.

## 5b. The equal-update comparison settles it

Amendment FE-A7 realised the pre-registered `fairness.equal_update_comparison` at update 300, the
nearest snapshot common to all four runs. This removes the seed-1 training-length confound
entirely, and it is the cleanest result of the round. Pooled, primary split:

| tag | C-P | F-P | **F-G − C-P** | Δ_F | interaction | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `best` (primary) | 0.028536 | 0.022031 | −0.006616 [−0.00736, −0.00585] | +0.000111 | +0.000066 | FE-FAIL |
| `final` (secondary) | 0.030004 | 0.024324 | −0.005851 [−0.00659, −0.00516] | +0.000171 | +0.000042 | FE-FAIL |
| **`u300` (equal updates)** | **0.031050** | **0.031352** | **+0.000130 [−0.00043, +0.00064]** | +0.000173 | +0.000038 | FE-FAIL |

At matched training length the Future-Exposed checkpoint is **not better at all** — F-P is very
slightly *worse* than C-P (+0.000302), and F-G − C-P is +0.000130 with a CI straddling zero. The
entire −0.0066 advantage in the primary analysis was training length, exactly as §5 decomposed
it. All three checkpoint selections agree: Δ_F is 0.50–0.70% against a 2% threshold, and the
interaction is 0.12–0.23% against a 1% threshold, with CIs containing zero throughout.

## 6. Secondary endpoint — action error, in millimetres

Best-of-K (K = 3, noise seeds 101/202/303), paired initial noise, 10 flow steps,
`cfg_guidance_scale` 1.0 — the release deployment settings. Primary split, translation error:

| | C-P | C-G | F-P | F-G | **Δ_F** |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed 0 | 33.409 mm | 33.551 mm | 33.430 mm | 33.531 mm | **−0.100 mm** [−0.278, +0.068] |
| seed 1 | 31.157 mm | 31.247 mm | 21.863 mm | 21.831 mm | **+0.032 mm** [−0.128, +0.190] |

Read against the reference scales fixed in round 1 — in-distribution chunk motion **85.0 mm**,
future-error treatment **0.904 mm**, flow-sampler nuisance **0.474 mm** — giving the model a
perfect future moves its action **less than re-drawing the sampler noise does**, by a factor of 5
to 15. On seed 0 the sign is *negative*: the true future makes the exposed model marginally
worse. Decision condition (e) is reported as "same direction" on `l2_norm_all7` (+0.00021,
+0.00038, both CIs containing zero) but **flips sign on the millimetre metric in seed 0** — it is
not evidence of anything.

## 7. Verdict, and what it does and does not establish

**FE-FAIL.** Core conditions (b), (c) and (d) all fail on the primary split: Δ_F's pooled lower
bound is below zero, Δ_F is 0.50% against a 2% threshold, and the interaction is 0.23% against a
1% threshold with a CI containing zero.

**What this establishes.** The Headroom round's open question is answered on the side of the
**channel**. It is not merely that a one-chunk oracle future is an inconsistent counterfactual —
here the future is the *consistent* counterfactual of the supervised actions by construction, the
head was *trained* to use it for 1,500 updates at a 50% exposure rate, and it still extracts
essentially nothing. Combined with the architectural fact established in round 1 — the action
head reads the future as an **unordered bag** (no positional embedding on the condition tokens;
permuting them changes the action by ~4e-7) — the most economical reading is that this future
channel carries almost no action-relevant information that the head can exploit, and that
scheduled sampling does not create such information.

**What this does NOT establish**, stated plainly:

- **The budget was spent badly.** At the pre-registered LR the checkpoint degrades; three of four
  runs stopped early with the model worse than at initialisation. A gentler LR, or far longer
  training, might let exposure teach something this round could not see. The LR was frozen before
  data and was not changed.
- **Seed 1's asymmetry was a real confound for every absolute comparison** — Control stopped at
  100 updates and Future-Exposed at 1,500 — but it is now resolved rather than merely flagged:
  the pre-registered equal-update comparison at update 300 (§5b, `SUMMARY_FE_u300.json`) removes
  it and reaches the same verdict, with F-G − C-P collapsing to +0.000130 [−0.00043, +0.00064].
  The confound inflated the *absolute* comparison; it never drove the interaction, and correcting
  it does not rescue exposure.
- **This is an offline result.** No simulator was run this round — `/home/zbh/LIBERO` is deleted
  and was deliberately not restored. Nothing here is a claim about task success.
- **It is a 50/50 exposure ratio, one architecture, one dataset.** A different ratio, a
  positional encoding on the future tokens, or a different conditioning mechanism are not tested.

## 8. Integrity

- LaWAM parameters never trained: `requires_grad = False` everywhere outside
  `policy_backend.flow`; `non_flow_trainable_names` empty in all four runs.
- Five frozen module hashes identical to their AUDIT 0 baselines after every run
  (`all_frozen_match_audit0: true`).
- All four runs started from the same Flow-Head initialisation
  (`all_started_from_same_flow_init: true`); all four final checkpoints distinct.
- Released checkpoint file untouched: sha256
  `df8e9c3e94900efbc21a8f347f1616bfc6f284c0de37bc13fb2f6384254905cc`; the round-1 anchor
  procedure is re-run in `CHECKPOINT_MANIFEST.json`.
- **No repository file modified** — `code_changes.patch` records an empty `git diff`.
- Training is bitwise reproducible (amendment FE-A5), so the future-condition mask is provably
  the only difference between the arms.
- Cached-path equivalence against the official `backend.forward()`: **exactly 0.0** on every
  tested sample.

Engineering defects found and recorded rather than hidden (AUDIT_FE.md §FE AUDIT 2): two AUDIT 1
gates I had specified wrongly (bitwise tolerance on a backward pass; a false claim that
`cfg_drop_prob == 0` implies train/eval agreement — the DiT carries 64 dropout modules at p=0.2);
an over-broad `Traceback` filter in a waiter that matched benign `rich` logger noise and launched
a concurrent pass, causing an OOM kill and a ~3 hour stall; a progress-rate display that divided
by total rather than per-split elapsed time and made a healthy process look dead; and a checkpoint
anchor briefly reported as mismatched because I used a new hashing procedure rather than round 1's.

## 9. Consequence for the research line

Per the protocol, FE-FAIL ends the Action-Critical Future line. Across six rounds the mechanism
was real and correctly ranked but never actionable: **U × S** carried complementary information
(round 1); **D × S** recovered 0.822 of the oracle correction under stress without needing
`H_real` (G2/DS/G3); repairing that information changed **no** task outcome (Headroom); and
training the head to use a real future changes the action by **less than the sampler nuisance**
(this round). `METHOD_HANDOFF_FE.md` is **not** written — that is reserved for FE-PASS. Method
Design is not started.
