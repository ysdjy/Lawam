# Future Exposure Test — AUDIT

Round FE. Pre-registered in `PROTOCOL_FE.yaml` (frozen; amendments append-only).

This file supersedes `AUDIT_FUTURE_EXPOSURE_superseded_pretrain_variant.md`, which audited a
`lawam_pretrain`-initialised design with recomputed train-only statistics. That design was
retired by the user's locked decisions of 2026-09-21; the file is retained unmodified as a
record and none of its conclusions are carried forward. Its machine-readable artefact
`PREFLIGHT_LOAD_AUDIT.json` and `TRAIN_ONLY_DATASET_STATISTICS.json` are likewise retained but
are NOT used by this round.

---

## FE AUDIT 0 — source, code, split, alignment and feasibility facts

**Status: complete. No optimizer step has been taken. No FE result exists.**
Machine-readable: `results/action_critical_reliability_probe/acr_future_exposure_20260920_231423/PREFLIGHT_FE_AUDIT0.json`
Produced by `research/action_critical_reliability_probe/future_exposure_test/scripts/fe0_facts.py`.

### Q0.1 — What exactly is this round asking?

If the Flow Action Head is trained with matched ground-truth future features, does conditioning
on `H_gt` then beat conditioning on `H_pred` on held-out demonstrations? Operationally: is there
a **Training Exposure × Future Quality interaction**, `(F-G − F-P) − (C-G − C-P)`?

The Headroom round ended at H-NO-HEADROOM and `METHOD_HANDOFF.md` named the one unresolved
question: is the wall the **intervention** (a one-chunk oracle future is an inconsistent
counterfactual) or the **channel** (this head, trained with scheduled sampling off, cannot use a
better future)? This round tests the channel by training the channel to use it.

### Q0.2 — What is the initialisation, and what is honestly claimed about it?

Both arms initialise from `results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt`,
1,419 state-dict keys, 2,555,179,360 runtime parameters — the same checkpoint every prior round
of this line used, with the same anchor. `lawam_pretrain` is present and verified but is **not
used**: initialising the Flow Head from pretrain would change the model's adaptation state and
could not answer whether the *current mature* LaWAM's future channel is usable.

The round is therefore called **adaptation-held-out**, and it does **not** claim the release
checkpoint has never seen these demonstrations. It almost certainly has — the released SFT
recipe used `train_split_all: true` over all 1,693 episodes. The exact guarantee is narrower:
no episode in the evaluation splits is used by **this round's** Flow-Head adaptation, in either
arm. Any prior exposure is common to both arms and cancels in the interaction, which is why the
endpoint is the interaction and not the absolute loss of either checkpoint.

### Q0.3 — What is trainable, and what is provably frozen?

| Group | Parameters | Trainable | Baseline `state_dict` SHA256 |
| --- | ---: | ---: | --- |
| `vlm` | 1,523,500,032 | 0 | `ea9f55ad044e7656a8533325c73f4e7131e2dc6d0b0b38599980823b19126a82` |
| `lam` | 725,095,008 | 0 | `fec680d5cb40778e63f429c8dbe4f15a1fcc40775980069de7ca15e26f9c4d35` |
| `vlm_to_lam` | 145,920 | 0 | `082683df604e813e69dc643b401428c92cf6739685fa9cd5d661ae3e3c636e09` |
| `act_query` | 16,384 | 0 | `6973cac5dbe7a0a9a5bbf9a3047650d70fb782c542068501219260f8dc62f85e` |
| `flow_action_query` | 16,384 | 0 | `7c9e1de21bf4feac96cc562322e22629d5d551ae62d9bf0ccdf274b98374ed7c` |
| **`flow`** | **306,405,632** | **306,405,632** | `477662833e3091d5974e9fa04d65cc47ac8d5463d44e8135b57834c461bd3658` |

Trainable total 306,405,632 = exactly the Flow-Head parameter count; frozen total 2,248,773,728;
`non_flow_trainable_names` is **empty**. Module modes: `backend.training=False`,
`flow.training=True`, `vlm.training=False`, `lam.training=False` — frozen modules in eval, the
Flow Head in train, as the official trainer would have them.

### Q0.4 — Why is it legitimate to back-propagate `loss_flow` alone?

`loss_total = loss_flow + 0.1·loss_perceptual + 0.1·loss_distill`. Neither auxiliary term is a
function of any Flow-Head parameter, so `∂(loss_perceptual + loss_distill)/∂θ_flow = 0` exactly.
Measured on a real batch through the **official** `backend.forward()`:
`loss_flow=0.004394`, `loss_perceptual=0.038337`, `loss_distill=0.006258`, `loss_total=0.008854`;
`aux_requires_grad=False` and **0 of 245** Flow tensors receive a non-`None` gradient from the
auxiliary terms. Gate passed. Back-propagating `loss_flow` alone is exact for the only trainable
module, and avoids building graphs through 2.25 B frozen parameters.

### Q0.5 — Is the future the model is asked to use actually the matched one?

`primary_videos` is `[view, time, C, H, W]` with `num_frames=2` and `video_delta_indices=[0,7]`:
frame 0 is `o_t`, frame 1 is `o_{t+7}`. 24 checks across 8 episodes × 3 starts, **all true**:

- the **future** frame of start `s` is bit-identical to the **current** frame of start `s+7`;
- the two frames of one sample are never identical (consecutive starts are real observations);
- `A_gt(s)[1:8]` is bit-identical to `A_gt(s+1)[0:7]`.

So `h_t`, `H_gt` and `A_gt` come from one trajectory and one time window by construction. This is
the structural difference from the Headroom round, where `H_real` was the future the *baseline
rollout* reached and therefore stopped being the counterfactual future once the repaired chunk
acted differently.

Descriptive separation on 2 samples (mean per-token norm): `‖H_pred−H_gt‖ = 5.03`,
`‖H_pred−h_t‖ = 6.71`, `‖H_gt−h_t‖ = 8.39`. `H_gt` is neither `H_pred` nor `h_t`.

### Q0.6 — Does the data exist, and is the split real?

`dataset/libero_merged_no_noops_20hz`: LeRobot v3.0, 1,693 episodes, 273,465 frames, 40 tasks,
20 Hz — present and complete. The `BLOCKED: ORIGINAL SFT TRAINING DATA UNAVAILABLE` gate does
**not** fire. No raw HDF5 is processed and the deleted `/home/zbh/LIBERO` simulator is not
needed, because this round is offline-only.

`SPLIT_MANIFEST.json` (sha256 `3483b293…97df3`) verified in this process: train 1,181 /
validation 167 / test_seen_task 167 / test_heldout_task 178 = 1,693; **all six pairwise overlaps
are 0**; the union is exactly episodes 0–1692. Restricting the loader to the 1,181 train episodes
yields **190,893** active steps, matching the manifest's claim exactly, and `len(dataset)` agrees.
182,626 of those are eligible starts (excluding the last 7 of each episode so no future frame is
clipped and no action is padded); no episode has zero eligible starts; episode lengths 75–505.

The stock `build_dataloaders(train_split_all=true)` selects all 1,693 episodes and is not used.
Subsetting mirrors the five assignments the official `_build_mode_split_from_trajectories()`
makes and then calls the dataset's own `_build_active_step_indexing()`. **No repository file is
modified.**

### Q0.7 — What did AUDIT 0 discover that changed the protocol?

Two things, both recorded as append-only amendments, both **before** any optimizer step:

- **FE-A1** — precision. Every previous round loaded the checkpoint with `use_bf16=True`, the
  deployment cast. That cast *cannot execute the official training forward*: it raises
  `RuntimeError: expected scalar type Float but found BFloat16` in `_compute_distill_loss →
  lam.get_latent_action → vq._encode`, because the LAM teacher's VQ runs in fp32. This round
  uses the training contract — fp32 weights plus the model's own `_cuda_autocast(bf16)` per
  stage. It is identical for both arms.
- **FE-A2** — addressing. `LeRobotMixtureDataset.__getitem__` treats its argument as an RNG
  *seed*, not an address: it calls `sample_step()`, which draws a random (dataset, trajectory,
  step). A pre-registered reproducible sample set is impossible through that entry point, so
  samples are fetched by explicit `(episode, start)` through the dataset's own
  `_select_video_keys_for_sample → get_step_data → transforms → _build_output_sample`. Only the
  address selection is replaced. Verified deterministic: repeating a fetch reproduces
  `primary_videos`, `wrist_images`, `action` and `state` bitwise.

### Q0.8 — Is it feasible on this machine, and what are the frozen engineering numbers?

RTX 5080, 15.5 GiB. Encoder peak **7.2 GiB allocated / 7.3 GiB reserved**, **0.136 s/sample**.
Cached frozen encoding is **3.11 MiB/sample** — `h_vlm [219,2048]` bf16, `h_t`/`H_pred`/`H_gt`
`[256,768]` fp32, `actions [50,32]`, `actions_mask`, `attention_mask`. Projected cache: 24.3 GiB
(8,000 train) + 1.6 GiB (512 validation) + 4.2 GiB (1,380 evaluation) ≈ 30 GiB, against a
pre-registered 100 GiB ceiling and 193 GiB free. **FE-A3**: the ceiling rule is discharged and
`n_train_samples` stays at 8,000. The cache is arm-independent and computed once, so the fp32
encoder and the optimizer are never resident at the same time.

Config facts read back off the live model, not off a YAML: `detach_future_feature=True`,
`enable_flow_h_t1_scheduled_sampling=False`, `flow_only_mode=False`, `future_prediction=True`,
`repeated_diffusion_steps=2`, `cfg_drop_prob=0.0`, `token_independent_noise=False`,
`use_state=False`, `horizon_sec=0.4`, `num_inference_steps=10`, `cfg_guidance_scale=1.0`.
`cfg_drop_prob=0.0` matters for fairness: the Flow Head draws **no** extra random number in
training mode, so the two arms' flow RNG streams cannot drift apart for that reason.

Confirmed in code and by execution: the eval-mode future selector returns `H_pred`
unconditionally, so the 2×2 evaluation **cannot** be implemented by toggling a config — the Flow
Head must be called with an explicit `h_t1_star`.

### Q0.9 — Normalization

`results/Checkpoints/libero/lawam_libero_sft_release/dataset_statistics.json`, sha256
`f25770e2efe656c7bf695efb44247cf6217ab43efdc8632c77bcdfcf640bd2ed`, covering 1,693 trajectories
and 273,465 transitions, with `action.mask = [T,T,T,T,T,T,F]`. Used unchanged for train,
validation and every evaluation split, in both arms. **Nothing is recomputed.** The superseded
`TRAIN_ONLY_DATASET_STATISTICS.json` is not used.

### Q0.10 — What is still open?

FE AUDIT 1 (cached-path equivalence, realised exposure ratio, paired RNG, freeze re-verification)
and the 20-update smoke run. Then formal training. **No FE verdict exists.**

---

## FE AUDIT 1 — the gates, and the two I had mis-specified

**Status: all seven gates pass. Smoke run complete. No pre-registered optimizer step has been
taken at the time of writing this section, and no FE result exists.**
Machine-readable: `.../acr_future_exposure_20260920_231423/PREFLIGHT_FE_AUDIT1.json`
Produced by `scripts/fe1_audit.py`.

### Q1.1 — Is the cached path the official path?

This round trains on a Stage A cache of the frozen encoding, because everything except the Flow
Head is frozen and in eval mode and augmentation is off, which makes that encoding a
deterministic function of the `(episode, start)` address. The claim that this is an *exact*
optimisation rather than an approximation is not assumed — it is measured.

**G1.** On four real samples, the flow loss computed from the cache is compared against the loss
the **official `backend.forward()`** produces for the same sample under the same flow RNG. To
make the comparison possible at all, the Flow Head's bound `forward` is temporarily wrapped to
seed immediately before it runs — necessary because `_compute_distill_loss` samples from the LAM
VQ *before* the flow call, so the global stream cannot simply be seeded beforehand. Result:

| episode | start | official `loss_flow` | cached `loss_flow` | abs diff |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 96 | 0.0006475899135693908 | 0.0006475899135693908 | **0.0** |
| 5 | 174 | 0.00047183450078591704 | 0.00047183450078591704 | **0.0** |
| 7 | 29 | 0.0014597981935366988 | 0.0014597981935366988 | **0.0** |
| 8 | 30 | 0.0448489710688591 | 0.0448489710688591 | **0.0** |

Bit-for-bit. The cached path *is* the official path.

**G2.** The VLM sequence length varies with the task instruction (measured 206–219 tokens) and
the collator right-pads to the batch maximum. A sample cached inside a 2-sample batch would
therefore carry padding decided by whichever sample shared its batch. The cache is built at
**batch size 1** — every sample at its own true length — and `collate_cached` re-creates the
official right padding at use time. Gate: four samples of lengths 219/218/213/212 evaluated
unpadded and then right-padded to 219 give **identical losses to the last bit** (max abs diff
0.0). Padding marked by `attention_mask == False` does not touch a sample's loss.

### Q1.2 — Is the future condition exactly what the two arms are supposed to get?

**G3**, over 4,000 drawn rows per arm: Control `H_gt` fraction **0.000** (exactly zero, not
approximately); Future-Exposed **0.491**. Critically, 481 samples received `H_pred` in one
diffusion repeat and `H_gt` in the other — confirming the mask is drawn over the **repeated**
batch, in the same order `_build_flow_future_condition` uses (repeat first, then draw), rather
than per-sample before the repeat.

The **smoke run** confirms this end-to-end on the real training loop: Control realised
`gt_row_fraction` **0.0**; Future-Exposed **0.528** over 320 rows — within 1σ of 0.5 (σ = 0.028
at that row count). The protocol's ±0.02 tolerance applies to the full run, where 24,000 rows
give σ = 0.0032.

### Q1.3 — Two gates I had specified wrongly

Both were my errors, both are recorded as amendment **FE-A5**, and in neither case was the fix to
loosen a tolerance.

**G4 — I asked for bitwise equality on a path that contains a backward pass.** Two
identically-configured 20-update runs differed by **1.4e-8**. That is CUDA reduction order in the
backward pass, not an RNG leak — an RNG leak would change τ and ε and move the loss by O(1). This
is the same shape of mistake as round 1's E4 check, where I set tolerance 0.0 for a cross-batch
comparison. The fix there was to compare the right thing; the fix here is to *remove the
nondeterminism* rather than tolerate it: under
`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the two runs are
**bitwise identical (max abs diff 0.0)**. Cost: none worth reporting — 0.266 s/update either way.
The gate also gained the exact statement of what it was always trying to prove: advancing the
mask generator by 0, 1 or 17 draws leaves the flow loss identical to the last bit
(0.0015973227564245462 three times over). The mask stream cannot perturb the flow stream.

**G5 — I asserted something false about the model.** The original gate required `flow.train()`
and `flow.eval()` to agree numerically "because `cfg_drop_prob == 0`". They differ by 0.0015 on a
0.0025 loss — 60%, nowhere near numerical noise. The assertion conflated two different dropouts:
`cfg_drop_prob` is the Flow Head's classifier-free-guidance dropout, and it is indeed 0, but the
**AlternateVLDiT carries 64 `nn.Dropout` modules at p = 0.2** (`final_dropout=True`), active in
train mode. That is the official training regulariser the release checkpoint was trained with,
not a defect. The corrected gate verifies the explanation decisively rather than assuming it:
setting every Dropout `p` to 0 makes train mode **exactly equal** eval mode. It then requires
determinism *separately* in each mode, which holds.

Consequence for pairing — and this matters: the dropout masks are drawn **inside** the flow
forward, *after* `torch.manual_seed(flow_seed)`, and their number does not depend on `h_t1_star`.
So the two arms receive identical dropout masks as well as identical τ and ε. Mode policy is now
explicit: **training** runs `flow.train()` (dropout on), matching the official trainer;
**validation and the 2×2 evaluation** run `flow.eval()` (dropout off), matching
`predict_action`'s deployment path.

### Q1.4 — Does anything except the Flow Head move?

**G6.** After five real optimizer steps, all five frozen module hashes are **unchanged**
(`vlm ea9f55ad…`, `lam fec680d5…`, `vlm_to_lam 082683df…`, `act_query 6973cac5…`,
`flow_action_query 7c9e1de2…`) and the Flow-Head hash **has** changed. The smoke run repeats this
check over 20 real updates in both arms: `frozen_unchanged: true`, `flow_changed: true`, both
times. **G7** re-asserts on the live path that the auxiliary losses give zero Flow-Head gradient.

### Q1.5 — Is the sample set what the protocol says?

Cache built: train **8,000** samples over **1,181** episodes, validation **512** over **167**,
`test_seen_task` **668** over **167**, `test_heldout_task` **712** over **178** — every split's
addresses touch every one of its episodes. All six realised pairwise episode overlaps are **0**.
Each split's loader was restricted to that split's episodes *before* any sample was fetched, so
disjointness is enforced mechanically rather than trusted. Total cache 31 GB, matching the
FE-A3 projection.

### Q1.6 — Feasibility, measured

0.266 s/update; ~10 minutes per 1500-update run including 15 validations; ~45 minutes for all
four runs. The smoke run took 41 s per arm for 20 updates with validation every 10.

### Q1.7 — What the smoke run is and is not

20 updates, both arms, seed 0, written under `smoke/`. It checked exactly four things: only the
Flow Head updates, the future-sampling ratio is right, the validation path runs, and the memory
and throughput are what AUDIT 0 projected. **Its checkpoints have been deleted** and its logs are
never read by the analysis. Formal training starts again from the released checkpoint and does
not consume any part of the 1,500-update budget.

### Q1.8 — Gate summary

| Gate | Result |
| --- | --- |
| G1 cached-path equivalence vs official `forward()` | **pass**, max abs diff 0.0 |
| G2 padding independence | **pass**, max abs diff 0.0 |
| G3 future condition (0% / ~50%, mask over repeated batch) | **pass** |
| G4 paired randomness + mask-stream independence | **pass**, bitwise |
| G5 determinism per mode + dropout explanation verified | **pass** |
| G6 freeze after real optimizer steps | **pass** |
| G7 auxiliary-loss gradient gate | **pass** |

`ALL_GATES_PASSED: true`. Training is authorised by the protocol. **No FE verdict exists.**

---

## FE AUDIT 2 — execution: what ran, and every defect in how it ran

### Q2.1 — Did the four pre-registered runs execute as specified?

Yes. `CHECKPOINT_MANIFEST.json`: `all_runs_completed: true`,
`all_started_from_same_flow_init: true`, `all_frozen_match_audit0: true`,
`all_exposure_ratios_within_tolerance: true`, `all_final_checkpoints_distinct: true`.

| run | stop | final | best | realised `H_gt` rows | target |
| --- | --- | ---: | ---: | ---: | ---: |
| Control seed 0 | early stop (patience) | 600 | 200 | 0.0000 | 0.0 |
| Control seed 1 | early stop (patience) | 500 | 100 | 0.0000 | 0.0 |
| Future-Exposed seed 0 | early stop (patience) | 600 | 200 | 0.4935 | 0.5 |
| Future-Exposed seed 1 | budget exhausted | 1500 | 1500 | 0.4986 | 0.5 |

The stopping rule was applied unchanged to all four. No budget was extended, no learning rate was
touched, no arm, ratio or seed was added.

### Q2.2 — Did the pre-registered learning rate work?

**No, and this is reported rather than repaired.** Three of four runs drove the validation loss
*above* its value at initialisation within a few hundred updates and early-stopped. Amendment
FE-A6 records the observation and the decision not to act on it: changing the learning rate after
seeing that would be result-driven tuning, which this round forbids. It is carried into
REPORT_FE.md §7 as a limitation on how far the verdict generalises.

### Q2.3 — Was the seed asymmetry handled?

Future-Exposed seed 1 ran the full 1,500 updates while Control seed 1 stopped at 500 (best 100).
Amendment FE-A7 invokes the `fairness.equal_update_comparison` clause pre-registered in §8 and
realises it at update 300 — the nearest validation-aligned snapshot saved in all four runs by the
frozen `SNAPSHOT_EVERY = 300` rule. Only the stopping updates and the training-log validation
curves were consulted to write it; no 2×2 cell existed at that moment.

### Q2.4 — Defects in execution

Four, all mine, all recorded rather than quietly fixed.

1. **Over-broad failure filter caused an OOM and a ~3 hour stall.** A background waiter used
   `grep -qE "Traceback"` as its failure signal. The `rich` logging library raises a benign
   `TypeError: 'Segment' object is not callable` while rendering a startup log table; the handler
   swallows it and the process continues normally. The filter matched that noise, declared the
   evaluation finished, and launched the equal-update pass concurrently while `fe4_manifest` was
   also loading a 7 GB checkpoint. Each evaluation process holds the frozen modules on CPU
   (~10 GB RSS), so three together exceeded 31 GB. The OOM killer took the u300 pass; the main
   evaluation survived but crawled for about three hours under memory pressure before recovering
   to 3.3 samples/s. **No data was lost** — the killed pass writes only at the end and had
   produced nothing, and no completed result was touched. Fix: wait on a completion-file count,
   never on log keywords, and run heavy passes strictly serially. This is the same family as the
   Headroom round's `pgrep` self-match: trusting a launcher's log instead of its output.

2. **A progress display that made a healthy process look dead.** The samples/s figure divided by
   total elapsed time rather than per-split elapsed time, so after the stall it printed `0.0/s`
   while the process was in fact doing 3.3/s. It was what made me believe the run had died. Fixed
   to per-split timing. Worth recording because a misleading instrument nearly caused me to kill
   and restart a healthy 3-hour job.

3. **A checkpoint anchor briefly reported as MISMATCHED when nothing was wrong.**
   `fe4_manifest.py` first hashed the raw checkpoint file with dtype and shape metadata and native
   dtype bytes, giving `7162f2d5…` against the round-1 anchor `37a53b8c…`. The weights were
   untouched; the **procedure** differed. Round 1's `g2_audit0.state_dict_hash` hashes the
   *loaded* runtime backend (loaded with `use_bf16=True`, the deployment cast), casts every tensor
   to float32 before taking bytes, and includes no dtype or shape metadata. The parameter count
   agreed exactly (2,555,179,360) under both procedures, which was the clue. The original
   procedure is now reproduced verbatim as `fe_common.legacy_state_dict_hash` so the digest is
   comparable across all six rounds. Recorded because "the checkpoint anchor failed" is exactly
   the kind of alarm that must never be reported without first checking one's own instrument.

4. **The two mis-specified AUDIT 1 gates** (amendment FE-A5), described in Q1.3 above: a bitwise
   tolerance demanded of a path containing a backward pass, and a false assertion that
   `cfg_drop_prob == 0` implies `flow.train()` equals `flow.eval()`. Neither was fixed by
   loosening a tolerance.

### Q2.4b — Anchor, resolved

Re-run with the reproduced round-1 procedure, the LaWAM `state_dict` digest is
`37a53b8c799c39725a18900eeaa687d1d2cebc24fca28d8e1f2881ca2870b523` over 2,555,179,360 parameters
— **exactly** the anchor carried since round 1. `lawam_checkpoint_untouched: true`. The released
checkpoint file sha256 is `df8e9c3e94900efbc21a8f347f1616bfc6f284c0de37bc13fb2f6384254905cc`.

### Q2.5 — What was NOT done

No simulator was run; `/home/zbh/LIBERO` remains deleted and was deliberately not restored. No raw
HDF5 was processed. No dataset statistics were recomputed. No repository file was modified
(`git status --porcelain --untracked-files=no` is empty). Nothing was pushed.

---

## FINAL FE AUDIT — verdict

### QF.1 — What is the verdict, and against which pre-registered rule?

**FE-FAIL**, on the rule frozen in `PROTOCOL_FE.yaml` §11 before any data existed. On the primary
split `test_seen_task`, three core conditions fail:

| condition | threshold | observed (pooled, `best`) | met |
| --- | --- | --- | --- |
| (a) Δ_F > 0 in both seeds | — | +0.000098, +0.000123 | yes |
| (b) pooled Δ_F lower bound > 0 | > 0 | **−0.000056** | **no** |
| (c) Δ_F ≥ 2% of F-P | 0.02 | **0.0050** | **no** |
| (d) interaction CI > 0 and ≥ 1% of C-P | 0.01 | **CI [−0.000006, +0.000142], 0.0023** | **no** |
| (e) action metrics same direction | — | same sign on `l2_norm_all7`, **sign flips on mm in seed 0** | equivocal |
| (f) deployment upper bound < 2% | 0.02 | ratio CI [−0.460, +0.075], unstable | no |

The verdict is identical for all three checkpoint selections — `best` (primary), `final`
(secondary) and `u300` (the equal-update comparison) — so it does not rest on which checkpoint the
stopping rule picked.

### QF.2 — Is the verdict an artefact of any confound the protocol named in advance?

The protocol named three, and each was measured rather than assumed.

- **FE-A4, tautological interaction.** Would inflate a PASS, not cause a FAIL. Irrelevant to the
  direction of this verdict, but its mandatory reporting caught the one place a threshold *was*
  met — `test_heldout_task` seed 1, interaction 1.52% of C-P with a CI excluding zero — and showed
  it was driven by Δ_C = −0.000361 (the Control arm being *hurt* by an out-of-distribution real
  future), not by Future-Exposed gaining. That split is auxiliary and that is one seed.
- **FE-A7, training-length asymmetry.** Resolved, not flagged: at matched update 300,
  F-G − C-P = +0.000130 [−0.000432, +0.000643], i.e. nothing. The −0.0066 apparent advantage was
  training length.
- **FE-A6, a learning rate that degrades the checkpoint.** Real, unrepaired by design, and the
  main limitation on generalisation. It is stated in REPORT_FE.md §7 as such.

### QF.3 — Could the result be a measurement failure rather than a finding?

The chain that would have to be broken is short, and every link was verified:

- the cached path reproduces the official `backend.forward()` `loss_flow` **bit-for-bit** (G1);
- right padding does not change a sample's loss (G2, exactly 0.0);
- Control received **exactly** 0.0000 real-future rows and Future-Exposed 0.4935 / 0.4986 (G3 and
  the training logs);
- the two arms are bitwise identical when the mask is disabled, so the mask is the only
  difference (G4 under deterministic kernels);
- `H_gt` is the matched future of the same time window, verified on 24 checks (AUDIT 0 Q0.5), and
  is demonstrably neither `H_pred` nor `h_t`;
- conditioning on `H_gt` **does** move the loss — the gate required it and it holds — so the
  channel is wired, it is just not useful.

A model that had learned to use the future would show it: it was trained on it for 1,500 updates
at a 49.9% exposure rate, and its own validation log shows `L(H_pred)` and `L(H_gt)` falling
together with a gap of +3e-6.

### QF.4 — What follows

`METHOD_HANDOFF_FE.md` is **not** written; the protocol reserves it for FE-PASS. Method Design is
**not** started. Per §11, FE-FAIL ends the Action-Critical Future line. No further variant, ratio,
seed or budget is added to rescue the result.

### QF.5 — Integrity, final

LaWAM `state_dict` anchor `37a53b8c…b523`, 2,555,179,360 parameters — unchanged, verified with the
round-1 procedure. Five frozen module hashes match their AUDIT 0 baselines after every run. No
repository file modified. Nothing pushed. Prior rounds' run directories, REPORTs and failure data
untouched; the superseded pretrain-variant protocol and audit are retained unmodified alongside
the ones that replaced them.
