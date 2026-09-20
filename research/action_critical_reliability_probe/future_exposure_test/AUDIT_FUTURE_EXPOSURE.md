# Future Exposure Test audit

## FE AUDIT 0 — official pretrain initialization (2026-09-20)

**Status: complete for source, configuration, split, statistics, and loading. No training or optimizer update has started.** This revision supersedes the initial audit that considered the released LIBERO SFT checkpoint. The release checkpoint is excluded from both arms because its original recipe used `train_split_all: true` and its statistics cover all 1,693 episodes.

### Source and code facts

- Branch: `research/action-critical-reliability-probe`; HEAD: `866690eddf328f4961f69cffa9fd3ea528e32362`.
- Both arms must initialize from the same official `jialei02/lawam_pretrain` revision `62b14a8e8990050ec8aeb1e1b8c8694d2bf60e84`: `results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt`. Its 7,174,214,645-byte file SHA256 is `9fe985abf0ef3d0868f9cc330f7248ade01aaf5891f94528a593eb17d6ba14cc`, matching the official LFS digest. All five repository files match the published sizes.
- The local base VLM file `results/Checkpoints/qwen3_weights/model.safetensors` has SHA256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`. Both arms must use this same file.
- The official `starVLA/config/training/train_libero.yaml` explicitly names `lawam_pretrain` as `trainer.pretrained_checkpoint` and sets `load_pretrained_policy_flow: true`. Its runtime horizon and dataset chunk are 0.4 seconds. The pretrain config uses 1.2 seconds, `data_mix: lam`, and framework name `LatentWorldVLAIndependent`; LIBERO uses `data_mix: libero` and `LaWAM`. The remaining config differences are `num_target_vision_tokens` absent versus `-1` and a 20,000 versus 10,000 ramp field. Use the LIBERO runtime config with pretrain weights, as the official SFT recipe does.
- In `lawam.py`, `_flow_h_t1_pred_prob()` returns 1.0 when scheduled sampling is disabled. `_build_flow_future_condition()` selects `H_pred` at that probability and otherwise matched `H_gt`; `forward()` passes the selected condition into the Flow Head. Thus Control is 100% `H_pred`, while enabled sampling with fixed start=end=0.5 makes Future-Exposed 50/50 with no ramp. `detach_future_feature: true` is mandatory.
- The ordinary evaluation selector returns `H_pred` unconditionally. Paired offline Flow loss must therefore call the Flow Head with an explicit `h_t1_star=H_pred` or `H_gt` while reusing the same noisy action, Flow time, and noise; simply toggling the config in `model.eval()` would not test the two conditions.
- The earlier Headroom `H_real` was captured on a baseline executed path, so a changed action could make it an inconsistent counterfactual future. Here `H_gt` and `A_gt` come from one fixed demonstration trajectory and time window; paired offline comparison does not create that mismatch.

### Checkpoint and runtime load compatibility

- Pretrain and released SFT checkpoint state dicts each contain 1,419 keys, with exactly the same key set, shape, and dtype. Pretrain contains all 245 Flow tensors (306,405,632 parameters).
- Building the LIBERO runtime model and calling the official `TrainerUtils.load_finetune_init_weights(..., load_pretrained_policy_flow=True)` succeeded. Every pretrain key exists in the runtime model with matching shape. All 245 runtime Flow tensors were checked equal to their pretrain source tensors after loading.
- The official relaxed loader warns about 1,002 runtime keys absent from the checkpoint. Of these, 245 `policy_action_head` and 625 `policy_vlm_adapter` keys are aliases of the loaded Flow/VLM modules; object identity was verified. The remaining 132 keys are Qwen language layers 16–27, supplied when the local base VLM is built. The released SFT checkpoint has the same 1,419-key schema. These 132 layers are frozen and included in the runtime VLM hash below; they must be identical across both arms.

### Dataset, alignment, split, and normalization

- Use only the existing `dataset/libero_merged_no_noops_20hz`; no raw HDF5 processing. The official LeRobot data has 1,693 episodes, 273,465 frames, 40 tasks, 20 Hz. Loader and collator smoke tests succeeded with an eight-action chunk and two video frames at offsets 0 and 7.
- For an eligible episode-local start index, `H_t` uses frame 0, `H_gt` frame 7, and `A_gt` actions 0–7 from the same episode. Exclude the last seven possible starts so the loader cannot clip a future frame or pad actions.
- Preserve `SPLIT_MANIFEST.json` unchanged (SHA256 `3483b2936e39352d5fdf973c4c18b04be3a3aa2ea017114038b4de7fb5097df3`). Disjoint complete-episode counts: train 1,181; validation 167; seen-task/unseen-episode test 167 (primary); held-out-task test 178 (auxiliary). Entire tasks 9, 19, 29, and 39 are held out.
- The released SFT statistics cover all 1,693 episodes and are forbidden. Pretrain statistics cover a different mixture and are also unsuitable for this LIBERO split. `TRAIN_ONLY_DATASET_STATISTICS.json` was computed from the 1,181 train episodes (190,893 transitions) in the already downloaded LeRobot Parquet files, with seven action and seven state dimensions. SHA256: `cfcfe12610f588ad60efece50a7ce3cdc2c3be592ed295def8399ab350f59464`. An official LeRobot loader smoke test confirmed that its action and state means are applied through `dataset_statistics_override`.
- The stock `build_dataloaders()` with the official `train_split_all: true` would select all episodes. A future training harness must explicitly subset episode IDs from the manifest and pass the train-only statistics override to every split; the stock unfiltered loader is prohibited. This is an FE AUDIT 1 gate before any optimizer step.
- The official pretrain checkpoint predates the LIBERO SFT initialization in the published recipe, and its config names `data_mix: lam`, rather than `libero`. An exact historical episode list for its broad pretraining is not published here, so pretraining-level duplicate exposure cannot be certified. The defined held-out guarantee is that neither **current LIBERO Flow Head finetune** trains on test episodes.

### Trainability and frozen runtime hashes

After actual pretrain loading, a preflight freeze assertion set only `policy_backend.flow` trainable. The hashes below are of full runtime module `state_dict()` values, including the 132 base VLM keys. Hash method: sorted relative key, NUL, dtype, NUL, shape tuple, NUL, raw tensor bytes. The exact machine-readable baseline is `PREFLIGHT_LOAD_AUDIT.json`.

| Runtime group | Parameters | Trainable | Baseline SHA256 |
| --- | ---: | ---: | --- |
| `vlm` | 2,127,532,032 | 0 | `7a7d7c66f43de04e02a3c5bbf4f629d9edb225ad0b1f623d270007724b37448e` |
| `lam` | 725,095,008 | 0 | `31b6cb29064933e9b52da45fd5317573c57085fca89455d95d36b0adf2216074` |
| `vlm_to_lam` | 145,920 | 0 | `514ca6ebda0f6e68b6f0409ffd111d56fd908df433f47bdc6ba6669a8dbb0bc8` |
| `act_query` | 16,384 | 0 | `d53f1fad9115aff4d5adcb1c16cc7e17fa2585df17ea8d59dcfd2bcd2c691c7c` |
| `flow_action_query` | 16,384 | 0 | `8f82ad7c38b4edf3cd198237ed432e738d24c10420d5b8a39f0de555420d9280` |
| `flow` | 306,405,632 | 306,405,632 | `9ea314de6e6f6c655c4ecfcd0efbf21d7aa10bfcbf479e8fa991315c26846c32` |

Unique frozen total: **2,852,805,728** parameters; unique trainable total: **306,405,632**. Before training, print and assert per-module `requires_grad` and parameter counts again. After training, recompute every frozen module hash and require exact equality.

### Fixed budget, endpoints, and remaining gates

- Two seeds `0,1`, batch size 1, accumulation 8, at most 300 optimizer updates per arm/seed, validation every 50 updates, patience 3, and a two-hour wall limit per arm/seed. Maximum is eight GPU-hours across four runs; actual throughput requires a no-update dry run. Never extend the budget after looking at results.
- The sole arm difference is future-condition sampling. Both arms share source checkpoint, frozen VLM source, train episodes, batch order, optimizer, learning rate, Flow time/noise, update count, and train-only normalization. A separate mask RNG must preserve paired Flow randomness.
- Primary test endpoint: paired `L(H_pred)-L(H_gt)` and its exposure interaction on seen-task/unseen-episode trajectories. Auxiliary: action errors and held-out tasks. Episode bootstrap CIs, two-seed consistency, deployment `F-P` versus `C-P`, and the FE-PASS/FE-FAIL/FE-TRADEOFF thresholds are locked in `PROTOCOL_FE.yaml`.
- FE AUDIT 1 remains pending: verify the actual 100/0 and approximately 50/50 masks, identical batch and Flow RNG traces, episode subsetting/statistics, time alignment, and no-update hardware capacity. FE AUDIT 2 and FINAL FE AUDIT are pending. No FE verdict exists.
