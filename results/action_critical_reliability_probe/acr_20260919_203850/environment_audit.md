# environment_audit.md — Step 0 (Action-Critical Reliability probe)

run_id: `acr_20260919_203850`　date: 2026-09-19/20　repo: `/home/zbh/Downloads/IsaacLab/Lawam_paper/LaWAM`

Machine-readable sources: `audit0_model.json` (loaded checkpoint, `lawam` env), `audit0_sim.json`
(`libero310` env). Everything below was read out of the **loaded model / running simulator**, not out of
`config.yaml` alone, unless explicitly marked as a config value.

---

## 1. Repository state

| item | value |
|---|---|
| commit | `7d27b9607c22034934a4b70347f8bfaba92bf692` ("security: remove exposed W&B API key") |
| branch | `main` (no pull, no reset, no checkout performed) |
| dirty files | `starVLA/model/framework/latent_world/runtime/output_mapper.py`, `.../runtime/runner.py`, `starVLA/model/framework/vlas/flowmatching_expert.py`, `starVLA/model/framework/vlas/lawam.py` (+113/−9 lines) |
| untracked | `research/`, `weights/` |
| origin of dirty changes | run `bd_20260914_093902` (previous branch diagnostic). They are **opt-in diagnostic hooks only**; copy preserved at `results/branch_diagnostic/bd_20260914_093902/code_changes.patch` |
| new code this round | `research/action_critical_reliability_probe/` (read-only w.r.t. the model so far) |
| disk | 24 GB free on `/` (97 % used) — data volume per phase must be budgeted |

## 2. Checkpoint and weights

| item | value |
|---|---|
| policy checkpoint | `results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt`, 7,174,214,645 B, mtime 2026-09-14 01:24:55 |
| loader | `deployment/model_server/server_policy.load_policy_from_checkpoint` → bf16 → cuda → eval (identical to the deployed server) |
| key counts on load | total 1419: vlm 493, lam 660, vlm_to_lam 19, flow 245, act_query 1, flow_action_query 1 |
| config | `results/Checkpoints/libero/lawam_libero_sft_release/config.yaml` |
| dataset statistics | `.../dataset_statistics.json` — franka only, 1,693 trajectories / 273,465 transitions; action min/max ±0.9375 (xyz), −0.258…0.356 / ±0.375 (rot), 0…1 (gripper, `mask=false`) |
| VLM | `results/Checkpoints/qwen3_weights` (Qwen3-VL-2B-Instruct) |
| LAM | `latent_action_model/logs/dino_large_vae/lam_release` (dino_large_vae) |
| DINOv3 | `weights/dinov3-vitb16-pretrain-lvd1689m` |
| checkpoint modified this round | **no** (read-only) |

## 3. Environments

| | model env | sim env |
|---|---|---|
| python | `/home/zbh/anaconda3/envs/lawam/bin/python` | `/home/zbh/anaconda3/envs/libero310/bin/python` |
| torch | 2.7.1+cu128 | 2.7.1+cpu |
| transformers | 5.2.0 | 4.57.6 |
| numpy | 1.26.4 | 1.26.4 |
| sim | — | mujoco 3.3.2, robosuite 1.4.0, `LIBERO_HOME=/home/zbh/LIBERO` |
| GPU | NVIDIA GeForce RTX 5080 (16 GB), 664 MiB used at audit time | — |

LIBERO suites available and loading: `libero_spatial` (10 tasks), `libero_object` (10), `libero_goal` (10);
each task exposes its init-state list (used later for *independent resets*). Task languages recorded in
`audit0_sim.json`.

## 4. Actual LaWAM interface being probed (verified on the loaded model)

Config values in effect (`audit0_model.json:config_effective`):

| item | value |
|---|---|
| `future_prediction` | **true** |
| `detach_future_feature` | true |
| `enable_flow_h_t1_scheduled_sampling` | false → `flow_h_t1_pred_prob = 1.0` (action head was trained **only** on predicted future, never on real future) |
| `action_horizon` (padding cap) | 50 |
| `horizon_sec` | 0.4 |
| `action_hz` (LIBERO client default) | 20.0 |
| effective chunk length | `floor(0.4 × 20) = 8` executed steps |
| `action_dim` | 32 (first 7 used by the franka embodiment) |
| flow `hidden_dim` / `vision_dim` | 1024 / 768 |
| `num_vision_tokens` | 256 |
| `num_target_vision_tokens` | **−1 → flow head has NO learnable future query tokens** (`flow.future_tokens is None`) |
| `use_state` | false (state never enters the action head) |
| `cfg_guidance_scale` | 1.0 → CFG path disabled at inference |
| `num_inference_steps` | 10 (Euler flow integration) |
| DiT | `AlternateVLDiT`, 16 layers, 16 heads, `attend_text_every_n_blocks=2` |

Measured tensor shapes (batch 1, real LIBERO observation):

| tensor | shape | dtype |
|---|---|---|
| `pred_latent` z | `[1, 1, 32]` | fp32 readout (model bf16) |
| `h_t` | `[1, 256, 768]` | — |
| `h_t1_pred` (**H_pred**) | `[1, 256, 768]` → **N = 256 tokens, D = 768** | — |
| `h_vlm` | `[1, 214, 2048]` | bfloat16 |
| flow initial noise | `[1, 50, 32]` | — |
| returned normalized actions | `[1, 8, 32]` | — |

Data path, as read from code (`lawam.py`, `flowmatching_expert.py`, `cross_attention_dit.py`):

```
o_t ─► Qwen3-VL ─► 8 <ACT_PH> queries ─► VLMToLAMQFormer ─► z [1,1,32]
o_t ─► DINOv3 penultimate + LayerNorm ─► h_t [1,256,768]
(h_t, z) ─► LAMDecoder_v2 (AdaLN, 12 layers) ─► H_pred = h_t1_pred [1,256,768]
cond_encoder_hidden = concat(h_t[256], H_pred[256], enc_vlm(h_vlm)[214])   # cross-attn keys/values
hidden_states       = action_features only (no state token, no future query token)
```

**How H_pred reaches the action head** — this is the exact interface the probe targets:
`AlternateVLDiT` alternates block roles. With 16 layers and `attend_text_every_n_blocks=2`:

- odd blocks 1,3,…,15 → self-attention over action tokens (no conditioning read),
- blocks 0,4,8,12 → cross-attention restricted to the **VLM** token half,
- blocks **2, 6, 10, 14** → cross-attention restricted to the **image** half, i.e. the 512 keys
  `[h_t(256) ‖ H_pred(256)]`.

So H_pred is consumed by exactly 4 cross-attention blocks, always jointly with h_t and with the same mask.
Consequence for this probe: (a) a per-token attention baseline is extractable from those 4 blocks and is
directly comparable between the h_t half and the H_pred half; (b) attention over H_pred is *not* separately
normalised — it shares a softmax with h_t, which must be stated when comparing attention to sensitivity.

Attention processor class in those blocks: `AttnProcessor2_0` (diffusers SDPA) — attention weights are not
materialised by default, so an explicit probe processor will be needed for the attention baseline.

## 5. Timing / horizon correspondence (needed for U later)

| item | value |
|---|---|
| policy query at time t | observation `o_t` (agentview + wrist, 180°-flipped, 256×256) |
| chunk executed | `a_t … a_{t+7}` — all 8 steps executed before the next query (official LIBERO client behaviour) |
| training future frame | `lerobot_datasets._sample_video_delta_indices`: `num_frames=2`, `sec_chunk=0.4` → indices `[0, 7]` |
| therefore H_pred targets | the observation **after 7 executed actions**, i.e. `o_{t+7}` |
| real-future encoder for U | must be `lam.extract_vision_features` on `o_{t+7}` with the identical 180° flip / 256×256 / ImageNet normalisation (helper already exists and was used in the prior run) |
| at inference | `primary_image` has a single frame, so the backend's internal `h_t1_gt` equals `h_t` — it is **not** a real future and must never be used as the U label |

## 6. Numerical floors measured today (3 observations, real images)

| quantity | value |
|---|---|
| action chunk repeatability, same inputs + same fixed initial noise, 4 repeats | **0.0 exactly** (max-abs over all dims) |
| `future_override = H_pred` (self-override) vs default path | **0.0** |
| `latent_override = z` (self-override) vs default path | **0.0** |
| action change from a *different* flow-noise seed | 0.013 – 0.020 max-abs (normalized units) |
| mean \|normalized action\| (first 7 dims) | 0.23 – 0.45 |

The zero repeatability floor is a favourable property: any non-zero action change under a token
perturbation is signal, not sampler noise. The flow-noise seed effect (≈0.015) remains the relevant
*nuisance* scale to compare against.

Simulator side (`audit0_sim.json`): snapshot → 8 replayed steps → restore → identical replay gives
**eef and qpos max-abs difference 0.0**; the controller/gripper internal state restore path from the prior
run still works unchanged. Re-rendering the same state after restore differs from the online frame by at
most 58/255 on isolated pixels (known EGL anti-aliasing + one 2 ms render-lag substep, quantified in the
prior run); mitigation carried over: **model inputs always come from the saved online frame**, never from a
re-render.

## 7. Feature scale statistics (basis for later perturbation calibration — not yet a decision)

Per-token L2 norms, 256 tokens × 3 observations:

| tensor | mean token norm | spread |
|---|---|---|
| `h_t` | **27.713** | std 0.000 (LayerNorm ⇒ every token has norm exactly √768 = 27.7128) |
| `H_pred` | 27.03 – 27.10 | std 0.36 – 0.42, min 25.4, max 27.6 |

`H_pred` element-wise std ≈ 0.976 – 0.978; per-dimension std across tokens: median 0.54, max ≈ 5.4.
Reference "semantic" distances in the same space:

- per-token L2 distance between `H_pred` and `h_t`: MSE 0.126 – 0.222 per element ⇒ ‖·‖₂ ≈ 9.8 – 13.1;
- prior run's two genuinely different futures u_A vs u_B: MSE 0.191 ⇒ ‖·‖₂ ≈ 12.1;
- per-token cos(H_pred, h_t): median 0.955 – 0.971, min 0.18 – 0.42.

So a *local* single-token perturbation should be well below ≈10 in L2; candidate scales of 1 %, 3 %, 10 %
of the token norm (0.28 / 0.83 / 2.77) are 2 – 25 % of a real semantic change. Fixed in `protocol.yaml`
before any sensitivity run.

## 8. Diagnostic interfaces present in the working tree (re-verified today)

| hook | location | status |
|---|---|---|
| `latent_override` | `LatentWorldPolicyBackend.predict_action` | present, default `None`, self-override diff 0.0 |
| `future_override` | same | present, default `None`, self-override diff 0.0 |
| `initial_noise` | `ConditionalFlowMatchingHead.sample_actions_cfg` | present, default `None`, fixed-noise repeat diff 0.0 |
| `return_diagnostics` | backend + `runner.infer_step` + `output_mapper` | present, returns z / h_t / H_pred / future_used / noise as fp32 CPU |
| `return_intermediates` | pre-existing upstream interface | present |
| shape/dtype/finite validation | `_validate_override_tensor` | present (raises `ValueError`) |
| attention extraction | — | **absent**, will have to be added (default-off) for the Step-4 attention baseline |
| per-token perturbation helper | — | absent; can be built entirely on top of `future_override` (no model change needed) |

## 9. Prior results that constrain this round (context, not conclusion)

Run `bd_20260914_093902` (same checkpoint, LIBERO-spatial task 2, 25 states) reported that replacing the
whole future with a *genuinely different, real* future changed the normalized action chunk by only
max-abs ≈ 0.034 and the executed end-effector position by ≤3 mm, while a branch switch requires ≈70 mm;
zeroing the entire future changed actions by ≈0.11, versus 0.16 – 0.47 for changing the instruction.

This is prior evidence that **global** future-utilisation is weak on this checkpoint, and it is precisely
what Step 1 must re-test properly (token level, multiple tasks/phases, calibrated ε, with a noise floor).
It is carried here as a prior, not as a result of this round.

## 10. Open items to resolve before or during Step 1

1. Attention extraction hook does not exist yet (needed only for the Step-4 baseline; default off + regression test).
2. Autograd-based S requires calling the backend outside `@torch.inference_mode` — feasible but secondary
   (finite differences are primary, per protocol).
3. Disk budget: 24 GB free. Feature tensors are 256×768 fp32 = 0.79 MB each; storing H_pred + H_real per
   state is ≈1.6 MB, which is fine, but per-token action chunks (256 tokens × seeds × ε) must be stored as
   summary metrics plus a small raw subset, not in full.
4. Task 2 online rollouts from the prior run can be reused as *candidate states*, but Step 1 requires at
   least 2 tasks; new rollouts on a second task are needed (server + official client path, as before).
