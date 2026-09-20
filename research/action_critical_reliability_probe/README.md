# Action-Critical Reliability probe

Direction-assessment round on the LaWAM LIBERO SFT checkpoint: are future-prediction **unreliability U** and
downstream **action sensitivity S** complementary, and does their combination explain action / execution
consequence better than U, S or attention alone?

No training, no loss, no architecture change, no checkpoint modification. **This round added zero lines of
model code** — it runs on the diagnostic hooks already present in the working tree from run
`bd_20260914_093902` (`latent_override`, `future_override`, `initial_noise`, `return_diagnostics`), plus
research-side code that is proven bit-identical to the untouched `predict_action` path.

Results: `results/action_critical_reliability_probe/<run_id>/` — start with `REPORT.md`, then `AUDIT_LOG.md`.

## Layout

- `configs/protocol.yaml` — pre-registration (thresholds, decision rules, amendments A1–A5 with reasons)
- `scripts/` — two conda envs: `libero310` for `s*`/`a*`, `lawam` for `m*`
  - `env_sim.sh`, `env_model.sh` — source before running
  - `probe_common.py` — paths, stats, json helpers; puts `research/branch_diagnostic/scripts` on `sys.path`
    so the *validated* model/sim helpers of the previous run are reused verbatim
  - `probe_model.py` — shared-encoding + **batched flow-head replay** (the sweep needs ~2.3k samplings per
    state and only `h_t1_star` changes), action distance metrics, deterministic perturbation directions,
    and the attention recorder (`record_attention`)
  - `a0_model_audit.py`, `a0_sim_audit.py` — AUDIT 0: interface, shapes, floors, feature scales
  - `m1_equivalence.py` — fast path == `predict_action` (0.0), in-batch floor, override validation
  - `m1b_batch_noise.py` — batch-noise characterisation (why baselines must be in-batch)
  - `s1_rollouts.py` — undisturbed rollouts of the unmodified policy (needs the websocket server)
  - `s2_select_states.py` — phase labelling + state export
  - `m2_sensitivity.py` — **Step 1**: per-token S (3 ε × 3 directions × 256 tokens) + whole-future controls
  - `s3_replay_futures.py` — snapshot replay: physics reproduction check + render-noise floor for U
  - `m3_uncertainty.py` — **Step 2 + Step 4 L1**: oracle U, E_action, per-token error decomposition
  - `m4_attention.py` — attention baseline with an SDPA-equivalence guard
  - `m5_consequence_chunks.py` / `s4_execute_consequences.py` — **Step 4 L2**: equal-budget causal test
  - `a1_analyze_sensitivity.py`, `a2_complementarity.py`, `a3_final.py` — analyses / decision inputs

## Reproduce

```bash
source research/action_critical_reliability_probe/scripts/env_model.sh   # sets RUN_DIR, MODEL_PY
$MODEL_PY research/action_critical_reliability_probe/scripts/a0_model_audit.py
$MODEL_PY research/action_critical_reliability_probe/scripts/m1_equivalence.py     # must pass
$MODEL_PY research/action_critical_reliability_probe/scripts/m1b_batch_noise.py

source research/action_critical_reliability_probe/scripts/env_sim.sh
$SIM_PY research/action_critical_reliability_probe/scripts/a0_sim_audit.py

# rollouts need the official policy server
$MODEL_PY deployment/model_server/server_policy.py \
    --ckpt_path results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt \
    --port 10093 --use_bf16 --idle_timeout -1 &
$SIM_PY .../s1_rollouts.py --suite libero_spatial --task_id 2 --episodes 0-9
$SIM_PY .../s1_rollouts.py --suite libero_object  --task_id 0 --episodes 0-9
$SIM_PY .../s1_rollouts.py --suite libero_goal    --task_id 8 --episodes 0-9 --fresh_env_per_episode
$SIM_PY .../s2_select_states.py
$SIM_PY .../s3_replay_futures.py
# stop the server before the sweep (it holds ~7 GB of the 16 GB GPU)

$MODEL_PY .../m2_sensitivity.py            # ~80 s/state
$MODEL_PY .../m3_uncertainty.py            # ~10 s/state
$MODEL_PY .../m4_attention.py
$MODEL_PY .../m5_consequence_chunks.py
$SIM_PY   .../s4_execute_consequences.py
$SIM_PY   .../a1_analyze_sensitivity.py && $SIM_PY .../a2_complementarity.py && $SIM_PY .../a3_final.py
```

Known environment quirks (see `AUDIT_LOG.md` AUDIT 0b): `libero_goal` needs `--fresh_env_per_episode`;
`libero_goal` task 0 has a missing asset region in this LIBERO install; building many envs in one process
corrupts LIBERO's object registry and produces misleading cascading errors.

Rollback of model code: nothing to roll back — this round changed none.
