# LaWAM two-branch diagnostic (Phase 0 interface validation + Phase 1 same-state A/B diagnosis)

Research question: given the same observation/instruction and two feasible short-horizon plans A/B, does
`z_i -> predicted_future_i -> action_chunk_i -> actual_future_i` hold for LaWAM (LIBERO SFT checkpoint)?

Everything is simulation only (LIBERO-Spatial task 2). No training, no weights modified, no git commit.

## Layout
- `configs/protocol.yaml`  pre-registered rules/thresholds (written before any B0/B1/B2 run).
- `scripts/` (two conda envs: `libero310` for `s*`/`a*` scripts, `lawam` for `m*` scripts)
  - `env_sim.sh`, `env_model.sh`         environment setup (source before running)
  - `sim_common.py`, `scripted.py`       env build (official adapter), full snapshot/restore, scripted OSC reference controller
  - `model_common.py`                    policy loading identical to `deployment/model_server/server_policy.py`
  - `s0_restore_test.py`, `s0b_render_and_dump.py`  Phase-0 check D (state restore), render-noise floor, projection calibration
  - `m1_phase0_tests.py`                 Phase-0 checks A/B/C + validation (`--mode baseline` on pristine code, `--mode patched`)
  - `s1_collect_candidates.py`           original policy online (official ModelClient + websocket server), snapshots at chunk boundaries
  - `s2a_probe_grasp_dirs.py`, `s2_reference_branches.py`  scripted grasp feasibility probe; snapshot rule + reference A/B + qualification
  - `m2_infer_conditions.py`             teacher z_A/z_B, u_A/u_B, B0/B1/B2 action chunks + diagnostics (3 fixed noise tensors)
  - `s3_execute_conditions.py`           R/B0/B1/B2 chunk execution from restored snapshots + executed-branch label + continuation
  - `s4_roi_pixels.py`, `m3_future_metrics.py`  future-feature metrics (global + ROI), predicted-branch label
  - `a1_aggregate.py`, `a2_cases.py`     metrics_by_state.csv, confusion matrices, summary.json, figures, representative cases
  - `m4_sensitivity.py`                  SUPPLEMENTARY (not pre-registered) conditioning-stream sensitivity diagnostic
- Model code changes (opt-in hooks, default behaviour unchanged): see `results/branch_diagnostic/<run_id>/code_changes.patch`
  (`latent_override`, `future_override`, `initial_noise`, `return_diagnostics` on `predict_action`).

## Reproduce (run id in `.current_run_id`)
```
source research/branch_diagnostic/scripts/env_sim.sh
$SIM_PY research/branch_diagnostic/scripts/s0_restore_test.py --run_dir $RUN_DIR
$SIM_PY research/branch_diagnostic/scripts/s0b_render_and_dump.py $RUN_DIR
source research/branch_diagnostic/scripts/env_model.sh
git stash   # (only for the pristine baseline)  && $MODEL_PY .../m1_phase0_tests.py --run_dir $RUN_DIR --mode baseline && git stash pop
$MODEL_PY research/branch_diagnostic/scripts/m1_phase0_tests.py --run_dir $RUN_DIR --mode patched
# server for online candidate collection
$MODEL_PY deployment/model_server/server_policy.py --ckpt_path $CKPT --port 10093 --use_bf16 --idle_timeout -1 &
$SIM_PY research/branch_diagnostic/scripts/s1_collect_candidates.py --run_dir $RUN_DIR --task_id 2 --episodes 0-29
$SIM_PY research/branch_diagnostic/scripts/s2_reference_branches.py --run_dir $RUN_DIR --task_id 2 --episodes 0-29
$MODEL_PY research/branch_diagnostic/scripts/m2_infer_conditions.py --run_dir $RUN_DIR
$SIM_PY research/branch_diagnostic/scripts/s3_execute_conditions.py --run_dir $RUN_DIR --phase pilot --states t02_ep000,...,t02_ep004
$SIM_PY research/branch_diagnostic/scripts/s3_execute_conditions.py --run_dir $RUN_DIR --phase confirm --states $(cat $RUN_DIR/confirm_states.txt)
$SIM_PY research/branch_diagnostic/scripts/s4_roi_pixels.py $RUN_DIR
$MODEL_PY research/branch_diagnostic/scripts/m3_future_metrics.py --run_dir $RUN_DIR
$SIM_PY research/branch_diagnostic/scripts/a1_aggregate.py --run_dir $RUN_DIR --phase confirm
$SIM_PY research/branch_diagnostic/scripts/a2_cases.py $RUN_DIR confirm
```
Rollback of model changes: `git checkout -- starVLA/` (or `git apply -R code_changes.patch`).
