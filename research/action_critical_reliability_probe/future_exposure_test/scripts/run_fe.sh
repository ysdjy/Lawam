#!/usr/bin/env bash
# FE round driver. Stages are separate on purpose: the protocol stops at gates.
#
#   ./run_fe.sh smoke      20-update engineering check, both arms (EXCLUDED from results)
#   ./run_fe.sh train      the pre-registered 4 runs (2 arms x seeds 0,1)
#   ./run_fe.sh eval       the 2x2 on the evaluation splits, for every trained checkpoint
#   ./run_fe.sh analyze    the interaction, the bootstrap, the decision rule
set -euo pipefail

export LAWAM_ROOT=/home/zbh/Downloads/IsaacLab/Lawam_paper/LaWAM
export PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # amendment FE-A5: bitwise-reproducible training
unset all_proxy ALL_PROXY || true
PY=/home/zbh/anaconda3/envs/lawam/bin/python
S="$LAWAM_ROOT/research/action_critical_reliability_probe/future_exposure_test/scripts"
RUN="$LAWAM_ROOT/results/action_critical_reliability_probe/acr_future_exposure_20260920_231423"
cd "$LAWAM_ROOT"

case "${1:-}" in
smoke)
  # Engineering only. Writes under smoke/ and is never read by the analysis.
  for arm in control future_exposed; do
    "$PY" "$S/fe_train.py" --arm "$arm" --seed 0 --max-updates 20 --val-every 10 \
      --patience 99 --out "$RUN/smoke" --smoke 2>&1 | tail -30
  done
  ;;
train)
  for seed in 0 1; do
    for arm in control future_exposed; do
      echo "=== TRAIN $arm seed $seed ==="
      "$PY" "$S/fe_train.py" --arm "$arm" --seed "$seed" \
        --max-updates 1500 --val-every 100 --patience 4 --out "$RUN/train"
    done
  done
  ;;
eval)
  # Amendment FE-A6: `best` is the primary evaluation (the checkpoint the stopping rule
  # selected); `final` is evaluated too and reported as a secondary so the choice is visible.
  for tag in best final; do
    for seed in 0 1; do
      for arm in control future_exposed; do
        ck="$RUN/train/flow_${arm}_seed${seed}_${tag}.pt"
        [ -f "$ck" ] || { echo "missing $ck"; exit 1; }
        echo "=== EVAL $arm seed $seed ($tag) ==="
        # The secondary `final` pass only needs the flow endpoint; decision condition (e)
        # is evaluated on the primary `best` checkpoints.
        extra=""; [ "$tag" = "final" ] && extra="--skip-actions"
        "$PY" "$S/fe2_evaluate.py" --checkpoint "$ck" --arm "$arm" --seed "$seed" \
          --tag "$tag" --out "$RUN/eval" $extra
      done
    done
  done
  ;;
analyze)
  "$PY" "$S/fe3_analyze.py" --metrics-dir "$RUN/eval" --tag best  --out "$RUN/SUMMARY_FE.json"
  "$PY" "$S/fe3_analyze.py" --metrics-dir "$RUN/eval" --tag final --out "$RUN/SUMMARY_FE_final_checkpoint.json"
  ;;
*)
  echo "usage: $0 {smoke|train|eval|analyze}" >&2; exit 2;;
esac
