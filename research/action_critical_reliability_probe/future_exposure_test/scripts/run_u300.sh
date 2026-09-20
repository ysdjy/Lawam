#!/usr/bin/env bash
# Amendment FE-A7: the pre-registered equal-update comparison, flow endpoint only.
set -euo pipefail
export LAWAM_ROOT=/home/zbh/Downloads/IsaacLab/Lawam_paper/LaWAM
export PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/zbh/anaconda3/envs/lawam/bin/python
S="$LAWAM_ROOT/research/action_critical_reliability_probe/future_exposure_test/scripts"
RUN="$LAWAM_ROOT/results/action_critical_reliability_probe/acr_future_exposure_20260920_231423"
cd "$LAWAM_ROOT"
for seed in 0 1; do for arm in control future_exposed; do
  ck="$RUN/train/flow_${arm}_seed${seed}_u300.pt"
  [ -f "$ck" ] || { echo "missing $ck"; exit 1; }
  echo "=== EVAL $arm seed $seed (u300) ==="
  "$PY" "$S/fe2_evaluate.py" --checkpoint "$ck" --arm "$arm" --seed "$seed" \
    --tag u300 --out "$RUN/eval" --skip-actions
done; done
