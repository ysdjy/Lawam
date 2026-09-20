#!/usr/bin/env bash
# Environment for model-side scripts (lawam env). Mirrors Lawam_paper/env_lawam.sh.
export LAWAM_ROOT=/home/zbh/Downloads/IsaacLab/Lawam_paper/LaWAM
export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset all_proxy ALL_PROXY
export MODEL_PY=/home/zbh/anaconda3/envs/lawam/bin/python
export RUN_ID=$(cat $LAWAM_ROOT/research/branch_diagnostic/.current_run_id)
export RUN_DIR=$LAWAM_ROOT/results/branch_diagnostic/$RUN_ID
export CUDA_VISIBLE_DEVICES=0
cd $LAWAM_ROOT
