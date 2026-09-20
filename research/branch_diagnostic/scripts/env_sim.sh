#!/usr/bin/env bash
# Environment for simulation-side scripts (libero310 env).
export LAWAM_ROOT=/home/zbh/Downloads/IsaacLab/Lawam_paper/LaWAM
export LIBERO_HOME=/home/zbh/LIBERO
export LIBERO_CONFIG_PATH=/home/zbh/LIBERO/libero
export PYTHONPATH=$LIBERO_HOME:$LAWAM_ROOT
export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export SIM_PY=/home/zbh/anaconda3/envs/libero310/bin/python
export MODEL_PY=/home/zbh/anaconda3/envs/lawam/bin/python
export RUN_ID=$(cat $LAWAM_ROOT/research/branch_diagnostic/.current_run_id)
export RUN_DIR=$LAWAM_ROOT/results/branch_diagnostic/$RUN_ID
cd $LAWAM_ROOT
