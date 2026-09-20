#!/usr/bin/env bash
# G3 Confirm stage C1 driver — frozen by PROTOCOL_G3.yaml amendments G3-A1..A4.
# 3 suites x 2 tasks x 10 independent resets x {ID, A_occlusion medium, B_camera_shift strong} = 180 rollouts.
# Family C is NOT part of the confirm stage. No S, no D x S, no ranking in this stage.
set -u
cd "$(dirname "$0")/../../../.." || exit 1
source research/action_critical_reliability_probe/scripts/env_sim.sh
RID=$(cat research/action_critical_reliability_probe/practical_significance/.current_run_id)
RUN=$LAWAM_ROOT/results/action_critical_reliability_probe/$RID/confirm
S=research/action_critical_reliability_probe/practical_significance/scripts/g3_rollouts.py

run () {   # suite task extra
  local suite=$1 tid=$2 extra=${3:-}
  $SIM_PY $S --run_dir "$RUN" --suite "$suite" --task_id "$tid" --episodes 0-9 \
          --family ID --severity none $extra 2>&1 | grep -E "ep[0-9]+:|INVALID"
  $SIM_PY $S --run_dir "$RUN" --suite "$suite" --task_id "$tid" --episodes 0-9 \
          --family A_occlusion --severity medium $extra 2>&1 | grep -E "ep[0-9]+:|INVALID"
  $SIM_PY $S --run_dir "$RUN" --suite "$suite" --task_id "$tid" --episodes 0-9 \
          --family B_camera_shift --severity strong $extra 2>&1 | grep -E "ep[0-9]+:|INVALID"
}

run libero_spatial 2
run libero_spatial 7
run libero_object 0
run libero_object 6
run libero_goal 8 --fresh_env_per_episode
run libero_goal 5 --fresh_env_per_episode
echo "CONFIRM_C1_ROLLOUTS_DONE"
