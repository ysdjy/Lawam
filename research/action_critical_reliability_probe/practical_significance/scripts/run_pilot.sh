#!/usr/bin/env bash
# G3 pilot driver: 3 tasks x 5 episodes x 10 conditions (ID + 3 families x 3 severities) = 150 rollouts.
# Conditions and severities come from the frozen PROTOCOL_G3.yaml; nothing here is tunable.
set -u
cd "$(dirname "$0")/../../../.." || exit 1
source research/action_critical_reliability_probe/scripts/env_sim.sh
RID=$(cat research/action_critical_reliability_probe/practical_significance/.current_run_id)
RUN=$LAWAM_ROOT/results/action_critical_reliability_probe/$RID
S=research/action_critical_reliability_probe/practical_significance/scripts/g3_rollouts.py

run_task () {           # suite task_id extra_flags
  local suite=$1 tid=$2 extra=${3:-}
  $SIM_PY $S --run_dir "$RUN" --suite "$suite" --task_id "$tid" --episodes 0-4 \
          --family ID --severity none $extra 2>&1 | grep -E "ep[0-9]+:|INVALID"
  for fam in A_occlusion B_camera_shift C_object_shift; do
    for sev in mild medium strong; do
      $SIM_PY $S --run_dir "$RUN" --suite "$suite" --task_id "$tid" --episodes 0-4 \
              --family "$fam" --severity "$sev" $extra 2>&1 | grep -E "ep[0-9]+:|INVALID"
    done
  done
}

run_task libero_spatial 2
run_task libero_object 0
run_task libero_goal 8 --fresh_env_per_episode
echo "PILOT_ROLLOUTS_DONE"
