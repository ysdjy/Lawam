"""Cross-task action/state shift audit on official Parquet data; no training."""
import argparse
import collections
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--robotwin', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
targets = ['lift_pot', 'open_microwave', 'hanging_mug', 'turn_switch', 'place_can_basket',
           'stack_blocks_three', 'stack_bowls_three', 'put_object_cabinet', 'stamp_seal']
# Explicit samples reviewed against released instructions and native task definitions.
# No assumption that task_index is an environment task ID or that all 500-row blocks are homogeneous.
starts = {'put_object_cabinet':1500, 'hanging_mug':5000, 'lift_pot':5500,
          'open_microwave':8500, 'place_can_basket':12500, 'stack_blocks_three':22000,
          'stamp_seal':24000}
manifest = {task:list(range(start,start+6)) for task,start in starts.items()}
manifest['stack_bowls_three'] = [23000,23004,23005]
manifest['turn_switch'] = [24500,24501,24503,24504,24505]
episode_task = {ep:task for task,episodes in manifest.items() for ep in episodes}
task_text = {r['task_index']:r['__index_level_0__'] for r in pq.read_table(a.dataset/'meta/tasks.parquet').to_pylist()}
mapping = []
selected = collections.defaultdict(list)
records = []
for file in sorted((a.dataset/'data').glob('**/*.parquet')):
    table = pq.read_table(file)
    episodes = np.asarray(table['episode_index'])
    task_ids = np.asarray(table['task_index'])
    for episode in np.unique(episodes):
        idx = np.flatnonzero(episodes==episode)
        task_id = int(task_ids[idx[0]])
        task = episode_task.get(int(episode))
        if task is None: continue
        # Fixed manually inspected samples; all numerical shift checks are automatic.
        ep = table.take(idx)
        frame = np.asarray(ep['frame_index'])
        if frame[0]!=0: continue
        state = np.stack(ep['observation.state'].to_pylist())
        action = np.stack(ep['action'].to_pylist())
        times = np.asarray(ep['timestamp'])
        if len(state)<40: continue
        selected[task].append(int(episode))
        mapping.append(dict(task=task, episode=int(episode), task_index=task_id, instruction=task_text[task_id], provenance='manually reviewed instruction and task source; no canonical environment ID in metadata'))
        shifts = {}
        for shift in [-2,-1,0,1,2]:
            start=max(0,-shift); end=min(len(state),len(state)-shift)
            error = action[start:end]-state[start+shift:end+shift]
            shifts[str(shift)] = dict(mean_abs=float(np.abs(error).mean()),
                                     max_abs=float(np.abs(error).max()),
                                     exact_fraction=float(np.mean(np.all(error==0,axis=1))))
        future_error = action[36:-1]-state[37:]
        records.append(dict(task=task, task_index=task_id, episode=int(episode), file=str(file),
                            frames=len(state), shifts=shifts,
                            timestamp_dt_max_error=float(np.max(np.abs(np.diff(times)-1/30))),
                            action_t36_equals_state_t37=bool(np.array_equal(action[36:-1],state[37:])),
                            future_shift_max_abs=float(np.abs(future_error).max()),
                            last_action_equals_last_state=bool(np.array_equal(action[-1],state[-1]))))
    del table
    if all(len(selected[t])==len(manifest[t]) for t in targets): break
assert all(len(selected[t])>=2 for t in targets), dict(selected)
report = dict(dataset_revision='560f9e9fecc7dbf826afd6042b418d632e319df1',
              mapping_warning='task_index labels instructions, not environment IDs; only listed samples mapped; full authoritative provenance unavailable',
              mapping_counts=dict(collections.Counter(r['task'] for r in mapping)),
              sampled_episodes=dict(selected), records=records,
              raw_simulator_conversion_provenance='unavailable; numeric equality alone does not prove realized-state provenance')
(a.output/'task_index_mapping.json').write_text(json.dumps(mapping,indent=2))
(a.output/'data_semantics_results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(samples=len(records), tasks=len(selected),
                     shift1_exact=sum(r['shifts']['1']['exact_fraction']==1 for r in records),
                     counts={t:len(v) for t,v in selected.items()})))
