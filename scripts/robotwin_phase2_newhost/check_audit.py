"""Small evidence check, including error-denominator and measured TCP geometry."""
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

root=Path(__file__).resolve().parents[2]
out=root/'results/robotwin_phase2_newhost'
paper=json.loads((out/'evidence/paper_table4.json').read_text())
assert len(paper)==50 and len({r['task'] for r in paper})==50
assert abs(sum(r['clean'] for r in paper)/50-92.64)<1e-10
assert abs(sum(r['randomized'] for r in paper)/50-89.8)<1e-10
sem=json.loads((out/'evidence/data_semantics_results.json').read_text())
assert len(sem['sampled_episodes'])==9 and len(sem['records'])==50
assert all(r['shifts']['1']['max_abs']==0 and r['action_t36_equals_state_t37'] for r in sem['records'])
assert all(r['shifts']['0']['max_abs']>0 for r in sem['records'])
for line in (out/'lift_pot_reset/episode.jsonl').read_text().splitlines():
    row=json.loads(line)
    if row['event']=='observation':
        for side in ['left','right']:
            state=row['state'];delta=np.asarray(state['tcp_center_world'][side][:3])-state['eef_world'][side][:3]
            assert abs(np.linalg.norm(delta)-.12)<1e-6
            assert state['gripper_measured_joint_qpos'][side]
spec=importlib.util.spec_from_file_location('screen',Path(__file__).with_name('screen.py'))
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
with tempfile.TemporaryDirectory() as temporary:
    screen.OUT=Path(temporary)
    (screen.OUT/'evidence').mkdir()
    (screen.OUT/'evidence/paper_task_mapping.json').write_bytes((out/'evidence/paper_task_mapping.json').read_bytes())
    for index,(status,success) in enumerate([('completed',False),('system_error',None),('seed_rejected',None)]):
        folder=screen.OUT/'episodes'/str(index);folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps(dict(task='lift_pot',condition='demo_clean',status=status,success=success)))
    cell=screen.summarize()[0]
    assert cell['completed']==1 and cell['failures']==1 and cell['system_errors']==1 and cell['seed_rejected']==1
    assert cell['local_success_percent']==0 and cell['new_method_percent']==''
print('PASS: paper extraction, 50-episode temporal shift, measured EEF/TCP separation, system-error denominator')
