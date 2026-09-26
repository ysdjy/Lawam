"""Audit saved completed rollouts: executed prefixes, clocks, RGB, and terminal evidence."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]/'results/robotwin_phase2_newhost'
records = []
for folder in sorted((root/'episodes').iterdir()):
    if not folder.is_dir() or not (folder/'summary.json').exists(): continue
    summary = json.loads((folder/'summary.json').read_text())
    if summary['status'] != 'completed': continue
    starts = ends = queries = 0
    chunk = None
    cursor = 0
    previous_tick = 0
    terminal = None
    for line in (folder/'episode.jsonl').open():
        event = json.loads(line)
        assert event['physics_tick'] >= previous_tick, folder
        assert abs(event['sim_seconds'] - event['physics_tick']/250) < 1e-10
        previous_tick = event['physics_tick']
        if event['event'] == 'policy_query':
            assert event['command_index'] == queries*36
            assert event['flow_seed'] == summary['seed']*10000+queries
            queries += 1
        elif event['event'] == 'observation':
            assert all((folder/path).is_file() for path in event['rgb_paths'])
            assert all(key in event['state'] for key in ['eef_world','tcp_center_world','joint_qpos','contacts'])
        elif event['event'] == 'full_absolute_action_chunk':
            chunk = event['actions']; cursor = 0
            assert len(chunk) == 36 and all(len(action) == 16 for action in chunk)
        elif event['event'] == 'action_start':
            assert event['command_index'] == starts
            assert event['executed_action'] == chunk[cursor], (folder, starts)
            cursor += 1; starts += 1
        elif event['event'] == 'action_end':
            ends += 1
            assert event['command_index'] == ends
        elif event['event'] == 'terminal_observation': terminal = event
    assert starts == ends == summary['episode_length']
    assert queries == summary['policy_queries'] == (starts+35)//36
    assert terminal is not None and terminal['physics_tick'] == summary['physics_ticks']
    if summary['success']: assert terminal['state']['native_success_predicate']
    assert all((folder/path).is_file() for path in ['video.mp4',summary.get('physical_trace_file','physical_trace.jsonl.gz'),'final_head.png'])
    records.append(dict(episode=folder.name,commands=starts,queries=queries,status='passed'))
(root/'evidence/rollout_integrity.json').write_text(json.dumps(records,indent=2))
print(json.dumps(dict(completed_checked=len(records),status='passed')))
