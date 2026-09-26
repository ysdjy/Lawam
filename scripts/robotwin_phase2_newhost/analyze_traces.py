"""Describe completed official episodes; metrics alone do not assign failure causes."""
import collections
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'results/robotwin_phase2_newhost'
reports=[]
for folder in sorted((ROOT/'episodes').iterdir()):
    if not folder.is_dir() or not (folder/'summary.json').exists():continue
    summary=json.loads((folder/'summary.json').read_text())
    if summary['status']!='completed':continue
    durations=[];position_errors=[];angle_errors=[];planner=collections.Counter();progress=[];queries=[]
    current=None;last_state=None
    for line in (folder/'episode.jsonl').open():
        event=json.loads(line)
        if event['event']=='policy_query':
            queries.append(dict(query=event['query'],command_index=event['command_index'],sim_seconds=event['sim_seconds']))
        elif event['event']=='planner_result':
            planner[str(event['side'])+':'+str(event['result'].get('status','unavailable'))]+=1
        elif event['event']=='action_start':current=event
        elif event['event']=='action_end':
            assert current is not None
            durations.append(event['sim_seconds']-current['sim_seconds'])
            action=np.asarray(current['executed_action'])
            state=event['state'];last_state=state
            for side,start in [('left',0),('right',8)]:
                realized=np.asarray(state['eef_world'][side])
                position_errors.append(float(np.linalg.norm(action[start:start+3]-realized[:3])))
                dot=abs(np.dot(action[start+3:start+7],realized[3:]))
                angle_errors.append(float(2*np.arccos(np.clip(dot,0,1))))
            objects=state['objects']
            articulation={name:obj['qpos'] for name,obj in objects.items() if isinstance(obj.get('qpos'),list)}
            progress.append(dict(command=event['command_index'],sim_seconds=event['sim_seconds'],articulation_qpos=articulation,
                                 predicate_details=state.get('predicate_details',{})))
            current=None
    assert len(durations)==summary['episode_length'], folder
    def stats(values):
        x=np.asarray(values)
        return dict(mean=float(x.mean()),p50=float(np.quantile(x,.5)),p95=float(np.quantile(x,.95)),max=float(x.max()))
    report=dict(task=summary['task'],condition=summary['condition'],seed=summary['seed'],success=summary['success'],
                commands=summary['episode_length'],simulated_seconds=summary['simulated_seconds'],
                command_duration_seconds=stats(durations),eef_target_vs_realized_position_m=stats(position_errors),
                eef_target_vs_realized_orientation_rad=stats(angle_errors),planner_status_counts=dict(planner),
                query_times=queries,final_state=last_state,progress=progress,
                interpretation='Descriptive end-of-planner tracking/progress; not a causal attribution or future-state target.')
    (folder/'trace_analysis.json').write_text(json.dumps(report,indent=2))
    reports.append({k:v for k,v in report.items() if k not in ['final_state','progress']})
(ROOT/'trace_analysis_index.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(dict(completed=len(reports),successes=sum(r['success'] for r in reports))))
