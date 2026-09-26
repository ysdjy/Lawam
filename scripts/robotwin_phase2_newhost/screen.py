"""Bounded single-worker fixed-seed screening; keeps system errors separate."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/robotwin_phase2_newhost'
PROTO = json.loads((OUT/'protocol.json').read_text())
FIELDS = ['task','condition','source','paper_percent','planned','completed','successes',
          'failures','system_errors','seed_rejected','local_success_percent','new_method_percent']


def summarize():
    paper = json.loads((OUT/'evidence/paper_task_mapping.json').read_text())
    by_task = {r['environment_module']:r for r in paper}
    summaries = []
    videos = []
    paths=list((OUT/'episodes').glob('*/summary.json'))+list((OUT/'attempts').glob('*/summary.json'))
    for path in sorted(paths):
        r = json.loads(path.read_text())
        summaries.append(r)
        if (path.parent/'video.mp4').exists():
            videos.append(dict(task=r['task'], condition=r['condition'], seed=r['seed'],
                               status=r['status'], success=r['success'], video=str(path.parent/'video.mp4'),
                               episode=str(path.parent/'episode.jsonl'), trace=str(path.parent/r.get('physical_trace_file','physical_trace.jsonl.gz')),
                               finalized=r['status']=='completed'))
    rows=[]
    cells=[('lift_pot','demo_clean',3)]+[(t,c,5) for t in PROTO['stage_B']['tasks'] for c in PROTO['stage_B']['conditions']]
    for task,cond,n in cells:
        selected=[r for r in summaries if r['task']==task and r['condition']==cond]
        completed=[r for r in selected if r['status']=='completed']
        successes=sum(r['success'] is True for r in completed)
        row=dict(task=task,condition=cond,source='local_baseline',
                 paper_percent=by_task[task]['paper_clean' if cond=='demo_clean' else 'paper_randomized'],
                 planned=n,completed=len(completed),successes=successes,failures=len(completed)-successes,
                 system_errors=sum(r['status']=='system_error' for r in selected),
                 seed_rejected=sum(r['status']=='seed_rejected' for r in selected),
                 local_success_percent=100*successes/len(completed) if completed else '',new_method_percent='')
        rows.append(row)
    with (OUT/'robotwin_screen_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    (OUT/'video_index.json').write_text(json.dumps(videos,indent=2))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['A','B','summary'],required=True)
    p.add_argument('--timeout',type=int,default=600)
    a=p.parse_args()
    if a.stage=='summary': print(json.dumps(summarize()));return
    assert (OUT/'server_load.json').exists(), 'Official checkpoint server must load successfully first'
    assert json.loads((OUT/'sapien_rt_none/result.json').read_text())['status']=='passed'
    assert 'NVIDIA GeForce RTX 5080' in (OUT/'evidence/vulkaninfo_summary.txt').read_text()
    if a.stage=='B':
        for asset in ['objects','textures']:
            assert json.loads((OUT/f'evidence/{asset}_verified.json').read_text())['crc_all_matched'], 'Complete randomized asset distribution required'
        control=summarize()[0]
        assert control['completed']>=3 and control['successes']>0, 'Stage A control gate incomplete; investigate before screening'
    checkpoint=ROOT/'results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt'
    stage=PROTO['stage_'+a.stage]
    tasks=[stage['task']] if a.stage=='A' else stage['tasks']
    conditions=[stage['condition']] if a.stage=='A' else stage['conditions']
    for task in tasks:
        for condition in conditions:
            for seed in stage['seeds']:
                target = stage.get('target_completed',stage.get('target_completed_per_cell',5))
                cell = next(r for r in summarize() if r['task']==task and r['condition']==condition)
                if cell['completed'] >= target: break
                out=OUT/'episodes'/f'{task}__{condition}__{seed}'
                if out.exists():
                    existing=json.loads((out/'summary.json').read_text())
                    assert existing['status'] in ['completed','seed_rejected'], 'System error/interrupted attempt needs repair and same-seed audit; cannot silently skip'
                    continue
                assert shutil.disk_usage(OUT).free>5*1024**3, 'Less than 5GiB free; stop safely'
                out.parent.mkdir(exist_ok=True)
                cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=8G','-p','MemorySwapMax=0',
                     'timeout','-k','10',str(a.timeout),'taskset','-c','16-23','env','-u','PYTHONPATH','PYTHONNOUSERSITE=1',
                     'PYTHONUNBUFFERED=1','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',
                     str(ROOT.parent/'.venvs/robotwin_poc/bin/python'), str(ROOT/'scripts/robotwin_phase2_newhost/episode.py'),
                     '--robotwin',str(ROOT.parent/'RoboTwin_poc'),'--official-code',str(OUT/'official_code'),
                     '--checkpoint',str(checkpoint),'--output',str(out),'--task',task,'--condition',condition,'--seed',str(seed),
                     '--serial-camera']
                print(json.dumps(dict(event='start',task=task,condition=condition,seed=seed)),flush=True)
                with (out.parent/(out.name+'.log')).open('w') as log:
                    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                summary_path=out/'summary.json'
                row=json.loads(summary_path.read_text()) if summary_path.exists() else dict(
                    source='local_baseline',task=task,condition=condition,seed=seed,success=None,episode_length=0)
                if result.returncode!=0 or row.get('status') not in ['completed','seed_rejected']:
                    out.mkdir(exist_ok=True)
                    row.update(status='system_error',success=None,process_exit_code=result.returncode,
                               error=row.get('error','worker failed or timed out; terminal outcome unknown'))
                    summary_path.write_text(json.dumps(row,indent=2))
                summarize()
                print(json.dumps(dict(event='end',task=task,condition=condition,seed=seed,
                                      status=row['status'],success=row['success'])),flush=True)
                status=json.loads((OUT/'status.json').read_text())
                status.update(stage=a.stage,last_episode=row,updated_unix=time.time())
                (OUT/'status.json').write_text(json.dumps(status,indent=2))
                if row['status']=='system_error':
                    raise SystemExit('System error: stop batch, retain seed, investigate before further episodes')


if __name__=='__main__': main()
