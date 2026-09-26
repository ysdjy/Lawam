"""Descriptive physical progress from official-policy action-end observations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[2]/'results/robotwin_phase2_newhost'
fig,axes=plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
for seed,color in [(100102,'#167D8D'),(100103,'#BA4938')]:
    folder=root/'episodes'/f'open_microwave__demo_clean__{seed}'
    report=json.loads((folder/'trace_analysis.json').read_text())
    progress=report['progress'];times=[r['sim_seconds'] for r in progress]
    qpos=[r['articulation_qpos']['microwave'][0] for r in progress]
    upper=report['final_state']['objects']['microwave']['qlimits'][0][1]
    axes[0].plot(times,np.array(qpos)/upper,color=color,label=f'{seed}: '+('success' if report['success'] else 'failure'))
axes[0].axhline(.6,color='black',ls='--',lw=1,label='native threshold')
axes[0].set(title='Microwave: achieved hinge progress',ylabel='qpos / upper joint limit')
axes[0].legend(fontsize=8)
for condition,color in [('demo_clean','#167D8D'),('demo_randomized','#BA4938')]:
    report=json.loads((root/'episodes'/f'stack_blocks_three__{condition}__100100'/'trace_analysis.json').read_text())
    progress=report['progress'];times=[r['sim_seconds'] for r in progress]
    delta=np.array([r['predicate_details']['layer2_delta'] for r in progress])
    label=condition.replace('demo_','')+': '+('success' if report['success'] else 'failure')
    axes[1].plot(times,np.max(np.abs(delta[:,:2]),axis=1)*100,color=color,label=label)
    axes[2].plot(times,delta[:,2]*100,color=color,label=label)
axes[1].axhline(2.5,color='black',ls='--',lw=1)
axes[1].set(title='Blocks: second-layer XY offset',ylabel='max absolute XY offset (cm)')
axes[2].axhspan(3.8,6.2,color='grey',alpha=.2,label='native height tolerance')
axes[2].axhline(5,color='black',ls='--',lw=1)
axes[2].set(title='Blocks: second-layer height',ylabel='block2.z - block1.z (cm)')
for axis in axes:
    axis.set_xlabel('Actual simulated time (s)');axis.grid(alpha=.2)
axes[1].legend(fontsize=8);axes[2].legend(fontsize=8)
fig.suptitle('Official checkpoint, local screening. Descriptive examples; not a causal comparison.',fontsize=11)
out=root/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'physical_failure_diagnostics.png',dpi=180)
fig.savefig(out/'physical_failure_diagnostics.pdf')
print(out/'physical_failure_diagnostics.png')
