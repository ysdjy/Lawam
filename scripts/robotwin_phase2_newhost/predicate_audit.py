"""Execute unchanged native predicates on counterfactual states, not rollouts."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

root=Path(__file__).resolve().parents[2]
robotwin=root.parent/'RoboTwin_poc'
def predicate(task):
    source=(robotwin/'envs'/f'{task}.py').read_text()
    cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef))
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='check_success')
    namespace={'np':np}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(robotwin/'envs'/f'{task}.py'),'exec'),namespace)
    return namespace['check_success']

def actor(position):
    return NS(get_pose=lambda:NS(p=np.asarray(position)))

bowls=NS(bowl1=actor([0,0,.04]),bowl2=actor([0,0,.07]),bowl3=actor([0,0,.11]),
         table_z_bias=0,is_left_gripper_open=lambda:True,is_right_gripper_open=lambda:True)
seal=NS(seal=actor([0,0,1.2]),target=actor([0,0,.741]),
        robot=NS(is_left_gripper_open=lambda:True,is_right_gripper_open=lambda:True))
mug=NS(mug=NS(get_functional_point=lambda i:np.array([0,0,1.5,1,0,0,0])),
       rack=NS(get_pose=lambda:NS(p=np.array([0,0,.8])),
               get_functional_point=lambda i:np.array([0,0,.9,1,0,0,0])),
       is_right_gripper_open=lambda:True)
rows=[]
for task,state,description in [
    ('stack_bowls_three',bowls,'Aligned bowl origins below tabletop; signed height errors pass without lower bound.'),
    ('stamp_seal',seal,'Seal 45.9 cm above target, XY equal and both gripper commands open; no contact requirement.'),
    ('hanging_mug',mug,'Mug functional point 60 cm above rack functional point; XY aligned, right command open.')]:
    value=bool(predicate(task)(state))
    assert value
    rows.append(dict(task=task,predicate_result=value,case=description,
                     evidence_type='counterfactual predicate unit check; not a physical rollout or success-rate observation'))
(root/'results/robotwin_phase2_newhost/evidence/predicate_counterexamples.json').write_text(json.dumps(rows,indent=2))
print('PASS: three unchanged native predicates admit counterfactual completion states; no simulator/task modified')
