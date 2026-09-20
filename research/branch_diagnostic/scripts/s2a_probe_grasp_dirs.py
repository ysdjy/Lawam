"""Probe which rim directions give a feasible scripted grasp (fixed before any model comparison)."""
import sys, json
from pathlib import Path
import numpy as np, imageio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_common as sc, scripted as sp
run_dir = Path(sys.argv[1]); out = run_dir / "phase0"; out.mkdir(exist_ok=True, parents=True)
suite = sc.load_suite(); res = {}
for tid in [2]:
    task = suite.get_task(tid); st = suite.get_task_init_states(tid); env, desc = sc.make_env(task, 0); e = sc.raw_env(env)
    bowl_name, plate_name = e.obj_of_interest[0], e.obj_of_interest[1]
    for dname, d in [("+y", (0, 1)), ("-y", (0, -1))]:
        for init_idx in [0, 1, 2, 3]:
            obs = sc.reset_to_init_state(env, st[init_idx]); rec = sp.Recorder(env, desc); rec.log(obs)
            bowl = sc.body_pos(env, bowl_name); hp = sp.hover_point(bowl, np.array(d, float))
            n = sp.move_to(env, hp, False, 80, rec)
            ok, n2 = sp.continuation(env, bowl_name, plate_name, np.array(d, float), rec)
            final_bowl = sc.body_pos(env, bowl_name)
            res[f"task{tid}_init{init_idx}_{dname}"] = {"success": ok, "steps": n + n2, "bowl_final": final_bowl.tolist(), "plate": sc.body_pos(env, plate_name).tolist()}
            print(dname, init_idx, ok, n + n2)
            imageio.mimwrite(out / f"probe_v2_task{tid}_init{init_idx}_{dname}.mp4", rec.frames, fps=20)
    env.close()
sc.write_json(out / "probe_grasp_dirs_v2.json", res)
