"""Offline ROI helper (libero310): project eef and bowl positions at the reference t7 frame to model-image pixels.
Used only for evaluation ROI masks; never given to the policy."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_common as sc  # noqa: E402

run_dir = Path(sys.argv[1])
manifest = [m for m in (json.loads(l) for l in open(run_dir / "reference_manifest.jsonl")) if m.get("qualified")]
suite = sc.load_suite(); task = suite.get_task(2); init_states = suite.get_task_init_states(2)
env, desc = sc.make_env(task, 0); env.reset(); env.set_init_state(init_states[0])
out = {}
for st in manifest:
    snap = sc.SimSnapshot.from_npz(np.load(st["files"]["snapshot"]))
    sc.restore_snapshot(env, snap)  # camera is static; projection only needs the camera matrix
    out[st["state_id"]] = {}
    for b in "AB":
        ref = np.load(st["files"][f"branch_{b}"])
        pts = np.stack([ref["eef"][7], ref["bowl"][7]])
        pix = sc.world_to_model_pixels(env, pts)
        out[st["state_id"]][b] = {"pixels": pix.tolist(), "names": ["eef_t7", "bowl_t7"]}
sc.write_json(run_dir / "roi_pixels.json", out)
print("wrote", run_dir / "roi_pixels.json", len(out))
env.close()
