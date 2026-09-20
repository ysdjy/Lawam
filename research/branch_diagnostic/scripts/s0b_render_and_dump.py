"""Quantify render nondeterminism at identical states; re-check projection; dump obs for model tests."""
import sys, json
from pathlib import Path
import numpy as np, imageio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_common as sc

run_dir = Path(sys.argv[1]); out = run_dir / "phase0"
d = np.load(out / "restore_test_snapshot.npz")
snap = sc.SimSnapshot.from_npz(d); X = d["X"]; img_s_online = d["img_s"]
suite = sc.load_suite(); task = suite.get_task(2); init_states = suite.get_task_init_states(2)
env, desc = sc.make_env(task, 0); env.reset(); env.set_init_state(init_states[0]); e = sc.raw_env(env)

def stats(a, b):
    diff = np.abs(a.astype(int) - b.astype(int))
    return {"max": int(diff.max()), "mean": float(diff.mean()), "frac_pixels_gt10": float((diff.max(-1) > 10).mean())}

res = {}
# render twice without stepping
obs1 = sc.restore_snapshot(env, snap); im1 = sc.flipped_agentview(obs1)
obs2 = e._get_observations(force_update=True) if "force_update" in e._get_observations.__code__.co_varnames else e._get_observations()
im2 = sc.flipped_agentview(obs2)
res["same_state_render_twice"] = stats(im1, im2)
res["restored_vs_online_snapshot_image"] = stats(im1, img_s_online)
# execute X twice from restore, compare frame-by-frame
seqs = []
for k in range(2):
    sc.restore_snapshot(env, snap); frames = []
    for a in X:
        obs, _, _, _ = env.step(list(map(float, a))); frames.append(sc.flipped_agentview(obs).copy())
    seqs.append(np.asarray(frames))
res["replay_twice_frames"] = stats(seqs[0], seqs[1])
res["replay_twice_frames_per_frame_max"] = [int(np.abs(seqs[0][i].astype(int) - seqs[1][i].astype(int)).max()) for i in range(len(X))]
# projection re-check
sc.restore_snapshot(env, snap)
names = list(e.obj_of_interest)
pts = np.stack([sc.eef_pos(env)] + [sc.body_pos(env, n) for n in names])
pix = sc.world_to_model_pixels(env, pts)
img = sc.flipped_agentview(e._get_observations()).copy()
for (r, c), col in zip(pix, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
    r, c = int(r), int(c); img[max(0, r-3):r+4, max(0, c-3):c+4] = col
imageio.imwrite(out / "projection_check_v2.png", img)
res["projection_v2_rowcol"] = {"eef": pix[0].tolist(), **{n: pix[i+1].tolist() for i, n in enumerate(names)}}
# dump observations for the model-side phase-0 tests: snapshot obs (online image) + obs after 7 steps of X
sc.restore_snapshot(env, snap)
recs = []
obs = e._get_observations()
def dump(obs, tag):
    ex, img, wrist = sc.policy_example(obs, desc)
    recs.append({"tag": tag, "primary": img, "wrist": wrist, "state": ex["state"], "lang": desc})
dump(obs, "snapshot_restored")
for i, a in enumerate(X[:8]):
    obs, _, _, _ = env.step(list(map(float, a)))
    if i in (6, 7): dump(obs, f"after_{i+1}_steps")
np.savez_compressed(out / "phase0_obs.npz", tags=np.array([r["tag"] for r in recs]), primary=np.stack([r["primary"] for r in recs]),
                    wrist=np.stack([r["wrist"] for r in recs]), state=np.stack([r["state"] for r in recs]), lang=np.array([desc]))
sc.write_json(out / "render_check.json", res); print(json.dumps(res, indent=1)); env.close()
