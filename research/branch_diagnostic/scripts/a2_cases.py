"""Step 6c: representative cases (libero310 env). Selection rule (fixed): for the confirmation phase, rank states by
B2 executed-switch score (mean s_e given input A minus mean s_e given input B); take the LOWEST, MEDIAN and HIGHEST
states plus the first pilot state. For each: an image panel (o_0 | ref A o_7 | ref B o_7 | B0 | B1-A | B1-B | B2-A | B2-B at o_7)
and a side-by-side video (R-A, R-B, B1-A, B1-B, B2-A, B2-B chunk frames, seed 101, 4 fps, labelled)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

run_dir = Path(sys.argv[1]); phase = sys.argv[2] if len(sys.argv) > 2 else "confirm"
rows = list(csv.DictReader(open(run_dir / f"analysis_{phase}" / "metrics_by_state.csv")))
rows = [r for r in rows if r["B2_exec_switch_score"] not in ("", "None")]
rows.sort(key=lambda r: float(r["B2_exec_switch_score"]))
sel = {"lowest_B2_switch": rows[0]["state_id"], "median_B2_switch": rows[len(rows) // 2]["state_id"], "highest_B2_switch": rows[-1]["state_id"], "first_pilot": "t02_ep000"}
out_dir = run_dir / "figures" / "cases"; out_dir.mkdir(parents=True, exist_ok=True)
vid_dir = run_dir / "videos" / "cases"; vid_dir.mkdir(parents=True, exist_ok=True)


def label(img, text):
    im = Image.fromarray(np.asarray(img)); d = ImageDraw.Draw(im); d.rectangle([0, 0, 256, 14], fill=(0, 0, 0)); d.text((3, 1), text, fill=(255, 255, 0)); return np.asarray(im)


meta = {}
for why, sid in sel.items():
    ph = "pilot" if sid in ("t02_ep000", "t02_ep001", "t02_ep002", "t02_ep003", "t02_ep004") else "confirm"
    ex_dir = run_dir / "executions" / sid
    o0 = np.load(run_dir / "reference" / sid / "o0.npz")["primary"]
    refA = np.load(run_dir / "reference" / sid / "branch_A_chunk.npz"); refB = np.load(run_dir / "reference" / sid / "branch_B_chunk.npz")
    tags = ["B0_s101", "B1_A_s101", "B1_B_s101", "B2_A_s101", "B2_B_s101"]
    execs = {t: np.load(ex_dir / f"{t}.npz") for t in tags}
    recs = {Path(json.loads(l)["files"]["execution"]).stem: json.loads(l) for l in open(run_dir / "episode_records.jsonl") if json.loads(l)["state_id"] == sid}
    panel = [label(o0, f"{sid} o_0 (t)"), label(refA["frames"][7], "ref A  o_7"), label(refB["frames"][7], "ref B  o_7")]
    for t in tags:
        r = recs[t]; panel.append(label(execs[t]["frames"][7], f"{t.replace('_s101','')} o_7 exec={r['executed_branch']}"))
    imageio.imwrite(out_dir / f"{why}_{sid}_panel.png", np.concatenate(panel, axis=1))
    # side-by-side video over the chunk (frames 0..8)
    seqs = [("R-A", refA["frames"]), ("R-B", refB["frames"]), ("B1-A", execs["B1_A_s101"]["frames"]), ("B1-B", execs["B1_B_s101"]["frames"]), ("B2-A", execs["B2_A_s101"]["frames"]), ("B2-B", execs["B2_B_s101"]["frames"])]
    frames = []
    for k in range(9):
        row = [label(s[min(k, len(s) - 1)], f"{n} t+{k} ({k*0.05:.2f}s)") for n, s in seqs]
        frames.append(np.concatenate([np.concatenate(row[:3], axis=1), np.concatenate(row[3:], axis=1)], axis=0))
    imageio.mimwrite(vid_dir / f"{why}_{sid}_sidebyside.mp4", frames, fps=4)
    meta[why] = {"state_id": sid, "phase": ph, "panel": str(out_dir / f"{why}_{sid}_panel.png"), "video": str(vid_dir / f"{why}_{sid}_sidebyside.mp4"),
                 "B2_exec_switch_score": next((r["B2_exec_switch_score"] for r in rows if r["state_id"] == sid), None),
                 "executed": {t: recs[t]["executed_branch"] for t in tags}, "success": {t: recs[t]["full_task_success"] for t in tags}}
json.dump({"selection_rule": __doc__, "cases": meta}, open(out_dir / "cases.json", "w"), indent=2)
print(json.dumps(meta, indent=1))
