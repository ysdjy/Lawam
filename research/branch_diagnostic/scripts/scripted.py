"""Scripted reference controller (libero310). Uses only the normal OSC_POSE delta interface via env.step."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_common as sc  # noqa: E402

RIM_RADIUS = 0.055          # akita bowl: 11.1 cm diameter (measured AABB, phase0)
RIM_TOP_DZ = 0.0526         # rim top z - bowl body z (0.951 - 0.8984)
HOVER_DZ = 0.035            # hover height above rim top for the pre-grasp point
GRASP_DEPTH = 0.020         # eef target below rim top when grasping
CARRY_DZ = 0.14             # carry height above rim top
PLATE_TOP_DZ = 0.0355       # plate top z - plate body z (0.938 - 0.9025)


class Recorder:
    def __init__(self, env, desc):
        self.env, self.desc = env, desc
        self.frames, self.wrist, self.states, self.records, self.actions = [], [], [], [], []

    def log(self, obs, action=None):
        ex, img, wrist = sc.policy_example(obs, self.desc)
        self.frames.append(img.copy())
        self.wrist.append(None if wrist is None else wrist.copy())
        self.states.append(ex["state"][0].copy())
        self.records.append(sc.obs_record(self.env, obs))
        if action is not None:
            self.actions.append(np.asarray(action, dtype=np.float64).copy())


def step_env(env, action, rec: Recorder | None):
    obs, _, done, _ = env.step([float(x) for x in action])
    if rec is not None:
        rec.log(obs, action)
    return obs, done


def hover_point(bowl_pos: np.ndarray, direction: np.ndarray) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-9)
    return np.array([bowl_pos[0] + RIM_RADIUS * d[0], bowl_pos[1] + RIM_RADIUS * d[1], bowl_pos[2] + RIM_TOP_DZ + HOVER_DZ])


def move_to(env, target, gripper_close, max_steps, rec, tol=0.006, gain=12.0, max_mag=0.9):
    n = 0
    for _ in range(max_steps):
        a = sc.scripted_step_action(env, target, gripper_close, gain=gain, max_mag=max_mag)
        obs, done = step_env(env, a, rec)
        n += 1
        if np.linalg.norm(sc.eef_pos(env) - target) < tol:
            break
    return n


def hold(env, gripper_close, steps, rec):
    for _ in range(steps):
        a = np.array([0, 0, 0, 0, 0, 0, 1.0 if gripper_close else -1.0])
        step_env(env, a, rec)


def branch_chunk_actions(env, target, rec, n_steps=sc.CHUNK_LEN, gain=12.0, max_mag=0.9):
    """The 8-action reference chunk for a branch: saturating P-control toward the branch hover point."""
    acts = []
    for _ in range(n_steps):
        a = sc.scripted_step_action(env, target, False, gain=gain, max_mag=max_mag)
        acts.append(a.copy())
        step_env(env, a, rec)
    return np.asarray(acts)


def continuation(env, bowl_name, plate_name, direction, rec, max_total=260):
    """After the chunk: finish approach, grasp the rim, carry to the plate, release. Returns (success, steps)."""
    n = 0
    bowl = sc.body_pos(env, bowl_name)
    hp = hover_point(bowl, direction)
    n += move_to(env, hp, False, 40, rec)
    grasp = hp.copy(); grasp[2] = bowl[2] + RIM_TOP_DZ - GRASP_DEPTH
    n += move_to(env, grasp, False, 40, rec, tol=0.005, max_mag=0.5)
    hold(env, True, 10, rec); n += 10
    carry = grasp.copy(); carry[2] = bowl[2] + RIM_TOP_DZ + CARRY_DZ
    n += move_to(env, carry, True, 40, rec, max_mag=0.6)
    plate = sc.body_pos(env, plate_name)
    # eef grasps the rim at bowl_center + r*dir, so bowl_center == eef - r*dir: put eef at plate + r*dir
    above = np.array([plate[0] + RIM_RADIUS * direction[0], plate[1] + RIM_RADIUS * direction[1], carry[2]])
    n += move_to(env, above, True, 80, rec, tol=0.008, max_mag=0.6)
    place = above.copy(); place[2] = plate[2] + PLATE_TOP_DZ + 0.06
    n += move_to(env, place, True, 40, rec, tol=0.01, max_mag=0.5)
    hold(env, False, 10, rec); n += 10
    up = place.copy(); up[2] += 0.08
    n += move_to(env, up, False, 20, rec, tol=0.01)
    hold(env, False, 5, rec); n += 5
    return bool(env.check_success()), n
