"""Shared helpers for the HSU round (simulator side, `libero310` env).

Reuses research/branch_diagnostic/scripts/sim_common.py and research/physical_context_probe/scripts/
pcp_common.py unmodified. New here: the physics level writer for Task B, the scripted oracle teacher,
and the action-space conversions between env actions and the checkpoint's normalized action space.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/zbh/Downloads/IsaacLab/Lawam_paper/LaWAM")
LIBERO_HOME = Path("/home/zbh/LIBERO")
OUT_ROOT = REPO / "results" / "hidden_state_utilization_probe"
CKPT_DIR = REPO / "results/Checkpoints/libero/lawam_libero_sft_release"
CKPT = CKPT_DIR / "final_model" / "pytorch_model.pt"
sys.path.insert(0, str(REPO / "research" / "physical_context_probe" / "scripts"))
sys.path.insert(0, str(REPO / "research" / "branch_diagnostic" / "scripts"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_HOME / "libero"))
for _k, _v in (("MUJOCO_GL", "egl"), ("PYOPENGL_PLATFORM", "egl"), ("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")):
    os.environ.setdefault(_k, _v)

import sim_common as sc  # noqa: E402
import pcp_common as pcp  # noqa: E402

SUITE, TASK_ID, SEED = "libero_goal", 5, 7
LEVELS = {"NOMINAL": 1.0, "HIGH": 2.0}
MU_NOMINAL = 0.95
MAX_STEPS = 500          # teacher + rollout budget: at HIGH friction the same stroke simply takes longer
CHUNK = 8
ACTION_LIMIT = 0.9375            # dataset action max/min on the translation dims


def context_value(mu_eff: float) -> float:
    """c = log(mu_eff / 0.95), the only context this round allows."""
    return float(math.log(float(mu_eff) / MU_NOMINAL))


def load_norm_stats() -> dict:
    return json.loads((CKPT_DIR / "dataset_statistics.json").read_text())["franka"]["action"]


def env_action_to_normalized(env_actions: np.ndarray, stats: dict, action_dim: int = 32) -> np.ndarray:
    """Inverse of the official eval post-processing (ModelClient.unnormalize_actions + gripper).

    env action = [world_vector(3), rotation_delta(3), gripper(+1 close / -1 open)].
    The gripper column has mask=0 in the statistics, so its RAW value IS the normalized value, and
    the eval path binarises it: raw > 0.5 -> env +1 (close).  Hence normalized gripper = 1.0 for a
    closing command and 0.0 for an opening one.
    """
    hi = np.asarray(stats["max"], np.float32)
    lo = np.asarray(stats["min"], np.float32)
    a = np.asarray(env_actions, np.float32).reshape(-1, 7)
    out = np.zeros((a.shape[0], action_dim), np.float32)
    out[:, :6] = np.clip(2.0 * (a[:, :6] - lo[:6]) / (hi[:6] - lo[:6]) - 1.0, -1.0, 1.0)
    out[:, 6] = np.where(a[:, 6] > 0, 1.0, 0.0)
    return out


def normalized_to_env_action(norm: np.ndarray, stats: dict) -> np.ndarray:
    """Forward direction, identical to what the LIBERO client does with a served chunk."""
    from examples.LIBERO.eval_files.libero_eval_core import _binarize_gripper_open, invert_gripper_action
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    raw = ModelClient.unnormalize_actions(normalized_actions=np.asarray(norm, np.float32),
                                          action_norm_stats=stats)
    return np.stack([np.concatenate([
        np.asarray(r[:3], np.float32), np.asarray(r[3:6], np.float32),
        invert_gripper_action(_binarize_gripper_open(np.asarray(r[-1:], np.float32)))]) for r in raw])


def noise(tag: str, q: int) -> np.ndarray:
    import torch
    base = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")
    g = torch.Generator(device="cpu").manual_seed(int((base + q) % (2 ** 31)))
    return torch.randn((1, 50, 32), generator=g, dtype=torch.float32).numpy()


# ----------------------------------------------------------------------------- environment
def make_task_env():
    suite = sc.load_suite(SUITE)
    task = suite.get_task(TASK_ID)
    env, desc = sc.make_env(task, SEED)
    return suite, task, env, desc


def reset_with_level(env, init_state, level: str):
    """Reset, then write the physics level. The Target handle is rebuilt AFTER the reset because
    LIBERO uses hard_reset=True and env.reset() reloads the MuJoCo model."""
    obs = sc.reset_to_init_state(env, init_state)
    tgt = pcp.Target("B", env)
    tgt.write_level("NOMINAL" if level == "NOMINAL" else "HIGH")
    _, contacts = tgt.contacts()
    mu_eff = max([c["friction"][0] for c in contacts], default=float("nan"))
    env._update_observables(force=True)
    return sc.raw_env(env)._get_observations(), tgt, mu_eff


# ----------------------------------------------------------------------------- oracle teacher
class HookDragTeacher:
    """Scripted oracle pusher for `push the plate to the front of the stove`.

    Geometry, frozen after ONE engineering sweep (AUDIT_HSU.md HSU AUDIT 1): the space BEHIND the
    plate is occupied by the wooden cabinet -- the robot's forearm collides with
    `wooden_cabinet_1_base` before the gripper ever reaches a push-from-behind pose -- so the only
    reachable strategy is the one the trained policy also uses: bring the OPEN gripper down just
    past the plate centre (z = 0.908, 4 cm along the goal direction), which puts the fingers behind
    the far rim, then drag along the goal direction. A stroke is repeated if the plate stalls.

    It is an oracle supervision generator, NOT a method and NOT a contribution. It reads the plate
    pose, the goal site and the physics level from the simulator; it never teleports, never writes
    object state, never applies a force to the plate and never replays a future rollout. Every
    action is an ordinary 7-dim LaWAM EEF action inside the dataset action range.
    """

    HOVER_Z = 1.00
    PUSH_Z = 0.908
    HOOK = 0.04           # m past the plate centre along the goal direction
    STEP = 0.03           # m of lead for the drag target
    GAIN = 0.7 / 0.05     # error (m) -> action units
    MAG = 0.35
    TOL = 0.006
    APPROACH_STEPS = 60
    DESCEND_STEPS = 40
    DRAG_STEPS = 200
    MAX_STROKES = 3
    STALL_STEPS = 25      # plate moved < STALL_EPS over this many steps -> re-stroke
    STALL_EPS = 0.002
    GRIPPER = -1.0        # open

    def __init__(self, env, tgt):
        self.env, self.tgt = env, tgt

    def _cmd(self, target):
        err = np.asarray(target) - sc.eef_pos(self.env)
        a = np.clip(err * self.GAIN, -self.MAG, self.MAG)
        return np.clip(np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, self.GRIPPER]), -ACTION_LIMIT, ACTION_LIMIT)

    def _goal_dir(self):
        p = self.tgt.plate_pose()[0][:2]
        v = self.tgt.goal_xy() - p
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-9), n

    def actions(self, max_steps: int = MAX_STEPS):
        """Yield one action at a time; the caller steps the env. Ends on success or the budget."""
        for _ in range(self.MAX_STROKES):
            plate = self.tgt.plate_pose()[0]
            d, _ = self._goal_dir()
            hook_xy = plate[:2] + d * self.HOOK
            for target, n in ((np.array([*hook_xy, self.HOVER_Z]), self.APPROACH_STEPS),
                              (np.array([*hook_xy, self.PUSH_Z]), self.DESCEND_STEPS)):
                for _ in range(n):
                    if np.linalg.norm(target - sc.eef_pos(self.env)) < self.TOL:
                        break
                    yield self._cmd(target)
            last, stalled = self.tgt.plate_pose()[0][:2].copy(), 0
            for _ in range(self.DRAG_STEPS):
                plate = self.tgt.plate_pose()[0]
                d, dist = self._goal_dir()
                if dist < 0.02:
                    break
                moved = float(np.linalg.norm(plate[:2] - last))
                stalled = stalled + 1 if moved < self.STALL_EPS else 0
                if moved >= self.STALL_EPS:
                    last = plate[:2].copy()
                if stalled > self.STALL_STEPS:
                    break                      # slipped off: lift and re-hook
                yield self._cmd(np.array([*(sc.eef_pos(self.env)[:2] + d * self.STEP), self.PUSH_Z]))


def run_teacher_episode(env, tgt, max_steps: int = MAX_STEPS, record: bool = False):
    """Run the teacher to success or the step budget. Returns a record dict."""
    teacher = HookDragTeacher(env, tgt)
    actions, obs_log, success, step = [], [], False, 0
    p0 = tgt.plate_pose()[0][:2].copy()
    d0 = float(np.linalg.norm(p0 - tgt.goal_xy()))
    contact_steps = 0
    obs = sc.raw_env(env)._get_observations()
    for a in teacher.actions(max_steps):
        if step >= max_steps:
            break
        if record and step % CHUNK == 0:
            obs_log.append({"step": step, "obs": obs})
        actions.append(np.asarray(a, np.float64).tolist())
        obs, _, _, _ = env.step(np.asarray(a, np.float64).tolist())
        hits, _ = tgt.contacts()
        contact_steps += int(hits > 0)
        step += 1
        if env.check_success():
            success = True
            break
    p1 = tgt.plate_pose()[0][:2]
    d1 = float(np.linalg.norm(p1 - tgt.goal_xy()))
    return {"success": bool(success), "steps": step, "actions": actions, "obs_log": obs_log,
            "plate_start_xy": p0.tolist(), "plate_end_xy": p1.tolist(),
            "dist_start": d0, "dist_end": d1, "progress_m": d0 - d1,
            "contact_steps": contact_steps,
            "max_abs_action": float(np.abs(np.asarray(actions)).max()) if actions else 0.0}
