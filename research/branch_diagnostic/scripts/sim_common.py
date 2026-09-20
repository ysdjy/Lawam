"""Simulation-side helpers for the branch diagnostic (runs in the `libero310` env).

Everything here reuses the official LIBERO eval semantics:
  * env construction  -> examples.LIBERO.eval_files.libero_benchmark_adapters._build_env
  * obs -> policy example -> examples.LIBERO.eval_files.libero_eval_core._build_policy_example_from_obs
  * action post-processing (unnormalize / gripper binarize+invert) -> ModelClient / libero_eval_core helpers
The only new pieces are: full snapshot/restore, a simple scripted eef controller, geometry/labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_HOME = Path(os.environ.get("LIBERO_HOME", "/home/zbh/LIBERO"))
for p in (str(REPO_ROOT), str(LIBERO_HOME)):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.LIBERO.eval_files.libero_benchmark_adapters import _build_env, quat2axisangle  # noqa: E402
from examples.LIBERO.eval_files.libero_eval_core import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _binarize_gripper_open,
    _build_policy_example_from_obs,
    invert_gripper_action,
)
from examples.LIBERO.eval_files.model2libero_interface import ModelClient  # noqa: E402

NUM_STEPS_WAIT = 10  # identical to EvalArgs.num_steps_wait used by the official eval
CHUNK_LEN = 8        # floor(horizon_sec=0.4 * action_hz=20); verified against checkpoint config at runtime
FUTURE_IDX = 7       # training video delta index of the supervised future frame (see lerobot_datasets._sample_video_delta_indices)


# ----------------------------------------------------------------------------- env
def load_suite(suite_name: str = "libero_spatial"):
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite_name]()


def make_env(task, seed: int):
    env, task_description = _build_env(task, LIBERO_ENV_RESOLUTION, seed)
    return env, task_description


def raw_env(env):
    return env.env  # robosuite / LIBERO BDDL env behind the ControlEnv wrapper


def reset_to_init_state(env, init_state):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


# ----------------------------------------------------------------------------- snapshot / restore
@dataclass
class SimSnapshot:
    sim_state: np.ndarray              # [time, qpos, qvel]
    qacc_warmstart: np.ndarray
    ctrl: np.ndarray
    act: np.ndarray
    timestep: int
    cur_time: float
    done: bool
    goal_pos: np.ndarray
    goal_ori: np.ndarray
    gripper_current_action: np.ndarray
    extra: dict = field(default_factory=dict)

    def to_npz_dict(self) -> dict[str, Any]:
        return {
            "sim_state": self.sim_state,
            "qacc_warmstart": self.qacc_warmstart,
            "ctrl": self.ctrl,
            "act": self.act,
            "timestep": np.int64(self.timestep),
            "cur_time": np.float64(self.cur_time),
            "done": np.bool_(self.done),
            "goal_pos": self.goal_pos,
            "goal_ori": self.goal_ori,
            "gripper_current_action": self.gripper_current_action,
        }

    @classmethod
    def from_npz(cls, d) -> "SimSnapshot":
        return cls(
            sim_state=np.asarray(d["sim_state"]),
            qacc_warmstart=np.asarray(d["qacc_warmstart"]),
            ctrl=np.asarray(d["ctrl"]),
            act=np.asarray(d["act"]),
            timestep=int(d["timestep"]),
            cur_time=float(d["cur_time"]),
            done=bool(d["done"]),
            goal_pos=np.asarray(d["goal_pos"]),
            goal_ori=np.asarray(d["goal_ori"]),
            gripper_current_action=np.asarray(d["gripper_current_action"]),
        )

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.sim_state).tobytes())
        h.update(np.ascontiguousarray(self.goal_ori).tobytes())
        h.update(np.ascontiguousarray(self.gripper_current_action).tobytes())
        return h.hexdigest()[:16]


def take_snapshot(env) -> SimSnapshot:
    e = raw_env(env)
    robot = e.robots[0]
    return SimSnapshot(
        sim_state=np.array(env.get_sim_state(), dtype=np.float64),
        qacc_warmstart=np.array(e.sim.data.qacc_warmstart, dtype=np.float64),
        ctrl=np.array(e.sim.data.ctrl, dtype=np.float64),
        act=np.array(e.sim.data.act, dtype=np.float64),
        timestep=int(e.timestep),
        cur_time=float(e.cur_time),
        done=bool(e.done),
        goal_pos=np.array(robot.controller.goal_pos, dtype=np.float64),
        goal_ori=np.array(robot.controller.goal_ori, dtype=np.float64),
        gripper_current_action=np.array(robot.gripper.current_action, dtype=np.float64),
    )


def restore_snapshot(env, snap: SimSnapshot):
    """Restore the full dynamic state (MuJoCo + controller + env counters) and regenerate obs."""
    e = raw_env(env)
    robot = e.robots[0]
    e.sim.set_state_from_flattened(np.array(snap.sim_state, dtype=np.float64))
    e.sim.data.qacc_warmstart[:] = snap.qacc_warmstart
    e.sim.data.ctrl[:] = snap.ctrl
    if snap.act.size:
        e.sim.data.act[:] = snap.act
    e.sim.forward()
    e.timestep = int(snap.timestep)
    e.cur_time = float(snap.cur_time)
    e.done = bool(snap.done)
    robot.controller.goal_pos = np.array(snap.goal_pos, dtype=np.float64)
    robot.controller.goal_ori = np.array(snap.goal_ori, dtype=np.float64)
    robot.controller.update(force=True)
    robot.gripper.current_action = np.array(snap.gripper_current_action, dtype=np.float64)
    env.check_success()
    env._post_process()
    env._update_observables(force=True)
    return e._get_observations()


# ----------------------------------------------------------------------------- geometry
def body_pos(env, body_name: str) -> np.ndarray:
    e = raw_env(env)
    return np.array(e.sim.data.body_xpos[e.obj_body_id[body_name]], dtype=np.float64)


def eef_pos(env) -> np.ndarray:
    return np.array(raw_env(env)._eef_xpos, dtype=np.float64)


def eef_quat(env) -> np.ndarray:
    return np.array(raw_env(env)._eef_xquat, dtype=np.float64)


def gripper_qpos(env, obs) -> np.ndarray:
    return np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)


def world_to_model_pixels(env, points_xyz: np.ndarray, camera: str = "agentview") -> np.ndarray:
    """Project world points to pixel (row, col) in the *model* image frame (after the [::-1, ::-1] flip)."""
    from robosuite.utils import camera_utils as CU

    e = raw_env(env)
    H = W = LIBERO_ENV_RESOLUTION
    w2c = CU.get_camera_transform_matrix(e.sim, camera, H, W)
    pix = CU.project_points_from_world_to_camera(np.asarray(points_xyz, dtype=np.float64), w2c, H, W)
    # robosuite renders images upside-down w.r.t. the row index used by project_points (they flip when
    # producing obs); the official eval then applies [::-1, ::-1] on the obs image.  Empirically calibrated
    # in s0_restore_test.py (see phase0_checks.json: 'projection_check').
    pix = np.asarray(pix)
    # Empirical calibration (phase0/projection_check*.png): robosuite's projected row already matches the
    # flipped model image; only the column must be mirrored.
    row = pix[..., 0]
    col = W - 1 - pix[..., 1]
    return np.stack([row, col], axis=-1)


# ----------------------------------------------------------------------------- scripted controller
def scripted_step_action(env, target_pos: np.ndarray, gripper_close: bool, gain: float = 12.0, max_mag: float = 0.9) -> np.ndarray:
    """P-control in eef space through the *same* OSC_POSE delta interface the policy uses.

    action[:3] in [-1,1] -> OSC scales by 0.05 m per unit; we saturate at max_mag (within the dataset's q99 ~0.94).
    Rotation deltas are zero (keeps goal orientation).  Gripper: +1 close, -1 open (LIBERO convention).
    """
    err = np.asarray(target_pos, dtype=np.float64) - eef_pos(env)
    a = np.clip(gain * err, -max_mag, max_mag)
    return np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, 1.0 if gripper_close else -1.0], dtype=np.float64)


def obs_record(env, obs) -> dict[str, Any]:
    """Compact per-step record (no images)."""
    e = raw_env(env)
    rec = {
        "eef_pos": eef_pos(env).tolist(),
        "eef_quat": eef_quat(env).tolist(),
        "gripper_qpos": gripper_qpos(env, obs).tolist(),
        "sim_time": float(e.sim.data.time),
        "timestep": int(e.timestep),
    }
    for name in e.obj_of_interest:
        rec[f"obj_{name}"] = body_pos(env, name).tolist()
    return rec


# ----------------------------------------------------------------------------- policy client helpers
def policy_example(obs, task_description: str):
    return _build_policy_example_from_obs(obs, task_description)


def normalized_to_env_action(normalized_chunk: np.ndarray, action_norm_stats: dict) -> np.ndarray:
    """Exact replica of the official client+eval post-processing for a [T,D] normalized chunk."""
    raw = ModelClient.unnormalize_actions(normalized_actions=np.asarray(normalized_chunk, dtype=np.float32), action_norm_stats=action_norm_stats)
    out = []
    for r in raw:
        wv = np.asarray(r[:3], dtype=np.float32)
        rot = np.asarray(r[3:6], dtype=np.float32)
        grip = invert_gripper_action(_binarize_gripper_open(np.asarray(r[-1:], dtype=np.float32)))
        out.append(np.concatenate([wv, rot, grip], axis=0))
    return np.asarray(out, dtype=np.float32)


def load_norm_stats(ckpt_dir: Path) -> dict:
    with open(Path(ckpt_dir) / "dataset_statistics.json") as f:
        stats = json.load(f)
    return stats["franka"]["action"]


# ----------------------------------------------------------------------------- misc
def sha256_of_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def write_json(path: Path, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(str(type(o)))


def append_jsonl(path: Path, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def flipped_agentview(obs) -> np.ndarray:
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def flipped_wrist(obs) -> Optional[np.ndarray]:
    if "robot0_eye_in_hand_image" in obs:
        return np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return None
