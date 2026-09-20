"""Model-side helpers (runs in the `lawam` env). Loads the policy exactly like deployment/model_server/server_policy.py."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # relative paths in config.yaml (qwen weights, LAM ckpt) are repo-relative

from deployment.model_server.server_policy import load_policy_from_checkpoint  # noqa: E402
from examples.LIBERO.eval_files.model2libero_interface import ModelClient  # noqa: E402
from starVLA.model.tools import read_mode_config  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
EMBODIMENT_ID = 25   # ModelClient default used by the official LIBERO eval
ACTION_HZ = 20.0     # ModelClient default used by the official LIBERO eval


def load_policy(ckpt_path: Path = DEFAULT_CKPT, use_bf16: bool = True):
    """Same as server_policy.main: from_pretrained -> bf16 -> cuda -> eval."""
    vla = load_policy_from_checkpoint(str(ckpt_path), use_bf16=use_bf16, device="cuda")
    model_config, norm_stats = read_mode_config(ckpt_path)
    return vla, model_config, norm_stats


def make_example(primary_img: np.ndarray, wrist_img: np.ndarray | None, state7: np.ndarray, lang: str, norm_stats: dict) -> dict[str, Any]:
    """Replicates ModelClient._prepare_example (state normalization, keys, dtypes)."""
    state_arr = np.asarray(state7, dtype=np.float32)
    if state_arr.ndim == 1:
        state_arr = state_arr[None, :]
    state_arr = ModelClient.normalize_state(state=state_arr, state_norm_stats=norm_stats["franka"]["state"])
    ex = {
        "primary_image": [np.asarray(primary_img)],
        "lang": str(lang),
        "state": state_arr,
        "embodiment_id": EMBODIMENT_ID,
        "action_hz": ACTION_HZ,
    }
    if wrist_img is not None:
        ex["wrist_image"] = [np.asarray(wrist_img)]
    return ex


def seed_all(seed: int):
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


def make_initial_noise(seed: int, action_horizon: int, action_dim: int, batch: int = 1) -> np.ndarray:
    """Deterministic CPU noise tensor N(0, I) with its own generator (independent of global RNG)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((batch, action_horizon, action_dim), generator=g, dtype=torch.float32).numpy()


def sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def imagenet_video_tensor(frames_uint8_hwc: list[np.ndarray], device="cuda") -> torch.Tensor:
    """[T,H,W,3] uint8 (already flipped + 256x256) -> [1,T,3,H,W] float imagenet-normalized, identical to
    batch_builder.prepare_frame_spatial_uint8 + imagenet_normalize (no crop since images are already 256x256 square)."""
    from starVLA.model.framework.latent_world.batch_utils import imagenet_normalize_video_, prepare_frame_spatial_uint8

    frames = [prepare_frame_spatial_uint8(f, target_hw=(256, 256), apply_center_crop_90=False) for f in frames_uint8_hwc]
    vid = torch.stack(frames, dim=0).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
    imagenet_normalize_video_(vid)
    return vid.to(device)


def write_json(path: Path, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _d(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(str(type(o)))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_d)
