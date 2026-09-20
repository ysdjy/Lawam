"""Shared helpers for the Future Exposure (FE) round.

Loads the released LIBERO SFT LaWAM, builds the OFFICIAL LeRobot dataset + latent-world train
collator, and restricts it to an explicit episode-id subset using only the dataset's own
public rebuild hook (`_build_active_step_indexing`). No repository file is modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
FE_ROOT = REPO_ROOT / "research" / "action_critical_reliability_probe" / "future_exposure_test"
RUN_DIR = REPO_ROOT / "results" / "action_critical_reliability_probe" / "acr_future_exposure_20260920_231423"
CKPT_DIR = REPO_ROOT / "results" / "Checkpoints" / "libero" / "lawam_libero_sft_release"
CKPT_FILE = CKPT_DIR / "final_model" / "pytorch_model.pt"
TRAIN_CFG = REPO_ROOT / "starVLA" / "config" / "training" / "train_libero.yaml"
SPLIT_MANIFEST = RUN_DIR / "SPLIT_MANIFEST.json"

# Frozen anchors (verified in every prior round of this research line).
LAWAM_STATE_DICT_SHA256 = "37a53b8c799c39725a18900eeaa687d1d2cebc24fca28d8e1f2881ca2870b523"
SPLIT_MANIFEST_SHA256 = "3483b2936e39352d5fdf973c4c18b04be3a3aa2ea017114038b4de7fb5097df3"

for _p in (str(REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO_ROOT)  # config.yaml carries repo-relative paths (qwen weights, LAM ckpt)


def enable_determinism() -> dict:
    """Amendment FE-A5. Without this, two identically-configured training runs differ by ~1.4e-8
    from CUDA reduction order in the backward pass. With it they are bitwise identical, so the
    future-condition mask is provably the ONLY difference between the two arms. Measured cost:
    none worth reporting (0.266 s/update either way)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return {
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": False,
    }


# ----------------------------------------------------------------------------- small utilities
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def seed_from(*parts: str) -> int:
    """Deterministic 63-bit seed; never uses python's per-process randomized hash()."""
    return int(sha256_text(*parts)[:16], 16) & ((1 << 63) - 1)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Size):
        return list(o)
    raise TypeError(str(type(o)))


def write_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def append_jsonl(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def legacy_state_dict_hash(module) -> tuple[str, int]:
    """The anchor procedure used since round 1 (`g2_audit0.state_dict_hash`), reproduced verbatim
    so the digest is comparable across all six rounds.

    It differs from `module_state_hash` below in three ways that all change the digest: it hashes
    the LOADED runtime backend (which round 1 loaded with use_bf16=True, the deployment cast), it
    casts every tensor to float32 before taking bytes, and it includes no dtype or shape metadata.
    Hashing the raw checkpoint file with dtype+shape metadata gives a different digest for the
    same weights -- that is a procedure difference, not a modified checkpoint.
    """
    import numpy as _np

    h = hashlib.sha256()
    n = 0
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        t = v.detach()
        h.update(_np.ascontiguousarray(t.float().cpu().numpy()).tobytes())
        n += t.numel()
    return h.hexdigest(), n


def module_state_hash(module) -> str:
    """Hash of a module's full state_dict (or of a bare Parameter): key, dtype, shape, raw bytes.

    `act_query` / `flow_action_query` are bare nn.Parameters on the backend, not submodules.
    """
    h = hashlib.sha256()
    sd = {"": module} if torch.is_tensor(module) else module.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k]
        h.update(k.encode()); h.update(b"\0")
        h.update(str(t.dtype).encode()); h.update(b"\0")
        h.update(str(tuple(t.shape)).encode()); h.update(b"\0")
        h.update(t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- model
def load_backend(use_bf16: bool = True):
    """Return (vla, backend). Identical loading path to every previous round."""
    from deployment.model_server.server_policy import load_policy_from_checkpoint

    vla = load_policy_from_checkpoint(str(CKPT_FILE), use_bf16=use_bf16, device="cuda")
    return vla, vla.policy_backend


FROZEN_GROUPS = ("vlm", "lam", "vlm_to_lam", "act_query", "flow_action_query")


def freeze_all_but_flow(backend) -> dict:
    """Set requires_grad=False everywhere except policy_backend.flow; return a count report."""
    for p in backend.parameters():
        p.requires_grad_(False)
    for p in backend.flow.parameters():
        p.requires_grad_(True)
    trainable = sum(p.numel() for p in backend.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in backend.parameters() if not p.requires_grad)
    flow_n = sum(p.numel() for p in backend.flow.parameters())
    named_flow = {id(p) for p in backend.flow.parameters()}
    leaked = [n for n, p in backend.named_parameters() if p.requires_grad and id(p) not in named_flow]
    return {
        "trainable_params": trainable,
        "frozen_params": frozen,
        "flow_params": flow_n,
        "trainable_equals_flow": trainable == flow_n,
        "non_flow_trainable_names": leaked,
    }


def set_train_modes(backend) -> None:
    """Frozen modules in eval(); the Flow Head in train(), exactly as the official trainer would."""
    backend.eval()
    backend.flow.train()


# ----------------------------------------------------------------------------- config
def load_train_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.load(str(TRAIN_CFG))


# ----------------------------------------------------------------------------- dataset
def load_split_manifest() -> dict:
    got = sha256_file(SPLIT_MANIFEST)
    if got != SPLIT_MANIFEST_SHA256:
        raise RuntimeError(f"SPLIT_MANIFEST.json sha256 mismatch: {got} != {SPLIT_MANIFEST_SHA256}")
    return json.loads(SPLIT_MANIFEST.read_text())


def build_dataset(cfg, dataset_statistics_override: dict | None = None):
    """The official LeRobot dataset, mode='all' (episode subsetting is applied afterwards)."""
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    return get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="all",
        balance_dataset_weights=bool(cfg.datasets.vla_data.get("balance_dataset_weights", True)),
        framework_name=cfg.framework.name,
        dataset_statistics_override=dataset_statistics_override,
    )


def build_collator(cfg, training: bool = True):
    from starVLA.dataloader import _build_latent_world_collator
    from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder

    policy_cfg = LatentWorldPolicyConfigBuilder(cfg).build()
    return _build_latent_world_collator(cfg, policy_cfg=policy_cfg, training=training)


def _inner_datasets(dataset) -> list:
    """Return the list of leaf LeRobot datasets under a (possibly mixture) wrapper."""
    for attr in ("datasets", "_datasets", "sub_datasets"):
        inner = getattr(dataset, attr, None)
        if isinstance(inner, (list, tuple)) and inner:
            return list(inner)
    return [dataset]


def restrict_to_episodes(dataset, episode_indices: Iterable[int]) -> dict:
    """Restrict every leaf dataset to the given GLOBAL episode ids.

    Mirrors the five assignments the official `_build_mode_split_from_trajectories()` makes and
    then calls the dataset's OWN `_build_active_step_indexing()`, which recomputes
    _subset_from_indices / _subset_lengths / _subset_cum_lengths / _subset_total_steps purely
    from _active_episode_indices.

    The stock build_dataloaders(train_split_all=true) would select all 1693 episodes; this is the
    explicit subsetting the FE protocol requires. No repository file is modified.
    """
    wanted = set(int(e) for e in episode_indices)
    report = {"requested_episodes": len(wanted), "leaves": []}
    for ds in _inner_datasets(dataset):
        all_ids = np.asarray(ds._all_trajectory_ids).astype(np.int64, copy=False)
        all_lens = np.asarray(ds._all_trajectory_lengths).astype(np.int64, copy=False)
        keep = np.array(
            [i for i, tid in enumerate(all_ids.tolist()) if int(tid) in wanted], dtype=np.int64
        )
        active_ids = all_ids[keep]
        active_lengths = all_lens[keep]

        ds._active_episode_indices = active_ids
        ds._active_traj_ids = active_ids
        ds._trajectory_ids = active_ids
        ds._trajectory_lengths = active_lengths
        ds._trajectory_id_to_index_active = {int(t): i for i, t in enumerate(active_ids.tolist())}
        ds._build_active_step_indexing()

        report["leaves"].append({
            "class": type(ds).__name__,
            "dataset_name": getattr(ds, "dataset_name", None),
            "n_all_episodes": int(len(all_ids)),
            "n_active_episodes": int(len(active_ids)),
            "n_active_steps": int(ds._subset_total_steps),
            "len_dunder": int(len(ds)),
        })
    report["total_active_steps"] = sum(l["n_active_steps"] for l in report["leaves"])
    report["total_active_episodes"] = sum(l["n_active_episodes"] for l in report["leaves"])
    return report


def episode_start_to_local_index(ds, episode_index: int, start: int) -> int:
    """Exact address of one (episode, start) sample in the ACTIVE subset of a leaf dataset.

    Inverse of the dataset's own `_local_to_abs_index`: the active subset concatenates whole
    episodes in `_active_episode_indices` order, so the local index is the cumulative length of
    the preceding active episodes plus the episode-local start.
    """
    active = np.asarray(ds._active_episode_indices).astype(np.int64, copy=False).tolist()
    pos = active.index(int(episode_index))
    prev_cum = 0 if pos == 0 else int(ds._subset_cum_lengths[pos - 1])
    length = int(ds._subset_lengths[pos])
    if not (0 <= int(start) < length):
        raise IndexError(f"start {start} out of range for episode {episode_index} (length {length})")
    return prev_cum + int(start)


def get_sample(mixture, ds, episode_index: int, start: int) -> dict:
    """One sample at an EXPLICIT (episode, start) address, through the official assembly.

    `LeRobotMixtureDataset.__getitem__` treats its argument as an RNG seed, not an address: it
    calls `sample_step()` which draws a random (dataset, trajectory, step). The FE protocol needs
    a fixed, reproducible sample set, so this reproduces the official body of `__getitem__`
    verbatim -- `_select_video_keys_for_sample` -> `get_step_data` ->
    `_get_transforms_for_selected_video_keys` -> `_build_output_sample` -- and changes ONLY how
    the address is chosen. No transform, normalization or assembly step is re-implemented.
    """
    local_idx = episode_start_to_local_index(ds, episode_index, start)
    video_keys = mixture._select_video_keys_for_sample(ds, local_idx)
    if not video_keys:
        raise ValueError(f"Dataset {ds.dataset_name} has no video keys configured.")
    raw = ds.get_step_data(int(episode_index), int(start), modality_keys_override={"video": video_keys})
    transforms = mixture._get_transforms_for_selected_video_keys(ds, video_keys)
    return mixture._build_output_sample(ds, transforms(raw), video_keys)


def episode_lengths(ds) -> dict[int, int]:
    """Active episode id -> length, from the dataset's own rebuilt indexing arrays."""
    active = np.asarray(ds._active_episode_indices).astype(np.int64, copy=False).tolist()
    lens = np.asarray(ds._subset_lengths).astype(np.int64, copy=False).tolist()
    return {int(e): int(l) for e, l in zip(active, lens)}
