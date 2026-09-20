"""Shared helpers for the action-critical-reliability probe.

Deliberately thin: the model loading / example building / noise helpers are *reused verbatim* from the
previous branch diagnostic (`research/branch_diagnostic/scripts/model_common.py`) so that this probe
talks to exactly the same policy interface that was already validated in run bd_20260914_093902.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BD_SCRIPTS = REPO_ROOT / "research" / "branch_diagnostic" / "scripts"
PROBE_ROOT = REPO_ROOT / "research" / "action_critical_reliability_probe"
RESULTS_ROOT = REPO_ROOT / "results" / "action_critical_reliability_probe"

for _p in (str(REPO_ROOT), str(BD_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def current_run_id() -> str:
    return (PROBE_ROOT / ".current_run_id").read_text().strip()


def run_dir(run_id: str | None = None) -> Path:
    return RESULTS_ROOT / (run_id or current_run_id())


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(str(type(o)))


def write_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def append_jsonl(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def stats(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "n": int(a.size),
    }


def env_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.executable}
    for mod in ("torch", "numpy", "transformers"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception as exc:  # pragma: no cover - diagnostic only
            out[mod] = f"unavailable: {exc}"
    out["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return out
