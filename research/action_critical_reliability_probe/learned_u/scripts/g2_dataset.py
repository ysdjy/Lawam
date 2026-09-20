"""Shared dataset / metric utilities for the Learned-U round (numpy + torch, no LaWAM forward).

Loads the per-state arrays produced by `g2_extract_u.py` (and, for holdout C, the prior round's
`uncertainty/*.npz` + `sensitivity/*.npz`), assembles the v1 feature vector, and provides the
state-level metrics and the episode bootstrap used everywhere in this round.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))

PRIOR_RUN = REPO / "results/action_critical_reliability_probe/acr_20260919_203850"
TOP_K = 64
TOP_Q = 0.75
EPS_MID_IDX = 1
S_METRIC = "l2_norm_all7"
LABEL_EPS = 1e-6


@dataclass
class StateData:
    state_id: str
    suite: str
    task_id: int
    episode: int
    phase: str
    split: str
    h_t: np.ndarray            # [256, 768] float32
    h_pred: np.ndarray         # [256, 768] float32
    z: np.ndarray              # [32] float32
    U_l2: np.ndarray           # [256] oracle label
    U_mse: np.ndarray
    U_floor_mse: np.ndarray

    @property
    def episode_key(self) -> str:
        return f"{self.suite}_t{self.task_id}_ep{self.episode}"

    def features(self) -> np.ndarray:
        """v1 feature vector x_j = [h_t_j, H_pred_j, H_pred_j - h_t_j, z] -> [256, 2336]."""
        diff = self.h_pred - self.h_t
        z = np.broadcast_to(self.z[None, :], (self.h_t.shape[0], self.z.shape[0]))
        return np.concatenate([self.h_t, self.h_pred, diff, z], axis=1).astype(np.float32)

    def simple_stats(self) -> dict[str, np.ndarray]:
        """The per-token statistics the mandatory no-training baselines are built from."""
        pred_norm = np.linalg.norm(self.h_pred, axis=-1)
        ht_norm = np.linalg.norm(self.h_t, axis=-1)
        diff = self.h_pred - self.h_t
        change_norm = np.linalg.norm(diff, axis=-1)
        cos = (self.h_pred * self.h_t).sum(-1) / (pred_norm * ht_norm + 1e-12)
        return {
            "pred_norm": pred_norm,
            "ht_norm": ht_norm,
            "change_norm": change_norm,
            "cos_dist": 1.0 - cos,
            "pred_mean": self.h_pred.mean(axis=-1),
            "pred_std": self.h_pred.std(axis=-1),
        }


def load_states(run_dir: Path, splits: list[str] | None = None) -> list[StateData]:
    split_rows = json.loads((run_dir / "splits.json").read_text())["rows"]
    out: list[StateData] = []
    for r in split_rows:
        if splits is not None and r["split"] not in splits:
            continue
        if r["source"] == "g2":
            d = np.load(run_dir / "u_dataset" / f"{r['state_id']}.npz")
            h_t, h_pred, z = d["h_t"], d["h_t1_pred"], d["z"].reshape(-1)
            U_l2, U_mse, floor = d["U_U_l2"], d["U_U_mse"], d["Ufloor_U_mse"]
        else:  # prior run (holdout C): same quantities, stored by m3_uncertainty / m2_sensitivity
            u = np.load(PRIOR_RUN / "uncertainty" / f"{r['state_id']}.npz")
            s = np.load(PRIOR_RUN / "sensitivity" / f"{r['state_id']}.npz")
            h_t, h_pred, z = s["h_t"], s["h_t1_pred"], s["z"].reshape(-1)
            U_l2, U_mse, floor = u["U_U_l2"], u["U_U_mse"], u["Ufloor_render_U_mse"]
        out.append(StateData(
            state_id=r["state_id"], suite=r["suite"], task_id=r["task_id"], episode=r["episode"],
            phase=r["phase"], split=r["split"],
            h_t=np.asarray(h_t, dtype=np.float32), h_pred=np.asarray(h_pred, dtype=np.float32),
            z=np.asarray(z, dtype=np.float32), U_l2=np.asarray(U_l2, dtype=np.float64),
            U_mse=np.asarray(U_mse, dtype=np.float64), U_floor_mse=np.asarray(floor, dtype=np.float64),
        ))
    return out


def load_S(state_id: str) -> np.ndarray:
    """Frozen per-token sensitivity of the prior round (holdout C only). Never recomputed here."""
    s = np.load(PRIOR_RUN / "sensitivity" / f"{state_id}.npz")
    metrics = [str(x) for x in s["metrics"]]
    return s["S_norm"][EPS_MID_IDX].mean(axis=0)[:, metrics.index(S_METRIC)].astype(np.float64)


def load_attention(state_id: str) -> np.ndarray:
    return np.load(PRIOR_RUN / "attention" / f"{state_id}.npz")["attention_future"].astype(np.float64)


# ----------------------------------------------------------------------------- metrics
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else float("nan")


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))).astype(np.float64) + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def ndcg_at_k(score: np.ndarray, gain: np.ndarray, k: int = TOP_K) -> float:
    order = np.argsort(-score)[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gain[order] * disc).sum())
    idcg = float((np.sort(gain)[::-1][:k] * disc).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def state_metrics(pred: np.ndarray, oracle: np.ndarray, k: int = TOP_K) -> dict[str, float]:
    thr = np.quantile(oracle, TOP_Q)
    label = oracle >= thr
    top_pred = set(np.argsort(-pred)[:k].tolist())
    top_true = set(np.argsort(-oracle)[:k].tolist())
    inter = len(top_pred & top_true)
    return {
        "spearman": spearman(pred, oracle),
        "pearson": pearson(pred, oracle),
        "top64_recall": inter / max(len(top_true), 1),
        "top64_precision": inter / max(len(top_pred), 1),
        "ndcg64": ndcg_at_k(pred, oracle, k),
        "auroc_top_quartile": auroc(pred, label),
    }


def boot_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """Bootstrap over EPISODES, never over tokens."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    stat = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        vals = np.concatenate([values[groups == g] for g in pick])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            stat.append(vals.mean())
    return [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))] if stat else [np.nan, np.nan]
