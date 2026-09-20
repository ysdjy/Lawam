"""Shared label/metric definitions (pre-registered in configs/protocol.yaml)."""
from __future__ import annotations

import numpy as np

S_E_THRESH = 0.25
PERP_MAX = 0.08
S_F_THRESH = 0.05
NOISE_FLOOR_MULT = 3.0
ROI_TOKEN_RADIUS = 2
GRID = 16
PATCH = 16


def executed_label(p, p_A, p_B, bowl_disp, failed=False):
    if failed or bowl_disp > 0.01:
        return "FAIL", float("nan"), float("nan")
    p, p_A, p_B = map(lambda x: np.asarray(x, dtype=np.float64), (p, p_A, p_B))
    axis = p_A - p_B
    mid = 0.5 * (p_A + p_B)
    s_e = float(np.dot(p - mid, axis) / (np.dot(axis, axis) + 1e-12))
    proj = mid + s_e * axis
    perp = float(np.linalg.norm(p - proj))
    if perp > PERP_MAX:
        return "OTHER", s_e, perp
    if s_e >= S_E_THRESH:
        return "A", s_e, perp
    if s_e <= -S_E_THRESH:
        return "B", s_e, perp
    return "OTHER", s_e, perp


def feat_mse(f, g, mask=None):
    f = np.asarray(f, dtype=np.float64).reshape(-1, f.shape[-1]); g = np.asarray(g, dtype=np.float64).reshape(-1, g.shape[-1])
    d = ((f - g) ** 2).mean(axis=1)  # per token
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.sum() == 0:
            return float("nan")
        return float(d[m].mean())
    return float(d.mean())


def predicted_label(f, u_A, u_B, noise_floor, mask=None):
    d_A = feat_mse(f, u_A, mask); d_B = feat_mse(f, u_B, mask)
    if not np.isfinite(d_A) or not np.isfinite(d_B):
        return "UNKNOWN", float("nan"), d_A, d_B
    s_f = (d_B - d_A) / (d_A + d_B + 1e-12)
    if abs(d_A - d_B) < NOISE_FLOOR_MULT * noise_floor or abs(s_f) < S_F_THRESH:
        return "UNKNOWN", float(s_f), d_A, d_B
    return ("A" if s_f > 0 else "B"), float(s_f), d_A, d_B


def roi_mask_from_pixels(pixels_rowcol, radius=ROI_TOKEN_RADIUS):
    """Token mask [256] (row-major 16x16 grid over the 256x256 model image) around given (row, col) pixels."""
    mask = np.zeros((GRID, GRID), dtype=bool)
    for r, c in np.asarray(pixels_rowcol).reshape(-1, 2):
        tr, tc = int(r) // PATCH, int(c) // PATCH
        r0, r1 = max(0, tr - radius), min(GRID, tr + radius + 1)
        c0, c1 = max(0, tc - radius), min(GRID, tc + radius + 1)
        mask[r0:r1, c0:c1] = True
    return mask.reshape(-1)
