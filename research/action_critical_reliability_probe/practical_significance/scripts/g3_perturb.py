"""G3 stress perturbations — exactly the nine severities frozen in PROTOCOL_G3.yaml.

Design rule that makes the U label impossible to misalign: families A and B perturb the image **at the
point where frames are recorded**, so `o_t` and the future frame `o_{t+7}` come from the same perturbed
stream by construction. There is no code path in which the policy sees a perturbed present and the U label
is computed from a clean future.

Family A  grey (128) rectangular cutout on the primary image only, centred on the target object's
          projection at the FIRST policy query and then frozen in image coordinates.
Family B  fixed diagonal translation of the primary image with edge-replication padding.
Family C  xy translation of the target object at episode initialisation, with the three pre-registered
          validity checks.

The wrist camera is never perturbed in A and B.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
import probe_common  # noqa: E402,F401  (must come first: it puts branch_diagnostic/scripts on sys.path)
import sim_common as sc  # noqa: E402

IMG = sc.LIBERO_ENV_RESOLUTION          # 256
PATCH = 16                              # 16x16 grid of 16 px patches -> one patch per future token
GREY = 128

SEVERITIES = {
    "A_occlusion":   {"mild": 48, "medium": 80, "strong": 112},        # px, = 9 / 25 / 49 tokens
    "B_camera_shift": {"mild": 6, "medium": 12, "strong": 24},         # px, = 0.375 / 0.75 / 1.5 patches
    "C_object_shift": {"mild": 0.02, "medium": 0.04, "strong": 0.06},  # metres
}
OBJECT_SHIFT_DIR = np.array([1.0, 1.0]) / np.sqrt(2.0)
MAX_BASE_DISTANCE = 0.75


@dataclass
class VisualPerturbation:
    """Deterministic, persistent perturbation of the primary image stream."""
    family: str
    severity: str
    anchor_rc: tuple[int, int] | None = None     # frozen occlusion centre (row, col)
    anchor_source: str = ""

    def size(self):
        return SEVERITIES[self.family][self.severity]

    def apply(self, primary: np.ndarray) -> np.ndarray:
        if self.family == "A_occlusion":
            return _occlude(primary, self.anchor_rc, int(self.size()))
        if self.family == "B_camera_shift":
            return _shift(primary, int(self.size()))
        return primary


def _occlude(img: np.ndarray, centre: tuple[int, int], size: int) -> np.ndarray:
    out = np.ascontiguousarray(img).copy()
    half = size // 2
    r = int(np.clip(centre[0], half, IMG - half - 1))
    c = int(np.clip(centre[1], half, IMG - half - 1))
    out[r - half:r + half, c - half:c + half] = GREY
    return out


def _shift(img: np.ndarray, px: int) -> np.ndarray:
    """Translate by (+px, +px) with edge replication (a camera-like viewpoint offset)."""
    padded = np.pad(np.ascontiguousarray(img), ((px, 0), (px, 0), (0, 0)), mode="edge")
    return np.ascontiguousarray(padded[:IMG, :IMG])


def occlusion_anchor(env, target_object: str) -> tuple[tuple[int, int], str]:
    """Target object's projected (row, col) in the flipped model image frame; (128,128) if unavailable."""
    try:
        pt = sc.body_pos(env, target_object).reshape(1, 3)
        rc = sc.world_to_model_pixels(env, pt)[0]
        r, c = int(round(float(rc[0]))), int(round(float(rc[1])))
        if 0 <= r < IMG and 0 <= c < IMG:
            return (r, c), "object_projection"
        return (IMG // 2, IMG // 2), f"fallback_centre(projection_out_of_frame r={r} c={c})"
    except Exception as exc:  # noqa: BLE001
        return (IMG // 2, IMG // 2), f"fallback_centre(projection_failed: {type(exc).__name__})"


# ----------------------------------------------------------------------------- family C
def _free_joint_qpos_addr(env, body_name: str) -> int | None:
    e = sc.raw_env(env)
    model = e.sim.model
    bid = e.obj_body_id[body_name]
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) == int(bid) and int(model.jnt_type[j]) == 0:   # 0 = mjJNT_FREE
            return int(model.jnt_qposadr[j])
    return None


def apply_object_shift(env, target_object: str, shift_m: float, init_object_xy: np.ndarray) -> dict:
    """Translate the target object in xy and run the three pre-registered validity checks.

    Returns a record; `valid` False means the episode must be marked invalid and RECORDED, never replaced.
    """
    e = sc.raw_env(env)
    addr = _free_joint_qpos_addr(env, target_object)
    rec: dict = {"target_object": target_object, "shift_m": float(shift_m),
                 "direction": OBJECT_SHIFT_DIR.tolist(), "joint_qpos_addr": addr}
    if addr is None:
        rec.update({"valid": False, "reason": "no free joint found for the target object"})
        return rec

    before = sc.body_pos(env, target_object).copy()
    qpos = e.sim.data.qpos
    qpos[addr:addr + 2] = qpos[addr:addr + 2] + OBJECT_SHIFT_DIR * shift_m
    e.sim.forward()
    after = sc.body_pos(env, target_object).copy()
    rec.update({"pos_before": before.tolist(), "pos_after": after.tolist(),
                "realised_shift_m": float(np.linalg.norm(after[:2] - before[:2]))})

    # check 1 — no contact with any body other than the table
    bid = e.obj_body_id[target_object]
    table_ids = {int(e.sim.model.body_name2id(n)) for n in e.sim.model.body_names
                 if "table" in n.lower()}
    bad_contacts = []
    for i in range(e.sim.data.ncon):
        con = e.sim.data.contact[i]
        b1 = int(e.sim.model.geom_bodyid[con.geom1])
        b2 = int(e.sim.model.geom_bodyid[con.geom2])
        if bid in (b1, b2):
            other = b2 if b1 == bid else b1
            if other not in table_ids and other != bid:
                bad_contacts.append(str(e.sim.model.body_id2name(other)))
    rec["contacts_with_non_table"] = sorted(set(bad_contacts))

    # check 2 — xy stays inside the init-state bounding box expanded by the shift magnitude
    lo = init_object_xy.min(axis=0) - shift_m
    hi = init_object_xy.max(axis=0) + shift_m
    rec["inside_init_bbox"] = bool(np.all(after[:2] >= lo) and np.all(after[:2] <= hi))
    rec["init_bbox"] = [lo.tolist(), hi.tolist()]

    # check 3 — within reach of the robot base
    base = np.asarray(e.sim.data.body_xpos[e.sim.model.body_name2id("robot0_base")], dtype=np.float64)
    rec["distance_to_base_m"] = float(np.linalg.norm(after[:2] - base[:2]))
    rec["within_reach"] = bool(rec["distance_to_base_m"] <= MAX_BASE_DISTANCE)

    rec["valid"] = bool(not bad_contacts and rec["inside_init_bbox"] and rec["within_reach"])
    if not rec["valid"]:
        rec["reason"] = ("contact:" + ",".join(rec["contacts_with_non_table"]) if bad_contacts else
                         "outside_init_bbox" if not rec["inside_init_bbox"] else "out_of_reach")
    return rec


def all_conditions() -> list[tuple[str, str]]:
    return [(f, s) for f in SEVERITIES for s in ("mild", "medium", "strong")]
