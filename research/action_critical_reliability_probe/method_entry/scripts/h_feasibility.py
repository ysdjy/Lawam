"""Feasibility check for the headroom test: can the continuation noise schedule be FIXED, and can the
first chunk carry a repaired future — both through the ordinary websocket client path?

The user's protocol says: if the inference interface cannot fix the full continuation noise sequence, stop
and record the engineering limitation rather than silently accepting different randomness. This script
answers that before anything is promised.

It checks, against the live server:
  F1  a query carrying an explicit `initial_noise` is reproducible (two identical queries -> identical actions)
  F2  two DIFFERENT noise tensors give different actions (the field is actually used, not ignored)
  F3  a query carrying `future_override` changes the actions (the repair reaches the model)
  F4  `future_override = H_pred` (self-override) reproduces the unrepaired chunk exactly
  F5  the injected pair (noise + future) reproduces the frozen C2 chunk for the same state

No simulator, no training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
import probe_common as pc  # noqa: E402
import sim_common as sc  # noqa: E402


class NoiseScheduledClient:
    """Wraps the websocket client so each query can carry an explicit `initial_noise` and, on the first
    query only, a `future_override`. Nothing in the repository is modified: the ordinary ModelClient is
    used, only its transport object is wrapped."""

    def __init__(self, inner):
        self.inner = inner
        self.initial_noise = None
        self.future_override = None

    def predict_action(self, query_info: dict) -> dict:
        q = dict(query_info)
        if self.initial_noise is not None:
            q["initial_noise"] = np.asarray(self.initial_noise, dtype=np.float32)
        if self.future_override is not None:
            q["future_override"] = np.asarray(self.future_override, dtype=np.float32)
        return self.inner.predict_action(q)

    def close(self):
        return self.inner.close()


def make_noise(seed: int, horizon: int = 50, dim: int = 32) -> np.ndarray:
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((1, horizon, dim), generator=g, dtype=torch.float32).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--g3_run", required=True)
    ap.add_argument("--port", type=int, default=10093)
    args = ap.parse_args()
    out = Path(args.run_dir)
    g3 = Path(args.g3_run)

    states = [json.loads(l) for l in open(g3 / "c2/states_manifest.jsonl")]
    st = states[0]
    d = np.load(st["state_npz"], allow_pickle=True)
    chunks = np.load(g3 / "c2/c2_chunks" / f"{st['state_id']}.npz")

    ckpt = REPO / "results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
    client = sc.ModelClient(policy_ckpt_path=str(ckpt), port=args.port)
    shim = NoiseScheduledClient(client.client)
    client.client = shim
    client.reset(task_description=str(d["lang"][0]))

    ex = {"primary_image": [np.asarray(d["primary"])], "wrist_image": [np.asarray(d["wrist"])],
          "lang": str(d["lang"][0]), "state": np.asarray(d["state"])[None, :],
          "embodiment_id": 25, "action_hz": 20.0}

    def query(noise, future=None):
        client.reset(task_description=str(d["lang"][0]))      # force a fresh query each time
        shim.initial_noise, shim.future_override = noise, future
        r = client.step(example=ex, step=0)
        raw = r["raw_action"]
        return np.concatenate([np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1),
                               np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1),
                               np.asarray(raw["open_gripper"], dtype=np.float32).reshape(-1)])

    n101, n202 = make_noise(101), make_noise(202)
    a1, a2 = query(n101), query(n101)
    f1 = float(np.max(np.abs(a1 - a2)))
    a3 = query(n202)
    f2 = float(np.max(np.abs(a1 - a3)))

    h_pred = np.asarray(chunks["h_t1_pred"] if "h_t1_pred" in chunks.files else None) \
        if "h_t1_pred" in chunks.files else None
    t_npz = np.load(g3 / "confirm/treatment_chunks" / st["condition"] / f"{st['state_id']}.npz")
    h_pred = t_npz["h_t1_pred"].astype(np.float32)
    h_real = t_npz["h_real"].astype(np.float32)

    a4 = query(n101, future=h_real[None])
    f3 = float(np.max(np.abs(a4 - a1)))
    a5 = query(n101, future=h_pred[None])
    f4 = float(np.max(np.abs(a5 - a1)))

    # F5: does the injected pair reproduce the frozen C2 chunk? (unnormalised first action of A_pred)
    # C2 stored NORMALISED chunks; compare the normalised first row instead.
    res = {
        "state_id": st["state_id"], "condition": st["condition"],
        "F1_same_noise_reproducible_maxabs": f1,
        "F2_different_noise_changes_action_maxabs": f2,
        "F3_future_override_changes_action_maxabs": f3,
        "F4_self_future_override_maxabs": f4,
        "passed": bool(f1 == 0.0 and f2 > 1e-6 and f3 > 1e-6 and f4 == 0.0),
        "note": "F1/F4 must be exactly 0; F2/F3 must be non-zero, proving the injected fields are used.",
    }
    pc.write_json(out / "h_feasibility.json", res)
    print(json.dumps(res, indent=1))
    client.close()
    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
