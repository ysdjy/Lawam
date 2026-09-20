"""H AUDIT 1 — infrastructure equivalence, on 6 states, before any formal headroom rollout.

Checks:
  E1  the four variants' FIRST action chunk, produced through the websocket path with injected
      `initial_noise` (+ `future_override` for B/C/D), reproduces the frozen C2 chunk
        A baseline        -> C2 `A_pred_seed101`        (no override at all: the server uses its live H_pred)
        B top-D repair    -> C2 `B_top_D_seed101`
        C top-(D x S)     -> C2 `D_top_D_times_S_seed101`
        D full oracle     -> C2 `A_realFuture_seed101`
  E2  snapshot restore is reproducible: restoring and replaying the same actions twice gives the same state
  E3  the continuation noise schedule is reproducible: the same schedule replayed gives identical actions,
      and a different schedule gives different ones

Variants B/C are built on the LIVE full-precision H_pred (hpred_fp32/), with only the 64 tokens of the
frozen selector replaced by the archived H_real — exactly the construction g3_c2_chunks.py used.

Nothing is trained; no repository file is modified.
Output: <run_dir>/h_audit1_equivalence.json  (exit 1 if any hard check fails)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research/action_critical_reliability_probe/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_common as pc  # noqa: E402
import sim_common as sc  # noqa: E402
from h_feasibility import NoiseScheduledClient, make_noise  # noqa: E402

CKPT = REPO / "results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
C2_KEY = {"A_baseline": "A_pred_seed101", "B_top_D": "B_top_D_seed101",
          "C_top_DS": "D_top_D_times_S_seed101", "D_full_oracle": "A_realFuture_seed101"}
TOL_CHUNK = 1e-5          # normalized action units; the websocket path adds float32 transport only
SCHEDULE_LEN = 40


def noise_schedule(state_id: str, schedule_id: int, n: int = SCHEDULE_LEN) -> list[np.ndarray]:
    seed = int.from_bytes(hashlib.sha256(f"{state_id}|{schedule_id}".encode()).digest()[:8], "big")
    return [make_noise((seed + k) % (2 ** 31)) for k in range(n)]


def build_variants(hp_npz, t_npz) -> dict[str, np.ndarray | None]:
    """A sends no override; B/C splice the frozen picks into the LIVE H_pred; D is the archived H_real."""
    h_pred = hp_npz["h_pred"].astype(np.float32)             # live full-precision tensor
    h_real = t_npz["h_real"].astype(np.float32)              # the same archive C2 spliced from
    out: dict[str, np.ndarray | None] = {"A_baseline": None}
    for name, key in (("B_top_D", "pick_B_top_D"), ("C_top_DS", "pick_D_top_D_times_S")):
        f = h_pred.copy()
        f[hp_npz[key]] = h_real[hp_npz[key]]
        out[name] = f[None]
    out["D_full_oracle"] = h_real[None]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--g3_run", required=True)
    ap.add_argument("--port", type=int, default=10093)
    args = ap.parse_args()
    run_dir, g3 = Path(args.run_dir), Path(args.g3_run)
    states = [json.loads(l) for l in open(g3 / "c2/states_manifest.jsonl")]

    # 2 stress conditions x 3 suites x 1 state = 6, taken deterministically (first episode of each)
    picked, seen = [], set()
    for s in states:
        k = (s["condition"], s["suite"])
        if k not in seen:
            seen.add(k)
            picked.append(s)
    assert len(picked) == 6, picked

    client = sc.ModelClient(policy_ckpt_path=str(CKPT), port=args.port)
    shim = NoiseScheduledClient(client.client)
    client.client = shim

    results, failures = [], []
    for st in states if False else picked:
        sid = st["state_id"]
        d = np.load(st["state_npz"], allow_pickle=True)
        hp = np.load(run_dir / "hpred_fp32" / f"{sid}.npz")
        tz = np.load(g3 / "confirm/treatment_chunks" / st["condition"] / f"{sid}.npz")
        c2 = np.load(g3 / "c2/c2_chunks" / f"{sid}.npz")
        variants = build_variants(hp, tz)
        ex = {"primary_image": [np.asarray(d["primary"])], "wrist_image": [np.asarray(d["wrist"])],
              "lang": str(d["lang"][0]), "state": np.asarray(d["state"])[None, :],
              "embodiment_id": 25, "action_hz": 20.0}
        n0 = make_noise(101)                                  # C2 used seed 101 for its stored chunks

        rec = {"state_id": sid, "condition": st["condition"], "suite": st["suite"]}
        for name, fut in variants.items():
            client.reset(task_description=str(d["lang"][0]))
            shim.initial_noise, shim.future_override = n0, fut
            client.step(example=ex, step=0)
            # the client caches the UNNORMALISED chunk; C2 stored the normalised one, so the reference is
            # put through the very same unnormalisation function before comparing
            got = np.asarray(client._get_slot_state(0).raw_actions, dtype=np.float64)
            ref = sc.ModelClient.unnormalize_actions(
                normalized_actions=np.asarray(c2[C2_KEY[name]], dtype=np.float32),
                action_norm_stats=client.action_norm_stats)
            err = float(np.max(np.abs(got[:, :7] - np.asarray(ref, dtype=np.float64)[:, :7])))
            rec[f"E1_{name}_maxabs_vs_C2"] = err
            if err > TOL_CHUNK:
                failures.append(f"{sid}: {name} first chunk differs from C2 by {err:.3e}")

        # E3 noise-schedule reproducibility (same schedule twice, then a different schedule)
        sch = noise_schedule(sid, 0)
        def first_two(schedule):
            client.reset(task_description=str(d["lang"][0]))
            shim.initial_noise, shim.future_override = schedule[0], None
            client.step(example=ex, step=0)
            return np.asarray(client._get_slot_state(0).raw_actions, dtype=np.float64).copy()
        a, b = first_two(sch), first_two(sch)
        c = first_two(noise_schedule(sid, 1))  # a different schedule must change the chunk
        rec["E3_same_schedule_maxabs"] = float(np.max(np.abs(a - b)))
        rec["E3_different_schedule_maxabs"] = float(np.max(np.abs(a - c)))
        if rec["E3_same_schedule_maxabs"] != 0.0:
            failures.append(f"{sid}: same noise schedule not reproducible ({rec['E3_same_schedule_maxabs']})")
        if rec["E3_different_schedule_maxabs"] == 0.0:
            failures.append(f"{sid}: different noise schedules produced identical actions")
        results.append(rec)
        print(f"{sid}: " + " ".join(f"{k.split('_')[1]}={rec[k]:.2e}" for k in rec if k.startswith("E1_"))
              + f" | sched_same={rec['E3_same_schedule_maxabs']:.1e} "
                f"sched_diff={rec['E3_different_schedule_maxabs']:.1e}", flush=True)
    client.close()

    # E2 snapshot restore reproducibility, on the same 6 states
    e2 = []
    env, cur = None, None
    for st in picked:
        key = (st["suite"], st["task_id"])
        if key != cur:
            if env is not None:
                env.close()
            suite = sc.load_suite(st["suite"])
            env, _ = sc.make_env(suite.get_task(st["task_id"]), 0)
            sc.reset_to_init_state(env, suite.get_task_init_states(st["task_id"])[st["episode_idx"]])
            cur = key
        snap = sc.SimSnapshot.from_npz(np.load(st["snapshot"]))
        acts = np.asarray(st["executed_chunk"], dtype=np.float64)
        runs = []
        for _ in range(2):
            sc.restore_snapshot(env, snap)
            for a in acts:
                env.step(a.tolist())
            runs.append(sc.eef_pos(env).copy())
        e2.append({"state_id": st["state_id"],
                   "restore_repeat_eef_maxabs_m": float(np.max(np.abs(runs[0] - runs[1])))})
        if e2[-1]["restore_repeat_eef_maxabs_m"] != 0.0:
            failures.append(f"{st['state_id']}: snapshot restore not reproducible")
    if env is not None:
        env.close()

    payload = {"n_states": len(picked), "tolerance_chunk": TOL_CHUNK,
               "E1_E3": results, "E2_snapshot_restore": e2,
               "failures": failures, "passed": not failures}
    pc.write_json(run_dir / "h_audit1_equivalence.json", payload)
    print(json.dumps({"passed": payload["passed"], "failures": failures}, indent=1))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
