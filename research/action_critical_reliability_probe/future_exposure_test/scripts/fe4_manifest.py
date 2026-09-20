"""FE checkpoint manifest + final integrity re-verification.

Re-asserts the round's anchors AFTER all training: the released LaWAM state_dict is untouched,
every frozen module hash still matches its AUDIT 0 baseline in every run, and the four trained
Flow Heads are distinct from each other and from the initialisation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_common as fc  # noqa: E402

TRAIN_DIR = fc.RUN_DIR / "train"
OUT = fc.RUN_DIR / "CHECKPOINT_MANIFEST.json"


def main() -> None:
    audit0 = json.loads((fc.RUN_DIR / "PREFLIGHT_FE_AUDIT0.json").read_text())
    base_frozen = audit0["frozen_module_hashes"]
    base_flow = audit0["flow_initial_hash"]

    man: dict = {
        "source_checkpoint": str(fc.CKPT_FILE.relative_to(fc.REPO_ROOT)),
        "source_checkpoint_file_sha256": fc.sha256_file(fc.CKPT_FILE),
        "lawam_state_dict_anchor_expected": fc.LAWAM_STATE_DICT_SHA256,
        "flow_initial_hash": base_flow,
        "frozen_baselines": base_frozen,
        "runs": [],
    }

    for log in sorted(TRAIN_DIR.glob("TRAIN_LOG_*.jsonl")):
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        start = next(r for r in rows if r.get("event") == "start")
        end = next((r for r in rows if r.get("event") == "end"), None)
        vals = [r for r in rows if r.get("event") == "validate"]
        entry = {
            "log": log.name,
            "arm": start["arm"], "seed": start["seed"], "p_pred": start["p_pred"],
            "completed": end is not None,
            "stopped": end["stopped"] if end else None,
            "final_update": end["final_update"] if end else None,
            "best_update": end["best_update"] if end else None,
            "best_val_loss_own": end["best_val_loss_own"] if end else None,
            "realised_gt_row_fraction": end["realised_gt_row_fraction"] if end else None,
            "target_gt_row_fraction": end["target_gt_row_fraction"] if end else None,
            "wall_seconds": end["wall_seconds"] if end else None,
            "n_validations": len(vals),
            "flow_hash_before": start["flow_hash_before"],
            "flow_hash_after": end["flow_hash_after"] if end else None,
            "flow_changed": end["flow_changed"] if end else None,
            "frozen_unchanged_within_run": end["frozen_unchanged"] if end else None,
            "frozen_matches_audit0_baseline": (end["frozen_hashes_after"] == base_frozen) if end else None,
            "flow_hash_before_matches_audit0": start["flow_hash_before"] == base_flow,
            "exposure_ratio_within_tolerance": (
                abs(end["realised_gt_row_fraction"] - end["target_gt_row_fraction"]) < 0.02
            ) if end else None,
        }
        for suffix in ("best", "final"):
            p = TRAIN_DIR / f"flow_{start['arm']}_seed{start['seed']}_{suffix}.pt"
            entry[f"{suffix}_checkpoint"] = p.name if p.exists() else None
            entry[f"{suffix}_checkpoint_sha256"] = fc.sha256_file(p) if p.exists() else None
        man["runs"].append(entry)

    finals = {(r["arm"], r["seed"]): r["final_checkpoint_sha256"] for r in man["runs"]}
    vals = [v for v in finals.values() if v]
    man["all_final_checkpoints_distinct"] = len(set(vals)) == len(vals)
    man["all_runs_completed"] = all(r["completed"] for r in man["runs"])
    man["all_frozen_match_audit0"] = all(r["frozen_matches_audit0_baseline"] for r in man["runs"])
    man["all_started_from_same_flow_init"] = all(r["flow_hash_before_matches_audit0"] for r in man["runs"])
    man["all_exposure_ratios_within_tolerance"] = all(
        r["exposure_ratio_within_tolerance"] for r in man["runs"])

    # The released LaWAM checkpoint must be what every prior round used. The anchor is recomputed
    # with the ORIGINAL round-1 procedure (loaded runtime backend, use_bf16=True, float32 bytes,
    # no dtype/shape metadata) so the digest is directly comparable across all six rounds.
    vla, backend = fc.load_backend(use_bf16=True)
    digest, n_params = fc.legacy_state_dict_hash(backend)
    man["lawam_state_dict_sha256_recomputed"] = digest
    man["lawam_state_dict_params"] = int(n_params)
    man["lawam_checkpoint_untouched"] = digest == fc.LAWAM_STATE_DICT_SHA256
    man["anchor_procedure"] = ("round-1 g2_audit0.state_dict_hash: sorted keys, key bytes, "
                               "float32 tensor bytes, no dtype/shape metadata, bf16-loaded backend")

    fc.write_json(OUT, man)
    print(json.dumps({k: v for k, v in man.items() if k != "runs"}, indent=2))
    for r in man["runs"]:
        print(f"  {r['arm']:15s} seed {r['seed']}  stop={r['stopped']:22s} "
              f"final_u={r['final_update']:5} best_u={r['best_update']:5} "
              f"gt_frac={r['realised_gt_row_fraction']:.4f} frozen_ok={r['frozen_matches_audit0_baseline']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
