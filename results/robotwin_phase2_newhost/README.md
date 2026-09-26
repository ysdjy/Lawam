# RoboTwin Phase 2 new-host handoff

This directory is the handoff for the official LaWAM RoboTwin baseline audit and screening completed on 2026-09-26.

## State

- Official checkpoint: `lawam_robotwin_sft_release`; verified SHA256 and strict parameter/alias load.
- Real closed loop: passed SAPIEN RGB, RoboTwin reset, policy query, native planner and completed episodes.
- Screening: 78 completed policy episodes passed integrity checks; 58 succeeded.
- Fixed candidate-pool limitation: `put_object_cabinet` has only 3 Clean and 2 Randomized expert-valid policy episodes. Seeds were not replaced.
- Final selection: Primary 1 `open_microwave`, Primary 2 `stack_blocks_three`, Control `lift_pot`.
- New method: not started; training steps: 0.

## Read first

1. `status.json`
2. `robotwin_screen_results.csv`
3. `protocol.json`
4. `../docs/robotwin_phase2_newhost/BASELINE_SCREEN.md`
5. `../docs/robotwin_phase2_newhost/TASK_SELECTION.md`
6. `../docs/robotwin_phase2_newhost/DATA_SEMANTICS.md`
7. `evidence/external_storage.jsonl`

The CSV separates `paper_percent`, `local_success_percent`, and the empty `new_method_percent` column. System errors and expert seed rejections are separate fields and are not silently counted as policy failures.

## Artifact policy

The repository contains the small, reviewable artifacts needed to understand and reproduce the work: source scripts, protocol, status, CSV summaries, episode summaries, trace-analysis summaries, failure reviews, audit logs, figures, and archive manifests. Raw videos, action JSONL and physical traces are much larger than a normal Git repository (the raw episode tree is about 14 GB); they remain in the verified external archive described by `evidence/external_storage.jsonl`. That manifest records the archive path, byte size and SHA256 for every moved artifact. Workspace files that were not moved remain at their exact relative paths and are listed in `evidence/artifact_manifest.jsonl`.

The original LaWAM worktree changes were preserved; this handoff commit does not reset or clean unrelated changes.
