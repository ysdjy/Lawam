# STATUS — G3 round is IN PROGRESS, not finished

run `acr_g3_practical_20260920_134247` · last updated 2026-09-20

This directory is pushed mid-round at the user's request. **There is no final decision yet.** Nothing here
should be read as a G3 verdict.

## What is done

| stage | status | artefacts |
|---|---|---|
| G3 AUDIT 0 — project state, frozen conclusions, weight hash | **complete** | `AUDIT_G3.md`, `g2_audit0_model.json`, `PROTOCOL_G3.yaml` |
| Pilot — 3 families × 3 severities, 150 rollout attempts | **complete** | `PILOT_RESULTS.csv`, `pilot_summary.json`, `treatment_*.jsonl`, `figures/g3_pilot.png` |
| G3 AUDIT 1 — severity selection by the pre-registered rule | **complete** | `AUDIT_G3.md` |
| Confirm **stage C1** — practical significance, 180 rollouts, 6 tasks | **complete** | `CONFIRM_RESULTS.csv`, `confirm_c1_summary.json`, `confirm/treatment_*.jsonl`, `figures/g3_confirm_c1.png` |
| G3 AUDIT 2 — C1 verdict | **complete** | `AUDIT_G3.md` |
| Confirm **stage C2** — D / S / D×S / attention ranking + equal-budget causal repair | **NOT STARTED** | — |
| One-chunk headroom test | **NOT STARTED** | — |
| FINAL G3 AUDIT, `REPORT_G3.md`, `NEXT_STAGE.md` | **NOT WRITTEN** | — |

Deliverables named in `PROTOCOL_G3.yaml` that do not exist yet, because their stage has not run:
`TOKEN_SELECTION_RESULTS.csv`, `EXECUTION_RESULTS.csv`, `FAILURE_STATES.jsonl`, `CRITICAL_STATES.jsonl`,
`REPORT_G3.md`, `NEXT_STAGE.md`.

## Where the evidence stands (stage C1 only)

Executed consequence of the future-prediction error, π(H_real) vs π(H_pred) at fixed flow noise from
identical snapshots, 60 episodes per condition over 6 tasks in 3 suites:

| condition | success | treatment mm (median / mean) | mean CI | nuisance mm | treatment / nuisance |
|---|---|---|---|---|---|
| ID | 95 % | 0.82 / 1.00 | [0.90, 1.12] | 0.44 | 1.77 |
| A occlusion 80 px | 78 % | 1.87 / 2.23 | [2.01, 2.45] | 0.61 | 3.15 |
| B camera shift 24 px | 68 % | 1.70 / 2.47 | [2.16, 2.81] | 0.49 | 3.57 |

All 35 failures are 250-step timeouts; 0 infrastructure errors; no episode excluded. The amplification is
~2× and is still only ~2 % of the chunk's 85 mm motion.

**Stage C2 has not run, so nothing is known yet about whether `D × S` locates the responsible tokens under
stress.** The G3 decision (`G3-PASS` / `G3-FAIL-PRACTICAL` / `G3-FAIL-RANKING` / `G3-INCONCLUSIVE`) requires
that stage and cannot be inferred from the table above.

## Not committed

`rollouts/`, `states/`, `treatment_chunks/` and the confirm equivalents (~7.4 GB of frames, snapshots and
raw action chunks) stay out of the repository. Everything committed is text, metrics or figures.
