# STATUS — G3 round COMPLETE

run `acr_g3_practical_20260920_134247` · finished 2026-09-20

All stages of the G3 round are done. **Final decision: G3-PASS, qualified on criterion 5.**
See `REPORT_G3.md` for the full report, `AUDIT_G3.md` for AUDIT 0/1/2/3 + FINAL, `NEXT_STAGE.md` for the
handover.

| stage | status |
|---|---|
| G3 AUDIT 0 — project state, frozen conclusions, weight hash | complete |
| Pilot — 3 families x 3 severities, 150 rollout attempts | complete |
| G3 AUDIT 1 — severity selection by the pre-registered rule | complete |
| Confirm stage C1 — practical significance, 180 rollouts, 6 tasks | complete |
| G3 AUDIT 2 — C1 verdict | complete |
| Confirm stage C2 — D / S / D x S / attention ranking, 120 states, 3,600 executed chunks | complete |
| G3 AUDIT 3 + FINAL G3 AUDIT | complete |
| One-chunk headroom test | **NOT RUN** (outside the scope set for C2; task-level headroom stays undemonstrated) |

Headline numbers: the future-error consequence roughly doubles under stress (0.82 -> 1.87 / 1.70 mm,
treatment/nuisance 1.77 -> 3.15 / 3.57), and `D x S` recovers 0.822 of the oracle correction versus 0.736
for `D`, 0.787 for `S`, 0.701 for `D x attention` and 0.432 for attention, statistically indistinguishable
from the oracle `U x S` (0.821). The advantage of `D x S` over `D` is +0.231 mm against a 0.51 mm sampler
nuisance on an 85 mm chunk.

Not committed: `rollouts/`, `states/`, `treatment_chunks/`, `c2/sensitivity/`, `c2/c2_chunks/` and the
confirm equivalents (raw frames, snapshots, per-token arrays and action chunks). Everything committed is
text, metrics or figures.
