# REPORT_HEADROOM.md — one-chunk task-level headroom test

run `acr_headroom_20260920_194346` · 2026-09-20 · branch `research/action-critical-reliability-probe` @ `60a9ae2`
LaWAM `state_dict` sha256 `37a53b8c…b523`, unchanged · **nothing was trained; no repository file modified**

768 continuation rollouts (480 main + 288 robustness), 0 infrastructure errors, nothing excluded.
This report answers three questions and nothing else.

---

## Q1 — Can a full future-feature oracle intervention change the final task outcome?

**No.**

| variant | final task success (120 states) |
|---|---|
| baseline | **95/120 = 0.792** |
| top-D repair | 92/120 = 0.767 |
| top-(D × S) repair | 91/120 = 0.758 |
| **full oracle (H_pred → H_real entirely)** | **90/120 = 0.750** |

full oracle − baseline = **−0.042**, CI [−0.092, +0.008], McNemar p = 0.227 (3 states rescued, 8 broken).
Of the **25** baseline failures, the full oracle rescues **3** and leaves **22** failing.
Completion time shows nothing either: −2.01 steps on average, median 0, CI [−4.44, +0.25].

The robustness subset (3 additional matched noise schedules, 72 state-schedule pairs) reproduces this:
baseline 0.694, full oracle 0.653.

## Q2 — Can the 64 tokens chosen by D × S produce a task-level rescue?

**No, not measurably.**

D × S − baseline = **−0.033**, CI [−0.083, +0.008], p = 0.289. Rescues **2**, damages **6**.
Remaining-steps difference: mean −0.67, median 0, CI [−3.50, +2.15]. On the robustness subset the paired
difference is exactly **0.000** [0.000, 0.000].

## Q3 — Is D × S a better basis for method design than D repair?

**Not at task level.** D × S − top-D = −0.008, CI [−0.058, +0.042], p = 1.000 (4 vs 5 discordant states).
At this endpoint the two are indistinguishable — which does not contradict G3, where D × S was clearly
better at the *chunk-deviation* endpoint (+0.122 recovery, +0.231 mm). The ranking advantage is real; it
simply does not reach the binary task outcome.

---

## Final decision: **H-NO-HEADROOM**

The pre-registered definition is met: the full `H_real` intervention improves neither success nor
completion time and produces no stable rescue pattern.

It is **not H-HARMFUL** — the point estimates are negative but every CI contains zero and every
McNemar p ≥ 0.23, so "stably reduces success" is not supported either.

## The leading explanation, named in the protocol before the experiment

`H_real` is the future the **baseline** rollout actually reached. Once the repaired chunk emits different
actions, that future is no longer the counterfactual future of those actions: the policy is conditioned on
a future belonging to a *different* action sequence. This is an oracle **future-feature intervention**, not
"the performance of a perfect world model", and the result should not be read as an upper bound on perfect
future prediction. A second, already documented factor: this checkpoint was trained with
`enable_flow_h_t1_scheduled_sampling=false`, so the action head has never been conditioned on a real future.

## What this does not overturn

G3 stands: under stress, `D × S` ranks action-critical future tokens better than `D`, `S`, attention and
`D × attention` (recovery 0.822 vs 0.736 / 0.787 / 0.432 / 0.701), indistinguishable from the oracle
`U × S`, and the executed one-chunk consequence is real (+0.231 mm against 0.51 mm of sampler noise).
What the headroom test adds is that this ~1–2 mm of influence, even spent optimally by an oracle, does not
move a binary task outcome in this setting.

## Breakdowns (for completeness, not for rescue)

By stress: occlusion baseline 0.883 → oracle 0.817; camera shift 0.700 → 0.683.
By source episode: originally-failed episodes 0.281 baseline / 0.281 oracle / 0.219 D × S; originally
successful 0.977 / 0.920 / 0.955.
By task the oracle helps only on `goal t8` (0.850 → 0.950) and hurts on `goal t5`, `object t0`, `spatial t7`.

## Deliverables
`PROTOCOL_HEADROOM.yaml` (with amendment H-A1), `AUDIT_HEADROOM.md` (H AUDIT 0/1/2 + FINAL),
`HEADROOM_RESULTS.csv` (120 paired rows), `RESCUE_CASES.csv`, `SUMMARY_HEADROOM.json`,
`continuation_records.jsonl` (768), `h_feasibility.json`, `h_audit1_equivalence.json`,
`figures/h_headroom.png`, `METHOD_HANDOFF.md`, `code_changes.patch`.
Code: `research/action_critical_reliability_probe/method_entry/`.
