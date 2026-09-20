# LONG_TERM_PLAN_G2.md — what G2 changes about the long-term plan (PLAN ONLY, nothing implemented)

Status: the learned-U line is **stopped** by its own pre-registered gate (Case G2-A). Nothing in this file
is authorised by the evidence; it exists so the reasoning is on record.

---

## 1. What G2 actually changed

| before G2 | after G2 |
|---|---|
| U is an oracle; unknown whether it exists at deployment | **U is available before execution**: `‖H_pred − h_t‖` predicts it at ρ ≈ 0.92, across unseen tasks and an unseen suite, at zero cost |
| the plan assumed a *Reliability Estimator* module had to be learned | a learned estimator is **not warranted**: it beats the free statistic only on tasks it trained on (testA +0.028) and loses on every held-out task (−0.044 / −0.034 / −0.009) |
| token-norm was an open confound | closed: pure `‖H_pred‖` reaches only 0.55–0.68, far below the change statistic |
| unknown whether a predictor could cheat via token position | measured: a token-index-only predictor reaches ρ ≈ −0.31, so position *is* exploitable — any future design must stay permutation-equivariant, as the v1 MLP provably is |

The architecture sketch in the prior round's `LONG_TERM_PLAN.md` therefore loses one of its boxes: the
"Reliability Estimator → U_hat" stage should be replaced by a **free feature**, not a learned head, unless
some future evidence overturns this.

## 2. The question that survives

Whether the *free* proxy retains the oracle mechanism:

> With the same 64/256 token budget, does `‖H_pred − h_t‖ × S` recover more of the oracle correction of the
> executed end-effector trajectory than `‖H_pred − h_t‖` alone — and how much of the oracle
> `U×S − U` margin does it keep?

It needs: the 90 frozen testC states (S, attention, snapshots, consequence machinery all already exist),
one logged amendment adding `top_change_norm × S` to the pre-registered selector list, and roughly one hour
of execution. It trains nothing. **It is not started without an explicit decision**, because the gate in
front of it returned negative.

## 3. Gates that remain, in order

| gate | question | why it is still ahead |
|---|---|---|
| G2′ | does the *free* U proxy preserve the U×S advantage in the equal-budget causal test? | §2; cheap, uses existing artefacts |
| G3 | with failure / OOD enrichment (occlusion, distractors, camera perturbation, unseen layout), does the reliability signal predict real action error or failure, and does S add to it? | the whole effect measured so far lives inside ≈0.9 mm of executed motion against ≈0.47 mm of sampler noise; without enrichment there is nothing to gate |
| G4 | can S be obtained online at all (amortised sensitivity, one-shot gradient proxy)? | S currently costs ~2,300 forwards per state; **deliberately untouched this round** so that a failure could be attributed to U rather than to two changed variables at once |
| G1 (unchanged from the prior round) | does a checkpoint trained with `scheduled_sampling=true` have a larger future-utilisation budget? | still the decisive experiment for whether any of this can matter for control |

## 4. What must never be inferred from G2

- That reliability information improves robot performance — nothing about performance was measured.
- That `‖H_pred − h_t‖` *is* U — it is a strong rank proxy (ρ ≈ 0.92), not a calibrated magnitude.
- That the learned MLP "failed" in the sense of being badly built — it reaches ρ 0.96 on unseen resets;
  it simply does not beat something free on unseen tasks, which is the only comparison that matters.
- That a bigger model would fix it. That escalation requires a new audit entry and, on present evidence,
  a reason why more capacity would transfer better than a statistic with no parameters at all.

## 5. Re-usable assets

`research/action_critical_reliability_probe/learned_u/`: `g2_extract_u.py` (U dataset built on the prior
round's own encoder and U definition), `g2_dataset.py` (episode-bootstrapped state-level metrics),
`g2_baselines.py` (the free baselines, including the corrected random floor and the token-index control),
`g2_train_u.py` / `g2_eval_u.py` (shared-token MLP + permutation audit), `g2_make_splits.py`
(leakage-asserted episode/task splits), `g2_audit0.py` (prior-run reproduction + LaWAM weight hash).
