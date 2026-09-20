# METHOD_HANDOFF.md

**Method Design is NOT started.** The headroom test returned **H-NO-HEADROOM**, and the protocol's rule for
that verdict is: report and stop, do not begin designing a side-channel. This file records what a future
decision would have to work with.

## 1. What five rounds established

| round | result |
|---|---|
| Round 1 | `S` is real and non-uniform; attention cannot replace it; oracle `U × S` > `U` |
| G2 | a learned `U` is unnecessary — the free `D = ‖H_pred − h_t‖` predicts it and transfers better |
| DS | `D → U` is length geometry; `D` is *predicted change magnitude*, not reliability; `D × S` ≈ oracle `U × S` |
| G3 | under stress the future-error consequence doubles (0.82 → 1.87/1.70 mm) and `D × S` still ranks best (0.822) |
| **Headroom** | **a one-chunk oracle correction does not change the final task outcome** (0.750 vs 0.792 baseline) |

The mechanism question is answered positively at every level *except* the one that matters for a method:
task outcome.

## 2. The wall, stated precisely

Correcting the future for one chunk — with a perfect oracle, at any token budget from 64 to all 256 —
rescues 3 of 25 failures and breaks 8 successes. The influence the future channel has (~1–2 mm of executed
end-effector motion, against ~0.5 mm of sampler noise on an 85 mm chunk) is real but does not reach a
binary task outcome.

## 3. The single most important unresolved question

**Is the wall the intervention or the channel?**

The oracle future used here belongs to the *baseline* action sequence. Once the repaired chunk acts
differently, that future is no longer its counterfactual future, so the experiment may be measuring
*inconsistent conditioning* rather than *better conditioning*. Distinguishing these needs an intervention
whose future is consistent with the actions it produces — for example a checkpoint trained with
`enable_flow_h_t1_scheduled_sampling=true` (so real futures are in-distribution as conditioning), or an
iterative scheme where the future is re-derived from the action being considered. **Neither was run, and
neither should be started without an explicit decision.**

If that question resolves against the channel, the Action-Critical Future line ends here: the ranking is
correct, the quantity it ranks is too small to matter, and the method has nothing to act on.

## 4. What must not happen next

No new U, no learned U, no new S definition, no new formula (`D²S`, `DS²`, …), no new proxy, no new
attention variant, no new stress family, no failure predictor, no change of token budget, phase or stress
to make the numbers move, and no "H-2" round that repeats this test with tweaked settings. The result is
what it is.

## 5. Re-usable assets

`research/action_critical_reliability_probe/method_entry/`: `h_feasibility.py` (proves the inference
interface can pin the whole continuation noise sequence and inject a repaired future, without patching any
repository file), `h_dump_hpred.py` (live full-precision `H_pred` dump), `h_audit1_equivalence.py`
(first-chunk equivalence against frozen chunks, snapshot restore, schedule reproducibility),
`h_continuation.py` (one-chunk intervention then unmodified continuation), `h_analyze.py` (paired McNemar +
episode bootstrap, the four case counts). Together with the four earlier rounds this reproduces everything
from rollout to task outcome without retraining anything.
