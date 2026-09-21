# Oracle Hidden-State Utilization Test — final report

Run: `hsu_20260921_084852` | Task B: `libero_goal/5`, “push the plate to the front
of the stove” | Physics: plate/table effective friction NOMINAL 0.95, HIGH 1.90 |
Verdict: **HSU-CAPACITY-ONLY (offline gate)**.

The teacher gate passed at **5/5 NOMINAL and 5/5 HIGH** with legal robot
actions. In the full 30+30 episode dataset it succeeded on 30/30 NOMINAL and
27/30 HIGH; three unsuccessful HIGH trajectories were kept and disclosed.
The Control and Hidden arms each trained **264,449 projector + 245,760 rank-4
cross-attention LoRA = 510,209 parameters**, with identical samples, order,
seed, optimizer, and 2,000 updates. The base 306M Flow Head and the rest of
LaWAM stayed frozen. The LoRA targets were q/k/v/out in eight cross-attention
DiT blocks.

At initialization, Control and Hidden actions were identical (maxabs 0).
Each differed from Original by maxabs 0.00099734 with pinned Flow noise due
to the biased `enc_vlm` projection of the added zero token. The original
literal `1e-5` equivalence test therefore failed; this was disclosed before
training and amendment HSU-A5 accepted a deviation below 10% of the
same-state Flow-sampler nuisance (measured 2.83%).

## Held-out offline result

Mean teacher-action L2 per normalized action element; lower is better.
The same ten test episodes and pinned Flow noise were used throughout.

| Condition | All chunks | NOMINAL | HIGH |
| --- | ---: | ---: | ---: |
| Original LaWAM | 0.197639 | 0.188349 | 0.202186 |
| Control adaptation (`C_P`) | **0.020017** | **0.019340** | **0.020348** |
| Hidden-Correct | 0.021811 | 0.022813 | 0.021321 |
| Hidden-Wrong | 0.021794 | 0.022841 | 0.021282 |
| Hidden-Shuffled | 0.021800 | 0.022829 | 0.021297 |

`C_P error − Hidden-Correct error = −0.001794` overall and `−0.000973`
on HIGH. `Hidden-Wrong error − Hidden-Correct error = −0.000017` overall and
`−0.000039` on HIGH. Correct context also loses to Shuffled by `0.000011`
overall. Episode-equal weighting confirms the result: Control 0.019430,
Hidden-Correct 0.021657; Hidden-Correct wins just **1/10** paired episodes.
At the selected Hidden checkpoint, swapping correct/wrong context changed
the frozen validation mini-set's sampled action by maxabs **0.00014919**
normalized units, indicating very weak context sensitivity. The shuffled
mapping was fixed before offline scoring, preserved the 5/5 marginal, and
mismatched six of ten episodes.

## Closed-loop result and decision

| Physics | Original | Control | Hidden-Correct | Hidden-Wrong |
| --- | --- | --- | --- | --- |
| NOMINAL | Not run | Not run | Not run | Not run |
| HIGH | Not run | Not run | Not run | Not run |

The pre-registered offline gate required Hidden-Correct to beat the
matched-capacity Control and wrong/shuffled context, especially on HIGH.
It failed, so the protocol forbids running closed-loop pilots. The prior
physical-context probe's Original policy rates (NOMINAL 20/20, HIGH 0/20)
are historical background only and were not repeated or counted as HSU
rollouts. Closed-loop success differences and contact/action-distribution
effects are **unmeasured** in this run.

The large gain over Original is attributable to the adaptation training
capacity and teacher imitation under this test. Correct oracle friction did
not add an independent offline benefit, so the verdict is
**HSU-CAPACITY-ONLY (offline gate)**. This constrains the tested
context-token plus rank-4 matched-LoRA design; it does not show that hidden
physics is generally useless. The run stops here, without further model or
controller changes.

Detailed checks: [`AUDIT_HSU.md`](AUDIT_HSU.md). Machine-readable results:
`results/hidden_state_utilization_probe/hsu_20260921_084852/`.
