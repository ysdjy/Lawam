# HSU audit — run `hsu_20260921_084852`

Protocol: [`PROTOCOL_HSU.yaml`](PROTOCOL_HSU.yaml). Branch at start:
`research/action-critical-reliability-probe`, HEAD
`ff523d756e42b9be2f2ce1cabf7c1f40ce31c806`. All HSU files are new; the
earlier physical-context probe was left untouched.

## HSU AUDIT 0 — model and matched capacity

Checkpoint: `results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt`.
Loaded backend state-dict SHA-256:
`d6b9533f28144469edddfcad716cb901eede7708e26a742f5f9ce1a64ab7a1f6`.
The base backend has 2,555,179,360 parameters; the Flow Head has 306,405,632.
The VLM, LAM, LaWM, and all base Flow weights are frozen.

| Arm | Projector | LoRA | Total trainable |
| --- | ---: | ---: | ---: |
| Control (`c=0`) | 264,449 | 245,760 | **510,209** |
| Hidden (`c=log(mu_eff/0.95)`) | 264,449 | 245,760 | **510,209** |

LoRA rank is 4 on q/k/v/out of the eight cross-attention DiT blocks (32
projections). The parameter count matches exactly and is below 5M. No base
parameter was found trainable. Source: `AUDIT0_FORWARD.json` in the run output.

## HSU AUDIT 1 — teacher and action legality

Five fixed initial states per physics level passed the gate: NOMINAL 5/5, HIGH
5/5. Effective friction read back as 0.95 and 1.90, respectively. Actions used
the normal seven-dimensional robot action interface and stayed within the
recorded dataset action range (maximum absolute component 0.9375); the teacher
did not teleport, write object pose, or apply force directly. Source:
`TEACHER_GATE.json` and `TEACHER_RESULTS.csv`.

## Forward audit — context consumption

The appended context is encoder token 721, inside the active VLM attention
mask. With all other inputs fixed and the gate temporarily opened, changing
only the physics scalar changed that token's K and V by maxabs 1.4003 and
1.2247 in a LoRA-adapted cross-attention block. Other tokens' K remained
unchanged, and the sampled action changed by maxabs 0.01134. Forward audit
passed. The gate was restored to zero before training.

## HSU AUDIT 2 — zero-init equivalence

Control and Hidden actions were identical at initialization (maxabs 0). Each
differed from unmodified LaWAM by maxabs 0.00099734 with pinned Flow noise.
The original literal `1e-5` criterion failed. The zero-valued extra token is
still projected through the biased `enc_vlm` layer and changes attention
slightly. This deviation was 2.83% of the measured Flow-sampler nuisance
(0.03521) on the same state. Amendment HSU-A5 records the revised criterion;
both arms meet it, but the literal failure remains visible.

## HSU AUDIT 3 — episode split

Sixty distinct episodes were collected, 30 NOMINAL and 30 HIGH. Initial states
0–19 / 20–24 / 25–29 are train / validation / test for both levels. There are
1,212 / 339 / 280 cached samples, with no episode shared across splits.
Successful episodes by split were NOMINAL 20/5/5 and HIGH 19/3/5. The three
unsuccessful HIGH trajectories (init 14, 22, 23) are retained and disclosed;
this is a supervision limitation. Source: `SPLIT.json`.

## Training engineering record

The first attempt exited before update 0 because `load_split` repeatedly
decompressed whole NPZ members for each sample. It now decompresses each
member once per episode; a complete train-plus-validation load used about
2.9 GiB RSS. Training runs inside a systemd scope with a 22 GiB memory limit
to protect the host, which has no swap.

A partial Control run showed a non-zero diagnostic action difference despite
constant context. This was DiT dropout during the diagnostic's train-mode
sampling. The partial artifacts are retained under `invalid_dropout_probe/`;
the diagnostic now uses eval mode for both paired samples, restores train
mode, and the Control run restarted from update 0. The model, data, optimizer,
seed, update count, and LoRA configuration were unchanged. See amendments
HSU-A6 and HSU-A7 in the protocol.

## HSU AUDIT 4 — offline context conditions

Both arms completed all 2,000 updates with the same 1,212 train and 339
validation samples and 510,209 trainable parameters. Best validation updates
were 1,400 for Control (loss 0.016695) and 1,300 for Hidden (loss 0.016175).
At the selected Hidden checkpoint, the paired correct-vs-wrong action maxabs
on the frozen validation mini-set was only 0.00014919 normalized action units;
the Control diagnostic was exactly zero. This indicates very weak context
use, even though the Hidden arm has learned a strong general teacher-action
imitation improvement.

The test set has 280 cached chunks from ten held-out episodes (five per
physics level). Each condition used the same cached inputs and pinned Flow
noise. The `H_SHUFFLED` map preserved five contexts of each level and changed
six of ten episode assignments. Lower action error is better:

| Condition | All chunks L2 | NOMINAL | HIGH |
| --- | ---: | ---: | ---: |
| ORIGINAL | 0.197639 | 0.188349 | 0.202186 |
| C_P | **0.020017** | **0.019340** | **0.020348** |
| H_CORRECT | 0.021811 | 0.022813 | 0.021321 |
| H_WRONG | 0.021794 | 0.022841 | 0.021282 |
| H_SHUFFLED | 0.021800 | 0.022829 | 0.021297 |

The principal signed benefits in action-error units are negative:
`C_P − H_CORRECT = −0.001794`, `H_WRONG − H_CORRECT = −0.000017`, and
`H_SHUFFLED − H_CORRECT = −0.000011`. On HIGH alone, these are `−0.000973`,
`−0.000039`, and `−0.000024`. Equal-weighting the ten episodes gives Control
0.019430 vs Hidden-Correct 0.021657; Hidden-Correct beats Control in only
one of ten episodes. This confirms that chunk count imbalance does not cause
the gate failure. Source: `OFFLINE_METRICS.csv` and `OFFLINE_SUMMARY.json`.

## FINAL HSU AUDIT — stop and verdict

The offline gate **failed**: Hidden-Correct did not beat the matched-capacity
Control, Wrong, or Shuffled conditions. Per the frozen protocol, no simulator
rollouts were run. `ROLLOUT_RESULTS.csv` has a header and no data rows, and
`SUMMARY_HSU.json` records `NOT_RUN_OFFLINE_GATE_FAIL` for all closed-loop
endpoints. The run verdict is **HSU-CAPACITY-ONLY (offline gate)**: adaptation
produced a large imitation gain over Original, but correct oracle physics
context added no independent gain under this specific context-token plus
rank-4 cross-attention LoRA setup. The three unsuccessful HIGH teacher
episodes limit the quality of supervision and are retained in this audit.
This result does not establish that hidden physical state has no value under
other conditioning designs. The protocol requires stopping this direction
here; no architecture or hyperparameter was changed in response to results.
