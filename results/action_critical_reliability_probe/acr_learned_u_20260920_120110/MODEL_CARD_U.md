# MODEL_CARD_U.md — shared-token U predictor (G2, v1)

**This model is a diagnostic instrument, not a component.** It was trained to answer one question — "is
per-token future-prediction unreliability predictable before execution?" — and the answer it produced is
that a *free statistic beats it on unseen tasks*. It must not be deployed, and no LaWAM parameter was
touched to produce it.

## Identity
| | |
|---|---|
| run | `acr_learned_u_20260920_120110` |
| files | `models/u_mlp_seed{0,1,2}.pt` (5.6 MB each), `models/feature_norm.npz` |
| in the git repository? | **no** — the three weight files are kept locally only, to keep the repo free of binary blobs. They are reproduced exactly by `g2_train_u.py` in ~2 s per seed from the extracted U dataset (seeds 0/1/2 fixed); the evaluation products that depend on them (`METRICS_U.csv`, `summary_learned_u.json`, the figure) *are* committed |
| training code | `research/action_critical_reliability_probe/learned_u/scripts/g2_train_u.py` |
| parameters | **1.46 M** per seed (budget 1–3 M) |
| training time | 1–2 s per seed (36–43 epochs, early-stopped) |
| LaWAM checkpoint | untouched; `state_dict` sha256 `37a53b8c799c39725a18900eeaa687d1d2cebc24fca28d8e1f2881ca2870b523`, verified in G2 AUDIT 0; size/mtime identical before and after training (`TRAIN_LOG.jsonl`) |

## Inputs — deployment-available only
`x_j = [h_t_j (768), H_pred_j (768), H_pred_j − h_t_j (768), z (32)] → 2336 dims`, standardised with
train-split statistics. **No token index, no positional id, no instruction embedding, no task id, and
nothing at or after t+H.** `H_real` reaches the model only through the scalar training label.

The exclusion of a token index is deliberate and consequential: the action head reads the future as an
unordered bag (permuting future tokens changes the action by ~4e-7), and a token-index-only predictor
reaches Spearman −0.31 against U, so a positional feature would have been a genuine shortcut.

## Architecture
`Linear(2336→512) → LayerNorm → GELU → Linear(512→512) → LayerNorm → GELU → Linear(512→1)`, weights shared
across all 256 tokens and applied independently → **permutation-equivariant by construction**, measured at
exactly **0.0** deviation on all 250 states.

## Training
Target `log(U_l2 + 1e-6)`; Huber loss (δ=1); AdamW, lr 3e-4, weight decay 1e-4; batches of 8 **states**
(tokens of a state are never split across batches); early stopping on validation Spearman, patience 20;
3 seeds, predictions averaged in log space. No ranking loss — pre-registered, so that "U is predictable"
is not confounded with "a top-64 selector can be fitted".

Data: 72 train states / 18,432 tokens from 4 tasks (episodes 0–5), 24 validation states (episodes 6–7).
Splits are episode-disjoint and task-disjoint by construction (`DATA_SPLIT.md`).

## Measured behaviour

| split | Spearman vs oracle U | top-64 recall | AUROC (top quartile) |
|---|---|---|---|
| train | 0.987 | 0.937 | — |
| val | 0.960 | — | — |
| testA — unseen resets, seen task | **0.959** | 0.857 | — |
| testB1 — unseen task, seen suite | **0.901** | 0.793 | — |
| testB2 — unseen task + suite | **0.840** | 0.758 | — |
| testC — prior round's 3 tasks | **0.909** | 0.821 | 0.960 |

Against the free baseline `‖H_pred − h_t‖` (paired, episode-bootstrapped):
**testA +0.028 [+0.020, +0.039] (wins), testB1 −0.044 [−0.050, −0.037] (loses), testB2 −0.034
[−0.058, −0.011] (loses), testC −0.009 [−0.026, +0.006] (ties).**
On testC the MLP is nevertheless better at the head of the ranking (recall@64 0.821 vs 0.762,
NDCG@64 0.932 vs 0.875, AUROC 0.960 vs 0.941).

## Known limitations
1. **Does not transfer across tasks.** The advantage over a zero-parameter statistic exists only on tasks
   seen in training. This is the finding, not a defect to be tuned away.
2. Mild overfitting (train 0.987 vs val 0.959), stable across 3 seeds — a feature-set limit, not an
   optimisation artefact.
3. Trained on 4 tasks from 2 suites, all in-distribution and (except one episode) successful; no OOD,
   occlusion or distractor data.
4. The label is an oracle built from a single rendered future frame; its own render-noise floor is ~16×
   below the signal, but 2 of 160 states have contact-rich replays whose floor is overestimated.
5. `testB2` is approach-only (10 states), because its task is a non-prehensile push with no grasp phase.

## Intended and forbidden use
Intended: reproducing the G2 analysis, and as a baseline if the question is ever revisited with a different
feature set. Forbidden: any use inside a policy, any claim of improved robot performance, any reporting of
the train/testA numbers without the held-out-task numbers beside them.
