# DATA_SPLIT.md — Learned-U (G2)

Generated mechanically from `PROTOCOL_G2.yaml:splits`. The **episode** is the minimum independent
unit; tokens and states are never split randomly. Once written this file is immutable: no split may
be changed after any result is seen.

| split | meaning | states | episodes | tokens |
|---|---|---|---|---|
| `train` | training-pool tasks, episodes 0-5 | 72 | 24 | 18,432 |
| `val` | training-pool tasks, episodes 6-7 (model selection / early stopping) | 24 | 8 | 6,144 |
| `testA` | training-pool tasks, episodes 8-9 — unseen resets of a SEEN task | 24 | 8 | 6,144 |
| `testB1` | libero_spatial t9 — unseen TASK, seen suite | 30 | 10 | 7,680 |
| `testB2` | libero_goal t5 — unseen task AND unseen suite (non-prehensile push) | 10 | 10 | 2,560 |
| `testC` | prior run's 3 tasks — unseen, and already carry frozen S / attention / consequences | 90 | 30 | 23,040 |

## Leakage checks

- `episode_disjoint_train_val`: **PASS**
- `episode_disjoint_train_testA`: **PASS**
- `episode_disjoint_val_testA`: **PASS**
- `no_state_id_in_two_splits`: **PASS**
- `task_disjoint_train_testB1`: **PASS**
- `task_disjoint_train_testB2`: **PASS**
- `task_disjoint_train_testC`: **PASS**

## Full listing

| suite | task | episode | phase | split | success | state_id |
|---|---|---|---|---|---|---|
| libero_object | 3 | 8 | approach | testA | True | `libero_object_t03_ep008_s0024_approach` |
| libero_object | 3 | 8 | pre_grasp | testA | True | `libero_object_t03_ep008_s0040_pre_grasp` |
| libero_object | 3 | 8 | manipulation | testA | True | `libero_object_t03_ep008_s0064_manipulation` |
| libero_object | 3 | 9 | approach | testA | True | `libero_object_t03_ep009_s0024_approach` |
| libero_object | 3 | 9 | pre_grasp | testA | True | `libero_object_t03_ep009_s0048_pre_grasp` |
| libero_object | 3 | 9 | manipulation | testA | True | `libero_object_t03_ep009_s0072_manipulation` |
| libero_object | 6 | 8 | approach | testA | True | `libero_object_t06_ep008_s0024_approach` |
| libero_object | 6 | 8 | pre_grasp | testA | True | `libero_object_t06_ep008_s0056_pre_grasp` |
| libero_object | 6 | 8 | manipulation | testA | True | `libero_object_t06_ep008_s0072_manipulation` |
| libero_object | 6 | 9 | approach | testA | True | `libero_object_t06_ep009_s0024_approach` |
| libero_object | 6 | 9 | pre_grasp | testA | True | `libero_object_t06_ep009_s0048_pre_grasp` |
| libero_object | 6 | 9 | manipulation | testA | True | `libero_object_t06_ep009_s0072_manipulation` |
| libero_spatial | 0 | 8 | approach | testA | True | `libero_spatial_t00_ep008_s0016_approach` |
| libero_spatial | 0 | 8 | pre_grasp | testA | True | `libero_spatial_t00_ep008_s0032_pre_grasp` |
| libero_spatial | 0 | 8 | manipulation | testA | True | `libero_spatial_t00_ep008_s0056_manipulation` |
| libero_spatial | 0 | 9 | approach | testA | True | `libero_spatial_t00_ep009_s0024_approach` |
| libero_spatial | 0 | 9 | pre_grasp | testA | True | `libero_spatial_t00_ep009_s0040_pre_grasp` |
| libero_spatial | 0 | 9 | manipulation | testA | True | `libero_spatial_t00_ep009_s0056_manipulation` |
| libero_spatial | 7 | 8 | approach | testA | True | `libero_spatial_t07_ep008_s0016_approach` |
| libero_spatial | 7 | 8 | pre_grasp | testA | True | `libero_spatial_t07_ep008_s0032_pre_grasp` |
| libero_spatial | 7 | 8 | manipulation | testA | True | `libero_spatial_t07_ep008_s0064_manipulation` |
| libero_spatial | 7 | 9 | approach | testA | True | `libero_spatial_t07_ep009_s0016_approach` |
| libero_spatial | 7 | 9 | pre_grasp | testA | True | `libero_spatial_t07_ep009_s0032_pre_grasp` |
| libero_spatial | 7 | 9 | manipulation | testA | True | `libero_spatial_t07_ep009_s0056_manipulation` |
| libero_spatial | 9 | 0 | approach | testB1 | True | `libero_spatial_t09_ep000_s0024_approach` |
| libero_spatial | 9 | 0 | pre_grasp | testB1 | True | `libero_spatial_t09_ep000_s0040_pre_grasp` |
| libero_spatial | 9 | 0 | manipulation | testB1 | True | `libero_spatial_t09_ep000_s0064_manipulation` |
| libero_spatial | 9 | 1 | approach | testB1 | True | `libero_spatial_t09_ep001_s0024_approach` |
| libero_spatial | 9 | 1 | pre_grasp | testB1 | True | `libero_spatial_t09_ep001_s0048_pre_grasp` |
| libero_spatial | 9 | 1 | manipulation | testB1 | True | `libero_spatial_t09_ep001_s0072_manipulation` |
| libero_spatial | 9 | 2 | approach | testB1 | True | `libero_spatial_t09_ep002_s0024_approach` |
| libero_spatial | 9 | 2 | pre_grasp | testB1 | True | `libero_spatial_t09_ep002_s0048_pre_grasp` |
| libero_spatial | 9 | 2 | manipulation | testB1 | True | `libero_spatial_t09_ep002_s0064_manipulation` |
| libero_spatial | 9 | 3 | approach | testB1 | True | `libero_spatial_t09_ep003_s0024_approach` |
| libero_spatial | 9 | 3 | pre_grasp | testB1 | True | `libero_spatial_t09_ep003_s0040_pre_grasp` |
| libero_spatial | 9 | 3 | manipulation | testB1 | True | `libero_spatial_t09_ep003_s0064_manipulation` |
| libero_spatial | 9 | 4 | approach | testB1 | True | `libero_spatial_t09_ep004_s0024_approach` |
| libero_spatial | 9 | 4 | pre_grasp | testB1 | True | `libero_spatial_t09_ep004_s0040_pre_grasp` |
| libero_spatial | 9 | 4 | manipulation | testB1 | True | `libero_spatial_t09_ep004_s0064_manipulation` |
| libero_spatial | 9 | 5 | approach | testB1 | True | `libero_spatial_t09_ep005_s0024_approach` |
| libero_spatial | 9 | 5 | pre_grasp | testB1 | True | `libero_spatial_t09_ep005_s0048_pre_grasp` |
| libero_spatial | 9 | 5 | manipulation | testB1 | True | `libero_spatial_t09_ep005_s0064_manipulation` |
| libero_spatial | 9 | 6 | approach | testB1 | True | `libero_spatial_t09_ep006_s0024_approach` |
| libero_spatial | 9 | 6 | pre_grasp | testB1 | True | `libero_spatial_t09_ep006_s0048_pre_grasp` |
| libero_spatial | 9 | 6 | manipulation | testB1 | True | `libero_spatial_t09_ep006_s0072_manipulation` |
| libero_spatial | 9 | 7 | approach | testB1 | True | `libero_spatial_t09_ep007_s0024_approach` |
| libero_spatial | 9 | 7 | pre_grasp | testB1 | True | `libero_spatial_t09_ep007_s0048_pre_grasp` |
| libero_spatial | 9 | 7 | manipulation | testB1 | True | `libero_spatial_t09_ep007_s0064_manipulation` |
| libero_spatial | 9 | 8 | approach | testB1 | True | `libero_spatial_t09_ep008_s0024_approach` |
| libero_spatial | 9 | 8 | pre_grasp | testB1 | True | `libero_spatial_t09_ep008_s0048_pre_grasp` |
| libero_spatial | 9 | 8 | manipulation | testB1 | True | `libero_spatial_t09_ep008_s0064_manipulation` |
| libero_spatial | 9 | 9 | approach | testB1 | True | `libero_spatial_t09_ep009_s0024_approach` |
| libero_spatial | 9 | 9 | pre_grasp | testB1 | True | `libero_spatial_t09_ep009_s0048_pre_grasp` |
| libero_spatial | 9 | 9 | manipulation | testB1 | True | `libero_spatial_t09_ep009_s0072_manipulation` |
| libero_goal | 5 | 0 | approach | testB2 | True | `libero_goal_t05_ep000_s0040_approach` |
| libero_goal | 5 | 1 | approach | testB2 | True | `libero_goal_t05_ep001_s0048_approach` |
| libero_goal | 5 | 2 | approach | testB2 | True | `libero_goal_t05_ep002_s0024_approach` |
| libero_goal | 5 | 3 | approach | testB2 | True | `libero_goal_t05_ep003_s0088_approach` |
| libero_goal | 5 | 4 | approach | testB2 | True | `libero_goal_t05_ep004_s0048_approach` |
| libero_goal | 5 | 5 | approach | testB2 | True | `libero_goal_t05_ep005_s0032_approach` |
| libero_goal | 5 | 6 | approach | testB2 | True | `libero_goal_t05_ep006_s0024_approach` |
| libero_goal | 5 | 7 | approach | testB2 | True | `libero_goal_t05_ep007_s0024_approach` |
| libero_goal | 5 | 8 | approach | testB2 | True | `libero_goal_t05_ep008_s0040_approach` |
| libero_goal | 5 | 9 | approach | testB2 | False | `libero_goal_t05_ep009_s0120_approach` |
| libero_goal | 8 | 0 | approach | testC | True | `libero_goal_t08_ep000_s0016_approach` |
| libero_goal | 8 | 0 | pre_grasp | testC | True | `libero_goal_t08_ep000_s0032_pre_grasp` |
| libero_goal | 8 | 0 | manipulation | testC | True | `libero_goal_t08_ep000_s0056_manipulation` |
| libero_goal | 8 | 1 | approach | testC | True | `libero_goal_t08_ep001_s0016_approach` |
| libero_goal | 8 | 1 | pre_grasp | testC | True | `libero_goal_t08_ep001_s0032_pre_grasp` |
| libero_goal | 8 | 1 | manipulation | testC | True | `libero_goal_t08_ep001_s0056_manipulation` |
| libero_goal | 8 | 2 | approach | testC | True | `libero_goal_t08_ep002_s0016_approach` |
| libero_goal | 8 | 2 | pre_grasp | testC | True | `libero_goal_t08_ep002_s0032_pre_grasp` |
| libero_goal | 8 | 2 | manipulation | testC | True | `libero_goal_t08_ep002_s0056_manipulation` |
| libero_goal | 8 | 3 | approach | testC | True | `libero_goal_t08_ep003_s0016_approach` |
| libero_goal | 8 | 3 | pre_grasp | testC | True | `libero_goal_t08_ep003_s0032_pre_grasp` |
| libero_goal | 8 | 3 | manipulation | testC | True | `libero_goal_t08_ep003_s0048_manipulation` |
| libero_goal | 8 | 4 | approach | testC | True | `libero_goal_t08_ep004_s0016_approach` |
| libero_goal | 8 | 4 | pre_grasp | testC | True | `libero_goal_t08_ep004_s0032_pre_grasp` |
| libero_goal | 8 | 4 | manipulation | testC | True | `libero_goal_t08_ep004_s0056_manipulation` |
| libero_goal | 8 | 5 | approach | testC | True | `libero_goal_t08_ep005_s0016_approach` |
| libero_goal | 8 | 5 | pre_grasp | testC | True | `libero_goal_t08_ep005_s0032_pre_grasp` |
| libero_goal | 8 | 5 | manipulation | testC | True | `libero_goal_t08_ep005_s0056_manipulation` |
| libero_goal | 8 | 6 | approach | testC | True | `libero_goal_t08_ep006_s0016_approach` |
| libero_goal | 8 | 6 | pre_grasp | testC | True | `libero_goal_t08_ep006_s0032_pre_grasp` |
| libero_goal | 8 | 6 | manipulation | testC | True | `libero_goal_t08_ep006_s0056_manipulation` |
| libero_goal | 8 | 7 | approach | testC | True | `libero_goal_t08_ep007_s0024_approach` |
| libero_goal | 8 | 7 | pre_grasp | testC | True | `libero_goal_t08_ep007_s0040_pre_grasp` |
| libero_goal | 8 | 7 | manipulation | testC | True | `libero_goal_t08_ep007_s0056_manipulation` |
| libero_goal | 8 | 8 | approach | testC | True | `libero_goal_t08_ep008_s0016_approach` |
| libero_goal | 8 | 8 | pre_grasp | testC | True | `libero_goal_t08_ep008_s0032_pre_grasp` |
| libero_goal | 8 | 8 | manipulation | testC | True | `libero_goal_t08_ep008_s0056_manipulation` |
| libero_goal | 8 | 9 | approach | testC | True | `libero_goal_t08_ep009_s0016_approach` |
| libero_goal | 8 | 9 | pre_grasp | testC | True | `libero_goal_t08_ep009_s0032_pre_grasp` |
| libero_goal | 8 | 9 | manipulation | testC | True | `libero_goal_t08_ep009_s0056_manipulation` |
| libero_object | 0 | 0 | approach | testC | True | `libero_object_t00_ep000_s0024_approach` |
| libero_object | 0 | 0 | pre_grasp | testC | True | `libero_object_t00_ep000_s0040_pre_grasp` |
| libero_object | 0 | 0 | manipulation | testC | True | `libero_object_t00_ep000_s0056_manipulation` |
| libero_object | 0 | 1 | approach | testC | True | `libero_object_t00_ep001_s0024_approach` |
| libero_object | 0 | 1 | pre_grasp | testC | True | `libero_object_t00_ep001_s0048_pre_grasp` |
| libero_object | 0 | 1 | manipulation | testC | True | `libero_object_t00_ep001_s0064_manipulation` |
| libero_object | 0 | 2 | approach | testC | True | `libero_object_t00_ep002_s0024_approach` |
| libero_object | 0 | 2 | pre_grasp | testC | True | `libero_object_t00_ep002_s0048_pre_grasp` |
| libero_object | 0 | 2 | manipulation | testC | True | `libero_object_t00_ep002_s0064_manipulation` |
| libero_object | 0 | 3 | approach | testC | True | `libero_object_t00_ep003_s0024_approach` |
| libero_object | 0 | 3 | pre_grasp | testC | True | `libero_object_t00_ep003_s0048_pre_grasp` |
| libero_object | 0 | 3 | manipulation | testC | True | `libero_object_t00_ep003_s0064_manipulation` |
| libero_object | 0 | 4 | approach | testC | True | `libero_object_t00_ep004_s0024_approach` |
| libero_object | 0 | 4 | pre_grasp | testC | True | `libero_object_t00_ep004_s0048_pre_grasp` |
| libero_object | 0 | 4 | manipulation | testC | True | `libero_object_t00_ep004_s0064_manipulation` |
| libero_object | 0 | 5 | approach | testC | True | `libero_object_t00_ep005_s0024_approach` |
| libero_object | 0 | 5 | pre_grasp | testC | True | `libero_object_t00_ep005_s0048_pre_grasp` |
| libero_object | 0 | 5 | manipulation | testC | True | `libero_object_t00_ep005_s0064_manipulation` |
| libero_object | 0 | 6 | approach | testC | True | `libero_object_t00_ep006_s0024_approach` |
| libero_object | 0 | 6 | pre_grasp | testC | True | `libero_object_t00_ep006_s0048_pre_grasp` |
| libero_object | 0 | 6 | manipulation | testC | True | `libero_object_t00_ep006_s0064_manipulation` |
| libero_object | 0 | 7 | approach | testC | True | `libero_object_t00_ep007_s0024_approach` |
| libero_object | 0 | 7 | pre_grasp | testC | True | `libero_object_t00_ep007_s0048_pre_grasp` |
| libero_object | 0 | 7 | manipulation | testC | True | `libero_object_t00_ep007_s0064_manipulation` |
| libero_object | 0 | 8 | approach | testC | True | `libero_object_t00_ep008_s0024_approach` |
| libero_object | 0 | 8 | pre_grasp | testC | True | `libero_object_t00_ep008_s0040_pre_grasp` |
| libero_object | 0 | 8 | manipulation | testC | True | `libero_object_t00_ep008_s0056_manipulation` |
| libero_object | 0 | 9 | approach | testC | True | `libero_object_t00_ep009_s0024_approach` |
| libero_object | 0 | 9 | pre_grasp | testC | True | `libero_object_t00_ep009_s0048_pre_grasp` |
| libero_object | 0 | 9 | manipulation | testC | True | `libero_object_t00_ep009_s0064_manipulation` |
| libero_spatial | 2 | 0 | approach | testC | True | `libero_spatial_t02_ep000_s0024_approach` |
| libero_spatial | 2 | 0 | pre_grasp | testC | True | `libero_spatial_t02_ep000_s0040_pre_grasp` |
| libero_spatial | 2 | 0 | manipulation | testC | True | `libero_spatial_t02_ep000_s0064_manipulation` |
| libero_spatial | 2 | 1 | approach | testC | True | `libero_spatial_t02_ep001_s0016_approach` |
| libero_spatial | 2 | 1 | pre_grasp | testC | True | `libero_spatial_t02_ep001_s0032_pre_grasp` |
| libero_spatial | 2 | 1 | manipulation | testC | True | `libero_spatial_t02_ep001_s0056_manipulation` |
| libero_spatial | 2 | 2 | approach | testC | True | `libero_spatial_t02_ep002_s0016_approach` |
| libero_spatial | 2 | 2 | pre_grasp | testC | True | `libero_spatial_t02_ep002_s0032_pre_grasp` |
| libero_spatial | 2 | 2 | manipulation | testC | True | `libero_spatial_t02_ep002_s0056_manipulation` |
| libero_spatial | 2 | 3 | approach | testC | True | `libero_spatial_t02_ep003_s0016_approach` |
| libero_spatial | 2 | 3 | pre_grasp | testC | True | `libero_spatial_t02_ep003_s0032_pre_grasp` |
| libero_spatial | 2 | 3 | manipulation | testC | True | `libero_spatial_t02_ep003_s0056_manipulation` |
| libero_spatial | 2 | 4 | approach | testC | True | `libero_spatial_t02_ep004_s0016_approach` |
| libero_spatial | 2 | 4 | pre_grasp | testC | True | `libero_spatial_t02_ep004_s0032_pre_grasp` |
| libero_spatial | 2 | 4 | manipulation | testC | True | `libero_spatial_t02_ep004_s0056_manipulation` |
| libero_spatial | 2 | 5 | approach | testC | True | `libero_spatial_t02_ep005_s0016_approach` |
| libero_spatial | 2 | 5 | pre_grasp | testC | True | `libero_spatial_t02_ep005_s0032_pre_grasp` |
| libero_spatial | 2 | 5 | manipulation | testC | True | `libero_spatial_t02_ep005_s0056_manipulation` |
| libero_spatial | 2 | 6 | approach | testC | True | `libero_spatial_t02_ep006_s0016_approach` |
| libero_spatial | 2 | 6 | pre_grasp | testC | True | `libero_spatial_t02_ep006_s0032_pre_grasp` |
| libero_spatial | 2 | 6 | manipulation | testC | True | `libero_spatial_t02_ep006_s0056_manipulation` |
| libero_spatial | 2 | 7 | approach | testC | True | `libero_spatial_t02_ep007_s0016_approach` |
| libero_spatial | 2 | 7 | pre_grasp | testC | True | `libero_spatial_t02_ep007_s0032_pre_grasp` |
| libero_spatial | 2 | 7 | manipulation | testC | True | `libero_spatial_t02_ep007_s0056_manipulation` |
| libero_spatial | 2 | 8 | approach | testC | True | `libero_spatial_t02_ep008_s0016_approach` |
| libero_spatial | 2 | 8 | pre_grasp | testC | True | `libero_spatial_t02_ep008_s0032_pre_grasp` |
| libero_spatial | 2 | 8 | manipulation | testC | True | `libero_spatial_t02_ep008_s0056_manipulation` |
| libero_spatial | 2 | 9 | approach | testC | True | `libero_spatial_t02_ep009_s0016_approach` |
| libero_spatial | 2 | 9 | pre_grasp | testC | True | `libero_spatial_t02_ep009_s0032_pre_grasp` |
| libero_spatial | 2 | 9 | manipulation | testC | True | `libero_spatial_t02_ep009_s0056_manipulation` |
| libero_object | 3 | 0 | approach | train | True | `libero_object_t03_ep000_s0024_approach` |
| libero_object | 3 | 0 | pre_grasp | train | True | `libero_object_t03_ep000_s0040_pre_grasp` |
| libero_object | 3 | 0 | manipulation | train | True | `libero_object_t03_ep000_s0064_manipulation` |
| libero_object | 3 | 1 | approach | train | True | `libero_object_t03_ep001_s0024_approach` |
| libero_object | 3 | 1 | pre_grasp | train | True | `libero_object_t03_ep001_s0040_pre_grasp` |
| libero_object | 3 | 1 | manipulation | train | True | `libero_object_t03_ep001_s0064_manipulation` |
| libero_object | 3 | 2 | approach | train | True | `libero_object_t03_ep002_s0016_approach` |
| libero_object | 3 | 2 | pre_grasp | train | True | `libero_object_t03_ep002_s0040_pre_grasp` |
| libero_object | 3 | 2 | manipulation | train | True | `libero_object_t03_ep002_s0064_manipulation` |
| libero_object | 3 | 3 | approach | train | True | `libero_object_t03_ep003_s0024_approach` |
| libero_object | 3 | 3 | pre_grasp | train | True | `libero_object_t03_ep003_s0048_pre_grasp` |
| libero_object | 3 | 3 | manipulation | train | True | `libero_object_t03_ep003_s0080_manipulation` |
| libero_object | 3 | 4 | approach | train | True | `libero_object_t03_ep004_s0024_approach` |
| libero_object | 3 | 4 | pre_grasp | train | True | `libero_object_t03_ep004_s0048_pre_grasp` |
| libero_object | 3 | 4 | manipulation | train | True | `libero_object_t03_ep004_s0072_manipulation` |
| libero_object | 3 | 5 | approach | train | True | `libero_object_t03_ep005_s0024_approach` |
| libero_object | 3 | 5 | pre_grasp | train | True | `libero_object_t03_ep005_s0048_pre_grasp` |
| libero_object | 3 | 5 | manipulation | train | True | `libero_object_t03_ep005_s0072_manipulation` |
| libero_object | 6 | 0 | approach | train | True | `libero_object_t06_ep000_s0024_approach` |
| libero_object | 6 | 0 | pre_grasp | train | True | `libero_object_t06_ep000_s0048_pre_grasp` |
| libero_object | 6 | 0 | manipulation | train | True | `libero_object_t06_ep000_s0072_manipulation` |
| libero_object | 6 | 1 | approach | train | True | `libero_object_t06_ep001_s0024_approach` |
| libero_object | 6 | 1 | pre_grasp | train | True | `libero_object_t06_ep001_s0056_pre_grasp` |
| libero_object | 6 | 1 | manipulation | train | True | `libero_object_t06_ep001_s0072_manipulation` |
| libero_object | 6 | 2 | approach | train | True | `libero_object_t06_ep002_s0024_approach` |
| libero_object | 6 | 2 | pre_grasp | train | True | `libero_object_t06_ep002_s0048_pre_grasp` |
| libero_object | 6 | 2 | manipulation | train | True | `libero_object_t06_ep002_s0072_manipulation` |
| libero_object | 6 | 3 | approach | train | True | `libero_object_t06_ep003_s0024_approach` |
| libero_object | 6 | 3 | pre_grasp | train | True | `libero_object_t06_ep003_s0056_pre_grasp` |
| libero_object | 6 | 3 | manipulation | train | True | `libero_object_t06_ep003_s0072_manipulation` |
| libero_object | 6 | 4 | approach | train | True | `libero_object_t06_ep004_s0024_approach` |
| libero_object | 6 | 4 | pre_grasp | train | True | `libero_object_t06_ep004_s0040_pre_grasp` |
| libero_object | 6 | 4 | manipulation | train | True | `libero_object_t06_ep004_s0064_manipulation` |
| libero_object | 6 | 5 | approach | train | True | `libero_object_t06_ep005_s0024_approach` |
| libero_object | 6 | 5 | pre_grasp | train | True | `libero_object_t06_ep005_s0048_pre_grasp` |
| libero_object | 6 | 5 | manipulation | train | True | `libero_object_t06_ep005_s0064_manipulation` |
| libero_spatial | 0 | 0 | approach | train | True | `libero_spatial_t00_ep000_s0016_approach` |
| libero_spatial | 0 | 0 | pre_grasp | train | True | `libero_spatial_t00_ep000_s0032_pre_grasp` |
| libero_spatial | 0 | 0 | manipulation | train | True | `libero_spatial_t00_ep000_s0056_manipulation` |
| libero_spatial | 0 | 1 | approach | train | True | `libero_spatial_t00_ep001_s0016_approach` |
| libero_spatial | 0 | 1 | pre_grasp | train | True | `libero_spatial_t00_ep001_s0032_pre_grasp` |
| libero_spatial | 0 | 1 | manipulation | train | True | `libero_spatial_t00_ep001_s0056_manipulation` |
| libero_spatial | 0 | 2 | approach | train | True | `libero_spatial_t00_ep002_s0024_approach` |
| libero_spatial | 0 | 2 | pre_grasp | train | True | `libero_spatial_t00_ep002_s0040_pre_grasp` |
| libero_spatial | 0 | 2 | manipulation | train | True | `libero_spatial_t00_ep002_s0056_manipulation` |
| libero_spatial | 0 | 3 | approach | train | True | `libero_spatial_t00_ep003_s0024_approach` |
| libero_spatial | 0 | 3 | pre_grasp | train | True | `libero_spatial_t00_ep003_s0040_pre_grasp` |
| libero_spatial | 0 | 3 | manipulation | train | True | `libero_spatial_t00_ep003_s0064_manipulation` |
| libero_spatial | 0 | 4 | approach | train | True | `libero_spatial_t00_ep004_s0016_approach` |
| libero_spatial | 0 | 4 | pre_grasp | train | True | `libero_spatial_t00_ep004_s0032_pre_grasp` |
| libero_spatial | 0 | 4 | manipulation | train | True | `libero_spatial_t00_ep004_s0056_manipulation` |
| libero_spatial | 0 | 5 | approach | train | True | `libero_spatial_t00_ep005_s0016_approach` |
| libero_spatial | 0 | 5 | pre_grasp | train | True | `libero_spatial_t00_ep005_s0032_pre_grasp` |
| libero_spatial | 0 | 5 | manipulation | train | True | `libero_spatial_t00_ep005_s0056_manipulation` |
| libero_spatial | 7 | 0 | approach | train | True | `libero_spatial_t07_ep000_s0016_approach` |
| libero_spatial | 7 | 0 | pre_grasp | train | True | `libero_spatial_t07_ep000_s0024_pre_grasp` |
| libero_spatial | 7 | 0 | manipulation | train | True | `libero_spatial_t07_ep000_s0048_manipulation` |
| libero_spatial | 7 | 1 | approach | train | True | `libero_spatial_t07_ep001_s0016_approach` |
| libero_spatial | 7 | 1 | pre_grasp | train | True | `libero_spatial_t07_ep001_s0032_pre_grasp` |
| libero_spatial | 7 | 1 | manipulation | train | True | `libero_spatial_t07_ep001_s0056_manipulation` |
| libero_spatial | 7 | 2 | approach | train | True | `libero_spatial_t07_ep002_s0016_approach` |
| libero_spatial | 7 | 2 | pre_grasp | train | True | `libero_spatial_t07_ep002_s0032_pre_grasp` |
| libero_spatial | 7 | 2 | manipulation | train | True | `libero_spatial_t07_ep002_s0064_manipulation` |
| libero_spatial | 7 | 3 | approach | train | True | `libero_spatial_t07_ep003_s0016_approach` |
| libero_spatial | 7 | 3 | pre_grasp | train | True | `libero_spatial_t07_ep003_s0032_pre_grasp` |
| libero_spatial | 7 | 3 | manipulation | train | True | `libero_spatial_t07_ep003_s0064_manipulation` |
| libero_spatial | 7 | 4 | approach | train | True | `libero_spatial_t07_ep004_s0016_approach` |
| libero_spatial | 7 | 4 | pre_grasp | train | True | `libero_spatial_t07_ep004_s0024_pre_grasp` |
| libero_spatial | 7 | 4 | manipulation | train | True | `libero_spatial_t07_ep004_s0056_manipulation` |
| libero_spatial | 7 | 5 | approach | train | True | `libero_spatial_t07_ep005_s0016_approach` |
| libero_spatial | 7 | 5 | pre_grasp | train | True | `libero_spatial_t07_ep005_s0032_pre_grasp` |
| libero_spatial | 7 | 5 | manipulation | train | True | `libero_spatial_t07_ep005_s0064_manipulation` |
| libero_object | 3 | 6 | approach | val | True | `libero_object_t03_ep006_s0024_approach` |
| libero_object | 3 | 6 | pre_grasp | val | True | `libero_object_t03_ep006_s0056_pre_grasp` |
| libero_object | 3 | 6 | manipulation | val | True | `libero_object_t03_ep006_s0176_manipulation` |
| libero_object | 3 | 7 | approach | val | True | `libero_object_t03_ep007_s0016_approach` |
| libero_object | 3 | 7 | pre_grasp | val | True | `libero_object_t03_ep007_s0040_pre_grasp` |
| libero_object | 3 | 7 | manipulation | val | True | `libero_object_t03_ep007_s0064_manipulation` |
| libero_object | 6 | 6 | approach | val | True | `libero_object_t06_ep006_s0024_approach` |
| libero_object | 6 | 6 | pre_grasp | val | True | `libero_object_t06_ep006_s0056_pre_grasp` |
| libero_object | 6 | 6 | manipulation | val | True | `libero_object_t06_ep006_s0072_manipulation` |
| libero_object | 6 | 7 | approach | val | True | `libero_object_t06_ep007_s0024_approach` |
| libero_object | 6 | 7 | pre_grasp | val | True | `libero_object_t06_ep007_s0048_pre_grasp` |
| libero_object | 6 | 7 | manipulation | val | True | `libero_object_t06_ep007_s0064_manipulation` |
| libero_spatial | 0 | 6 | approach | val | True | `libero_spatial_t00_ep006_s0016_approach` |
| libero_spatial | 0 | 6 | pre_grasp | val | True | `libero_spatial_t00_ep006_s0032_pre_grasp` |
| libero_spatial | 0 | 6 | manipulation | val | True | `libero_spatial_t00_ep006_s0056_manipulation` |
| libero_spatial | 0 | 7 | approach | val | True | `libero_spatial_t00_ep007_s0024_approach` |
| libero_spatial | 0 | 7 | pre_grasp | val | True | `libero_spatial_t00_ep007_s0040_pre_grasp` |
| libero_spatial | 0 | 7 | manipulation | val | True | `libero_spatial_t00_ep007_s0056_manipulation` |
| libero_spatial | 7 | 6 | approach | val | True | `libero_spatial_t07_ep006_s0016_approach` |
| libero_spatial | 7 | 6 | pre_grasp | val | True | `libero_spatial_t07_ep006_s0032_pre_grasp` |
| libero_spatial | 7 | 6 | manipulation | val | True | `libero_spatial_t07_ep006_s0064_manipulation` |
| libero_spatial | 7 | 7 | approach | val | True | `libero_spatial_t07_ep007_s0016_approach` |
| libero_spatial | 7 | 7 | pre_grasp | val | True | `libero_spatial_t07_ep007_s0024_pre_grasp` |
| libero_spatial | 7 | 7 | manipulation | val | True | `libero_spatial_t07_ep007_s0056_manipulation` |
