# 官方权重本地 screening

本阶段只用官方RoboTwin SFT release；无新方法训练。paper published、local baseline、new method分列。完整policy闭环已通过：Lift Pot Clean完成3/3成功，专家拒绝1条、系统错误0条。候选任务screening已完成；Put Object Cabinet 因固定候选池中专家有效 seed 不足而保留为欠采样 cell。

冻结配置见 `results/robotwin_phase2_newhost/protocol.json`。阶段A Lift Pot Clean按固定候选池100000–100009完成3条；阶段B八个任务两种condition按固定候选池100100–100119筛选（仅screening；筛选前扩池理由见下文）。除 Put Object Cabinet 外，15个cell各完成5条；Cabinet Clean完成3条、Randomized完成2条，不能补换seed。保留910000–910099未用seed。官方专家seed合法性检查保留，但不根据policy结果替换seed；拒绝与系统错误单列。

replan=36，action ensemble关闭，16D绝对EEF动作，native planner完整执行，flow RNG按`seed*10000+query_index`固定。指令类型seen沿用LaWAM官方deploy_policy.yml（两个condition均如此；RoboTwin randomized YAML自身写unseen，差异显式保存）。评测采用单worker，条件/版本/权重不随任务分数变化。

每条episode应包含summary.json、episode.jsonl、physical_trace.jsonl.gz、video.mp4与query_rgb/。JSONL记录完整policy输出、反归一化chunk、执行action及planner结果；trace记录每个物理step实测状态/contacts。视频是每动作一帧的10fps诊断回放，不是匀速物理时间录像，frame↔physics_tick映射在JSONL。

小样本仅定位失败机制与检查链路；不够宣布复现论文、超过论文或统计显著提升。系统错误不进入已完成episode成功率分母，同时必须另报总计划数、完成数、拒绝数和系统错误数，防止隐性筛seed。

## 固定候选池补充（筛选前冻结）

初始A的100001被官方expert合法性过滤拒绝，发生在policy调用前。为满足有效episode数量，固定A候选池100000–100009，完成3条即止；B候选池100100–100119，每cell完成5条即止。B尚未开始时即冻结此池。只允许expert拒绝后推进至下一固定候选，任何policy失败均计入完成数，不能替换；系统错误中止并修复同seed。原协议保存在 `evidence/protocol_initial_before_policy.json`，修订理由和时间写入protocol.json。报告同时列出全部rejected和实际policy seeds，不隐藏过滤。

## 阶段A实测

| local baseline seed | outcome | commands | physics ticks | queries | simulated seconds |
|---|---|---:|---:|---:|---:|
| 100000 | completed / True | 100 | 8681 | 3 | 34.724 |
| 100001 | seed_rejected / None | 0 | 0 | 0 | 0.0 |
| 100002 | completed / True | 106 | 9111 | 3 | 36.444 |
| 100003 | completed / True | 104 | 9266 | 3 | 37.064 |

Paper published: Lift Pot Clean100%、Randomized99%；local baseline目前仅Clean3/3，new method未运行。这个小样本支持链路与高成功率控制可用，不足以确认论文50任务均值。

## 阶段B实测汇总

| task | local Clean | local Randomized | paper Clean / Rand | completed / planned |
|---|---:|---:|---:|---:|
| open_microwave | 40% (2/5) | 60% (3/5) | 41 / 43 | 10/10 |
| hanging_mug | 40% (2/5) | 80% (4/5) | 51 / 43 | 10/10 |
| turn_switch | 20% (1/5) | 80% (4/5) | 47 / 56 | 10/10 |
| place_can_basket | 60% (3/5) | 60% (3/5) | 92 / 65 | 10/10 |
| stack_blocks_three | 100% (5/5) | 60% (3/5) | 90 / 75 | 10/10 |
| stack_bowls_three | 100% (5/5) | 100% (5/5) | 90 / 80 | 10/10 |
| put_object_cabinet | 100% (3/3) | 100% (2/2) | 90 / 82 | 5/10 |
| stamp_seal | 100% (5/5) | 100% (5/5) | 89 / 88 | 10/10 |

以上均为 local baseline；paper published 值是论文 Table 4 的100-trial cell；new method 尚未运行。系统错误单独保留在 CSV，不进入 local success 分母。完整性检查：78 个 completed episode 通过 `check_rollouts.py`，其中58成功；这是跨 cell 的小样本汇总，不是论文复现声明。

## 阶段B结论

按机制清楚、失败可观察、物理标签可得、且不是单纯最低分筛选，推荐主任务为 `open_microwave` 与 `stack_blocks_three`，控制任务为 `lift_pot`。`turn_switch` 是备选，但 clean 只有1/5且随机化4/5，显示较强的姿态/接触方向依赖；`hanging_mug` 多阶段且 predicate 风险较高。`stack_bowls_three`、`stamp_seal` 本地全成功，`put_object_cabinet` 欠采样，均不用于主实验结论。
