# 任务选择审计（官方权重 screening 后锁定）

官方权重本地 screening、视频和物理 trace 已完成。以下 local baseline 仅用于小规模机制筛选，不宣称复现论文100-trial统计。

论文值来自 [LaWAM Table 4](https://arxiv.org/html/2606.15768v1#A1.T4)，50个唯一任务、100个cell已重新解析，Clean均值92.64%、Randomized89.80%、等权91.22%；论文每cell100 trials。源数据见 `paper_table4.json`。

| 环境模块 | paper Clean / Rand % | 提升空间 | TCP位置 | 姿态 | 接触/关节 | 失败可观察性与判据风险 |
|---|---:|---|---|---|---|---|
| open_microwave | 41 / 43 | 大 | 抓把手、保持轨迹 | 高，沿铰链运动 | hinge/contact | qpos >= 最大关节值×0.6，进度清晰；不能归结为纯位置 |
| hanging_mug | 51 / 43 | 大 | 多阶段抓取/挂置 | 高 | handover/hanging | 功能点XY接近架中心、Z>0.86、右夹爪命令打开；不要求松手后稳定挂住，研究干净性偏低 |
| turn_switch | 47 / 56 | 大 | 接近与按压方向 | 高 | articulation/contact | qpos距上限0.05以内；能观察进度，需区分接触方向、姿态和位置 |
| place_can_basket | 92 / 65 | Clean小/Rand中 | 双臂相对位置 | 中高 | basket/contact | 篮/罐高度、篮轴方向、距离、接触篮且不接触桌面；物体状态可能更关键 |
| stack_blocks_three | 90 / 75 | Clean有限/Rand中 | 高，厘米级放置 | 中高 | stacking | 两层相对XYZ阈值[.025,.025,.012]m及夹爪打开；没有持续稳定判据 |
| stack_bowls_three | 90 / 80 | 中到有限 | 高 | 高，碗嵌套 | stacking/contact | 高度判断是 signed difference < .02，缺少绝对值/下界，低于目标也能通过高度项；高优先级判据风险 |
| put_object_cabinet | 90 / 82 | 有限 | 放入抽屉 | 中高 | articulation | 物体相对高度在(.007,.12)m且XY接近功能点、夹爪打开；未直接要求抽屉qpos或完整容纳 |
| stamp_seal | 89 / 88 | 有限 | XY精度1cm | 潜在高但判据不体现 | 接触理应重要 | 原判据仅XY接近+双夹爪打开，不要求盖章接触、Z或印记；不适合直接声称接触物理改善 |
| lift_pot | 100 / 99 | 小，控制候选 | 双TCP距把手均<3cm | 锅轴方向>0.8 | 双臂约束 | 锅Z>.82m且双TCP/把手几何阈值；当前仍只是控制候选，等本地验证 |

所有任务的 current EEF/TCP、真实关节、夹爪关节、物体世界pose、contact 可从当前模拟器获取。微波炉/开关/柜子等关节实体还可读 qpos/limits。future TCP 只能由后续实际trace得到，不是reset时可直接查询的已实现未来。发布Parquet本身不含contact/object/articulation标签，不应假装官方数据已有全套物理真值。

在没有本地视频/trace前，不把低分解释成“缺TCP”，不排除感知、语言、动作归一化、版本或渲染差异。Hanging Mug 的多阶段耦合及原成功判据、Stack Bowls Three 的单侧高度条件、Stamp Seal 的接触缺失使它们目前的研究干净性较弱；这是源码风险，不是本地失败结论。

## 最终任务选择

**Primary 1 — `open_microwave`。** 论文 Clean/Randomized 为41/43%，本地为2/5和3/5。失败 episode 中 planner 全为 Success、EEF target tracking 约0.7–0.8 mm均值，但铰链 qpos 只达到约0.05–0.52 rad，低于0.6×上限判据；视频与 contact/关节 trace 能区分抓手脱离、铰链进度停滞。它提供 articulation progress、handle-relative TCP、接触和姿态标签，适合测试未来物理状态是否比当前视觉动作更能保持可执行接触。

**Primary 2 — `stack_blocks_three`。** 论文 Clean/Randomized 为90/75%，本地为5/5和3/5。随机失败中 block 相对偏移约8–10 cm，planner 成功且多数 EEF target error低；predicate 对相对 XYZ 的阈值明确，物体 pose、TCP、接触和释放后的未来状态都可记录。它与微波炉的铰链/接触机制不同，能检验几何放置和未来物体状态这一类物理信息。

**Control — `lift_pot`。** 论文 Clean/Randomized 为100/99%，本地 Clean 为3/3；成功判据是锅体高度、双 TCP 与把手几何及姿态约束，适合检测物理信息分支是否破坏原有控制能力。它不是主实验刷分任务。

不选 `hanging_mug` 是因为多阶段 handover/hang 与 predicate 不要求稳定悬挂；不选 `turn_switch` 为主任务是因为小样本 clean/randomized 差异大，且接触方向和姿态难从 TCP 位置单独解释；不选 `place_can_basket` 是因为主要失败是物体掉落和篮筐接触，适合作为后续扩展但机制更混杂；不选 `stack_bowls_three`、`stamp_seal` 是因为本地全成功且 predicate 分别存在高度单侧条件、缺少接触/Z约束；`put_object_cabinet` 因固定候选池有效样本不足而不作结论。

保留910000–910099 seed，从未用于此次筛选或调参。不以新方法效果反向修改上述选择。

下一阶段若baseline与任务筛选通过，最小实验应一次只检验一种物理信息：从实际未来TCP几何标签做离线可预测性/时间对齐检查，再比较等容量正确信息与打乱信息控制；是否采用位置或关节进度由选定失败机制决定。当前不增加多头、不开始RT1/RT2训练。

## 判据的独立反例检查

`predicate_audit.py` 从本地源码抽取原 `check_success` 函数原样执行，在明确标注的反事实状态下验证了：低于桌面的对齐碗、目标上方45.9cm的印章、架上方60cm的杯功能点，都可满足各自判据的相应完整条件。结果保存在 `predicate_counterexamples.json`。这属于逻辑单元检查，不是物理rollout，不证明本地发生过误判，也不计入成功率。原任务源码和成功判据未改。

## 固定候选池补充（筛选前冻结）

初始A的100001被官方expert合法性过滤拒绝，发生在policy调用前。为满足有效episode数量，固定A候选池100000–100009，完成3条即止；B候选池100100–100119，每cell完成5条即止。B尚未开始时即冻结此池。只允许expert拒绝后推进至下一固定候选，任何policy失败均计入完成数，不能替换；系统错误中止并修复同seed。原协议保存在 `evidence/protocol_initial_before_policy.json`，修订理由和时间写入protocol.json。报告同时列出全部rejected和实际policy seeds，不隐藏过滤。
