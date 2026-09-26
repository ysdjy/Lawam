# 当前阻塞与续跑入口

机器状态以 `results/robotwin_phase2_newhost/status.json` 和每条episode summary为准。本轮官方 baseline screening 已结束；没有启动任何新方法训练。

## 固定入口

- 原工作树不适合直接当无修改baseline；官方隔离代码在 `results/robotwin_phase2_newhost/official_code`。
- `scripts/robotwin_phase2_newhost/server.py`：先核对官方7.17GB权重SHA256，再检查完整load缺失/多余键，绑定127.0.0.1:10097。每query按固定种子采样。
- `episode.py`：单seed真实native simulator episode；默认不改任务动作或成功条件，只有进程内RT去噪器设none。固定timeout由外层提供，异常不计policy失败。
- `screen.py --stage A`：3个Lift Pot Clean固定seed；`--stage B`：8任务×2条件×5固定seed。B要求A至少3条完成且至少1次成功，作为额外高成功率控制链路检查；不根据候选分数调配置。
- `screen.py --stage summary`：重建CSV和视频索引，无需GPU。
- `data_semantics.py`、`predicate_audit.py`、`check_audit.py`：离线语义/判据/分母检查，不训练、不改模拟器。

运行Python必须分开：simulator使用 `../.venvs/robotwin_poc/bin/python`；policy使用系统 `/usr/bin/python3.10`，仅该进程设置 `PYTHONPATH=/home/zbh/anaconda3/envs/lawam/lib/python3.10/site-packages`，并将官方code路径置顶。不要修改现有conda、系统驱动或认证。完整命令与启动日志保留在本次执行证据中。

若发生系统错误，先保留日志/部分jsonl/trace，定位基础设施后才允许同seed新attempt；不得静默跳过错误或改seed。screening 已有证据并已锁定 Primary1=`open_microwave`、Primary2=`stack_blocks_three`、Control=`lift_pot`。`put_object_cabinet` 的两个 cell 分别只有3/5、2/5个有效 policy episode，原因是固定候选池专家拒绝过多；不得补换seed，报告中标为欠采样。系统错误、超时和原生崩溃均保留且未计入 policy success。910000–910099保持未用。

## 当前剩余限制

- 所有 screening cell 是5条量级，属于机制筛选，不是论文100-trial复现或统计显著性检验。
- RTX 5080 上 SAPIEN RT + OIDN 会报不支持；本轮固定使用 RT renderer、denoiser=`none`，已通过1800×3 camera连续取图压力门。该渲染差异需在正式实验报告中保持显式。
- 个别早期 episode 遇到 `get_picture` hang、矩阵诊断 SIGSEGV、json序列化 SIGFPE 和墙钟超时；同 seed 已保留原始 attempt 并在串行取图/CPU线程限制后复核。它们是 infrastructure/system-error ledger，不可当作 policy failure。
- predicate 审计发现 Stack Bowls 的高度条件和 Stamp Seal 的接触条件存在逻辑反例；没有修改任务源码，也没有用这些任务做主结论。
- 下一步仍需先做选定任务的离线未来-TCP/对象状态时间对齐与可预测性检查，再设计单一物理信息变量的最小方法实验；当前不能声称 TCP 注入会提升成功率。

## 固定候选池补充（筛选前冻结）

初始A的100001被官方expert合法性过滤拒绝，发生在policy调用前。为满足有效episode数量，固定A候选池100000–100009，完成3条即止；B候选池100100–100119，每cell完成5条即止。B尚未开始时即冻结此池。只允许expert拒绝后推进至下一固定候选，任何policy失败均计入完成数，不能替换；系统错误中止并修复同seed。原协议保存在 `evidence/protocol_initial_before_policy.json`，修订理由和时间写入protocol.json。报告同时列出全部rejected和实际policy seeds，不隐藏过滤。
