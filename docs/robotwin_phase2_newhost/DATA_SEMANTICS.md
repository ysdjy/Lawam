# RoboTwin 数据与控制语义审计

本文件只讨论本次核查的 RoboTwin；旧 LIBERO/PoC 指标不作为证据。数值结果见 `results/robotwin_phase2_newhost/evidence/data_semantics_results.json`，可重跑脚本为 `scripts/robotwin_phase2_newhost/data_semantics.py`。

## 发布数据与任务映射

- 官方 `jialei02/robotwin_merged` 固定 revision `560f9e9fecc7dbf826afd6042b418d632e319df1`。元数据为 LeRobot v3，27,500 episodes、6,077,247 frames、30 fps、16D state/action。
- `task_index` 指向 23,607 条语言指令，不是 0–49 的环境编号。`meta/tasks.parquet` 保存指令文本；episode metadata 没有权威的 environment task ID。
- 论文 50 个名字逐一核对本地模块存在，并保存源文件 SHA256，见 `paper_task_mapping.json`。本次 9 个候选名字均对应小写下划线模块。
- 数据抽样使用人工核对指令与源码的明确 episode 列表，再自动读出 task_index，见 `task_index_mapping.json`。没有用 episode 除以 500 自动宣称任务身份。初版通配模板/关键词映射误把 “switch arms” 等当任务名称，结果已标记 `invalid_heuristic_*`，不用于结论。
- 数据完整 Parquet 与元数据已下载；训练视频未全量下载。没有训练、重算或覆盖发布 normalization statistics。

## 自动数值核查

9 个任务共 50 个 episode，逐 episode 内检验 shift -2/-1/0/+1/+2，禁止跨 episode 比较。全部 50 个样本的非末帧均有 **action[k] == observation.state[k+1]，16 个浮点分量完全一致**；shift 0 的误差非零。全部样本有 `action[t+36] == state[t+37]`。末帧 action 不等于最后 state，不能把最后动作视为已观测的未来状态。

这证明发布数组含一步前移的标签关系。它没有单独证明 state 最初来自哪版原始转换脚本、是否重采样，或原始采样的真实物理间隔。官方README第399–403行说明数据源于 LingBot/Robbyant EEF 数据并转换为 v3；精确转换脚本与原始物理时间戳尚未取得，不把数值等式当作完整物理溯源。

## 坐标与字段

当前 SAPIEN/RoboTwin 实现（`6dde571...`）的 `get_obs().endpose` 调用 `get_arm_pose` → `get_*_ee_pose`。位置为世界坐标、单位米，四元数为 wxyz；16D 排列为 `[L_xyz,L_wxyz,L_gripper,R_xyz,R_wxyz,R_gripper]`。

`robot._trans_endpose` 读取实际 end-link 的 `global_pose`，乘机器人配置的 global/delta rotation，然后沿工具 x 轴平移。EEF 用 `gripper_bias - 0.12`；`get_*_tcp_pose` 用 `gripper_bias`。因此同一状态的 simulator TCP/gripper center 在 EEF 工具 x 方向前方 0.12m，姿态一致。TCP 是该机器人模型定义的夹爪中心，不代表动态接触点。trace 同时保存 raw end-link、变换后的 EEF、TCP，禁止互换。

`get_*_arm_jointState` 返回 drive target；`get_*_arm_real_jointState` 的机械臂分量才取 articulation qpos，但末尾夹爪仍是命令值。`get_normal_real_gripper_val` 也读取 drive target。故本次另从 articulation 的真实 qpos 抽取夹爪关节；命令开度与实际夹爪关节分别记录。原 success predicate 的 gripper-open 判断仍保持官方命令值语义，没有修改。

## 36 帧、chunk 与真实控制时间

- 发布 checkpoint：`data_mix=robotwin_eef_30hz`，`use_state=false`，内部 action_dim=32、最大 horizon=50，flow horizon=1.2s，10 次 flow integration。原始16D动作从32D输出切取并用发布 statistics 反归一化；四元数单独归一化。
- 官方 EEF adapter 的夹爪输出为连续值（`action_binary_indices=()`），但第7/15维仍执行 `1-g`（`action_invert_indices=(7,15)`）。夹爪方向转换与二值阈值是两件不同的事；本次完整沿用官方转换。
- native夹爪命令1为打开、0为闭合；`is_*_gripper_open`检查命令>0.8。发布数据/模型侧与native执行侧的开合方向由上述`1-g`转换连接。实际关节开度仍须读qpos，不能由命令“打开”推断已经完全打开。
- 当前训练 reader 使用 `int(1.2*30)=36`，动作 offsets 为 0..35；`num_frames=2` 的图像 offsets 为 **[0,35]**。因此当前源码的 future RGB 是 t+35（名义1.1667s），而第36个动作 action[t+35] 数值对应 state[t+36]。这是一帧差异，不能写成训练 future RGB 和最终动作状态都是 t+36。
- Flow 时间编码 `i/hz` 为 0..35/30，输出有效36步；最大50步不是在线实际执行50步。官方 auto eval 默认 `replan_steps=36`、ensemble off；本次固定同设置。
- 在线 `take_action(..., action_type='ee')` 接收一个绝对 EEF 目标，转换为 end-link planner goal，左右 CuRobo 规划后执行变长轨迹。场景 timestep 为1/250秒；规划失败会走50步回退，success可提前结束。**36个planner命令不保证等于1.2秒模拟时间**。
- 当前 RoboTwin collect 配置 save_freq=15，在250Hz下也不能直接解释为30Hz。发布30fps是数据坐标，不足以证明原始真实采样时钟。

在线每条 action 记录起止 physics_tick、完整 planner result、实际qpos/EEF/TCP及contact；query记录完整normalized输出和反归一化chunk，action_start记录真正取出的前缀。视频按动作帧10fps回放，真实时间须读取jsonl，不能由视频播放时长反推。

下一步的 future label 必须从实际 trace 按明确物理时间插值/匹配，并保留跨episode、terminal和观测缺失mask；不能直接把预测 action target 当作 realized future TCP。未来训练前需先完成 baseline 与任务锁定。

## 轨迹存储与完整性

运行中逐步写入`physical_trace.jsonl.gz`。完成后可无损转换为`physical_trace.jsonl.zst`；实际文件名在episode的`summary.json.physical_trace_file`和视频索引中，`trace_storage.json`记录解压后SHA256及完整回验结果。转换不减少采样频率、不删除任何字段。可用`zstd -dc physical_trace.jsonl.zst`流式读取，避免一次载入数GB未压缩文本。中断attempt不做完整性宣称，保留原始部分文件。

`joint_qpos.left/right`各保存对应articulation的完整qpos；当前双臂共用实体，因此两数组相同，不是左右臂状态相同。六自由度的`joint_drive_target`按左右臂配置单独排列。涉及无限关节范围的qlimits沿用Python JSON的Infinity表示；动作、EEF、TCP及图像均检查有限值，不能把无限关节上界当NaN观测。

大trace的已完成文件另归档到`/media/zbh/4E4A-BDAC/LaWAM_robotwin_phase2_newhost_20260925/`，工作区同名路径保留符号链接。复制前后字节SHA256与路径映射见`evidence/external_storage.jsonl`；读取这些trace需要该盘保持挂载。未移动视频、报告、权重或数据集，未删除旧实验。

v4记录器中，hanging_mug功能点直接保存位置和完整世界变换矩阵，避免仅为记录而重复求四元数；早期记录是xyz+wxyz。逐physics tick不额外重算原判据，字段标明unavailable并指向action_end/terminal的实际检查；原任务自身逐step的成功检查没有改变，物理状态与contacts仍每tick保留。

录像格式有记录版本：早期为头/左腕/右腕横排；第二次额外右腕读回卡死后，改为每动作头部图像，以减少额外渲染读回。三路policy原始RGB仍在每次query保存。每动作录像帧与物理时钟的关系未改变。`check_rollouts.py`逐条验证执行前缀与chunk一致、query/seed/时钟对应及终局证据。

## 本次数值审计的实际 task_index

这些数字仅定位所列样本的语言标签，不是任务统一编号。

| 环境任务 | 本次样本 task_index |
|---|---|
| hanging_mug | 3588, 3589, 3590, 3591, 3592, 3593 |
| lift_pot | 4085, 4086, 4087, 4088, 4089, 4090 |
| open_microwave | 6829, 6830, 6831, 6832, 6833, 6834 |
| place_can_basket | 10671, 10672, 10673, 10674, 10675, 10676 |
| put_object_cabinet | 938, 939, 940, 941, 942, 943 |
| stack_blocks_three | 19581, 19582, 19583, 19584, 19585, 19586 |
| stack_bowls_three | 19796, 19800, 19801 |
| stamp_seal | 20623, 20624, 20625, 20626, 20627, 20628 |
| turn_switch | 21121, 21122, 21124, 21125, 21126 |

## 首条真实闭环的时钟证据

官方Lift Pot seed100000成功：100个命令、8,681个physics tick（34.724s）、3次query。query在命令0/36/72发生，物理时钟分别0/13.040/26.328s。每次输出36×16动作；第三chunk只执行28个动作即触发原成功判据。此证据直接否定“native replan36恒等1.2s执行”的假设，不涉及改变动作执行。

`trace_analysis.json`的目标跟踪误差比较每个planner命令结束时的实际EEF与该命令绝对EEF目标，统计包含左右两臂和闲置臂。它不是接触点误差、不是物体目标关系误差，也不是模型未来状态预测误差；毫米级目标跟踪并不意味着抓持/放置正确。相邻`action_start/end`按事件顺序配对：start的command_index为执行前计数，end为已完成计数，比start加1。
