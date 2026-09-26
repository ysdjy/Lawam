# 本机环境审计

执行日期2026-09-25，工作区 `/home/zbh/Downloads/IsaacLab/Lawam_paper`。所有状态以下方实际日志为准，不根据“新机器”描述推断硬件。证据目录 `results/robotwin_phase2_newhost/evidence/`。

| 项目 | 本次实测 |
|---|---|
| 主机/GPU | zbh-MS-7D90；RTX5080 16303MiB；driver580.126.09，MIG不适用 |
| RAM/存储 | 31GiB RAM，无swap；起始可用RAM26GiB、磁盘35GiB；下载与运行逐步监控 |
| Vulkan | 系统未装vulkaninfo；仅下载Ubuntu vulkan-tools deb并解包至results/tools，无系统安装。`vulkaninfo --summary` exit0，显示NVIDIA RTX5080。其他ICD一条warning不影响NVIDIA枚举 |
| 设备权限 | `/dev/dri/renderD128` 存在；用户不在render/video组但ACL明确给zbh读写；没有修改权限 |
| SAPIEN | 3.0.0b1；default renderer40帧、RT32spp depth8连续40帧读回通过；原OIDN报unsupported/invalid handle，none去噪配置无该错误 |
| Simulator Python | `.venvs/robotwin_poc/bin/python`，系统Python3.10.12；torch2.7.1+cu128含sm120 |
| 规划器 | mplib0.2.1，nvidia-curobo0.7.8；当前机器人用CuRobo。真实Lift Pot reset已触发规划器初始化 |
| Policy Python | `/usr/bin/python3.10`，进程级PYTHONPATH引用已有lawam site-packages；torch2.7.1+cu128、transformers5.2.0；官方policy imports通过，无全局配置改动 |
| LaWAM工作树 | HEAD5c9d6b5，起始11个tracked文件已有旧实验修改。保存initial_git_status和initial_worktree.patch；不覆盖或reset |
| 官方代码 | 独立detached worktree固定origin/main `7d27b9607c22034934a4b70347f8bfaba92bf692`；模型与动作adapter均从此导入；权重路径只用symlink复用 |
| RoboTwin | 现有干净checkout `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`。相较LingBot推荐2eeec322，已对照robot.py及Lift Pot/Microwave/两种stack源码完全一致；base变化主要为采集/保存辅助逻辑，diff保存供审计。不能证明就是论文运行版本 |
| 官方权重 | `jialei02/lawam_robotwin_sft_release@f1c4f9a36bddc3e8243e1da8e2b7fb329f84fb86`，大小7,174,214,645字节，预期SHA256 `a52031302c6dc5b813982227255add8d2acb839149a4b90908b179a8f66adbeb`；下载/实际加载状态见status.json与server日志 |
| 官方数据 | `jialei02/robotwin_merged@560f9e9fecc7dbf826afd6042b418d632e319df1`，完整Parquet和metadata；不把本地旧PoC数据当官方baseline |
| 资产 | 官方RoboTwin2.0 revision3dc3b798668feb99ac61cc9086d84cbcc3d79186，通过HTTP Range和ZIP CRC补齐；Randomized需要完整unseen纹理和完整物体候选集合，不能靠减少杂物库制造容易分布 |

## 已通过的门槛与差异

1. Vulkan真实NVIDIA枚举通过。
2. SAPIEN连续take_picture/RGB读回通过；RT需关闭不兼容OIDN，保留该渲染差异，不称严格论文复现。
3. Lift Pot：先完成reset/三相机/25个physics tick检查（不计成功率），随后用官方权重完成seed100000/100002/100003三条完整闭环，3/3成功。实测双EEF/TCP相距0.12m。原始视频、逐动作日志和逐physics tick trace均已保存。

重用环境与官方requirements存在版本差异（例如websockets16.1.1而README要求15.0.1），所有实际包版本已存 `sim_packages.txt`。先测试已有环境，不为了凑版本重装系统。XPolicyLab子模块未初始化，本次直接使用官方LaWAM websocket/native task路径，不依赖它。

## 资源约束与非侵入性

一次只跑一个simulator worker与一个policy server，分别限制内存；simulator PyTorch显存上限32%，policy60%；限制线程数，单episode有wall timeout；OOM/timeout计系统错误，不能伪装失败或0%成功率。保留已有GPU桌面/ToDesk及所有其他作业。未申请管理员权限、改驱动、改Claude/Codex认证、训练或覆盖checkpoint，未提交/推送。

## 官方权重实际加载（本轮）

完整下载后SHA256与发布LFS对象一致。官方框架重复注册VLM/flow，导致738个别名在普通state_dict检查中显示missing；逐项验证它们与已加载主键共享同一存储、形状、stride与dtype，并核对全部1,419个主键张量的加载值一致，真实缺失0、多余0。最初保守检查中止的日志保留为 `server_initial_alias_check.log`，没有用那次失败加载跑episode。随后官方模型已bfloat16部署在127.0.0.1:10097，初始GPU约5.75GiB；加载审计见 `server_load.json`。未修改官方网络结构、参数或动作算法。

## 长轨迹读回故障与修复记录

Open Microwave Clean seed100103首次尝试在动作1270后卡住，连续两次120秒线程栈停在 `camera.get_picture('Color')`；GPU无计算负载，未见内核日志异常。仅终止本轮worker，记system_error，终局未知；原始attempt移入 `results/.../attempts/`，保留在CSV系统错误计数中。不能把它当策略失败或成功。

独立RT-none三相机1800轮长测通过。诊断视频采集补上每次取图前的camera pose与`scene.update_render()`同步，避免调用会改变灯光随机序列的`env._update_render()`；未改policy观测入口、动作或物理参数。随后从同seed完整重跑。该同步是候选修补，不宣称已证明故障根因；修复时间与差异见 `evidence/renderer_hang_repair.json`。每动作120秒无进展时输出线程栈并终止，防止再次无限挂起。

RT32spp关闭OIDN后存在可见颗粒噪声，这是本地baseline与发布实验之间的视觉域差异。小样本成功率和失败不能直接因果归于缺少物理信息。

## 后续稳定性审计（跨2026-09-26）

筛选过程中共保留以下真实系统错误，均不进入成功率分母：Open Microwave Clean100103诊断相机读回挂起；Randomized100109诊断相机挂起、随后一次server空闲退出导致连接超时；Hanging Mug Clean100102及Randomized100108在`numpy.linalg.eigh`原生崩溃；Place Can Basket Randomized100102在JSON编码中SIGFPE，100105在原生成功分支`get_obs()`相机读回挂起。完整中断attempt与日志留存，未当作policy失败。初始化UnStableError和已核对的expert抓取目标异常按官方候选过滤处理，最初保守分类也保留了复核记录。

显式渲染同步、减少额外腕部录像取图、单线程OpenBLAS均没有单独消除所有故障。记录器随后改用功能点矩阵，避免重复求四元数；不再在native `scene.step`内部额外重算判据。主机为i9-14900K，存在5.7/6.0GHz-max及4.4GHz-max核心。只对本次进程限制CPU亲和性：simulator16–23，policy24–31，并保留内存限制；没有改主机频率、BIOS、驱动、现有Python包或其他作业。10万次数值/JSON/gzip检查通过，但这不能证明硬件根因或排除硬件问题。内核日志可见部分时段NVIDIA invalid-object-handle警告，未据此断言根因。

CPU隔离后原生终局相机读回仍挂起，按停止条件暂停广泛筛选。最后进行独立串行相机验证：逐相机`take_picture → get_picture`，相机顺序、RGB转换、场景/灯光及动作不变。篮筐同seed100105初始三路PNG与旧读回逐字节一致，随后700步正常完成失败；独立Lift Pot100000复核100步成功。仅在这两项通过后恢复剩余候选。新配置下若再有原生故障，本次不再增加兼容性修补。协议的每次修补时间与适用范围均留在`protocol.json`。

本次属于带兼容性修补的官方权重本地screening，不能称严格发布环境复现。固定seed不保证逐位相同仿真轨迹；已观察到同seed重试的关节轨迹不同。后续任何新方法比较必须先锁定最终runtime并复核其baseline，不能把不同恢复阶段的差异当方法收益。
