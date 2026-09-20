# LaWAM 阶段0/阶段1 诊断报告：同状态双方案（A/B）对应关系

run_id: `bd_20260914_093902`　日期: 2026-09-14　目录: `results/branch_diagnostic/bd_20260914_093902/`
代码: `research/branch_diagnostic/`（README.md 含完整复现命令）　模型改动: `code_changes.patch`（4 个文件，全部为默认关闭的可选接口）

---

## 1. 本轮验证什么，不验证什么

验证链路：`z_i → predicted_future_i → action_chunk_i → actual_future_i`，在同一仿真快照、同一观测、同一指令下，A/B 两个真实可行的短期方案能否被 LaWAM 各环节保持。

- A. 参考方案是否真的可行（R 组）
- B. LAM 教师是否把 A/B 编码为可区分的 z（z_A vs z_B）
- C. LaWM 是否按 z 预测出正确且可区分的未来（B1 组的预测未来）
- D. 动作头是否按未来条件执行对应方案（B2 组：给真实未来 u_A/u_B）
- E. 模型内部预测与实际执行是否对应（预测标签 vs 执行标签）

不做：多候选自主生成、训练/微调、真机、长时规划、benchmark、文献结论。本轮 B1/B2 全部是 **oracle 诊断**（教师 z 与真实未来来自参考轨迹），不代表模型自主规划能力。

## 2. 本地实际版本与模型数据流（按代码核对，非按概念图）

环境清单见 `environment_manifest.json`。仓库 commit `7d27b96`，起始时仅 `weights/` 未跟踪；checkpoint `lawam_libero_sft_release/final_model/pytorch_model.pt`（7.17 GB，2026-09-14 01:24）；LAM `dino_large_vae/lam_release`；Qwen3-VL-2B；DINOv3-ViT-B/16（本地重建）；RTX 5080 16 GB；`lawam`（torch 2.7.1+cu128）与 `libero310`（mujoco 3.3.2, robosuite 1.4.0）两个 conda 环境。旧冒烟结果（libero_spatial 20/20）只作为部署可用的线索，未计入本轮。

核对到的数据流（文件：`starVLA/model/framework/vlas/lawam.py`, `flowmatching_expert.py`, `latent_world/batch_builder.py`, `latent_world/runtime/runner.py`, `examples/LIBERO/eval_files/*`, `latent_action_model/core/*`）：

| 项目 | 实际情况 |
|---|---|
| 潜在动作 z | VLM 序列中 8 个 `<ACT_PH>` 查询 → `VLMToLAMQFormer`（1 个可学习 query 交叉注意）→ **单个** z，shape `[B,1,32]`，float（模型整体 bf16）；`z0` 范数 ≈ 2.1 |
| 当前视觉特征 h_t | `lam.extract_vision_features(primary_image)`：DINOv3 倒数第二层 + LayerNorm，`[B,256,768]`（16×16 token，256×256 图） |
| 预测未来 h_t1_pred | `lam.decoder(h_t, z)`（LAMDecoder_v2，AdaLN 12 层）→ `[B,256,768]` |
| `h_t1_gt` 变量 | 推理时 `primary_image` 只有 1 帧，`features[:, -1]` 与 `features[:, 0]` 相同，**不是真实未来**（阶段0已核实） |
| 动作头条件 | `cat(h_t, h_t1_star, enc_vlm(h_vlm))`（AlternateVLDiT 交替对图像/VLM token 交叉注意）；`use_state=false`（state 不进入动作头，仅归一化后传输）；CFG scale 1.0（不启用）；10 步流匹配；初始噪声 `torch.randn` 在 `sample_actions_cfg` 内生成 |
| 时间 | `horizon_sec=0.4`，`action_hz=20` → 有效动作 `floor(0.4×20)=8`；`action_horizon=50` 只是 padding 上限；服务端返回 8 步，客户端缓存并**全部执行 8 步**后再查询 |
| 训练监督未来帧 | `lerobot_datasets._sample_video_delta_indices`：`num_frames=2`, `sec_chunk=0.4` → 视频索引 `[0, 7]`，即 **执行 7 个动作后的观测 o_7** |
| LAM 教师 | `_run_lam_teacher`：帧 `[0,7]`（`num_frames=2`），VAE 推理取 `mu`，无随机；训练时 distill loss（mse, 权重 0.1）把策略 z 拉向教师 z；`unfreeze_lam_decoder=true` |
| 训练时动作头的未来条件 | `enable_flow_h_t1_scheduled_sampling=false` → `_flow_h_t1_pred_prob()=1.0` → **动作头训练中只见过预测未来 h_t1_pred（detach），从未以真实未来 h_t1_gt 作条件** |
| 图像处理 | agentview / wrist 均 `[::-1, ::-1]`（180° 旋转），256×256，无裁剪；LAM 输入 ImageNet 归一化 |
| 动作反归一化 | min/max 线性反归一化，末维夹爪 `<0.5→0, ≥0.5→1` 再 `1-2·(v>0.5)` 再取反 → LIBERO 中 +1 闭合 / −1 张开 |
| 仿真 | OSC_POSE delta 控制，20 Hz，25 个 2 ms 物理子步；控制器 `goal_ori` 在零旋转指令时**持久**（属于需恢复的内部状态） |

时间对齐记录：输入观测时间 t（策略查询步 k 的在线图像）；执行控制区间 a_t…a_{t+7}（8 步，0.4 s）；未来监督帧 t+7；实际对比帧 = 执行 7 个动作后的 o_7（主）与 o_8（辅）。未拿完整 8 步后的状态与预测未来相比较。

## 3. 阶段0检查结果（`phase0_checks.json`，容差在阶段1前声明）

| 检查 | 结果 |
|---|---|
| D 状态恢复（`phase0/restore_test.json`） | 完整恢复（qpos/qvel/time + warmstart + ctrl + timestep/cur_time/done + OSC goal_pos/goal_ori + 夹爪 current_action）后回放 16 步：qpos/qvel 最大差 7.8e-15，eef/物体 0.0；从 npz 载入新 env 同样 7.8e-15；`reset+set_init_state+回放前缀` 也为 0.0。只恢复 qpos/qvel 的朴素方式差 0.19（说明控制器/夹爪内部状态必须恢复） |
| 渲染噪声底线（`phase0/render_check.json`） | 同一状态渲染两次差 0；物理完全相同的两次回放，逐帧最大像素差 ≤92 但均值 0.09、>10 的像素占 0.26%（EGL 反走样噪声）；恢复后重渲染 vs 在线观测：均值 0.18（robosuite 渲染滞后 1 个 2 ms 子步，eef 差 0.2 mm）。**因此所有条件的模型输入一律使用保存的在线图像 o_0** |
| 投影标定 | `phase0/projection_check_v2.png`：eef/碗/盘子投影点落在正确位置（列需镜像，行不需要）；仅用于离线 ROI |
| A 修改前后等价 | 未打补丁的原代码先跑基线（3 观测 × 3 种子，`model_baseline_pristine.npz`）；打补丁后默认路径动作、h_t、h_t1_pred 与基线差 **0.0**（9/9） |
| B 自替换等价 | `latent_override=自身 z` → 动作/未来差 0.0；`future_override=自身预测` → 0.0；显式传入与种子采样相同的噪声 → 与默认路径差 0.0；噪声往返差 0.0078（fp32→bf16 转换，实际使用的 bf16 噪声已保存） |
| C 重复一致性 | 固定噪声重复 5 次：0.0；原代码同种子重复：0.0 |
| 校验 | 错 shape / NaN / 错 token 数 / 错噪声 shape 全部抛 `ValueError` |
| 干预有效性 | `−z` 使预测未来 maxabs 变化 13.9、动作变化 0.035（说明接口生效，也是后文现象的先兆） |
| 参考控制器探针 | v1 四个方向全部失败（放置目标符号错误，已修正并记录）；v2 修正后 +y/−y 各 4/4 成功，±x 因夹指沿 y 闭合不可行 → 方案对固定为 **A=+y 缘、B=−y 缘** |

结论：阶段0 全部通过，未放宽任何容差。

## 4. 双方案构造与筛选（`reference_manifest.jsonl`，规则见 `protocol.yaml`，在模型对照前固定）

- 任务：libero_spatial task 2 “pick up the black bowl from table center and place it on the plate”（碗孤立可见、周围空间充足）。只检查了这 1 个任务，达到合格数量后未扩展任务。
- 候选起点：原版策略经官方 websocket 服务 + 官方 `ModelClient` 在线运行 **30 个不同 reset episode**（init_state 0–29，全部成功，91–99 步），在每个策略查询步保存完整快照。
- 快照规则（仅依赖参考可行性，与模型表现无关）：最早的查询步，满足夹爪张开、尚无闭合指令、碗位移 <2 mm、eef 水平距碗 ≤8 cm、高于碗 10–24 cm。结果：20 个状态取 step 24，9 个取 step 16。
- 参考方案：从快照出发，8 步脚本 OSC 动作（P 增益 12，|a|≤0.9，零旋转）朝 A/B 预抓取悬停点运动，然后固定脚本续作（下降—闭合—抬起—移到盘上—下放—松开），成功以 LIBERO `On(bowl, plate)` 判定（xy <3 cm 且接触）。
- 合格条件：A、B 续作均成功；chunk 回放 eef 差 ≤1e-6；chunk 内碗位移 ≤5 mm；t+7 时 eef_A 与 eef_B 距离 ≥3 cm。
- 结果：**30 个候选，29 个合格**；`t02_ep007` 因 B 续作失败被排除（原因记录在 manifest）。合格状态 t+7 分离 0.0667–0.0762 m（均值 0.0708）。
- 可区分性：真实未来 u_A vs u_B 特征 MSE 0.191（均值），渲染噪声底线 0.0054–0.0090（约 20–35 倍）；R 组实际未来特征按锚点判定 A→A 20/20，B→B 20/20。教师 z：cos(z_A,z_B)=0.79（min 0.66），‖z_A−z_B‖=1.41；策略 z0 与 z_A 的 cos=0.96、与 z_B 0.79；范数 z0 2.12 / z_A 2.19 / z_B 2.15（尺度一致，未见明显分布差）；用 o_8 代替 o_7 提取教师 z 仅偏移 0.22/0.33（远小于 A–B 差 1.41）。

## 5. 四组对照与实际执行数量

固定项：快照、o_0 在线图像、腕部图像、状态、指令、h_vlm（由相同输入确定性重算）、动作后处理、每个种子的初始噪声张量（`conditions/initial_noise.npz` + sha256，3 个种子 101/202/303，跨所有状态相同）。每次执行前完整恢复快照，chunk 内不重新查询策略。短期 chunk 之后运行“与执行标签匹配的脚本续作”得到 full_task_success（OTHER/FAIL 不运行）。

| 阶段 | 状态数 | 核心短期执行 | 续作执行 |
|---|---|---|---|
| pilot（ep000–004） | 5 | 85（R10 + B0 15 + B1 30 + B2 30） | 85 |
| confirm（ep005,006,008–025） | 20 | **340**（R40 + B0 60 + B1 120 + B2 120） | 310 |
| 备用合格状态 ep026–029 | 4 | 未使用 | — |

非核心执行（单独统计）：在线候选采集 30 episode（约 2,850 控制步）；参考构造 120 次（29 状态 × 2 方案 × [chunk+续作, chunk 回放]，含被排除者）；抓取探针 16 个完整 episode；恢复测试约 12 段回放；模型前向：阶段0 约 70 次，条件推理 29×15=435 次策略前向 + 教师/特征提取，补充敏感性 5 状态 × 19 次。基础设施失败：**0**（`failures.jsonl` 为空）。

## 6. 主要指标与两张混淆矩阵（confirm，20 状态；`analysis_confirm/summary.json`, `metrics_by_state.csv`）

**混淆矩阵 1：输入/参考方案 → 预测方案**（按执行记录计数，每状态 3 种子；B1 预测由 LaWM 解码器给出；B2 的“预测”就是替换进去的真实未来，按构造为正确，不算预测能力）

| 输入 | A | B | UNKNOWN |
|---|---|---|---|
| B1 输入 A（z_A） | 57 | 0 | 3 |
| B1 输入 B（z_B） | 48 | **0** | 12 |
| B2 输入 A（u_A） | 60 | 0 | 0 |
| B2 输入 B（u_B） | 0 | 60（按构造） | 0 |
| B0 无干预 | 57 | 0 | 3 |

**混淆矩阵 2：输入/参考方案 → 实际执行方案**（t+7 eef 位置，规则见 protocol.yaml）

| 输入 | A | B | OTHER | FAIL |
|---|---|---|---|---|
| R 参考 A | 20 | 0 | 0 | 0 |
| R 参考 B | 0 | 20 | 0 | 0 |
| B0 | 54 | 0 | 6 | 0 |
| B1 输入 A | 54 | 0 | 6 | 0 |
| B1 输入 B | 54 | **0** | 6 | 0 |
| B2 输入 A | 54 | 0 | 6 | 0 |
| B2 输入 B | 54 | **0** | 6 | 0 |

6 个 OTHER 全部来自同 2 个状态（ep005、ep021）：策略自身 chunk 朝 A 侧移动但 s_e=0.18–0.21，低于 0.25 阈值，并非朝 B；5 种条件下 s_e 相差 <0.006。

按状态汇总（配对 bootstrap，2000 次）：

| 指标 | B1-A | B1-B | B2-A | B2-B |
|---|---|---|---|---|
| 实际方案跟随率 | 0.90 [0.75, 1.00] | **0.00 [0.00, 0.00]** | 0.90 [0.75, 1.00] | **0.00 [0.00, 0.00]** |
| 全链对应率（输入=预测=执行） | 0.90 | 0.00 | 0.90（预测按构造） | 0.00 |
| 双方案均保持的状态数 | B1: 0/20 | | B2: 0/20 | |
| 脚本续作成功（信息性） | 50/60 | 53/60 | 49/60 | 47/60 |

B2 与 B1 的 B 跟随率之差 95% CI = [0, 0]。执行位置差异量化（同状态同种子 t+7 eef 距 B0）：B1-A 0.17 mm、B1-B 0.42 mm、B2-A 0.98 mm、B2-B 1.95 mm（最大 3.1 mm）；B0 种子间 0.27 mm；而 A–B 参考分离 70.8 mm。归一化动作 maxabs 变化：B1-B vs B0 0.009，B2-B vs B0 0.034，B2-A vs B2-B 0.029（参考 A/B chunk 之间为 1.13）。

**未来预测误差（MSE，256×768 特征）**

| 预测 | 对实际 o_7 | 对正确锚点 | 对错误锚点 | s_f（>0 偏 A） | ROI s_f |
|---|---|---|---|---|---|
| B0 预测 | 0.118 | uA 0.135 | uB 0.234 | +0.27 | — |
| B1 输入 A | 0.122 | 0.129 | 0.238 | +0.30 | +0.42 |
| B1 输入 B | 0.132 | **0.209** | **0.152** | **+0.16（仍偏 A）** | +0.20 |

参考：u_A–u_B 0.191；h_t 到 u_A 0.194、到 u_B 0.232。LaWM 解码器对 z 的响应：z_B 使解码未来相对 B0 预测移动 MSE 0.023（需要 ≈0.19 才能到 u_B）；−z0 0.081；3·z0 0.57（能大幅改变输出，说明解码器并非对 z 完全不敏感，而是在真实 z 范围内响应很小）。

**补充敏感性（非预注册，pilot 5 状态，种子 101，`sensitivity.json`）**：动作 maxabs 变化——未来替换为 u_B 0.031、全零 0.110、CFG 无条件嵌入 0.129、朝 u_B 方向 3 倍外推 0.167；z 替换 z_B 0.009、−z0 0.051、3·z0 0.116；仅换指令 0.16–0.47；仅换当前图像（用 o_7 作当前帧）0.18–0.23；换噪声种子 0.013；真实 A/B chunk 差 1.13。

## 7. 代表性案例（`figures/cases/cases.json`，选取规则：按 B2 执行切换分数取最低/中位/最高 + 第一个 pilot 状态，非“最典型失败”）

- `lowest_B2_switch_t02_ep010`、`median_B2_switch_t02_ep014`、`highest_B2_switch_t02_ep017`、`first_pilot_t02_ep000`：面板 `figures/cases/*_panel.png`（o_0 | ref A o_7 | ref B o_7 | B0 | B1-A | B1-B | B2-A | B2-B 的 o_7），对照视频 `videos/cases/*_sidebyside.mp4`（R-A, R-B, B1-A, B1-B, B2-A, B2-B，标注方案/组别/时间，4 fps）。
- 三类结果：成功保持 A（所有状态，输入 A）；无法切换到 B（所有状态，输入 B，执行与 B0 几乎重合）；无法判断（ep005/ep021 的 OTHER，属于策略自身运动幅度小，不是方案差异；B1-B 有 12 条预测 UNKNOWN，是解码器输出落在 A/B 之间且差距低于阈值）。
- 全部参考视频：`videos/reference/`；全部执行视频：`videos/pilot/`, `videos/confirm/`（每条执行记录含路径）。

## 8. 分别分析：模型问题 / 数据分布问题 / 工程问题

**工程（已排除）**：恢复精确（1e-15）、参考回放精确（0.0）、接口等价（0.0）、无失败记录、动作后处理复用官方代码、chunk 内不重查询、噪声跨条件固定并校验。渲染噪声与 2 ms 渲染滞后已量化，不影响结论（模型输入用同一保存图像）。

**表示与预测层（B、C）**：教师 z 与真实未来特征均能区分 A/B（cos 0.79 vs 同方案 0.96；特征 MSE 比噪声底线高 20–35 倍），因此不是“表示不能表达差异”（排除情况 2）。但 LaWM 解码器在给 z_B 时只把预测向 B 移动一小部分（对 uA 0.152 / 对 uB 0.209，仍偏 A），60 条中 0 条被判为 B。可能原因：SFT 中解码器被解冻并只用**策略自身的 z** 做感知损失，而策略 z 对同一状态几乎只对应其固有方案（z0≈z_A，cos 0.96），解码器在 z_B 这一距离（‖Δz‖≈1.4）上没有被训练成可控的方向；这属于训练分布问题与模型接口共同作用，本轮不能区分二者。

**动作头（D，主要发现）**：给定完全正确的真实未来 u_B（绕过 LaWM），动作几乎不变（maxabs 0.034，eef ≤3 mm，而方案切换需要约 70 mm）。即使把未来 token 换成全零或 CFG 无条件嵌入，变化也只有 0.11–0.13；相比之下换指令或换当前图像（进入 h_vlm/h_t）变化 0.16–0.47。机制上与训练配置一致：动作头在训练中只以 `h_t1_pred`（detach）为条件，而 `h_t1_pred` 是 (h_t, h_vlm) 的确定性函数，未来 token 不携带 h_vlm 之外的信息，模型无需依赖它即可拟合。需要说明的解释边界：固定 h_vlm 后替换未来是训练中未出现的条件组合（真实特征与预测特征分布也不同：预测对实际 MSE≈0.12），因此严格地说本轮证明的是“在该 checkpoint 上，未来/潜在动作通道在真实 z/真实未来范围内不是短期动作的控制通道”，而不能断言动作头在所有输入分布下“忽略未来”。

## 9. 当前证据支持什么，不支持什么

支持：
- 情况 4（B2 也无法跟随 A/B），且 B1 同样无法跟随；B0 稳定执行其固有方案 A（+y 缘），任务成功率高（脚本续作 53/60；原策略在线 30/30）。
- 两个真实方案在几何、视觉特征、教师潜在动作三层都可区分；参考控制与恢复接口可靠。
- 现有 checkpoint 中 “z → 预测未来” 对 z 的响应不足以切换方案，“未来 → 动作” 在真实特征范围内接近无响应。

不支持 / 不能下的结论：
- 不能称 “LaWAM 忽略未来” 为普适结论（只测了 1 个任务、1 个 checkpoint、0.4 s 视界、20+5 个状态，且干预属于分布外条件组合）。
- 不能据此声称需要某种新损失或“已证明科学缺口”；也不能评价模型自主生成多方案的能力（本轮候选全部来自 oracle）。
- 续作成功率之间的差异（47–53/60）来自固定脚本对略微不同起点的鲁棒性，不代表方案差异。

## 10. 下一步最小建议（本轮未执行任何训练）

1. 对照实验：用本轮同样的双方案数据（29 状态 × A/B 参考轨迹，含 z_B/u_B）做**普通微调**，再复跑 B1/B2；若跟随率上升，说明是数据分布问题而非结构问题。
2. 若能获得 `enable_flow_h_t1_scheduled_sampling=true`（训练中混入真实未来）的 checkpoint，直接复跑本协议，检验动作头是否学会使用未来 token。
3. 训练侧诊断：度量动作头对未来 token 的归因/梯度占比（无需改变研究问题）。
4. 只有在 1–3 表明未来通道可控后，再研究自主多候选生成与候选选择。

## 附：交付文件

`environment_manifest.json`, `protocol.yaml`, `code_changes.patch`, `phase0_checks.json`（含 `phase0/` 原始 json/png/mp4）, `reference_manifest.jsonl`（30 条，含所有候选与排除原因）, `episode_records.jsonl`（425 条：pilot 85 + confirm 340，每条含 run_id/快照/条件/种子/噪声路径/执行步数/标签/文件路径）, `future_metrics.jsonl`, `analysis_{pilot,confirm,all}/{summary.json, metrics_by_state.csv, confusion_matrices.json}`, `sensitivity.json`, `failures.jsonl`（空）, `execution_counts.jsonl`, `conditions/<state>/model_outputs.npz`（z、u、预测未来、归一化动作、噪声）, `executions/<state>/<tag>.npz`（帧、eef、物体、送入环境的动作）, `reference/<state>/`, `videos/`, `figures/`, `logs/`。
