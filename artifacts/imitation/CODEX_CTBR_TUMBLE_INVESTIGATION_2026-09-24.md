# Circular-12 BC 单步角速度 spike 排查（2026-09-24）

## 结论

已复现 `bc_14_verticalnorm_v1.pt` 的单步巨大角速度。**直接原因是物理碰撞冲量（A），不是 CTBR/PID/混控输出过大。**此前显示 `contact_force_max_n=0` 的记录取自 `wrapped.step(action)` **之前**；撞击发生在该控制步最后一个 2.5 ms 物理子步，之后环境因 collision 立即重置。因此，旧日志把撞击前的 0 N 与下一状态的大角速度排在相邻行，造成接触力与角速度错配（B：logger 采样时序）。

这解释了 spike，但**没有消除 BC 撞门**。BC 仍会偏离专家轨迹、贴近门框；现无充分证据支持改 PID、限力矩、改碰撞阈值或伪造过门。因此本次未修改控制/评分源码、未重训，不能声称达到 `zero_tumble_gate=true`。不要进入 PPO/估计器阶段。

## 复现证据

在 `ExpertDemo-v0`、seed=1、14 m/s 下，独立逐子步探针首次捕获 >10 rad/s 的事件：第 2 回合第 564 个 10 ms 控制步、Gate 1 附近。

| 量 | 撞击前 | 第 1–3 个 2.5 ms 子步 | 第 4 个子步 |
| --- | ---: | ---: | ---: |
| 机体角速度模长 | 1.093 rad/s | 约 1.09 rad/s | **56.097 rad/s** |
| PhysX 直接接触力 | 0 N | 0 N | **2097.824 N** |
| contact sensor 接触力 | 0 N | 0 N | **2097.824 N** |

同一步 BC–Expert 动作 MAE = **0.03594**；BC 归一化动作 `[0.255, -0.095, -0.173, 0.291]`，并无极端指令。PID 的 P/I/D 都在毫牛·米量级；最终实际力矩 `[-0.00356, 0.00307, -0.00179] N·m`。`omega_ref == omega_real`（当前 motor model disabled），allocation 的 actual wrench 与 desired wrench 一致。撞击前没有 gate pass/miss；事后 gate index 回到 0 是 episode reset，**不是切 target 引起的 action discontinuity**。

机身主刚体惯量取 PhysX `get_inertias()`；用第 3→4 子步的 `I·Δω/0.0025 s` 粗估等效力矩约 `[-5.50, -8.70, 72.69] N·m`，模长 **73.4 N·m**。这是角动量量级检查，不是含约束/陀螺项的精确逆动力学；但它远高于 CTBR 的 `(0.28, 0.28, 0.14) N·m` 限值，也远高于实际输出。数据在 [first_spike_gt10.json](codex_ctbr_debug/first_spike_gt10.json)。

另一独立复现（第 1 回合第 512 步、Gate 12 附近）：约 1.00→6.99 rad/s，撞击子步接触力 2111 N，实际力矩约 0.001 N·m；见 [first_spike.json](codex_ctbr_debug/first_spike.json)。两次都只在**接触力出现的同一物理子步**跳变。

Gate 1 事件撞前机身位置约 `(-0.298, 1.031, 2.084) m`，该门中心约 `(0.015, 0.013, 2.078) m`；高度远离地面、位置贴近门框区域。日志确认发生了物理接触，但没有接触对的 prim 名称，故不把具体碰撞物体身份说成已被直接测定。

## 控制链与时序核查

- 环境 `sim.dt=1/400=0.0025 s`、`decimation=4`，控制步为 0.01 s；PID 用 `env.step_dt`，motor（若启用）用 `env.physics_dt`。
- normalized action 被限制在 `[-1,1]`，期望 body rate 上限为 `(4,4,2) rad/s`；PID D 项仅对实测 rate 求导，不对 command 求导，积分有边界，moment command 有硬限幅。
- allocation 按有限 rotor thrust 分配，并输出 `omega_ref`；禁用 motor model 时 `omega_real=omega_ref`；复现时 actual wrench 与 desired wrench 数值匹配，排除混控巨大力矩。
- Isaac Lab 本地 `articulation.py::set_external_force_and_torque` 接受的是**机体系**力和力矩；`write_data_to_sim` 进一步调用 PhysX `apply_forces_and_torques_at_position(..., is_global=False)`。当前 API 没有要求调用方传 `is_global` 的参数；CTBR 施力方向/单位未发现错误。
- `ManagerBasedRLEnv.step` 在 4 个物理子步之后才算 termination 并 reset。旧 evaluator 的接触读取在 `wrapped.step` 前，只能看到上一步结果；而 spike 步撞击后立即 reset。逐子步探针直接比较 sensor 与 PhysX view，两者同子步同为约 2.1 kN，排除这两次事件的 contact sensor 覆盖缺失。
- Gate target 在上述 >10 rad/s 事件里没有 pass/miss 转换；不支持 G（切门时序）为直接诱因。H（BC 累积轨迹误差）可解释为何会撞门，但不是角速度在一个子步内爆发的力源。

## 质量基线与未完成项

现有 20 回合 BC contactdiag audit：平均过门 **18.2**、完整单圈率 **45%**、inversion **5**、gross attitude excursion **9**、`zero_tumble_gate=false`。相同示范任务的专家 20 回合：平均过门 **44.7**、完整单圈率 **100%**、inversion/gross 均 **0**。见 [BC summary](bc_14_verticalnorm_v1_contactdiag/summary.json) 与 [expert summary](expert_demo_audit_14/summary.json)。

**After：未产生。**没有经过证实且不作弊的最小控制器修复可消除 BC 撞门；因此未运行“修复后 20 回合”审计，也未宣称验收目标达成。后续应在保持真实碰撞和真实过门评分的前提下，针对临门边缘的 BC 轨迹偏差/安全裕量开展新的示范与闭环验证；这是策略质量问题，不应靠屏蔽 tumble 或改碰撞检测掩盖。

## 复现命令

```bash
./.conda-env/bin/python scripts/imitation/debug_ctbr_spike.py \
  --checkpoint artifacts/imitation/bc_14_verticalnorm_v1.pt \
  --output artifacts/imitation/codex_ctbr_debug/first_spike_gt10.json \
  --episodes 5 --seed 1 --spike-threshold-radps 10 \
  --device cuda:0 --headless
```

原 20 回合审计（不变更评分）：

```bash
./.conda-env/bin/python scripts/imitation/evaluate_circular12_flight_quality.py \
  --task Isaac-Drone-Racer-Swift-CTBR-GT-Circular12-ExpertDemo-v0 \
  --controller bc --checkpoint artifacts/imitation/bc_14_verticalnorm_v1.pt \
  --episodes 20 --target-speed-mps 14 \
  --output-dir artifacts/imitation/codex_ctbr_debug/audit20 \
  --device cuda:0 --headless
```

本次仅新增临时诊断脚本 `scripts/imitation/debug_ctbr_spike.py` 和本报告；未修改生产控制逻辑、未 commit/push。

## 验证

- 逐子步仿真复现：2 次，均捕获同子步碰撞力与角速度跳变。
- `python -m py_compile scripts/imitation/debug_ctbr_spike.py`：通过。
- `pytest` 的仿真依赖测试收集在普通 Python 下因缺 `omni.kit` 失败；非仿真 imitation/静态子集 15 passed。较宽静态子集 52 passed、2 failed；失败分别是旧 GT policy deployment contract 文本检查、旧 multigate trainer 文本检查，与本次新增脚本无关。
- `git diff --check`：通过；新增文件另以尾随空格扫描检查，未发现尾随空格。
