# 实验与项目计划

> 状态：当前唯一权威执行计划
>
> 更新：2026-08-01
>
> 当前门：E0 证据封存
>
> 计划边界：只推进到 M1 候选机制的去留；后续数据集、论文和通用平台不在当前范围

## 1. 目标与原则

本项目回答一个聚焦问题：低质量批次是否会通过 AdamW 持久状态损害后续正常学习，以及限制可疑梯度的持久状态写入能否在不删样本、不使用干净参考集、不增加前后向的条件下改善单卡 PEFT 训练。

执行原则：

1. **先证伪，后确认**：候选方法先通过最小 Pilot，有稳定信号才设计正式确认。
2. **原始证据先于文档结论**：未绑定原始产物和哈希的 PASS 只能记为 `REPORTED`。
3. **统计需求决定运行数**：不预先假设“8 seeds 就足够”，也不以无上限加 seed 挽救小效应。
4. **现有链路优先**：复用已通过的直接 PowerShell、Python runner 和 artifact 哈希模式；只修复会影响正确性、证据完整性或精确恢复的问题。
5. **测试集隔离**：M1 期间不构造、不加载 CIFAR-100 official test。

## 2. 当前证据账本

| 证据 | 原始位置 | 已支持的结论 | 不支持的结论 | 状态 |
|---|---|---|---|---|
| 旧 EuroSAT 48 runs | `experiments/phase1_eurosat/outputs/` | 运行时间、显存和 LoRA 开发区域 | noisy-data 有效性或方法选择 | `PILOT_ONLY` |
| M0 四轨迹 | `tmp/m0-full-matrix-20260731.zip` | label-flip 的 AdamW state-only AUC 三个 seed 均为正，均值约 `0.6543`；SGD state-only 为 `0` | input degradation 的效应稳定性 | `VERIFIED_LOCAL` |
| M0.5 载体归因 | `tmp/m05-full-20260731.zip` | `m-only` AUC 三个 seed 均为正 | Protect-M 或 Protect-MV 的结构选择；leave-one-seed-out 结论不稳定 | `INCONCLUSIVE_STRUCTURE` |
| M0.6 独立确认 | Windows 原始包尚未绑定到本计划 | 后续文档记录为正式 PASS | 在本地原始包、哈希和独立重算完成前，不作为新 GPU 矩阵的唯一授权 | `REPORTED_NEEDS_BINDING` |
| M1 calibration | Windows；clean 源码起点 `fefb50f`，noisy 源码 `1e566ba` | clean 已运行，noisy 当前正在运行；相关本地测试 `5 passed`、Ruff 通过 | 四个 cells 未入库前，不固定 Pilot 预算或宣称性能 | `NOISY_RUNNING` |

当前只有两个科学结论可以固定：

- 标签翻转场景下，AdamW 持久状态具有可测的跨步影响；
- 现有证据不足以在 Protect-M 和 Protect-MV 之间做出结构选择。

## 3. 执行漏斗

### E0：证据封存

**目标**：把已发生的实验变成可追溯的决策输入，不开新科学矩阵。

必做：

1. 从 Windows 收回 M0.6 bridge、20 个 evidence bundles 和 `COMPLETE.json`；核对 source commit、文件哈希、`test_loaded=false`、完整运行数和失败/重试记录。
2. 按冻结规则独立重算 M0.6 的 `M`、`A_both`、95% 区间、符号一致性和 pulse 异质性；记录 `PASS` 或 `FAIL`，不改阈值。
3. 收回 clean calibration seeds `101/102` 和正在运行的 noisy calibration seeds `101/102`；核对 source commit、split、noise/input binding、步数、有限值、恢复记录、时间和显存。不干预当前运行，不因结果不理想重跑。
4. 对 noisy cells 从私有 audit/source 重生成并比对标签哈希，独立确认 `train 101 -> noise 1101`、`train 102 -> noise 1102`；同时确认完成目录不是由其他 commit、config 或 bundle 复用而来。

**通过门**：M0.6 原始包完整且独立重算通过；clean/noisy 各两个 calibration cells 均形成完整、可审计的有效产物，足以冻结 Pilot 公共训练预算。

**失败处理**：M0.6 科学门失败则停止 optimizer-admission 方法线；产物不完整则先恢复证据，不用新运行替换旧运行。calibration 的已验证系统故障只允许一次完全同配置重试；重试仍失败则 E0 为 `INCONCLUSIVE`，不启动 Pilot。

### M1-P0：最小实现门

**目标**：只交付 Pilot 必需的候选机制、噪声数据和可比性检查。

必做：

1. 从现有 Protect-M/Protect-MV 草案中冻结一个共享 detector 和一个预先声明的默认配置；Pilot 不做 detector grid search。精度直接继承已运行 calibration 的 FP32/AMP-disabled 路径，不新增 AMP 设计任务。
2. 两个候选只允许“拒绝步是否保护 `v`”一处差异；关闭保护时必须与 AdamW 一致。
3. 使用现有 `m1_noisy_data.py` 路线生成严格 40% instance-dependent noise；只增加 Pilot 冻结 noise seeds `1301–1303` 的入口，并补充一个最小检查，覆盖确定性、每类 160 个错标、公开/私有字段隔离和篡改拒绝。
4. 审核 CAdam 的原论文公式、官方代码/许可证和 ViT-LoRA 兼容性。CAdam 作为当前最近的机制基线，不用新建基线平台。
5. 在 CPU 做 AdamW parity、state hold/commit、checkpoint resume 和禁止信息输入检查；在 GPU 只做一个短 sentinel，确认单次前后向、无 OOM/非有限值且开销不超过 5%。

**通过门**：实现一致性、数据隔离、短运行与开销检查全部通过。

**失败处理**：修正确定性实现错误后可重测；机制不可实现、需要额外前后向或无法满足开销门则停止。

### M1-Pilot：24-run 证伪实验

**目标**：用最小、成对的矩阵判断候选机制是否值得进入调参和正式确认。Pilot 是开发证据，不是论文确认证据。

| 维度 | 固定值 |
|---|---|
| 方法 | AdamW、CAdam、Protect-M、Protect-MV |
| 条件 | clean、同一协议的 40% instance-dependent noise |
| 训练 seeds | `301, 302, 303` |
| noise seeds | `1301, 1302, 1303` |
| augmentation seeds | `10301, 10302, 10303` |
| 总数 | `4 methods × 2 conditions × 3 paired seeds = 24 runs` |
| 数据门 | tuning validation 用于 epoch 曲线；development gate 只在全部方法/配置冻结后打开；official test 始终封闭 |

公共训练预算由 clean/noisy 四个 calibration cells 共同产生。对每个条件，先按 epoch 计算两个 seeds 的 tuning accuracy 算术均值 `a_e` 和全程最大值 `a_max`；选择最早的 epoch `e`，使 `a_e >= 0.99 * a_max`，且 `e..e+5` 六次评估均满足 `a_j >= a_max - 0.002`。若不存在该 epoch，该条件取 `50`。最终预算为 clean/noisy 两个 epoch 的较大值，限制在 `[20, 50]` 并转为精确 optimizer-step 数。对所有方法固定相同 optimizer steps、data accesses、evaluation cadence 和批次/增强 manifest。Pilot 不做 best-checkpoint 报告和中途停止。

候选 `c` 相对 CAdam 的主要开发量为：

```text
Delta_noisy[c,s] = Acc_noisy[c,s] - Acc_noisy[CAdam,s]
Delta_clean[c,s] = Acc_clean[c,s] - Acc_clean[AdamW,s]
```

候选只在以下条件全部成立时进入下一门：

- 三个 `Delta_noisy` 均为正，且均值至少 `+1.0` 个百分点；
- 相对 AdamW 的平均 clean drop 不超过 `0.5` 个百分点，任一 seed 不超过 `1.0` 个百分点；
- 固定时间的 noisy 改善为正，平均 wall-clock 和峰值 VRAM 开销均不超过 `5%`；
- 不使用已知噪声率、干净标签、样本删除、额外模型/阶段/前后向；
- 配对、数据隔离、有限值、失败/重试和 artifact 哈希审计全部通过。

若两个候选都通过，默认选择更简单的 Protect-M；只有 Protect-MV 的平均 noisy 优势比 Protect-M 至少高 `0.5` 个百分点、三个 seed 均不差，且没有额外超限开销时才选 Protect-MV。

Pilot 不做 p-value，不因边界结果加 seed：

- 无候选通过：`NO-GO`，停止方法线；
- 完整运行因系统故障无法形成：`INCONCLUSIVE`，只允许精确恢复或一次同配置系统重试；
- 至少一个候选通过：`PILOT-GO`，冻结候选并进入确认设计。

### M1-Confirm：条件式正式确认

本阶段现在不启动、不生成运行矩阵。`PILOT-GO` 后只在本文件中增补一次并经用户审批，不新建规划文档。

届时必须冻结：

1. 候选结构、源码、有限调参空间和 Pilot 中未使用的确认 seeds；
2. AdamW、最强简单机制基线、冻结候选和一个直接 noisy-label 基线；
3. 有限、对称的调参预算，并在打开 confirmation gate 前冻结每个方法的唯一配置；不恢复现有的六方法 × 12 配置通用平台；
4. 基于 Pilot 配对方差的保守功效计算：noisy superiority 使用两侧 `alpha=0.05`、目标效应 `1.0pp`，clean non-inferiority 使用单侧 `alpha=0.05`、边界 `0.5pp`，目标 power 至少 `80%`；
5. 确认 seed 数不少于 `8`、不超过 `12`。若保守功效计算需要超过 `12` 个配对 seeds，该效应不适合当前单卡预算，直接 `NO-GO` 而不降低统计标准。

只有 M1-Confirm 通过，才能规划真实噪声数据或论文证据。

## 4. ARIS 的有限作用

当前只复用 ARIS 的三个架构原则：

```text
计划 -> 最小实现/短 sentinel -> 原始产物 -> 独立审阅 -> 下一门
```

- `EXPERIMENT_PLAN.md` 描述跑什么与停止条件；
- `EXPERIMENT_TRACKER.md` 只描述当前状态；
- 审阅者直接读取原始 artifacts，不使用执行者摘要代替证据；
- E0、M1-Pilot 和 M1-Confirm 各审阅一次，不在每个小步生成新文档或启动循环审稿。

ARIS 的 W2/W3、paper pipeline、research wiki、全量 skill 控制层和通用实验编排不是当前交付。

## 5. 基础设施与文档冻结

当前不执行：

- Windows Coordinator、通用队列、新状态机或新 Control API；
- 通用 `m1_protocol`/Schema/Artifact DAG/后续功效平台；
- 六方法 × 12 配置的大规模调参框架；
- 新数据集、新骨干、音频、full fine-tuning 或论文生成；
- 为每个决策新建 spec、plan、schema 和带时间戳副本。

只有以下三类问题允许修改底层链路：

1. 会改变科学结论的正确性缺陷；
2. 会破坏数据隔离、配对、有限值或 artifact 完整性的缺陷；
3. 阻止已批准运行完成或精确恢复的实际故障。

## 6. 当前顺序与总预算

```text
收回/审计 M0.6 和 clean/noisy calibration
  -> 冻结最小候选与噪声链路
  -> CPU 检查 + GPU sentinel
  -> 24-run M1-Pilot
  -> NO-GO / INCONCLUSIVE / PILOT-GO
  -> 仅 PILOT-GO 后计划 8–12 seed 正式确认
```

- E0：不新增科学 GPU 运行；只等待当前 noisy calibration 完成并收回已有结果。
- M1-P0：一个短 GPU sentinel，其余为 CPU/本地审计。
- M1-Pilot：严格上限 24 runs，Windows 单并发。
- M1-Confirm：当前不占用预算；将来上限 12 paired seeds，超出则停止。

## 7. 结束定义

当前计划在下列任一情况结束：

- E0 否定 M0.6 的冻结机制门；
- M1-P0 无法交付一个符合单模型、单阶段、单次前后向和 5% 开销门的候选；
- M1-Pilot 返回 `NO-GO`；
- 正式确认所需样本超过单卡预算上限；
- 最近文献或官方代码已等价实现核心 recurrence；
- 收益必须依赖干净参考、已知噪声率、额外模型/阶段或样本删除。

到达结束定义后，先保存负面证据和失败解释，再决定是否返回新问题；不通过新平台、新数据或降低门槛挽救当前主线。
