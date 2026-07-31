# 实验计划

**Problem**：低质量批次是否会通过 optimizer state 产生跨步骤污染。  
**Method Thesis**：将梯度的即时参数影响与持久状态写入分开，可能在不筛样本的情况下限制低质量数据的长期伤害。  
**Date**：2026-07-31

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1（主张）AdamW 状态会传播单次异常批次的影响 | 主线成立的必要条件 | State-only 在 norm matching 后产生非零 clean-loss excess AUC，且持续性高于无动量 SGD | B1, B2 |
| C2（支持）即时/持久解耦可减少污染而不删除难例 | 决定是否值得形成方法 | 候选降低污染 AUC/半衰期，干净性能非劣且开销 ≤5% | B3, B4 |

**Anti-claims to rule out**

- 效应只来自异常梯度 norm 更大；
- 效应只是脉冲步骤造成的参数偏移；
- 差异来自 batch、augmentation 或 RNG 不一致；
- 候选只是 clipping、β 调整或 PNM/AdaPNM 的改名。

## Paper Storyline

- Main paper must prove：状态污染存在；解耦机制有效且低成本。
- Appendix can support：更多 β、pulse severity、full-finetune 边界。
- Experiments intentionally cut：文本第三模态、配置搜索、一般 data valuation。

## Experiment Blocks

### Block 0：Windows 全链路 smoke

- Claim tested：无；只验证基础设施。
- Dataset / split / task：EuroSAT Phase1 既有 Runner。
- Compared systems：无。
- Metrics：状态转换、CUDA 启动、artifact manifest。
- Setup：config 00、seed 99、1 epoch、600s、单并发。
- Success criterion：`HTTP 201 → queued → running → succeeded → artifacts`。
- Failure interpretation：只定位 transport/policy/runner/CUDA/artifact 层。
- Priority：MUST-RUN。

### Block 1：四轨迹因果分解

- Claim tested：C1。
- Dataset / split / task：CIFAR-100，45k/5k，test untouched。
- Compared systems：Control、Parameter-only、State-only、Full。
- Metrics：clean-loss excess AUC、`D_m`、`D_v`、`D_theta`、half-life。
- Setup：ViT-B/16 + LoRA r8；500-step warm-up；128-step deterministic replay；3 seeds。
- Construction：分别生成 clean/corrupt 候选 step 后交叉组合参数与状态；无动量 SGD 的 State-only 必须与 Control 数值一致。
- Success criterion：AdamW State-only 的配对效应方向一致，至少一类 pulse 的 95% CI 排除 0。
- Failure interpretation：若 State-only 无效，optimizer-state 不是可辩护主机制。
- Table / figure target：机制图 1、主表 1。
- Priority：MUST-RUN。

### Block 2：替代解释排除

- Claim tested：C1 与 anti-claims。
- Compared systems：AdamW、无动量 SGD；原始 pulse 与 norm-matched pulse。
- Metrics：与 Block 1 相同，加 replay checksum；固定在 `h={0,1,2,4,8,16,32,64,128}` 评测 validation probe。
- Success criterion：norm matching 后仍存在；AdamW 持续性高于无动量 SGD；所有 replay checksum 一致。
- Failure interpretation：若简单 norm 或即时参数扰动即可解释，停止主线。
- Table / figure target：主图 2、附表 A1。
- Priority：MUST-RUN。

### Block 3：候选解耦机制

- Claim tested：C2。
- Compared systems：AdamW、clipping、β retuning、PNM/AdaPNM、候选。
- Metrics：污染 AUC/half-life、干净 loss、wall-clock、VRAM。
- Success criterion：候选降低主要终点，开销 ≤5%，且优势不被简单基线消除。
- Failure interpretation：若强简单基线等效，不形成新方法。
- Table / figure target：主表 2。
- Priority：MUST-RUN，但仅在 B1/B2 通过后。

### Block 4：最小外部有效性

- Claim tested：C2。
- Dataset：CIFAR-100 clean、instance-dependent noise、mixed defects；后续 FSDnoisy18k。
- Compared systems：最强简单基线、最强直接 noisy-label 基线、候选。
- Metrics：accuracy、calibration、wall-clock、VRAM、clean non-inferiority。
- Success criterion：噪声条件 ≥1pp，干净下降 ≤0.5pp，开销 ≤5%。
- Failure interpretation：只能解释脉冲而不能改善训练，则降级为诊断结果。
- Priority：MUST-RUN（视觉）；音频在视觉通过后执行。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---:|---|---|---|
| S0 | Windows smoke | 1 | 完整状态链与 artifacts | ≤10 min | Runner 版本不一致 |
| M0 | 污染机制 | 12 bundles | C1 五项门禁 | 预计 6–12 GPUh | deterministic replay |
| M1 | 候选去留 | 18–24 | ≥1pp、clean ≤0.5pp、overhead ≤5% | 预计 15–30 GPUh | 简单基线等效 |
| F1 | 视觉正式证据 | 顺序停止 | 3–5 seeds 与真实噪声 | 通过 M1 后估算 | 单卡周期 |
| F2 | 音频外部验证 | 顺序停止 | 冻结规则迁移 | 通过 F1 后估算 | 模态依赖 |

## Compute and Data Budget

- M0 上限：12 experiment bundles。
- M1 上限：24 runs。
- 并发：Windows 单卡固定为 1。
- 数据准备：CIFAR-100 split IDs、pulse manifests、预生成增强。
- Biggest bottleneck：四轨迹 deterministic replay 与 optimizer state 的无泄漏分叉。

## Risks and Mitigations

- CUDA 非确定性：启用确定性算法，保存并核验 RNG、batch IDs 与 augmentation manifest。
- State-only 构造错误：用解析性二次模型单元测试验证状态/参数交换。
- 指标自归一化：统一以 Control norm 加固定 epsilon 为分母，保存原始值。
- 测试集泄漏：M0 不加载 CIFAR-100 test。
- 服务器契约漂移：smoke 先验证；M0 使用新 project ID 和版本化 matrix。

## Final Checklist

- [x] Main paper tables are covered
- [x] Novelty is isolated
- [x] Simplicity is defended
- [x] Frontier contribution is explicitly not claimed
- [x] Nice-to-have runs are separated from must-run runs
