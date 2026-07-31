# Research Brief

> 输入对象：ARIS `/idea-discovery`
>
> 当前阶段：文献地形、候选机制查新与研究评审；禁止启动新 GPU 实验
>
> 成果门槛：CCF-A Full/Regular 或学校正式认定 T1

## Problem Statement

用户在单张桌面端 RTX 5080 上训练自己的模型时，训练集质量往往低于预期：可能同时存在错标、输入退化、分布外样本、重复低信息样本和类别不平衡。现有方法常把问题简化为已知噪声率下的人工对称错标，或依靠双模型、多阶段清洗、外部大模型和额外干净数据，未必适合固定单卡预算。

当前研究域是：在未知且可能混合的数据缺陷下，如何估计样本对当前训练过程的真实价值，并在固定 GPU 预算内分配有限计算。这里必须区分“样本/标签是否可信”和“继续训练该样本能否带来干净评测收益”。单卡、PEFT、图像或音频都是约束与验证场景，不单独构成贡献。

## Background

- **Field**：机器学习、数据中心 AI、鲁棒训练、参数高效微调。
- **Sub-area**：低质量数据训练、noisy-label learning、sample selection、data valuation、curriculum learning、compute-aware training。
- **Mandatory closest works**：CleaR（ACL 2024）、TURN（IJCAI 2024）、Delora（ACL Findings 2025）、RACT（2026 preprint）、Normalized Losses（ICML 2020）。
- **What I already tried**：EuroSAT + ViT-B/16 + LoRA，16 个配置 × 3 seeds，共 48 次开发性运行。
- **What did not establish the new claim**：旧实验训练标签干净，且存在 epoch 不一致、split/train seed 未分离、把 `config × seed` 当作选择单位、重复模拟 RNG 未变化、测试集已参与开发判断等问题。

详细材料：

- `docs/superpowers/specs/2026-07-31-high-quality-noisy-data-research-design.md`
- `docs/research-direction/07_recommendation.md`
- `docs/research-direction/08_novelty_audit_protocol.md`
- `experiments/phase1_eurosat/`

## Constraints

- **Compute**：一张桌面端 RTX 5080；最终必须同时报告 wall-clock、optimizer steps、数据访问次数、峰值显存、额外模型/阶段和存储。
- **Pilot budget**：通过查新后，机制探针最多 12—24 次运行；该数字是上限，不是预设网格。
- **Target**：只接受 CCF-A Full/Regular 或学校正式 T1 的证据路径。
- **Evidence discipline**：技术结论使用原论文/正式 proceedings；预印本必须标注；测试集只在方案冻结后使用。
- **Modality**：视觉用于最小机制开发；真实噪声音频可作外部验证；文本仅在最终主张确实需要第三模态时加入。

## What I'm Looking For

- [x] 从明确现实问题出发发现可验证的新机制。
- [x] 对候选机制进行最近工作查新和顶会级批判性评审。
- [x] 只为通过门禁的候选生成单卡实验计划。
- [ ] 现在写论文。
- [ ] 现在启动大规模训练。

## Domain Knowledge

- 可靠性高不等于训练价值高：困难但正确的样本可能有高价值；重复的干净样本可能边际价值很低。
- 现有小损失筛选容易把困难干净样本误判为噪声。
- “更鲁棒”若来自更多训练、额外模型、更强数据增强或额外干净集，在固定计算预算下可能并不成立。
- 有希望的机制必须产生最近方法没有的可证伪预测，而不只是组合已知信号或更换 LoRA rank。

## Non-Goals

- 不研究 AutoResearch 或 ARIS 本身。
- 不把 PEFT 配置选择可靠性、`/selection-audit`、LCB/Bootstrap 包装成主贡献。
- 不把 CleanLab、鲁棒损失、双 adapter、clean routing 或 rank schedule 的直接拼装命名为新方法。
- 不为了“通用”而机械铺开图像、音频和文本全部组合。
- 不以 CCF-B/T2、Findings、Short、Demo、Workshop 或预印本作为达标成果。

## Existing Results

旧 48 次运行不是正式证据，但仍可用于：

- 单卡训练链路、运行时间和显存估计；
- LoRA 的开发性超参数区域；
- 逐样本 logits 和日志格式；
- 发现正式协议必须独立设置 `split_seed`、`train_seed`、`noise_seed` 并保存 sample IDs。

它不能用于证明 noisy-data 方法有效，也不能继续用同一测试集选择正式方法。
