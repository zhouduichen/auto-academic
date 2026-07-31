# Research Brief

> 输入对象：ARIS `/idea-discovery`
>
> 当前阶段：主线已冻结，进行 optimizer-state 定向查新；禁止启动新 GPU 实验
>
> 成果门槛：CCF-A Full/Regular 或学校正式认定 T1

## Problem Statement

用户在单张桌面端 RTX 5080 上训练自己的模型时，训练集质量往往低于预期：可能同时存在错标、输入退化、分布外样本、重复低信息样本和类别不平衡。现有方法常把问题简化为已知噪声率下的人工对称错标，或依靠双模型、多阶段清洗、外部大模型和额外干净数据，未必适合固定单卡预算。

冻结的研究主线是：分析低质量批次如何通过 AdamW/动量的一阶、二阶状态产生跨步骤污染，并设计一种低成本机制，将梯度的即时影响与持久状态写入解耦。目标是在未知混合缺陷下，不筛样本、不知道噪声率、不使用干净参考集或额外模型，改善单卡 PEFT 训练。

## Background

- **Field**：机器学习、数据中心 AI、鲁棒训练、参数高效微调。
- **Sub-area**：低质量数据训练、noisy-label learning、robust optimization、momentum/Adam dynamics、参数高效微调。
- **Mandatory closest works**：PNM/AdaPNM、robust momentum/Adam、gradient clipping/aggregation，以及 CleaR、TURN、IDO、DSS、CADS。
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
- [ ] 完成 optimizer-state 候选的最后定向查新和顶会级批判性评审。
- [x] 冻结带停止条件的单卡长期实验路径。
- [ ] 现在写论文。
- [ ] 现在启动大规模训练。

## Domain Knowledge

- 一次有害梯度不仅影响当前更新，还可能通过动量和二阶矩改变后续正常更新。
- 直接删除异常梯度可能同时删除困难但正确的学习信号；因此主线区分即时影响与持久影响。
- “更鲁棒”若来自更多训练、额外模型、更强数据增强或额外干净集，在固定计算预算下可能并不成立。
- 候选必须产生“污染半衰期缩短”的可证伪预测，并优于 clipping、SGD 和 PNM/AdaPNM。

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
