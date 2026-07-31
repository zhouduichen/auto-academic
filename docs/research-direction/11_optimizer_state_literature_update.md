# 11 — Optimizer-State 定向文献更新

## 范围与截止时间

本轮 L0 检索冻结于 2026-07-31（Asia/Shanghai），目标不是为尚未生成的 M1 候选方法证明新颖性，而是候选无关地回答三个问题：噪声或异常梯度如何污染 Adam 类持久状态；已有工作如何检测、抑制、重置或稳健化这种污染；严格单模型、单阶段、无干净参考、未知噪声率和低额外算力条件下还剩下什么机制空间。

默认时间窗为 2022–2026；只有定义优化器状态、稳健动量或关键基线所必需的基础论文向前回溯。ARIS 只用于发现、去重和审计留痕；技术事实仅以正式论文主页、官方论文 PDF、DOI 元数据或作者官方代码为依据。

## 已有证据状态

- 工作区中不存在独立的 `papers/` 或 `literature/` 本地论文库，不能把缓存或搜索结果误计为已下载全文。
- `.aris/verify-papers/candidate_papers.json` 含 40 条候选；辅助验证记录的总体 verdict 是 `WARN`，reason code 为 `verify_papers_invocation_failed`。原因是 arXiv TLS 握手反复超时，因此 40 条在 ARIS helper 层全部保持 `UNVERIFIED`。
- `docs/research-direction/09_adjacent_work_claim_matrix.csv` 含 12 条逐篇深读记录。这些记录分别以官方 proceedings、arXiv 或期刊页面复核，但不会反向改写上述 ARIS forensic verdict。
- 现有 12 条主要覆盖 noisy-label PEFT、数据选择、训练预算和时间统计；它们是本轮检索的输入，不足以替代 optimizer-state 定向检索。

## 检索主题与查询注册表

首轮共预注册 18 条查询，每个主题各三条，完整机器可读记录位于 `.aris/literature/l0_search_log.jsonl`。

| 主题 | 目标 | 查询数 |
|---|---|---:|
| `optimizer_state_contamination` | 噪声/异常梯度对一阶、二阶矩及未来步的持续影响 | 3 |
| `selective_moment_update` | 跳过、门控或分离当前参数更新与持久状态写入 | 3 |
| `robust_adam_momentum` | 稳健动量、重尾梯度与 spike-aware Adam 变体 | 3 |
| `gradient_agreement` | 时间一致性、梯度方向/余弦对齐和样本选择 | 3 |
| `noisy_peft` | noisy-label PEFT/LoRA 与优化器机制的交叉 | 3 |
| `low_overhead_direct` | 单模型、单阶段、无干净集、低额外开销边界 | 3 |

注册时间为 2026-07-31T23:18:33+08:00。注册与执行分开记录：当前 18 条均为 `search_registered`，后续每次实际检索会追加来源、时间、结果引用、去重键和验证状态，不覆盖原始注册条目。

## 候选发现与去重

发现结果将按 `DOI > arXiv ID > normalized title` 去重。正式发表版取代预印本的 venue 元数据，但保留 arXiv 链接。首轮发现尚未开始，因此本节目前只定义审计口径，不作数量或新颖性判断。

## 直接威胁深读

“直接威胁”指可能实现相同持久状态控制合同、控制同一状态载体，或在相同约束下形成强简单基线的工作。所有此类工作都必须进入公式级深读；当前尚未从本轮注册查询中形成直接威胁队列。

## 主题综合

综合只在主来源验证和直接威胁深读后形成。报告将严格区分：对当前 batch 参数更新的影响、对未来步骤持久状态的影响、检测信号、额外模型/forward/backward/stage，以及是否需要干净参考或已知噪声率。

## 饱和检查

关闭 L0 需要两轮互不重复的反查查询连续不再增加新的 `critical` 或 `deep_read` 工作。首轮 18 条查询尚未执行，因此当前不宣称饱和。

## L1 待候选冻结的精确问题

L1 只在 M1 数据支持并冻结精确 recurrence 后启动，逐项检查候选的状态载体、门控信号、当前更新效应、持久效应与每个直接先验是否数学或实现等价。L0 不提前填写 candidate-overlap，也不把内部的 A/T1 证据门槛表述为会议录用保证。
