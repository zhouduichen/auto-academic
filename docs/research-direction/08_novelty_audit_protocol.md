# 08 — Novelty Audit Protocol

> Frozen date: 2026-07-31
>
> Publication floor: CCF-A Full/Regular paper or formal institutional T1
>
> Authorized activity: literature and existing-artifact analysis only

## 1. Research domain

本次审计围绕以下现实问题寻找 A/T1 级研究空白：

> 在单张桌面端 RTX 5080 的固定计算预算下，面对错标、输入退化、分布外样本、重复低信息样本和不平衡等未知或混合数据缺陷，如何让模型把有限训练计算用于真正有价值的信号？

问题域不等于最终论文题目。“单卡”“PEFT”“噪声”“跨模态”都只是约束或场景，不能单独构成新颖性。

AutoResearch 仅提供固定预算、基线先行、逐轮验证和结果留痕的实验规划参考。ARIS 仅提供计划、完整性、结果—结论和拒稿理由审计参考。二者都不是研究对象或论文贡献。

本阶段不是另建一套文献分析流程，而是执行 ARIS 的 `/idea-discovery`：

```text
/research-lit
  → /idea-creator
  → /novelty-check
  → /research-review
  → /research-refine-pipeline
```

- `RESEARCH_BRIEF.md` 是该流程的输入；
- `docs/research-direction/09_adjacent_work_claim_matrix.csv` 是经论文真实性校验后的持久证据表；
- `idea-stage/IDEA_REPORT.md` 是唯一的人类可读主报告，文献地形、候选、查新、评审和决定均折叠进入该文件；
- 外部评审原始记录只保存在 `.aris/traces/`；
- `/kill-argument` 留到论文主张和完整草稿稳定后使用，不能替代当前的 idea-level `/research-review`；
- `/prior-art-search` 是专利/FTO 工作流，本次学术查新不调用。

## 2. Definitions

### 2.1 Data defect

`data defect` 指可能降低训练价值的数据问题，包括：

- 标签错误；
- 模糊、截断、压缩、低信噪比等输入退化；
- 与目标任务无关的分布外样本；
- 重复、近重复或低信息样本；
- 类别不平衡及其与噪声的耦合。

### 2.2 Sample reliability

`sample reliability` 指当前样本及其观测标签有效的概率。它回答“这个样本是否可信”，不等价于该样本对训练是否有价值。

### 2.3 Sample training value

`sample training value` 指在指定模型状态和计算预算下，对该样本投入一个单位训练计算所带来的预期干净评测收益。困难但正确的样本可能可靠性较高且训练价值较高；重复的干净样本可能可靠性高但边际训练价值低。

### 2.4 Compute budget

`compute budget` 同时包含：

- GPU wall-clock；
- optimizer steps；
- 样本/数据访问次数；
- 峰值显存；
- 辅助模型与额外训练阶段；
- 逐样本状态和中间产物存储；
- 外部模型、人工审核或重标注成本。

### 2.5 Direct prior

`direct prior` 指只需替换数据集、骨干模型或参数值，就能实现候选主张和核心机制的已有工作。

### 2.6 Irreducible difference

`irreducible difference` 指若用最近已有方法替换，候选机制的独特预测或行为就会消失的结构差异。换模态、换损失系数、换 LoRA rank、增加已知信号的加权和都不自动构成不可约差异。

## 3. Search sources and query families

### 3.0 ARIS source routing

`/research-lit` 使用 `web, semantic-scholar, openalex`，并在可用时加入本地论文库；arXiv 元数据通过 ARIS 固定适配器检索。所有候选先经 `verify_papers.py` 的 arXiv → CrossRef → Semantic Scholar 三层校验，再进入分析。无法验证的条目保留并明确标记 `UNVERIFIED`，不得静默删除或当作已确认文献。

检索阶段只做发现、去重、论文事实提取和主题综合。候选机制生成后，`/novelty-check` 才按 3—5 条核心技术主张进行最近六个月并发工作查新；随后 `/research-review` 以 NeurIPS/ICML 审稿标准评估创新、可证伪性、证据缺口和单卡可行性。

### 3.1 Primary sources

技术结论只使用：

- PMLR；
- ACL Anthology；
- CVF Open Access；
- OpenReview；
- AAAI Proceedings；
- IJCAI Proceedings；
- ACM Digital Library；
- IEEE Xplore；
- JMLR；
- 论文作者提供且由正式论文链接的代码或项目页。

arXiv 仅用于尚未正式发表的近期工作，并必须标注 `preprint`。博客、聚合站、新闻稿和二手综述只能发现线索，不能支持 claim-level 结论。

### 3.2 Query families

每组检索词分别与 `2024`、`2025`、`2026` 组合，并追踪核心工作引用链：

```text
robust parameter-efficient fine-tuning noisy labels
LoRA label noise memorization rank curriculum
PEFT clean routing noisy supervision
dual adapter noisy label detection
sample valuation training dynamics noisy data
forgetting events mislabeled examples
hard clean samples versus mislabeled sample selection
gradient alignment sample reweighting noisy labels
influence functions data valuation pretrained models
coreset selection low-quality data fixed compute
compute-aware curriculum learning fixed budget
single-model noisy-label learning efficient
mixed corruption label noise OOD input degradation
open-set noisy label learning out-of-distribution contamination
noisy long-tailed learning pretrained models
duplicate low-information sample training value
audio noisy labels parameter-efficient fine-tuning
web audio weak labels robust training
cross-modal sample quality estimation
cross-modal noisy supervision robust learning
data cleaning AutoML budget allocation
data-centric AI compute allocation training
unknown noise rate robust fine-tuning
real-world noisy labels foundation model adaptation
```

## 4. Inclusion criteria

纳入工作必须满足：

1. 改变低质量数据下的训练、数据使用、样本选择、路由、加权、标签处理或计算分配；
2. 研究预训练模型，或其机制能够合理迁移到 PEFT；
3. 方法与实验信息足以进行 claim-level 比较；
4. 2024—2026 工作强制覆盖；更早工作只在定义强基线或核心机制时纳入；
5. 同一工作的预印本与正式版本只保留一行，以正式版本为准；
6. 最终矩阵至少包含 50 篇直接相关工作，而不是用宽泛综述或弱关键词匹配凑数。

最低类别覆盖：

- robust PEFT：8 篇；
- training dynamics / sample selection：10 篇；
- data valuation / influence：8 篇；
- compute-aware / curriculum：8 篇；
- mixed defects / OOD contamination：8 篇；
- real-noise audio：5 篇；
- cross-modal noisy supervision：3 篇。

同一工作可以覆盖多个类别，但总数仍需至少 50 篇。

## 5. Exclusion criteria

排除：

1. 无法追溯到原论文的二手总结；
2. 与训练数据质量无关的纯推理鲁棒性；
3. 与数据缺陷无关的硬件噪声；
4. 仅因标题出现 “noise” 而与本问题无机制关系的工作；
5. 只展示应用案例、无法形成公平基线比较的工作；
6. 同一工作的重复预印本或项目页记录；
7. 无法威胁任何候选主张、也不是必要强基线的外围论文。

## 6. Evidence extraction schema

每篇论文记录：

- 标题、年份、发表状态、正式 venue、主来源与代码；
- 任务模态、骨干、PEFT 方法；
- 数据缺陷类型与是否假设噪声率已知；
- 是否需要干净参考集、额外模型、额外训练阶段；
- 核心机制与论文主张；
- 机制证据与最强结果；
- 是否报告 wall-clock、GPU hours、VRAM 或仅报告可训练参数；
- 作者报告的局限和审计发现的局限；
- 与候选 A/B 的直接重叠；
- `high`、`medium`、`low` direct-prior risk。

`high` 表示问题与机制均实质相同；`medium` 表示问题或机制相同但非两者；`low` 表示强基线或邻近证据。未知值写为 `not_reported`，不能留空或自行推测。

## 7. Candidate gate

候选仅在以下条件全部成立时通过：

1. 没有 direct prior 实现同一机制；
2. 机制产生至少一个最近方法不产生的独特可证伪预测；
3. 强简单基线不能表达相同干预；
4. 关键性能比较能在单张 RTX 5080 上公平完成；
5. 完整研究有可信的 CCF-A/T1 证据路径；
6. 最强 kill argument 有具体、可验证的回答；
7. 预计收益不是额外训练、更多调参或更多辅助模型的直接结果；
8. 候选能够同时接受效果、效率、干净数据非劣性和真实低质量数据验证。

任何一项失败即为 `FAIL`。

## 8. Stop rules

立即停止候选方向：

- 最近方法可通过换数据集或骨干直接完成；
- 不可约差异只是已知信号的新加权；
- 只在人工对称错标上可能有效；
- 无法区分困难干净样本与错误样本；
- 需要多卡、多个完整模型或大规模外部 API 才成立；
- 最小机制探针无法在 12—24 次运行内区分候选机制和最近基线；
- 完整证据只能支撑低于 CCF-A/T1 的投稿目标。

若全部候选失败，输出 `NO_GO` 并继续寻找问题，不降低目标级别。

## 9. Audit trail

每次检索记录：

- 检索日期；
- 查询字符串；
- 搜索来源；
- 纳入论文；
- 排除的高相似论文及原因；
- 预印本/正式发表状态核验；
- 对候选方向造成的新增风险。

所有支持和反对证据进入同一矩阵。后续候选作者不得删除高风险 prior、静默改变 `high/medium/low` 判定或用性能结果倒推新颖性。

检索矩阵不是第二份综述。其唯一职责是保存可核验的逐篇证据；综合结论只写入 `idea-stage/IDEA_REPORT.md`。
