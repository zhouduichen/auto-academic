# 04 — 缺口分析 (Gap Analysis)

> 状态：已修订（v2）| 更新日期：2026-07-30
> 基于 98 篇论文 + 对 AutoPEFT/BIPEFT/MetaPEFT/DataMaster 的重新评估

---

## 0. 关键修订：为什么原"预算感知 PEFT Agent"不够新

进一步审查发现五篇密集相邻工作，使原方案的核心主张（"自动选择 PEFT 方法 + 超参数 + 数据清洗"）变成已有模块的拼装：

| 工作 | 出版 | 核心能力 | 与我们的冲突 |
|---|---|---|---|
| AutoPEFT | TACL | 贝叶斯优化自动选 PEFT 配置 | 自动 PEFT 选择 |
| BIPEFT | EMNLP 2024 Findings | 预算引导的迭代 PEFT 搜索 | 预算感知 PEFT 搜索 |
| MetaPEFT | CVPR 2025 | 元学习视 PEFT 超参数 | 学习如何选择 HP |
| Data Cleaning + AutoML | 2022 | 数据清洗作为 AutoML 变量 | 数据质量×AutoML |
| DataMaster | arXiv 2026 | Agent 数据发现→清洗→转换 | Agent 管数据 |

**但所有这些系统有一个共同盲区**：它们没有问"选出来的配置到底可不可靠"。它们全部在最大化验证准确率——即使验证集只有 100 个样本且含有标签错误。

---

## 1. 已被充分解决/过饱和的问题

### A. PEFT 簇

| 问题 | 过饱和证据 | 继续投入风险 |
|---|---|---|
| LoRA 新变体 | 20+ 变体（DoRA/AdaLoRA/DyLoRA/VeRA/PiSSA/LoKr/LoHa…），边际增益 0.5-2% | **高** — 新变体极难在 CCF-B/C 发表 |
| CoOp/CoCoOp 后续 Prompt Tuning | PromptSRC 已达到 ~80% Harmonic Mean 平台期；新方法典型增益 1-3% | **高** — 饱和 |
| PEFT 纯综述 | 2025-2026 年已有多篇（TechRxiv, Information Fusion, Neurocomputing, TACL） | **高** — 无新增实证的综述无发表空间 |
| 单数据集/单种子 PEFT 比较 | CVPR 2025 证明超参调优改变方法排名 | **极高** — 已属方法论缺陷 |

### B. AutoML 簇

| 问题 | 过饱和证据 | 继续投入风险 |
|---|---|---|
| 标准 HPO（<20 维，充足预算） | BOHB/DEHB/SMAC3 高度成熟，Optuna 已成为工程标准 | **高** |
| 无约束 NAS | NAS 基准（NAS-Bench-101/201/301）已成标准；DARTS 类方法不再新颖 | **高** |
| 单一保真度 BO | 多保真度方法（BOHB/DEHB/ASHA）全面超越，单保真度 BO 无竞争力 | **中** |

### C. AI Agent 簇

| 问题 | 过饱和证据 | 继续投入风险 |
|---|---|---|
| "又一个 LLM Agent 做 AutoML" | MLR-Bench 证明 ~80% 结果伪造；不加验证的 Agent 方案无发表价值 | **极高** |
| LLM 文献综述 Agent | PaperQA2 已超人类；单纯文献 Agent 不再新颖 | **高** |
| MLE-bench 刷榜 | 多家已逼近天花板（ML-Master 56.44%）；边际提升论文空间有限 | **中** |

### D. 工程图簇

| 问题 | 过饱和证据 | 继续投入风险 |
|---|---|---|
| P&ID/CAD 符号检测 | YOLO/Faster R-CNN 已成熟；ICDAR 2024 竞赛已标准化 | **高** |
| 流程图分类 | ViT 在 UML 上已达 92.7% 准确率（Torcal et al.） | **中** |

---

## 2. 仍然可信的研究空白

### 空白 1：PEFT 自动选择的可靠性——选择偏差与排名稳定性 🥇 [已修订]

**描述**：现有 AutoPEFT/BIPEFT/MetaPEFT 都能自动搜索 PEFT 配置，但**没有一个问"选出来的配置是否可靠"**。具体缺失：
- 搜索到的"最优"配置在不同种子/划分下是否稳定？
- 搜索配置越多，验证集过拟合是否使选择更差？
- 小验证集 + 标签噪声下，"选验证准确率最高"的策略是否系统性地差？

**文献证据**：
- AutoPEFT（TACL）、BIPEFT（EMNLP 2024）、MetaPEFT（CVPR 2025）：全部最大化验证准确率——没有风险规避目标或可靠性评估
- Pecher et al.（ACM CS 2025）：种子变化可逆转模型排名（415 篇论文）——但视觉 PEFT 交叉场景未被覆盖
- NeurIPS 2021 HPO 选择偏差论文：一般 HPO 中的选择后悔值问题——但无人落到 PEFT + 小数据场景
- CVPR 2025 统一研究（Mai et al.）：方法趋同但未问"排名是否稳定"

**为什么不是被忽视的盲区**：自动搜索和可靠性评估是正交维度。所有已有系统在"能不能找到"上竞争，没有人在问"找到的是不是真的"。

**CCF 适配**：诊断性基准 → TMLR / NeurIPS D&B / CCF-B 会议 / CCF-C 起步。

---

### 空白 2：视觉 PEFT 稳定性与方差系统基准 🥈

**描述**：对主流视觉 PEFT 方法（LoRA/SSF/VPT/Adapter/Convpass）进行 5-10 种子的系统方差分析，报告排名稳定性和置信区间。

**文献证据**：
- Pecher et al.（ACM CS 2025）：种子变化可逆转模型排名，但 415 篇论文分析中仅 161 篇超越简单识别
- PEFT-Bench（EACL 2026）：首个统一 PEFT 基准，但（1）仅 NLP/LLM（2）未做方差分析
- Bean et al.（NeurIPS 2025）：仅 16% 基准使用统计检验
- CVPR 2025 统一研究已证明方法论可行但未扩展到多种子

**为什么不是被忽视的盲区**：稳定性危机已被识别（Pecher et al.），但具体到视觉 PEFT 的系统方差分析尚未被任何人执行。这是一个"已知急需但无人动手"的空白。

**CCF 适配**：诊断性基准论文 → TMLR / NeurIPS D&B / CCF-B 期刊。

---

### 空白 3：LLM Agent vs 传统 AutoML 在等计算预算下的公平比较 🥉

**描述**：在相同的 GPU 时间预算下（无 API 依赖），LLM 引导的 HPO 是否优于 BOHB/Hyperband？

**文献证据**：
- 零篇被调研论文按 GPU 时间归一化比较 LLM Agent 与传统 AutoML
- MLE-bench 的 8 次尝试 vs 100 次 BO 试验并非计算归一化
- Raponi et al.（IEEE TEVC 2025）：低预算下 BO 确实优于随机搜索，但未涉及 LLM Agent
- MLR-Bench 关键发现：~80% Agent 结果伪造 → 输出可测量指标（验证准确率）的优化 Agent 情况不同

**为什么不是被忽视的盲区**：Agent 社区和 AutoML 社区之间几乎没有对话。Agent 论文对比未调优基线，AutoML 论文不对比 LLM Agent。交叉空白。

**CCF 适配**：实证比较论文 → TMLR / CCF-B 会议；可能获得高引用。

---

### 空白 4：SysML v2 公开数据集 🏅

**描述**：创建首个公开的 SysML v2 图示数据集（8 类型，1000-2000 张，结构化标注）。

**文献证据**：
- SysML v2 是 OMG 2023 标准，8 种图示类型（BDD/IBD/PAR/REQ/ACT/SD/STMC/UC）
- **零个公开的 SysML v2 标注数据集**
- UML：Torcal 2,626 张（2024）；P&ID：PID2Graph（2024）；CAD：ArchCAD-400K 413K 块（2025）
- CircuitSense 证明感知-推理鸿沟存在于所有工程图类型
- SysML v2 Pilot Implementation（Eclipse, Apache 2.0 开源）

**为什么不是被忽视的盲区**：SysML v2 太新（2023），学术界尚未赶上来。创建时间窗口有限（先发优势将在 2-3 年内消失）。

**CCF 适配**：数据集论文 → NeurIPS Datasets & Benchmarks / LREC / CCF-B 期刊。

---

### 空白 5：极端少样本（1-2 shot）下的联合数据选择 + PEFT 方法选择

**描述**：当你只能标注 5-20 个样本时，联合决定标注哪些样本 + 使用哪种 PEFT 方法。

**文献证据**：
- 主动学习 + PEFT 交叉领域几乎空白
- 大多数 PEFT 基准使用 16-shot 设定；1-2 shot 性能缺乏特征化
- CoCoOp 在 1-shot 下因文本-图像不对齐而急剧退化
- DHA（TMLR 2022）：联合优化 DA+HPO+NAS，但不涉及 PEFT 方法选择或主动学习
- Convpass 证明视觉特化 PEFT 在极少样本下优势更明显

**CCF 适配**：方法论文 → ECCV/ICCV workshop / CCF-C 会议。

---

### 空白 6 🆕：数据质量感知的 PEFT 方法自动选择

**描述**：先自动评估数据集质量（噪声水平、标签错误率），再根据估计的质量水平自动选择（PEFT 方法 + LoRA 秩 + 训练策略）。

**文献证据**：
- Steele (2026)：理论证明最优 LoRA 秩随噪声率递减——r* ∝ (1+η)^(-1/(2α+1))
- Delora (ACL 2025)：双 LoRA 在 20-40% 噪声下 +3-10%，但需人工决定是否启用双适配器模式
- CleanLab (Northcutt et al., 2021)：已有成熟噪声检测工具，但未与 PEFT 方法选择联动
- **目前无工作将"数据质量评估"作为 PEFT 方法选择的输入特征**

**CCF 适配**：方法论文 → ECAI / ACL / CCF-B 会议。

---

### 空白 7 🆕：预算感知的数据清洗 × 模型训练联合优化

**描述**：给定固定总预算（如 4 GPU 小时 + $10 API 额度），如何最优分配资源给数据清洗（CleanLab 检测、重标注、丢弃）vs 模型训练（方法选择、HPO）？

**文献证据**：
- FHLR (2025)：仅 100 个专家标签即可修复严重噪声——证明清洗预算是极其高效的
- Delora (ACL 2025)：GPT-4o 重标注需要 API 费用——但未讨论总成本是否最优
- **零篇论文显式优化"清洗预算 vs 训练预算"的 trade-off**

**CCF 适配**：系统/优化论文 → AutoML Conference / CCF-B 会议。

---

### 空白 8 🆕：噪声鲁棒 PEFT 方法的系统基准

**描述**：在统一的噪声水平（10%/30%/50%）和噪声类型（对称/非对称/实例依赖）下，系统比较 Delora/LoPE/RACT/RFedLR/标准 LoRA 的 Pareto 前沿。

**文献证据**：
- 已有 ≥5 种噪声鲁棒 PEFT 方法，但每个方法在不同条件下报告结果
- PEFT-Bench（EACL 2026）对 NLP PEFT 统一基准，但未涉及噪声鲁棒性维度

**CCF 适配**：诊断性基准 → TMLR / NeurIPS D&B。

---

## 3. 可行性评估表

| 空白 | 16GB 可行？ | 数据公开？ | CCF-B/C 可行？ | 1-2 天否证实验？ | 竞赛转化？ | 总体可行性 |
|---|---|---|---|---|---|---|
| 1. 预算感知 PEFT 选择 | ✅ LoRA/SSF/Adapter 全适配 | ✅ VTAB-1K, FGVC | ✅ ECAI, ICME | ✅ 3方法×3预算 | ✅ 可演示系统 | 🟢 高 |
| 2. PEFT 稳定性基准 | ✅ 仅需运行已有方法 | ✅ 同上 | ✅ 诊断论文 | ✅ 3方法×3数据集×5种子 | ⚠️ 基准≠产品 | 🟢 高 |
| 3. Agent vs BO 比较 | ✅ 本地 BO 可行 | ✅ 同上 | ✅ 实证对比 | ⚠️ 需多配置运行 | ⚠️ 比较结果可能负面 | 🟡 中 |
| 4. SysML v2 数据集 | N/A（数据创建） | ✅ 创建后公开 | ✅ NeurIPS D&B | ❌ 数据收集需时间 | ✅ IEEE 竞赛潜力 | 🟡 中 |
| 5. 极端少样本选择 | ✅ 样本少训练快 | ✅ CUB-200, miniImageNet | ✅ CCF-C | ✅ 2策略×3方法 | ⚠️ 领域太窄 | 🟢 高 |
| 6 🆕 数据质量感知 PEFT | ✅ Delora实测可行 | ✅ CIFAR-10N/CIFAR-FS | ✅ ECAI, ACL | ✅ CleanLab+2 LoRA秩 | ✅ 质量报告 | 🟢 高 |
| 7 🆕 清洗×训练联合优化 | ✅ 小数据清洗快 | ✅ CleanLab免费 | ✅ AutoML Conf | ✅ 3预算×3策略 | ⚠️ 系统级 | 🟢 高 |
| 8 🆕 噪声鲁棒 PEFT 基准 | ✅ 运行已有方法 | ✅ 同上 | ✅ TMLR | ✅ 3方法×3噪声率 | ⚠️ 基准 | 🟢 高 |

---

## 4. 最强反对证据

在确定选题前，必须诚实地面对以下反对证据：

1. **CVPR 2025 研究的反向解读**：如果所有 PEFT 方法调参后趋同，那"自动方法选择"的增量可能极小——选什么方法都差不多，Agent 的贡献在哪？
2. **BOHB 可能已经足够**：BOHB + LoRA + 合理默认值可能已经是最优解，Agent 的额外推理开销可能反向降低效率。
3. **XAutoLM 的部分覆盖**：EMNLP 2025 的 XAutoLM 已将元学习应用于 LM 微调自动选择，扩展到视觉可能在方法层面缺乏足够的新颖性。
4. **Agent 伪造率问题**：MLR-Bench 证明 ~80% Agent 实验结果是伪造的——优化 Agent 的"伪造"形式可能是：选择了一个表面上好的配置，但实际由于过拟合小验证集导致。
5. 🆕 **干净数据假设**：大多数 PEFT 论文假设干净标签——如果真实世界数据集普遍有 3-10% 标签错误（Northcutt et al.），现有 PEFT 方法结论在噪声数据上可能被推翻。
6. 🆕 **CleanLab 预处理可能已足够**：如果在微调前运行 CleanLab 并丢弃/重标注错误标签，标准 LoRA 的性能可能已经接近噪声特化的 Delora/RACT——专用噪声方法的增量价值不大。

**这些反对证据必须在最小否证实验中直接面对，不得回避。**
