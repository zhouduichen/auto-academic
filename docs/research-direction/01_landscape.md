# 01 — 原始研究地图 (Research Landscape)

> 状态：已完成（A/B/C/D/E/F 六簇全覆盖）| 更新日期：2026-07-30
>
> ⚠️ 本文件保存原始宽口径文献地图，不代表当前选题或投稿目标。2026-07-31 起，当前研究合同以 [`04_gap_analysis.md`](04_gap_analysis.md)、[`05_candidate_topics.md`](05_candidate_topics.md)、[`06_venue_fit.md`](06_venue_fit.md) 和 [`07_recommendation.md`](07_recommendation.md) 为准。

## 总体概述

原始调研围绕**“面向小数据任务的预算感知自动微调 Agent”**展开，横跨多个相互关联的文献簇。该表述现仅用于说明当时的检索范围；预算感知 Agent 已不再是当前研究主线。以下是原始调研识别的关键关系：

```
                    ┌──────────────────────────────┐
                    │   C. AI 自动实验/科研 Agent    │
                    │   (LLM as experiment planner)  │
                    └──────────────┬───────────────┘
                                   │ 自动决策
                    ┌──────────────▼───────────────┐
                    │   B. 预算感知 AutoML/HPO       │
                    │   (搜索策略与资源分配)          │
                    └──────────────┬───────────────┘
                                   │ 优化目标
                    ┌──────────────▼───────────────┐
                    │   A. 小数据 PEFT               │
                    │   (可选的微调方法池)            │
                    └──────────────┬───────────────┘
                                   │ 验证领域
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
┌───────────▼──────────┐  ┌───────▼──────────┐  ┌───────▼──────────┐
│  D. 工程图/SysML 理解  │  │  自然图像分类     │  │  E. 评测基准     │
│  (副线验证场)          │  │  (标准验证场)     │  │  (方法论贡献)    │
└──────────────────────┘  └──────────────────┘  └──────────────────┘
```

**核心洞察**：CVPR 2025 的统一研究表明，不同 PEFT 方法在充分调参后性能趋同——这意味着**方法选择不如自动调参重要**，直接支持本项目的核心假设。

---

## A. 小数据视觉/多模态参数高效微调

### 发展时间线

| 年份 | 里程碑 | 关键工作 |
|---|---|---|
| 2021 | NLP 侧 LoRA 提出 | Hu et al., LoRA (NeurIPS) |
| 2022 | 视觉 PEFT 大爆发 | CoOp/CoCoOp (CVPR), VPT (ECCV), SSF (NeurIPS), AdaptFormer (NeurIPS), Tip-Adapter (ECCV) |
| 2023 | 多模态 prompt + 正则化 | MaPLe (CVPR), KgCoOp (CVPR), PromptSRC (ICCV), PLOT (ICLR), TaskRes (CVPR) |
| 2024 | LoRA 变体 + 视觉特化 | DoRA (ICML), Convpass (ECAI) |
| 2025 | 系统化研究 + 统一评测 | CVPR 2025 统一研究 (Mai et al.), MetaPEFT (CVPR), CLIP-AST (CVPR), CDRA-SPT (CVPR) |
| 2026 | 综述+最佳实践 | TACL 2026 综述 (Szep et al.) |

### 关键派别

1. **Prompt Tuning 派**（CoOp → CoCoOp → MaPLe → KgCoOp → PromptSRC）：在输入端添加可学习向量，代表方法 PromptSRC 仅需 46K 参数
2. **Adapter 派**（CLIP-Adapter → Tip-Adapter → AdaptFormer → Convpass）：插入轻量级瓶颈模块，Convpass 引入卷积归纳偏置
3. **LoRA 派**（LoRA → DoRA → LoRA+ → CDRA-SPT）：低秩分解权重更新，r=8-16 适合 16GB 显存
4. **特征变换派**（SSF → TaskRes）：仅学习缩放+偏移参数（SSF 仅 0.3M 参数），零推理开销
5. **免训练派**（Tip-Adapter → ADAPT）：无梯度更新，基于检索的适配
6. **测试时自适应派**（TPT → PromptAlign → SyTTA）：推理时动态调整

### 核心发现

- **CVPR 2025 统一研究**（Mai et al.）：充分超参调优后，所有 PEFT 方法性能趋同（VTAB-1K 上）——甚至简单 BitFit 也能匹配复杂方法
- **不同 PEFT 方法产生互补性错误**，可通过集成提升性能
- **SSF** 仅需 0.3M 参数即超越全量微调 11.48%（VTAB-1K），是极端预算场景首选
- **Convpass**（ECAI 2024）证明视觉特化 PEFT 方法优于 NLP 迁移方法
- **TACL 2026 综述**建议：QLoRA + 中等 rank（16-32）+ 2-4 层冻结 = 16GB GPU 最优策略

### 代表性的过饱和区域

- LoRA 变体已有 20+ 种（DoRA, AdaLoRA, DyLoRA, VeRA, PiSSA, LoKr, LoHa...），边际收益递减至 0.5-2%
- CoOp/CoCoOp 后续 prompt 方法高度同质化，新方法典型提升 1-3%
- 多篇 PEFT 综述在 2025-2026 年出版，纯综述论文空间有限

---

## B. 预算感知 AutoML 与自动微调

### 发展时间线

| 年份 | 里程碑 | 关键工作 |
|---|---|---|
| 2017-18 | 多保真度革命 | Hyperband (JMLR 2018), BOHB (ICML 2018) |
| 2019-20 | 框架成熟 | Optuna (KDD 2019), ASHA (MLSys 2020), HEBO (NeurIPS 2020 BBO Challenge) |
| 2021 | 进化+多保真度 | DEHB (IJCAI 2021), DHA (TMLR) — 联合优化 DA+HPO+NAS |
| 2022 | 系统化基准 | SMAC3 (JMLR), DyHPO (NeurIPS), HPOBench, YAHPO Gym |
| 2023 | Transformer 替代模型 | LC-PFN (NeurIPS) — 单次前向传播替代 MCMC，~10,000x 加速 |
| 2024 | 冻结-解冻 BO | FT-PFN/iFBO (ICML), FanG-HPO, FlexHB |
| 2025 | 元学习+AutoML | XAutoLM (EMNLP) — 支持 LoRA 的元学习自动微调，FedPop (AAAI) |

### 关键派别

1. **序列模型优化派**（SMAC3, HEBO）：GP/RF 替代模型引导试探，采样高效但 O(n³)
2. **多保真度 Bandit 派**（Hyperband, ASHA, BOHB, DEHB, FlexHB）：廉价部分评估 + 早停剪枝
3. **PFN/Transformer 替代派**（LC-PFN, FT-PFN/iFBO）：单次前向传播替代迭代替代模型拟合
4. **种群/进化派**（PBT, GPBT, FedPop）：训练中动态调整超参的种群
5. **元学习/热启动派**（XAutoLM）：利用历史实验初始化 HPO 先验——冷启动缓解关键
6. **Green AI/预算感知派**（FanG-HPO, Green AutoML）：将能量/CO2 约束纳入优化目标

### 核心发现

- **低预算（<10 次试探）下随机搜索最强**；~50 次试探是 BO 的交叉点
- **多保真度方法改变等式**：即使 10-50 次"完整"评估，Hyperband 在低保真度下评估数百配置
- **PEFT + 多保真度 HPO + 元学习热启动三者协同放大**——每个组件放大其他组件的收益
- LC-PFN 实现近乎零成本的早停决策（~10,000x 快于 MCMC）
- XAutoLM（EMNLP 2025）：4.5x 评估时间减少，7x 搜索误差降低，但仅限 NLP
- **若 PEFT 单次试验成本为全量微调的 1%，50 次 LoRA 试验 + BOHB 早停 ≈ 单次全量微调的 20% 成本**

### 代表性工作

| 论文 | 问题 | 方法 | 核心结果 |
|---|---|---|---|
| Hyperband (JMLR 2018) | 多臂 Bandit HPO | 逐次减半 bracket | >10x vs BO 加速 |
| BOHB (ICML 2018) | 结合 BO + Hyperband | TPE + 逐次减半 | 10-100x 更快达最优 |
| DEHB (IJCAI 2021) | 替代 BOHB 的 TPE 开销 | 差分进化 + Hyperband | 比 BOHB 快至 32x |
| HEBO (JAIR 2022) | 异方差+非平稳 HPO | 输入/输出变形 + Pareto 采集 | 108 任务 SOTA |
| LC-PFN (NeurIPS 2023) | 快速学习曲线外推 | Transformer 替代 MCMC | ~10,000x 加速 |
| FT-PFN/iFBO (ICML 2024) | 冻结-解冻 BO | PFN 替代 + MFPI-random | 成本敏感 HPO SOTA |
| XAutoLM (EMNLP 2025) | LLM 微调自动选择 | 元学习 + AutoML | 4.5x 时间减少；LoRA 支持 |
| FlexHB (2024) | 固定 bracket 低效 | 自适应 bracket + 全局排序 | 6.9x vs MFES-HB |

---

## C. AI 自动实验/科研 Agent

### 发展时间线

| 年份 | 里程碑 | 关键工作 |
|---|---|---|
| 2023 | 基础模块 | Coscientist (Nature), MLAgentBench (NeurIPS) |
| 2024 H1 | 首批自主系统 | ChemCrow (Nature MI), Data Interpreter (ACL), SciAgents |
| 2024 H2 | 端到端论文生成 | AI Scientist v1, PaperQA2, MLE-bench (OpenAI/ICLR 2025) |
| 2025 H1 | 基准爆发 | PaperBench (ICML), ScienceAgentBench (ICLR), CORE-Bench (TMLR), MLR-Bench (NeurIPS) |
| 2025 H2 | 可扩展性 | AI Scientist v2 (树搜索+ICLR workshop 录用), ML-Master (MLE-bench 榜首), PiML |

### 三级自主性分类（Zheng et al., EMNLP 2025）

- **Level 1**：LLM 作为工具——增强人类（AlphaFold, GNoME）
- **Level 2**：LLM 作为分析师——信息处理自主权
- **Level 3**：LLM 作为科学家——主要研究阶段的自主执行

### 关键派别

1. **端到端流水线派**（AI Scientist v1/v2）：想法生成→代码→实验→论文→自动审稿
2. **工具增强派**（ChemCrow, Coscientist）：LLM + 18 专业工具（SMILES, 反应预测, 安全检测）
3. **多 Agent 派**（SciAgents, Data Interpreter）：专业角色分工协作
4. **树搜索派**（AIDE, ML-Master）：ML 工程 = 代码空间搜索（draft/debug/improve）
5. **文献 Agent 派**（PaperQA2）：5 工具 RAG — 超人类文献综合

### 核心发现（⚠️ 含关键负面证据）

- **Agent 价值来源层次**：试验次数 >> 工具编排 >> LLM 推理质量
- **失败率触目惊心**：42% 实验失败（Beel et al.）；~80% 结果伪造（MLR-Bench）；57% 手稿幻觉
- **MLE-bench**：o1-preview+AIDE 金牌率 16.9%（pass@1），34.1%（8 次尝试）— 但 8 次尝试在单 GPU 上不现实
- **PaperBench**：最佳 Agent 21.0% vs 人类 PhD 41.4% — Agent 写得出代码但跑不通
- **关键区分**：优化 Agent（输出=可测量验证准确率）vs 科研 Agent（输出=论文主张）— 伪造问题对优化 Agent 影响较小
- **核心未探索问题**：**LLM Agent 在等 GPU 预算下能否优于 BOHB？** — 零篇论文做此比较

### 代表性工作

| 论文 | 问题 | 方法 | 核心结果 |
|---|---|---|---|
| AI Scientist v1 (2024) | 端到端科研全周期 | LLM 流水线+自动审稿 | $15/篇；42% 失败率 |
| AI Scientist v2 (2025) | 开放探索+无模板 | 树搜索+多 LLM | ICLR workshop 录用 |
| PaperQA2 (2024) | 文献综合 | 5 工具 RAG Agent | 超人类文献准确率 |
| ChemCrow (Nature MI 2024) | 化学实验 Agent | GPT-4+18 化学工具 | 自主合成分子 |
| AIDE (2025) | ML 工程优化 | 解空间树搜索 | MLE-bench 主导框架 |
| MLAgentBench (NeurIPS 2023) | ML Agent 基准 | 15 任务 ReAct | Claude Opus 37.5% 成功率 |
| MLE-bench (ICLR 2025) | ML 工程基准 | 75 Kaggle 竞赛 | o1+AIDE 16.9% 金牌率 |
| PaperBench (ICML 2025) | 论文复现基准 | 20 论文+8316 任务 | Agent 21.0% vs 人类 41.4% |
| MLR-Bench (NeurIPS 2025) | ML 研究基准 | 201 研究任务 | ~80% 结果伪造/无效 |
| Beel et al. (SIGIR 2025) | 独立评估 | AI Scientist 复现 | 100% 新颖性误分类 |

---

## D. 工程图、系统架构图与 SysML 理解

### 发展时间线

| 年份 | 里程碑 | 关键工作 |
|---|---|---|
| 2022-23 | CNN 时代图重建 | GraphDecoder (IEEE VIS/TVCG) |
| 2024 | VLM 时代开启 | FlowLearn (ECAI), DesignQA (ASME), Relationformer/PID2Graph, GD&T VLM 微调, FlowVQA (ACL) |
| 2025 | 结构化理解突破 | CircuitSense (NeurIPS), Draw with Thought/Plot2XML (ACM MM), TextFlow (NAACL), CAD2Program (AAAI), SysTemp, V2G-Audit |
| 2026 | 工程 VLM 基准 | Enginuity (arXiv 2026.06) |

### 关键派别

1. **端到端 VLM 派**：直接用 GPT-4V/Claude 理解工程图——高感知能力但推理失败
2. **解耦管线派**（TextFlow, V2G-Audit）：视觉编码器 + 结构化中间表示 + 推理器分离
3. **Transformer 图重建派**（Relationformer）：端到端节点+边联合检测
4. **神经符号派**（AGREE-Dog, GSP Verifier）：学习感知 + 符号验证混合
5. **Multi-Agent 生成派**（SysTemp, Cibrian et al.）：多 Agent 协作 + ANTLR 语法验证

### 核心发现

- **感知-推理鸿沟已被量化**：CircuitSense (NeurIPS 2025) 发现 MLLM 感知准确率 >85%，但符号推理 <19%
- **V2G-Audit** 证明：像素级 MLLM 理解在连接标注上仅 6% 准确率，加入向量图结构后提升至 67%（+61%）
- **SysML v2 生成**：Cibrian et al. (2025) 通过 RAG + ANTLR 达到 100% 语法正确率
- **语义正确性评估仍是未解决问题**——当前所有方法评估语法有效性，而非语义正确性
- **关键数据集缺口**：SysML v2 图示没有公开的结构化标注数据集（UML、P&ID、CAD 已有）

### 代表性工作

| 论文 | 问题 | 方法 | 核心结果 |
|---|---|---|---|
| Relationformer/PID2Graph (2024) | P&ID 图→图结构 | DETR 变体 | 边检测 +25% vs 模块化基线 |
| CircuitSense (NeurIPS 2025) | 电路理解基准 | 8006 题分层评测 | >85% 感知 vs <19% 推理 |
| Draw with Thought (ACM MM 2025) | 科学图→mxGraph XML | CoT+MLLM | CLIP-score 与人类评估 ρ=0.825 |
| V2G-Audit (2025) | 工程图审计 | 向量→属性图+GSP 验证 | 6%→67% 连接标注 |
| SysTemp (PAAMS 2025) | NL→SysML v2 | 4-Agent 协作+模板 | 80% 语法正确 (有模板) |
| Cibrian et al. (2025) | 工业 SysML v2 生成 | RAG+ANTLR+自纠正 | 100% 语法正确 |
| DesignQA (ASME 2025) | 工程文档 VQA | 5 MLLM 评估基准 | MLLM 跨文档推理薄弱 |
| AGREE-Dog (2025) | MBSE 形式化验证 | LLM+AGREE 模型检查器 | 84.6% 一轮修复 |

---

## E. 评测基准、数据集和诊断性研究

### 发展时间线

| 年份 | 里程碑 | 关键工作 |
|---|---|---|
| 2021 | 基准设计科学 | "Are We Learning Yet?" (NeurIPS) |
| 2024 | FSL 评估可靠性 | Shimabucoro et al., Evaluating the Evaluators (TMLR) |
| 2024 | AutoML 公平比较 | AMLB 框架 (JMLR), 短预算基准 (Jurado et al.) |
| 2025 | 构造效度系统评估 | Bean et al., Measuring What Matters (NeurIPS) |
| 2025 | PEFT 稳定性综述 | Pecher et al. (ACM Computing Surveys) |
| 2025 | 负面结果方法学 | Min-P 重分析, Jerge & Evans (TMLR) |
| 2025-26 | PEFT 基准涌现 | PEFT-Bench (EACL 2026), ArchCAD-400K (NeurIPS), DrafterBench, Enginuity |

### 关键发现

- **评估标准薄弱是系统性机会**：
  - 仅 16% 的 LLM 基准使用统计检验（Bean et al., NeurIPS 2025）
  - 仅 53.4% 的基准提供构造效度证据
  - FSL 验证准确率有 >50% 概率偏离真实值超过 10 个绝对百分点（Shimabucoro et al., TMLR 2024）

- **负面结果论文有发表路径**：ICBINB@ICLR 接收率 ~70%，TMLR 接受诊断性研究，Min-P 重分析成功发表

- **工程图公开数据集现状**：UML（Torcal 2,626 张）、P&ID（PID2Graph）、CAD（ArchCAD-400K 413K 块）、电路（CircuitSense 8,006 题）、科学图（Plot2XML 247 张）——但 SysML v2 为零

### 代表性工作

| 论文 | 类型 | 核心发现 |
|---|---|---|
| Pecher et al. (ACM CS 2025) | PEFT 稳定性综述 | 种子变化可完全逆转模型排名；415 篇论文分析 |
| PEFT-Bench (EACL 2026) | PEFT 统一基准 | 27 NLP 数据集+7 方法，但无视觉模型 |
| Mai et al. (CVPR 2025) | PEFT 诊断研究 | 充分调参后方法趋同；互补性错误可集成 |
| Shimabucoro et al. (TMLR 2024) | FSL 评估诊断 | FSL 验证不可靠；最佳估计器 MAE 9.5-12.7% |
| AMLB (JMLR 2024) | AutoML 基准 | AutoGluon 5 分钟 > 其他 1 小时 |
| Bean et al. (NeurIPS 2025) | 基准设计科学 | 445 基准系统评估；8 条改进建议 |
| Jerge & Evans (TMLR 2025) | 负面结果 | 每种推理方法至少对 1 个基准有负面影响 |
| Min-P 重分析 (ICLR 2025) | 负面结果/重分析 | 数据遗漏+多重比较未校正后，min-p 无优势 |

---

## F. 脏数据/低质量数据/噪声标签训练

> 状态：已完成 | 更新日期：2026-07-30

### 发展时间线

| 年份 | 里程碑 | 关键工作 |
|---|---|---|
| 2015-18 | 早期噪声鲁棒损失 | GCE (2018), SCE (2019), Co-teaching (2018) |
| 2019-20 | 半监督+样本选择 | DivideMix (ICLR 2020), JoCoR (2020), ELR (NeurIPS 2020) |
| 2021 | Confident Learning | Pervasive Label Errors (NeurIPS), CleanLab (JAIR Best Paper 2024) |
| 2022-23 | 真实噪声基准 | CIFAR-10N/100N, DataPerf (NeurIPS 2023) |
| 2024 | 实例依赖噪声+课程学习 | Time-Consistency Curriculum (TPAMI), Manifold DivideMix |
| 2025 | **PEFT×噪声鲁棒爆发** | Delora (ACL), RFedLR (NeurIPS), LoPE (NeurIPS), FHLR, BatMan-CLR |
| 2026 | LoRA 噪声鲁棒理论 | Why LoRA Resists Label Noise (arXiv:2602) |

### 关键派别

1. **损失函数工程派**：设计天然抗噪的损失函数（MAE/GCE/SCE/Bi-Tempered）——复合损失在高噪声率下一致优于简单替代
2. **样本选择派**（Co-teaching 家族）：双网络小损失筛选——2024-25 趋势是打破确认偏差循环
3. **半监督重构派**（DivideMix 家族）：将噪声样本视为无标签→半监督学习——2024-25 整合对比学习
4. **噪声转移矩阵派**：显式建模 P(noisy|true)，通过前向/反向修正损失——实例依赖噪声下估计困难
5. **Confident Learning / 数据为中心派**：模型无关的标签错误检测（CleanLab）——2024 IJCAI-JAIR Best Paper
6. **PEFT 特化噪声鲁棒派（2025-26）**：LoRA 低秩约束天然抗噪；双适配器（Delora/LoPE/RACT）解耦噪声处理

### 核心发现

- **标签错误普遍存在**：ImageNet 测试集约 6% 错误标签；3.3% 跨 10 基准平均——CleanLab 应作为微调前标准预处理步骤
- **小数据 × 噪声是乘性交互**（非加性）：噪声对小数据集的破坏性远超大数据集；BatMan-CLR 展示 60% 噪声下标准元学习器精度下降 34%
- **FHLR（2025）**：仅需 100 个专家标签即可在 60% 噪声下获得 43 个百分点的提升——高预算感知
- **LoRA 天然抗噪**：Steele (2026) 理论证明——秩-r LoRA 的 memorization 容量有限，低秩天然抑制噪声记忆；最优秩随噪声率递减
- **双适配器架构一致优于单适配器**：Delora (ACL 2025): +3-10% 噪声下精度，仅 +13.6MB；LoPE (NeurIPS 2025): 无需数据清洗

### 代表性工作

| 论文 | 问题 | 方法 | 核心结果 |
|---|---|---|---|
| Pervasive Label Errors (NeurIPS 2021) | 基准标签错误 | Confident Learning | ImageNet ~6% 错误；ResNet-18 > ResNet-50 修正后 |
| Delora (ACL 2025) | PEFT 噪声标签 | 双 LoRA（干净/噪声）+ GPT-4o 重标注 | +3-10% 噪声下；仅 +13.6MB；84.53% @ 60% 噪声 |
| Why LoRA Resists Label Noise (arXiv 2026) | LoRA 噪声鲁棒理论 | 3 定理+RACT 噪声检测 | 91.1% F1 噪声检测；最优秩与噪声率负相关 |
| FHLR (Scientific Reports 2025) | 小数据+高噪声 | 三阶段+100 专家标签+参数平均 | 74.1% vs 30.2% @ 60% 噪声 |
| BatMan-CLR (ECML-PKDD 2025) | 元学习噪声 | 流形采样+对比损失 | 60% 噪声下退化仅 1.1-9.4% vs 34% |
| Data-Centric AI Survey (ACM CS 2025) | 数据质量优先 | 全生命周期分类 | 数据质量 = 一等关注 |
| RFedLR (NeurIPS 2025) | 联邦 LoRA 噪声 | 敏感性感知参数更新 | 83.12% vs 75.28% @ 20% PairFlip |
| LoPE (NeurIPS 2025) | 无数据清洗噪声鲁棒 | 非对称 LoRA+投毒专家 MoE | 架构级噪声鲁棒 |

---

## 跨簇交叉领域

从已有调研中识别出以下跨簇交叉机会：

### 1. 预算感知 PEFT 方法选择（A × B）
- **机会**：CVPR 2025 发现所有 PEFT 方法充分调参后趋同，但调参本身消耗预算
- **问题**：如何在固定 GPU 预算内自动选择最优（方法 + 超参 + 数据子集）组合？
- **已有线索**：SSF（0.3M 参数，零推理开销）是极端预算最优；LoRA r=8-16 适合中等预算；QLoRA 是 16GB GPU 最优

### 2. PEFT 稳定性诊断基准（A × E）
- **机会**：Pecher et al. 综述揭示稳定性危机，但 PEFT-Bench 未做方差分析
- **问题**：对主流 PEFT 方法做 5-10 种子的系统方差分析，报告排名稳定性和置信区间
- **历史 CCF 适配判断**：诊断性基准可能适配 CCF-B/C，但低于当前 CCF-A/T1 硬门槛，不能作为现行投稿路线

### 3. 工程图的自动微调验证（A × D）
- **机会**：工程图需领域特化微调，且具有客观正确性标准（约束满足）
- **问题**：自动微调 Agent 在自然图像上训练，泛化至工程图领域的效果如何？
- **价值**：将 SysML/工程图定位为自动微调方法的泛化验证场

### 4. Agent 决策与 AutoML 搜索的公平比较（B × C）
- **机会**：LLM Agent 声称能做实验规划，但缺乏与非 LLM 方法（BO, Hyperband）的公平比较
- **问题**：在相同计算预算下，LLM Agent 的搜索效率是否优于随机搜索 + Hyperband？
- **发表潜力**：公平比较研究本身就是有价值的方法论贡献

### 5. SysML v2 数据集缺口（D × E）
- **机会**：SysML v2 是 2023 年发布的新标准（OMG），完全没有公开的标注数据集
- **问题**：创建一个 1,000-2,000 张 SysML v2 图示（8 类型）的结构化标注数据集
- **历史 CCF 适配判断**：数据集工作需要达到 CCF-A/T1 证据标准；CCF-B 期刊不再是现行目标

### 6. 数据质量感知的 PEFT 方法选择（A × F）🆕
- **机会**：Delora/RACT 证明噪声水平影响最优 PEFT 配置；Steele 证明最优 LoRA 秩随噪声递减
- **问题**：能否先评估数据质量，再根据估计的噪声水平自动选择（PEFT 方法 + 秩 + 训练策略）？
- **关键线索**：CleanLab 噪声检测 + Delora 高噪声 + 标准 LoRA 低噪声 = 完整数据质量感知管线

### 7. 预算感知的数据清洗 × 模型训练联合优化（B × F）🆕
- **机会**：清洗数据的预算 vs 训练模型的预算是可互换的——应联合优化
- **问题**：在固定总成本下，多少预算应分配给数据清洗，多少给模型训练？
- **文献空白**：零篇论文显式优化此 trade-off——潜在的核心贡献点

### 8. 噪声鲁棒 PEFT 的系统基准（E × F）🆕
- **机会**：已有多种噪声鲁棒 PEFT 方法（Delora/LoPE/RACT/RFedLR），但无系统比较
- **问题**：在统一噪声水平（10%/30%/50%）和噪声类型（对称/非对称/实例依赖）下，各方法的 Pareto 前沿是什么？
- **CCF 适配**：诊断性基准论文——可发表
