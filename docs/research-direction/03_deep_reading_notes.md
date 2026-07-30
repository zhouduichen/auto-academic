# 03 — 精读笔记 (Deep Reading Notes)

> 状态：已完成 | 更新日期：2026-07-29
> 总计：30 篇精读（⭐ 核心 15 篇）

---

## A 簇：小数据 PEFT（8 篇精读，4 核心）

### A1 ⭐ Lessons and Insights from a Unifying Study of PEFT in Visual Recognition
- **基本信息**：Mai et al., 2025, CVPR
- **链接**：[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Mai_Lessons_and_Insights_from_a_Unifying_Study_of_Parameter-Efficient_Fine-Tuning_CVPR_2025_paper.html)
- **研究问题**：不同 PEFT 方法（VPT, Adapter, LoRA, BitFit, SSF）在视觉识别中真正差异有多大？
- **贡献类型**：诊断性实证研究
- **核心方法**：对 VTAB-1K 上所有主流 PEFT 方法进行系统超参调优后公平比较
- **关键发现**：
  - 充分超参调优后，所有 PEFT 方法性能趋同——即便简单的 BitFit 也能匹配复杂方法
  - 不同 PEFT 方法产生互补性错误，可有效集成
  - PEFT 在多 shot 场景同样有效，且比全量微调更好保持分布偏移鲁棒性
- **复现性**：代码未公开（截至撰写时）
- **与本项目关系**：最主要动机来源——方法选择不如自动调参重要，直接支持本项目的 Agent 定位
- **精读日期**：2026-07-29

### A2 ⭐ SSF: Scaling & Shifting Your Features
- **基本信息**：Lian et al., 2022, NeurIPS
- **链接**：[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html) | [GitHub](https://github.com/dongzelian/SSF)
- **研究问题**：能否仅用极少量参数（缩放+偏移）实现超越全量微调的迁移性能？
- **贡献类型**：PEFT 方法
- **核心方法**：每个操作（MSA, MLP, LayerNorm）添加可学习 scale (γ) + shift (β) 参数，通过重参数化实现零推理开销
- **关键结果**：仅 0.3M 参数；VTAB-1K 超越全量微调 11.48%；FGVC 超越全量微调 2.46%
- **复现性**：代码开源
- **与本项目关系**：极端预算场景首选方法，0.3M 参数 + 零推理开销 = 16GB GPU 最优
- **精读日期**：2026-07-29

### A3 Convpass: Convolutional Bypasses Are Better Vision Transformer Adapters
- **基本信息**：Jie et al., 2024, ECAI
- **链接**：[arXiv:2207.07039](https://arxiv.org/abs/2207.07039) | [GitHub](https://github.com/JieShibo/PETL-ViT)
- **研究问题**：NLP 起源的 PEFT 方法缺乏视觉归纳偏置，视觉特化设计能否更好？
- **贡献类型**：PEFT 方法
- **核心方法**：并行卷积瓶颈块（恢复 2D 空间结构后卷积）替代 NLP 起源的 Adapter/LoRA
- **关键结果**：VTAB-1K 76.6%（vs LoRA 74.5%、全量微调 68.9%），仅 0.33M 参数
- **复现性**：代码开源
- **与本项目关系**：关键洞察——视觉特化 PEFT 优于 NLP 迁移 PEFT，本系统应优先搜索视觉特化方法
- **精读日期**：2026-07-29

### A4 ⭐ Fine-tuning Large Language Models with Limited Data: A Survey and Practical Guide
- **基本信息**：Szep et al., 2026, TACL
- **链接**：[arXiv:2411.09539](https://arxiv.org/abs/2411.09539) | [ACL Anthology](https://aclanthology.org/2026.tacl-1.17/)
- **研究问题**：在有限数据和 GPU 预算约束下，微调 LLM 的最佳实践是什么？
- **贡献类型**：综述/实践指南
- **核心方法**：系统回顾约束微调的实证证据，整理为可操作建议
- **关键发现**：
  - QLoRA + 中等 rank（16-32）+ 2-4 层冻结 = 16GB GPU 最优策略
  - 数据质量 > 数据数量；仅 100K token 继续预训练就有效
  - 随机搜索减少 HP 调优时间 30%
- **与本项目关系**：直接提供系统默认配置和预算分配策略
- **精读日期**：2026-07-29

### A5 LoRA: Low-Rank Adaptation of Large Language Models
- **基本信息**：Hu et al., 2021, NeurIPS (Microsoft)
- **链接**：[arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | [GitHub](https://github.com/microsoft/LoRA)
- **研究问题**：能否用极低秩分解矩阵替代全量权重更新？
- **贡献类型**：基础 PEFT 方法
- **核心方法**：W + BA，r << min(d,k)，推理时合并至原权重，零推理开销
- **关键结果**：10,000x 更少参数；3x 更少 GPU 内存；匹配或超越全量微调
- **与本项目关系**：基石方法，r=8-16 在 ViT-B/16 上轻松适配 16GB 显存
- **精读日期**：2026-07-29

### A6 ⭐ DoRA: Weight-Decomposed Low-Rank Adaptation
- **基本信息**：Liu et al., 2024, ICML (NVIDIA/HKUST)
- **链接**：[ICML 2024](https://icml.cc/virtual/2024/poster/35052) | [GitHub](https://github.com/NVlabs/DoRA)
- **研究问题**：将预训练权重分解为幅值和方向后分别微调，能否提升 LoRA 表达能力？
- **贡献类型**：PEFT 方法（LoRA 增强）
- **核心方法**：W = m * (V/||V||_c) + BA，幅值独立学习，方向低秩更新
- **关键结果**：视觉-语言任务一致优于 LoRA；无推理开销增加
- **与本项目关系**：LoRA 的有力升级选项，极低数据量下额外表达能力可能关键
- **精读日期**：2026-07-29

### A7 KgCoOp: Knowledge-Guided Context Optimization
- **基本信息**：Yao et al., 2023, CVPR (中科院自动化所)
- **链接**：[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yao_Visual-Language_Prompt_Tuning_With_Knowledge-Guided_Context_Optimization_CVPR_2023_paper.html) | [GitHub](https://github.com/htyao89/KgCoOp)
- **研究问题**：如何防止 prompt tuning 中灾难性遗忘 CLIP 预训练知识？
- **贡献类型**：PEFT 正则化方法
- **核心方法**：在交叉熵损失基础上添加可学习 prompt 与手工 prompt 嵌入间的欧氏距离正则化
- **关键结果**：仅增加一行代码的损失项（无推理开销），Harmonic mean 77.0（最佳之一）
- **与本项目关系**：设计哲学可泛化——最小化实现成本的简单正则化可显著提升性能
- **精读日期**：2026-07-29

### A8 CoOp / CoCoOp
- **基本信息**：Zhou et al., 2022 (IJCV/CVPR), NTU Singapore
- **链接**：[GitHub](https://github.com/KaiyangZhou/CoOp)
- **研究问题**：能否用可学习连续向量替代 CLIP 的手工 prompt？
- **贡献类型**：基础 Prompt Learning 方法
- **核心方法**：CoOp：静态可学习上下文向量；CoCoOp：图像条件元网络生成动态 prompt
- **关键结果**：16-shot CoOp 63.33%（vs CLIP 零样本 56.77%）；CoCoOp 解决基类过拟合但推理变慢
- **与本项目关系**：CLIP PEFT 的标准起点，CoOp 代码库是基准测试平台
- **精读日期**：2026-07-29

---

## B 簇：预算感知 AutoML（6 篇精读，3 核心）

### B1 ⭐ BOHB: Robust and Efficient Hyperparameter Optimization at Scale
- **基本信息**：Falkner et al., 2018, ICML
- **链接**：[ICML 2018](https://proceedings.mlr.press/v80/falkner18a.html) | [HpBandSter](https://automl.github.io/HpBandSter/)
- **研究问题**：Hyperband 随机采样配置 + BO 采样效率高但慢，能否结合两者优势？
- **贡献类型**：算法（混合型多保真度 BO）
- **核心方法**：Hyperband 的逐次减半 bracket 调度 + TPE 替代 GP（KDE 建模好坏配置密度比 l(x)/g(x)）
- **关键结果**：比纯 Hyperband 和纯 BO 快 10-100x 达到最低测试误差；ResNet-20 CIFAR-10: 2.78±0.09%
- **与本项目关系**：单 GPU 预算约束 HPO 的最强默认选择；TPE 原生支持混合/离散空间
- **精读日期**：2026-07-29

### B2 ⭐ LC-PFN: Efficient Bayesian Learning Curve Extrapolation
- **基本信息**：Adriaensen et al., 2023, NeurIPS
- **链接**：[NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html) | [GitHub](https://github.com/automl/LC-PFN)
- **研究问题**：能否用单次前向传播替代 MCMC 实现贝叶斯学习曲线外推？
- **贡献类型**：模型（Transformer 替代）
- **核心方法**：Transformer 在 1000 万合成右删失学习曲线上预训练，推理时单次前向传播产生后验预测分布
- **关键结果**：比 MCMC 快 ~10,000x；预测早停在 45/53 数据集上实现 2-6x 加速
- **与本项目关系**：近乎零成本的早停决策——预算感知 Agent 的核心组件
- **精读日期**：2026-07-29

### B3 ⭐ XAutoLM: Efficient Fine-Tuning of Language Models via Meta-Learning and AutoML
- **基本信息**：Estevanell-Valladares et al., 2025, EMNLP
- **链接**：[ACL Anthology](https://aclanthology.org/2025.emnlp-main.399/)
- **研究问题**：能否利用历史微调实验的元学习加速未来的 AutoML 搜索？
- **贡献类型**：系统/框架
- **核心方法**：从过去成败中提取任务级和系统级元特征，构建"经验感知先验"偏置采样；支持全量/部分/LoRA 微调
- **关键结果**：平均评估时间减少 4.5x；搜索误差率降低至 7x；5/6 任务超越零样本优化器
- **与本项目关系**：最直接相关的工作——展示元学习可以大幅降低 AutoML 搜索成本，但仅限 NLP
- **精读日期**：2026-07-29

### B4 FT-PFN / iFBO: In-Context Freeze-Thaw Bayesian Optimization
- **基本信息**：Rakotoarison et al., 2024, ICML
- **链接**：[ICML 2024](https://proceedings.mlr.press/v235/rakotoarison24a.html)
- **研究问题**：能否用 PFN 替代在线更新替代模型实现冻结-解冻 BO？
- **贡献类型**：算法 + 模型
- **核心方法**：FT-PFN 替代模型 + MFPI-random 多保真度采集函数；单次前向传播无需在线训练
- **关键结果**：比深度 GP/深度集成快 10-100x；三个基准套件上新 SOTA
- **与本项目关系**：成本敏感 HPO 的当前 SOTA——理论上最优的资源分配
- **精读日期**：2026-07-29

### B5 DEHB: Evolutionary Hyperband
- **基本信息**：Awad et al., 2021, IJCAI
- **链接**：[IJCAI 2021](https://doi.org/10.24963/ijcai.2021.296) | [GitHub](https://github.com/automl/DEHB)
- **研究问题**：BOHB 的 TPE 替代有 O(n³) 开销，进化算法能否更高效？
- **贡献类型**：算法
- **核心方法**：用差分进化（DE）替代 BOHB 的 TPE；常数运行时开销；连续表示离散超参
- **关键结果**：比 BOHB 快至 32x；高维离散问题特别强
- **与本项目关系**：NAS 和高维 HPO 场景的备选优化器
- **精读日期**：2026-07-29

### B6 HEBO: Pushing The Limits of Sample-Efficient HPO
- **基本信息**：Cowen-Rivers et al., 2022, JAIR (Huawei Noah's Ark)
- **链接**：[JAIR](https://doi.org/10.1613/jair.1.13643) | [GitHub](https://github.com/huawei-noah/HEBO)
- **研究问题**：标准 GP 替代假设同方差和平稳性——HPO 中是否普遍违反？
- **贡献类型**：算法 + 实证分析
- **核心方法**：非线性输入/输出变形 + 多目标采集函数集成（Pareto 前沿解决采集冲突）
- **关键结果**：108 任务上显著超越现有优化器；异方差和非平稳在 HPO 中普遍存在
- **与本项目关系**：解释了朴素 GP-BO 为何失败——微调任务中噪声随超参剧烈变化
- **精读日期**：2026-07-29

---

## C 簇：AI 科研 Agent（6 篇精读，3 核心）

### C1 ⭐ PaperBench: Evaluating AI's Ability to Replicate AI Research
- **基本信息**：Starace et al., 2025, ICML (OpenAI)
- **链接**：[ICML 2025](https://proceedings.mlr.press/v267/starace25a.html) | [GitHub](https://github.com/openai/preparedness)
- **研究问题**：AI Agent 能否从零复现 SOTA ML 论文？
- **贡献类型**：基准
- **核心方法**：20 篇 ICML 2024 Spotlight/Oral 论文 × 8,316 可评分任务 × LLM 评分器 (F1=0.83)
- **关键结果**：最佳 Agent 21.0% vs 人类 PhD 41.4%（3 篇子集）；o1 工作 1 小时后性能停滞
- **与本项目关系**：最严格测量——Agent 在集成/测试/运行代码上失败，而非代码生成本身
- **精读日期**：2026-07-29

### C2 ⭐ Independent Evaluation of the AI Scientist
- **基本信息**：Beel et al., 2025, ACM SIGIR Forum
- **链接**：[arXiv:2502.14297](https://arxiv.org/abs/2502.14297)
- **研究问题**：Sakana AI 宣称的"自主科学发现"是否经得起独立检验？
- **贡献类型**：独立评估/负面结果
- **核心方法**：在两台机器上安装运行 AI Scientist v1，使用 MovieLens-100k，系统分析输出
- **关键发现**：
  - 42% 实验失败；57% 手稿包含幻觉数字
  - 100% 新颖性误分类；自动审稿人拒绝了全部 7 篇 AI 论文和 9/10 篇人类论文
  - 代码修改仅 +8% 字符/迭代
- **与本项目关系**：最清醒的数据——自主研究 Agent 的"自主"严重高估；42% 失败率
- **精读日期**：2026-07-29

### C3 ⭐ MLR-Bench: Evaluating AI Agents on Open-Ended ML Research
- **基本信息**：Chen et al., 2025, NeurIPS 2025 (Datasets & Benchmarks)
- **链接**：[arXiv:2505.19955](https://arxiv.org/abs/2505.19955) | [GitHub](https://github.com/chchenhui/mlrbench)
- **研究问题**：端到端 ML 研究（想法→实验→论文）中 Agent 产生有效结果的比例？
- **贡献类型**：基准
- **核心方法**：201 个 workshop 研究任务；MLR-Judge（LLM 评审 + 专家验证评分标准）
- **关键发现**：**~80% Agent 产生的实验结果系伪造或无效**——LLM 写连贯论文但无法做有效实验
- **与本项目关系**：区分"自动调参"与"自动科研"的关键证据——我们的 Agent 输出可测量指标而非论文主张
- **精读日期**：2026-07-29

### C4 AIDE: AI-Driven Exploration in the Space of Code
- **基本信息**：Jiang et al., 2025, Weco AI
- **链接**：[arXiv:2502.13138](https://arxiv.org/abs/2502.13138) | [GitHub](https://github.com/WecoAI/aideml)
- **研究问题**：ML 工程能否被框架化为代码优化问题并通过树搜索解决？
- **贡献类型**：系统/框架
- **核心方法**：解空间树搜索——节点=Python 脚本，操作=draft/debug/improve，搜索策略选择节点
- **关键结果**：MLE-bench 金牌数 4x vs 最佳线性 Agent；有效提交率 92.4%（vs 63.6%）
- **与本项目关系**：MLE-bench 主导框架（OpenAI/METR/Sakana/Meta 均使用）；硬编码搜索策略>学习策略暗示当前价值来自编排而非学习
- **精读日期**：2026-07-29

### C5 PaperQA2: Language Agents Achieve Superhuman Synthesis
- **基本信息**：Skarlinski et al., 2024, FutureHouse
- **链接**：[arXiv:2409.13740](https://arxiv.org/abs/2409.13740) | [GitHub](https://github.com/Future-House/paper-qa)
- **研究问题**：LLM Agent 能否在文献搜索/总结/矛盾检测上超越人类专家？
- **贡献类型**：系统 + 基准
- **核心方法**：5 工具 RAG Agent（Paper Search, Gather Evidence, Generate Answer, Citation Traversal）
- **关键结果**：LitQA2 准确率 66.0%（超人类）；WikiCrow 文章比人类 Wikipedia 更准确
- **与本项目关系**：文献综述 Agent 的黄金标准——但仅限于文献任务，不涉及实验设计
- **精读日期**：2026-07-29

### C6 MLE-bench: Evaluating ML Agents on ML Engineering
- **基本信息**：Chan et al., 2024, ICLR 2025 (OpenAI)
- **链接**：[arXiv:2410.07095](https://arxiv.org/abs/2410.07095) | [GitHub](https://github.com/openai/mle-bench)
- **研究问题**：AI Agent 能否在真实世界 ML 竞赛中表现得像 ML 工程师？
- **贡献类型**：基准
- **核心方法**：75 个 Kaggle 竞赛 × 三级难度 × 反作弊措施（GPT-4o 日志审计 + 抄袭检测）
- **关键结果**：o1-preview + AIDE：16.9% 奖牌率（pass@1），34.1%（8 次尝试）
- **与本项目关系**：ML 工程 Agent 的事实标准基准——但预算非归一化，且 8 次尝试在单 GPU 上不现实
- **精读日期**：2026-07-29

---

## D 簇：工程图/SysML（5 篇精读，2 核心）

### D1 ⭐ CircuitSense: Bridging Visual Comprehension and Symbolic Reasoning
- **基本信息**：Akbari et al., 2025, NeurIPS 2025 (Datasets & Benchmarks)
- **链接**：[circuitsense-benchmark.github.io](https://circuitsense-benchmark.github.io) | [arXiv:2509.22339](https://arxiv.org/abs/2509.22339)
- **研究问题**：MLLM 在工程图理解中感知与推理的差距有多大？
- **贡献类型**：基准
- **核心方法**：8006+ 题分层基准（感知→分析→设计）；6 层级
- **关键结果**：闭源 MLLM 感知准确率 >85%，符号推导/分析推理 <19%
- **与本项目关系**：量化了工程图理解的感知-推理鸿沟——Cluster D 基线
- **精读日期**：2026-07-29

### D2 ⭐ Beyond Pixels: Vector-to-Graph Framework for Reliable Schematic Auditing
- **基本信息**：2025, arXiv:2602.11678
- **链接**：[arXiv:2602.11678](https://arxiv.org/abs/2602.11678) | [GitHub](https://github.com/gm-embodied/V2G-Audit)
- **研究问题**：像素驱动 MLLM 处理工程图为何不可靠？
- **贡献类型**：方法 + 实证
- **核心方法**：CAD 基元→属性图（节点=组件，边=连接）+ GSP 验证器（拉普拉斯特征值分析）
- **关键结果**：总体准确率 12%（仅 MLLM）→ 47%（MLLM+V2G）；连接标注 6% → 67%（+61%）
- **与本项目关系**：直接验证 Cluster D 核心假设——像素级 VLM 理解工程图需图结构增强
- **精读日期**：2026-07-29

### D3 Draw with Thought / Plot2XML
- **基本信息**：2025, ACM Multimedia 2025
- **链接**：[Hugging Face](https://huggingface.co/datasets/miracle10/DwT_drawio)
- **研究问题**：科学图→可编辑结构化表示（mxGraph XML）
- **贡献类型**：方法 + 数据集
- **核心方法**：免训练 CoT + Draw.io 验证器多轮精炼
- **关键结果**：89% XML 有效性；CLIP-score 与人类评估 ρ=0.825
- **与本项目关系**：图像-结构化图恢复最佳方法，mxGraph XML 格式 + CLIP-score 评估协议可直接借用
- **精读日期**：2026-07-29

### D4 Relationformer / PID2Graph
- **基本信息**：Stürmer et al., 2024, arXiv:2411.13929
- **链接**：[Zenodo 数据集](https://zenodo.org/records/14803338)
- **研究问题**：P&ID 图端到端图结构提取
- **贡献类型**：方法 + 数据集
- **核心方法**：DETR 变体同时检测对象和关系
- **关键结果**：边检测比模块化基线 +25%（首个公开 P&ID 图结构数据集）
- **与本项目关系**：P&ID 领域最强基线，Relationformer 架构和数据集可直接复用
- **精读日期**：2026-07-29

### D5 Agent-Based Valid SysML v2 Generation
- **基本信息**：Cibrian et al., 2025, Computers in Industry
- **链接**：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0166361525001150)
- **研究问题**：NL→SysML v2 模型如何保证语法正确？
- **贡献类型**：系统
- **核心方法**：LLM + RAG（精选 SysML v2 示例） + ANTLR 验证 + 诊断反馈自纠正
- **关键结果**：20 prompts 100% 语法正确；全面超越纯 LLM
- **与本项目关系**：RAG+ANTLR+自纠正模式是 SysML 验证组件的蓝图
- **精读日期**：2026-07-29

---

## E 簇：基准与诊断（3 篇精读，不额外标记核心）

### E1 A Survey on Stability of Learning with Limited Labelled Data
- **基本信息**：Pecher et al., 2025, ACM Computing Surveys
- **链接**：[ACM DL](https://dl.acm.org/doi/10.1145/3691339)
- **核心发现**：415 篇论文系统分析；种子变化可完全逆转模型排名；元学习中性能偏差高达 90%
- **与本项目关系**：PEFT 基准设计的基础文献——证明方差分析非可选而是必须
- **精读日期**：2026-07-29

### E2 Measuring What Matters: Construct Validity in LLM Benchmarks
- **基本信息**：Bean et al., 2025, NeurIPS 2025
- **链接**：[NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1967e0fc3aa6cbbace562f5cb8e3954e-Abstract-Datasets_and_Benchmarks_Track.html)
- **核心发现**：445 基准中仅 53.4% 提供构造效度证据；仅 16% 使用统计检验；8 条可操作建议
- **与本项目关系**：设计新基准的检查清单——8 条建议应作为必选项
- **精读日期**：2026-07-29

### E3 Evaluating the Evaluators: Are Validation Methods for FSL Fit for Purpose?
- **基本信息**：Shimabucoro et al., 2024, TMLR
- **链接**：[Edinburgh Research](https://www.research.ed.ac.uk/en/publications/evaluating-the-evaluators-are-validation-methods-for-few-shot-lea/)
- **核心发现**：FSL 验证准确率 >50% 概率错误超过 10 个绝对百分点；最佳估计器（5-fold CV）MAE 9.5-12.7%
- **与本项目关系**：FSL 评估不可靠的量化证据——任何 PEFT 基准必须解决此问题
- **精读日期**：2026-07-29

---

---

## F 簇：脏数据/低质量数据/噪声标签训练（5 篇精读，3 核心）

### F1 ⭐ Pervasive Label Errors in Test Sets Destabilize ML Benchmarks
- **基本信息**：Northcutt et al., 2021, NeurIPS (Datasets & Benchmarks) / JAIR (IJCAI-JAIR Best Paper Prize 2024)
- **链接**：[arXiv:2103.14749](https://arxiv.org/abs/2103.14749) | [CleanLab](https://github.com/cleanlab/label-errors)
- **研究问题**：基准测试集中的标签错误有多普遍？它们如何扭曲模型排名？
- **贡献类型**：实证分析 + 算法 + 修正数据集
- **核心方法**：Confident Learning — 使用模型预测概率估计噪声标签与真实标签的联合分布，基于置信计数剪枝可能的标签错误
- **关键结果**：10 个基准平均 3.3% 错误率；ImageNet ~6%；51% CL 标记被人工确认；修正后 ResNet-18 > ResNet-50
- **复现性**：CleanLab 开源库，labelerrors.com 提供修正数据集
- **与本项目关系**：**标准预处理步骤**——任何微调前应先运行 CleanLab
- **精读日期**：2026-07-30

### F2 ⭐ Why LoRA Resists Label Noise: A Theoretical Framework
- **基本信息**：Steele, 2026, arXiv:2602.00084（预印本）
- **链接**：[arXiv:2602.00084](https://arxiv.org/abs/2602.00084)
- **研究问题**：LoRA 的噪声鲁棒性有没有理论解释？能否根据噪声水平选择最优秩？
- **贡献类型**：理论 + 算法
- **核心方法**：三个定理——(1) 记忆容量界，(2) 最优秩与噪声率负相关 r*=O((n/(d(1+η)))^(1/(2α+1)))，(3) 时间分离——干净模式早期学习，噪声后期记忆。RACT 算法用秩差异做噪声检测
- **关键结果**：91.1% F1 噪声检测（AG News）；91.46% 准确率保持；推荐怀疑噪声时使用更低秩
- **复现性**：代码随论文发布
- **与本项目关系**：**关键理论支撑**——为预算感知 Agent 中"噪声水平 → LoRA 秩选择"提供理论依据
- **精读日期**：2026-07-30

### F3 ⭐ Delora: Dual Low-Rank Adaptation for Noisy Label Detection
- **基本信息**：Yuan et al. (浙江大学), 2025, ACL 2025 Findings
- **链接**：[ACL Anthology](https://aclanthology.org/2025.findings-acl.792/)
- **研究问题**：传统小损失样本选择造成确认偏差循环——初始错误选择→后续训练恶化
- **贡献类型**：方法（双 LoRA 架构）
- **核心方法**：Clean LoRA（记忆干净数据）+ Noisy LoRA（记忆错误标签）；Noisy LoRA 输出 = 可学习、样本依赖的阈值。第二阶段用 GPT-4o 重标注噪声样本
- **关键结果**：+3.26% 至 +10% 噪声下准确率增益；仅 +13.6MB 参数和 +3.2GB 显存开销；60% 对称噪声下 84.53% 准确率
- **复现性**：ACL Anthology 开源
- **与本项目关系**：**直接可实现**——双 LoRA 设计在 RTX 5080 16GB 上完全可行；解耦噪声检测和分类的架构模式可直接借用
- **精读日期**：2026-07-30

### F4 FHLR: Few-Shot Human-in-the-Loop Refinement for Label Noise
- **基本信息**：Saeed et al., 2025, Scientific Reports
- **链接**：[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11794423/)
- **研究问题**：小传感器数据集在标签噪声下严重退化；现有方法在高噪声率下失败
- **贡献类型**：方法（三阶段 + 人在回路）
- **核心方法**：种子训练（标签平滑+EMA）→ ~100 专家修正标签微调 → 种子模型与微调模型的加权参数平均
- **关键结果**：60% 对称噪声下 FHLR 74.1% vs 次优方法 30.2%（43 个百分点提升）
- **与本项目关系**：**预算感知范式**——仅需 100 个专家标签即可修复严重噪声，展示人在回路的成本效益
- **精读日期**：2026-07-30

### F5 BatMan-CLR: Making Few-Shot Meta-Learners Resilient Against Label Noise
- **基本信息**：Galjaard et al., 2025, ECML-PKDD
- **链接**：[ECML-PKDD 2025](https://dl.acm.org/doi/abs/10.1007/978-3-032-06106-5_15)
- **研究问题**：元学习器在元训练中的标签噪声下准确率灾难性下降（高达 34%）
- **核心方法**：流形/批次流形采样将噪声监督任务转化为半监督任务 + 对比损失
- **关键结果**：60% 错误标签下退化仅 1.1%（miniImageNet）和 9.4%（CIFAR-FS）vs 基线 34%
- **与本项目关系**：直接针对小样本+噪声双重挑战——智能采样而非架构修改
- **精读日期**：2026-07-30

---

## 核心论文统计

| 簇 | 精读 | 核心 |
|---|---|---|
| A - PEFT | 8 | 4 |
| B - AutoML | 6 | 3 |
| C - Agent | 6 | 3 |
| D - 工程图 | 5 | 2 |
| E - 基准 | 3 | 0 |
| F - 脏数据 | 5 | 3 |
| **合计** | **30** | **15** |
| D - 工程图 | 5 | 2 |
| E - 基准 | 3 | 0 |
| **合计** | **25** | **12** |
