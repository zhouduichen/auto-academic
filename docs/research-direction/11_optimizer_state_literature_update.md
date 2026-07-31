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

注册时间为 2026-07-31T23:18:33+08:00。注册与执行分开记录：18 条原始记录保持 `search_registered`，首轮实际检索另以 `-W` 后缀追加，不覆盖注册条目。六个主题还分别执行了 ARIS arXiv 与 Semantic Scholar helper；OpenAlex helper 因本机缺少它声明的 `requests` 依赖而显式失败。Semantic Scholar 除一组结果外主要返回 HTTP 429。这些失败均保留在日志中，不能被计作零命中或饱和证据。

## 候选发现与去重

发现结果按 `DOI > arXiv ID > normalized title` 去重。正式发表版取代预印本的 venue 元数据，但保留 arXiv 链接。首轮共固化 20 个 work：12 个进入 `deep_read`，8 个作为机制或应用 `context`；完整字段见 `12_optimizer_state_claim_matrix.csv`。

去重中发现一个必须显式保留的版本陷阱：`arXiv:2502.17055` 的早期版本对应 Stable-SPAM（O06），当前 v3 已改为 GradientStabilizer（O07）。O06 因此锚定 ICLR SCOPE 的 OpenReview 正式页面，O07 锚定当前 arXiv/ICML 版本；二者不能合并成一篇同名同方法工作。

首轮候选形成四组：

1. 状态污染因果与测量：Adam 基础 recurrence（O01）、shuffle order 的 fixed-clock memory 分析（O09）。
2. 直接稳健化或重置：TAdam（O02）、AdaR（O04）、SPAM（O05）、Stable-SPAM（O06）、GradientStabilizer（O07）、Adam-Rel（O10）。
3. 状态通道路由或交叉耦合：DP-AdamBC（O03）、Adaptive-OGP（O08）、AdaMomentum（O20）。
4. 一致性和 noisy-label 应用：Cautious Optimizers（O11）、GAL（O12）、ELR（O13）以及 noisy-PEFT/近期强基线（O14–O17）、MGUP（O18）、adaptive t-momentum（O19）。

## 直接威胁深读

“直接威胁”指可能实现相同持久状态控制合同、控制同一状态载体，或在相同约束下形成强简单基线的工作。首轮最重要的结论不是“没有先例”，而是先例已经覆盖了相邻机制的大部分组成件：

- TAdam（O02）以 Student-t/Mahalanobis 权重稳健化一阶矩写入，而二阶矩沿用 Adam。这是最接近“重点保护一阶矩”的历史先例；但异常梯度对当前一步和未来一阶矩的作用被同一个权重同时削弱，且 raw square 仍进入 `v`。
- SPAM（O05）用 `g_i^2/v_i` 识别 spike，先裁剪当前梯度，再周期性清零一、二阶矩；Stable-SPAM（O06）进一步使用历史范数统计。两者均同时改变当前作用与未来状态，并不提供二者分离。
- GradientStabilizer（O07）保留瞬时梯度方向，但用历史梯度范数的 EMA mean/RMS 比替换幅值，然后把变换后的梯度送入底层优化器。因此它是强低成本替代，但 `m/v` 接收的也是被变换后的信号。
- AdaR（O04）维护两套一、二阶矩并周期性切换/重置；它控制“记忆长度”，没有质量检测器，也不只保护一阶矩。
- Adaptive-OGP（O08）已经明确建立了 moment-path routing：修改后梯度写入 `m`，原梯度平方写入 `v`。它使“把不同信号路由给不同 Adam 状态”不能再作为宽泛新颖性主张；但该方法需要任务边界、K=50 梯度缓存与周期 SVD，目标是持续学习投影，不是标签噪声。
- Cautious Optimizers（O11）使用当前梯度与 optimizer update 的逐坐标符号一致性，但在标准 `m/v` 已更新之后才遮罩参数步。因此“agreement signal”已有直接先例，而“只改变持久写入位置”仍是必须单独验证的差别。
- GAL（O12）直接把跨模型梯度一致性用于抑制 noisy-label memorization，但需要三模型流程；它排除了“梯度一致性首次用于标签噪声”的表述，不等价于单模型状态门控。
- DP-AdamBC（O03）证明可只校正二阶矩通道，但依赖已知 DP 高斯噪声方差。它不能外推为未知标签噪声下的校正器。

O09 提供了关键因果支持：即使局部样本多重集相同，fixed-clock momentum/AdamW memory 也可能令顺序差异产生一阶端点效应；但该结果是局部测量理论，不是 noisy-label 方法，也不能替代本项目自己的多 seed 验证。

## 主题综合

首轮证据把方法空间压缩为一个更严格的问题：是否能在未知标签噪声、单模型、单阶段、单次 backward 的条件下，利用已有状态或同一步轻量信号，减少可疑梯度写入某个持久状态通道，同时保留一个可解释且可消融的当前学习作用。这个表述是检索后形成的研究问题，不是新颖性结论。

已经被覆盖的宽泛主张包括：

- “异常梯度会长期污染 Adam moments”（O05、O07、O09）；
- “稳健化 Adam 的一阶矩”（O02、O19）；
- “周期性重置一、二阶矩”（O04–O06）；
- “不同信号进入不同 moment pathway”（O03、O08、O20）；
- “梯度—动量/梯度—梯度一致性用于门控”（O11、O12、O18）；
- “PEFT 容量或路由可以抵抗标签噪声”（O14–O16）。

在候选冻结前仍可保留、但不能宣称成立的差异是：`current update effect` 与 `persistent state effect` 的显式分离；单模型 noisy-label PEFT 中的 `m-only` 证据；以及相对 AdamW/clipping/TAdam/SPAM/Cautious 的真实 wall-clock–accuracy Pareto 改善。任何 M1 后方法都至少要对 O02、O05、O07、O08、O11 做逐式对照。

最强拒稿论点目前是：候选可能只是 TAdam 的鲁棒一阶矩、Cautious 的 agreement mask 与 SPAM 的污染叙事的重新组合；如果没有“保留当前作用、只阻断未来写入”的精确 recurrence、对应反事实消融和 noisy-PEFT 实证，就不足以形成独立方法贡献。

## 饱和检查

关闭 L0 需要两轮互不重复的反查查询连续不再增加新的 `critical` 或 `deep_read` 工作。首轮 18 条查询、三类 helper 和 20-work 去重矩阵已完成；当前尚未执行 S1/S2 反查，因此不宣称饱和。

## L1 待候选冻结的精确问题

L1 只在 M1 数据支持并冻结精确 recurrence 后启动，逐项回答：

1. 门控发生在 raw gradient、`m` 写入、`v` 写入还是最终 parameter step？
2. 可疑 batch 的当前方向/幅值是否仍影响参数；若保留，通过哪一项 recurrence 实现？
3. 与 TAdam（O02）的 Student-t first-moment weight 在何种条件下等价或不等价？
4. 与 SPAM/GradientStabilizer（O05/O07）的 upstream transform 相比，是否仅把同一缩放移了位置？
5. 与 Adaptive-OGP（O08）的 moment routing 相比，信号、状态方向和额外计算合同是否实质不同？
6. 与 Cautious/MGUP（O11/O18）的 alignment mask 相比，是否只剩介入位置差异，且该差异是否由实验因果支持？
7. 对 `m-only`、`v-only`、`m/v coupled`、current-only、state-only 的反事实消融能否唯一支持主张？

L0 不提前填写 candidate-overlap，也不把内部的 A/T1 证据门槛表述为会议录用保证。
