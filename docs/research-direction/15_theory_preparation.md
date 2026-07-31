# AdamW 状态污染理论准备与候选接口

> 日期：2026-07-31
> 目的：在候选机制冻结前，只固化标准 AdamW 的可验证对象、假设层级、反例和代码映射。
> 禁止：本文件不声称候选已有收敛保证，不从单脉冲外推持续/系统性噪声，也不把局部状态差
> 直接解释为最终泛化差。

## 1. 记号与标准递推

对 optimizer step `t=1,2,...`，设观测 mini-batch 梯度为 `g_t`，一阶矩 `m_t`、二阶矩
`v_t`，学习率 `eta_t`，AdamW 参数为 `beta_1,beta_2,epsilon,lambda`。逐坐标平方和除法记为
`⊙` 与 `/`：

```text
m_t = beta_1 m_{t-1} + (1-beta_1) g_t
v_t = beta_2 v_{t-1} + (1-beta_2) (g_t ⊙ g_t)
mhat_t = m_t / (1-beta_1^t)
vhat_t = v_t / (1-beta_2^t)
u_t = mhat_t / (sqrt(vhat_t) + epsilon)
theta_t = (1-eta_t lambda) theta_{t-1} - eta_t u_t
```

若使用 AMSGrad，则 `vhat_t` 前还有逐坐标历史最大值，必须作为不同递推单独证明；不能沿用本文件
的标准 AdamW 精确式。实际代码若使用 per-parameter-group step、capturable tensor step、梯度累积、
loss scaling 或跳步，公式中的 `t` 必须替换为真实 optimizer state step，而不是 batch/epoch 编号。

## 2. 单次污染脉冲的精确状态传播

令 clean 对照在时刻 `tau` 的梯度为 `c_tau`，污染分支观测

```text
g_tau = c_tau + delta,
```

并在 `tau` 前状态完全相同。首先研究**共同外生未来梯度**反事实：两个分支在 `tau` 后被强制
输入相同的 `c_{tau+1},c_{tau+2},...`。这不是实际训练的全部动力学，而是隔离 optimizer memory
的精确冲激响应。在 `k>=0` 时：

```text
Delta m_{tau+k}
  = beta_1^k (1-beta_1) delta

Delta v_{tau+k}
  = beta_2^k (1-beta_2)
    [2 c_tau ⊙ delta + delta ⊙ delta]

Delta mhat_{tau+k}
  = Delta m_{tau+k} / (1-beta_1^(tau+k))

Delta vhat_{tau+k}
  = Delta v_{tau+k} / (1-beta_2^(tau+k)).
```

因此一阶与二阶状态的记忆时间尺度分别由 `beta_1`、`beta_2` 控制，但这不意味着参数影响按同样
速度或同一符号衰减：更新同时依赖分子 `mhat` 和分母 `sqrt(vhat)+epsilon`，且随后参数分离会
改变未来梯度。

M0.6 的 state-only 分支在脉冲点恢复参数、选择性置换状态，并回放同一 batch/augmentation；
它给出接近该反事实的实验实现，但 horizon > 0 后参数已经分离，未来梯度一般不再相同。因此
上式是 horizon-zero 与强制共同梯度实验的精确式，不是 M0.6 全轨迹的无条件闭式解。

## 3. 实际耦合动力学

令污染分支相对 clean 分支的未来梯度差为

```text
e_t = g_t^(p) - g_t^(c).
```

它同时包含后续新污染和由 `Delta theta` 引起的内生梯度差。对 `t>tau`：

```text
Delta m_t = beta_1 Delta m_{t-1} + (1-beta_1)e_t

Delta v_t = beta_2 Delta v_{t-1}
          + (1-beta_2)[2 g_t^(c) ⊙ e_t + e_t ⊙ e_t]

Delta theta_t = (1-eta_t lambda)Delta theta_{t-1}
              - eta_t [F(mhat_t^(p),vhat_t^(p))
                        - F(mhat_t^(c),vhat_t^(c))],

F(m,v) = m/(sqrt(v)+epsilon).
```

展开第一式可得历史误差卷积：

```text
Delta m_t = beta_1^(t-tau)(1-beta_1)delta
          + (1-beta_1) sum_{j=tau+1}^t beta_1^(t-j)e_j.
```

二阶矩也有对应卷积，但其输入含交叉项和平方项，不能把 `Delta v` 解释成简单“噪声能量”。
任何候选理论若忽略 `e_t`，必须明确标成 fixed-gradient/state-only 局部结论。

### 3.1 分母耦合的无光滑下界表达

逐坐标对任意 `a,b>=0`：

```text
|1/(sqrt(a)+epsilon) - 1/(sqrt(b)+epsilon)|
  <= |sqrt(a)-sqrt(b)| / epsilon^2
  <= sqrt(|a-b|) / epsilon^2.
```

由此可以得到不要求 `v` 远离零的 Hölder 型控制：

```text
|F(m_p,v_p)-F(m_c,v_c)|
  <= |m_p-m_c|/epsilon
   + |m_c| sqrt(|v_p-v_c|)/epsilon^2.
```

若要得到对 `Delta v` 线性的 Lipschitz 界，必须额外假设所有相关坐标
`v >= v_min > 0`；常数会显式依赖 `v_min`，不能隐藏。理论与数值实验都应同时报告实际
`sqrt(vhat)+epsilon` 的分位数，防止用极松的 `1/epsilon` 界伪装解释力。

## 4. 假设层级

| 层级 | 假设 | 可以研究 | 不能外推 |
|---|---|---|---|
| A0 deterministic finite-horizon | 给定初始化、batch、augmentation 与梯度序列，递推确定 | 精确冲激响应、状态置换、代码等价 | 概率、期望风险、泛化 |
| A1 bounded | `||g_t||<=G`，污染 `||delta_t||<=D` | 有限窗状态/参数上界 | 重尾异常、无界损失 |
| A2 centered stochastic | 条件均值无偏，方差/次高斯参数有界 | 高概率或期望界、seed 方差 | 系统性偏差、相关噪声 |
| A3 heavy-tailed | 仅有限 `p` 阶矩或污染分布重尾 | robust aggregation/截断的尾界 | 次高斯集中率 |
| A4 mixture contamination | 独立或条件独立 Bernoulli 污染，比例/上界可定义 | 误接纳/误拒绝项、比例依赖界 | 持续 burst、课程式相关性 |
| A5 biased / temporally correlated | 污染有非零均值、burst 或 Markov 相关 | 偏差累计、恢复时间、失败边界 | 独立混合模型结论 |
| A6 systematic/endogenous | 错误方向与任务梯度长期一致，且梯度依赖已污染参数 | 不可识别性、停止条件 | 无 side information 的普遍检测保证 |

每个 theorem/lemma 必须在标题旁写明使用的最高层级。A1/A2 下通过的候选，不能在摘要中被描述成
对 A5/A6 “robust”，除非后续独立证明和实验同时覆盖。

## 5. 不可识别反例

考虑只观察当前/历史梯度、optimizer state 和内部随机数的任意算法 `A`。构造两个世界：

- 世界 P（真实目标漂移）：clean stochastic gradient 为 `h_t`，没有污染；
- 世界 Q（系统性错误）：目标本应给出 `h_t-b`，但每一步标签/数据机制都加入固定偏差 `b`，
  所以算法仍观察到 `h_t`。

两个世界给算法的观测序列和初态分布完全相同，因此 `A` 的 admission、状态和参数轨迹分布也
完全相同。若 `A` 在 P 中必须接纳 `b` 所代表的真实新方向，它就无法仅凭这些观测在 Q 中保证
拒绝同一个方向；反之亦然。

**结论边界：** 没有 clean reference、任务边界、多视图一致性、已知噪声模型或其他结构性假设时，
持续且与真实学习方向一致的系统性错误梯度一般不可识别。候选最多声称在冻结的可观测模式和
假设下减少持久污染，不能声称普遍识别 noisy gradients。R5 必须实测这个反例族；若方法稳定
接纳系统性错误，应撤销对应安全/检测主张，而不是调阈值掩盖。

## 6. 可测量量与理论—实验映射

| 理论对象 | 代码/轨迹字段 | 最小实验 | 判定用途 |
|---|---|---|---|
| `Delta m_t` | `exp_avg` 差、norm/cosine/逐层分位数、hash | M0.6 `m-only` 与 control | 一阶状态传播 |
| `Delta v_t` | `exp_avg_sq` 差及分母分位数 | `v-only` 与 control | 二阶与分母耦合 |
| interaction | `state-both - m-only - v-only` AUC | M0.6 factorial | 禁止把 m/v 当独立可加 |
| `Delta theta_t` | trainable parameter delta、update delta | parameter-only/state branches | 状态影响到参数的桥 |
| `e_t` | 同 batch 上两分支 raw gradient 差 | 短 horizon paired replay | 内生反馈卷积 |
| admission error | synthetic clean/noisy offline tag 与 gate score | 只在合成训练离线分析 | false accept/reject；tag 不进算法 |
| hard-clean harm | class、difficulty stratum、clean gate loss | R5 冻结分层 | 安全界而非总体均值 |
| systematic bias | 人工构造长期同向 pulse/burst | R5 负对照 | 触发不可识别停止条件 |

所有 state snapshot 必须在同一语义点采集（推荐 optimizer step 前 raw grad、state write 后、parameter
update 后），否则不同实现的 `t` 会错一位。混合精度下同时记录 unscaled fp32 gradient summary、
state dtype 和 overflow skip；跳步时 `step`、m、v、参数均不得静默改变。

## 7. 候选冻结后必须填写的接口

以下字段当前故意留空，只有精确候选 recurrence 与代码冻结后才能填写：

```yaml
candidate_version: null
observed_signal: null
ephemeral_current_path: null
persistent_state_write_m: null
persistent_state_write_v: null
history_state: null
admission_function: null
initialization_and_bias_correction: null
weight_decay_order: null
mixed_precision_and_skip_semantics: null
nonstandard_hyperparameters: []
```

候选理论义务：

1. 写出即时参数通路与 persistent `m/v/history` 的逐步 recurrence；
2. 证明“保护关闭”时与冻结 AdamW 数值等价，或逐项解释不可等价处；
3. 在 A0 下给出单脉冲精确差，在所声称的最高概率层级给出有限窗上界；
4. 上界显式含误接纳、误拒绝、分母耦合、weight decay 和内生 `e_t`；
5. 每个符号映射到实际 state key、dtype、update order 与单元测试；
6. 用第 5 节反例说明不保证什么，并将其变成 R5 负对照。

## 8. 理论硬停止

满足任一项就不能把 C6 写成方法贡献：候选 recurrence 与文献中的现有方法等价；证明只适用于
与代码不同的 update order；需要候选实际不具备的 clean reference/已知 noise rate；把固定未来
梯度结论表述成真实耦合训练保证；关键常数只能取到使上界无信息；或系统性偏置反例与论文的
宽泛鲁棒性措辞矛盾且不愿收窄主张。
