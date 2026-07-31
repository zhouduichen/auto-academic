# M1 后统计、功效与数据门合同

> 冻结日期：2026-07-31（未查看 M1 confirmation 结果）
> 适用范围：只规定 M1 完成后 R2–R5 如何用已封存的成对方差规划新数据门。
> 不追溯修改：M0.6 和 M1 的 10-seed / 8-seed 判定、阈值与 `GO/NO-GO/INCONCLUSIVE`
> 仍以其现有 specification 为唯一标准。

## 1. 统计单元和共同端点

- 实验单元是**独立训练 seed**。同 seed 内方法共享 initialization、train order、noise 与
  augmentation manifest，方法差构成 paired observation。
- 同 seed 的多个 noise pulse、checkpoint、epoch、batch、类别、样本或预测都不是独立重复。
- 若一个 train seed 有多个预注册 noise seed，先按预注册权重在 train-seed 内平均，再跨 seed 推断；
  除非设计明确把二者做成独立交叉随机化且训练本身独立。
- primary fixed-step endpoint 是最终预注册 optimizer step 的指标，不选 best checkpoint。
- primary fixed-wall-clock endpoint 是不超过 AdamW 配对 wall-clock budget 的最后一个预注册评价点，
  不在 checkpoints 间插值。
- 准确率差全部用 percentage points（pp），不是相对百分比。

设 primary paired differences 为 `d_s = metric(candidate,s)-metric(reference,s)`，`n` 为完整配对
seed 数，`dbar` 和 `sd` 分别为均值与 `ddof=1` 标准差。两侧 95% t interval：

```text
dbar ± t_(0.975,n-1) sd/sqrt(n).
```

clean non-inferiority margin 为 `Delta_NI=0.5 pp`，其一侧 95% lower bound：

```text
L_NI = dbar - t_(0.95,n-1) sd/sqrt(n).
```

只有 `L_NI > -0.5` 才通过。边界相等不算通过。

## 2. M1 后允许继续的首要条件

只有完整 M1 判为 `GO` 才能用本合同启动 R2/R3。`NO-GO` 停止算法线；`INCONCLUSIVE` 不允许
用功效函数追加同一门 seeds。M1 中任何缺失 formal cell、test isolation 失败或无效配对先按原
协议处理，不能抽取“看起来正常的 seeds”作为后续方差输入。

## 3. 唯一功效函数

### 3.1 方差输入

对未来某个以 paired pp difference 表示的 primary endpoint，从 M1 完整 8-seed 对应差值计算
`s_M1`。使用正态方差枢轴的一侧 80% upper confidence bound：

```text
sigma_U80 = sqrt((n0-1) s_M1^2 / chi2_(0.20,n0-1)),  n0=8.
```

规划标准差：

```text
same-domain mechanism endpoint:
  sigma_plan = max(0.25 pp, sigma_U80)

new dataset or backbone endpoint:
  sigma_plan = max(0.25 pp, 1.25 * sigma_U80)
```

`0.25 pp` 是防止偶然零/近零方差导致荒谬小样本的冻结 floor；`1.25` 是 synthetic-to-real 或
结构迁移的异质性 inflation。不能用 M1 的 observed mean 替换目标效应，也不能在看见新 gate
结果后降低 inflation。若 M1 没有语义匹配的 paired endpoint，则该指标不能成为 powered primary
claim；必须在打开数据门前给出外部独立方差来源或把它降为 exploratory。

### 3.2 目标效应

- C3 真实噪声 noisy superiority 的 minimum meaningful effect 固定为 `delta=1.0 pp`；
- clean non-inferiority 以真实均值差 `0 pp` 规划，距 `-0.5 pp` 边界的效应为 `0.5 pp`；
- candidate 对 direct baseline 的 fixed-step Pareto margin 为 `-0.5 pp`，不把“未显著差”当等价；
- C2、C4、C5 若使用不同 primary endpoint，必须由科学意义在其 design 中预先给出 `delta`/margin，
  不得使用该 gate 的 pilot mean 倒推。

### 3.3 精确 paired-t power

superiority 使用两侧 `alpha=0.05`（与两侧 95% interval 一致）。对候选样本数 `n`：

```text
nu = n-1
ncp = sqrt(n) * delta / sigma_plan
c = t_(0.975,nu)
power_sup(n) = P[T_nu(ncp) > c] + P[T_nu(ncp) < -c].
```

non-inferiority 使用一侧 `alpha=0.05`，在真实 clean difference 规划值为 0 时：

```text
ncp_NI = sqrt(n) * 0.5 / sigma_plan_clean
c_NI = t_(0.95,nu)
power_NI(n) = P[T_nu(ncp_NI) > c_NI].
```

共同主要 endpoint 是 intersection-union test（IUT）：noisy superiority 与 clean NI 必须都过。
为使联合成功率有至少 0.80 的保守下界，分别要求 `power_sup>=0.90` 和 `power_NI>=0.90`；由
union bound，两个失败概率之和不超过 0.20。无需因 IUT 再调低各自 alpha，因为拒绝总 null
必须拒绝每个 component null。

样本量算法：

```text
n_sup = first n>=5 with power_sup(n)>=0.90
n_NI  = first n>=5 with power_NI(n)>=0.90
n_required = max(5, n_sup, n_NI)
```

`n_required` 向上取整为完整配对 seeds。每个阶段必须先依据 M1 固定输入机械算出它，再根据
M1 实测吞吐制定一个纯资源性的 `n_cap`，二者都在下载/打开新数据门前提交。若
`n_required > n_cap`，结论是**该主张在当前资源下不可检验**，不是把 seed 数砍到 5 或降低功效。
本 L0 文件不在 M1 方差出现前伪造最终 seed 数，也不预设一个方便的 `n_cap`。

参考实现必须使用 `scipy.stats.t`、`scipy.stats.nct` 和 `scipy.stats.chi2`，并写入 SciPy version、
输入差值 hash、所有分位数、逐 n power 与最终选择。独立单元测试用 Monte Carlo（固定 RNG，至少
200,000 draws）验证解析 power 误差不超过 0.005；Monte Carlo 只校验函数，不参与选 n。

## 4. Claim-by-claim 判定

### C2 因果归因（R2）

primary contrasts 必须在 design 中从以下关系选择并冻结：完整候选 vs persistent-state admission
关闭；完整候选 vs rate-matched random admission。两者回答不同反例，不能择优报告。至少要求：

- 完整候选相对 state-disabled 达到预注册实际效应和 interval；
- rate-matched random 不足以解释完整收益；
- 阻断即时通路若获得主要收益，则触发“收益来自丢弃/削弱当前困难样本”硬停止；
- state hash 与 recurrence audit 证明消融只改变声明的通路。

R2 的多个机制对照按闭合检验或预注册固定顺序执行；未进入 primary 顺序的 ablation 是 secondary。

### C3 真实噪声（R3）

共同主要判定：

1. candidate - strongest simple 的 noisy/real-label fixed-step top-1 mean 至少 `+1.0 pp` 且两侧
   95% paired interval lower endpoint `>0`；
2. candidate - AdamW 的 clean evaluation 一侧 95% lower endpoint `>-0.5 pp`；
3. fixed-wall-clock noisy mean improvement `>0`；
4. wall-clock 与 peak VRAM 各不超过设计中冻结的 5% overhead；
5. candidate 不被 direct baseline 在 accuracy、wall-clock、VRAM 三项严格支配；
6. hard-clean、minority/worst-class 安全界全部通过。

前两项是 IUT co-primary；3–6 是必须同时满足的 guardrails。失败任一项都不能宣称 C3 支持。

### C4/C5 与 direct Pareto

- C4 每次只更换一个实质因素（dataset scale、backbone family 或 full-FT），不做结果驱动的完整
  笛卡尔积。第一项预注册为 primary，其余按固定顺序 replication。
- C5 每个安全 endpoint 使用预注册非劣 margin；总体准确率改善不能覆盖 minority/hard-clean
  失败。若系统性偏置不可识别反例成立，应明确撤销“检测所有噪声”的主张。
- candidate 对 direct baseline 的 `-0.5 pp` fixed-step 非劣或 fixed-wall-clock superiority 是
  Pareto guardrail，不是通过普通双侧“p>0.05”证明等价。

## 5. Multiplicity 和报告层级

1. 每个 claim 只有一个 primary contrast 或一个全部必须通过的 IUT 集合。
2. IUT components 不做 alpha 分摊；任一 component 不通过则整个 claim 不通过。
3. 同一 claim family 的多个 secondary hypotheses 用 Holm step-down 调整，报告 raw 与 adjusted p、
   同时给 effect 和 CI。只有 adjustment 后通过的可称 confirmatory secondary。
4. 多数据集支持同一主张时使用预注册 fixed sequence：primary dataset 失败即停止正式主张；后续
   数据集结果仅 exploratory。若设计要求同时检验多个数据集，则对数据集 family 使用 Holm。
5. balanced accuracy、macro-F1、NLL、ECE、AUC、各 class 和 admission diagnostics 若未被指定为
   primary/safety gate，均为 secondary/exploratory，不能因显著而升级。
6. 不以 Bayesian、bootstrap 或 permutation sensitivity 替代冻结 t-test 主判定；可以并列报告，
   但分歧必须解释，不能挑有利方法。

## 6. 缺失、失败、重试和异常值

- 基础设施故障只允许 exact retry 一次：同 source、config、seed、manifest、idempotency key；失败
  artifact 永久保留。不同 GPU、依赖或 batch size 不叫 exact retry。
- OOM、NaN/Inf、numerical divergence、algorithm timeout、状态损坏和超出预注册资源上限是方法
  失败，不可当基础设施故障重抽 seed。
- 不删除 outlier，不 winsorize，不根据 Cook distance/箱线图重跑；全部 paired effects 原样报告。
- 不做均值/last observation imputation。任何 formal cell 缺失则该方法/claim 不能 GO。
- 同一 seed 的一方失败时保留另一方 artifact，但 paired formal comparison 失败；不能临时改成
  unpaired test。
- overflow skip、zero-step、重复 sample ID、错 noise manifest、test access、hash mismatch 或非有限
  metric 均 fail closed。
- 预注册的 exact retry 仍失败时不追加替代 seed。新 seeds 只允许在新的用户批准设计和未使用
  数据门中出现，旧结论保持失败/不确定。

## 7. Test-once 与跨门复用

- tuning 可按冻结 search budget 使用；development 只做结构/继续决策；final 在全矩阵完成、代码和
  analysis hash 冻结后一次性评价。
- 看过 development 后可以停止或冻结，但不能改方法再在同 development 上称 confirmatory。
- 看过 final 后发生纯报告 bug，可用同 prediction 文件重算；若需重新训练、换 checkpoint 或改变
  prediction，则必须新数据门，原 final 保留。
- M1 seeds/confirmation gate 只用于 M1 与功效输入，不能与 R2–R5 合并成更大的正式样本。
- 一个真实数据集的 tuning/development/final 不能同时为另一算法版本提供 fresh gate。
- 公开 test labels 即使人人可得，也要通过本地访问 guard；“没有手工看标签”不等于没有 test leakage。

## 8. 机器可读证据字段

每个 stage 必须产生不可变 `STATISTICAL_DESIGN.json` 与 `STATISTICAL_RESULT.json`。最小字段：

```json
{
  "design_version": null,
  "claim_id": null,
  "method_versions": {},
  "dataset_version": null,
  "gate_hashes": {"tuning": null, "development": null, "final": null},
  "experimental_unit": "paired_train_seed",
  "seed_manifest_hash": null,
  "primary_contrasts": [],
  "endpoint_units": "percentage_points",
  "fixed_step": null,
  "fixed_wall_clock_rule": null,
  "alpha": 0.05,
  "target_marginal_power": 0.90,
  "minimum_effect_pp": null,
  "noninferiority_margin_pp": null,
  "m1_difference_hash": null,
  "m1_sd_pp": null,
  "sigma_u80_pp": null,
  "heterogeneity_multiplier": null,
  "sigma_floor_pp": 0.25,
  "sigma_plan_pp": null,
  "n_required": null,
  "n_cap": null,
  "multiplicity_family": null,
  "holm_order": [],
  "retry_policy": "one_exact_infrastructure_retry",
  "source_commit": null,
  "config_hash": null,
  "analysis_code_hash": null,
  "environment_lock_hash": null,
  "test_access_log_hash": null,
  "paired_differences": [],
  "estimate_pp": null,
  "confidence_interval": null,
  "raw_p": null,
  "adjusted_p": null,
  "guardrails": {},
  "decision": null,
  "artifact_manifest_sha256": null
}
```

设计文件在数据门打开前填写到 `n_cap`；结果文件只能引用设计 hash，不允许覆盖设计。判定程序必须
从 raw per-seed metrics 重算 paired differences，而不是信任执行者摘要或 ARIS 文本结论。

## 9. 冻结顺序与硬停止

```text
封存 M1 全部 64-run artifact
  -> 独立验证 M1 GO
  -> 计算并提交 M1 paired variance 输入
  -> 冻结下一 claim、minimum effect、comparisons、n_required/n_cap
  -> 冻结方法/数据/analysis hashes
  -> 才允许下载或打开新数据门
```

若功效不可承担、没有独立 fresh gate、formal cell 缺失、primary 不通过、方法被严格支配、或安全
界失败，结论是停止/收窄主张。不得通过增加未预注册 endpoints、改用最优 seed、降低非劣界、
事后单侧检验或把 exploratory 结果升级来挽救。
