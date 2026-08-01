# M1 可执行协议包设计

> 日期：2026-08-01（Asia/Shanghai）
> 状态：设计已批准；尚未实现
> 适用仓库：`auto-academic`
> 目的：把 M1 中不依赖候选 recurrence、GPU 训练或未来 gate 结果的协议，冻结为可校验、
> 可哈希、可由 CPU 测试验证的最小包。

## 1. 决策摘要

本设计采用一个独立的 `experiments.m1_protocol` 包作为 M1 协议的单一机器可读来源。它包含：

1. AdamW、Clip-AdamW、beta-retuned AdamW、AdaPNM 与 ELR 的精确 12 点网格；
2. 以 Xia 等人 Algorithm 2 为主干、加入数值归一化与 exact-rate 控制的 40% synthetic
   instance-dependent label-noise 协议；
3. 分阶段 Pydantic 模型、生成后提交的 JSON Schema、稳定 JSON 编码和 artifact 哈希规则；
4. M1 后 paired-t 功效的纯数学计算器和带 `M1_DECISION=GO` 授权门的规划入口；
5. 不下载数据、不加载模型、不启动 GPU 的 CPU 单元测试和结构测试。

本设计同时修正原 M1 规格中的一个循环依赖：不能先“用调好的 AdamW 校准预算”，再使用
该预算调 AdamW。预算校准必须改用预先冻结、与搜索结果无关的 calibration anchor。

本包不是训练器、任务平台或 Windows 控制层。它不实现 Protect-M/Protect-MV，不读取
development/confirmation gate，不提交 GPU 任务，也不扩大 M1 或 M1 后实验范围。

## 2. 与现有规格的关系

本设计是以下文档的窄范围补充：

- `docs/superpowers/specs/2026-07-31-m06-m1-quality-first-experiment-design.md`；
- `docs/research-direction/13_baseline_compatibility_matrix.md`；
- `docs/research-direction/14_dataset_protocol_matrix.md`；
- `docs/research-direction/16_post_m1_statistics_contract.md`。

它不改变以下已冻结内容：

- CIFAR-100 的 `40,000 / 2,500 / 2,500 / 5,000 / untouched test` 分区；
- calibration、tuning、M1-Dev、M1-Confirm 的 seed 隔离；
- successive-halving 的 `12 -> 4 -> 2` 结构和每方法 18 full-run-equivalent 预算；
- M1-Dev、M1-Confirm 的方法矩阵、效应门、clean non-inferiority、效率门和停止规则；
- M1-Confirm 的 8 个配对 seeds 和完整 64-run 要求；
- M1 后统计合同中的方差上界、功效、multiplicity、缺失和重试规则。

它对原 M1 规格第 7–8 节作一项规范性修正：原文 `Calibration uses tuned-default AdamW`
被本设计第 4 节的固定 calibration anchor 和非循环执行顺序取代。实现计划必须先同步修订
原文，且必须在任何 M1 calibration/tuning artifact 产生前完成。

## 3. 范围与非目标

### 3.1 本轮交付范围

```text
experiments/m1_protocol/
  __init__.py
  models.py
  canonical.py
  grids.py
  noise.py
  power.py
  validate.py

  specs/
    baseline_grids.json
    noise_protocol.json

  schemas/
    baseline_grid_spec.schema.json
    noise_protocol_spec.schema.json
    calibration_result.schema.json
    adamw_selection.schema.json
    resolved_baseline_grids.schema.json
    noise_manifest_header.schema.json
    post_m1_power_input.schema.json
    post_m1_power_result.schema.json

tests/
  test_m1_protocol_models.py
  test_m1_grids.py
  test_m1_noise.py
  test_m1_power.py

docs/research-direction/
  17_m1_executable_protocol.md
```

### 3.2 明确不做

- 不实现 CIFAR loader、ViT、LoRA、optimizer training loop 或 checkpoint；
- 不下载 CIFAR-100、预训练权重或任何未来数据集；
- 不生成当前正式 40,000-sample noise artifact；
- 不实现或猜测 Protect-M/Protect-MV recurrence 和数值网格；
- 不把协议入口加入 `arw` CLI，不创建通用实验平台；
- 不连接 Windows，不排队、不停止、不修改 GPU 任务；
- 不生成 M1 `GO/NO-GO/INCONCLUSIVE` 结论或未来数据门最终 seed 数；
- 不创建论文 claim/evidence 模板。

## 4. 消除 calibration 与 tuning 的循环依赖

### 4.1 固定 calibration anchor

预算校准使用下列、在任何 M1 tuning 结果之前冻结的 anchor：

```text
optimizer       = AdamW
learning_rate   = 3e-4
weight_decay    = 0.01
betas           = (0.9, 0.999)
eps             = 1e-8
amsgrad         = false
```

模型、LoRA、batch、augmentation、数据分区、AMP、scheduler 和 parameter groups 与 M1 公共
协议一致。该点与 M0 使用的 AdamW 中心配置一致，但其选择依据是预先存在的协议锚点，不是
M1 tuning 表现。

Calibration anchor：

- 只使用 train seeds `101, 102`；
- clean 无 noise seed；noisy 使用 `1101, 1102`；
- 最多 50 epochs；
- 只在 tuning validation 上执行原规格定义的预算判定；
- 不进入任何方法性能表、显著性检验或 strongest-baseline 选择。

### 4.2 非循环执行顺序

绑定顺序为：

```text
冻结 split、noise protocol、calibration anchor 和 baseline grid spec
  -> 运行 calibration anchor
  -> 冻结 fixed optimizer-step budget 与 G_ref
  -> 在相同 budget 下先调 AdamW 的 12 个配置
  -> 冻结 AdamW selection artifact
  -> 解析其他 baseline 的绝对网格
  -> 在相同 successive-halving schedule 下调其他方法
```

AdamW 和其他方法都使用同一个 calibration-derived step budget，所以不存在 AdamW 先使用
50-epoch 搜索而其他方法使用较短预算的不对称。

### 4.3 `G_ref` 定义

Calibration anchor 的每一步记录 AMP unscale 后、任何 clipping 前、覆盖全部可训练参数的
global L2 gradient norm。定义：

```text
G_ref = median(
  finite global gradient norms from both conditions and both calibration seeds,
  after dropping the first 10% optimizer steps of each run
)
```

若任一正式 calibration cell 缺失、出现非有限 norm、optimizer-step 数不一致或所得
`G_ref <= 0`，calibration 失败关闭，不能解析 Clip-AdamW 网格。

## 5. Artifact DAG 与生命周期

预注册 spec 和运行后 observation 分开建模：

```text
baseline_grid_spec.json -----+
                              +--> calibration_result.json
noise_protocol_spec.json ----+             |
                                            +--> adamw_selection.json
                                                      |
                                                      +--> resolved_baseline_grids.json
```

### 5.1 预提交 spec

仓库当前只提交：

- `specs/baseline_grids.json`；
- `specs/noise_protocol.json`；
- 对应 JSON Schema。

### 5.2 运行后 artifact

以下文件在真实上游结果产生前不存在，不能用占位数值提交：

- `calibration_result.json`；
- `adamw_selection.json`；
- `resolved_baseline_grids.json`；
- 正式 noise training/audit bundles；
- `post_m1_power_result.json`。

每个下游 artifact 必须包含：

- 自身 schema ID/version；
- 所有直接上游文件的 SHA-256；
- source commit；
- `uv.lock` SHA-256；
- 创建时间仅作审计，不进入语义 hash；
- canonical payload SHA-256。

任一上游字节变化，下游 validator 必须拒绝；不得静默重算并覆盖旧 artifact。

## 6. 配置模型、Schema 与 canonical JSON

### 6.1 Pydantic 规则

所有模型使用 Pydantic v2 strict configuration：

```text
extra = forbid
strict = true
allow_inf_nan = false
```

禁止：

- 字符串到数值的隐式转换；
- 未声明字段；
- NaN、正负 Infinity；
- 重复 config ID；
- 未知 method、unit、artifact role 或 lifecycle status；
- 把 unresolved spec 当作 resolved grid 使用。

Schema 由 `uv.lock` 锁定的 Pydantic `2.13.4` 生成。生成和测试统一执行
`uv run --locked`。只承诺在锁定环境中可字节级重生成，不承诺任意 Pydantic 版本间相同。

### 6.2 Artifact-specific schema

不同阶段使用不同模型，不以一个含大量 optional 字段的“万能 schema”表示生命周期：

- `BaselineGridSpec`：只接受预注册相对/绝对定义；
- `CalibrationResult`：要求四个 calibration cells 和 `G_ref`；
- `AdamWSelection`：要求完整 tuning provenance 和唯一选中 config；
- `ResolvedBaselineGrids`：要求所有引用已解析为绝对数值；
- `NoiseProtocolSpec`：只表示生成算法；
- `NoiseManifestHeader`：只表示数组索引、shape、dtype 与 hash；
- `PostM1PowerInput/Result`：只表示 M1 后统计规划。

### 6.3 稳定 JSON

`canonical.py` 使用标准库实现仓库内稳定编码：

```text
encoding       = UTF-8
sort_keys      = true
separators     = (",", ":")
ensure_ascii   = false
allow_nan      = false
final newline  = exactly one
```

SHA-256 对最终文件字节计算。该编码是仓库协议，不宣称完整实现 RFC 8785。时间戳、绝对路径、
hostname 和临时目录不能进入 canonical payload。

## 7. Baseline 12 点网格

### 7.1 公共 optimizer 约定

- optimizer 只接收 `requires_grad=True` 的 LoRA query/value 与分类头参数；
- LoRA A/B 和分类头 weight 进入 decay group；bias 进入 no-decay group；
- 若未来出现可训练 normalization 参数，其 scale/bias 进入 no-decay group；
- clean/noisy 必须使用同一选中 config；
- Adam 类 `eps=1e-8`；
- AdamW、Clip-AdamW、beta-retuned AdamW 和 ELR 的 AdamW 使用 `amsgrad=false`；
- 所有非 AdamW 方法继承最终选中 AdamW 的绝对 LR、WD 与 parameter-group 规则；
- 方法不得通过隐藏 parameter group 或 condition-specific override 改写公共参数。

### 7.2 AdamW

```text
learning_rate in {3e-5, 1e-4, 3e-4, 1e-3}
weight_decay  in {0.0, 0.01, 0.1}
betas            = (0.9, 0.999)
```

按 learning-rate 主序、weight-decay 次序生成恰好 12 个稳定 ID。

### 7.3 Clip-AdamW

沿用选中 AdamW，global max norm 为：

```text
max_norm = G_ref * 2**e
e in {-3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
       0.0,  0.5,  1.0,  1.5,  2.0,  2.5}
```

裁剪顺序固定为 AMP unscale、全局范数计算、一次 global L2 clipping、optimizer step。禁止逐层、
逐 tensor 或逐 parameter-group clipping。关闭裁剪的等价性测试使用独立的 `max_norm=inf`
测试输入，不作为 12 点 tuning config。

### 7.4 Beta-retuned AdamW

```text
beta1 in {0.5, 0.7, 0.9, 0.95}
beta2 in {0.95, 0.99, 0.999}
```

恰好 12 点，并必须包含标准 `(0.9, 0.999)` 等价 anchor。

### 7.5 AdaPNM

```text
beta1 in {0.7, 0.9, 0.95}
beta2    = 0.999
beta3 in {0.25, 0.5, 1.0, 2.0}
eps      = 1e-8
amsgrad  = true
decoupled = true
```

恰好 12 点，包含作者实现的 `beta3=1` 中心值。AMSGrad 的额外状态、VRAM 和时间全部收费。
算法来源为 [PNM/AdaPNM 论文与作者代码入口](https://proceedings.mlr.press/v139/xie21h.html)。

### 7.6 ELR

保留选中 AdamW optimizer，只改变 ELR target EMA 和 loss：

```text
beta   in {0.7, 0.9, 0.99}
lambda in {1.0, 3.0, 5.0, 7.0}
```

恰好 12 点，包含 CIFAR-100 官方配置中心 `(beta=0.9, lambda=7)`。算法来源为
[ELR NeurIPS 2020 论文页](https://proceedings.neurips.cc/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html)。

### 7.7 Candidate 边界

当前 `baseline_grids.json` 不出现 Protect-M/Protect-MV 的 JSON 实例。`models.py` 只定义未来
`CandidateGridPair` 的接口约束：

- 两边各恰好 12 个配置；
- config ID 和全部参数逐项相同；
- 唯一允许差异为 `protection_target = m | mv`；
- 最多三个非标准 tunable hyperparameters；
- 公共 AdamW 参数从冻结 selection 继承；
- protection disabled 的独立测试必须与 AdamW 状态和参数更新等价。

候选 recurrence 冻结后必须经新设计修订创建 `candidate_grid_pair.json`，不能由当前实现自动填充。

## 8. 40% Xia-style exact-rate IDN

### 8.1 来源与主张边界

生成结构以 Xia 等人 NeurIPS 2020 论文 Algorithm 2 为主干：样本污染率来自截断正态，错误
类别概率由样本特征与 clean-class-specific 随机矩阵决定。原始来源：
[Part-dependent Label Noise: Towards Instance-dependent Label Noise](https://proceedings.neurips.cc/paper/2020/file/5607fe8879e4fd269e88387e8cb30b7e-Paper.pdf)。

本协议加入了标准化/L2 normalization、per-class exact-rate calibration 和 fixed-size PPS，故只能
称为：

> Xia-style instance-dependent noise with preregistered normalization and exact-rate control.

不能声称逐字复现论文生成器，也不能把单一 synthetic mechanism 包装成人类标签噪声普适结论。

### 8.2 输入和禁止信息

生成器只接受：

- split role 明确为 train 的 40,000 个稳定 sample IDs；
- 对应原始 CIFAR-100 `32 x 32 x 3` uint8 图像；
- 对应 clean labels，仅限离线合成；
- 已校验的 train-split digest；
- noise seed；
- `noise_protocol.json`。

生成器接口不接受 tuning、development、confirmation、official test 样本或统计。输入必须每类
恰好 400 个，ID 唯一。任何数量、角色、class count 或 digest 不符立即失败。

### 8.3 特征变换

按 stable sample ID 排序后，在 float64 中：

1. uint8 转为 `[0,1]`；
2. 只用 40,000 train 图像计算 RGB channel mean/std；
3. 逐通道标准化；
4. 展平为 3,072 维；
5. 每样本 L2 normalize 得到 `x_i`。

若 channel std 或样本 L2 norm 非正/非有限，生成失败。保存 channel statistics、预处理配置和
feature-transform digest。该变换名冻结为 `cifar100-train-standardized-l2-pixels-v1`。

### 8.4 RNG 域分离

统一使用 NumPy `Generator(PCG64DXSM)`。每个流的 128-bit entropy 定义为：

```text
SHA256(
  UTF8("m1-idn-v1") || 0x00 ||
  uint64_big_endian(noise_seed) || 0x00 ||
  UTF8(domain)
)[:16]
```

域至少包括：

```text
base-rate/<class-id>
matrix/<class-id>
pps-order/<class-id>
pps-start/<class-id>
destination/<sample-id>
```

不使用 Python `hash()`、全局 NumPy RNG 或不同阶段共享的可变 RNG 状态。

### 8.5 基础污染率与 exact-rate calibration

对每个样本独立生成：

```text
q_i ~ TruncNormal(mean=0.4, std=0.1, lower=0, upper=1)
```

数值实现将 `q_i` 限制在 `[1e-12, 1-1e-12]`，并同时保存未限制的 draw 和用于计算的值。

对每个 clean class `y`，求唯一 `delta_y`：

```text
pi_i = sigmoid(logit(q_i) + delta_y)
sum(pi_i for clean_class y) = 160
```

使用固定次数、固定 bracket 的 float64 bisection，不依赖结果驱动容差。结果必须满足每类求和
绝对误差 `<= 1e-10`，且每个 `0 < pi_i < 1`。`q_i` 是基础 draw，`pi_i` 才是 exact-rate
抽样的一阶 inclusion probability；二者不得混称。

### 8.6 Fixed-size systematic PPS

每个 clean class：

1. 使用 `pps-order/<class-id>` 对 400 个稳定 ID 产生随机排列；
2. 使用 `pps-start/<class-id>` 生成 `U in [0,1)`；
3. 令 thresholds 为 `U + {0,1,...,159}`；
4. 在排列后的 cumulative `pi_i` 上以 `searchsorted(..., side="right")` 选择样本；
5. 为避免累计舍入影响，将最后一个 cumulative value 显式设为 `160.0`。

因为每个 `pi_i < 1` 且总和为整数 160，该算法每类恰好选择 160 个不同样本，并保留一阶
inclusion probability `pi_i`。全局必须恰好 16,000 个 corruption IDs。

### 8.7 错误目标类别

对每个 noise seed 和 clean class：

```text
W_y in R^(3072 x 100)
W_y[a,b] ~ Normal(0,1), float64
```

对样本 `i`：

```text
logit_ij = x_i^T W_y[:,j] / tau_destination
tau_destination = 1.0
r_i,y_i = 0
r_i,j = off-diagonal softmax(logit_i), j != y_i
```

仅对被 PPS 选中的样本，用独立 `destination/<sample-id>` 流从 `r_i` 抽取 noisy label，保证
`noisy_label != clean_label`。完整一阶 transition row 为：

```text
T_i,y_i = 1 - pi_i
T_i,j   = pi_i * r_i,j, j != y_i
```

所有 softmax 使用 subtract-max 稳定实现。每行 `T_i` 必须有限、非负，且 float64 行和误差
`<= 1e-12`。

### 8.8 跨平台复现边界

大规模 float matrix multiplication 的末位可能受 BLAS/硬件影响。正式规则是：

- 在一个 `uv run --locked` 的可信 CPU 预处理环境中生成一次正式 bundles；
- 保存最终 transition、noisy labels 和全部 digest；
- Windows 训练端只验证并消费 training bundle，不重新生成；
- artifact 字节和 SHA-256 是正式训练输入身份；
- 其他平台重生成只做容差内科学复查，不要求大矩阵输出 bit-exact；
- golden hash 只覆盖不依赖 BLAS 的 RNG、bisection、PPS 和小型显式矩阵 fixture。

## 9. Training bundle 与 audit bundle 隔离

正式生成产生两个不共享父目录能力的 bundle：

```text
m1-noise-training-<opaque-id>/
  sample_ids.npy
  noisy_labels.npy
  public_manifest.json

m1-noise-audit-<opaque-id>/
  clean_labels.npy
  corruption_mask.npy
  q_base_raw.npy
  q_base_used.npy
  inclusion_probability.npy
  transition_probabilities.npy
  private_manifest.json
```

两个 bundle 由控制层私有映射关联。Windows noisy training job 只接收 training bundle 的精确
路径和 hash；candidate optimizer API 不接收 bundle 路径，只接收正常 batch gradient/state。

`public_manifest.json` 采用字段白名单，只含：

- schema/version；
- opaque artifact ID；
- sample count；
- `sample_ids.npy`、`noisy_labels.npy` 的相对文件名、dtype、shape、SHA-256；
- source protocol public ID；
- public-manifest payload hash。

它不含 clean labels、corruption mask、noise rate、noise seed、`q_i`、`pi_i` 或 transition。

Audit bundle 保存合成诊断所需全部信息，但不复制到普通训练工作目录。该设计减少正常 runner、
candidate 和日志意外获得 clean-reference signal 的能力，不声称能防御恶意代码主动读取原始
CIFAR metadata；它是接口/能力隔离，不是安全沙箱。

所有 `.npy` 必须：

- `allow_pickle=False`；
- 使用 schema 声明的显式 dtype/endian；
- 与 manifest shape 完全一致；
- 按 stable sample ID 对齐；
- 通过文件 SHA-256 验证。

## 10. 功效计算器

### 10.1 两层接口

`power.py` 提供：

1. `paired_t_power(...)` 等纯数学函数，不依赖 workflow 状态；
2. `plan_post_m1(...)` 协议入口，要求 sealed M1 `GO` artifact 和完整 provenance。

纯数学函数可供单元测试和独立复核使用。协议入口必须拒绝：

- `NO-GO` 或 `INCONCLUSIVE`；
- 少于或多于 8 个 M1 配对 seeds；
- 重复/错配 seed ID；
- 缺失 formal cell；
- 非有限 difference；
- 与未来 primary endpoint 语义不匹配的 M1 variance；
- 无效 source/result hash。

### 10.2 方差规划

对 8 个 paired pp differences：

```text
s_M1 = sample standard deviation, ddof=1
sigma_U80 = sqrt(7 * s_M1^2 / chi2_ppf(0.20, df=7))
```

规划 context 是互斥 enum：

```text
same_domain
new_dataset_or_backbone
```

并定义：

```text
same_domain:
  sigma_plan = max(0.25 pp, sigma_U80)

new_dataset_or_backbone:
  sigma_plan = max(0.25 pp, 1.25 * sigma_U80)
```

dataset 与 backbone 同时变化也只应用一次 1.25，不把两个 boolean 连乘。若 M1 没有语义匹配
endpoint，必须提供在新 design 中冻结的独立外部方差，或把指标降为 exploratory。

### 10.3 精确 noncentral-t power

Superiority：

```text
df = n - 1
ncp = sqrt(n) * delta / sigma_plan
critical = t.ppf(0.975, df)
power_sup = nct.sf(critical, df, ncp) + nct.cdf(-critical, df, ncp)
```

Clean non-inferiority，以真实均值差 `0 pp` 和 margin `0.5 pp` 规划：

```text
ncp_NI = sqrt(n) * 0.5 / sigma_plan_clean
critical_NI = t.ppf(0.95, df)
power_NI = nct.sf(critical_NI, df, ncp_NI)
```

两个 IUT component 各要求 `>=0.90`，从 `n=5` 开始逐整数搜索：

```text
n_required = max(n_sup, n_NI, 5)
```

不得用 M1 observed mean 替代目标效应。`n_cap` 不传入数学求解器；资源层在独立 design 中
冻结 `n_cap`，再输出 `feasible` 或 `resource-infeasible`。资源不足不能降低功效、效应门或
自动截断为方便的 seed 数。

### 10.4 参考测试值

在 `delta=1.0 pp`、NI margin `0.5 pp`、每 component power `0.90` 下：

| `sigma_plan` | `n_sup` | `n_NI` |
|---:|---:|---:|
| `0.5 pp` | 5 | 11 |
| `1.0 pp` | 13 | 36 |
| `2.0 pp` | 44 | 139 |

输出保存 SciPy version、输入 paired differences hash、全部分位数、每个候选 `n` 的 power 和
最小满足值。

## 11. Validator 与失败关闭

`validate.py` 是薄的本包入口，可通过 `python -m experiments.m1_protocol.validate` 调用；它不
调度训练、不连接控制 API。每个子命令只验证一个明确 artifact type。

### 11.1 通用失败

- schema/version 不支持；
- JSON extra field、错误 type/unit 或非有限值；
- canonical payload/file hash 不符；
- source commit 或 `uv.lock` hash 不符；
- 上游 hash 不符或生命周期顺序越级；
- 路径逃出 artifact root；
- `.npy` 需要 pickle、dtype/shape/endian 不符；
- 重复、缺失或未排序 stable ID。

### 11.2 网格失败

- 任一激活方法不是恰好 12 个唯一 config；
- AdamW 组合或稳定 ID 不完整；
- beta-retuned AdamW 缺少标准等价 anchor；
- AdaPNM 缺少 `beta3=1`；
- ELR 缺少 `(0.9,7)`；
- 相对配置在缺失 `G_ref`/AdamW selection 时被解析；
- clean/noisy override 不一致；
- 非 AdamW 方法覆盖公共 LR/WD/parameter groups；
- 未冻结 candidate 被表示成 resolved 方法。

### 11.3 Noise 失败

- 输入不是 40,000 个唯一 train IDs 或每类不是 400；
- 任一非-train split 信息进入生成器；
- 每类 corruption count 不是 160 或总数不是 16,000；
- corrupted label 等于 clean label；
- 每类 `sum(pi)` 误差超过 `1e-10`；
- transition 非有限、为负或行和误差超过 `1e-12`；
- 输入顺序改变后按 ID 对齐的非-BLAS步骤结果改变；
- public manifest 泄露 private 字段；
- array/hash/protocol/generator digest 不符。

## 12. CPU 测试设计

### 12.1 模型与 Schema

- valid round-trip；
- strict type 和 extra-field 拒绝；
- NaN/Inf 拒绝；
- Schema 在锁定环境中重生成后字节一致；
- canonical JSON 与 golden hash；
- 不同 artifact stage 不能互相冒充。

### 12.2 网格

- 五类 baseline 各恰好 12 个稳定 ID；
- AdamW Cartesian product 完整；
- Clip exponent 解析和 `G_ref` provenance；
- inheritance 禁止隐藏 override；
- required anchor 存在；
- duplicate/boundary/unknown method 失败；
- future `CandidateGridPair` 只允许 protection target 不同。

### 12.3 Noise

快速小 fixture 覆盖：

- RNG domain separation；
- q clipping 与 delta bisection；
- exact-k systematic PPS；
- transition、不同标签和 row sum；
- sample input order invariance；
- training/audit bundle 字段隔离；
- 文件 hash、dtype、shape 和 `allow_pickle=False`。

另做一个不执行大规模随机矩阵乘法的全规模结构测试：40,000 IDs、100 classes、每类 400，
验证 exact 160/class、exact 16,000 total、数组对齐与内存规模。小 fixture 用重复 seeds 验证
systematic PPS 的 empirical inclusion frequency 与 `pi_i` 在预注册容差内一致。

大矩阵 BLAS 结果不设跨硬件 golden hash；只用显式小矩阵验证公式。

### 12.4 Power

- 第 10.4 节三个解析参考值；
- 返回的 `n` 达标且 `n-1` 不达标；
- SD floor 和一次性 1.25 inflation；
- `n0=8`、paired ID 和 semantic endpoint gate；
- fixed RNG、至少 200,000 draws 的 Monte Carlo，解析 power 误差 `<=0.005`；
- Monte Carlo 只验证函数，不参与选 `n`；
- 非 GO 状态只阻止 `plan_post_m1`，不阻止纯数学测试。

所有测试在 Mac 本地 CPU 执行，不读取 CIFAR、模型权重、M1 gate 或 Windows 状态。

## 13. 文档与使用说明

`docs/research-direction/17_m1_executable_protocol.md` 只说明：

- 每个 spec/artifact 的职责和生命周期；
- 如何在锁定环境中生成 Schema、验证 spec 和运行 CPU tests；
- calibration -> AdamW -> resolved baseline 的正确顺序；
- 如何生成/传递 training bundle 而不传递 audit bundle；
- M1 GO 后如何机械调用功效规划器；
- 常见失败的含义和恢复边界。

它不包含真实训练命令、Windows 凭据、未来 gate 值或候选 recurrence。

## 14. 验收条件

实现阶段只有全部满足才完成：

1. 原 M1 规格的 calibration 循环依赖已同步修订且无相反表述；
2. 两个 committed specs 通过对应 Pydantic 模型和 JSON Schema；
3. 五个 baseline 的 exact 12-grid 与本设计逐值一致；
4. 未冻结 candidate 不存在 JSON 实例；
5. noise 核心算法、bundle 隔离和全部失败关闭测试通过；
6. power 解析值、最小 `n` 和 200,000-draw Monte Carlo 测试通过；
7. `uv run --locked pytest`、相关 Ruff 检查通过；
8. 无 CIFAR/model 下载、无 GPU 使用、无 Windows mutation；
9. 除设计要求的文件外，不修改用户现有未跟踪目录或实验输出；
10. 实现 commit 与测试结果有清晰 handoff，但不自动启动 M1。

## 15. 后续顺序

本设计提交并经用户复核后，下一步只使用 `writing-plans` 生成逐文件实现计划。代码实现、原
M1 规格修订和测试必须在该计划获准后进行。候选机制、M1 训练和 M1 后实验仍分别需要各自的
结果门与用户授权。
