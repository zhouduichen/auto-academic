# M1 40% Instance-Dependent Label Noise Annex

> 日期：2026-08-01（Asia/Shanghai）
> 状态：用户已批准；正式 artifact 仅允许从已提交实现生成
> 适用仓库：`auto-academic`
> 上位规格：`2026-07-31-m06-m1-quality-first-experiment-design.md` 第 3、5、8、11、12 节
> 已批准的算法设计：`2026-08-01-m1-executable-protocol-package-design.md` 第 5、6、8、9、11、12 节

## 1. 目的与效力

本 annex 冻结 M1 唯一允许的 synthetic instance-dependent label-noise（IDN）生成协议。
它把上位设计中的 40% 要求细化为可实现、可哈希、可失败关闭的输入、算法和 artifact
合同。它不授权生成正式 artifact、运行 calibration、查看 tuning/development/confirmation
结果或启动 GPU 任务。

若本 annex 与上位 M1 设计冲突，采用更严格且不扩大数据访问的一方；任何改变样本集合、特征、
噪声分布、seed 映射或训练可见字段的修订，都必须在查看受影响阶段结果之前审批并使用全新版本号。
本协议只能称为：

> Xia-style instance-dependent noise with preregistered normalization and exact-rate control.

它不是 Xia 等人生成器的逐字复现，也不支持把一个 synthetic mechanism 外推为普遍的人类标注噪声。

## 2. 冻结结论

正式 noisy condition 使用以下唯一方案：

- 数据只来自 M1 split seed `20260801` 分出的 40,000 个 CIFAR-100 training-role 样本；
- 特征为仅由这 40,000 张原始 uint8 图像计算的 channel-standardized、sample-L2-normalized
  3,072 维像素；不使用预训练模型、validation/gate/test 数据或训练结果；
- base corruption draw 为 `TruncNormal(mean=0.4, std=0.1, lower=0, upper=1)`；
- 每个 clean class 单独校准 inclusion probabilities，并用 fixed-size systematic PPS 恰好选择
  `160/400` 个样本，合计恰好 `16,000/40,000`；
- wrong-class destination 由 clean-class-specific 随机矩阵与每样本特征决定，且 noisy label
  永远不同于 clean label；
- 正式生成只在一个冻结、锁依赖的可信 CPU 环境执行一次；Windows 训练端只验证和消费
  已封存 artifact，不重新生成；
- training bundle 与 private audit bundle 分离；noisy runner 和 candidate 不获得 clean labels、
  corruption mask、真实 noise rate、noise seed、propensity 或 transition probabilities；
- 同一 paired noise seed 的所有方法、配置和允许的 exact retry 使用字节完全相同的 training
  bundle hash。

不用以下替代方案：预训练 embedding IDN 会引入额外模型和权重 provenance；训练中模型 logits
IDN 会让噪声依赖 train seed、checkpoint 或方法。两者都会改变当前科学问题和配对结构。

## 3. 数据能力边界与 test 隔离

### 3.1 唯一上游数据

split 构造进程只允许实例化一次官方 CIFAR-100 training split，等价接口必须固定
`train=True, transform=None`。禁止实例化、下载、探测或统计 `train=False`。noise generator 的
接口只接受已封存的 40,000-sample train-role bundle，不接受 torchvision dataset 对象、路径
通配符或任意 split role。

输入 train-role bundle 必须包含：

- 40,000 个唯一 stable sample IDs；
- 与 ID 对齐的原始 `uint8[32,32,3]` RGB 图像；
- 与 ID 对齐的 clean labels，仅在离线生成和 private audit 能力域可见；
- dataset archive、extracted training payload、split spec、train-ID、train-image、
  train-clean-label digests，字段名与第 4 节一致；
- `split_role="train"`、`split_seed=20260801` 和每类恰好 400 的证明字段；
- `test_constructed=false`、`test_loaded=false`、`test_access_count=0`。

缺少任一字段或数量、role、digest 不符时，必须在任何随机数生成前失败。

### 3.2 Stable sample ID 与 M1 split

样本身份绑定到官方 CIFAR-100 Python training array 的原始零基索引：

```text
sample_id = "cifar100-python-train-v1/" + zero_pad(original_index, 5)
original_index in [0, 49999]
```

ID 不随分区、排列、augmentation、train seed 或 noise seed 改变。原始图像内容 hash 与 clean
label 另存，不能用内容 hash 代替 ID，以免重复图像合并。

为使本 annex 自包含，本文把与现有 M0 `stratified_split` 同型的下列算法作为待用户批准的
唯一默认；批准本文即冻结该算法：

```text
rng = numpy.random.Generator(numpy.random.PCG64(20260801))
for class_id in 0..99:
    ids = ascending original indices whose clean_label == class_id
    require len(ids) == 500
    rng.shuffle(ids)
    train        += ids[0:400]
    tuning       += ids[400:425]
    development  += ids[425:450]
    confirmation += ids[450:500]
rng.shuffle(train)
rng.shuffle(tuning)
rng.shuffle(development)
rng.shuffle(confirmation)
```

四次最终 shuffle 的调用顺序也是协议的一部分。NumPy 精确版本由 `uv.lock` 冻结。每个 role
分别保存有序 ID 数组、按该顺序的 clean-label 数组及各自 SHA-256；另保存按 sample ID 排序的
`{sample_id, clean_label, split_role}` canonical table digest。生成噪声前只把 train role 传入
private generator；其他三个 role 的 ID、标签、图像和统计都不进入其进程。

### 3.3 训练期能力隔离

noisy training data adapter 接收 training bundle 中的 `sample_ids` 和 `noisy_labels`，并按 ID
读取 image-only store。它不得载入原始 CIFAR targets 或 private audit 路径。candidate optimizer
API 只接收普通 batch 的参数、梯度、optimizer state 和 noisy training targets 所产生的训练信号；
它不接收 artifact 路径、noise seed、clean label、mask、propensity 或 transition。

clean condition 作为独立正式 cell 使用普通 clean training targets；这不允许 noisy cell 访问
同一 ID 的 clean target。training 和 audit bundle 不共享父目录能力。该隔离是接口与工作目录
能力控制，不宣称抵御主动绕过路径边界的恶意代码；正式 source review 必须确认不存在该绕过。

## 4. Canonical encoding 与 digest

JSON 均使用仓库 canonical 编码：UTF-8、keys 按字典序、separators `(",", ":")`、
`ensure_ascii=false`、`allow_nan=false`、文件末尾恰好一个换行。SHA-256 对最终文件字节计算。
绝对路径、hostname、临时目录和时间戳不进入 semantic payload；创建时间只可进入非语义审计层。
任何 `*_payload_sha256` 都先从对象中移除该 hash 字段，再对剩余 canonical payload 计算，最后
写回字段；禁止自引用 hash 或把非语义审计层混入 semantic payload。

数组使用 `.npy`、`allow_pickle=False`、C-contiguous，schema 固定 dtype、endianness、shape 和
sample-ID order。正式 schema 至少区分以下 digest：

- `dataset_archive_sha256`：批准的官方 CIFAR-100 Python archive bytes；
- `dataset_train_payload_sha256`：archive 中抽取但不反序列化 test 的 training payload bytes；
- `split_spec_sha256`：第 3.2 节 canonical 算法 spec；
- `train_ids_sha256`、`train_images_sha256`、`train_clean_labels_sha256`；
- `feature_extractor_sha256`：第 5 节 canonical transform spec；
- `feature_matrix_sha256`：按 stable ID 排序的 float64 feature bytes；
- `noise_protocol_sha256`：本 annex 对应的机器可读 noise spec；
- `generator_source_sha256`：生成器相关 tracked source files 的 canonical file list 与 hash；
- `generator_digest`：下列 canonical payload 的 SHA-256：

```text
{
  schema/version,
  noise_protocol_sha256,
  dataset_archive_sha256,
  dataset_train_payload_sha256,
  split_spec_sha256,
  train_ids_sha256,
  train_images_sha256,
  train_clean_labels_sha256,
  feature_extractor_sha256,
  generator_source_sha256,
  source_commit,
  uv_lock_sha256,
  numpy_version,
  scipy_version,
  trusted_cpu_profile_id
}
```

任何上游字节变化都产生新 artifact identity；不得静默更新 digest 或覆盖旧 artifact。

## 5. 特征提取

所有步骤按 stable sample ID 字典序处理，并使用 float64：

1. 把原始 `uint8[H,W,C]` 精确转换为 `[0,1]`：`z = uint8 / 255.0`；
2. 仅在 40,000 train-role 图像及全部像素上，按 RGB channel 计算 float64 mean 和 population
   standard deviation（`ddof=0`）；
3. 逐通道执行 `(z - mean) / std`；
4. 按固定 `H,W,C` C-order 展平成 3,072 维；
5. 每样本除以其 float64 L2 norm，得到 `x_i`。

channel std、L2 norm 或任一输出若非有限或不大于零，生成失败。保存 channel mean/std 的
float64 bytes、统计 digest、transform spec、feature extractor digest 和最终 feature matrix
digest。正式 transform ID 为：

```text
cifar100-train-standardized-l2-pixels-v1
```

不做 resize、crop、flip、ImageNet normalization、PCA、learned embedding 或 label-conditioned
feature transformation。训练 augmentation manifest 不参与噪声特征。

## 6. RNG 与域分离

所有噪声随机流使用锁版本 NumPy `Generator(PCG64DXSM)`，禁止 Python `hash()`、`random`、
全局 NumPy RNG、Torch RNG 或跨域共享可变 RNG state。

对每个 domain，计算：

```text
h = SHA256(
      UTF8("m1-idn-v1") || 0x00 ||
      uint64_big_endian(noise_seed) || 0x00 ||
      UTF8(domain)
    )
entropy = unsigned_big_endian_integer(h[0:16])
rng(domain) = numpy.random.Generator(numpy.random.PCG64DXSM(entropy))
```

`noise_seed` 必须是 `[0, 2^64-1]` 的整数。domains 恰好为：

```text
base-rate/<class-id-decimal>
matrix/<class-id-decimal>
pps-order/<class-id-decimal>
pps-start/<class-id-decimal>
destination/<stable-sample-id>
```

class ID 使用无前导零的十进制。stable sample ID 使用第 3.2 节原字符串。生成器保存 domain
scheme ID 和每个 domain 的前 16-byte entropy hex，但不向 training bundle公开。

## 7. Base propensity 与 exact-rate calibration

### 7.1 截断正态 draw

每类按 stable ID 排序，使用 `base-rate/<class-id>` 流连续产生 400 个 float64 `U in [0,1)`。
通过锁版本 SciPy `ndtr/ndtri` 的 inverse-CDF 生成：

```text
mu = 0.4
sigma = 0.1
a = (0.0 - mu) / sigma
b = (1.0 - mu) / sigma
q_raw = mu + sigma * ndtri(ndtr(a) + U * (ndtr(b) - ndtr(a)))
q_used = clip(q_raw, 1e-12, 1 - 1e-12)
```

`q_raw` 与 `q_used` 都保存到 private audit。非有限值或区间外值失败。`q_raw` 是 base draw，
不是正式 inclusion probability。

### 7.2 Per-class calibration

对 clean class `y` 求 `delta_y`：

```text
pi_i = sigmoid(logit(q_used_i) + delta_y)
sum(pi_i for clean_class y) = 160
```

使用 float64 stable logit/sigmoid 和固定 128 次 bisection，初始 bracket `[-64.0, 64.0]`。
先验证 lower sum `<160` 且 upper sum `>160`。每次取 float64 midpoint；若 sum `<160` 更新
lower，否则更新 upper；128 次后用最终两端 midpoint 计算 `delta_y` 与 `pi_i`。禁止以运行时
容差提前退出。

每类必须满足：所有 `0 < pi_i < 1`、值有限，且 `abs(sum(pi_i)-160) <= 1e-10`。否则失败。
`pi_i` 才是 fixed-size design 的一阶 inclusion probability。

## 8. Exact 16,000 corruption set

每个 clean class 独立执行 fixed-size systematic PPS：

1. `pps-order/<class-id>` 对该类 400 个 stable-ID-sorted positions 调用一次 `permutation(400)`；
2. 按该 permutation 重排 ID 和 `pi_i`；
3. `pps-start/<class-id>` 调用一次 `random()` 得到 float64 `U in [0,1)`；
4. thresholds 为 `U + arange(160, dtype=float64)`；
5. cumulative 为顺序 float64 `cumsum(pi)`，并把最后一项显式设为 `160.0`；
6. 对每个 threshold 使用 `searchsorted(cumulative, threshold, side="right")`；
7. 把 160 个位置映射回 stable IDs。

每类必须得到 160 个不同 ID；重复、越界或计数不符立即失败。100 类 union 必须恰好包含
16,000 个不同 ID，且 class counts 恰好都是 160。不得以 Bernoulli 后调、全局 top-k、重复抽样、
删补样本或结果驱动重抽来修复。

## 9. Instance-dependent wrong-class destination

对每个 clean class `y`，使用 `matrix/<class-id>` 流调用一次：

```text
W_y = rng.standard_normal(size=(3072, 100), dtype=float64)
```

数组为 C-order。对该类每个样本：

```text
tau_destination = 1.0
logit_ij = dot(x_i, W_y[:, j]) / tau_destination
r_i,y = 0
r_i,j = exp(logit_ij - max_{k != y}(logit_ik)) /
        sum_{k != y} exp(logit_ik - max_{l != y}(logit_il)),  j != y
T_i,y = 1 - pi_i
T_i,j = pi_i * r_i,j,  j != y
```

所有矩阵乘法和 softmax 为 float64。`r_i` 和完整 `T_i` 对全部 40,000 样本计算并保存到
private audit。每行必须有限、非负，`r_i,y=0`，且 `abs(sum(T_i)-1) <= 1e-12`。

仅对第 8 节选中的 corrupted ID，使用独立 `destination/<stable-sample-id>` 流调用一次
`choice(100, p=r_i)` 得到 noisy label。未选中样本的 noisy label 等于 clean label。最终必须验证：

- corruption mask 恰好 16,000 个 true；
- 每个 corrupted sample 的 `noisy_label != clean_label`；
- 每个 uncorrupted sample 的 `noisy_label == clean_label`；
- 所有 labels 是 `[0,99]` 的整数；
- 在至少两个 clean classes 内，propensity 或 wrong-class distribution 随 sample feature 变化；
  若所有样本行意外完全相同则失败，而不是把结果称为 instance-dependent。

## 10. 正式 artifact 边界

每个 noise seed 生成两个兄弟 artifact，由控制层私有 registry 映射；training 侧不能从 opaque ID
推导 private 路径：

```text
m1-noise-training-<opaque-id>/
  sample_ids.npy
  noisy_labels.npy
  public_manifest.json
  sha256_manifest.json

m1-noise-audit-<opaque-id>/
  clean_labels.npy
  corruption_mask.npy
  q_base_raw.npy
  q_base_used.npy
  inclusion_probability.npy
  destination_probabilities.npy
  transition_probabilities.npy
  channel_statistics.npy
  private_manifest.json
  sha256_manifest.json
```

`sample_ids.npy` 与所有 label arrays 按 stable sample ID 字典序对齐，dtype/shape 冻结为：

| Array | dtype | shape |
|---|---|---|
| `sample_ids.npy` | `|S30` ASCII bytes | `(40000,)` |
| `noisy_labels.npy` | `|u1` | `(40000,)` |
| `clean_labels.npy` | `|u1` | `(40000,)` |
| `corruption_mask.npy` | `|b1` | `(40000,)` |
| `q_base_raw.npy` | `<f8` | `(40000,)` |
| `q_base_used.npy` | `<f8` | `(40000,)` |
| `inclusion_probability.npy` | `<f8` | `(40000,)` |
| `destination_probabilities.npy` | `<f8` | `(40000,100)` |
| `transition_probabilities.npy` | `<f8` | `(40000,100)` |
| `channel_statistics.npy` | `<f8`；row 0 mean、row 1 std，column 顺序 RGB | `(2,3)` |

`|S30` 的每项必须是第 3.2 节恰好 30-byte 的 ASCII ID，无 NUL padding、截断或替代编码。

### 10.1 Public manifest 白名单

public manifest 只允许：

- schema ID/version、opaque artifact ID、sample count；
- `sample_ids.npy`、`noisy_labels.npy` 的相对文件名、dtype、shape、SHA-256；
- opaque public protocol handle（不得编码或反推出 noise rate、noise seed 或 private ID）、
  train-ID digest；
- source commit、`uv.lock` digest、public payload hash；
- `test_constructed=false`、`test_loaded=false`、`test_access_count=0`。

禁止出现 clean labels、mask、noise rate、noise seed、class corruption counts、`q`、`pi`、`r`、
`T`、private artifact ID/path 或可关联 private registry 的字段。schema 使用 `extra=forbid`。

### 10.2 Private manifest

private manifest 保存全部输入/output hashes、noise seed、class counts、channel statistics、delta、
RNG entropy、每个 clean class 的 `W_y` float64-bytes SHA-256、feature/generator digests、每个
array schema、验证结果和 public bundle SHA-256。
private audit 可用于 synthetic-noise memorization 等离线诊断，但不得被训练 runner、candidate、
hyperparameter ranking 代码或训练日志读取。

两个 `sha256_manifest.json` 覆盖各自目录除自身外的全部普通文件，记录相对路径、byte size 和
SHA-256。manifest 还保存其自身 canonical payload SHA-256。所有路径必须解析后仍位于 artifact
root，禁止 symlink、junction、absolute path 和 `..`。

## 11. Seed 映射与配对

seed 映射冻结如下；clean condition 不生成或接收 noise bundle：

| Stage | Train seeds | Noise seeds | Augmentation seeds |
|---|---|---|---|
| Calibration anchor | `101, 102` | `1101, 1102` | `10101, 10102` |
| Tuning | `201, 202` | `1201, 1202` | `10201, 10202` |
| M1-Dev | `301, 302, 303` | `1301, 1302, 1303` | `10301, 10302, 10303` |
| M1-Confirm | `401–408` | `1401–1408` | `10401–10408` |

表中位置一一对应。每个 `(stage, train_seed)` 的 noisy cell 只能引用对应 noise bundle hash；
所有方法、配置、rung、恢复和 exact retry 均复用该 hash。successive-halving 后进入更高 rung 时
不得重新生成 noise。不同 stage 的 noise seed、artifact ID 和 private registry entry 不复用。

对同一个 `(stage, train_seed)`，clean/noisy conditions 除训练 target 与 noisy-only bundle binding
外，共享相同 initialization、optimizer-step/data-access budget、batch order 和 augmentation manifest。
同一 condition 内所有方法的这些 hashes 也必须相同。私有 orchestrator 保存 noise seed 到 opaque
bundle 的映射；training process、candidate 和普通 run log 都不能读取该映射。

每个正式 run manifest 同时绑定：split hash、training noise bundle hash（noisy only）、batch-order
manifest hash、augmentation manifest hash、train seed、augmentation seed、source commit、lock digest、
method config hash 和 idempotency key。runner 启动前验证全部绑定；运行中 sample ID 顺序和 noisy
labels 只能来自已验证 bundle。

## 12. 一次生成、跨平台边界与重试

大规模 BLAS dot product 的末位可能随硬件/库变化。正式 noise artifacts 因此只在批准的
`trusted_cpu_profile_id` 上、用 `uv run --locked` 和 clean tracked source commit 生成一次。
该环境保存 OS/architecture、Python/NumPy/SciPy/BLAS versions、thread count、CPU model、source
commit、`uv.lock` hash。正式 artifact bytes 与 SHA-256 是训练输入身份。

其他平台可作科学复核，但不得替换正式 artifact。跨平台复核要求 exact 匹配非 BLAS 的 RNG、
bisection、PPS 和小型显式矩阵 fixtures；大矩阵 probabilities 只做预注册 tolerance 检查，不能
用复核输出训练。

失败与重试：

- 正式生成的 verified infrastructure failure 可 exact retry 一次，输入 bytes、source、lock、
  CPU profile、seed、protocol 和 idempotency key 必须相同；失败 artifact 保留；
- retry 成功输出必须与已有完整 partial-file hash（若有）和 deterministic fixtures 一致；若两个
  完整输出字节不同，视为 generator/provenance defect，不得选择其中之一；
- 已成功 sealed bundle 不得重生成或覆盖；复制必须 hash-identical；
- unfavorable corruption pattern 不是失败，禁止重抽 seed；
- non-finite、错误计数、label collision、hash mismatch、split/test/capability violation 是协议失败，
  不是可忽略 warning；
- split、noise、pairing、source 或 test-isolation defect 使受影响 gate 整体无效。修复需新设计
  版本和该 gate 全部未使用 seeds；
- 正式 training run 只有经验证的 infrastructure failure 可按上位 M1 第 12 节 exact retry 一次；
  OOM、数值发散、算法 timeout、非有限 state 和不利结果不得作为 infrastructure retry。

## 13. 生成前与消费前验证

正式 generator 必须在写 seal 前验证：

1. 输入正好 40,000 个唯一 train-role IDs，每类 400；其他 role 数量为零；
2. dataset/split/ID/image/clean-label/protocol/source/lock digests 全部匹配；
3. channel stats、features、`q`、`pi`、`r`、`T` 全部有限；
4. 每类 `sum(pi)` 误差 `<=1e-10`；
5. 每类 corruption count 为 160，总数为 16,000，无重复；
6. corrupted label 全部改变，uncorrupted label 全部保持；
7. transition 非负且 row-sum 误差 `<=1e-12`；
8. public schema 无 private 字段，两个 bundle 不共享父目录能力；
9. 所有 array dtype/shape/endian/order 与 hashes 匹配；
10. `test_constructed=false`、`test_loaded=false`、test access count 为零。

training runner 每次启动必须重新验证 public manifest、file manifest、train-ID/split/source/lock
绑定和 test-isolation flags。runner 不复算或修补 artifact。任一检查失败时该 run 不启动。

## 14. 预注册测试

实现计划至少包含以下 CPU-only tests，且测试不得下载 CIFAR、权重或读取任何 M1 result：

- canonical JSON、array schema、path containment 和 golden digest；
- RNG domain separation、锁版本 deterministic fixtures；
- inverse-CDF truncated-normal、q clipping、fixed-128 bisection；
- small-fixture exact-k systematic PPS、输入顺序不变性、empirical inclusion frequency；
- wrong-class softmax、label inequality、transition row sums；
- 40,000 IDs/100 classes 的全规模结构测试：exact 160/class、exact 16,000、数组对齐与内存边界；
- public/private field isolation，public extra-field fail closed；
- wrong split role、重复/缺失 ID、错误 class count、hash/dtype/shape/endian 失败；
- test loader construction sentinel：任何 `train=False` 构造即失败；
- 同一 paired noise seed 跨 methods/configs/rungs 的 training bundle hash 相同；
- exact retry identity 与不允许 overwrite 的 lifecycle test。

## 15. Primary-source provenance

生成结构以 Xia 等人 NeurIPS 2020 Algorithm 2 为主干：样本污染率来自截断正态，错误类别
概率由样本特征与 clean-class-specific 随机矩阵决定。Primary source：

- Xia et al., *Part-dependent Label Noise: Towards Instance-dependent Label Noise*,
  NeurIPS 2020，论文：<https://proceedings.neurips.cc/paper/2020/file/5607fe8879e4fd269e88387e8cb30b7e-Paper.pdf>。

本 annex 的 train-only standardization、sample L2 normalization、per-class logit calibration、
fixed-size PPS、严格 bundle 隔离和 exact-rate 条件是本项目的预注册扩展。报告必须明确区分原论文
主干与这些扩展。

## 16. 用户审阅项

下列项目在更早的冻结 M1 文档中仍未给出足够的字节级定义，并会改变样本身份或正式 noisy
labels；用户批准本 annex 时需一并确认。本文均给出推荐默认，因此没有结果产生后再选择的空间。

1. **M1 split 的精确 RNG/切片顺序。** 推荐批准第 3.2 节的 PCG64、逐类
   `400/25/25/50` 连续切片和固定 role shuffle 顺序；它与 M0 split 代码同型。替换 RNG 或切片
   顺序会改变所有 M1 样本集合，必须在 calibration 前决定。
2. **官方 CIFAR source bytes 与 ID mapping。** 推荐在实现验收时、任何正式 split/noise 生成前，
   提交官方 Python archive 与 extracted training payload bytes 的实际 SHA-256，以及 50,000 个原始索引到
   第 3.2 节 stable ID 的 mapping digest。若实际 source digest 不同，禁止自动接受。
3. **可信 CPU 正式生成 profile。** 推荐指定一台固定 CPU 主机、单线程或固定 BLAS thread count、
   锁定 `uv.lock`，生成一次并 seal。由于 destination dot product 的末位可能改变 categorical draw，
   profile 必须在正式 noise artifact 前命名和提交，不能生成多个平台版本后择优。
4. **Calibration augmentation seed。** 上位 executable protocol 已冻结 train/noise seeds
   `101–102/1101–1102`，但没有给出 augmentation seed 数值。推荐冻结为 `10101, 10102`，与
   train seeds 按位置一一对应，并在任何 calibration run 前提交；它不改变 noise bundle，但会
   改变 calibration trajectory 和预算。

除这四项外，feature family、40% exact-rate 方式、`q` 分布、per-class balance、destination
temperature、noise seeds、training/audit visibility 和 retry 规则已由上位批准设计或本 annex
唯一化，不应再次作为可调选项。

## 17. 批准门

只有在以下条件全部满足后，才能实现或生成正式 M1 noisy artifacts：

- 用户明确批准第 16 节四项及本文整体；
- annex 与对应 machine-readable noise spec 在同一 clean source commit 提交；
- schema、validator 和 CPU tests 先通过；
- split/source/CPU-profile digests 已冻结，未读取任何 M1 calibration/tuning/gate 结果；
- candidate mechanism 仍遵守 noisy cell 不可见 clean-reference signal 的独立 freeze gate；
- 生成动作获得单独授权。

本 annex 的批准本身不授权实现代码、下载数据、生成 artifact、运行 calibration 或派发 GPU 任务。
