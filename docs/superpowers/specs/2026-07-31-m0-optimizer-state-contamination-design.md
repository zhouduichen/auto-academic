# M0 优化器状态污染实验设计

> 日期：2026-07-31
>
> 状态：设计冻结，等待实现与 Windows 提交

## 目标

M0 只回答一个机制问题：

> 单个低质量批次是否会通过 AdamW 的一阶、二阶状态，在后续恢复为干净训练后继续造成可测量的额外损失？

M0 不比较最终论文方法，不使用测试集，不把 Windows smoke job 当作科学证据。

## 两层执行

### 层 1：Windows 提交链路 smoke

复用已经部署的 `reliablepeft-phase1` Runner：

- config：`config_00`
- seed：`99`
- epochs：`1`
- batch size：`32`
- time budget：`600s`
- max parallel：`1`

成功链：`HTTP 201 → queued → running → succeeded → artifact manifest`。

该任务只验证 Mac → ARW API → Windows Runner → CUDA → artifact 的链路。失败按发生阶段分类，不解释模型结果。

### 层 2：M0 因果分解实验

Windows Runner 增加独立的 `optimizer-state-m0` 固定执行入口。API 只接受类型化实验矩阵，不接受 raw command、路径、环境变量或任意代码执行。

## M0 实验单位

一个实验包由同一 warm-up checkpoint 分叉出的四条轨迹组成：

| 轨迹 | 参数即时更新 | optimizer state 写入 | 估计对象 |
|---|---|---|---|
| Control | 干净梯度 | 干净梯度 | 基准 |
| Parameter-only | 异常梯度 | 干净梯度 | 即时参数扰动 |
| State-only | 干净梯度 | 异常梯度 | optimizer-state 污染 |
| Full | 异常梯度 | 异常梯度 | 真实联合效应 |

构造顺序固定：从同一脉冲前状态分别计算 clean/corrupt 两次候选 step，得到
`(theta_clean, state_clean)` 与 `(theta_corrupt, state_corrupt)`；随后交叉组合
Parameter-only 与 State-only。无动量 SGD 没有持久 state，因此其 State-only
必须与 Control 数值一致，用作实现正确性的负对照。

分叉前保存：

- model state；
- optimizer state；
- CUDA/CPU/NumPy RNG；
- AMP scaler（若启用）；
- 固定 batch IDs 与预生成增强参数。

脉冲后四条轨迹重放完全相同的 128 个干净 batch。

## 设置

- 数据：CIFAR-100，固定 `45k train / 5k validation / 10k untouched test`。
- 模型：`google/vit-base-patch16-224-in21k`，LoRA rank 8，target 为 query/value。
- 训练：分类头与 LoRA 可训练；backbone 其余参数冻结。
- warm-up：固定 500 optimizer steps。
- optimizers：AdamW 与无动量 SGD。
- 脉冲：
  - `label_flip`：对脉冲 batch 做确定性 derangement，100% 标签改变；
  - `input_degradation`：对脉冲 batch 做固定强度高斯噪声与模糊，标签不变。
- norm control：异常梯度整体缩放到干净梯度的 L2 norm，排除“只是梯度更大”的解释。
- seeds：3 个独立 `train_seed`；`split_seed` 与 `pulse_seed` 单独冻结。
- 运行量：`2 optimizers × 2 pulses × 3 seeds = 12 experiment bundles`。

## 指标

主要终点：

- `state_only_clean_loss_excess_auc_128`：State-only 相对 Control 的未来 128 步干净验证损失差面积。

次要终点：

- 一阶矩相对距离 `D_m(h)`；
- 二阶矩相对距离 `D_v(h)`；
- 参数相对距离 `D_theta(h)`；
- loss recovery half-life；
- 与 Control clean gradient 的 cosine；
- wall-clock、峰值显存、数据访问次数。

轨迹在 `h={0,1,2,4,8,16,32,64,128}` 上评测固定 validation probe；
AUC 按这些观测点的梯形积分计算。

距离分母使用 Control 状态 norm 加固定数值稳定项，不使用各轨迹自身最大值归一化。

## 分析

- 以 experiment bundle 为实验单位，所有比较保持 seed、checkpoint、batch 与增强配对。
- 主要比较：AdamW State-only vs Control。
- 机制对照：无动量 SGD State-only vs Control。
- 报告每个 seed 原始轨迹、配对差、均值、95% bootstrap CI 与效应量。
- M0 不做多数据集推广，不使用 test accuracy。

## Go / No-Go

只有同时满足以下条件才进入候选算法开发：

1. AdamW 的 State-only loss excess AUC 在两类脉冲中方向一致；
2. 至少一类脉冲的 95% 配对 CI 排除 0；
3. AdamW 的污染半衰期明显长于无动量 SGD；
4. norm matching 后效应仍存在；
5. 重放审计证明分叉后的 batch IDs、增强与 RNG 一致。

任一关键条件失败，停止 optimizer-state 主线，不通过追加数据集挽救。

## 产物

每个实验包保存：

- `run_config.json`
- `split_ids.json`
- `replay_manifest.json`
- `checkpoint_manifest.json`
- `trajectory_metrics.jsonl`
- `summary.json`
- `stdout.log`
- 文件级 SHA-256 manifest

测试集在 M0 中不加载。

## 安全与失败处理

- smoke job 与 M0 使用不同 project/plan ID。
- smoke 固定复用已成功执行的 source commit
  `847dcced0a55bf9fca29f663b0a9c9a3e9d115a8`。
- API 幂等键防止重复提交。
- 超时、CUDA OOM、数据缺失和 checksum 失败均产生明确失败状态。
- 不使用 SSH 执行实验；SSH 只保留为管理员部署渠道。
- ARIS 的 `experiment-bridge`、`experiment-queue`、`run-experiment` 和 `monitor-experiment` 在当前 Stage A2 为 `skill_not_activated`，因此实际提交仅通过项目已有的类型化 ARW API 完成。
