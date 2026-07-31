# 基线兼容性、许可证与计算收费矩阵

> 审计日期：2026-07-31
> 适用协议：M1 的模型、数据门、一次前后向、fixed-step 与 fixed-wall-clock 规则均以
> `2026-07-31-m06-m1-quality-first-experiment-design.md` 为准。
> 边界：本文件冻结基线角色和阻塞条件，不实现方法、不读取 M1 gate、不启动训练。

## 1. 结论先行

1. **M1 simple baseline 采用 AdaPNM，而不是非自适应 PNM。** AdaPNM 与 AdamW 同属
   自适应、解耦 weight decay 的单次反向路径，官方实现可直接包裹 LoRA 参数；代价是两套交替
   一阶矩和一套二阶矩，全部进入显存和时间收费。
2. **M1 direct baseline 采用 ELR。** DSS 的 CVPR 2026 官方仓库在审计 commit 未发现
   LICENSE/COPYING 文件，因而不能把作者代码纳入可复现实验。原 M1 协议已经预先规定
   “DSS 许可证或复现不通过则回退 ELR”，所以这个选择不依赖任何 M1 结果。
3. AdamW、global-norm Clip-AdamW、beta-retuned AdamW、AdaPNM 与 ELR 都满足单模型、
   单阶段、每 batch 一次 forward/backward 的主路径。任何额外状态和操作仍计入 VRAM、
   fixed-step wall-clock 与 fixed-wall-clock 准确率。
4. TAdam、ADOPT、GradientStabilizer 只保留为 M1 后的机制对照候选；CAdam、SPAM 和 DSS
   当前有代码/许可证或协议阻塞，不进入正式矩阵。

## 2. 审计口径

- “官方代码”仅指论文作者或论文主页指向的仓库；没有许可证不能推定允许复制、修改或分发。
- 仓库审计固定到下表 commit；最终运行还必须记录实际 vendor/source commit、Python 包版本、
  lockfile digest、CUDA/cuDNN、编译选项和本地补丁哈希。
- “一次 backward”只表示算法主路径不重复反向；loss 的额外表、选择统计、范数归约和状态张量
  仍然按真实 wall-clock、VRAM 与数据访问收费。
- optimizer 只接收 `requires_grad=True` 的 LoRA query/value 与分类头参数。AMP 下所有基于梯度的
  变换都发生在 `GradScaler.unscale_(optimizer)` 之后、`optimizer.step()` 之前。
- 不允许方法读取 clean reference、真实 noise rate、被污染样本的 clean label 或训练期间的验证样本。

## 3. M1 正式比较矩阵

| 角色 | 方法 | 主来源 / 官方代码 | 审计 commit 与 License | 核心路径和额外收费 | ViT-LoRA 包装位置 | M1 判定 / 阻塞 |
|---|---|---|---|---|---|---|
| anchor | AdamW | [PyTorch AdamW 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) / [PyTorch](https://github.com/pytorch/pytorch) | source HEAD `70147dd482f51f736c227969bf0e5b147c893b22`；BSD-3-Clause；实验用版本以后由 lockfile 冻结 | 一次 backward；每参数 `exp_avg`、`exp_avg_sq`、`step`，可选 AMSGrad；decoupled weight decay | 原生 optimizer，只传可训练 LoRA/头参数 | 进入 M1；其 12 配置调参先执行并提供 clean reference |
| simple | global-norm Clip-AdamW | [PyTorch `clip_grad_norm_` 源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/clip_grad.py) | 同上 | 一次 backward；无持久新增状态；每步全局范数归约和缩放；记录 clip 前/后范数及是否触发 | AMP unscale 后，对全部可训练参数一次全局裁剪，再执行 AdamW | 进入 M1；不得逐层/逐参数组偷偷改变裁剪定义 |
| simple | beta-retuned AdamW | [PyTorch AdamW 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | 同上 | 与 AdamW 完全同路径；只改变预注册 `betas`，无算法性额外状态 | AdamW 构造参数 | 进入 M1；beta 网格与其他方法一样恰好 12 个配置 |
| simple | **AdaPNM** | [PNM 论文](https://proceedings.mlr.press/v139/xie21h.html) / [作者仓库](https://github.com/zeke-xie/Positive-Negative-Momentum) | `9ffd201fd2ae3c8fb5fb9766fcf6bcf2c07617df`；MIT | 一次 backward；交替写入 `exp_avg`/`neg_exp_avg`，另有 `exp_avg_sq`，官方默认可选/启用 AMSGrad 时再有 `max_exp_avg_sq`；所有额外状态完整收费 | 用 AdaPNM optimizer 替换 AdamW，仍只传 LoRA/头参数；采用 decoupled weight decay 路径 | **冻结为 compatible PNM-family choice**；非自适应 PNM 不作为主比较。必须在调参前冻结 AMSGrad 开关和 12 配置网格 |
| direct | **ELR** | [NeurIPS 2020 论文页](https://proceedings.neurips.cc/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html) / [作者仓库](https://github.com/shengliu66/ELR) | `934af53434a336b6db80d05d7649d23216e8ca6d`；MIT | 单模型、单阶段、一次 forward/backward；保存每个训练样本的 EMA target；40,000×100 float32 约 16 MB，另加索引访问与 ELR loss 时间 | 保留 AdamW；loss 接收稳定 sample ID、logits、标签，EMA 表仅含 40k train ID | **冻结为 M1 direct baseline**；`beta`、`lambda` 与公共 LR 等组成恰好 12 配置；clean/noisy 共用一个配置 |

### 3.1 为什么是 AdaPNM

官方实现中的 AdaPNM 保留 Adam 类二阶预条件，同时通过相邻时间段的正负动量组合改变一阶路径；
这比不含二阶矩的 PNM 更接近 M1 的 AdamW 参照系。选择依据是优化器家族与协议兼容性，不是
对候选的预期胜负。实现前必须写最小状态测试：奇偶步分别更新哪一个一阶 buffer、另一个是否
保持、bias correction、AMSGrad 和 weight decay 是否与冻结公式一致。

### 3.2 为什么 DSS 回退为 ELR

[DSS 官方 CVPR 2026 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.html)
指向作者仓库。审计 `Aliinton/DSS@ebbe5dd705584814d1214fe6cc1f1fb40658cf58`
时，仓库根目录与递归文件清单均未发现 LICENSE、COPYING 或等价授权文本。代码虽有单模型
`train.py` 和 CLIP 路径，但还包含默认 30 epoch warm-up、预测轨迹、MDA/CCS 选择器和样本选择；
若在无许可证条件下自行搬运，复现与分发均不可接受。故按 M1 第 10 节预注册顺序回退到 ELR。

如果作者以后增加许可证，也不能在当前 M1 confirmation gate 打开后换回 DSS；只能在未使用的
新数据门和新设计版本中重新审计。

## 4. M1 后机制对照候选

| 方法 | 来源 / 官方代码 | commit / License | 相对 AdamW 的变化 | 当前角色与原因 |
|---|---|---|---|---|
| TAdam | [论文](https://arxiv.org/abs/2003.00179) / [作者仓库](https://github.com/Mahoumaru/TAdam) | `5387e5c02c08c2c9d95a3c8510135535d3043672`；未发现许可证 | 以 Student-t 型鲁棒权重抑制一阶矩异常，二阶矩仍接收原始平方梯度；一次 backward | 适合检验“只鲁棒化 m 是否足够”，但代码许可与现代 PyTorch 维护状态不满足 M1 正式基线 |
| ADOPT / ADOPTW | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/84d286e32bbee8fa3a86ee9c50e00081-Abstract-Conference.html) / [作者仓库](https://github.com/iShohei220/adopt) | `6468572b4b3688a2f056b796b071da540568520e`；Apache-2.0 | 当前梯度由前一步二阶矩归一化，再更新状态；一次 backward，带预注册 clip schedule | 可隔离“当前梯度与同一步 v 的时序解耦”，但不是 label-noise direct baseline；留作 R2 机制对照 |
| GradientStabilizer | [论文](https://arxiv.org/abs/2502.17055) / [作者仓库](https://github.com/TianjinYellow/GradientStabilizer) | `107504bad04693ba2c5efe9ac8b5d39fa77b0e37`；MIT | AdamW 前的梯度范数稳定变换；约两个标量 EMA 状态，一次 backward | 可检验简单上游稳定化是否解释收益；只在 M1 GO 后按 R2 新设计考虑 |
| SPAM | [论文](https://arxiv.org/abs/2501.06842) / [作者仓库](https://github.com/TianjinYellow/SPAM-Optimizer) | `e8185fbe695fdc449953b4a1f96121df5bf0d268`；未发现许可证 | 梯度 spike clipping 加周期性状态 reset；一次 backward，但存在重置周期、warm-up 与额外状态行为 | 机制重叠较宽且无许可证；不进入 M1。若只重写公式会形成新的非官方实现，必须另行审计 |
| CAdam | [论文](https://arxiv.org/abs/2411.19647) | 未找到作者公开代码；License 不适用/实现授权未知 | 噪声感知系数作用于参数更新；标准 m/v 仍先写入 | 适合作为“参数门而非状态写门”的概念消融；无官方实现，不能作为当前正式基线 |
| DSS | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.html) / [作者仓库](https://github.com/Aliinton/DSS) | `ebbe5dd705584814d1214fe6cc1f1fb40658cf58`；未发现许可证 | 预测轨迹、类边际校正、candidate class trend 与样本选择；warm-up 后选择 | 算法强但许可证阻塞；M1 依预注册回退 ELR，不允许凭结果换回 |

## 5. 实现和计费合同

### 5.1 固定公共路径

每个方法必须共用：数据 ID 与顺序、augmentation/noise manifest、模型初始化、LoRA target modules、
classification head、batch size、precision、梯度累积、scheduler、optimizer steps、评价 cadence 和
checkpoint endpoint。方法不得通过不同的数据增强、额外 epoch 或 best checkpoint 获利。

### 5.2 方法专属收费字段

每 run 至少记录：

- `forward_count`、`backward_count`、`optimizer_step_count`、`examples_seen`；
- `train_wall_seconds`、每 step p50/p95、fixed-wall-clock 截止 checkpoint；
- `peak_allocated_vram_bytes` 与 `peak_reserved_vram_bytes`；
- optimizer state 每个 key 的 dtype、shape、元素数和总 bytes；
- per-example 表和 trajectory 缓存 bytes；
- preprocessing/warm-up/额外阶段时间，即使不进入主训练循环；
- trainable parameters、全部模型参数、source commit、patch hash 和配置 hash。

### 5.3 预实现测试

正式调参前，CPU 小张量测试必须证明：

1. AdamW、Clip-AdamW 与 beta-retuned AdamW 在关闭各自差异时逐步等价；
2. clipping 只在 unscale 后执行，非有限范数 fail closed；
3. AdaPNM 的奇偶 buffer、二阶矩、AMSGrad、bias correction 和 decoupled decay 符合冻结实现；
4. ELR 的 target 表只由 train sample ID 读写，未见过的 ID、重复 ID 或 gate ID 立即报错；
5. 所有方法每 batch 恰好一次 forward/backward，test loader 构造计数为零。

## 6. 冻结状态

| 项目 | 状态 |
|---|---|
| AdamW / Clip-AdamW / beta-retuned AdamW | M1 已冻结角色；具体 12 配置网格仍须在 gate 前提交 |
| PNM-family | **AdaPNM 已冻结**；不得在看见 M1 tuning/gate 后切换 PNM |
| direct baseline | **ELR 已冻结**；DSS 因许可证阻塞退出当前 M1 |
| 候选方法 | 本文件不定义；等待候选机制 specification 的既定 freeze gate |
| M1 后机制对照 | 仅列候选，不构成 GPU 授权；必须依 M1 数据建立新设计 |
