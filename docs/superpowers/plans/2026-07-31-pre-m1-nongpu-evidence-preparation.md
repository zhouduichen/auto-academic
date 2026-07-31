# Pre-M1 Non-GPU Evidence Preparation Plan

> 日期：2026-07-31  
> 执行位置：Mac 本地  
> 边界：只完成 L0 中不依赖 M0.6/M1 结果的基线、数据、理论和统计准备；不实现候选、不启动训练、不选择 M1 后实验。

**目标：** 完成 `13_baseline_compatibility_matrix.md`、`14_dataset_protocol_matrix.md`、`15_theory_preparation.md` 与 `16_post_m1_statistics_contract.md`，使 M1 结果出来后只需按数据填充决策，而不再临时改变比较、数据门、理论对象或统计规则。

**原则：** 论文和代码事实只引用官方论文页、正式 PDF、作者仓库、数据集主页或正式框架文档；找不到许可证就明确写“未发现”，不能推定可复用。数据集只做候选排序与访问风险审计，不提前下载或打开未来测试门。理论只推标准 AdamW 与候选无关反例。统计只冻结函数和门，不用尚不存在的 M1 方差伪造最终样本量。

## Task 1：基线兼容性与许可审计

- 核查 AdamW、global-norm clipping、beta-retuned AdamW、PNM/AdaPNM、DSS、ELR，以及 L0 新发现的 TAdam、CAdam、ADOPT、SPAM/GradientStabilizer。
- 每项记录：主来源、官方代码、审计 commit/tag、license、核心计算路径、额外模型/前后向/阶段/状态、是否需要干净参考或噪声率、ViT-LoRA/AdamW 包装位置、M1 可比性和阻塞条件。
- 明确区分：M1 已冻结候选、M1 simple baseline、M1 direct baseline、只用于 M1 后机制对照、因成本/协议不兼容而排除。
- 产物：`docs/research-direction/13_baseline_compatibility_matrix.md`。

验收：每个进入正式比较候选的基线都有可追溯来源、许可证结论和确定的计算收费路径；任何未知项都转成显式 blocker。

## Task 2：数据集协议与数据门审计

- 核查当前 M1 的 CIFAR-100 合成噪声门，以及候选真实噪声数据：CIFAR-10N/CIFAR-100N、Clothing1M、WebVision、Food-101N 和可选 FSDnoisy18k。
- 每项记录：官方来源、label provenance、规模、split/test 角色、评价脚本、license/terms、下载/存储、单 RTX 5080 风险、是否有可信 clean evaluation、已知偏差和泄漏风险。
- 只给出 `ready / conditional / reject-for-now` 排序及启动前置条件，不在 M1 前选择后续数据集，不下载数据，不查看未来测试标签。
- 产物：`docs/research-direction/14_dataset_protocol_matrix.md`。

验收：每个数据集都有独立 tuning/development/final gate 方案或明确说明为何当前不能形成合法数据门。

## Task 3：候选无关理论准备

- 固化 AdamW 的 `m/v/parameter` 递推、单次污染脉冲的精确状态差传播、参数影响的分母耦合表达和可测量量。
- 列出 bounded、heavy-tailed、biased、temporally correlated 与 systematic contamination 的假设层级，禁止把一个层级的结论外推到另一个。
- 构造“持续一致的错误梯度与真实目标漂移观察等价”的不可识别反例。
- 建立待候选冻结后填写的 recurrence、代码状态、定理义务、反事实实验映射模板。
- 产物：`docs/research-direction/15_theory_preparation.md`。

验收：所有现阶段公式都只依赖标准 AdamW；候选专属上界、阈值和常数保持空接口，并有清楚的停止/反例条件。

## Task 4：M1 后统计合同

- 冻结 paired experimental unit、fixed-step/fixed-wall-clock endpoint、clean non-inferiority、superiority、Pareto 与安全性决策函数。
- 定义 M1 方差进入未来 seed 数的唯一功效函数、保守方差规则、最小/最大可行样本规则和打开新数据门前的冻结顺序。
- 定义主张层级与 multiplicity：共同主要终点使用 intersection-union；同一主张的次要比较使用 Holm；探索性指标不升级为主要证据。
- 定义缺失、失败、重试、无穷/NaN、异常值、跨门复用和 test-once 规则，以及机器可读证据字段。
- 产物：`docs/research-direction/16_post_m1_statistics_contract.md`。

验收：合同不含从未来结果倒推的方差或 seed 数；给定 M1 成对差和未来最小效应后，可机械地产生样本量与 GO/NO-GO 判定。

## Task 5：交叉审计与提交

- 检查四份文档中的方法角色、数据门、成本口径、非劣界和停止规则一致。
- 用脚本检查必需标题、来源 URL、license 字段、功效公式和禁止项。
- 运行 `git diff --check`；仅提交四份文档及本计划，不提交 `.aris/`、`tmp/` 或实验输出。

验收命令：

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("docs/research-direction")
required = {
    "13_baseline_compatibility_matrix.md": ["License", "M1", "ViT-LoRA", "阻塞"],
    "14_dataset_protocol_matrix.md": ["数据门", "License", "RTX 5080", "test"],
    "15_theory_preparation.md": ["AdamW", "不可识别", "候选冻结后"],
    "16_post_m1_statistics_contract.md": ["paired", "non-inferiority", "Holm", "功效"],
}
for name, needles in required.items():
    text = (root / name).read_text()
    for needle in needles:
        assert needle in text, (name, needle)
print("non-GPU evidence documents valid")
PY
git diff --check
```

