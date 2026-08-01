# 实验跟踪表

> 只记录执行状态；实验理由、门槛与停止规则以 `EXPERIMENT_PLAN.md` 为准。
>
> 当前门：E0 证据封存。

| ID | 任务 | 状态 | 完成/启动条件 | 备注 |
|---|---|---|---|---|
| E0-01 | 核对本地 M0/M0.5 原始包 | DONE | 本地 zip 可读，summary 与已记录结论一致 | M0 label-flip AdamW state-only 均值 AUC 约 0.6543；M0.5 结构归因不确定 |
| E0-02 | 收回 M0.6 bridge + 20 bundles | TODO | Windows 原始产物、`COMPLETE.json` 和哈希可用 | 当前只有文档中的 formal PASS 记录 |
| E0-03 | 独立重算 M0.6 冻结门 | BLOCKED | E0-02 完成 | 不修改阈值、不补 seed |
| E0-04 | 收回 clean calibration `101/102` | VERIFY | Windows 运行状态和产物可读 | 本地 runner 测试 3/3 通过；正式结果未入库 |
| E0-05 | 审计 clean calibration 与冻结 Pilot 预算 | BLOCKED | E0-04 产物完整 | 只使用 final/epoch 轨迹，不报 best checkpoint |
| P0-01 | 冻结单一 detector 配置和公共 AMP 规则 | BLOCKED | E0 通过 | 不做 detector grid search |
| P0-02 | 完成 Protect-M/Protect-MV 最小实现与 CPU 检查 | BLOCKED | P0-01 冻结 | 关闭保护时与 AdamW 一致 |
| P0-03 | 补齐 `m1_noisy_data.py` 的 Pilot seeds 与最小证据检查 | BLOCKED | E0 通过 | 仅增 `1301–1303`；检查确定性、精确 40%、私有字段隔离、篡改拒绝 |
| P0-04 | CAdam 原论文/官方代码/许可证/兼容性审计 | BLOCKED | E0 通过 | 只审计一个最近机制基线 |
| P0-05 | Windows GPU sentinel | BLOCKED | P0-01 至 P0-04 通过 | 只做短运行、一次前后向和 5% 开销门 |
| P1-01 | 24-run M1-Pilot | BLOCKED | E0 和 P0 全绿 | 4 methods × 2 conditions × 3 seeds，单并发 |
| P1-02 | Pilot artifact audit 与 `NO-GO/INCONCLUSIVE/PILOT-GO` | BLOCKED | P1-01 矩阵完整 | 审阅者直读原始产物 |
| C1-01 | M1-Confirm 设计与 seed 功效计算 | DEFERRED | 只有 `PILOT-GO` | 在原 `EXPERIMENT_PLAN.md` 内增补；不新建 plan/spec |

## 冻结项

- `experiments/m1_protocol/` 通用协议/Schema/后续功效平台：`HOLD`，不继续扩建。
- Windows Coordinator、新 Control API、通用队列和完整 ARIS W1→W3：`OUT_OF_SCOPE`。
- 新数据集、音频、新骨干、full fine-tuning 和论文写作：`DEFERRED_UNTIL_CONFIRM_GO`。
