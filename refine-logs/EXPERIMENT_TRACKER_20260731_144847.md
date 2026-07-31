# 实验跟踪表

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| S0-001 | S0 | Windows 全链路 smoke | Phase1 config00 / seed99 / 1 epoch | EuroSAT pilot | state chain, artifacts | MUST | DONE | exp=d9a87ed3d096459b；v1→v6 全事件链成功；142s；3.2GB VRAM；7 个产物及 SHA256 完整；仅作基础设施证据 |
| M0-A-L-0 | M0 | label pulse 因果分解 | AdamW / 4 branches / seed0 | CIFAR100 train-val | AUC, Dm, Dv, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-A-L-1 | M0 | label pulse 因果分解 | AdamW / 4 branches / seed1 | CIFAR100 train-val | AUC, Dm, Dv, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-A-L-2 | M0 | label pulse 因果分解 | AdamW / 4 branches / seed2 | CIFAR100 train-val | AUC, Dm, Dv, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-A-I-0 | M0 | input pulse 因果分解 | AdamW / 4 branches / seed0 | CIFAR100 train-val | AUC, Dm, Dv, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-A-I-1 | M0 | input pulse 因果分解 | AdamW / 4 branches / seed1 | CIFAR100 train-val | AUC, Dm, Dv, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-A-I-2 | M0 | input pulse 因果分解 | AdamW / 4 branches / seed2 | CIFAR100 train-val | AUC, Dm, Dv, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-S-L-0 | M0 | label pulse 对照 | SGD no momentum / 4 branches / seed0 | CIFAR100 train-val | AUC, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-S-L-1 | M0 | label pulse 对照 | SGD no momentum / 4 branches / seed1 | CIFAR100 train-val | AUC, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-S-L-2 | M0 | label pulse 对照 | SGD no momentum / 4 branches / seed2 | CIFAR100 train-val | AUC, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-S-I-0 | M0 | input pulse 对照 | SGD no momentum / 4 branches / seed0 | CIFAR100 train-val | AUC, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-S-I-1 | M0 | input pulse 对照 | SGD no momentum / 4 branches / seed1 | CIFAR100 train-val | AUC, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |
| M0-S-I-2 | M0 | input pulse 对照 | SGD no momentum / 4 branches / seed2 | CIFAR100 train-val | AUC, Dtheta, half-life | MUST | BLOCKED | 等待 M0 Runner |

## 当前执行门禁

- 2026-07-31 14:48：本地 M0 固定执行器、数值负对照和类型化客户端已通过。
- Windows 只注册了 `karpathy-autoresearch` 与 `reliablepeft-phase1`；尚无
  `optimizer-state-m0` 固定项目，因此 12 个 M0 包继续保持 `BLOCKED`，禁止误投旧 Runner。
