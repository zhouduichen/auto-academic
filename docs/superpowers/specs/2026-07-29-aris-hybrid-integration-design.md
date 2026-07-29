# 完整 ARIS × AutoAcademic × Karpathy AutoResearch 混合架构设计

**日期：** 2026-07-29  
**状态：** 混合架构方向已获用户批准；等待本书面规格复核  
**范围：** 完整 ARIS 研究工作流接入、Mac 控制面、Windows 安全 Runner 与 Karpathy AutoResearch 执行内核的分阶段融合

## 1. 目标

保留 ARIS 的完整研究生命周期能力，包括选题、文献、实验规划、代码审查、实验桥接、自动评审、论文写作、审计和可恢复运行；同时禁止 ARIS 通过原生 SSH、`screen`、`tmux`、任意远端 shell、Vast.ai 或 Modal 绕过 AutoAcademic 控制面。

最终系统只有一条训练执行路径：

```text
ARIS research workflow
        ↓ typed experiment intent
AutoAcademic adapter on Mac
        ↓ HTTPS + bearer scope + idempotency
Windows Coordinator / policy gate
        ↓ immutable approved job
Windows Runner
        ↓ isolated worktree + fixed runner profile
karpathy/autoresearch on CUDA
```

Mac 负责编排、展示和提交意图，不运行训练、不保存服务端权威状态，也不直接控制 GPU 进程。Windows 在 Mac 断线后仍能继续已批准实验，并保存任务状态、审计事件和产物哈希。

## 2. 已确认的基础

- Workbench：`/Users/huangjiahao/自动化科研/auto-academic`，远端为 `https://github.com/zhouduichen/auto-academic.git`。
- Karpathy 上游：`/Users/huangjiahao/自动化科研/karpathy-autoresearch`，固定 commit `228791fb499afffb54b46200aca536f79142f117`，保持 clean。
- 完整 ARIS 评估基线：`https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git`，当前本地固定 commit `53562a7c64cc1d55946cba1fb8a8416137143d14`，MIT License。
- ARIS Codex 包当前包含 81 个 skills，并通过项目本地 `.agents/skills/<name>` 暴露。
- Tailscale、MagicDNS 和 peer ping 已通过；ARIS 正常路径只需 Windows Control API `8443`，该端口仍需由 Windows 服务、防火墙或 Tailnet policy 打通。Windows `22` 不属于融合依赖，仅允许不同人工身份按 break-glass 流程临时访问。
- AutoAcademic 已有 OpenAPI 1.0、严格 Pydantic 模型、项目/任务只读契约和 7 项离线测试。

ARIS 上游更新前必须重新执行 inventory、危险入口和契约差异审查；不得用浮动 `main` 直接替换已验收版本。

## 3. 方案选择

### 3.1 采用：干净完整上游 + 项目本地不可变快照 + 安全覆盖层

完整 ARIS 作为独立、固定 commit、可更新但不直接修改的兄弟 clone。AutoAcademic 保存受控安装器、profile、adapter skills 和契约。安装器从固定 Git tree 构建内容寻址的项目本地快照；普通 skills 来自该 ARIS 快照，可能触达执行面的 skills 使用同一快照中的 AutoAcademic 同名覆盖版本。`.agents/skills` 只链接到不可变快照，不链接到 live clone。

该方案同时满足：

- 完整 ARIS 内容可追溯且可与上游更新；
- 安全改动集中在 AutoAcademic，便于测试和审计；
- 不 fork 或污染 Karpathy 与 ARIS 的干净上游；
- 研究工作流按原 ARIS skill 名称组合，无需重写整条 W1→W3 链；
- 执行 adapter 可独立升级，不影响文献、写作和审计能力。

### 3.2 不采用：直接修改 ARIS clone

直接改上游最省初始文件，但会把安全策略、上游升级和本地补丁混在一个仓库，难以证明使用的是完整、未漂移的 ARIS，也容易在 pull/rebase 时恢复原生 SSH 路径。

### 3.3 不采用：全局安装 ARIS 后靠提示词约束

全局安装会影响所有 Codex 项目；提示词不能形成可测试的执行边界，同名 skill 的解析顺序也可能让原生 SSH/`screen` 入口重新出现，因此不满足 fail-closed 要求。

## 4. 仓库与工作区拓扑

```text
/Users/huangjiahao/自动化科研/
├── aris/                         # 完整、干净、固定 commit 的 ARIS clone
├── karpathy-autoresearch/        # 完整、干净、固定 commit 的训练内核
├── auto-academic/                # 控制面、契约、adapter 与安装器
└── workspaces/
    └── karpathy-autoresearch/    # ARIS 研究产物与项目本地 skills
        ├── .agents/skills/
        ├── .aris/
        │   └── vendor/aris/<profile-digest>/
        ├── idea-stage/
        ├── refine-logs/
        ├── review-stage/
        ├── research-wiki/
        └── paper/
```

`aris/` 和 `karpathy-autoresearch/` 都是只读来源，不保存研究运行产生的文件。研究产物只进入 `workspaces/karpathy-autoresearch/`；AutoAcademic 仓库只保存产品代码、测试、契约和文档。项目本地 vendor snapshot 按 profile digest 生成，安装完成后只读；ARIS clone 后续 checkout 不会改变已安装工作区的行为。

## 5. “完整 ARIS”的定义

“完整”指完整上游仓库中的 81 个 mainline skills、81 个同名 Codex mirror、30 个 shared references、review overlays、tools、模板、测试和许可证都保留并固定版本，不是只复制 `research-pipeline` 或少数提示词。受控 Codex profile 暴露 81 个 Codex skill 名称，其中未获批准的名称解析到 blocked stub；未激活的 mainline、其他宿主 overlay 和辅助资产仍保留在完整上游与 vendor inventory 中。

受控 snapshot 中保留全部 81 个 Codex skill；运行时 profile 对每个上游 skill 名称必须恰好给出一个解析目标，但采用 deny-by-default，而不是假设未列入 blocklist 的 skill 都安全：

- 通过内容、工具权限、依赖调用和外部副作用审查的 skill 链接到 snapshot 内的上游 Codex mirror；
- 已实现等价安全 adapter 的执行型 skill 链接到 snapshot 内的 AutoAcademic 覆盖版本；
- 未审查、新增、未知或尚无安全 adapter 的执行型 skill 链接到无执行工具权限的 `blocked-skill`，返回稳定 `skill_not_activated`，不能直接链接上游实现。
- 当前已知必须覆盖或阻断的最低清单包括：
  - `experiment-bridge`
  - `run-experiment`
  - `experiment-queue`
  - `monitor-experiment`
  - `auto-review-loop`
  - `auto-review-loop-minimax`
  - `result-to-claim`
  - `training-check`
  - `serverless-modal`
  - `vast-gpu`
  - `qzcli`
  - `dse-loop`

前八个覆盖版本保留原工作流职责，但把执行、监控、停止和证据读取改为 `arw`/Control API。`serverless-modal`、`vast-gpu`、`qzcli` 和 `dse-loop` 在本 profile 中默认 fail closed：不能创建云资源、外部 GPU job、任意进程或后台循环。未来若实现经过同等审查的 typed provider adapter，必须通过独立规格和门禁后才能激活。

`research-pipeline`、`idea-discovery`、`paper-writing`、审计链和其他上游 skills 保持完整；它们调用同名执行 skills 时自然进入安全覆盖层。

完整内容不等于完整权限。融合 profile 的宿主工具 broker 只暴露按 skill 审批的文件、文献 connector、内容寻址 helper 和类型化 AutoAcademic adapter；上游 `allowed-tools` 声明不能自行扩大权限。broker 不授予通用远端 shell、SSH key、Windows shell、云 GPU 凭据、Runner 凭据或 Karpathy canonical repo 的写权限。Tailnet/egress policy 和结果 provenance 门禁作为系统级第二道边界：即使提示内容要求旁路，本地或非 API 结果也不能成为 canonical experiment evidence。

## 6. 受控安装与更新

AutoAcademic 提供 `arw aris verify`、`arw aris plan` 和 `arw aris install`：

1. `verify` 检查 ARIS origin、固定 commit、clean 状态、MIT License、81/81 mirror parity、30 个 shared references、内部引用、skill capability inventory 和覆盖/阻断清单。
2. `plan` 只输出变更计划，不写文件，报告新增、复用、冲突、覆盖和移除项。
3. `install` 只接受显式工作区路径；父目录或现有同名真实目录冲突时中止，不覆盖用户文件。
4. 安装器复用 ARIS 官方 inventory、选择、冲突和 manifest 语义，但不创建指向 live clone 的链接；它在临时目录复制固定 Git tree 内容、注入覆盖、生成逐文件 SHA-256 manifest，校验后原子发布只读 snapshot。
5. 原生高风险 skill 从未进入可解析 profile；`.agents/skills` 只在 snapshot 通过全量校验后指向它。
6. `.aris/autoacademic-profile.json` 记录 ARIS commit、Workbench commit、profile digest、全部 skill 名称、激活状态、能力集合、解析目标、逐文件摘要和生成时间；新 skill 默认 `blocked`。
7. 安装结束必须重新遍历所有链接和 snapshot 文件并校验目标/摘要；任何缺失、重复、越界或错误目标使安装失败。每次 ARIS run 启动前由 broker 重算 profile digest，加载后锁定同一内容视图，结束后再次核验。
8. `update` 不自动跟随上游。新 commit 先在新的临时 snapshot 运行 inventory diff、覆盖差异测试、ARIS 自带目标测试和 AutoAcademic quality gate，通过后才原子切换 profile；旧 snapshot 保留到回滚窗口结束。

部署环境中 snapshot 和 `.agents/skills` 由 agent 不具写权限的 broker/独立 OS 身份持有，运行 sandbox 只把它们只读映射给 ARIS；ARIS 仅能写研究 workspace 中声明的 artifact roots。签名 manifest 的私钥不进入 agent 环境。只做 `chmod` 不算通过部署门禁。

安装和卸载只管理 manifest 中登记的链接。卸载不得删除研究产物、用户文件或不属于本 profile 的 skill。

## 7. 实验意图契约

ARIS 不向 API 发送任意 shell command、SSH 主机、远端路径、环境变量字典或凭据。提交对象是结构化的实验意图：

```json
{
  "project_id": "karpathy-autoresearch",
  "source_commit": "228791fb499afffb54b46200aca536f79142f117",
  "title": "ARIS candidate sanity run",
  "plan_id": "aris_20260729_example",
  "candidate": {
    "patch_sha256": "<64 lowercase hex>",
    "patch": "<unified diff limited to train.py>"
  },
  "matrix": {
    "seeds": [42],
    "time_budget_seconds": 300,
    "max_parallel": 1
  }
}
```

`source_url`、`adapter_type`、固定 argv、environment/image、mutable/protected paths 和 expected outputs 全部从服务端 project registry 的 `autoresearch_v1` profile 解析，不接受客户端覆盖。服务端和 Runner 必须再次验证，而不能信任 Mac 已验证：

- `source_commit` 与项目 pin 完全一致；
- patch 哈希正确、格式可解析，修改路径属于 `mutable_paths`；
- `prepare.py`、配置的 protected paths、Git 元数据和 Runner 文件不能修改；
- patch AST/内容策略拒绝 `subprocess`、`os.system`、动态下载、额外依赖安装和未授权网络访问；该检查只是预筛，不作为安全隔离替代品；
- adapter 只允许固定的 runner profile，不接受 raw command；
- seed、时间、并发、输出类型和总预算在服务端上限内；
- 同一个 `Idempotency-Key` 只产生一个 submission；
- 未批准或策略拒绝的 submission 永不进入 Runner 队列。

## 8. API 与状态机

在现有 `/api/v1` 上增加：

| 操作 | HTTP | 权限 |
|---|---|---|
| 提交实验意图 | `POST /api/v1/experiments` | `experiments.submit` |
| 列出实验 | `GET /api/v1/experiments` | `experiments.read` |
| 查看实验 | `GET /api/v1/experiments/{id}` | `experiments.read` |
| 读取事件 | `GET /api/v1/experiments/{id}/events` | `experiments.read` |
| 取消实验 | `POST /api/v1/experiments/{id}/cancel` | `experiments.cancel` |
| 读取产物 manifest | `GET /api/v1/experiments/{id}/artifacts` | `artifacts.read` |

状态机固定为：

```text
submitted → policy-validating → waiting-approval → queued → running
          ↘ policy-rejected                      ↘ failed
                                                ↘ succeeded
                                                ↘ cancelled
```

终态不可回退。取消必须带 reason、expected version 和幂等键；Runner 已结束时返回当前终态，不伪造取消成功。每次状态变化产生不可变 `audit_event_id`、request ID、actor、旧/新状态、版本和时间。

Coordinator 在 policy validation 后生成 canonical job manifest，规范化并固定 project/source、candidate patch、matrix、runner profile、environment/image、预算、expected outputs 和 retry policy，计算 `job_manifest_sha256`。人类或 bounded policy 的审批必须绑定 exact manifest digest 与 task version；任何 patch、matrix、profile、预算或环境变化都产生新版本并重新审批。进入队列和 Runner 启动前分别复验 digest、版本与审批绑定，Runner report 也回写同一 digest。

Tailnet policy 只允许 ARIS/AutoAcademic 自动化身份访问 Windows Control API 的 `8443`，拒绝 `22`、RDP、SMB 和 Runner 内部端口。人工 break-glass 运维必须使用不同身份、临时授权和独立审计，凭据不暴露给 ARIS。

## 9. ARIS adapter 行为

### 9.1 `experiment-bridge`

保留解析计划、生成/检查候选代码、sanity-first、结果汇总、tracker 和 handoff；部署阶段改为生成结构化 submission。融合 profile 固定 `AUTO_PROCEED=false`、`AUTO_DEPLOY=false`、`CODE_REVIEW=true`，不能由研究 prompt 关闭。首次 GPU 预算承诺、候选 patch 和 job matrix 必须经明确人类批准；后续只有服务端签名的 bounded policy 才能自动批准同一计划范围内的重试。

### 9.2 `run-experiment` 与 `experiment-queue`

单任务和批任务分别构造同一种 experiment intent。队列调度、GPU 分配、OOM retry、wave dependency 和 crash recovery 归 Windows Runner 所有，Mac skill 不轮询 `screen`，也不在本机启动后台进程。

### 9.3 `monitor-experiment`

只调用 read API，使用 cursor/event version 增量读取状态；日志和结果通过同源 artifact 端点下载并校验大小与 SHA-256。连接失败显示最后一次已知的非权威快照，并明确标记 stale，不猜测实验失败。

### 9.4 `auto-review-loop`、`result-to-claim` 与 `training-check`

只使用已验证的服务端产物和审计记录形成结论。需要追加实验或停止运行时再次走 submit/cancel API。没有产物、哈希失败或审计状态不绿时，claim 保持 pending/blocked，不能由模型自行改成 supported。

执行完成是 Type-A 事实，不等于科学结论被接受。claim supported、实验完整和 submission-ready 属于 Type-B 判断，必须由不同模型族直接读取一手 artifacts 并产生可追踪 verdict；同模型家族的 subagent review 只能标记 provisional。

## 10. Windows Runner 边界

Runner 按每个 submission 创建独立 worktree，checkout 固定 source commit，验证 patch 后只修改 `train.py`。执行命令由 `autoresearch_v1` profile 固定，不从请求拼接 shell。运行环境、数据缓存和 CUDA 依赖由 Windows 节点预置并版本化；训练任务默认无外网 egress。

候选 `train.py` 被视为不可信代码。Runner 使用无登录权限、无主机凭据、无 secret 的专用低权限身份和 OS 强隔离边界；canonical source、数据/评估、Runner binary 和环境只读，只有 attempt workspace/artifact staging 可写。每个 attempt 进入独立 Windows Job Object/容器或等价进程边界，强制完整进程树跟踪、可执行文件策略、网络 deny、CPU/RAM/GPU/time/disk 上限和终止后零孤儿校验。不得挂载用户 home、SSH 配置、Git push 凭据或其他任务 workspace。Python 动态行为不能只靠 AST denylist 防御。

不复用 ARIS legacy queue scheduler。Windows Coordinator 必须实现 per-run namespace、单活 scheduler lease、原子 GPU reservation、typed failure reason、有界 retry、终态收敛和 artifact provenance，避免原实现的跨队列 screen 冲突、`failed_other` 不收敛、glob completion 误判和 `shell=True` 风险。

每个运行至少记录：

- submission、job 和 worktree ID；
- source commit、patch SHA-256、环境镜像/profile 版本；
- GPU 型号、seed、时间预算、开始/结束时间和退出码；
- stdout/stderr 或结构化日志的产物哈希；
- `run.json` 指标与生成它的 job ID；
- 策略、审批、取消和紧急停止审计事件。

Runner 不 push Karpathy 上游，不修改干净 clone，不接受 Mac 共享目录作为可写执行目录。

## 11. 故障处理与 fail-closed 规则

- ARIS pin、inventory 或覆盖目标不匹配：拒绝安装/更新。
- API capability 缺失或版本不兼容：本地拒绝提交。
- TLS、鉴权或网络失败：不回退 SSH，不启动本地训练。
- 请求结果不明确：用原幂等键查询，不生成第二个 submission。
- patch、路径或预算不合法：服务端 `policy-rejected`，不进入队列。
- Windows 重启：从持久状态恢复 queued/running 审计，未知进程标记 reconciliation-required，不直接重跑。
- 产物哈希不一致：隔离下载并标记 integrity-failed，不交给 ARIS 下游声明链。
- Mac 断线：Windows 继续已批准任务；Mac 恢复后从事件版本续读。
- emergency stop 仍按原 MAC-P1 规则单独实现，不由 ARIS 自动触发。

## 12. 分阶段实施

### 阶段 A：完整 ARIS 基座与安全 profile

将完整 ARIS clone 固定到正式兄弟目录；实现不可变 snapshot、profile manifest、全量 capability inventory、deny-by-default verifier、dry-run/install/uninstall、已批准的安全 adapter 和 blocked stub。所有测试使用临时目录，不发送网络请求，不安装到全局 Codex。

**完成门禁：** 81 个上游 skill 名称全部有且只有一个解析目标；未知/新增/执行型 skill 未经批准一律 blocked；篡改 snapshot、manifest 或 symlink 时 runtime preflight 失败；live clone checkout 不改变已安装 profile；ARIS 与 Karpathy clone 均 clean；`make quality` 通过。

### 阶段 B：实验契约、客户端与 CLI adapter

扩展 OpenAPI/Pydantic，交付 `arw experiments submit/list/show/events/cancel` 和 artifact manifest 读取。用 MockTransport 覆盖鉴权、幂等、版本冲突、能力缺失、超时和秘密脱敏。

**完成门禁：** raw command/host/path/env 字段无法通过模型；所有写请求有 scope、幂等和审计 ID；网络失败不触发任何旁路；默认测试零监听端口。

### 阶段 C：Windows Coordinator 与 `autoresearch_v1` Runner

实现服务端状态机、策略验证、审批、持久队列、独立 worktree、固定命令执行、产物 manifest 和恢复流程。先用无 CUDA 的 fake executor 完成契约和故障测试，再在 Windows CUDA 节点做真机验收。

**完成门禁：** protected path、错误 commit、超预算和重复请求均被拒绝；审批后变更 patch、matrix、profile、environment 或预算必须生成新 version 并重新审批，旧 digest 在入队和启动门禁均失败；恶意 `train.py` 无法读取 home/凭据/其他 workspace、写入只读挂载、联网、启动未允许可执行文件、遗留子进程或突破 CPU/RAM/GPU/time/disk 限额；重启可恢复；Runner 不接受任意 shell；Karpathy 上游保持 clean。

### 阶段 D：ARIS 实验纵向切片

让已批准的适配型覆盖 skill 使用阶段 B 客户端；所有外部算力、任意进程和后台循环 backend 保持 fail closed。运行最小的 research-pipeline 路径：计划 → 候选 patch → 人工批准 → Windows sanity → 结果/哈希 → ARIS claim/review handoff。

**完成门禁：** 抓取的调用证据中没有 SSH、`screen`、`tmux`、Modal/Vast、Qz CLI、后台进程或 Mac 训练；每个结果能追溯到 submission/job/patch/source commit；结果完整性 verifier 与不同模型族 claim review 通过；独立 Reviewer 给出 PASS。

### 阶段 E：完整 W1→W3 与 submission assurance

在实验纵向切片通过后，运行 Idea Discovery → Experiment Bridge → Auto Review Loop → Paper Writing。NARRATIVE_REPORT 中每个数字必须链接 canonical run/artifact；citation、claim、experiment integrity、paper claim 和 kill-argument 等适用审计不得 silent skip。

**完成门禁：** W1、W1.5、W2、W3 的固定 artifacts 全部生成且通过各自 acceptance；执行事实与 Type-B 科学判断分离；paper PDF 编译成功、强制 audits 为允许终态，并保留人类最终提交签字。

### 阶段 F：恢复、断网与紧急控制

演练 Mac 断网/恢复、Coordinator/Runner 重启、幂等重放、done-but-unaccepted resume、预算耗尽、取消、artifact 部分上传和 emergency stop。emergency stop 仍只允许人类显式确认，不由 ARIS 自动触发。

**完成门禁：** Mac 断线后 Windows 继续已批准任务并可续读；重启无重复 attempt、无孤儿 GPU PID；resume 不跳过未接受阶段；emergency stop 冻结新调度并终止完整进程树；审计与上游完整性保持不变。

## 13. 测试策略

- 单元：profile inventory、symlink 冲突、manifest、模型严格性、状态机和策略函数。
- 契约：OpenAPI examples 与 Pydantic 双向一致；客户端/服务端版本与 feature 握手。
- 属性/安全：patch 路径穿越、绝对路径、Git 元数据、重复字段、额外字段、预算边界、恶意文件名。
- 集成：MockTransport、fake Runner、进程重启、重复 idempotency key、部分产物和哈希损坏。
- 审批不可变性：审批后逐项篡改 patch、matrix、profile、environment、预算和 manifest version；旧 digest 必须在 queue/Runner 两处 fail closed。
- Runner 逃逸：恶意候选尝试读取 home/secret/相邻 workspace、写 canonical/只读 mount、网络连接、动态导入与启动子进程、制造 orphan、耗尽 CPU/RAM/GPU/time/disk；每项都必须被 OS 边界阻止并产生审计事件。
- 上游兼容：ARIS inventory diff、capability classification、覆盖/阻断 skills 与上游职责清单差异、ARIS 目标测试；注入新增执行型 skill 时必须默认 blocked。
- 真机：Tailscale TLS、低权限 Token、Windows CUDA sanity、断网恢复和审计核对。

本地测试不得连接 Windows、监听端口、访问 SSH key、启动 Docker/GPU 或打印 Token。真机测试使用独立标记并只在阶段 C 本地门禁通过后运行。

## 14. 非目标

- 不把 ARIS、Karpathy 或研究产物合并进同一个 Git 历史；
- 不修改或 push 两个干净上游 clone；
- 不支持 Mac CPU/MPS/CUDA 正式训练；
- 不在本 profile 支持 Vast.ai、Modal 或任意 SSH backend；
- 不在阶段 A/B 启动 Windows 服务或宣称远端部署完成；
- 不用共享可写目录替代 API、审计和 artifact hash；
- 不允许 ARIS 自主触发 emergency stop。

## 15. 总完成定义

只有以下条件全部满足，才宣称“完整 ARIS 混合融合完成”：

1. 完整 ARIS 固定版本和全部 skill inventory 可验证；
2. 所有未审查、未知和高风险 skills 都被阻断或通过受控覆盖层，旁路测试为零；
3. ARIS 能完成 W1→W1.5→W2→W3 的 artifact handoff；
4. 所有实验只能通过结构化 API、服务端策略和 Windows Runner 执行；
5. Karpathy 上游保持固定、clean、不可被 Runner push；
6. 任务、审批、运行、取消、结果和声明拥有端到端 provenance；
7. 完整 W1→W3 artifacts、Type-B verdicts、submission audits 和人类最终签字可追溯；
8. Mac 断线不终止 Windows 已批准实验，恢复后能续读，emergency stop 与重启演练无孤儿；
9. 本地质量门禁、fake Runner 集成、Windows CUDA 真机和独立安全审查全部 PASS。
