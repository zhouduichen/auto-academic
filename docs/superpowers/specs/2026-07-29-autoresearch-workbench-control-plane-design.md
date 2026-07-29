# AutoResearch Workbench Mac 控制面设计规格

**日期：** 2026-07-29  
**状态：** 推荐架构已获用户批准；等待书面规格复核  
**本规格范围：** MAC-P1 控制 CLI。MAC-P2 开发提交流程和 Windows 服务栈分别编写后续规格。

## 1. 目标与预期效果

在 Mac 上提供安全、稳定、可脚本化的 `arw` CLI，使用户能够查看 Windows 节点状态、项目、任务、实验和报告，并在明确确认后执行审批、拒绝和紧急停止。Mac 只通过 Tailscale 上的 TLS API 访问服务端，不连接数据库、Redis、Docker 或 Runner。

`karpathy/autoresearch` 保持为独立、只在 Windows CUDA 环境运行的研究内核。Workbench 通过不可变的仓库 URL 和 commit 引用它，不修改上游，不在 Mac 安装 CUDA 依赖或运行训练。

本设计用需求追踪矩阵和分层验收保证最终效果覆盖原 MAC_PLAN，而不是只生成一个能启动但不能完成控制任务的 CLI。

## 2. 已知环境与门禁拆分

当前 Mac 环境、配置、Tailscale、MagicDNS、目标节点在线状态和 `tailscale ping` 已通过。Windows 节点 `100.88.143.10` 的 TCP 22/8443 仍超时，阻塞范围在 Windows 服务监听、防火墙或 Tailnet ACL/grant。

为避免 Windows 未就绪让本地开发完全停摆，采用两个互不混淆的门禁：

- **本地构建门禁：** API 契约、CLI、MockTransport 单元测试、静态检查和安全审查通过。该门禁不要求 Windows 在线。
- **远端集成门禁：** 真实 Windows API、TLS、鉴权、审计和下载端到端通过。该门禁通过前，不得宣称 MAC-P1 已部署完成。

这允许先完成可靠客户端，但不会把模拟测试冒充双机验收。

## 3. 架构决策

采用“独立 Workbench 仓库 + 固定上游引用 + 契约优先的纵向切片”。

### 3.1 仓库边界

```text
/Users/huangjiahao/自动化科研/karpathy-autoresearch/        # 干净的训练内核上游 clone
/Users/huangjiahao/自动化科研/auto-academic/       # 新建 Workbench 主仓库
```

现有 `~/Projects/autoresearch-workbench` 是干净的 Karpathy 上游 clone。实施时先验证 origin、HEAD 和 clean 状态，再移动为 `~/Projects/karpathy-autoresearch`；随后在空出的原路径初始化新的 Workbench Git 仓库。移动不得改变上游 commit、分支或 origin。

### 3.2 Workbench 目录

```text
auto-academic/
├── pyproject.toml
├── uv.lock
├── Makefile
├── README.md
├── contracts/
│   └── openapi.yaml
├── configs/
│   └── projects/
│       └── karpathy-autoresearch.yaml
├── src/arw/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   ├── client.py
│   ├── errors.py
│   ├── output.py
│   └── artifacts.py
├── tests/
│   ├── test_config.py
│   ├── test_contract.py
│   ├── test_client.py
│   ├── test_cli_read.py
│   ├── test_cli_write.py
│   ├── test_artifacts.py
│   └── test_security.py
└── docs/superpowers/
    ├── specs/
    └── plans/
```

不在第一阶段拆分多个 Python 包，也不生成客户端代码。`models.py` 是 OpenAPI 与运行时模型之间的单一类型边界，避免过早引入代码生成链。

## 4. 分阶段交付

### Slice 0：契约与基础

交付 OpenAPI、Pydantic 模型、配置加载、错误模型、输出规范和进程内 MockTransport。此阶段不执行真实网络写操作。

### Slice 1：只读控制

实现：

```text
arw doctor
arw status
arw projects list
arw tasks list --status waiting-approval
```

`doctor` 只诊断本机配置、文件权限、DNS、TCP 和 TLS，不读取 SSH 私钥或输出 Token。其他命令只调用 API。

### Slice 2：受控状态变更

实现：

```text
arw approve <TASK_ID>
arw reject <TASK_ID> --reason "..."
arw experiments compare <RUN_A> <RUN_B>
```

审批和拒绝先读取任务当前版本，再提交 `expected_version`。任务已变化时返回冲突，不覆盖新状态。

### Slice 3：报告与紧急控制

实现：

```text
arw reports daily --download
arw emergency-stop --confirm
```

报告必须完成路径、大小和 SHA-256 验证后才进入最终目录。紧急停止最后实现，必须同时满足本地确认、服务端权限 scope 和服务端审计。

### Slice 4：真实 Windows 集成

使用真实 TLS、低权限 Token 和 Windows API 运行契约测试及完整命令验收。该阶段完成后才能关闭 MAC-P1 远端集成门禁。

## 5. API 契约

### 5.1 公共约定

- 控制 API 固定前缀 `/api/v1`。
- `GET /healthz` 仅返回最小存活信息，不返回数据库、队列或主机敏感细节。
- 除 `/healthz` 外所有接口必须使用 `Authorization: Bearer <token>`。
- 每个响应包含 `api_version` 和 `request_id`。
- 错误响应使用稳定的机器可读 `code`、安全的 `message` 和可选 `details`；不得包含堆栈、Token 或数据库信息。
- 列表接口使用 `limit` 与不透明 `cursor` 分页，CLI 自动读取下一页，但保留服务端最大页数限制。
- 客户端发送 `X-ARW-Client-Version`；服务端元数据声明兼容客户端范围和能力集合。

### 5.2 能力握手

`GET /api/v1/meta` 返回：

```json
{
  "api_version": "1.0",
  "request_id": "req_...",
  "server_version": "0.1.0",
  "minimum_client_version": "0.1.0",
  "features": ["projects.read", "tasks.read", "tasks.decide", "reports.read"]
}
```

CLI 在状态变更前检查版本和 feature。能力缺失时本地拒绝，不向未知接口试探写请求。

### 5.3 端点映射

| CLI | HTTP |
|---|---|
| `arw status` | `GET /healthz`、`GET /api/v1/meta`、`GET /api/v1/status` |
| `arw projects list` | `GET /api/v1/projects` |
| `arw tasks list` | `GET /api/v1/tasks?status=...` |
| 审批前读取 | `GET /api/v1/tasks/{task_id}` |
| `arw approve` | `POST /api/v1/tasks/{task_id}/approve` |
| `arw reject` | `POST /api/v1/tasks/{task_id}/reject` |
| `arw experiments compare` | `GET /api/v1/experiments/compare?run_a=...&run_b=...` |
| `arw reports daily` | `GET /api/v1/reports/daily` |
| 报告文件 | `GET /api/v1/artifacts/{artifact_id}` |
| `arw emergency-stop` | `POST /api/v1/emergency-stop` |

### 5.4 幂等与并发

- 每次 CLI 调用生成一个 UUID `Idempotency-Key`。
- 同一次调用中的网络重试复用该键；不同人工调用不复用本地历史键。
- 服务端以任务状态机和唯一决策记录保证业务幂等，不能依赖 Mac 保存权威状态。
- 审批/拒绝提交 `expected_version`；服务端原子比较，不匹配则返回 `409 task_version_conflict`。
- 成功响应必须包含 `audit_event_id`、最终状态和最终版本。
- Emergency stop 不自动重试；连接结果不明确时显示 request ID 并要求用户查询状态，禁止盲目再次发送。

## 6. 配置和身份

配置优先级：

1. 当前进程环境变量；
2. `~/.config/arw/env`；
3. `~/.config/arw/node.yaml` 中的非秘密默认值。

读取 `env` 前必须验证：普通文件、当前用户所有、权限不宽于 `600`。不满足时拒绝读取。支持：

```text
ARW_SERVER
ARW_API_TOKEN
ARW_CA_BUNDLE   # 可选；私有 CA 文件路径
```

TLS 默认使用系统信任库；提供私有 CA 时仍验证证书链和主机名。不存在 `--insecure`。Token 不接受命令行传参，不进入日志、异常、JSON 输出、shell history 或仓库。

## 7. HTTP Client

`arw.client` 是唯一允许发送 HTTP 请求的模块，使用 `httpx`：

- 默认 connect timeout 3 秒、read timeout 15 秒；报告流下载使用单独的 60 秒 read timeout。
- GET 请求仅对连接失败和 502/503/504 做最多 2 次指数退避重试。
- 状态变更请求只在明确未收到任何响应时、且复用同一幂等键的前提下最多重试 1 次；emergency stop 例外，永不自动重试。
- 默认不跟随重定向。
- Authorization header 只发送给配置中的同源 HTTPS 主机。
- 401/403、404、409、超时/TLS、5xx 映射为稳定本地错误类型。

## 8. CLI 输出与退出码

默认输出紧凑表格；`--json` 输出 UTF-8 JSON，顶层包含 `schema_version`，字段稳定、键排序且不含秘密。stdout 只放结果，stderr 放诊断，便于脚本组合。

| 退出码 | 含义 |
|---:|---|
| 0 | 成功 |
| 2 | 参数错误或缺少本地确认 |
| 3 | 认证或授权失败 |
| 4 | 资源不存在 |
| 5 | 状态或版本冲突 |
| 6 | DNS、TCP、TLS 或超时失败 |
| 7 | 服务端错误或版本不兼容 |
| 8 | 产物路径、大小或哈希校验失败 |

错误默认不打印原始响应体；调试模式也只显示脱敏后的 request ID、状态码和错误 code。

## 9. 报告下载安全

报告 manifest 只包含 `artifact_id`、相对文件名、字节数和 SHA-256，不允许服务端返回任意外部 URL。CLI 通过固定同源 artifact 端点下载，并执行：

1. 拒绝绝对路径、`..`、空名称、路径分隔符逃逸和重复名称；
2. 在 Exports 目录内创建权限 `600` 的临时文件；
3. 流式计算字节数和 SHA-256；
4. 失败时删除临时文件；
5. 校验通过后原子重命名为最终文件。

同名最终文件默认不覆盖；只有后续明确设计的版本化文件名策略允许并存。

## 10. 上游训练内核适配

Workbench 项目配置保存：

```yaml
project_id: karpathy-autoresearch
adapter_type: autoresearch_v1
source_url: https://github.com/karpathy/autoresearch
source_commit: 228791fb499afffb54b46200aca536f79142f117
execution_platform: windows_cuda
mac_execution_allowed: false
mutable_paths:
  - train.py
protected_paths:
  - prepare.py
```

MAC-P1 只读取并展示这些信息。MAC-P2 才提交远端实验请求；Windows Runner 负责独立 worktree、容器、数据、GPU 和正式结果。Mac 不运行 `uv sync`、`prepare.py` 或 `train.py`。

## 11. 测试与质量门禁

技术栈为 Python 3.12、`uv`、Typer、httpx、Pydantic；开发工具为 pytest、respx、ruff 和 mypy。依赖写入 lockfile，`make quality` 固定执行：

```text
ruff format --check .
ruff check .
mypy src
pytest -q
```

测试覆盖：

- 配置优先级、owner/mode 拒绝逻辑和秘密脱敏；
- OpenAPI 示例与 Pydantic 模型一致；
- 能力握手、版本不兼容和 feature 缺失；
- 401、403、404、409、429、5xx、DNS、TCP、TLS、超时；
- GET 和写请求的差异化重试策略；
- 任务版本冲突、幂等键复用和 audit ID；
- emergency stop 无 `--confirm` 时发送零请求，且永不自动重试；
- JSON schema version、stdout/stderr 分离和稳定退出码；
- artifact 路径穿越、跨域、重定向、部分文件、大小和哈希错误；
- 所有正常质量测试仅使用 MockTransport，不监听本地端口。

真实 Windows 契约测试使用独立标记，不属于默认 `make quality`；其失败阻止远端集成门禁，但不改变已通过的本地构建结果。

## 12. 原规划需求追踪

| 原 MAC_PLAN 能力 | 交付 Slice | 关键验收 |
|---|---:|---|
| `arw status` | 1 | health/meta/status 模型和离线错误测试 |
| `arw projects list` | 1 | 分页、表格和稳定 JSON |
| `arw tasks list` | 1 | 状态过滤和分页 |
| `arw approve` | 2 | expected_version、幂等、audit event |
| `arw reject` | 2 | 必填 reason、版本冲突、audit event |
| `experiments compare` | 2 | 双 run 校验和稳定输出 |
| `reports daily --download` | 3 | 同源、路径、大小、SHA-256、原子落盘 |
| `emergency-stop --confirm` | 3 | 零误触请求、scope、审计、无自动重试 |
| 401/403/超时测试 | 0–3 | MockTransport 完整覆盖 |
| 稳定 JSON/表格 | 1–3 | snapshot/schema 测试 |
| Windows 真机控制 | 4 | TLS、Token、审计和端到端测试 |
| Mac 不保存服务端权威状态 | 全阶段 | 架构审查与无本地状态文件测试 |
| Mac 不运行持久服务/GPU | 全阶段 | 进程/端口审查和依赖检查 |

## 13. 非目标

本规格不实现：

- Windows PostgreSQL、Redis、MLflow、Coordinator、Runner 或 Dashboard；
- Git 分支、push、远端实验提交和报告产物之外的 MAC-P2 工作流；
- Mac light worker、CPU/MPS 训练或正式性能结果；
- 公网暴露、关闭 TLS 校验、共享 SMB 可写目录；
- 对 Karpathy 上游训练代码的修改。

## 14. 完成定义

### 本地构建完成

- Slice 0–3 的对应测试和 `make quality` 全部通过；
- 上游仓库保持 clean，Workbench 是独立仓库；
- 独立 Reviewer 未发现秘密泄露、越权网络、路径穿越或 Mac 持久服务；
- 每条原 MAC-P1 命令都有实现、稳定输出和失败测试。

### MAC-P1 部署完成

- Windows `https://autoresearch-5080:8443/healthz` 在不关闭 TLS 校验时返回成功；
- 能力握手兼容；
- 查询、审批、拒绝、对比、报告下载和 emergency stop 在真实 API 上通过；
- 审批/拒绝/emergency stop 均在 Windows 产生可核对的审计事件；
- 下载报告的 SHA-256 与服务端 manifest 一致；
- Mac 断网后 Windows 状态和实验继续独立运行。

只有两组完成定义都满足，才宣称达到原 MAC-P1 预期规划效果。

