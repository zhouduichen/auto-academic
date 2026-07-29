# ARW Foundation and Read-Only CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the new `auto-academic` Workbench repository and deliver Slice 0–1: a versioned API contract, secure local configuration, read-only HTTP client, `arw doctor`, `arw status`, `arw projects list`, and `arw tasks list` with complete offline tests.

**Architecture:** Keep `karpathy/autoresearch` as a clean sibling upstream clone. Build a Python 3.12 Workbench repository using contract-first Pydantic models, one network boundary in `ArwClient`, and Typer commands tested through `httpx.MockTransport`; no Mac background service is started.

**Tech Stack:** Python 3.12, uv, Typer, httpx, Pydantic, PyYAML, pytest, respx, ruff, mypy, OpenAPI 3.1.

## Global Constraints

- Repository remote is `https://github.com/zhouduichen/auto-academic.git`.
- Preserve `karpathy/autoresearch` at commit `228791fb499afffb54b46200aca536f79142f117`; do not install its dependencies or modify its files.
- Do not start PostgreSQL, Redis, MLflow, Coordinator, Worker, Runner, Docker, CUDA, or any persistent Mac service.
- Do not read `~/.ssh`, Keychain exports, or print API-token values.
- TLS verification is mandatory; no `--insecure` option exists.
- Default tests use MockTransport and do not listen on local ports.
- Each implementation task uses a dedicated branch/worktree and is reviewed before merge.
- The unavailable `superpowers:subagent-driven-development` package is replaced by fresh Codex subagents plus independent reviewer gates.

---

### Task 1: Repository migration and project foundation

**Files:**
- Move without editing: `/Users/huangjiahao/Projects/autoresearch-workbench` → `/Users/huangjiahao/Projects/karpathy-autoresearch`
- Create repository: `/Users/huangjiahao/Projects/autoresearch-workbench`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/pyproject.toml`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/Makefile`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/README.md`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/__init__.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/__main__.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/cli.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/configs/projects/karpathy-autoresearch.yaml`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/tests/test_package.py`
- Copy: design spec to `/Users/huangjiahao/Projects/autoresearch-workbench/docs/superpowers/specs/2026-07-29-autoresearch-workbench-control-plane-design.md`
- Copy: this plan to `/Users/huangjiahao/Projects/autoresearch-workbench/docs/superpowers/plans/2026-07-29-arw-foundation-readonly-cli.md`

**Interfaces:**
- Consumes: clean upstream clone and empty GitHub repository.
- Produces: importable `arw` package version `0.1.0`, locked Python environment, quality command, and pinned upstream project metadata.

- [ ] **Step 1: Verify and move the upstream clone**

Run:

```bash
test ! -e /Users/huangjiahao/Projects/karpathy-autoresearch
test "$(git -C /Users/huangjiahao/Projects/autoresearch-workbench remote get-url origin)" = "https://github.com/karpathy/autoresearch"
test "$(git -C /Users/huangjiahao/Projects/autoresearch-workbench rev-parse HEAD)" = "228791fb499afffb54b46200aca536f79142f117"
test -z "$(git -C /Users/huangjiahao/Projects/autoresearch-workbench status --porcelain=v1 --untracked-files=all)"
mv /Users/huangjiahao/Projects/autoresearch-workbench /Users/huangjiahao/Projects/karpathy-autoresearch
```

Expected: all preconditions exit 0; the clean upstream repository exists only at `karpathy-autoresearch` with unchanged origin and HEAD.

- [ ] **Step 2: Clone the Workbench remote and create `main`**

Run:

```bash
git clone https://github.com/zhouduichen/auto-academic.git /Users/huangjiahao/Projects/autoresearch-workbench
git -C /Users/huangjiahao/Projects/autoresearch-workbench symbolic-ref HEAD refs/heads/main
```

Expected: clone warns that the repository is empty; local branch is `main`.

- [ ] **Step 3: Write the failing package smoke test**

```python
from arw import __version__


def test_package_version() -> None:
    assert __version__ == "0.1.0"
```

Run: `cd /Users/huangjiahao/Projects/autoresearch-workbench && python3 -m pytest tests/test_package.py -q`

Expected: FAIL because the package and project environment do not exist.

- [ ] **Step 4: Create the package foundation**

`pyproject.toml`:

```toml
[project]
name = "auto-academic"
version = "0.1.0"
description = "Mac control client and shared API contract for AutoResearch Workbench"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
  "httpx>=0.28,<1",
  "pydantic>=2.11,<3",
  "pyyaml>=6.0,<7",
  "typer>=0.16,<1",
]

[project.scripts]
arw = "arw.cli:app"

[dependency-groups]
dev = [
  "mypy>=1.17,<2",
  "pytest>=8.4,<9",
  "respx>=0.22,<1",
  "ruff>=0.12,<1",
  "types-pyyaml>=6.0,<7",
]

[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/arw"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "S", "RUF"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "S105", "S106"]

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["arw"]
```

`Makefile`:

```make
.PHONY: format quality test

format:
	uv run ruff format .
	uv run ruff check --fix .

quality:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest -q

test:
	uv run pytest -q
```

`src/arw/__init__.py`:

```python
__version__ = "0.1.0"
```

`src/arw/__main__.py`:

```python
from arw.cli import app


if __name__ == "__main__":
    app()
```

`src/arw/cli.py`:

```python
import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Control AutoResearch Workbench from the Mac."""
```

`README.md`:

````markdown
# auto-academic

Mac control client and shared API contract for AutoResearch Workbench.

The Mac client never runs CUDA training or persistent Workbench services. The pinned
`karpathy/autoresearch` source executes only on the Windows CUDA node.

## Development

```bash
uv sync --dev
make quality
```
````

`configs/projects/karpathy-autoresearch.yaml`:

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

- [ ] **Step 5: Copy the approved design and plan, then lock dependencies**

Run:

```bash
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp /Users/huangjiahao/Documents/Codex/2026-07-29/an/outputs/docs/superpowers/specs/2026-07-29-autoresearch-workbench-control-plane-design.md docs/superpowers/specs/
cp /Users/huangjiahao/Documents/Codex/2026-07-29/an/outputs/docs/superpowers/plans/2026-07-29-arw-foundation-readonly-cli.md docs/superpowers/plans/
uv sync --dev --python 3.12
uv run pytest tests/test_package.py -q
uv run arw --help
```

Expected: dependency lock succeeds; smoke test PASS; the foundation CLI help exits 0.

- [ ] **Step 6: Run quality and commit foundation**

Run:

```bash
make quality
git add pyproject.toml uv.lock Makefile README.md src tests configs docs
git commit -m "chore: initialize autoresearch workbench"
git push -u origin main
```

Expected: quality PASS; initial commit pushed to `origin/main`.

---

### Task 2: Read-only OpenAPI contract and runtime models

**Files:**
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/contracts/openapi.yaml`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/models.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/tests/test_contract.py`

**Interfaces:**
- Consumes: Python package from Task 1.
- Produces: `HealthResponse`, `MetaResponse`, `StatusResponse`, `ProjectListResponse`, and `TaskListResponse` with field names matching OpenAPI examples.

- [ ] **Step 1: Write failing contract-model tests**

```python
from pathlib import Path

import yaml

from arw.models import (
    HealthResponse,
    MetaResponse,
    ProjectListResponse,
    StatusResponse,
    TaskListResponse,
)


CONTRACT = Path(__file__).parents[1] / "contracts" / "openapi.yaml"


def test_contract_defines_read_only_paths() -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert document["openapi"] == "3.1.0"
    assert set(document["paths"]) == {
        "/healthz",
        "/api/v1/meta",
        "/api/v1/status",
        "/api/v1/projects",
        "/api/v1/tasks",
    }


def test_models_accept_contract_examples() -> None:
    HealthResponse.model_validate(
        {"api_version": "1.0", "request_id": "req_1", "status": "ok"}
    )
    MetaResponse.model_validate(
        {
            "api_version": "1.0",
            "request_id": "req_2",
            "server_version": "0.1.0",
            "minimum_client_version": "0.1.0",
            "features": ["projects.read", "tasks.read"],
        }
    )
    StatusResponse.model_validate(
        {
            "api_version": "1.0",
            "request_id": "req_3",
            "data": {
                "node_id": "autoresearch-5080",
                "state": "ready",
                "active_tasks": 1,
                "waiting_approval": 2,
            },
        }
    )
    ProjectListResponse.model_validate(
        {
            "api_version": "1.0",
            "request_id": "req_4",
            "items": [
                {
                    "project_id": "karpathy-autoresearch",
                    "name": "Karpathy AutoResearch",
                    "source_url": "https://github.com/karpathy/autoresearch",
                    "source_commit": "228791fb499afffb54b46200aca536f79142f117",
                    "execution_platform": "windows_cuda",
                }
            ],
            "next_cursor": None,
        }
    )
    TaskListResponse.model_validate(
        {
            "api_version": "1.0",
            "request_id": "req_5",
            "items": [
                {
                    "task_id": "task_1",
                    "project_id": "karpathy-autoresearch",
                    "title": "baseline",
                    "status": "waiting-approval",
                    "version": 3,
                }
            ],
            "next_cursor": None,
        }
    )
```

Run: `uv run pytest tests/test_contract.py -q`

Expected: FAIL because `contracts/openapi.yaml` and `arw.models` do not exist.

- [ ] **Step 2: Implement exact runtime models**

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Envelope(StrictModel):
    api_version: Literal["1.0"]
    request_id: str = Field(min_length=1)


class HealthResponse(Envelope):
    status: Literal["ok", "degraded"]


class MetaResponse(Envelope):
    server_version: str
    minimum_client_version: str
    features: set[str]


class StatusData(StrictModel):
    node_id: str
    state: Literal["ready", "degraded", "stopped"]
    active_tasks: int = Field(ge=0)
    waiting_approval: int = Field(ge=0)


class StatusResponse(Envelope):
    data: StatusData


class ProjectSummary(StrictModel):
    project_id: str
    name: str
    source_url: HttpUrl
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_platform: str


class ProjectListResponse(Envelope):
    items: list[ProjectSummary]
    next_cursor: str | None = None


class TaskSummary(StrictModel):
    task_id: str
    project_id: str
    title: str
    status: str
    version: int = Field(ge=1)


class TaskListResponse(Envelope):
    items: list[TaskSummary]
    next_cursor: str | None = None
```

- [ ] **Step 3: Write the OpenAPI document**

```yaml
openapi: 3.1.0
info:
  title: AutoResearch Workbench Control API
  version: 1.0.0
servers:
  - url: https://autoresearch-5080:8443
paths:
  /healthz:
    get:
      operationId: health
      responses:
        "200":
          description: Minimal health response
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"
              example:
                api_version: "1.0"
                request_id: req_1
                status: ok
  /api/v1/meta:
    get:
      operationId: meta
      security:
        - bearerAuth: []
      responses:
        "200":
          description: Version and feature metadata
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/MetaResponse"
              example:
                api_version: "1.0"
                request_id: req_2
                server_version: 0.1.0
                minimum_client_version: 0.1.0
                features: [projects.read, tasks.read]
  /api/v1/status:
    get:
      operationId: status
      security:
        - bearerAuth: []
      responses:
        "200":
          description: Workbench status
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/StatusResponse"
              example:
                api_version: "1.0"
                request_id: req_3
                data:
                  node_id: autoresearch-5080
                  state: ready
                  active_tasks: 1
                  waiting_approval: 2
  /api/v1/projects:
    get:
      operationId: listProjects
      security:
        - bearerAuth: []
      parameters:
        - $ref: "#/components/parameters/Limit"
        - $ref: "#/components/parameters/Cursor"
      responses:
        "200":
          description: Project page
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ProjectListResponse"
              example:
                api_version: "1.0"
                request_id: req_4
                items:
                  - project_id: karpathy-autoresearch
                    name: Karpathy AutoResearch
                    source_url: https://github.com/karpathy/autoresearch
                    source_commit: 228791fb499afffb54b46200aca536f79142f117
                    execution_platform: windows_cuda
                next_cursor: null
  /api/v1/tasks:
    get:
      operationId: listTasks
      security:
        - bearerAuth: []
      parameters:
        - $ref: "#/components/parameters/Limit"
        - $ref: "#/components/parameters/Cursor"
        - name: status
          in: query
          required: false
          schema:
            type: string
      responses:
        "200":
          description: Task page
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/TaskListResponse"
              example:
                api_version: "1.0"
                request_id: req_5
                items:
                  - task_id: task_1
                    project_id: karpathy-autoresearch
                    title: baseline
                    status: waiting-approval
                    version: 3
                next_cursor: null
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
  parameters:
    Limit:
      name: limit
      in: query
      required: false
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 100
    Cursor:
      name: cursor
      in: query
      required: false
      schema:
        type: [string, "null"]
  schemas:
    Envelope:
      type: object
      required: [api_version, request_id]
      properties:
        api_version:
          type: string
          const: "1.0"
        request_id:
          type: string
          minLength: 1
    HealthResponse:
      allOf:
        - $ref: "#/components/schemas/Envelope"
        - type: object
          required: [status]
          properties:
            status:
              type: string
              enum: [ok, degraded]
    MetaResponse:
      allOf:
        - $ref: "#/components/schemas/Envelope"
        - type: object
          required: [server_version, minimum_client_version, features]
          properties:
            server_version:
              type: string
            minimum_client_version:
              type: string
            features:
              type: array
              uniqueItems: true
              items:
                type: string
    StatusData:
      type: object
      additionalProperties: false
      required: [node_id, state, active_tasks, waiting_approval]
      properties:
        node_id:
          type: string
        state:
          type: string
          enum: [ready, degraded, stopped]
        active_tasks:
          type: integer
          minimum: 0
        waiting_approval:
          type: integer
          minimum: 0
    StatusResponse:
      allOf:
        - $ref: "#/components/schemas/Envelope"
        - type: object
          required: [data]
          properties:
            data:
              $ref: "#/components/schemas/StatusData"
    ProjectSummary:
      type: object
      additionalProperties: false
      required: [project_id, name, source_url, source_commit, execution_platform]
      properties:
        project_id:
          type: string
        name:
          type: string
        source_url:
          type: string
          format: uri
        source_commit:
          type: string
          pattern: "^[0-9a-f]{40}$"
        execution_platform:
          type: string
    ProjectListResponse:
      allOf:
        - $ref: "#/components/schemas/Envelope"
        - type: object
          required: [items, next_cursor]
          properties:
            items:
              type: array
              items:
                $ref: "#/components/schemas/ProjectSummary"
            next_cursor:
              type: [string, "null"]
    TaskSummary:
      type: object
      additionalProperties: false
      required: [task_id, project_id, title, status, version]
      properties:
        task_id:
          type: string
        project_id:
          type: string
        title:
          type: string
        status:
          type: string
        version:
          type: integer
          minimum: 1
    TaskListResponse:
      allOf:
        - $ref: "#/components/schemas/Envelope"
        - type: object
          required: [items, next_cursor]
          properties:
            items:
              type: array
              items:
                $ref: "#/components/schemas/TaskSummary"
            next_cursor:
              type: [string, "null"]
```

Run: `uv run pytest tests/test_contract.py -q`

Expected: PASS.

- [ ] **Step 4: Run quality and commit**

Run:

```bash
make quality
git add contracts/openapi.yaml src/arw/models.py tests/test_contract.py
git commit -m "feat: define read-only API contract"
```

Expected: quality PASS; one focused contract commit.

---

### Task 3: Secure configuration loading

**Files:**
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/config.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/tests/test_config.py`

**Interfaces:**
- Consumes: `~/.config/arw/node.yaml`, `~/.config/arw/env`, and process environment.
- Produces: `Settings(server: str, api_token: SecretStr, ca_bundle: Path | None, connect_timeout: float, read_timeout: float)` and `load_settings(*, environ: Mapping[str, str] | None, home: Path | None)`.

- [ ] **Step 1: Write failing precedence and permission tests**

```python
import os
from pathlib import Path

import pytest

from arw.config import ConfigError, load_settings


def write_home(tmp_path: Path, mode: int = 0o600) -> Path:
    arw_dir = tmp_path / ".config" / "arw"
    arw_dir.mkdir(parents=True)
    (arw_dir / "node.yaml").write_text(
        "server:\n  api_url: https://node.example:8443\n  token_env: ARW_API_TOKEN\n",
        encoding="utf-8",
    )
    env_file = arw_dir / "env"
    env_file.write_text(
        "ARW_API_TOKEN=file-token\nARW_SERVER=https://file.example:8443\n",
        encoding="utf-8",
    )
    env_file.chmod(mode)
    return tmp_path


def test_process_environment_wins(tmp_path: Path) -> None:
    home = write_home(tmp_path)
    settings = load_settings(
        environ={"ARW_API_TOKEN": "process-token", "ARW_SERVER": "https://process.example"},
        home=home,
    )
    assert settings.server == "https://process.example"
    assert settings.api_token.get_secret_value() == "process-token"


def test_private_env_file_is_loaded(tmp_path: Path) -> None:
    home = write_home(tmp_path)
    settings = load_settings(environ={}, home=home)
    assert settings.server == "https://file.example:8443"
    assert settings.api_token.get_secret_value() == "file-token"


def test_group_readable_env_file_is_rejected(tmp_path: Path) -> None:
    home = write_home(tmp_path, mode=0o640)
    with pytest.raises(ConfigError, match="permissions"):
        load_settings(environ={}, home=home)


def test_placeholder_token_is_rejected(tmp_path: Path) -> None:
    home = write_home(tmp_path)
    with pytest.raises(ConfigError, match="ARW_API_TOKEN"):
        load_settings(
            environ={"ARW_API_TOKEN": "REPLACE", "ARW_SERVER": "https://node.example"},
            home=home,
        )
```

Run: `uv run pytest tests/test_config.py -q`

Expected: FAIL because `arw.config` does not exist.

- [ ] **Step 2: Implement secure loading**

```python
from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr


class ConfigError(RuntimeError):
    pass


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    server: str
    api_token: SecretStr
    ca_bundle: Path | None = None
    connect_timeout: float = 3.0
    read_timeout: float = 15.0


def _private_env(path: Path) -> dict[str, str]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ConfigError("ARW env path must be a regular file")
    if info.st_uid != os.getuid():
        raise ConfigError("ARW env file must be owned by the current user")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ConfigError("ARW env file permissions must not exceed 600")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ConfigError("ARW env file contains an invalid assignment")
        values[key.strip()] = value.strip()
    return values


def _https_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ConfigError("ARW_SERVER must be an HTTPS URL")
    return value.rstrip("/")


def load_settings(
    *, environ: Mapping[str, str] | None = None, home: Path | None = None
) -> Settings:
    process = dict(os.environ if environ is None else environ)
    root = Path.home() if home is None else home
    config_dir = root / ".config" / "arw"
    node_path = config_dir / "node.yaml"
    env_path = config_dir / "env"
    file_values = _private_env(env_path) if env_path.exists() else {}
    raw_node: object = (
        yaml.safe_load(node_path.read_text(encoding="utf-8")) if node_path.exists() else {}
    )
    if raw_node is None:
        raw_node = {}
    if not isinstance(raw_node, dict):
        raise ConfigError("ARW node configuration must be a mapping")
    raw_server: object = raw_node.get("server", {})
    if not isinstance(raw_server, dict):
        raise ConfigError("ARW node server configuration must be a mapping")
    node_url = raw_server.get("api_url", "")
    if not isinstance(node_url, str):
        raise ConfigError("ARW node api_url must be a string")
    server = process.get("ARW_SERVER") or file_values.get("ARW_SERVER")
    if not server:
        server = node_url
    token = process.get("ARW_API_TOKEN") or file_values.get("ARW_API_TOKEN", "")
    if not token or token == "REPLACE":
        raise ConfigError("ARW_API_TOKEN is missing or still uses the replacement value")
    ca_value = process.get("ARW_CA_BUNDLE") or file_values.get("ARW_CA_BUNDLE")
    ca_bundle = Path(ca_value).expanduser() if ca_value else None
    if ca_bundle is not None and not ca_bundle.is_file():
        raise ConfigError("ARW_CA_BUNDLE must name an existing file")
    return Settings(
        server=_https_url(server),
        api_token=SecretStr(token),
        ca_bundle=ca_bundle,
    )
```

- [ ] **Step 3: Run tests, quality, and commit**

Run:

```bash
uv run pytest tests/test_config.py -q
make quality
git add src/arw/config.py tests/test_config.py
git commit -m "feat: load secure Mac client configuration"
```

Expected: tests and quality PASS; no secret values appear in output.

---

### Task 4: Read-only HTTP client, retry policy, and pagination

**Files:**
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/errors.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/client.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/tests/test_client.py`

**Interfaces:**
- Consumes: `Settings` and response models.
- Produces: context-managed `ArwClient` with `health()`, `meta()`, `status()`, `list_projects()`, and `list_tasks(status: str | None)`.

- [ ] **Step 1: Write failing client tests**

```python
import httpx
import pytest
from pydantic import SecretStr

from arw.client import ArwClient
from arw.config import Settings
from arw.errors import (
    AuthError,
    ConflictError,
    ConnectionError,
    ContractError,
    NotFoundError,
    ServerError,
)


def settings() -> Settings:
    return Settings(
        server="https://node.example:8443",
        api_token=SecretStr("test-secret"),
    )


def test_health_is_unauthenticated_and_meta_is_authenticated() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/healthz":
            return httpx.Response(
                200,
                json={"api_version": "1.0", "request_id": "h", "status": "ok"},
            )
        return httpx.Response(
            200,
            json={
                "api_version": "1.0",
                "request_id": "m",
                "server_version": "0.1.0",
                "minimum_client_version": "0.1.0",
                "features": ["projects.read"],
            },
        )

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        client.health()
        client.meta()
    assert "Authorization" not in seen[0].headers
    assert seen[1].headers["Authorization"] == "Bearer test-secret"


def test_project_pagination_accumulates_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        item = {
            "project_id": f"project-{cursor or 'first'}",
            "name": "Project",
            "source_url": "https://github.com/karpathy/autoresearch",
            "source_commit": "2" * 40,
            "execution_platform": "windows_cuda",
        }
        return httpx.Response(
            200,
            json={
                "api_version": "1.0",
                "request_id": "p",
                "items": [item],
                "next_cursor": "next" if cursor is None else None,
            },
        )

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        projects = client.list_projects()
    assert [project.project_id for project in projects] == ["project-first", "project-next"]


def test_task_filter_is_forwarded() -> None:
    seen_status: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_status.append(request.url.params.get("status"))
        return httpx.Response(
            200,
            json={"api_version": "1.0", "request_id": "t", "items": [], "next_cursor": None},
        )

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        assert client.list_tasks("waiting-approval") == []
    assert seen_status == ["waiting-approval"]


def test_repeated_cursor_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"api_version": "1.0", "request_id": "p", "items": [], "next_cursor": "loop"},
        )

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ContractError, match="cursor"):
            client.list_projects()


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, AuthError), (403, AuthError), (404, NotFoundError), (409, ConflictError), (500, ServerError)],
)
def test_status_codes_map_to_safe_errors(status: int, error: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="server-secret-body", headers={"X-Request-ID": "req_safe"})

    with ArwClient(settings(), transport=httpx.MockTransport(handler), sleeper=lambda _: None) as client:
        with pytest.raises(error) as caught:
            client.meta()
    assert "server-secret-body" not in str(caught.value)
    assert "test-secret" not in str(caught.value)


def test_redirect_is_not_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://attacker.example/"})

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ContractError, match="redirect"):
            client.meta()


def test_three_transport_failures_map_to_connection_error() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectTimeout("hidden details", request=request)

    with ArwClient(
        settings(), transport=httpx.MockTransport(handler), sleeper=lambda _: None
    ) as client:
        with pytest.raises(ConnectionError, match="connection failed"):
            client.meta()
    assert attempts == 3
```

Run: `uv run pytest tests/test_client.py -q`

Expected: FAIL because the client and errors do not exist.

- [ ] **Step 2: Implement stable error classes**

```python
class ArwError(RuntimeError):
    exit_code = 7


class AuthError(ArwError):
    exit_code = 3


class NotFoundError(ArwError):
    exit_code = 4


class ConflictError(ArwError):
    exit_code = 5


class ConnectionError(ArwError):
    exit_code = 6


class ServerError(ArwError):
    exit_code = 7


class ContractError(ArwError):
    exit_code = 7
```

- [ ] **Step 3: Implement `ArwClient`**

```python
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import httpx
from pydantic import ValidationError

from arw import __version__
from arw.config import Settings
from arw.errors import (
    AuthError,
    ConflictError,
    ConnectionError,
    ContractError,
    NotFoundError,
    ServerError,
)
from arw.models import (
    HealthResponse,
    MetaResponse,
    ProjectListResponse,
    ProjectSummary,
    StatusResponse,
    TaskListResponse,
    TaskSummary,
)

ModelT = TypeVar("ModelT")


class ArwClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = settings.api_token.get_secret_value()
        self._sleeper = sleeper
        verify: bool | str = str(settings.ca_bundle) if settings.ca_bundle else True
        timeout = httpx.Timeout(settings.read_timeout, connect=settings.connect_timeout)
        self._client = httpx.Client(
            base_url=settings.server,
            verify=verify,
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
            headers={"X-ARW-Client-Version": __version__},
        )

    def __enter__(self) -> ArwClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get(
        self,
        path: str,
        *,
        authenticated: bool = True,
        params: dict[str, object] | None = None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._token}"} if authenticated else {}
        for attempt in range(3):
            try:
                response = self._client.get(path, params=params, headers=headers)
            except httpx.TransportError:
                if attempt == 2:
                    raise ConnectionError(f"GET {path}: connection failed") from None
                self._sleeper(0.1 * (2**attempt))
                continue
            if response.status_code in {502, 503, 504} and attempt < 2:
                self._sleeper(0.1 * (2**attempt))
                continue
            return self._validate_status(path, response)
        raise AssertionError("retry loop did not return or raise")

    @staticmethod
    def _validate_status(path: str, response: httpx.Response) -> httpx.Response:
        status = response.status_code
        request_id = response.headers.get("X-Request-ID", "unknown")
        message = f"GET {path}: HTTP {status} (request_id={request_id})"
        if status in {401, 403}:
            raise AuthError(message)
        if status == 404:
            raise NotFoundError(message)
        if status == 409:
            raise ConflictError(message)
        if 300 <= status < 400:
            raise ContractError(f"GET {path}: redirect refused (request_id={request_id})")
        if 400 <= status < 500:
            raise ContractError(message)
        if status >= 500:
            raise ServerError(message)
        return response

    @staticmethod
    def _parse(response: httpx.Response, model: type[ModelT]) -> ModelT:
        try:
            return model.model_validate(response.json())  # type: ignore[attr-defined,no-any-return]
        except (ValueError, ValidationError):
            raise ContractError("server response does not match API contract") from None

    def health(self) -> HealthResponse:
        return self._parse(self._get("/healthz", authenticated=False), HealthResponse)

    def meta(self) -> MetaResponse:
        return self._parse(self._get("/api/v1/meta"), MetaResponse)

    def status(self) -> StatusResponse:
        return self._parse(self._get("/api/v1/status"), StatusResponse)

    def list_projects(self) -> list[ProjectSummary]:
        items: list[ProjectSummary] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(100):
            params: dict[str, object] = {"limit": 100}
            if cursor is not None:
                params["cursor"] = cursor
            page = self._parse(self._get("/api/v1/projects", params=params), ProjectListResponse)
            items.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                return items
            if cursor in seen:
                raise ContractError("server returned a repeated pagination cursor")
            seen.add(cursor)
        raise ContractError("server exceeded the 100-page pagination limit")

    def list_tasks(self, status: str | None = None) -> list[TaskSummary]:
        items: list[TaskSummary] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(100):
            params: dict[str, object] = {"limit": 100}
            if status is not None:
                params["status"] = status
            if cursor is not None:
                params["cursor"] = cursor
            page = self._parse(self._get("/api/v1/tasks", params=params), TaskListResponse)
            items.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                return items
            if cursor in seen:
                raise ContractError("server returned a repeated pagination cursor")
            seen.add(cursor)
        raise ContractError("server exceeded the 100-page pagination limit")
```

- [ ] **Step 4: Run tests, quality, and commit**

Run:

```bash
uv run pytest tests/test_client.py -q
make quality
git add src/arw/errors.py src/arw/client.py tests/test_client.py
git commit -m "feat: add secure read-only API client"
```

Expected: retry, header, status mapping, pagination, and redaction tests PASS.

---

### Task 5: Stable output and read-only Typer commands

**Files:**
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/output.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/cli.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/tests/test_cli_read.py`

**Interfaces:**
- Consumes: `load_settings()`, `ArwClient`, and Pydantic models.
- Produces: `arw status`, `arw projects list`, `arw tasks list`, stable `--json`, and fixed exit-code behavior.

- [ ] **Step 1: Write failing CLI tests**

```python
import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

import arw.cli as cli
from arw.errors import AuthError, ConnectionError
from arw.models import (
    HealthResponse,
    MetaResponse,
    ProjectSummary,
    StatusData,
    StatusResponse,
    TaskSummary,
)

runner = CliRunner()


class FakeClient:
    def health(self) -> HealthResponse:
        return HealthResponse(api_version="1.0", request_id="h", status="ok")

    def meta(self) -> MetaResponse:
        return MetaResponse(
            api_version="1.0",
            request_id="m",
            server_version="0.1.0",
            minimum_client_version="0.1.0",
            features={"projects.read", "tasks.read"},
        )

    def status(self) -> StatusResponse:
        return StatusResponse(
            api_version="1.0",
            request_id="s",
            data=StatusData(
                node_id="autoresearch-5080",
                state="ready",
                active_tasks=1,
                waiting_approval=2,
            ),
        )

    def list_projects(self) -> list[ProjectSummary]:
        return [
            ProjectSummary(
                project_id="karpathy-autoresearch",
                name="Karpathy AutoResearch",
                source_url="https://github.com/karpathy/autoresearch",
                source_commit="2" * 40,
                execution_platform="windows_cuda",
            )
        ]

    def list_tasks(self, status: str | None = None) -> list[TaskSummary]:
        assert status == "waiting-approval"
        return [
            TaskSummary(
                task_id="task_1",
                project_id="karpathy-autoresearch",
                title="baseline",
                status="waiting-approval",
                version=1,
            )
        ]


def install_client(monkeypatch: pytest.MonkeyPatch, client: object) -> None:
    @contextmanager
    def fake_context() -> Iterator[object]:
        yield client

    monkeypatch.setattr(cli, "client_context", fake_context)


def test_status_json(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, FakeClient())
    result = runner.invoke(cli.app, ["--json", "status"])
    assert result.exit_code == 0
    document = json.loads(result.stdout)
    assert document["schema_version"] == "1.0"
    assert document["data"]["status"]["data"]["node_id"] == "autoresearch-5080"


def test_status_table(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, FakeClient())
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "NODE" in result.stdout
    assert "autoresearch-5080" in result.stdout


def test_project_and_task_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, FakeClient())
    projects = runner.invoke(cli.app, ["projects", "list"])
    tasks = runner.invoke(cli.app, ["tasks", "list", "--status", "waiting-approval"])
    assert projects.exit_code == 0
    assert "karpathy-autoresearch" in projects.stdout
    assert tasks.exit_code == 0
    assert "task_1" in tasks.stdout


@pytest.mark.parametrize(
    ("error", "code"),
    [(AuthError("auth failed"), 3), (ConnectionError("connection failed"), 6)],
)
def test_safe_error_exit(monkeypatch: pytest.MonkeyPatch, error: Exception, code: int) -> None:
    class FailingClient(FakeClient):
        def status(self) -> StatusResponse:
            raise error

    install_client(monkeypatch, FailingClient())
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == code
    assert result.stdout == ""
    assert "Error:" in result.stderr
```

Run: `uv run pytest tests/test_cli_read.py -q`

Expected: FAIL because CLI and output modules do not exist.

- [ ] **Step 2: Implement output helpers**

`src/arw/output.py` exports:

```python
import json


def json_document(data: object) -> str:
    payload = {"schema_version": "1.0", "data": data}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max([len(headers[index]), *(len(row[index]) for row in text_rows)])
        for index in range(len(headers))
    ]
    lines = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        for row in text_rows
    )
    return "\n".join(lines)
```

- [ ] **Step 3: Implement the Typer app**

```python
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NoReturn

import typer

from arw.client import ArwClient
from arw.config import ConfigError, load_settings
from arw.errors import ArwError
from arw.output import json_document, table

app = typer.Typer(no_args_is_help=True)
projects_app = typer.Typer(no_args_is_help=True)
tasks_app = typer.Typer(no_args_is_help=True)
app.add_typer(projects_app, name="projects")
app.add_typer(tasks_app, name="tasks")


@dataclass(frozen=True)
class CliState:
    json_output: bool


@app.callback()
def main(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit stable JSON output."),
) -> None:
    ctx.obj = CliState(json_output=json_output)


def _state(ctx: typer.Context) -> CliState:
    state = ctx.find_root().obj
    if not isinstance(state, CliState):
        raise RuntimeError("CLI state was not initialized")
    return state


@contextmanager
def client_context() -> Iterator[ArwClient]:
    with ArwClient(load_settings()) as client:
        yield client


def _fail(error: ConfigError | ArwError) -> NoReturn:
    code = error.exit_code if isinstance(error, ArwError) else 2
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=code)


@app.command("status")
def status_command(ctx: typer.Context) -> None:
    try:
        with client_context() as client:
            health = client.health()
            meta = client.meta()
            status = client.status()
    except (ConfigError, ArwError) as error:
        _fail(error)
    state = _state(ctx)
    if state.json_output:
        typer.echo(
            json_document(
                {
                    "health": health.model_dump(mode="json"),
                    "meta": meta.model_dump(mode="json"),
                    "status": status.model_dump(mode="json"),
                }
            )
        )
        return
    data = status.data
    typer.echo(
        table(
            ("NODE", "STATE", "ACTIVE", "WAITING", "SERVER", "HEALTH"),
            [
                (
                    data.node_id,
                    data.state,
                    data.active_tasks,
                    data.waiting_approval,
                    meta.server_version,
                    health.status,
                )
            ],
        )
    )


@projects_app.command("list")
def projects_list(ctx: typer.Context) -> None:
    try:
        with client_context() as client:
            projects = client.list_projects()
    except (ConfigError, ArwError) as error:
        _fail(error)
    if _state(ctx).json_output:
        typer.echo(json_document([item.model_dump(mode="json") for item in projects]))
        return
    typer.echo(
        table(
            ("PROJECT", "NAME", "PLATFORM", "COMMIT"),
            [
                (item.project_id, item.name, item.execution_platform, item.source_commit)
                for item in projects
            ],
        )
    )


@tasks_app.command("list")
def tasks_list(
    ctx: typer.Context,
    status: str | None = typer.Option(None, "--status"),
) -> None:
    try:
        with client_context() as client:
            tasks = client.list_tasks(status)
    except (ConfigError, ArwError) as error:
        _fail(error)
    if _state(ctx).json_output:
        typer.echo(json_document([item.model_dump(mode="json") for item in tasks]))
        return
    typer.echo(
        table(
            ("TASK", "PROJECT", "STATUS", "VERSION", "TITLE"),
            [
                (item.task_id, item.project_id, item.status, item.version, item.title)
                for item in tasks
            ],
        )
    )
```

- [ ] **Step 4: Run tests, quality, and commit**

Run:

```bash
uv run pytest tests/test_cli_read.py -q
make quality
git add src/arw/output.py src/arw/cli.py tests/test_cli_read.py
git commit -m "feat: add read-only arw commands"
```

Expected: CLI tests and full quality PASS.

---

### Task 6: Local doctor and Slice 0–1 integration gate

**Files:**
- Modify: `/Users/huangjiahao/Projects/autoresearch-workbench/src/arw/cli.py`
- Create: `/Users/huangjiahao/Projects/autoresearch-workbench/tests/test_doctor.py`
- Modify: `/Users/huangjiahao/Projects/autoresearch-workbench/README.md`

**Interfaces:**
- Consumes: config validation and read-only client.
- Produces: `arw doctor`, documented commands, and final local acceptance evidence.

- [ ] **Step 1: Write failing doctor tests**

```python
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

import arw.cli as cli
from arw.errors import ConnectionError
from arw.models import HealthResponse, MetaResponse

runner = CliRunner()


class HealthyClient:
    def health(self) -> HealthResponse:
        return HealthResponse(api_version="1.0", request_id="h", status="ok")

    def meta(self) -> MetaResponse:
        return MetaResponse(
            api_version="1.0",
            request_id="m",
            server_version="0.1.0",
            minimum_client_version="0.1.0",
            features={"projects.read", "tasks.read"},
        )


def install_client(monkeypatch: pytest.MonkeyPatch, client: object) -> None:
    @contextmanager
    def fake_context() -> Iterator[object]:
        yield client

    monkeypatch.setattr(cli, "client_context", fake_context)


def test_doctor_success(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, HealthyClient())
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "config" in result.stdout and "PASS" in result.stdout
    assert "tls" in result.stdout and "api" in result.stdout
    assert "test-secret" not in result.stdout + result.stderr


def test_doctor_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingClient(HealthyClient):
        def health(self) -> HealthResponse:
            raise ConnectionError("connection failed")

    install_client(monkeypatch, FailingClient())
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 6
    assert "config" in result.stdout and "PASS" in result.stdout
    assert "Error: connection failed" in result.stderr
    assert "test-secret" not in result.stdout + result.stderr
```

Run: `uv run pytest tests/test_doctor.py -q`

Expected: FAIL because `doctor` is not registered.

- [ ] **Step 2: Implement doctor**

Add this command to `src/arw/cli.py`:

```python
@app.command("doctor")
def doctor_command() -> None:
    try:
        with client_context() as client:
            client.health()
            client.meta()
    except ConfigError as error:
        _fail(error)
    except ArwError as error:
        typer.echo(table(("CHECK", "RESULT"), [("config", "PASS")]))
        _fail(error)
    typer.echo(
        table(
            ("CHECK", "RESULT"),
            [("config", "PASS"), ("tls", "PASS"), ("api", "PASS")],
        )
    )
```

- [ ] **Step 3: Document user commands and phase limitations**

Add README sections with:

````markdown
## Commands

```bash
arw doctor
arw status
arw projects list
arw tasks list --status waiting-approval
```

Use `--json` before the command for stable machine output.

## Current phase

Slice 0–1 is read-only. Approval, rejection, experiment comparison, report downloads,
emergency stop, and Windows live integration are delivered by later reviewed slices.
````

- [ ] **Step 4: Run complete local acceptance**

Run:

```bash
make quality
uv run arw --help
uv run arw projects --help
uv run arw tasks --help
git status --short
git -C /Users/huangjiahao/Projects/karpathy-autoresearch status --short
git -C /Users/huangjiahao/Projects/karpathy-autoresearch rev-parse HEAD
```

Expected: quality PASS; all commands appear in help; only Task 6 changes are present before commit; upstream status is clean and HEAD remains `228791fb499afffb54b46200aca536f79142f117`.

- [ ] **Step 5: Commit and push the reviewed slice**

Run after independent review:

```bash
git add src/arw/cli.py tests/test_doctor.py README.md
git commit -m "feat: add local control client diagnostics"
git push origin main
```

Expected: `origin/main` contains the complete reviewed Slice 0–1 history.
