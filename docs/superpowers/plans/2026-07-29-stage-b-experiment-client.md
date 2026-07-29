# Stage B Experiment Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the complete offline-tested Mac experiment control surface: strict experiment contract/models, secure settings, one HTTP client boundary, verified artifact downloads, and `arw experiments` commands.

**Architecture:** Extend the existing OpenAPI/Pydantic boundary without adding a server. `ArwClient` is the only network-capable module and is exercised through `httpx.MockTransport`; CLI commands construct strict intent models and delegate to it. Artifact bytes remain same-origin and enter the destination only after path, size, and SHA-256 verification.

**Tech Stack:** Python 3.12, existing Pydantic 2, PyYAML, httpx 0.28, Typer, pytest, respx, ruff, mypy, OpenAPI 3.1.

## Global Constraints

- Work only on `feat/stage-b-experiments`; do not modify ARIS profile/vendor files or implement Windows services.
- Preserve existing five read-only endpoints and all 17 baseline tests.
- Control API prefix remains `/api/v1`; `/healthz` alone is unauthenticated.
- Requests never accept `source_url`, `adapter_type`, raw command/argv, host, remote path, environment dictionaries, credentials, or `--insecure`.
- Every response contains `api_version: "1.0"` and non-empty `request_id`; models reject extra fields.
- Every write checks a feature capability and uses one UUID `Idempotency-Key` across retries in the same invocation.
- HTTP never follows redirects; Bearer credentials are sent only to the configured same-origin HTTPS server.
- Default tests use MockTransport and temporary directories; no listener, Windows, Tailscale, SSH, Docker, or GPU is used.
- `make quality` is the final gate.

---

### Task 1: Experiment OpenAPI and strict runtime models

**Files:**
- Modify: `contracts/openapi.yaml`
- Modify: `src/arw/models.py`
- Modify: `tests/test_contract.py`
- Create: `tests/test_experiment_contract.py`

**Interfaces:**
- Produces `CandidatePatch`, `ExperimentMatrix`, `ExperimentSubmitRequest`, `ExperimentState`, `ExperimentSummary`, `ExperimentDetail`, `ExperimentEvent`, `ExperimentCancelRequest`, `ArtifactEntry`, their list/envelope responses, and `ErrorResponse`.
- Adds POST/list/show/events/cancel/artifact-manifest operations plus same-origin artifact bytes.

- [ ] **Step 1: Write failing contract/model tests**

Test exact new paths, Bearer security, required idempotency header on both POSTs, cursor pagination, OpenAPI examples accepted by the matching Pydantic model, patch SHA mismatch rejection, state/result invariants, extra field rejection, and absence of forbidden execution fields from `ExperimentSubmitRequest.model_fields`.

- [ ] **Step 2: Run red tests**

Run: `uv run pytest tests/test_contract.py tests/test_experiment_contract.py -q`
Expected: collection fails because experiment models do not exist.

- [ ] **Step 3: Implement models and contract**

Use these exact model boundaries:

```python
ExperimentState = Literal[
    "submitted", "policy-validating", "waiting-approval", "policy-rejected",
    "queued", "running", "failed", "succeeded", "cancelled",
]

class CandidatePatch(StrictModel):
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patch: str = Field(min_length=1, max_length=1_000_000)

class ExperimentMatrix(StrictModel):
    seeds: list[int] = Field(min_length=1, max_length=32)
    time_budget_seconds: int = Field(ge=30, le=3600)
    max_parallel: int = Field(ge=1, le=8)

class ExperimentSubmitRequest(StrictModel):
    project_id: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    title: str = Field(min_length=1, max_length=200)
    plan_id: str = Field(min_length=1, max_length=100)
    candidate: CandidatePatch
    matrix: ExperimentMatrix

class ExperimentCancelRequest(StrictModel):
    reason: str = Field(min_length=1, max_length=500)
    expected_version: int = Field(ge=1)
```

`CandidatePatch` validates SHA-256 over UTF-8 patch bytes. Matrix validates unique seeds. Terminal experiment detail requires result for `succeeded`, safe error for `failed`, and neither for non-terminal states. All OpenAPI object schemas use `additionalProperties: false`, or `unevaluatedProperties: false` when composed with `allOf`.

Change the old exact path equality test to assert the original five paths are a subset; do not remove any old-path assertion.

- [ ] **Step 4: Run green tests and quality**

Run: `uv run pytest tests/test_contract.py tests/test_experiment_contract.py -q && make quality`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add contracts/openapi.yaml src/arw/models.py tests/test_contract.py tests/test_experiment_contract.py
git commit -m "feat: define experiment control contract"
```

### Task 2: Secure local settings and stable errors

**Files:**
- Create: `src/arw/config.py`
- Create: `src/arw/errors.py`
- Create: `tests/test_config.py`
- Create: `tests/test_errors.py`

**Interfaces:**
- Produces `Settings`, `load_settings(*, environ=None, home=None)`, `ArwError` subclasses, and `map_http_error(response)`.

- [ ] **Step 1: Write failing settings/error tests**

Cover environment > `~/.config/arw/env` > node YAML precedence; env file must be regular, current-user owned, and mode no broader than `0600`; server must be HTTPS; token is `SecretStr`; CA bundle must exist. Cover stable exit codes 2/3/4/5/6/7/8 and safe parsing of an `ErrorResponse` without returning raw bodies.

- [ ] **Step 2: Run red tests**

Run: `uv run pytest tests/test_config.py tests/test_errors.py -q`
Expected: collection fails because modules are absent.

- [ ] **Step 3: Implement settings/errors**

```python
class Settings(StrictModel):
    server: str
    api_token: SecretStr
    ca_bundle: Path | None = None
    connect_timeout: float = Field(default=3.0, gt=0)
    read_timeout: float = Field(default=15.0, gt=0)
    artifact_read_timeout: float = Field(default=60.0, gt=0)

def load_settings(
    *, environ: Mapping[str, str] | None = None, home: Path | None = None
) -> Settings: ...
```

Error classes: `ConfigError(2)`, `AuthError(3)`, `NotFoundError(4)`, `ConflictError(5)`, `NetworkError(6)`, `ServerError(7)`, `ArtifactError(8)`. `ArwError` stores only safe message, code, request ID, and status.

- [ ] **Step 4: Run green tests and quality**

Run: `uv run pytest tests/test_config.py tests/test_errors.py -q && make quality`

- [ ] **Step 5: Commit**

```bash
git add src/arw/config.py src/arw/errors.py tests/test_config.py tests/test_errors.py
git commit -m "feat: load secure control settings"
```

### Task 3: Single HTTP client boundary

**Files:**
- Create: `src/arw/client.py`
- Create: `tests/test_experiment_client.py`

**Interfaces:**
- Produces context-managed `ArwClient(settings, *, transport=None, sleeper=time.sleep)` with `meta`, `submit_experiment`, `list_experiments`, `get_experiment`, `list_events`, `cancel_experiment`, `get_artifact_manifest`, and `stream_artifact`.

- [ ] **Step 1: Write failing MockTransport tests**

Assert exact methods/paths/query/body; Authorization and `X-ARW-Client-Version`; feature absence causes zero writes; one invocation reuses idempotency key on a connection retry; separate calls use different keys; GET retries only transport failures and 502/503/504 twice; redirects are errors; response validation is strict; status/TLS/timeout errors map to stable local classes.

- [ ] **Step 2: Run red tests**

Run: `uv run pytest tests/test_experiment_client.py -q`
Expected: collection fails because `ArwClient` is absent.

- [ ] **Step 3: Implement `ArwClient`**

Create one `httpx.Client` with `follow_redirects=False`, same-origin base URL, TLS verification from system trust or an `ssl.SSLContext` using the CA bundle, separate connect/read timeouts, and headers containing the client version. Every public method returns the Task 1 Pydantic type. `stream_artifact` is a context manager yielding a checked `httpx.Response` and uses the 60-second artifact read timeout.

- [ ] **Step 4: Run green tests and quality**

Run: `uv run pytest tests/test_experiment_client.py -q && make quality`

- [ ] **Step 5: Commit**

```bash
git add src/arw/client.py tests/test_experiment_client.py
git commit -m "feat: add experiment control client"
```

### Task 4: Verified artifact download

**Files:**
- Create: `src/arw/artifacts.py`
- Create: `tests/test_artifacts.py`

**Interfaces:**
- Produces `download_artifact(client: ArwClient, artifact: ArtifactEntry, destination: Path) -> Path`.

- [ ] **Step 1: Write failing filesystem tests**

Cover valid streamed download; reject empty/absolute/nested/`..` names; reject duplicate final names; enforce declared byte size and SHA-256; clean partial temp files on transport/size/hash failure; final file mode is `0600`; no redirect or external URL is accepted.

- [ ] **Step 2: Run red tests**

Run: `uv run pytest tests/test_artifacts.py -q`
Expected: collection fails because downloader is absent.

- [ ] **Step 3: Implement downloader**

Create the destination directory if needed, reject symlink destinations, create the temporary file with `mkstemp` in the destination, stream and hash bytes, then publish without overwrite using same-filesystem `os.link(temp, final)` followed by temp unlink. On every failure remove the temp and raise `ArtifactError`.

- [ ] **Step 4: Run green tests and quality**

Run: `uv run pytest tests/test_artifacts.py -q && make quality`

- [ ] **Step 5: Commit**

```bash
git add src/arw/artifacts.py tests/test_artifacts.py
git commit -m "feat: verify downloaded experiment artifacts"
```

### Task 5: Stable output and experiments CLI

**Files:**
- Create: `src/arw/output.py`
- Modify: `src/arw/cli.py`
- Create: `tests/test_cli_experiments.py`
- Create: `tests/test_security.py`

**Interfaces:**
- Produces `arw experiments submit/list/show/events/cancel/artifacts`; `--json` emits sorted JSON with top-level `schema_version: "1"`.

- [ ] **Step 1: Write failing Typer tests**

Inject a fake `ArwClient` factory. Cover every command success, patch hash construction, seed parsing, cancel reason/version, artifact manifest display/download, JSON stability, stdout/stderr separation, local error exit codes, missing confirmation/arguments sending zero requests, and help containing no token/insecure/host/command/argv/env options.

- [ ] **Step 2: Run red tests**

Run: `uv run pytest tests/test_cli_experiments.py tests/test_security.py -q`
Expected: commands are absent.

- [ ] **Step 3: Implement output and CLI**

Keep the existing `aris` group unchanged. Add an `experiments` Typer group. `submit` reads one local unified diff and accepts project/commit/title/plan/seeds/time/max-parallel. `list/show/events/artifacts` are read-only. `cancel` requires reason and expected version. Commands open one `ArwClient(load_settings())`, close it deterministically, and convert `ArwError.exit_code` to `typer.Exit` after writing diagnostics to stderr.

- [ ] **Step 4: Run green tests and quality**

Run: `uv run pytest tests/test_cli_experiments.py tests/test_security.py -q && make quality`

- [ ] **Step 5: Commit**

```bash
git add src/arw/output.py src/arw/cli.py tests/test_cli_experiments.py tests/test_security.py
git commit -m "feat: expose experiment control CLI"
```

### Task 6: Stage B integration gate

**Files:**
- Modify only if a discovered defect requires it: Stage B files from Tasks 1-5.

**Interfaces:**
- Produces one clean branch with complete local Stage B evidence.

- [ ] **Step 1: Run targeted suites**

```bash
uv run pytest -q tests/test_experiment_contract.py tests/test_config.py tests/test_errors.py \
  tests/test_experiment_client.py tests/test_artifacts.py tests/test_cli_experiments.py \
  tests/test_security.py
```

Expected: all targeted tests pass with no skip/xfailed.

- [ ] **Step 2: Run full quality**

Run: `make quality`
Expected: Ruff format/check, strict mypy, and the complete pytest suite pass.

- [ ] **Step 3: Verify scope and cleanliness**

```bash
git diff --check main...HEAD
git status --short
git diff --name-only main...HEAD
```

Expected: no uncommitted files; no ARIS profile/vendor or Windows server file changed.

## Plan Self-Review

- Contract, models, settings, errors, client, artifacts, output, and all six experiment CLI operations are covered.
- The only network boundary is `src/arw/client.py`; all default tests use MockTransport.
- Existing response envelope, strict model, cursor pagination, dependencies, and quality tooling are reused.
- No server, database, scheduler, runner, ARIS activation, SSH, raw command, or new dependency is introduced.
- All types and method names are consistent across tasks; no floating version or implementation placeholder remains.
