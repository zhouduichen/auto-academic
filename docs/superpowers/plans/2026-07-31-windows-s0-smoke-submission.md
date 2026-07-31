# Windows S0 Smoke Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore typed client compatibility with the deployed Windows Phase1 API and submit one audited one-epoch smoke job.

**Architecture:** Extend the shared `ExperimentMatrix` schema with the three already-deployed Phase1 fields, expose them through the existing `arw experiments submit` command, then submit and monitor one bounded job through HTTPS. This plan does not implement M0 or use SSH for experiment execution.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, httpx, pytest, OpenAPI YAML, ARW Windows API.

## Global Constraints

- Project: `reliablepeft-phase1`.
- Source commit: `847dcced0a55bf9fca29f663b0a9c9a3e9d115a8`.
- Smoke matrix: config 0, seed 99, 1 epoch, batch size 32, 600 seconds, max parallel 1.
- Never print API tokens or use `verify=False`.
- Submit only through the typed HTTPS API; no SSH experiment execution.
- Smoke results are infrastructure evidence, not research evidence.
- Preserve unrelated untracked `.agents/`, `.aris/`, `.codex/`, Phase1 outputs, and `tmp/`.

---

### Task 1: Align ExperimentMatrix with the deployed Phase1 contract

**Files:**
- Modify: `src/arw/models.py`
- Modify: `contracts/openapi.yaml`
- Modify: `tests/test_experiment_contract.py`

**Interfaces:**
- Consumes: existing `ExperimentMatrix` and `ExperimentSubmitRequest`.
- Produces: optional `config_id: int | None`, `epochs: int`, and `batch_size: int` fields accepted in submit and detail responses.

- [ ] **Step 1: Write the failing contract test**

Add:

```python
def test_phase1_matrix_fields_match_deployed_server() -> None:
    patch = "x"
    request = ExperimentSubmitRequest.model_validate(
        {
            "project_id": "reliablepeft-phase1",
            "source_commit": "a" * 40,
            "title": "smoke",
            "plan_id": "s0",
            "candidate": {
                "patch_sha256": sha256(patch.encode()).hexdigest(),
                "patch": patch,
            },
            "matrix": {
                "seeds": [99],
                "time_budget_seconds": 600,
                "max_parallel": 1,
                "config_id": 0,
                "epochs": 1,
                "batch_size": 32,
            },
        }
    )
    assert request.matrix.config_id == 0
    assert request.matrix.epochs == 1
    assert request.matrix.batch_size == 32
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
uv run pytest tests/test_experiment_contract.py::test_phase1_matrix_fields_match_deployed_server -q
```

Expected: FAIL because `ExperimentMatrix` rejects the deployed fields.

- [ ] **Step 3: Implement the runtime and OpenAPI fields**

Add to `ExperimentMatrix`:

```python
config_id: int | None = Field(default=None, ge=0, le=63)
epochs: int = Field(default=10, ge=1, le=100)
batch_size: int = Field(default=32, ge=1, le=256)
```

Add matching optional OpenAPI properties:

```yaml
config_id: {type: integer, minimum: 0, maximum: 63, nullable: true}
epochs: {type: integer, minimum: 1, maximum: 100, default: 10}
batch_size: {type: integer, minimum: 1, maximum: 256, default: 32}
```

- [ ] **Step 4: Run contract tests**

Run:

```bash
uv run pytest tests/test_experiment_contract.py tests/test_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arw/models.py contracts/openapi.yaml tests/test_experiment_contract.py
git commit -m "fix: align experiment matrix with Windows runner"
```

---

### Task 2: Expose Phase1 fields in the typed CLI

**Files:**
- Modify: `src/arw/cli.py`
- Modify: `tests/test_cli_experiments.py`

**Interfaces:**
- Consumes: Task 1 `ExperimentMatrix` fields.
- Produces: `arw experiments submit --config-id --epochs --batch-size`.

- [ ] **Step 1: Extend the failing CLI test**

Add these arguments to `test_submit_builds_hash_and_matrix`:

```python
"--config-id", "0",
"--epochs", "1",
"--batch-size", "32",
```

Add assertions:

```python
assert request.matrix.config_id == 0
assert request.matrix.epochs == 1
assert request.matrix.batch_size == 32
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
uv run pytest tests/test_cli_experiments.py::test_submit_builds_hash_and_matrix -q
```

Expected: FAIL because the options do not exist.

- [ ] **Step 3: Implement the CLI options**

Add bounded options to `experiments_submit`:

```python
config_id: Annotated[int | None, typer.Option("--config-id", min=0, max=63)] = None,
epochs: Annotated[int, typer.Option("--epochs", min=1, max=100)] = 10,
batch_size: Annotated[int, typer.Option("--batch-size", min=1, max=256)] = 32,
```

Pass them into `ExperimentMatrix`.

- [ ] **Step 4: Run CLI and full quality tests**

Run:

```bash
uv run pytest tests/test_cli_experiments.py tests/test_experiment_client.py -q
make quality
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arw/cli.py tests/test_cli_experiments.py
git commit -m "feat: submit typed Phase1 smoke parameters"
```

---

### Task 3: Submit and verify S0-001

**Files:**
- Create: `experiments/smoke/phase1_s0_candidate.patch`
- Modify after completion: `refine-logs/EXPERIMENT_TRACKER.md`
- Create after completion: `refine-logs/EXPERIMENT_TRACKER_<timestamp>.md`
- Modify after completion: `MANIFEST.md`

**Interfaces:**
- Consumes: typed CLI from Task 2 and configured `~/.config/arw`.
- Produces: one Windows experiment ID, terminal state, event chain, and artifact manifest.

- [ ] **Step 1: Create the audited placeholder patch**

Create exactly:

```diff
--- a/train.py
+++ b/train.py
@@ -1,1 +1,1 @@
-# AutoResearch placeholder
+# S0-001 typed Windows infrastructure smoke
```

The deployed Phase1 executor ignores this patch; record that limitation explicitly.

- [ ] **Step 2: Submit one bounded job**

Run:

```bash
uv run arw experiments submit experiments/smoke/phase1_s0_candidate.patch \
  --project-id reliablepeft-phase1 \
  --source-commit 847dcced0a55bf9fca29f663b0a9c9a3e9d115a8 \
  --title "[S0] typed Windows smoke — config00 seed99 epoch1" \
  --plan-id s0-001-20260731 \
  --seed 99 \
  --time-budget 600 \
  --max-parallel 1 \
  --config-id 0 \
  --epochs 1 \
  --batch-size 32 \
  --json
```

Expected: HTTP 201 represented as JSON with a new experiment ID and non-terminal state.

- [ ] **Step 3: Monitor without starting a second job**

Poll:

```bash
uv run arw experiments show <experiment_id> --json
uv run arw experiments events <experiment_id> --json
```

Expected ordered states: submitted/policy-validating/queued/running/succeeded. A skipped transient state is acceptable if events preserve it.

- [ ] **Step 4: Verify artifacts**

Run:

```bash
uv run arw experiments artifacts <experiment_id> --json
```

Expected: non-empty manifest with valid SHA-256 values. If terminal state is failed, record the typed error and do not resubmit automatically.

- [ ] **Step 5: Update the ARIS tracker and manifest**

Set `S0-001` to `DONE` only for `succeeded + non-empty artifact manifest`; otherwise set `FAILED` with the exact failure layer. Write a new timestamped tracker before updating the fixed latest copy, then append both files to `MANIFEST.md`.

- [ ] **Step 6: Commit evidence metadata**

```bash
git add experiments/smoke/phase1_s0_candidate.patch refine-logs/EXPERIMENT_TRACKER*.md MANIFEST.md
git commit -m "test: record Windows S0 smoke submission"
```
