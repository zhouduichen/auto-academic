# M0.5 State-Carrier Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the three-seed M0.5 experiment that causally separates AdamW first-moment, second-moment, and coupled state contamination while lowering CPU work and preserving the valid M0 numerical protocol.

**Architecture:** Keep the causal state operations in `src/arw/m0_core.py`, move deterministic planned-batch preparation into a focused reusable module, and add a separate M0.5 runner rather than changing the semantics of the valid M0 runner. M0.5 advances branches batch-major so one prepared GPU batch feeds all five branches, then a pure analysis script applies the preregistered cross-seed attribution rules.

**Tech Stack:** Python 3.12 on Mac, existing Windows `uv` Python runtime, PyTorch, torchvision, Transformers, PEFT, pytest, Ruff, mypy, PowerShell, SSH/Tailscale.

## Global Constraints

- Do not change batch size 32, precision, model, LoRA rank, warmup steps, replay horizons, split seed, pulse seed, or label-flip construction from valid M0.
- Load the pretrained ViT backbone fail-closed; missing or unexpected backbone weights abort the run.
- Never construct or load the CIFAR-100 test split.
- Use only `label_flip + AdamW` and seeds `0, 1, 2`.
- Affinitize the Windows process tree to at most 10 of 24 logical processors; use 4 main PyTorch threads, one inter-op thread, and two one-thread DataLoader workers.
- Execute one seed at a time. Do not add services, schedulers, APIs, or runner infrastructure.
- The smoke run must use a separate output directory and must never be included in scientific analysis.
- Every run fails closed on state-schema mismatch, RNG mismatch, non-finite values, missing horizons, source-commit mismatch, pretrained-loading anomalies, or missing artifacts.

---

### Task 1: Integrate the Valid Pretrained-Backbone Fix

**Files:**
- Modify by cherry-pick: `experiments/m0_optimizer_state/m0_run.py`
- Modify by cherry-pick: `tests/test_m0_runner.py`

**Interfaces:**
- Consumes: commit `95cadc4` on `codex/fix-m0-vit-backbone`.
- Produces: fail-closed `build_model(config: RunConfig) -> nn.Module` on the implementation branch.

- [ ] **Step 1: Create an implementation branch and cherry-pick only the code fix**

```bash
git switch -c codex/m05-state-carrier
git cherry-pick 95cadc4
```

Expected: the branch contains the ViT loading fix but not result commit `aecadcc`.

- [ ] **Step 2: Run the focused loading tests**

```bash
uv run pytest tests/test_m0_runner.py -q
```

Expected: all M0 runner tests pass, including missing/unexpected backbone-weight rejection.

- [ ] **Step 3: Verify the branch history**

```bash
git log -4 --oneline
```

Expected: `95cadc4` is present and `aecadcc` is absent.

---

### Task 2: Add Exact AdamW Moment Composition

**Files:**
- Modify: `src/arw/m0_core.py`
- Modify: `tests/test_m0_core.py`

**Interfaces:**
- Consumes: `CandidateState`, `BranchState`, and AdamW optimizer state dictionaries.
- Produces: `compose_adamw_state(clean, corrupt, corrupt_moments) -> OptimizerState` and `build_state_carrier_branches(clean, corrupt) -> dict[str, BranchState]`.

- [ ] **Step 1: Write failing branch-construction tests**

Add tests that create one clean and one norm-matched corrupt AdamW candidate and assert:

```python
branches = build_state_carrier_branches(clean, corrupt)
assert tuple(branches) == (
    "control", "m_only", "v_only", "state_both", "parameter_only"
)
assert optimizer_moment_distance(
    branches["m_only"].optimizer, corrupt.optimizer, "exp_avg"
) == 0.0
assert optimizer_moment_distance(
    branches["m_only"].optimizer, clean.optimizer, "exp_avg_sq"
) == 0.0
assert optimizer_moment_distance(
    branches["v_only"].optimizer, clean.optimizer, "exp_avg"
) == 0.0
assert optimizer_moment_distance(
    branches["v_only"].optimizer, corrupt.optimizer, "exp_avg_sq"
) == 0.0
```

Also assert that all five states have identical parameter-group metadata and identical AdamW step counters.

- [ ] **Step 2: Run the tests and observe the missing interface failure**

```bash
uv run pytest tests/test_m0_core.py -q
```

Expected: collection or execution fails because the two new functions do not exist.

- [ ] **Step 3: Implement strict state composition**

Implement `compose_adamw_state` by deep-copying the clean state, validating identical state IDs, parameter groups, tensor shapes, dtypes, devices, and step counters, then replacing only the requested keys from `{"exp_avg", "exp_avg_sq"}`. Reject missing moment keys and unsupported optimizer schemas with `ValueError`.

Build branches exactly as:

```python
return {
    "control": BranchState(clean.parameters, clean.optimizer, clean.loss, clean.gradient_norm),
    "m_only": BranchState(
        clean.parameters,
        compose_adamw_state(clean.optimizer, corrupt.optimizer, frozenset({"exp_avg"})),
        clean.loss,
        corrupt.gradient_norm,
    ),
    "v_only": BranchState(
        clean.parameters,
        compose_adamw_state(clean.optimizer, corrupt.optimizer, frozenset({"exp_avg_sq"})),
        clean.loss,
        corrupt.gradient_norm,
    ),
    "state_both": BranchState(
        clean.parameters, corrupt.optimizer, clean.loss, corrupt.gradient_norm
    ),
    "parameter_only": BranchState(
        corrupt.parameters, clean.optimizer, corrupt.loss, corrupt.gradient_norm
    ),
}
```

- [ ] **Step 4: Run focused and existing core tests**

```bash
uv run pytest tests/test_m0_core.py -q
```

Expected: all tests pass and the existing M0 branch behavior is unchanged.

- [ ] **Step 5: Commit the state intervention**

```bash
git add src/arw/m0_core.py tests/test_m0_core.py
git commit -m "feat: isolate AdamW moment carriers"
```

---

### Task 3: Add the Low-CPU Planned-Batch Pipeline

**Files:**
- Create: `experiments/m0_optimizer_state/planned_batches.py`
- Create: `tests/test_m05_planned_batches.py`
- Modify: `experiments/m0_optimizer_state/m0_run.py`

**Interfaces:**
- Consumes: CIFAR-100 training dataset and the existing deterministic augmentation values.
- Produces: `BatchPlan`, `build_batch_plans`, `PlannedBatchDataset`, `configure_cpu_runtime()`, `worker_init_fn(worker_id)`, and `move_batch(batch, device)`.

- [ ] **Step 1: Write deterministic equivalence and thread-limit tests**

Test that a `PlannedBatchDataset` item is exactly equal to the former `materialize_batch` CPU result for fixed sample IDs, crop coordinates, and flips. Test that `worker_init_fn(0)` calls `torch.set_num_threads(1)`, and that `move_batch` preserves values and labels on CPU.

- [ ] **Step 2: Run the new tests and observe import failure**

```bash
uv run pytest tests/test_m05_planned_batches.py -q
```

Expected: FAIL because `planned_batches.py` does not exist.

- [ ] **Step 3: Implement the plan-indexed dataset**

The module must expose this shape:

```python
class PlannedBatchDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, dataset: CIFAR100, plans: list[BatchPlan]) -> None:
        self._dataset = dataset
        self._plans = tuple(plans)

    def __len__(self) -> int:
        return len(self._plans)

    def __getitem__(self, plan_index: int) -> tuple[Tensor, Tensor]:
        return materialize_cpu_batch(self._dataset, self._plans[plan_index])

def configure_cpu_runtime() -> None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

def worker_init_fn(_worker_id: int) -> None:
    torch.set_num_threads(1)

def move_batch(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
```

Move `BatchPlan`, `build_batch_plans`, and the existing deterministic image transform into
this module, then make `m0_run.py` import them. This avoids a circular dependency when
`m05_run.py` uses the same plan type. Do not alter sampling, resize, crop, flip,
normalization, or interpolation behavior.

- [ ] **Step 4: Run data-pipeline and M0 regression tests**

```bash
uv run pytest tests/test_m05_planned_batches.py tests/test_m0_runner.py -q
```

Expected: all tests pass with byte-identical transformed tensors.

- [ ] **Step 5: Commit the data pipeline**

```bash
git add experiments/m0_optimizer_state/planned_batches.py \
  experiments/m0_optimizer_state/m0_run.py tests/test_m05_planned_batches.py
git commit -m "perf: prepare deterministic batches with two workers"
```

---

### Task 4: Implement Batch-Major M0.5 Replay

**Files:**
- Create: `experiments/m0_optimizer_state/m05_run.py`
- Create: `tests/test_m05_runner.py`

**Interfaces:**
- Consumes: valid M0 warmup/candidate helpers, `build_state_carrier_branches`, and planned GPU batches.
- Produces: `run(config, data_dir, output_dir) -> dict[str, object]` for one AdamW label-flip seed.

- [ ] **Step 1: Write a failing five-branch smoke test**

Use the existing fake CIFAR-100 and tiny vision model with `warmup_steps=1`, `replay_steps=2`, and `batch_size=2`. Assert branch order, horizons, `test_loaded is False`, and `train_steps == 1 + 2 + 5 * 2`.

- [ ] **Step 2: Write a failing branch-major parity test**

Evolve the same five tiny-model branches using a simple test-only branch-major reference and production batch-major replay. Compare every final trainable parameter, `exp_avg`, `exp_avg_sq`, step counter, and per-branch RNG hash with `rtol=0, atol=0`.

- [ ] **Step 3: Run the focused tests and observe failure**

```bash
uv run pytest tests/test_m05_runner.py -q
```

Expected: FAIL because `m05_run.py` does not exist.

- [ ] **Step 4: Implement per-branch runtime state**

Use a runtime record with independent branch and RNG snapshots:

```python
@dataclass
class RuntimeBranch:
    state: BranchState
    rng: RngState

def advance_branch(
    runtime: RuntimeBranch,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch: tuple[Tensor, Tensor],
) -> RuntimeBranch:
    restore_branch(model, optimizer, runtime.state)
    restore_rng_state(runtime.rng)
    training_step(model, optimizer, criterion, *batch)
    return RuntimeBranch(
        state=snapshot_branch(model, optimizer, runtime.state),
        rng=capture_rng_state(),
    )
```

For each replay batch, move it to the GPU once, then advance branches in frozen order `control`, `m_only`, `v_only`, `state_both`, `parameter_only`. At measurement horizons, restore and evaluate each branch independently against the control snapshot.

- [ ] **Step 5: Add timing without synchronization in the training path**

Measure DataLoader wait using `time.perf_counter()` around iterator retrieval. Measure aggregate step wall time around the five branch updates. Store `data_wait_seconds`, `branch_step_seconds`, `data_wait_fraction`, and `examples_per_second`; do not call `torch.cuda.synchronize()` inside every training step.

- [ ] **Step 6: Run parity and smoke tests**

```bash
uv run pytest tests/test_m05_runner.py -q
```

Expected: all five branch states and RNG states match branch-major reference exactly.

- [ ] **Step 7: Commit the replay engine**

```bash
git add experiments/m0_optimizer_state/m05_run.py tests/test_m05_runner.py
git commit -m "feat: run M0.5 replay batch-major"
```

---

### Task 5: Add Audited M0.5 Artifacts and CLI Validation

**Files:**
- Modify: `experiments/m0_optimizer_state/m05_run.py`
- Modify: `tests/test_m05_runner.py`

**Interfaces:**
- Consumes: one completed five-branch trajectory.
- Produces: the frozen M0 artifact set plus M0.5 branch-construction and timing fields.

- [ ] **Step 1: Extend the artifact test**

Require exactly these files:

```python
expected = {
    "run_config.json",
    "split_ids.json",
    "replay_manifest.json",
    "checkpoint_manifest.json",
    "branch_construction_manifest.json",
    "trajectory_metrics.jsonl",
    "summary.json",
    "sha256_manifest.json",
}
```

Assert that the construction manifest contains per-branch parameter, `exp_avg`,
`exp_avg_sq`, optimizer-step, and RNG hashes at horizon zero. Assert source commit, timing
fields, all nine horizons, finite metrics, and `test_loaded: false`.

- [ ] **Step 2: Run the test and observe missing artifact fields**

```bash
uv run pytest tests/test_m05_runner.py -q
```

Expected: FAIL on missing construction manifest or timing fields.

- [ ] **Step 3: Implement fail-closed artifact generation**

Use stable sorted JSON, write artifacts only inside the requested output directory, and
write `sha256_manifest.json` last. Abort before success status when any invariant fails.
Expose only these scientific CLI values:

```text
--seed {0,1,2}
--data-dir PATH
--output-dir PATH
--warmup-steps 500
--replay-steps 128
--batch-size 32
--probe-size 256
--device cuda
```

Optimizer and pulse are fixed internally to `adamw` and `label_flip`.

- [ ] **Step 4: Run M0.5 and M0 regression tests**

```bash
uv run pytest tests/test_m05_runner.py tests/test_m0_runner.py tests/test_m0_core.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the audited runner**

```bash
git add experiments/m0_optimizer_state/m05_run.py tests/test_m05_runner.py
git commit -m "feat: audit M0.5 state-carrier artifacts"
```

---

### Task 6: Implement the Preregistered Cross-Seed Attribution

**Files:**
- Create: `src/arw/m05_analysis.py`
- Create: `experiments/m0_optimizer_state/m05_analyze.py`
- Create: `tests/test_m05_analysis.py`

**Interfaces:**
- Consumes: three successful M0.5 `summary.json` files for seeds 0, 1, and 2.
- Produces: `attribution_report.json` with `m_dominant`, `v_dominant`, `coupled`, or `inconclusive`.

- [ ] **Step 1: Write table-driven failing tests for all four decisions**

Each fixture supplies `A_m`, `A_v`, and `A_both` for three seeds. Assert eligibility,
median interaction, leave-one-seed-out stability, and the final decision. Include a sign-
unstable fixture that must return `inconclusive` even when its mean is positive.

- [ ] **Step 2: Run the tests and observe import failure**

```bash
uv run pytest tests/test_m05_analysis.py -q
```

Expected: FAIL because the analysis module does not exist.

- [ ] **Step 3: Implement pure attribution functions**

Expose typed functions:

```python
@dataclass(frozen=True)
class SeedEffects:
    seed: int
    a_m: float
    a_v: float
    a_both: float

def _classify_once(effects: list[SeedEffects]) -> str:
    m_median = median(effect.a_m for effect in effects)
    v_median = median(effect.a_v for effect in effects)
    both_median = median(effect.a_both for effect in effects)
    interaction = median(
        effect.a_both - effect.a_m - effect.a_v for effect in effects
    )
    if not all(effect.a_both > 0 for effect in effects):
        return "inconclusive"
    m_eligible = all(effect.a_m > 0 for effect in effects) and abs(m_median) >= 0.25 * abs(both_median)
    v_eligible = all(effect.a_v > 0 for effect in effects) and abs(v_median) >= 0.25 * abs(both_median)
    if abs(interaction) >= 0.25 * abs(both_median):
        return "coupled"
    if m_eligible and (not v_eligible or abs(m_median) >= 2 * abs(v_median)):
        return "m_dominant"
    if v_eligible and (not m_eligible or abs(v_median) >= 2 * abs(m_median)):
        return "v_dominant"
    if m_eligible and v_eligible:
        return "coupled"
    return "inconclusive"

def classify_carrier(effects: list[SeedEffects]) -> dict[str, object]:
    if {effect.seed for effect in effects} != {0, 1, 2} or len(effects) != 3:
        raise ValueError("exactly one effect for each seed 0, 1, and 2 is required")
    if not all(
        isfinite(value)
        for effect in effects
        for value in (effect.a_m, effect.a_v, effect.a_both)
    ):
        raise ValueError("effects must be finite")
    decision = _classify_once(effects)
    leave_one_out = [
        _classify_once([effect for effect in effects if effect.seed != held_out])
        for held_out in (0, 1, 2)
    ]
    if decision == "inconclusive" or any(item != decision for item in leave_one_out):
        decision = "inconclusive"
    return {"decision": decision, "leave_one_out": leave_one_out}
```

Implement exactly the 25% eligibility/interaction, 2x dominance, all-positive, and leave-
one-seed-out rules from the approved design. Require exactly seeds `{0, 1, 2}` and reject
non-finite or duplicate input.

- [ ] **Step 4: Implement the read-only CLI**

`m05_analyze.py` accepts the three bundle directories and one output path, verifies every
SHA-256 manifest before reading summaries, writes stable JSON, and returns nonzero for an
invalid or incomplete evidence set.

- [ ] **Step 5: Run analysis tests**

```bash
uv run pytest tests/test_m05_analysis.py -q
```

Expected: all decision and validation tests pass.

- [ ] **Step 6: Commit the analysis gate**

```bash
git add src/arw/m05_analysis.py experiments/m0_optimizer_state/m05_analyze.py \
  tests/test_m05_analysis.py
git commit -m "feat: classify M0.5 optimizer-state carriers"
```

---

### Task 7: Validate Locally, Smoke on Windows, Then Run Three Seeds

**Files:**
- Modify only if validation reveals a scoped defect: files introduced in Tasks 1–6.
- Generate on Windows: `C:\arw-results\m05\smoke\*`
- Generate on Windows: `C:\arw-results\m05\seed0\*`, `seed1\*`, `seed2\*`
- Generate on Windows: `C:\arw-results\m05\attribution_report.json`

**Interfaces:**
- Consumes: committed implementation branch and Windows repo `C:\arw-m0`.
- Produces: one excluded smoke bundle, three valid bundles, and one audited attribution report.

- [ ] **Step 1: Run complete local quality gates**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Expected: formatting, lint, strict typing, and the full test suite pass.

- [ ] **Step 2: Confirm a clean scoped diff and commit any validation fix**

```bash
git status --short
git diff --check
```

Expected: only pre-existing unrelated untracked paths remain; implementation files are committed.

- [ ] **Step 3: Push the implementation branch and update the Windows worktree**

```bash
git push -u origin codex/m05-state-carrier
ssh 33135@autoresearch-5080 "powershell -NoProfile -Command \"Set-Location C:\\arw-m0; git fetch origin; git switch --track -c codex/m05-state-carrier origin/codex/m05-state-carrier\""
```

Expected: Windows `HEAD` equals the pushed source commit. If Windows has local result-only commits,
switching branches preserves them. If the M0.5 branch already exists locally, use
`git switch codex/m05-state-carrier` followed by
`git merge --ff-only origin/codex/m05-state-carrier`.

- [ ] **Step 4: Run a bounded Windows smoke with the low-CPU policy**

Run the transient command through Windows `start` so the affinity and priority are
inherited by `uv` and its Python/DataLoader children:

```powershell
cmd.exe /d /c start "" /wait /belownormal /affinity 3FF `
  uv run python -m experiments.m0_optimizer_state.m05_run `
    --seed 0 --data-dir C:\arw-data --output-dir C:\arw-results\m05\smoke `
    --warmup-steps 2 --replay-steps 2 --batch-size 2 --probe-size 8 --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

The hexadecimal mask `3FF` exposes exactly ten logical processors. Do not register a
scheduled task or service.

- [ ] **Step 5: Audit the smoke and exclude it from evidence**

Expected: exit code 0, pretrained loading 200/200, five branches, required artifacts,
finite metrics, `test_loaded: false`, CPU affinity limited to 10 logical processors, and
no source-commit mismatch. Delete nothing; retain it under the dedicated `smoke` path.

- [ ] **Step 6: Run seeds 0, 1, and 2 sequentially**

For each seed use the frozen defaults and output `C:\arw-results\m05\seed<seed>`. Start
the next seed only after the previous artifact manifest validates. Preserve the same
`0x3FF` affinity, `BelowNormal` priority, two workers, and one-run-at-a-time policy.

- [ ] **Step 7: Generate the attribution report**

```powershell
uv run python -m experiments.m0_optimizer_state.m05_analyze `
  C:\arw-results\m05\seed0 C:\arw-results\m05\seed1 C:\arw-results\m05\seed2 `
  --output C:\arw-results\m05\attribution_report.json
```

Expected: one of the four preregistered carrier decisions with all per-seed AUCs,
interaction contrasts, eligibility checks, and leave-one-seed-out decisions.

- [ ] **Step 8: Archive and copy the evidence to Mac**

Create `C:\arw-results\m05\m05-full-20260731.zip`, calculate SHA-256, copy it to
`auto-academic/tmp/`, verify the hash and ZIP integrity on Mac, and report the carrier
decision before designing or implementing M1.
