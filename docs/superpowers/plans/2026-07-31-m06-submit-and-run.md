# M0.6 Submit-and-Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parameterize the existing direct M0.5 runner just enough to execute the approved 20-bundle M0.6 matrix, validate one bridge bundle, and start the sequential Windows run immediately.

**Architecture:** Keep the existing causal runner and five state-carrier branches. Add explicit replay/probe/pulse randomness, two missing audit artifacts, a bundle finalizer for redirected stdout, and one bounded PowerShell loop. Do not build an API, scheduler, service, M1 runner, or new training pipeline.

**Tech Stack:** Python 3.12, PyTorch, pytest, Ruff, mypy, PowerShell, Git, SSH, Windows RTX 5080.

## Global Constraints

- Run only the approved M0.6 bridge and 20 evidence bundles.
- Keep model, LoRA, batch size, warm-up, replay horizons, precision, and optimizer unchanged.
- Use train seeds `3–12`, pulse seeds `271828` and `161803`, replay seed `420000 + train_seed`, probe seed `20260806`, and split seed `20260731`.
- Run one bundle at a time with Windows `BelowNormal` priority and affinity mask `0x3FF`.
- Never load CIFAR-100 test data.
- Do not implement M1 or change the candidate algorithm.
- Preserve all existing untracked local outputs.

---

### Task 1: Parameterize Independent M0.6 Randomness

**Files:**
- Modify: `experiments/m0_optimizer_state/m0_run.py`
- Modify: `tests/test_m0_runner.py`

**Interfaces:**
- Consumes: existing `RunConfig`, `build_batch_plans`, `run`, and CLI.
- Produces: optional `replay_seed` and `probe_seed` fields plus CLI flags `--split-seed`, `--pulse-seed`, `--replay-seed`, and `--probe-seed`.

- [ ] **Step 1: Write failing tests**

Add tests that assert an explicit replay seed changes only replay plans, an explicit probe
seed is shared across pulse replicates, and `corrupt_pulse` receives the exact pulse seed
without adding the train seed.

- [ ] **Step 2: Run the focused tests and observe failure**

Run:

```bash
uv run pytest tests/test_m0_runner.py -q
```

Expected: failure because `RunConfig` and the CLI do not expose the new seeds.

- [ ] **Step 3: Implement the minimal seed split**

Extend `RunConfig` with:

```python
replay_seed: int | None = None
probe_seed: int | None = None
```

When `replay_seed is None`, retain the legacy single-stream plan generation so the bridge
can reproduce pilot seed 0. Otherwise build warm-up plus pulse plans from `seed` and replay
plans from `replay_seed`. Use `probe_seed` directly when present, otherwise preserve the
legacy `split_seed + seed` rule. Pass `pulse_seed` directly to `corrupt_pulse`.

Allow any nonnegative train seed at the CLI and reject negative seed inputs.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/test_m0_runner.py tests/test_m0_core.py -q
```

Expected: all tests pass, including legacy bridge behavior.

---

### Task 2: Complete the Evidence Bundle Audit

**Files:**
- Modify: `experiments/m0_optimizer_state/m0_run.py`
- Create: `experiments/m0_optimizer_state/finalize_bundle.py`
- Modify: `tests/test_m0_runner.py`
- Create: `tests/test_m06_finalize_bundle.py`

**Interfaces:**
- Consumes: five branch states and an output directory containing redirected `stdout.log`.
- Produces: `branch_construction_manifest.json`, `provenance.json`, and a final verified `sha256_manifest.json`.

- [ ] **Step 1: Write artifact tests**

Require each state-attribution bundle to contain hashes for branch parameters, `exp_avg`,
`exp_avg_sq`, optimizer steps, and RNG at horizon zero. Require provenance to contain the
source commit and `uv.lock` digest.

Test `finalize_bundle(output_dir)` rejects a missing/failed summary, non-finite trajectory,
wrong branch/horizon set, `test_loaded != false`, or missing stdout. Assert it rewrites the
SHA-256 manifest over every file except itself.

- [ ] **Step 2: Run and observe failure**

```bash
uv run pytest tests/test_m0_runner.py tests/test_m06_finalize_bundle.py -q
```

- [ ] **Step 3: Implement branch and provenance manifests**

Hash branch parameters and moment-specific optimizer projections with the existing stable
state hashing helper before replay. Resolve `git rev-parse HEAD`, hash `uv.lock`, and fail
closed when either value is unavailable in a real CUDA run.

Expose:

```python
def finalize_bundle(output_dir: Path) -> None:
    """Validate a completed bundle and atomically rewrite its file manifest."""
```

The CLI is:

```bash
uv run python -m experiments.m0_optimizer_state.finalize_bundle OUTPUT_DIR
```

- [ ] **Step 4: Run artifact tests**

```bash
uv run pytest tests/test_m0_runner.py tests/test_m06_finalize_bundle.py -q
```

Expected: all pass.

---

### Task 3: Add the Bounded Windows Launcher

**Files:**
- Create: `scripts/run_m06.ps1`

**Interfaces:**
- Consumes: expected source commit, `C:\arw-data`, old M0.5 seed0 artifacts, and the fixed M0.6 matrix.
- Produces: one excluded bridge directory, twenty finalized evidence directories, and a master log under `C:\arw-results\m06`.

- [ ] **Step 1: Implement fail-closed launcher preflight**

The script accepts `-ExpectedCommit`, verifies clean Git state and exact HEAD, sets its
process priority to `BelowNormal` and affinity to `0x3FF`, checks CUDA, data path, and old
seed0 evidence, then refuses to run if any check fails.

- [ ] **Step 2: Implement bridge**

Run seed0 with legacy replay/probe semantics and pulse seed `314159`. Redirect stdout to
the bridge directory, finalize the bundle, then require the new and archived seed0
`trajectory_metrics.jsonl` and `checkpoint_manifest.json` hashes to match exactly.

- [ ] **Step 3: Implement the evidence loop**

For train seeds `3..12` and pulse seeds `271828,161803`, run:

```powershell
uv run python -m experiments.m0_optimizer_state.m0_run `
  --optimizer adamw --pulse label_flip --state-attribution `
  --seed $trainSeed --split-seed 20260731 --pulse-seed $pulseSeed `
  --replay-seed (420000 + $trainSeed) --probe-seed 20260806 `
  --data-dir C:\arw-data --output-dir $bundleDir
```

Finalize and validate each bundle before starting the next. Existing valid completed
bundles may be skipped; partial or invalid directories fail closed and are never deleted.

---

### Task 4: Validate, Commit, and Push the Frozen Runner

**Files:**
- All files from Tasks 1–3.

- [ ] **Step 1: Run local quality gates**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
git diff --check
```

Expected: every command succeeds.

- [ ] **Step 2: Commit only scoped files**

```bash
git add experiments/m0_optimizer_state/m0_run.py \
  experiments/m0_optimizer_state/finalize_bundle.py \
  tests/test_m0_runner.py tests/test_m06_finalize_bundle.py scripts/run_m06.ps1
git commit -m "feat: submit audited M0.6 confirmation matrix"
git push origin codex/m05-direct-experiment
```

Record the resulting commit as the immutable M0.6 source commit.

---

### Task 5: Deploy and Start M0.6

**Files:**
- No new files.

- [ ] **Step 1: Fast-forward Windows**

Use read-only preflight first, then:

```powershell
Set-Location C:\arw-m0
git fetch origin
git merge --ff-only origin/codex/m05-direct-experiment
```

Require clean status and exact source commit.

- [ ] **Step 2: Start the launcher detached**

Start one detached PowerShell process with `scripts\run_m06.ps1`, the immutable commit, and
redirected master stdout/stderr under `C:\arw-results\m06`. Record PID and start time.

- [ ] **Step 3: Monitor the bridge and first evidence bundle**

Wait for bridge success, exact pilot hash match, and completion of the first M0.6 bundle.
Report PID, commit, completed/total bundles, current bundle, artifact path, and any failure.
Do not wait silently longer than 60 seconds between status updates.
