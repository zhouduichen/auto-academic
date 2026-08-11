# Causal Transport Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and dispatch a fail-closed, staged causal-transport experiment that tests whether optimizer-state dominance after a local noisy exposure reverses to parameter dominance after accumulated noisy-label PEFT.

**Architecture:** A pure evaluator freezes the carrier-ratio and stage decisions. A paired-exposure bundle runner warms one model on clean data, forks clean/noisy exposure from the exact same state for a fixed dose, then reuses the crossed replay core for `CC/CN/NC/NN` continuations. A staged orchestrator runs held-out CAdam Gate A first, then AdamW multi-dose Gate B, and prepares CIFAR-100N Gate C only after Gate B passes.

**Tech Stack:** Python 3.12, PyTorch, torchvision, NumPy, existing ViT/LoRA runners, pytest, Ruff, PowerShell Scheduled Tasks, SHA-256 JSON artifacts.

## Global Constraints

- Existing M0.6, M1, M1.1, and `2eee6e1301e24bac3f984f5e243bd4c5189a8f11` artifacts are immutable discovery inputs.
- Confirmation uses three new training seeds `401`, `402`, and `403`; paired seed is the experimental unit.
- CIFAR-100 Gate B doses are exactly `(1, 8, 64, 1250)` optimizer steps after a 500-step clean warm-up.
- `R = log2((abs(state_effect) + 1e-12) / (abs(parameter_effect) + 1e-12))`.
- State dominance requires `R >= 1`; parameter dominance requires `R <= -1`; an interaction at least as large as both main effects invalidates the main-effect label.
- Test loaders are never constructed. Tuning data are used only at the frozen horizons 128 and 512.
- Windows execution uses one RTX 5080 task at a time, CPU affinity decimal `4095`, no GPU power cap, and the fixed venv `C:\arw-m0\.venv\Scripts\python.exe`.
- New code must be TDD, atomic on terminal writes, resumable by dose, source-commit bound, and preserve all legacy causal-attribution behavior.

---

## File structure

- Create `experiments/causal_transport_gate.py`: pure carrier classification, stage evaluation, contract validation, and atomic decisions.
- Create `experiments/causal_transport_bundle.py`: paired clean/noisy exposure, snapshot persistence, common continuation, and one factorial bundle.
- Create `experiments/causal_transport_orchestrator.py`: exact Gate A/B matrices, input validation, staged execution, and artifact collection.
- Create `experiments/causal_transport_sentinel.py`: CUDA functional/resource sentinel for the new bundle path.
- Create `tests/test_causal_transport_gate.py`: mechanical decision tests.
- Create `tests/test_causal_transport_bundle.py`: exposure pairing, state crossing, resume, and isolation tests.
- Create `tests/test_causal_transport_orchestrator.py`: exact job order, fail-closed staging, and stale-decision tests.
- Modify `experiments/causal_attribution_replay.py`: expose the already-tested measurement core without changing legacy request semantics.
- Modify `tests/test_causal_attribution_replay.py`: regression test for the extracted measurement core.
- Conditional after Gate B: create `experiments/cifar100n_transport_data.py` and `tests/test_cifar100n_transport_data.py` for sealed human-noise inputs.

---

### Task 1: Freeze carrier metrics and stage decisions

**Files:**
- Create: `experiments/causal_transport_gate.py`
- Create: `tests/test_causal_transport_gate.py`

**Interfaces:**
- Produces: `carrier_ratio(effects: dict[str, float]) -> float`
- Produces: `classify_effect(effects: dict[str, float]) -> str`
- Produces: `evaluate_anchor_pair(local: dict[str, object], accumulated: dict[str, object]) -> dict[str, object]`
- Produces: `evaluate_stage(records: dict[str, dict[str, object]], required_seeds: tuple[int, ...], stage: Literal["A", "B", "C"]) -> dict[str, object]`
- Produces: `write_decision(path: Path, decision: dict[str, object]) -> None`

- [ ] **Step 1: Write failing metric and interaction tests**

```python
def test_carrier_ratio_and_interaction_override() -> None:
    assert gate.carrier_ratio({"parameter": 1.0, "state": 4.0, "interaction": 0.5}) == pytest.approx(2.0)
    assert gate.classify_effect({"parameter": 1.0, "state": 4.0, "interaction": 0.5}) == "state"
    assert gate.classify_effect({"parameter": 4.0, "state": 1.0, "interaction": 0.5}) == "parameter"
    assert gate.classify_effect({"parameter": 1.0, "state": 2.0, "interaction": 3.0}) == "interaction"
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run: `uv run --locked pytest tests/test_causal_transport_gate.py -q`

Expected: collection fails because `experiments.causal_transport_gate` does not exist.

- [ ] **Step 3: Implement finite carrier classification**

```python
EPSILON = 1e-12

def carrier_ratio(effects: dict[str, float]) -> float:
    values = {key: float(effects[key]) for key in ("parameter", "state", "interaction")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("carrier effects must be finite")
    return math.log2((abs(values["state"]) + EPSILON) / (abs(values["parameter"]) + EPSILON))

def classify_effect(effects: dict[str, float]) -> str:
    parameter, state, interaction = (abs(float(effects[key])) for key in ("parameter", "state", "interaction"))
    ratio = carrier_ratio(effects)
    if interaction >= max(parameter, state):
        return "interaction"
    if ratio >= 1.0:
        return "state"
    if ratio <= -1.0:
        return "parameter"
    return "mixed"
```

- [ ] **Step 4: Add failing replicated anchor tests**

```python
def _summary(kind: str) -> dict[str, object]:
    values = {
        "state": {"parameter": 1.0, "state": 4.0, "interaction": 0.5},
        "parameter": {"parameter": 4.0, "state": 1.0, "interaction": 0.5},
        "mixed": {"parameter": 1.0, "state": 1.0, "interaction": 0.5},
    }[kind]
    return {
        "effects": {
            "clean_loss_excess_auc_128": dict(values),
            "tuning_loss_h512": dict(values),
        },
        "source_commit": "a" * 40,
        "test_loaded": False,
    }

def test_stage_requires_two_of_three_anchor_crossings() -> None:
    passing = {"local": _summary("state"), "accumulated": _summary("parameter")}
    reverse = {"local": _summary("parameter"), "accumulated": _summary("state")}
    result = gate.evaluate_stage({"401": passing, "402": passing, "403": {"local": _summary("mixed"), "accumulated": _summary("mixed")}}, (401, 402, 403), "B")
    assert result["decision"] == "GO"
    assert gate.evaluate_stage({"401": passing, "402": passing, "403": reverse}, (401, 402, 403), "B")["decision"] == "NO-GO"
```

- [ ] **Step 5: Implement anchor and stage evaluation**

Evaluate both `clean_loss_excess_auc_128` and `tuning_loss_h512`. An anchor crosses only when local class is `state`, accumulated class is `parameter`, neither interaction dominates, and both outcome families agree. Gate A requires both seeds; Gates B/C require at least two of three and reject an opposite twofold reversal in the third.

- [ ] **Step 6: Test atomic stale-decision replacement and fail-closed malformed inputs**

Create the parent directory, write JSON to `path.with_suffix('.tmp')`, flush, then `replace(path)`. Reject missing keys, booleans-as-numbers, non-finite values, wrong source commit, `test_loaded != false`, duplicated seeds, and an existing decision from another contract hash.

- [ ] **Step 7: Run tests and lint**

Run: `uv run --locked pytest tests/test_causal_transport_gate.py -q`

Expected: all tests pass.

Run: `uv run --locked ruff check experiments/causal_transport_gate.py tests/test_causal_transport_gate.py`

Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add experiments/causal_transport_gate.py tests/test_causal_transport_gate.py
git commit -m "feat: freeze causal transport decisions"
```

### Task 2: Extract a reusable crossed measurement core

**Files:**
- Modify: `experiments/causal_attribution_replay.py`
- Modify: `tests/test_causal_attribution_replay.py`

**Interfaces:**
- Consumes: existing `CheckpointState`, `build_factorial_branches`, `run_cached_branches`, `factorial_effects`.
- Produces: `measure_factorial_replay(model, optimizer, branches, cached, probe, tuning_loader, device) -> tuple[list[dict[str, object]], dict[str, object]]`.

- [ ] **Step 1: Write a failing parity test**

Construct a tiny model, two valid `CheckpointState` values, two cached batches, and a probe loader. Assert the extracted function returns all four branches, frozen horizons, finite effects, and bitwise-identical rows on two calls with the same RNG.

- [ ] **Step 2: Verify the failure**

Run: `uv run --locked pytest tests/test_causal_attribution_replay.py -q`

Expected: fail because `measure_factorial_replay` is absent.

- [ ] **Step 3: Extract the existing measurement body without changing formulas**

Move the nested `measure_branch`, endpoint aggregation, and effect aggregation from `run_replay` into the new function. Pass `common_rng=capture_common_rng()` internally, retain `HORIZONS`, and return:

```python
return rows, {"endpoints": endpoints, "effects": effects, "test_loaded": False}
```

`run_replay` must call this function and write byte-for-byte equivalent schema fields for legacy requests.

- [ ] **Step 4: Run legacy and new tests**

Run: `uv run --locked pytest tests/test_causal_attribution_replay.py tests/test_causal_attribution_gate.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/causal_attribution_replay.py tests/test_causal_attribution_replay.py
git commit -m "refactor: expose factorial replay measurement"
```

### Task 3: Build paired exposure bundles

**Files:**
- Create: `experiments/causal_transport_bundle.py`
- Create: `tests/test_causal_transport_bundle.py`

**Interfaces:**
- Consumes: `clean.Config`, `clean._build_model`, `clean._build_optimizer`, `replay.measure_factorial_replay`, sealed M1 image/noise/tuning stores.
- Produces: `TransportRequest` dataclass.
- Produces: `PairedTransportDataset` with `batch(position, epoch, noisy) -> tuple[Tensor, Tensor]`.
- Produces: `run_bundle(request: TransportRequest) -> dict[str, object]`.

- [ ] **Step 1: Write failing paired-data tests**

```python
def test_paired_batch_changes_only_labels(fake_stores: Stores) -> None:
    data = bundle.PairedTransportDataset(fake_stores.image, fake_stores.training, augmentation_seed=10401)
    clean_images, clean_labels = data.batch(0, epoch=0, noisy=False)
    noisy_images, noisy_labels = data.batch(0, epoch=0, noisy=True)
    assert torch.equal(clean_images, noisy_images)
    assert not torch.equal(clean_labels, noisy_labels)
```

The fake store includes stable sample IDs, image rows, train order, noisy labels, and clean CIFAR targets. Assert identical crop/flip draws and exact stable-ID alignment.

- [ ] **Step 2: Write failing exposure-state tests**

Use a tiny model and optimizer. After a two-step clean warm-up, restore the common snapshot twice, apply dose `1` with paired clean/noisy labels, and assert the resulting snapshots have equal optimizer clocks, identical frozen tensors, and different trainable parameters/moments.

- [ ] **Step 3: Implement request validation and paired batches**

```python
@dataclass(frozen=True)
class TransportRequest:
    method: Literal["adamw", "cadam"]
    seed: int
    dose: Literal[1, 8, 64, 1250]
    warmup_steps: int
    continuation: Literal["clean", "noisy"]
    config: clean.Config
    data_dir: Path
    image_store: Path
    tuning_store: Path
    noisy_bundle: Path
    expected_input_binding: dict[str, object]
    output: Path
    source_commit: str
    device: str = "cuda"
```

Require `warmup_steps == 500` outside tests, `config.seed == seed`, `config.method == method`, `augmentation_seed == 10000 + seed`, exact dose membership, current source commit, and `test_loaded=false` seals.

- [ ] **Step 4: Implement warm-up and paired exposure**

Build one model/optimizer. Run 500 clean batches, capture the common `BranchState` and RNG, then for each condition restore both and apply exactly `dose` paired batches. Cache images once per step and select clean/noisy labels without changing order or augmentation. Validate optimizer step clocks equal `500 + dose`.

- [ ] **Step 5: Persist resumable exposure snapshots**

Before replay, atomically write `clean-exposure.pt` and `noisy-exposure.pt` with schema `causal-transport-exposure/1`, source commit, contract hash, seed, dose, warm-up steps, model state, optimizer state, and RNG hash. On restart, load only when both files and their SHA-256 entries match; otherwise fail closed rather than mixing one old and one new snapshot.

- [ ] **Step 6: Run the four crossed branches**

Build `CC/CN/NC/NN` from the paired exposure states. Cache 512 subsequent paired batches; select continuation labels from the request while keeping images and RNG common. Call `replay.measure_factorial_replay` and write `trajectory_metrics.jsonl`, `factorial_effects.json`, `summary.json`, `replay_manifest.json`, and `sha256_manifest.json`.

- [ ] **Step 7: Add nonfinite, resume, no-test, and config-mismatch tests**

Assert a non-finite exposure state writes no success summary; a valid snapshot pair skips warm-up/exposure on resume; one missing/corrupt snapshot rejects; mismatched IDs reject; and no CIFAR constructor is ever called with `train=False`.

- [ ] **Step 8: Run focused tests and lint**

Run: `uv run --locked pytest tests/test_causal_transport_bundle.py tests/test_causal_attribution_replay.py -q`

Expected: all tests pass.

Run: `uv run --locked ruff check experiments/causal_transport_bundle.py tests/test_causal_transport_bundle.py`

Expected: `All checks passed!`

- [ ] **Step 9: Commit**

```bash
git add experiments/causal_transport_bundle.py tests/test_causal_transport_bundle.py
git commit -m "feat: add paired causal transport bundles"
```

### Task 4: Orchestrate Gate A and Gate B

**Files:**
- Create: `experiments/causal_transport_orchestrator.py`
- Create: `tests/test_causal_transport_orchestrator.py`

**Interfaces:**
- Consumes: `run_bundle`, legacy `ReplayRequest`, `evaluate_stage`.
- Produces: `build_gate_a_matrix()`, `build_gate_b_matrix()`, `run_gate_a(args)`, `run_gate_b(args)`.

- [ ] **Step 1: Write failing exact-matrix tests**

Gate A must contain CAdam local bundle jobs for seeds 301/302 and legacy epoch-20 clean/noisy replays, with no CAdam base capture. Gate B must contain exactly `3 seeds x 4 doses x 2 continuations = 24` bundles for seeds 401/402/403.

- [ ] **Step 2: Write failing stage-order tests**

Mock all jobs. Assert Gate B is never called when Gate A writes `NO-GO`; stale terminal decisions are removed before validation; a valid completed job is reused; and a job from another source commit or contract hash fails.

- [ ] **Step 3: Implement immutable contract creation**

Write `CAUSAL_TRANSPORT_CONTRACT.json` before any CUDA job with exact seeds, doses, warm-up, endpoints, thresholds, store hashes, legacy discovery decision hash, source commit, lock hash, and `test_loaded=false`. Refuse to overwrite a different contract.

- [ ] **Step 4: Implement Gate A collection**

For each seed, map the dose-1 CAdam bundle to the local anchor and the existing epoch-20 CAdam factorial replay to the accumulated anchor. Evaluate both outcome families, write `GATE_A_DECISION.json`, and exit code 2 on valid `NO-GO`.

- [ ] **Step 5: Implement Gate B collection**

Generate or validate three sealed noise bundles for noise seeds 1401/1402/1403 from the same private-source and image-store hashes. Execute doses in increasing order per seed and seeds serially. Use dose 1 and 1250 as gate anchors; retain doses 8/64 only for the transition curve. Write `GATE_B_DECISION.json` atomically.

- [ ] **Step 6: Add resource accounting and recovery tests**

Sum bundle wall-clock seconds, enforce source/contract equality, require exact 24 summaries for a complete Gate B, and preserve valid dose outputs across restart. A missing job remains pending; it is not counted as scientific failure.

- [ ] **Step 7: Run tests and lint**

Run: `uv run --locked pytest tests/test_causal_transport_orchestrator.py tests/test_causal_transport_gate.py tests/test_causal_transport_bundle.py -q`

Expected: all tests pass.

Run: `uv run --locked ruff check experiments/causal_transport_orchestrator.py tests/test_causal_transport_orchestrator.py`

Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add experiments/causal_transport_orchestrator.py tests/test_causal_transport_orchestrator.py
git commit -m "feat: orchestrate causal transport gates"
```

### Task 5: Add the CUDA sentinel

**Files:**
- Create: `experiments/causal_transport_sentinel.py`
- Modify: `tests/test_causal_transport_bundle.py`

**Interfaces:**
- Produces: `run_sentinel(args) -> dict[str, object]` and `CAUSAL_TRANSPORT_SENTINEL_PASS.json`.

- [ ] **Step 1: Write failing sentinel contract tests**

Require checks named `paired_images_exact`, `labels_only_difference`, `optimizer_clock_match`, `factorial_carriers_exact`, `deterministic_replay`, `resume_atomic`, `nonfinite_rejected`, and `test_isolated`.

- [ ] **Step 2: Implement a tiny CUDA bundle**

Use two warm-up steps, dose one, two cached continuation batches, and a tiny model. Hash both executions and require exact equality. Record peak VRAM, elapsed seconds, device name, total VRAM, source commit, lock hash, and `test_loaded=false`.

- [ ] **Step 3: Enforce resource and stale-pass rules**

Delete only the sentinel pass path before validation. Pass only with CUDA, all functional checks true, peak VRAM below 8 GiB, and elapsed time below 300 seconds. Write atomically.

- [ ] **Step 4: Run local tests and lint**

Run: `uv run --locked pytest tests/test_causal_transport_bundle.py -q`

Expected: all tests pass without requiring local CUDA.

Run: `uv run --locked ruff check experiments/causal_transport_sentinel.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add experiments/causal_transport_sentinel.py tests/test_causal_transport_bundle.py
git commit -m "feat: gate causal transport CUDA execution"
```

### Task 6: Conditional CIFAR-100N Gate C adapter

**Files:**
- Create only after Gate B `GO`: `experiments/cifar100n_transport_data.py`
- Create only after Gate B `GO`: `tests/test_cifar100n_transport_data.py`
- Modify only after Gate B `GO`: `experiments/causal_transport_orchestrator.py`

**Interfaces:**
- Produces: `seal_cifar100n(archive: Path, output: Path, label_set: str) -> dict[str, object]`.
- Produces paired clean/human-noisy labels with the same stable sample IDs consumed by `PairedTransportDataset`.

- [ ] **Step 1: Stop unless Gate B is a provenance-valid `GO`**

Run the Gate B decision validator before downloading, extracting, or opening CIFAR-100N labels. A missing, stale, or `NO-GO` decision exits without creating Gate C files.

- [ ] **Step 2: Write failing archive and label tests**

Reject wrong archive hash, unknown label-set name, wrong length, labels outside `[0, 99]`, ID mismatch, any test array, and an output directory that already contains a different seal.

- [ ] **Step 3: Implement sealed human-noise input**

Persist only stable training IDs and the predeclared CIFAR-100N noisy-label vector. Record official URL, archive SHA-256, label-set name, clean-label SHA-256, source commit, and `test_loaded=false`. Never construct the CIFAR-100 test split.

- [ ] **Step 4: Extend the matrix with three seeds and doses 1/1250**

Gate C contains exactly `3 seeds x 2 doses x 2 continuations = 12` bundle jobs and uses the same 2/3 anchor-crossing rule as Gate B.

- [ ] **Step 5: Test, lint, and commit**

Run: `uv run --locked pytest tests/test_cifar100n_transport_data.py tests/test_causal_transport_orchestrator.py -q`

Expected: all tests pass.

Run: `uv run --locked ruff check experiments/cifar100n_transport_data.py tests/test_cifar100n_transport_data.py`

Expected: `All checks passed!`

```bash
git add experiments/cifar100n_transport_data.py tests/test_cifar100n_transport_data.py experiments/causal_transport_orchestrator.py tests/test_causal_transport_orchestrator.py
git commit -m "feat: add CIFAR-100N transport confirmation"
```

### Task 7: Regression, ARIS evidence, and Windows dispatch

**Files:**
- Create runtime artifacts on Windows only under `C:\arw-results\causal-transport-<sha>`.

**Interfaces:**
- Consumes: exact current commit, contract, sentinel, sealed stores, Gate A/B/C orchestrator.
- Produces: staged decision JSON files, driver logs, `DRIVER_EXIT.json`, and compact mirrored evidence.

- [ ] **Step 1: Run the complete local regression**

Run: `uv run --locked pytest tests/test_causal_transport_gate.py tests/test_causal_transport_bundle.py tests/test_causal_transport_orchestrator.py tests/test_causal_attribution_replay.py tests/test_causal_attribution_gate.py tests/test_m1_calibration_clean.py tests/test_m1_calibration_noisy.py -q`

Expected: all tests pass.

Run Ruff only on touched files; separately record unrelated pre-existing full-tree failures without modifying user-owned files.

- [ ] **Step 2: Refresh the ARIS novelty evidence**

Search primary sources for optimizer-state transplantation, causal breaks/reset, training-memory witnesses, state-aware unlearning, and noisy-label PEFT through 2026-08-11. Freeze closest-work URLs and the irreducible difference in `NOVELTY_AUDIT.json`; unverified papers remain marked unverified.

- [ ] **Step 3: Push and deploy an exact detached commit**

Require a clean tracked worktree, push the branch, fetch on Windows, detach at the exact 40-character SHA, and verify `git rev-parse HEAD` before CUDA execution.

- [ ] **Step 4: Run the CUDA sentinel**

Run with `C:\arw-m0\.venv\Scripts\python.exe`. Do not register the formal task unless the current-commit sentinel pass exists and all functional/resource fields are true.

- [ ] **Step 5: Register one recoverable scheduled task**

Create `ARW-CausalTransport` with `MultipleInstances=IgnoreNew`, `ExecutionTimeLimit=P2D`, CPU affinity `4095`, direct native Python invocation, stdout/stderr append logs, and `ErrorActionPreference=Continue` only around the native process. The driver writes the exact native exit code to `DRIVER_EXIT.json`.

- [ ] **Step 6: Execute staged gates without user prompts**

Run Gate A. On valid `NO-GO`, stop scientifically with exit code 2. On `GO`, continue Gate B automatically. Only Gate B `GO` authorizes Task 6 and Gate C. Infrastructure failure resumes from valid dose artifacts; it never changes scientific thresholds.

- [ ] **Step 7: Mirror compact evidence and update the plan**

Copy decisions, summaries, manifests, hashes, novelty audit, temperatures, and wall-clock records to the Mac. Leave raw snapshots and trajectories on Windows. Mark the active stage complete only when its terminal artifact is provenance-valid.

---

## Execution choice

The user explicitly selected direct inline execution and asked for no further question/answer checkpoints. Implement tasks sequentially in this session. Do not spawn subagents. Stop only on a scientific gate decision or a genuine authority/data blocker.
