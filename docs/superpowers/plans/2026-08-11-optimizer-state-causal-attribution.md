# Optimizer-State Causal Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and dispatch a fail-closed ten-day evidence gate that reuses M0.6/M1/M1.1 artifacts and adds only the checkpoint captures and 2x2 parameter/state replay interventions needed to attribute the noisy-label PEFT performance gap.

**Architecture:** Existing clean/noisy runners gain opt-in epoch checkpoint archives without changing their default behavior. The noisy runner also gains an explicit, hash-pinned reuse path for the already-sealed M1 seed-301/302 inputs; its default current-commit rule remains unchanged. A pure evidence module validates legacy artifacts and freezes one pre-GPU prediction; a replay module validates and crosses clean/noisy parameters with clean/noisy optimizer state under cached common continuations; a gate module orchestrates the exact matrix and writes one mechanical decision. A separate CUDA sentinel validates checkpoint compatibility and deterministic branch replay before the orchestrator may start.

**Tech Stack:** Python 3.12, PyTorch/CUDA, torchvision CIFAR-100, transformers/PEFT ViT-LoRA, NumPy, pytest, Ruff, PowerShell scheduled tasks on Windows.

## Global Constraints

- One Windows RTX 5080; Windows Python is always `C:\arw-m0\.venv\Scripts\python.exe` and never `uv run --locked`.
- Windows CPU affinity is `0xFFF` (12 cores); GPU power is unrestricted.
- At most four AdamW capture jobs, twelve mandatory replay bundles, and four conditional CAdam replay bundles.
- Expected new GPU use is under four hours; hard ceiling is eight hours including one infrastructure retry.
- Seeds are exactly `301` and `302`; AdamW capture checkpoints are exactly epochs `1`, `2`, and `3`.
- Primary replay is the frozen noisy continuation through horizon `512`; secondary clean continuation reports clean-loss-excess AUC through `128`.
- No development, confirmation, or official test split is loaded; every artifact records `test_loaded=false`.
- Existing M0.6, M1, M1.1, and epoch-20 CAdam artifacts are immutable inputs; failed candidates remain failed.
- Reuse the existing `ad1da682...` seed-301/302 public noise/image/tuning stores only when all four manifest hashes exactly match their frozen M1 `input_binding.json` records; do not regenerate noise.
- No threshold sweep, extra seed, longer continuation, new optimizer, or favorable-noise search is authorized.
- CUDA evidence requires a clean tracked worktree and exact current source commit.

---

## File map

- Modify `experiments/m1_calibration_clean.py`: opt-in epoch checkpoint archival shared by clean and noisy runners.
- Modify `experiments/m1_calibration_noisy.py`: call the shared archive validation, preserve archived checkpoints in manifests, and add an explicit hash-pinned legacy-input reuse path.
- Modify `tests/test_m1_calibration_clean.py`: default/archived checkpoint behavior, invalid configuration, and fail-closed legacy-input binding tests.
- Create `experiments/causal_attribution_evidence.py`: legacy inventory, SHA binding, and frozen prediction.
- Create `tests/test_causal_attribution_evidence.py`: evidence validation and prediction rule tests.
- Create `experiments/causal_attribution_replay.py`: checkpoint validation, CC/CN/NC/NN branch construction, cached continuations, metrics, and artifact writing.
- Create `tests/test_causal_attribution_replay.py`: atomic transplant, incompatibility rejection, deterministic replay, and isolation tests.
- Create `experiments/causal_attribution_sentinel.py`: current-commit CUDA functional/resource sentinel.
- Create `experiments/causal_attribution_gate.py`: exact capture/replay orchestration and mechanical GO/NO-GO evaluator.
- Create `tests/test_causal_attribution_gate.py`: matrix, orchestration, stale-result, gate, and conditional CAdam tests.

---

### Task 1: Opt-in epoch checkpoint capture and sealed-input reuse

**Files:**
- Modify: `experiments/m1_calibration_clean.py`
- Modify: `experiments/m1_calibration_noisy.py`
- Test: `tests/test_m1_calibration_clean.py`

**Interfaces:**
- Consumes: existing `Config`, `_checkpoint`, clean/noisy `run`, noisy runner reuse of `clean.Config`, and frozen baseline `input_binding.json` files.
- Produces: `Config.checkpoint_epochs: tuple[int, ...]`, `_validate_checkpoint_epochs(config)`, `checkpoint-epoch{N}.pt` archives with the unchanged checkpoint schema, and noisy `_validate_input_provenance(..., expected_input_binding)` with a strict default and exact opt-in reuse.

- [ ] **Step 1: Write failing archive and validation tests**

Add a small direct checkpoint test plus one tiny-run assertion:

```python
from pathlib import Path

import pytest

from experiments import m1_calibration_noisy as noisy


def test_checkpoint_epoch_validation_is_exact() -> None:
    valid = m1.Config(
        seed=301,
        augmentation_seed=10_301,
        epochs=3,
        checkpoint_epochs=(1, 2, 3),
    )
    assert m1._validate_checkpoint_epochs(valid) == (1, 2, 3)

    for epochs in ((2, 1), (1, 1), (0,), (4,)):
        invalid = m1.Config(
            seed=301,
            augmentation_seed=10_301,
            epochs=3,
            checkpoint_epochs=epochs,
        )
        with pytest.raises(ValueError, match="checkpoint_epochs"):
            m1._validate_checkpoint_epochs(invalid)


def test_checkpoint_archives_do_not_replace_resume_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_checkpoint(path: Path, *args: object) -> None:
        calls.append(path.name)
        path.write_bytes(path.name.encode())

    monkeypatch.setattr(m1, "_checkpoint", fake_checkpoint)
    config = m1.Config(
        seed=301,
        augmentation_seed=10_301,
        epochs=3,
        checkpoint_epochs=(1, 3),
    )
    m1._write_epoch_checkpoints(
        tmp_path, object(), object(), 1, m1._validate_checkpoint_epochs(config)
    )
    assert calls == ["checkpoint.pt", "checkpoint-epoch1.pt"]


def test_legacy_input_reuse_requires_exact_frozen_binding() -> None:
    public = {
        "source_commit": "a" * 40,
        "uv_lock_sha256": "old-lock",
        "test_loaded": False,
    }
    current = {"source_commit": "b" * 40, "uv_lock_sha256": "new-lock"}
    actual = {
        "noise_bundle_sha256": "1" * 64,
        "image_store_sha256": "2" * 64,
        "tuning_store_sha256": "3" * 64,
        "source_commit": "a" * 40,
        "test_loaded": False,
    }
    assert noisy._validate_input_provenance(public, current, actual, dict(actual)) == {
        "runner_source_commit": "b" * 40,
        "input_source_commit": "a" * 40,
        "reused_sealed_input": True,
    }
    forged = {**actual, "noise_bundle_sha256": "9" * 64}
    with pytest.raises(RuntimeError, match="frozen input binding"):
        noisy._validate_input_provenance(public, current, actual, forged)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
uv run --locked pytest tests/test_m1_calibration_clean.py -q
```

Expected: FAIL because `Config.checkpoint_epochs`, checkpoint helpers, and `_validate_input_provenance` do not exist.

- [ ] **Step 3: Add the opt-in archive implementation**

Add the field and helpers to `experiments/m1_calibration_clean.py`:

```python
@dataclass(frozen=True)
class Config:
    checkpoint_epochs: tuple[int, ...] = ()


def _validate_checkpoint_epochs(config: Config) -> tuple[int, ...]:
    values = config.checkpoint_epochs
    if tuple(sorted(set(values))) != values or any(
        isinstance(epoch, bool) or not 1 <= epoch <= config.epochs for epoch in values
    ):
        raise ValueError("checkpoint_epochs must be unique increasing epochs in [1, epochs]")
    return values


def _write_epoch_checkpoints(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    archive_epochs: tuple[int, ...],
) -> None:
    _checkpoint(output_dir / "checkpoint.pt", model, optimizer, next_epoch)
    if next_epoch in archive_epochs:
        _checkpoint(
            output_dir / f"checkpoint-epoch{next_epoch}.pt",
            model,
            optimizer,
            next_epoch,
        )
```

At the start of both clean and noisy `run`, call `_validate_checkpoint_epochs(config)`. Replace each direct `_checkpoint(checkpoint_path, ...)` call with `_write_epoch_checkpoints(output_dir, ...)`. Add an appendable CLI option to both entry points:

```python
parser.add_argument("--checkpoint-epoch", type=int, action="append", default=[])
# Config construction
checkpoint_epochs=tuple(args.checkpoint_epoch),
```

Do not alter `_checkpoint` or the default empty tuple; legacy behavior must remain byte-schema compatible.

- [ ] **Step 4: Add the explicit sealed-input reuse path**

Keep current behavior when no frozen binding is passed. Add a keyword-only `expected_input_binding: dict[str, object] | None = None` to noisy `run`. Compute the actual three manifest hashes before training and validate them as follows:

```python
def _validate_input_provenance(
    public: dict[str, object],
    current: dict[str, object],
    actual_binding: dict[str, object],
    expected_input_binding: dict[str, object] | None,
) -> dict[str, object]:
    if expected_input_binding is None:
        if current["source_commit"] != public["source_commit"]:
            raise RuntimeError("runner/noise source commit mismatch")
        if current["uv_lock_sha256"] != public["uv_lock_sha256"]:
            raise RuntimeError("runner/noise lock mismatch")
        return {
            "runner_source_commit": current["source_commit"],
            "input_source_commit": public["source_commit"],
            "reused_sealed_input": False,
        }
    required = {
        "noise_bundle_sha256",
        "image_store_sha256",
        "tuning_store_sha256",
        "source_commit",
        "test_loaded",
    }
    if set(expected_input_binding) != required or expected_input_binding != actual_binding:
        raise RuntimeError("frozen input binding mismatch")
    if expected_input_binding["source_commit"] != public["source_commit"]:
        raise RuntimeError("frozen input binding source mismatch")
    if expected_input_binding["test_loaded"] is not False or public["test_loaded"] is not False:
        raise RuntimeError("test isolation flag failed")
    return {
        "runner_source_commit": current["source_commit"],
        "input_source_commit": public["source_commit"],
        "reused_sealed_input": True,
    }
```

The gate loads the expected binding only from the corresponding immutable M1 AdamW noisy baseline directory. The new capture `input_binding.json` records the actual hashes, old input source commit, current runner source commit, and `reused_sealed_input=true`. Add noisy CLI option `--expected-input-binding`, load it as JSON, and pass it to `run`. No code path may rewrite a reused store or its manifest.

- [ ] **Step 5: Run clean/noisy runner tests**

Run:

```bash
uv run --locked pytest tests/test_m1_calibration_clean.py tests/test_m1_pilot_runner.py tests/test_m11_rescue.py -q
uv run --locked ruff check experiments/m1_calibration_clean.py experiments/m1_calibration_noisy.py tests/test_m1_calibration_clean.py
```

Expected: all tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 6: Commit Task 1**

```bash
git add experiments/m1_calibration_clean.py experiments/m1_calibration_noisy.py tests/test_m1_calibration_clean.py
git commit -m "feat: archive causal attribution checkpoints"
```

---

### Task 2: Immutable legacy evidence inventory and prediction

**Files:**
- Create: `experiments/causal_attribution_evidence.py`
- Create: `tests/test_causal_attribution_evidence.py`

**Interfaces:**
- Consumes: `m06_root`, `m1_root`, `m11_decision`, and SHA-bearing artifact files.
- Produces: `inventory_legacy(...) -> dict[str, object]`, `derive_prediction(inventory) -> dict[str, object]`, `write_legacy_evidence(...)`, `LEGACY_EVIDENCE_INVENTORY.json`, and `LEGACY_EVIDENCE_PREDICTION.json`.

- [ ] **Step 1: Write failing fixture-based evidence tests**

Create fixtures with exactly 20 M0.6 summaries and the two frozen decisions:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import causal_attribution_evidence as evidence


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _legacy_fixture(root: Path, parameter_auc: float, state_auc: float) -> tuple[Path, Path, Path]:
    m06 = root / "m06"
    for index in range(20):
        cell = m06 / f"train{index + 3}-pulse161803"
        _write_json(
            cell / "summary.json",
            {
                "status": "succeeded",
                "bundle_id": f"m06-{index}",
                "branches": {
                    "parameter_only": {"clean_loss_excess_auc_128": parameter_auc},
                    "state_both": {"clean_loss_excess_auc_128": state_auc},
                },
                "source_commit": "a" * 40,
                "test_loaded": False,
            },
        )
        _write_json(cell / "sha256_manifest.json", [])
    _write_json(m06 / "COMPLETE.json", {"status": "succeeded", "completed_bundles": 20})
    m1 = root / "m1"
    _write_json(
        m1 / "M1_PILOT_DECISION.json",
        {"status": "succeeded", "decision": "NO-GO", "source_commit": "b" * 40, "test_loaded": False},
    )
    m11 = root / "M11_RESCUE_DECISION.json"
    _write_json(
        m11,
        {"status": "succeeded", "decision": "NO-GO", "source_commit": "c" * 40, "test_loaded": False},
    )
    return m06, m1, m11


@pytest.mark.parametrize(
    ("parameter_auc", "state_auc", "dominant"),
    [(4.0, 1.0, "parameter"), (1.0, 4.0, "state"), (1.0, 1.5, "interaction")],
)
def test_prediction_uses_frozen_twofold_rule(
    tmp_path: Path, parameter_auc: float, state_auc: float, dominant: str
) -> None:
    m06, m1, m11 = _legacy_fixture(tmp_path, parameter_auc, state_auc)
    inventory = evidence.inventory_legacy(m06, m1, m11)
    prediction = evidence.derive_prediction(inventory)
    assert prediction["checkpoint_predictions"] == {"1": dominant, "2": dominant, "3": dominant}
    assert prediction["test_loaded"] is False


def test_inventory_fails_closed_on_wrong_bundle_count(tmp_path: Path) -> None:
    m06, m1, m11 = _legacy_fixture(tmp_path, 2.0, 1.0)
    (m06 / "train3-pulse161803" / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="20 complete M0.6 bundles"):
        evidence.inventory_legacy(m06, m1, m11)
```

- [ ] **Step 2: Run tests and confirm import failure**

```bash
uv run --locked pytest tests/test_causal_attribution_evidence.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement strict inventory and prediction functions**

The core rule must be deterministic and use only existing M0.6 branch AUC values:

```python
def derive_prediction(inventory: dict[str, object]) -> dict[str, object]:
    m06 = inventory["m06"]
    if not isinstance(m06, dict):
        raise RuntimeError("M0.6 inventory is malformed")
    parameter = statistics.median(abs(float(value)) for value in m06["parameter_auc_128"])
    state = statistics.median(abs(float(value)) for value in m06["state_auc_128"])
    if parameter >= 2 * state:
        dominant = "parameter"
    elif state >= 2 * parameter:
        dominant = "state"
    else:
        dominant = "interaction"
    return {
        "rule": "median-absolute-auc-twofold-v1",
        "parameter_median_abs_auc_128": parameter,
        "state_median_abs_auc_128": state,
        "checkpoint_predictions": {str(epoch): dominant for epoch in (1, 2, 3)},
        "source_artifact_sha256": inventory["inventory_sha256"],
        "test_loaded": False,
    }
```

`inventory_legacy` must validate every summary as finite, `status=succeeded`, `test_loaded=false`, unique bundle ID, and SHA-manifest present; validate exact M1/M1.1 `NO-GO` decisions; and include SHA-256 for every consumed file. `write_legacy_evidence` must write both JSON files atomically with `clean._write_json` and refuse to overwrite a prediction whose source hash differs.

- [ ] **Step 4: Add the CLI and run focused checks**

The CLI accepts exact roots and one output directory:

```python
parser.add_argument("--m06-root", type=Path, required=True)
parser.add_argument("--m1-root", type=Path, required=True)
parser.add_argument("--m11-decision", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
```

Run:

```bash
uv run --locked pytest tests/test_causal_attribution_evidence.py -q
uv run --locked ruff check experiments/causal_attribution_evidence.py tests/test_causal_attribution_evidence.py
```

Expected: tests pass and Ruff passes.

- [ ] **Step 5: Commit Task 2**

```bash
git add experiments/causal_attribution_evidence.py tests/test_causal_attribution_evidence.py
git commit -m "feat: freeze legacy causal evidence"
```

---

### Task 3: Crossed checkpoint replay engine

**Files:**
- Create: `experiments/causal_attribution_replay.py`
- Create: `tests/test_causal_attribution_replay.py`

**Interfaces:**
- Consumes: clean/noisy checkpoint paths, their config/provenance files, one replay regime (`clean` or `noisy`), data stores, seed, checkpoint epoch, method, and output root.
- Produces: `CheckpointState`, `ReplayRequest`, `load_checkpoint_state`, `validate_checkpoint_pair`, `build_factorial_branches`, `factorial_effects`, `capture_common_rng`, `measure_branch`, `run_cached_branches`, `run_replay`, `trajectory_metrics.jsonl`, `factorial_effects.json`, `replay_manifest.json`, and `summary.json`.

- [ ] **Step 1: Write pure crossed-state and fail-closed tests**

Use a tiny FP32 model and initialized AdamW states:

```python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments import causal_attribution_replay as replay


def _state(seed: int, step: int = 2) -> replay.CheckpointState:
    torch.manual_seed(seed)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
    for _ in range(step):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
    return replay.CheckpointState(
        parameters=deepcopy(model.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()),
        next_epoch=1,
    )


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_factorial_branches_cross_only_requested_carriers() -> None:
    clean = _state(1)
    noisy = _state(2)
    branches = replay.build_factorial_branches(clean, noisy)
    assert tuple(branches) == ("CC", "CN", "NC", "NN")
    assert all(
        torch.equal(branches["CN"].parameters[key], clean.parameters[key])
        for key in clean.parameters
    )
    assert all(
        torch.equal(branches["NC"].parameters[key], noisy.parameters[key])
        for key in noisy.parameters
    )
    _assert_nested_equal(branches["CN"].optimizer, noisy.optimizer)
    _assert_nested_equal(branches["NC"].optimizer, clean.optimizer)


def test_checkpoint_pair_rejects_step_mismatch_atomically() -> None:
    clean = _state(1, step=2)
    noisy = _state(2, step=3)
    with pytest.raises(ValueError, match="step counters"):
        replay.validate_checkpoint_pair(clean, noisy)


def test_factorial_effect_formula() -> None:
    effects = replay.factorial_effects({"CC": 1.0, "CN": 2.0, "NC": 4.0, "NN": 8.0})
    assert effects == {"parameter": 4.5, "state": 2.5, "interaction": 3.0}
```

- [ ] **Step 2: Run tests and confirm module failure**

```bash
uv run --locked pytest tests/test_causal_attribution_replay.py -q
```

Expected: FAIL because the replay module does not exist.

- [ ] **Step 3: Implement checkpoint validation and branch construction**

Use immutable dataclasses and deep copies:

```python
@dataclass(frozen=True)
class CheckpointState:
    parameters: dict[str, Tensor]
    optimizer: dict[str, Any]
    next_epoch: int


@dataclass(frozen=True)
class ReplayRequest:
    clean_checkpoint: Path
    noisy_checkpoint: Path
    continuation: Literal["clean", "noisy"]
    seed: Literal[301, 302]
    checkpoint_epoch: Literal[1, 2, 3, 20]
    method: Literal["adamw", "cadam"]
    data_dir: Path
    image_store: Path
    tuning_store: Path
    noisy_bundle: Path
    output: Path
    source_commit: str


def build_factorial_branches(
    clean_state: CheckpointState, noisy_state: CheckpointState
) -> dict[str, BranchState]:
    validate_checkpoint_pair(clean_state, noisy_state)
    return {
        "CC": BranchState(deepcopy(clean_state.parameters), deepcopy(clean_state.optimizer), 0.0, 0.0),
        "CN": BranchState(deepcopy(clean_state.parameters), deepcopy(noisy_state.optimizer), 0.0, 0.0),
        "NC": BranchState(deepcopy(noisy_state.parameters), deepcopy(clean_state.optimizer), 0.0, 0.0),
        "NN": BranchState(deepcopy(noisy_state.parameters), deepcopy(noisy_state.optimizer), 0.0, 0.0),
    }


def factorial_effects(values: dict[str, float]) -> dict[str, float]:
    if set(values) != {"CC", "CN", "NC", "NN"} or any(
        not math.isfinite(value) for value in values.values()
    ):
        raise ValueError("factorial values must contain four finite branches")
    return {
        "parameter": 0.5 * ((values["NC"] - values["CC"]) + (values["NN"] - values["CN"])),
        "state": 0.5 * ((values["CN"] - values["CC"]) + (values["NN"] - values["NC"])),
        "interaction": values["NN"] - values["NC"] - values["CN"] + values["CC"],
    }
```

`validate_checkpoint_pair` must compare exact model keys, tensor shape/dtype, optimizer parameter groups, parameter-state IDs and keys, moment shape/dtype, and every `step` counter. It must reject booleans, non-finite tensors, empty states, mismatched `next_epoch`, or checkpoints whose top-level keys differ from `model`, `optimizer`, and `next_epoch`.

`load_checkpoint_state(model, checkpoint_path)` loads the full model state strictly into the just-built model, then stores only `m0.capture_trainable_state(model)` in `CheckpointState.parameters`; frozen base weights and buffers are validated by strict loading but are not passed to `m0.restore_trainable_state`. It deep-copies the optimizer state and integer `next_epoch`. This filtering is required because `BranchState.parameters` is a trainable-parameter snapshot, whereas the runner checkpoint contains the full `model.state_dict()`.

- [ ] **Step 4: Write deterministic replay tests before the runner**

Monkeypatch model/data builders with tiny objects. Assert:

```python
def test_common_replay_core_is_bitwise_deterministic() -> None:
    clean = _state(1)
    noisy = _state(2)
    branches = replay.build_factorial_branches(clean, noisy)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
    cached = [
        (torch.tensor([[1.0, -1.0]]), torch.tensor([0])),
        (torch.tensor([[0.5, 2.0]]), torch.tensor([1])),
    ]
    criterion = torch.nn.CrossEntropyLoss()

    def measure(branch: str, horizon: int) -> dict[str, object]:
        total = sum(float(parameter.detach().sum()) for parameter in model.parameters())
        return {"branch": branch, "horizon": horizon, "parameter_sum": total, "test_loaded": False}

    rng = replay.capture_common_rng()
    first = replay.run_cached_branches(
        model, optimizer, branches, cached, criterion, rng, (0, 1, 2), measure
    )
    second = replay.run_cached_branches(
        model, optimizer, branches, cached, criterion, rng, (0, 1, 2), measure
    )
    assert first == second
    assert {row["branch"] for row in first} == {"CC", "CN", "NC", "NN"}
    assert all(row["test_loaded"] is False for row in first)
```

Expected first run: FAIL because `run_cached_branches` is missing.

- [ ] **Step 5: Implement cached common continuation and metrics**

Implement `ReplayConfig` with exact allowed values and use the existing clean/noisy dataset classes. Starting at an epoch boundary, build one deterministic loader for `next_epoch`, cache exactly 512 CPU batches once, and replay the same batch tensors through CC, CN, NC, and NN after restoring a shared RNG snapshot.

`capture_common_rng` is a thin public wrapper around `m0.capture_rng_state`. `measure_branch` accepts the branch name, horizon, restored model/optimizer, and CC references, and returns only finite JSON-native metrics with `test_loaded=false`. `run_cached_branches` implements the loop below and takes a measurement callback so the tiny deterministic unit test does not instantiate ViT/PEFT.

Record probe metrics at the frozen horizons and full tuning loss/accuracy at horizon 512. CC runs first and supplies reference parameters, moments, gradient, predictions, and probe loss. Other branches record `d_theta`, `d_m`, `d_v`, gradient cosine, probe loss excess, and Jensen-Shannon divergence to CC. Use existing `arw.m0_core` capture/restore/distance functions rather than reimplementing tensor traversal.

The core branch loop must have this shape:

```python
for branch_name in ("CC", "CN", "NC", "NN"):
    restore_branch(model, optimizer, branches[branch_name])
    m0.restore_rng_state(common_rng)
    for horizon in range(513):
        if horizon in HORIZONS:
            rows.append(measure_branch(branch_name, horizon, model, optimizer, references))
        if horizon < 512:
            images, labels = cached_batches[horizon]
            m0.training_step(model, optimizer, criterion, images.to(device), labels.to(device))
    tuning_loss, tuning_accuracy = clean._evaluate(model, tuning_loader, device)
```

All JSON writes are atomic; rerunning a completed bundle validates its source hash and returns without recomputation. A partial bundle without a valid summary fails rather than silently appending.

- [ ] **Step 6: Run replay tests, regression, and lint**

```bash
uv run --locked pytest tests/test_causal_attribution_replay.py tests/test_m0_core.py tests/test_m0_runner.py -q
uv run --locked ruff check experiments/causal_attribution_replay.py tests/test_causal_attribution_replay.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add experiments/causal_attribution_replay.py tests/test_causal_attribution_replay.py
git commit -m "feat: add crossed optimizer-state replay"
```

---

### Task 4: CUDA sentinel, exact orchestrator, and decision gate

**Files:**
- Create: `experiments/causal_attribution_sentinel.py`
- Create: `experiments/causal_attribution_gate.py`
- Create: `tests/test_causal_attribution_gate.py`

**Interfaces:**
- Consumes: current source commit, sentinel pass, legacy inventory/prediction, hash-pinned sealed M1 noise stores for seeds 301/302, existing M1/CAdam roots, and the replay engine.
- Produces: `build_mandatory_matrix`, `validate_inputs`, `evaluate_attribution(record, *, require_cadam)`, `run_capture_cell`, `run_replay_cell`, `collect_record`, `run_gate`, `CAUSAL_ATTRIBUTION_SENTINEL_PASS.json`, `ADAMW_ATTRIBUTION_DECISION.json`, and final `CAUSAL_ATTRIBUTION_DECISION.json`.

- [ ] **Step 1: Write exact-matrix and pure decision tests**

```python
def passing_attribution_record(dominant: str) -> dict[str, object]:
    if dominant == "parameter":
        values = {"parameter": 4.0, "state": 1.0, "interaction": 0.5}
    elif dominant == "state":
        values = {"parameter": 1.0, "state": 4.0, "interaction": 0.5}
    else:
        values = {"parameter": 1.0, "state": 1.0, "interaction": 2.0}
    return {
        "mandatory_artifacts": 12,
        "effects": {
            str(seed): {str(epoch): dict(values) for epoch in (1, 2, 3)}
            for seed in (301, 302)
        },
        "prediction": {"checkpoint_predictions": {str(epoch): dominant for epoch in (1, 2, 3)}},
        "cadam_confirmation": {"301": dominant, "302": dominant},
        "gpu_hours": 2.0,
        "test_loaded": False,
    }


def test_mandatory_matrix_is_exact() -> None:
    cells = gate.build_mandatory_matrix()
    assert len(cells) == 12
    assert {(cell.seed, cell.epoch, cell.continuation) for cell in cells} == {
        (seed, epoch, continuation)
        for seed in (301, 302)
        for epoch in (1, 2, 3)
        for continuation in ("clean", "noisy")
    }


def test_gate_requires_replicated_twofold_dominance_and_prediction_match() -> None:
    record = passing_attribution_record(dominant="parameter")
    result = gate.evaluate_attribution(record, require_cadam=True)
    assert result["decision"] == "GO"
    assert all(result["checks"].values())

    record["effects"]["302"]["2"] = {"parameter": 0.5, "state": 1.0, "interaction": 0.2}
    assert gate.evaluate_attribution(record, require_cadam=True)["decision"] == "NO-GO"


def test_cadam_confirmation_is_required_after_adamw_go() -> None:
    record = passing_attribution_record(dominant="state")
    record["cadam_confirmation"] = {"301": "parameter", "302": "state"}
    assert gate.evaluate_attribution(record, require_cadam=True)["decision"] == "NO-GO"
```

- [ ] **Step 2: Write orchestration and stale-artifact tests**

Monkeypatch capture and replay calls. Assert the order is evidence -> sentinel validation -> four captures -> twelve AdamW replays -> AdamW decision -> zero or four CAdam replays -> final decision. Assert `CAUSAL_ATTRIBUTION_DECISION.json` is unlinked before any validation so a failed rerun cannot leave a stale pass.

```python
def test_orchestrator_runs_exact_jobs_and_removes_stale_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    output = tmp_path / "output"
    args = argparse.Namespace(
        data_dir=tmp_path / "data",
        image_store=tmp_path / "images",
        tuning_store=tmp_path / "tuning",
        bundle_301=tmp_path / "bundle301",
        bundle_302=tmp_path / "bundle302",
        legacy_m06_root=tmp_path / "m06",
        legacy_m1_root=tmp_path / "m1",
        legacy_m11_decision=tmp_path / "m11.json",
        sentinel=tmp_path / "sentinel.json",
        output=output,
    )
    monkeypatch.setattr(gate, "source_commit", lambda: "c" * 40)

    def fake_validate(*_: object) -> dict[str, object]:
        calls.append("validate")
        return passing_attribution_record("parameter")

    monkeypatch.setattr(gate, "validate_inputs", fake_validate)
    monkeypatch.setattr(gate, "run_capture_cell", lambda *_: calls.append("capture"))

    def fake_replay(cell: object, *_: object) -> None:
        calls.append("cadam-replay" if getattr(cell, "method") == "cadam" else "adamw-replay")

    monkeypatch.setattr(gate, "run_replay_cell", fake_replay)
    monkeypatch.setattr(
        gate,
        "collect_record",
        lambda *_: passing_attribution_record("parameter"),
    )
    stale = args.output / "CAUSAL_ATTRIBUTION_DECISION.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"decision":"GO"}')
    result = gate.run_gate(args)
    assert result["decision"] == "GO"
    assert calls[0] == "validate"
    assert calls[1:5] == ["capture"] * 4
    assert calls[5:17] == ["adamw-replay"] * 12
    assert calls[17:] == ["cadam-replay"] * 4
    assert calls.count("capture") == 4
    assert calls.count("adamw-replay") == 12
    assert calls.count("cadam-replay") == 4
    assert json.loads(stale.read_text())["source_commit"] == "c" * 40
```

- [ ] **Step 3: Run tests and confirm missing modules**

```bash
uv run --locked pytest tests/test_causal_attribution_gate.py -q
```

Expected: FAIL because sentinel and gate modules do not exist.

- [ ] **Step 4: Implement the mechanical evaluator**

For each seed/checkpoint primary noisy-continuation bundle, choose the largest absolute effect among `parameter`, `state`, and `interaction`. A main effect is eligible only when its magnitude is at least twice the other main effect; interaction is eligible only when its magnitude is at least each main effect. Require the same eligible dominant component in both seeds at at least two checkpoints and equality with the frozen prediction at those checkpoints.

Return named checks and never mutate the input:

```python
checks = {
    "mandatory_complete": mandatory_complete,
    "finite_and_isolated": finite_and_isolated,
    "replicated_at_two_checkpoints": replicated_at_two_checkpoints,
    "twofold_or_interaction_rule": effect_rule_passed,
    "legacy_prediction_matches": prediction_matches,
    "cadam_matches_when_required": cadam_matches,
    "gpu_hours_at_most_8": 0 < gpu_hours <= 8.0,
}
return {**record, "checks": checks, "decision": "GO" if all(checks.values()) else "NO-GO"}
```

- [ ] **Step 5: Implement the CUDA sentinel**

The sentinel must remove its old pass file, require CUDA and a clean current commit, validate one real clean/noisy checkpoint pair, construct four branches, run eight cached steps twice from the same RNG, verify bitwise deterministic model and optimizer outcomes, require finite metrics, and record peak VRAM plus elapsed seconds. It writes `CAUSAL_ATTRIBUTION_SENTINEL_RESULT.json` always and the pass artifact only on success.

Required functional keys are:

```python
required = {
    "checkpoint_schema",
    "step_clock_match",
    "factorial_carriers_exact",
    "deterministic_replay",
    "nonfinite_rejected",
    "test_isolated",
}
```

- [ ] **Step 6: Implement exact capture/replay orchestration**

`run_gate` performs four current-commit AdamW capture cells with the exact frozen M1 hyperparameters: `epochs=3`, `checkpoint_epochs=(1, 2, 3)`, `batch_size=32`, `learning_rate=3e-4`, `weight_decay=0.1`, `betas=(0.9, 0.999)`, `split_seed=20260801`, `augmentation_seed=10000+seed`, `lora_rank=8`, and no gradient clipping. Each noisy capture passes the matching immutable M1 AdamW `input_binding.json` to the explicit reuse path. It then runs twelve mandatory replays and writes an AdamW decision. Only when that decision is `GO` may it replay the four existing epoch-20 CAdam pairs. The final decision is atomic and returns exit code 2 on `NO-GO`.

The CLI has exact required paths:

```python
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--image-store", type=Path, required=True)
parser.add_argument("--tuning-store", type=Path, required=True)
parser.add_argument("--bundle-301", type=Path, required=True)
parser.add_argument("--bundle-302", type=Path, required=True)
parser.add_argument("--legacy-m06-root", type=Path, required=True)
parser.add_argument("--legacy-m1-root", type=Path, required=True)
parser.add_argument("--legacy-m11-decision", type=Path, required=True)
parser.add_argument("--sentinel", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
```

- [ ] **Step 7: Run all new tests and legacy regression**

```bash
uv run --locked pytest \
  tests/test_causal_attribution_evidence.py \
  tests/test_causal_attribution_replay.py \
  tests/test_causal_attribution_gate.py \
  tests/test_m1_calibration_clean.py \
  tests/test_m1_pilot_runner.py \
  tests/test_m11_rescue.py \
  tests/test_m0_core.py \
  tests/test_m0_runner.py -q
uv run --locked ruff check \
  experiments/causal_attribution_evidence.py \
  experiments/causal_attribution_replay.py \
  experiments/causal_attribution_sentinel.py \
  experiments/causal_attribution_gate.py \
  tests/test_causal_attribution_evidence.py \
  tests/test_causal_attribution_replay.py \
  tests/test_causal_attribution_gate.py
```

Expected: all tests pass and Ruff passes.

- [ ] **Step 8: Commit Task 4**

```bash
git add experiments/causal_attribution_sentinel.py experiments/causal_attribution_gate.py tests/test_causal_attribution_gate.py
git commit -m "feat: gate causal optimizer-state attribution"
```

---

### Task 5: Final audit, exact Windows deployment, and dispatch

**Files:**
- Verify only; do not add launcher scripts or generated evidence to Git.

**Interfaces:**
- Consumes: the four implementation commits, Windows repo `C:\arw-m0`, existing roots `C:\arw-results\m06`, `C:\arw-results\m1-pilot`, and `C:\arw-results\m11-rescue-bb5282f`.
- Produces: a verified immutable-input record, sentinel pass, hidden scheduled task `ARW-CausalAttribution`, and final decision artifact under a commit-keyed result root.

- [ ] **Step 1: Run the full local gate**

```bash
uv run --locked pytest -q
uv run --locked ruff check .
git diff --check
git status --short --branch
```

Expected: all tests and Ruff pass; only known unrelated untracked user files remain.

- [ ] **Step 2: Push and deploy the exact clean commit**

```bash
git push origin codex/m05-direct-experiment
git rev-parse HEAD
```

Record the 40-character commit as `CAUSAL_SHA`. Deploy it exactly:

```powershell
Set-Location C:\arw-m0
git fetch origin codex/m05-direct-experiment
$causalSha = (git rev-parse origin/codex/m05-direct-experiment).Trim()
git checkout --detach $causalSha
if ((git status --porcelain --untracked-files=no).Length -ne 0) {
    throw 'tracked Windows worktree is dirty'
}
```

If GitHub transport fails, create and copy a bundle instead of editing Windows source files:

```bash
causal_sha="$(git rev-parse HEAD)"
git bundle create "/tmp/causal-${causal_sha}.bundle" codex/m05-direct-experiment
scp "/tmp/causal-${causal_sha}.bundle" 33135@autoresearch-5080:C:/arw-transfer/
```

```powershell
Set-Location C:\arw-m0
$bundle = Get-ChildItem C:\arw-transfer\causal-*.bundle | Sort-Object LastWriteTime -Descending | Select-Object -First 1
git fetch $bundle.FullName codex/m05-direct-experiment
$causalSha = (git rev-parse FETCH_HEAD).Trim()
git checkout --detach $causalSha
```

- [ ] **Step 3: Verify and reuse the exact seed-301/302 sealed inputs**

Validate the existing stores against both their seals and the frozen baseline bindings; do not regenerate them:

```powershell
Set-Location C:\arw-m0
$python = 'C:\arw-m0\.venv\Scripts\python.exe'
$dataRoot = 'C:\arw-data\m1-pilot-noise-ad1da68'
$expected = @{
    1301 = '6542747c61b91881c2279f20df67087add7459fd3573ddf75afa79f2729e3458'
    1302 = '8fe5cab46a646471dcfc8bbdfba589451b4edfd13e519d7aa77d28165e12d9b7'
}
foreach ($seed in 1301, 1302) {
    $validated = & $python -m experiments.m1_noisy_data validate `
      --training "$dataRoot\training\pilot-$seed" `
      --image-store "$dataRoot\image-store" `
      --tuning-store "$dataRoot\tuning-store" | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $validated.sample_count -ne 40000) {
        throw "public validation failed for $seed"
    }
    $manifestHash = (Get-FileHash -Algorithm SHA256 `
      "$dataRoot\training\pilot-$seed\manifest.json").Hash.ToLowerInvariant()
    $baselineBinding = Get-Content `
      "C:\arw-results\m1-pilot\adamw\noisy-seed$($seed - 1000)\input_binding.json" |
      ConvertFrom-Json
    if ($validated.source_commit -ne 'ad1da6820d96523c47b58ab9fa47311ef9cf17a9' `
        -or $validated.test_loaded -or $manifestHash -ne $expected[$seed] `
        -or $baselineBinding.noise_bundle_sha256 -ne $manifestHash) {
        throw "frozen provenance or isolation failed for $seed"
    }
}
if ((Get-FileHash -Algorithm SHA256 "$dataRoot\image-store\manifest.json").Hash.ToLowerInvariant() `
    -ne '260d913d4fa8efe25bb5251a0862201a2d380d3d45a7b19348280258f52fd34d') {
    throw 'frozen image-store hash mismatch'
}
if ((Get-FileHash -Algorithm SHA256 "$dataRoot\tuning-store\manifest.json").Hash.ToLowerInvariant() `
    -ne '1ee64b5c0958eef67ccc6a0fdfb7461a75a0a16e7fa024b4ae21e20e99a29333') {
    throw 'frozen tuning-store hash mismatch'
}
```

- [ ] **Step 4: Run the real CUDA sentinel before registration**

```powershell
$causalSha = (git -C C:\arw-m0 rev-parse HEAD).Trim()
$shortSha = $causalSha.Substring(0, 8)
C:\arw-m0\.venv\Scripts\python.exe -m experiments.causal_attribution_sentinel `
  --clean-checkpoint C:\arw-results\m1-pilot\adamw\clean-seed301\checkpoint.pt `
  --noisy-checkpoint C:\arw-results\m1-pilot\adamw\noisy-seed301\checkpoint.pt `
  --output "C:\arw-results\causal-sentinel-$shortSha"
if ($LASTEXITCODE -ne 0) { throw 'causal attribution sentinel failed' }
```

Expected: `CAUSAL_ATTRIBUTION_SENTINEL_PASS.json` exists, is bound to `CAUSAL_SHA`, and every functional key is true.

- [ ] **Step 5: Register and start one hidden continuous Windows task**

Write the exact commit-bound driver, start Python as a child, and apply `0xFFF` affinity to that child:

```powershell
Set-Location C:\arw-m0
$causalSha = (git rev-parse HEAD).Trim()
$shortSha = $causalSha.Substring(0, 8)
$driver = "C:\arw-run\causal-attribution-$shortSha.ps1"
$script = @'
$ErrorActionPreference = 'Stop'
Set-Location C:\arw-m0
$python = 'C:\arw-m0\.venv\Scripts\python.exe'
$causalSha = (git rev-parse HEAD).Trim()
$shortSha = $causalSha.Substring(0, 8)
$dataRoot = 'C:\arw-data\m1-pilot-noise-ad1da68'
$sentinel = "C:\arw-results\causal-sentinel-$shortSha\CAUSAL_ATTRIBUTION_SENTINEL_PASS.json"
$output = "C:\arw-results\causal-attribution-$shortSha"
New-Item -ItemType Directory -Force -Path $output | Out-Null
$arguments = @(
    '-m', 'experiments.causal_attribution_gate',
    '--data-dir', 'C:\arw-data\cifar100',
    '--image-store', "$dataRoot\image-store",
    '--tuning-store', "$dataRoot\tuning-store",
    '--bundle-301', "$dataRoot\training\pilot-1301",
    '--bundle-302', "$dataRoot\training\pilot-1302",
    '--legacy-m06-root', 'C:\arw-results\m06',
    '--legacy-m1-root', 'C:\arw-results\m1-pilot',
    '--legacy-m11-decision', 'C:\arw-results\m11-rescue-bb5282f\M11_RESCUE_DECISION.json',
    '--sentinel', $sentinel,
    '--output', $output
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -PassThru `
    -RedirectStandardOutput "$output\driver.stdout.log" `
    -RedirectStandardError "$output\driver.stderr.log"
$process.ProcessorAffinity = [IntPtr]0xFFF
$process.WaitForExit()
exit $process.ExitCode
'@
New-Item -ItemType Directory -Force -Path C:\arw-run | Out-Null
Set-Content -Path $driver -Value $script -Encoding UTF8
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$driver`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User '33135'
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit (New-TimeSpan -Days 2) -Hidden
Register-ScheduledTask -TaskName 'ARW-CausalAttribution' -Action $action `
  -Trigger $trigger -Settings $settings -User '33135' -Force | Out-Null
Start-ScheduledTask -TaskName 'ARW-CausalAttribution'
```

Register `ARW-CausalAttribution` with `MultipleInstances IgnoreNew` and a two-day execution limit. The task must run inventory -> four captures -> twelve replays -> optional four CAdam replays -> final decision without an SSH window.

- [ ] **Step 6: Verify dispatch and monitor the mechanical stop**

Verify dispatch mechanically:

```powershell
Get-ScheduledTask -TaskName 'ARW-CausalAttribution' | Get-ScheduledTaskInfo
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object CommandLine -Like '*causal_attribution_gate*' |
  Select-Object ProcessId, CommandLine
Get-Process python | Select-Object Id, ProcessorAffinity, CPU
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader
git -C C:\arw-m0 rev-parse HEAD
Get-Content "C:\arw-results\causal-attribution-$shortSha\driver.stdout.log" -Tail 30
```

Require the gate child affinity to equal decimal `4095`. The terminal artifact is `CAUSAL_ATTRIBUTION_DECISION.json`; exit code 2 with a valid `NO-GO` artifact is a scientific result, not an infrastructure failure.

Do not dispatch any follow-on experiment after either `GO` or `NO-GO`. Mirror compact metrics, manifests, decision JSON, and SHA-256 hashes to the Mac; leave raw checkpoints on Windows.

---

## Plan self-review result

- Spec coverage: all eight spec sections map to Tasks 1-5, including legacy reuse, exact 2x2 intervention, mandatory/conditional counts, endpoints, fail-closed gate, integrity, compute cap, and ten-day dispatch.
- Placeholder scan: no deferred code, unnamed validation, or unspecified test step remains.
- Type consistency: checkpoint epoch archives feed `CheckpointState`; replay bundles feed `evaluate_attribution`; source-commit sentinel and data roots feed `run_gate`.
