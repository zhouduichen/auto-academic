# CIFAR-10N Cross-Model Repair Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seal CIFAR-10N without test access, calibrate a discovery-only repair selector, and run the frozen 20-bundle ResNet/ViT independent confirmation matrix.

**Architecture:** Add a sealed CIFAR-10N adapter beside the existing CIFAR-100 stores, then minimally generalize the transport request for dataset, class count, model family, and an excluded diagnostic prefix. A separate calibration module freezes model-specific thresholds from discovery snapshots; a confirmation orchestrator binds every input and evaluates mechanism and policy gates atomically.

**Tech Stack:** Python 3.12, PyTorch, torchvision, Transformers, PEFT, NumPy, pytest, Ruff, uv, Windows PowerShell, RTX 5080.

## Global Constraints

- Use only official CIFAR-10 training images and CIFAR-10N `worse_label`; never inspect another CIFAR-10N label vector.
- Never construct the official CIFAR-10 test split; keep development and sealed-confirmation training partitions unloaded.
- Split seed is `20260801`; training/tuning/development/confirmation counts are 40,000/2,500/2,500/5,000.
- Models are pinned ResNet-18 full fine-tuning and ViT-B/16 IN21K rank-8 LoRA, each with ten output classes.
- Seeds are exactly 701--705; dose is 1,250; warm-up is 500; diagnostic prefix is eight batches; horizons are 0--512 as frozen in the spec.
- Execute all 20 bundles serially regardless of interim scientific values, with at most 12 CPU library threads and no GPU power cap.
- Test, development, or confirmation partitions cannot influence selection, execution, stopping, or primary analysis.
- Infrastructure failures may resume exactly; scientific failures may not change seeds, thresholds, labels, endpoints, or gates.

---

### Task 1: Seal the exact CIFAR-10N training inputs

**Files:**
- Create: `experiments/cifar10n_transport_data.py`
- Create: `tests/test_cifar10n_transport_data.py`

**Interfaces:**
- Produces: `cifar10_split(clean_labels: np.ndarray, split_seed: int = 20260801) -> dict[str, np.ndarray]`.
- Produces: `seal_cifar10n(cifar_root: Path, label_file: Path, output: Path, *, expected_label_sha256: str) -> dict[str, object]`.
- Produces: `validate_cifar10n(root: Path) -> dict[str, object]` returning paths and manifest hashes for `training`, `image-store`, and `tuning-store`.

- [ ] **Step 1: Write failing split and isolation tests**

Create synthetic 10-class labels with 5,000 examples per class. Assert exact per-class allocations of 4,000/250/250/500, stable hashes, no overlap, and full coverage. Monkeypatch `torchvision.datasets.CIFAR10` and assert only `train=True` is accepted.

```python
def test_cifar10_split_is_exact_and_disjoint():
    labels = np.repeat(np.arange(10), 5_000)
    split = data.cifar10_split(labels)
    assert {name: len(ids) for name, ids in split.items()} == {
        "train": 40_000, "tuning": 2_500,
        "development": 2_500, "confirmation": 5_000,
    }
    assert len(np.unique(np.concatenate(list(split.values())))) == 50_000
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `uv run --locked pytest tests/test_cifar10n_transport_data.py -q`

Expected: collection fails because `experiments.cifar10n_transport_data` does not exist.

- [ ] **Step 3: Implement exact label validation and sealed stores**

Load only `torch.load(label_file, map_location="cpu", weights_only=False)["worse_label"]` and reject extra label selection. Validate shape `(50000,)`, integer dtype, range 0--9, class coverage, and the caller-supplied file SHA-256 before loading. Construct CIFAR-10 with `train=True, download=False`. Write:

```text
output/training/sample_ids.npy
output/training/noisy_labels.npy
output/training/manifest.json
output/image-store/sample_ids.npy
output/image-store/images.npy
output/image-store/clean_labels.npy
output/image-store/train_order.npy
output/image-store/manifest.json
output/tuning-store/images.npy
output/tuning-store/clean_labels.npy
output/tuning-store/manifest.json
output/SPLIT_MANIFEST.json
```

Use stable IDs `cifar10-train-00000` through `cifar10-train-49999`. Public stores contain only training and tuning IDs. Record the development and confirmation hashes/counts in `SPLIT_MANIFEST.json` without writing their images or labels. Every manifest contains:

```python
{
    "dataset": "cifar10n",
    "label_set": "worse_label",
    "split_seed": 20260801,
    "test_loaded": False,
    "development_loaded": False,
    "confirmation_loaded": False,
}
```

- [ ] **Step 4: Verify seal corruption and forbidden-access failures**

Test wrong file hash, missing/extra/out-of-range labels, wrong CIFAR order, a modified `.npy`, a second label key request, and any `train=False` constructor. Each must raise before producing a valid manifest.

Run: `uv run --locked pytest tests/test_cifar10n_transport_data.py -q`

Expected: all tests pass.

- [ ] **Step 5: Lint and commit the data adapter**

```bash
uv run --locked ruff check experiments/cifar10n_transport_data.py tests/test_cifar10n_transport_data.py
git diff --check
git add experiments/cifar10n_transport_data.py tests/test_cifar10n_transport_data.py
git commit -m "feat: seal CIFAR-10N transport inputs"
```

### Task 2: Generalize full-state transport without changing legacy semantics

**Files:**
- Modify: `experiments/causal_transport_bundle.py`
- Modify: `experiments/repair_handoff_models.py`
- Modify: `tests/test_causal_transport_bundle.py`
- Modify: `tests/test_resnet_full_state_repair.py`
- Create: `tests/test_cifar10n_transport_bundle.py`

**Interfaces:**
- Extends: `TransportRequest` with `dataset_name: Literal["cifar100", "cifar10n"] = "cifar100"`, `num_labels: int = 100`, and `diagnostic_prefix_steps: int = 0`.
- Produces: `repair_handoff_models.build_transport_model(config: clean.Config, *, num_labels: int = 100) -> nn.Module`.
- Produces: `stream_label_nll(model, batches, device, *, num_labels: int) -> float`.

- [ ] **Step 1: Write failing backward-compatibility and CIFAR-10N tests**

Assert an old request serializes with CIFAR-100 defaults and still passes every existing test. Add a synthetic CIFAR-10N request and assert ten-logit ResNet/ViT heads, no CIFAR-100 constructor, exactly eight excluded diagnostic plans, 512 replay plans after the prefix, and a summary field:

```python
"diagnostic": {
    "prefix_steps": 8,
    "normalized_stream_label_nll": expected,
    "prefix_plan_sha256": expected_hash,
}
```

- [ ] **Step 2: Confirm RED without touching model or runner code**

Run:

```bash
uv run --locked pytest tests/test_cifar10n_transport_bundle.py tests/test_causal_transport_bundle.py -q
```

Expected: new request fields and model-class arguments are unavailable.

- [ ] **Step 3: Add ten-class pinned model construction**

Keep the existing default at 100 labels. For ResNet pass `num_labels`, retain exact allowed classifier mismatch validation, and require full fine-tuning. For ViT construct the pinned backbone, set `num_labels`, attach a fresh classifier, then apply rank-8 LoRA to `q_proj`/`v_proj` with the classifier saved. Add `vit_artifact_binding()` parallel to `resnet_artifact_binding()` and bind config/model hashes and revision.

```python
def build_transport_model(config: clean.Config, *, num_labels: int = 100) -> nn.Module:
    if num_labels not in {10, 100}:
        raise ValueError("transport num_labels must be 10 or 100")
    if config.model_id == RESNET_MODEL_ID:
        return _build_resnet(config, num_labels=num_labels)
    if config.model_id == VIT_MODEL_ID:
        return _build_vit_lora(config, num_labels=num_labels)
    raise ValueError("unsupported transport model_id")
```

- [ ] **Step 4: Add dataset routing and diagnostic-prefix exclusion**

For `cifar100`, preserve the existing constructor and behavior byte-for-byte when the new fields have defaults. For `cifar10n`, load `clean_labels.npy` from the validated image store and never instantiate a torchvision test dataset. Generate `warmup + dose + diagnostic_prefix_steps + 512` plans; score the prefix at the restored noisy-exposure model without updates; pass only the remaining 512 plans to factorial replay.

Normalize score as:

```python
score = mean_cross_entropy_on_observed_labels / math.log(request.num_labels)
```

Record the prefix plan hash and assert optimizer/model/RNG state is unchanged by scoring.

- [ ] **Step 5: Run full transport regression**

```bash
uv run --locked pytest \
  tests/test_cifar10n_transport_bundle.py \
  tests/test_causal_transport_bundle.py \
  tests/test_causal_attribution_replay.py \
  tests/test_resnet_full_state_repair.py \
  tests/test_resnet_full_state_stage2.py -q
uv run --locked ruff check \
  experiments/causal_transport_bundle.py \
  experiments/repair_handoff_models.py \
  tests/test_cifar10n_transport_bundle.py
git diff --check
```

Expected: all tests and Ruff pass; legacy request tests remain unchanged in meaning.

- [ ] **Step 6: Commit generalized transport**

```bash
git add experiments/causal_transport_bundle.py experiments/repair_handoff_models.py \
  tests/test_causal_transport_bundle.py tests/test_resnet_full_state_repair.py \
  tests/test_cifar10n_transport_bundle.py
git commit -m "feat: support CIFAR-10N cross-model transport"
```

### Task 3: Freeze the discovery-only repair policy

**Files:**
- Create: `experiments/repair_policy_calibration.py`
- Create: `tests/test_repair_policy_calibration.py`

**Interfaces:**
- Produces: `candidate_thresholds(scores: list[float]) -> list[float]`.
- Produces: `leave_one_seed_out(scores: list[DiscoveryScore]) -> dict[str, object]`.
- Produces: `calibrate_policy(inputs: CalibrationInputs, output: Path) -> dict[str, object]` writing `REPAIR_POLICY_DECISION.json` atomically.

- [ ] **Step 1: Write failing pure threshold tests**

Use three paired seeds with clean scores below noisy scores. Require adjacent midpoints, smallest-threshold tie breaking, seed-paired leave-one-out predictions, balanced accuracy 1.0, and direction `score < threshold -> CN`, otherwise `NC`. Add overlapping scores that yield `NO-GO` below 5/6.

```python
def test_threshold_ties_choose_smallest_and_freeze_direction():
    result = calibration.leave_one_seed_out(PAIRED_DISCOVERY_SCORES)
    assert result["decision"] == "GO"
    assert result["rule"] == "lt:CN,ge:NC"
    assert result["threshold"] == min(result["maximizing_thresholds"])
```

- [ ] **Step 2: Confirm RED**

Run: `uv run --locked pytest tests/test_repair_policy_calibration.py -q`

Expected: import failure.

- [ ] **Step 3: Implement pure calibration and fail-closed discovery binding**

Bind corrected ResNet seeds 601--603 and valid ViT seeds 501--503, their contracts, snapshot manifests, input stores, source commits, model hashes, and continuation plans. Accept complete-state v2 snapshots. Accept legacy ViT snapshots only after proving the model has no persistent mutable buffers omitted by the snapshot and every trainable tensor is present; reject legacy ResNet snapshots.

Reconstruct the first eight continuation plans and compute normalized stream-label NLL at the noisy-exposure model. Do not update weights, optimizer, or RNG. Write scores and thresholds before CIFAR-10N parsing. Remove a stale decision before validation.

- [ ] **Step 4: Test atomicity, isolation, and exact replay**

Assert missing/modified snapshots, wrong seed/model/contract, mutable omitted buffers, nonfinite scores, data access flags, and changed RNG/model state all fail without a decision. Rerunning from valid inputs must reproduce identical decision bytes.

Run:

```bash
uv run --locked pytest tests/test_repair_policy_calibration.py tests/test_causal_transport_bundle.py -q
uv run --locked ruff check experiments/repair_policy_calibration.py tests/test_repair_policy_calibration.py
git diff --check
```

- [ ] **Step 5: Commit the calibration gate**

```bash
git add experiments/repair_policy_calibration.py tests/test_repair_policy_calibration.py
git commit -m "feat: freeze discovery-only repair policy"
```

### Task 4: Add the immutable 20-bundle confirmation gate

**Files:**
- Create: `experiments/cifar10n_repair_confirmation.py`
- Create: `tests/test_cifar10n_repair_confirmation.py`

**Interfaces:**
- Produces: `build_jobs(output: Path) -> list[ConfirmationJob]` with exactly 20 jobs.
- Produces: `write_contract(...) -> dict[str, object]` writing `CIFAR10N_CONFIRMATION_CONTRACT.json`.
- Produces: `evaluate_confirmation(...) -> dict[str, object]`.
- Produces CLI commands `preflight`, `run`, and `finalize`.

- [ ] **Step 1: Write failing exact-matrix and gate tests**

Generate synthetic summaries/trajectories for seeds 701--705, models `resnet18`/`vit_lora`, and clean/noisy continuations. Require 20 unique jobs, full-state schema v2, exact request/model/data bindings, and finite 4x11 trajectories.

Create passing mechanism data with five negative cross-model seed contrasts, at least four negative seeds per model, and at least four action-switch seeds per model. Create policy data satisfying both-model fixed-policy improvement, 8/10 unit wins, and oracle regret 0.25. Flip each condition separately and require the corresponding `NO-GO`.

- [ ] **Step 2: Confirm RED**

Run: `uv run --locked pytest tests/test_cifar10n_repair_confirmation.py -q`

Expected: module import fails.

- [ ] **Step 3: Implement the contract and preflight**

The contract binds the approved design SHA, source commit, `uv.lock`, official label file and sealed-store hashes, split manifest, model revisions/hashes, policy decision/hash, exact 20 jobs, optimizer, diagnostic prefix, horizons, all gate constants, test isolation, and excluded prior artifacts. `preflight` loads both ten-class models on CUDA, performs two finite updates, exact complete-state restore, optimizer restore, prefix-score no-mutation check, and peak-VRAM check without loading CIFAR-10N outcomes.

- [ ] **Step 4: Implement serial run/resume and atomic decisions**

Run every job through `causal_transport_bundle.run_bundle`, rerun completed-bundle validation to prove exact resume, and do not evaluate interim scientific values. `finalize` validates all checksums without GPU work and writes:

```text
CIFAR10N_MECHANISM_DECISION.json
CIFAR10N_POLICY_DECISION.json
CIFAR10N_CONFIRMATION_REPORT.json
```

Mechanism calculations use `margin = NC - CN` and the equal-model seed aggregation frozen in the spec. Policy calculations use trapezoidal clean-loss-excess AUC over 32/64/128/256, equal continuation weights, fixed CN/NC/NN comparators, per-context oracle, and the explicit zero-gap rule.

- [ ] **Step 5: Run complete local verification**

```bash
uv run --locked pytest \
  tests/test_cifar10n_transport_data.py \
  tests/test_cifar10n_transport_bundle.py \
  tests/test_repair_policy_calibration.py \
  tests/test_cifar10n_repair_confirmation.py \
  tests/test_causal_attribution_replay.py \
  tests/test_causal_transport_bundle.py \
  tests/test_resnet_full_state_repair.py \
  tests/test_resnet_full_state_stage2.py -q
uv run --locked ruff check \
  experiments/cifar10n_transport_data.py \
  experiments/repair_policy_calibration.py \
  experiments/cifar10n_repair_confirmation.py \
  experiments/causal_transport_bundle.py \
  experiments/repair_handoff_models.py
git diff --check
```

Expected: all tests and lint checks pass.

- [ ] **Step 6: Commit and push the frozen runner**

```bash
git add experiments/cifar10n_repair_confirmation.py tests/test_cifar10n_repair_confirmation.py
git commit -m "feat: add CIFAR-10N repair confirmation gate"
git push origin codex/m05-direct-experiment
```

### Task 5: Seal data, dispatch once, and preserve the evidence package

**Files:**
- Runtime only: `C:\arw-data\cifar10n-confirmation-<commit>\`
- Runtime only: `C:\arw-results\cifar10n-confirmation-<commit>\`

**Interfaces:**
- Consumes the exact committed runner, policy decision, official CIFAR-10/CIFAR-10N files, and Windows CUDA environment.
- Produces the sealed stores, preflight, 20 bundles, three final decisions, logs, telemetry, and SHA-256 manifests.

- [ ] **Step 1: Create a clean detached Windows worktree and verify code**

Fetch or transfer the exact commit, create `C:\arw-cifar10n-<shortsha>`, verify `git rev-parse HEAD`, run the focused test set, and set:

```powershell
$env:OMP_NUM_THREADS="12"
$env:MKL_NUM_THREADS="12"
$env:OPENBLAS_NUM_THREADS="12"
$env:NUMEXPR_NUM_THREADS="12"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

- [ ] **Step 2: Calibrate and freeze policy before parsing CIFAR-10N**

Run the discovery calibration against exact existing artifacts. Require an atomic decision and archive its hash. If policy calibration is `NO-GO`, keep that immutable result and continue mechanism confirmation without a policy claim.

- [ ] **Step 3: Download, hash, and seal only the approved data**

Download the official CIFAR-10N file without opening it, compute SHA-256, compare it with the command-line expected hash, then run `seal_cifar10n`. Validate all manifests twice and confirm all three access flags remain false. Do not enumerate alternate label-vector values.

- [ ] **Step 4: Run CUDA preflight and launch a persistent scheduled task**

Use `C:\arw-m0\.venv\Scripts\python.exe`, the clean worktree, CUDA, four data workers, and 12-thread environment limits. Register one Windows scheduled task with `MultipleInstances=IgnoreNew`, battery continuation enabled, no execution-time limit, and combined runner log. Do not set a GPU power limit.

- [ ] **Step 5: Confirm live execution and complete the matrix**

Confirm task state, Python command line, nonzero GPU utilization, VRAM, and temperature. On completion require exactly 20 checksum-valid bundles and both decision files. Infrastructure failures resume; scientific `NO-GO` does not trigger reruns.

- [ ] **Step 6: Mirror and audit the final evidence**

Copy contracts, manifests, summaries, trajectories, decisions, logs, telemetry, and checksums back to the local immutable results tree. Verify source and artifact hashes after transfer, then run the result-to-claim audit before drafting or expanding experiments.
