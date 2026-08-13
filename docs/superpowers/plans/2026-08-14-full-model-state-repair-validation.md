# Full-Model-State Repair Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make crossed ResNet repair interventions restore every model parameter and persistent buffer, prove branch-order/resume invariance, and run a two-bundle correctness sentinel before any scientific matrix expansion.

**Architecture:** Upgrade causal transport checkpoints from trainable parameters to exact `model.state_dict()` snapshots while keeping optimizer state as an independently crossed factor. Refactor replay measurement so CC-relative metrics are computed after raw branch execution, allowing standard and reversed branch orders to be compared. Add a separate fail-closed ResNet full-state sentinel and leave all prior ResNet artifacts immutable and excluded.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, Ruff, uv, Windows PowerShell, RTX 5080.

## Global Constraints

- Paper validity and contribution quality take priority over compute reuse.
- The 30 ViT bundles remain discovery evidence; all 24 old ResNet bundles are excluded from scientific estimates.
- Full model state means every exact `model.state_dict()` entry, including BatchNorm `running_mean`, `running_var`, and `num_batches_tracked`.
- Stage 1 runs only seed 601, dose 1250, clean/noisy continuation; Stage 2 is forbidden until every correctness check passes.
- Test labels remain unavailable; CPU libraries use at most 12 threads; GPU execution is serial.

---

### Task 1: Restore complete model state across every branch

**Files:**
- Modify: `experiments/causal_attribution_replay.py`
- Modify: `experiments/causal_transport_bundle.py`
- Modify: `tests/test_causal_attribution_replay.py`
- Modify: `tests/test_causal_transport_bundle.py`

**Interfaces:**
- Produces: `capture_model_state(model: nn.Module) -> dict[str, Tensor]`, `restore_model_state(model: nn.Module, state: dict[str, Tensor]) -> None`, and `CheckpointState.model_state`.
- Preserves: four carrier combinations `CC`, `CN`, `NC`, `NN` and optimizer-state schema.

- [ ] **Step 1: Add a failing BatchNorm isolation test**

Create a small `Conv2d -> BatchNorm2d -> Linear` model, expose clean/noisy states, mutate all BatchNorm buffers during one branch, restore another branch, and assert exact equality for every `state_dict()` entry. Also assert `validate_checkpoint_pair` rejects missing, extra, mismatched, empty, or nonfinite floating model-state tensors.

- [ ] **Step 2: Confirm the regression is RED**

Run:

```bash
uv run --locked pytest tests/test_causal_attribution_replay.py tests/test_causal_transport_bundle.py -q
```

Expected: the BatchNorm buffer isolation test fails because current snapshots contain only trainable parameters.

- [ ] **Step 3: Implement exact model-state capture, validation, and restore**

Use detached clones of every `model.state_dict()` tensor. Restore with `model.load_state_dict(deepcopy(state), strict=True)` and verify the resulting state is tensor-for-tensor equal. Rename the checkpoint carrier to `model_state`; snapshot schema becomes `causal-transport-exposure/2` and stores `model_state`, `optimizer`, `next_epoch`, and the contract hash. Version-1 snapshots fail closed.

- [ ] **Step 4: Cross complete model state with optimizer state**

Build `CC/CN/NC/NN` from complete clean/noisy model states. `run_paired_exposure`, common-state restoration, snapshot save/load, replay restoration, and state-distance diagnostics must all use complete model state. Do not alter legacy `m0_core.BranchState` semantics used by non-transport experiments.

- [ ] **Step 5: Run focused and legacy regression tests**

```bash
uv run --locked pytest \
  tests/test_causal_attribution_replay.py \
  tests/test_causal_transport_bundle.py \
  tests/test_causal_transport_sentinel.py \
  tests/test_resnet_repair_handoff_sentinel.py \
  tests/test_resnet_repair_curve.py -q
uv run --locked ruff check experiments/causal_attribution_replay.py experiments/causal_transport_bundle.py tests/test_causal_attribution_replay.py tests/test_causal_transport_bundle.py
```

Expected: all pass and Ruff reports `All checks passed!`.

- [ ] **Step 6: Commit the carrier correction**

```bash
git add experiments/causal_attribution_replay.py experiments/causal_transport_bundle.py tests/test_causal_attribution_replay.py tests/test_causal_transport_bundle.py
git commit -m "fix: restore complete model state in transport branches"
```

### Task 2: Make replay independent of branch execution order

**Files:**
- Modify: `experiments/causal_attribution_replay.py`
- Modify: `tests/test_causal_attribution_replay.py`

**Interfaces:**
- Produces: `run_cached_branches(..., branch_order: tuple[str, ...] = BRANCH_NAMES)` and order-independent `measure_factorial_replay`.

- [ ] **Step 1: Write a failing standard-versus-reversed test**

Run the BatchNorm model using orders `("CC", "CN", "NC", "NN")` and `("NN", "NC", "CN", "CC")`. Require identical rows after canonical sorting and identical endpoints/effects. Reject duplicated, missing, or unknown branch names.

- [ ] **Step 2: Refactor raw measurement and CC normalization**

During branch execution, store raw probe loss, probabilities, gradients, and state snapshots without requiring CC to run first. After all branches complete, index the CC record for each horizon and derive `clean_loss_excess`, distances, gradient cosine, and prediction JS. Canonically sort rows by `BRANCH_NAMES` and horizon before endpoint calculation.

- [ ] **Step 3: Verify deterministic order invariance**

```bash
uv run --locked pytest tests/test_causal_attribution_replay.py -q
uv run --locked ruff check experiments/causal_attribution_replay.py tests/test_causal_attribution_replay.py
git diff --check
```

Expected: both orders are bitwise identical on CPU; malformed orders fail closed.

- [ ] **Step 4: Commit order-independent replay**

```bash
git add experiments/causal_attribution_replay.py tests/test_causal_attribution_replay.py
git commit -m "fix: make factorial replay branch-order invariant"
```

### Task 3: Add and dispatch the two-bundle ResNet correctness sentinel

**Files:**
- Create: `experiments/resnet_full_state_repair.py`
- Create: `tests/test_resnet_full_state_repair.py`
- Runtime only: `C:\arw-results\resnet-full-state-<commit>\`

**Interfaces:**
- Consumes: pinned ResNet adapter, sealed CIFAR stores, full-state transport core, seed 601, dose 1250, clean/noisy continuation.
- Produces: immutable contract, CUDA preflight, standard/reversed diagnostics, two scientific summaries, and `FULL_STATE_SENTINEL_DECISION.json`.

- [ ] **Step 1: Write failing exact-matrix and fail-closed gate tests**

Require exactly two condition bundles and two reversed-order diagnostics. The gate passes only when source/model/data bindings match, all summaries succeed, snapshot schema is version 2, all horizon-0 deterministic metrics match across continuations, standard/reversed results match exactly after canonical sorting, resume hashes match, every endpoint is finite, and `test_loaded=false`.

- [ ] **Step 2: Implement the immutable contract and sentinel runner**

Bind source commit, `uv.lock`, pinned model hashes, sealed store/noise hashes, excluded-artifact hashes, seed/dose/continuations, optimizer, horizons, both branch orders, and test isolation. Remove stale decisions before validation; write decisions atomically; reuse only checksum-valid completed artifacts.

- [ ] **Step 3: Run the complete local gate**

```bash
uv run --locked pytest \
  tests/test_resnet_full_state_repair.py \
  tests/test_causal_attribution_replay.py \
  tests/test_causal_transport_bundle.py \
  tests/test_resnet_repair_handoff_sentinel.py \
  tests/test_resnet_repair_curve.py -q
uv run --locked ruff check experiments/resnet_full_state_repair.py tests/test_resnet_full_state_repair.py
git diff --check
```

Expected: all pass, Ruff passes, and diff check is empty.

- [ ] **Step 4: Commit, push, and create a detached Windows worktree**

```bash
git add experiments/resnet_full_state_repair.py tests/test_resnet_full_state_repair.py
git commit -m "feat: add full-model-state ResNet sentinel"
git push origin codex/m05-direct-experiment
```

On Windows, fetch the exact commit into a clean detached worktree, use the existing CUDA environment, set offline model loading and 12-thread limits, and run focused tests plus CUDA preflight.

- [ ] **Step 5: Run only Stage 1 and stop for its decision**

Launch the exact four executions (two standard scientific bundles plus two reversed diagnostics) serially and independently of SSH. Require `FULL_STATE_SENTINEL_DECISION.json`. Mirror summaries, trajectories, snapshots manifests, contract, decision, logs, and hashes. Do not launch Stage 2 unless the decision is `GO`.

