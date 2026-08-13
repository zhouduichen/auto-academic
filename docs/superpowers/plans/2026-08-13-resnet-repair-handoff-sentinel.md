# ResNet Repair-Handoff Sentinel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a fail-closed ResNet-18 external sentinel that tests whether repair preference switches from optimizer state at dose 8 to parameters at dose 64.

**Architecture:** Add one pinned Hugging Face ResNet model adapter and route only the exact ResNet model ID through it while preserving the existing ViT-LoRA path byte-for-byte. A separate sentinel runner builds the 12-bundle matrix, binds model/data/source hashes into one contract, reuses the existing paired exposure and factorial replay engine, and writes one atomic decision. A short CUDA preflight must pass before Windows dispatch.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face Transformers, NumPy, existing causal-transport primitives, pytest, Ruff, SHA-256, RTX 5080.

## Global Constraints

- Preserve the previous Gate B and symmetric-noise `NO-GO` artifacts.
- Use exactly seeds `(601, 602, 603)`, doses `(8, 64)`, and continuations `("clean", "noisy")`.
- Reuse sealed symmetric noise bundles `(1501, 1502, 1503)` in fixed order.
- Use `microsoft/resnet-18` at revision `65a5785d9156231087c481e0c7dd33a5ff6f7e3e`; bind local model/config file hashes into the contract. The originally drafted `b84c5cd73e9544fa1b67d690748d13a4bdb29267` revision was rejected during implementation because it contains no `model.safetensors`.
- Use full-parameter AdamW, learning rate `3e-4`, weight decay `0.1`, betas `(0.9, 0.999)`, epsilon `1e-8`, batch size 32, warm-up 500, no scheduler.
- Keep official test inaccessible and require `test_loaded=false`.
- Limit CPU libraries to 12 threads and run GPU bundles serially.
- Do not authorize the later 50–90 hour matrix.

---

## File structure

- Create `experiments/repair_handoff_models.py`: pinned ResNet construction and local artifact binding.
- Modify `experiments/causal_transport_bundle.py`: select the ResNet adapter only for its exact model ID.
- Create `tests/test_repair_handoff_models.py`: adapter routing, loading audit, trainability, and hash tests.
- Create `experiments/resnet_repair_handoff_sentinel.py`: CUDA preflight, contract, jobs, gate, resume, and CLI.
- Create `tests/test_resnet_repair_handoff_sentinel.py`: exact matrix, contract, gate, non-finite, and preflight tests.
- Runtime only: `C:\arw-results\resnet-repair-handoff-<commit>\` and its local evidence mirror.

### Task 1: Add the pinned ResNet model adapter

**Files:**
- Create: `experiments/repair_handoff_models.py`
- Modify: `experiments/causal_transport_bundle.py`
- Create: `tests/test_repair_handoff_models.py`

**Interfaces:**
- Produces: `build_transport_model(config: clean.Config) -> nn.Module`.
- Produces: `resnet_artifact_binding(*, local_files_only: bool) -> dict[str, str]`.
- Existing `bundle.run_bundle(request)` consumes the model router without changing its public signature.

- [ ] **Step 1: Write failing routing and audit tests**

```python
def test_default_transport_model_preserves_vit_path(monkeypatch):
    expected = nn.Linear(1, 1)
    monkeypatch.setattr(models.clean, "_build_model", lambda config: expected)
    assert models.build_transport_model(_config(VIT_MODEL_ID)) is expected


def test_resnet_load_allows_only_replaced_classifier(monkeypatch):
    fake = _FakeResNet()
    monkeypatch.setattr(
        models.ResNetForImageClassification,
        "from_pretrained",
        lambda *args, **kwargs: (
            fake,
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [
                    ("classifier.1.weight", (1000, 512), (100, 512)),
                    ("classifier.1.bias", (1000,), (100,)),
                ],
                "error_msgs": [],
            },
        ),
    )
    assert models.build_transport_model(_config(models.RESNET_MODEL_ID)) is fake
    assert all(parameter.requires_grad for parameter in fake.parameters())
```

Add failures for an extra missing key, an unexpected key, an extra mismatched key, and an unknown model ID.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `uv run --locked pytest tests/test_repair_handoff_models.py -q`

Expected: collection fails because `experiments.repair_handoff_models` does not exist.

- [ ] **Step 3: Implement exact model construction and artifact binding**

```python
RESNET_MODEL_ID = "microsoft/resnet-18"
RESNET_REVISION = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
VIT_MODEL_ID = "google/vit-base-patch16-224-in21k"
ALLOWED_HEAD_MISMATCH = {"classifier.1.weight", "classifier.1.bias"}


def build_transport_model(config: clean.Config) -> nn.Module:
    if config.model_id == VIT_MODEL_ID:
        return clean._build_model(config)
    if config.model_id != RESNET_MODEL_ID:
        raise ValueError("unsupported transport model_id")
    model, loading = ResNetForImageClassification.from_pretrained(
        RESNET_MODEL_ID,
        revision=RESNET_REVISION,
        num_labels=100,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        use_safetensors=True,
        local_files_only=True,
    )
    _validate_loading_info(loading)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("ResNet sentinel requires full-parameter fine-tuning")
    return model
```

`resnet_artifact_binding` resolves `config.json` and `model.safetensors` through the pinned revision, rejects absent/non-files, and returns exact SHA-256 values with the model ID and revision.

- [ ] **Step 4: Route causal bundles through the adapter**

Replace only:

```python
model = clean._build_model(request.config).to(device)
```

with:

```python
model = repair_models.build_transport_model(request.config).to(device)
```

Keep optimizer construction, request serialization, snapshots, replay, and artifact schemas unchanged.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
uv run --locked pytest \
  tests/test_repair_handoff_models.py \
  tests/test_causal_transport_bundle.py \
  tests/test_causal_transport_orchestrator.py -q
uv run --locked ruff check \
  experiments/repair_handoff_models.py \
  experiments/causal_transport_bundle.py \
  tests/test_repair_handoff_models.py
```

Expected: all tests pass; Ruff reports `All checks passed!`.

- [ ] **Step 6: Commit the adapter**

```bash
git add experiments/repair_handoff_models.py experiments/causal_transport_bundle.py tests/test_repair_handoff_models.py
git commit -m "feat: add ResNet transport model adapter"
```

### Task 2: Add the exact repair-handoff sentinel

**Files:**
- Create: `experiments/resnet_repair_handoff_sentinel.py`
- Create: `tests/test_resnet_repair_handoff_sentinel.py`

**Interfaces:**
- Produces: `build_jobs(output: Path) -> list[SentinelJob]`.
- Produces: `write_contract(...) -> dict[str, object]`.
- Produces: `evaluate_sentinel(summaries: dict[str, dict[str, object]], *, contract_sha256: str) -> dict[str, object]`.
- Produces CLI commands: `preflight` and `run`.

- [ ] **Step 1: Write failing matrix and decision tests**

```python
def test_build_jobs_is_exact_matrix(tmp_path: Path) -> None:
    assert [job.key for job in sentinel.build_jobs(tmp_path)] == [
        f"seed{seed}-dose{dose}-{continuation}"
        for seed in (601, 602, 603)
        for dose in (8, 64)
        for continuation in ("clean", "noisy")
    ]


def test_gate_requires_short_state_and_long_parameter_repairs() -> None:
    record = sentinel.evaluate_sentinel(_passing_summaries(), contract_sha256="a" * 64)
    assert record["decision"] == "GO"
    assert record["checks"] == {
        "exact_valid_matrix": True,
        "dose64_all_12_parameter_repair": True,
        "dose8_at_most_3_parameter_repair": True,
        "dose8_each_seed_at_most_1": True,
    }
```

Add separate tests that flip one dose-64 margin, create four total dose-8 positives, create two positives within one dose-8 seed, remove a bundle, mismatch request metadata, change a contract, load test data, or insert a non-finite endpoint; each must fail closed.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `uv run --locked pytest tests/test_resnet_repair_handoff_sentinel.py -q`

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement the contract and job matrix**

```python
TRAINING_SEEDS = (601, 602, 603)
DOSES = (8, 64)
CONTINUATIONS = ("clean", "noisy")


def build_jobs(output: Path) -> list[SentinelJob]:
    return [
        SentinelJob(seed, dose, continuation, output / "sentinel" / key)
        for seed in TRAINING_SEEDS
        for dose in DOSES
        for continuation in CONTINUATIONS
        for key in (f"seed{seed}-dose{dose}-{continuation}",)
    ]
```

The immutable contract binds source commit, `uv.lock`, prior discovery and sentinel decisions, exact model artifact binding, store bindings, seeds, doses, continuations, optimizer configuration, outcomes, and `test_loaded=false`. Existing differing contracts fail closed.

- [ ] **Step 4: Implement the continuous repair gate**

For each bundle and primary outcome compute:

```python
parameter_repair = endpoints["NN"][outcome] - endpoints["CN"][outcome]
state_repair = endpoints["NN"][outcome] - endpoints["NC"][outcome]
repair_margin = parameter_repair - state_repair
```

Require all 12 dose-64 margins positive. At dose 8 require no more than three positive margins overall and no more than one among the four repeated observations for each seed. Store all margins and diagnostic carrier classes in the atomic decision.

- [ ] **Step 5: Implement CUDA preflight and dispatch guard**

`preflight` loads the pinned model locally, runs two deterministic 224-pixel forward/backward AdamW steps, verifies finite loss, nonempty optimizer state, exact capture/restore, model binding, CUDA availability, peak VRAM below 8 GB, and `test_loaded=false`. It atomically writes `RESNET_CUDA_SENTINEL.json` with the source commit and model binding.

`run` refuses to create a contract unless that artifact reports `passed` and exactly matches the current commit and model binding.

- [ ] **Step 6: Run focused and full regression gates**

Run:

```bash
uv run --locked pytest \
  tests/test_resnet_repair_handoff_sentinel.py \
  tests/test_repair_handoff_models.py \
  tests/test_causal_transport_bundle.py \
  tests/test_causal_transport_gate.py \
  tests/test_attribution_transport_sentinel.py -q
uv run --locked ruff check \
  experiments/resnet_repair_handoff_sentinel.py \
  tests/test_resnet_repair_handoff_sentinel.py
git diff --check
```

Expected: all tests pass, Ruff passes, and the diff check is empty.

- [ ] **Step 7: Commit the sentinel**

```bash
git add experiments/resnet_repair_handoff_sentinel.py tests/test_resnet_repair_handoff_sentinel.py
git commit -m "feat: add ResNet repair handoff sentinel"
```

### Task 3: Validate and dispatch on Windows

**Files:**
- Runtime only: `C:\arw-sentinel-resnet\`
- Runtime only: `C:\arw-results\resnet-repair-handoff-<commit>\`
- Runtime only: local `results/resnet-repair-handoff-<commit>\`

**Interfaces:**
- Consumes the two committed implementation tasks and existing sealed Windows stores.
- Produces one immutable CUDA sentinel, contract, 12 summaries, and `REPAIR_HANDOFF_DECISION.json`.

- [ ] **Step 1: Push the exact clean commit and create an isolated Windows worktree**

Verify the original Windows worktree remains untouched. Fetch the branch and add a detached worktree at the exact implementation commit. Confirm `git status --short --untracked-files=no` is empty there.

- [ ] **Step 2: Prefetch and bind the pinned model**

Download only revision `65a5785d9156231087c481e0c7dd33a5ff6f7e3e` once, then run `resnet_artifact_binding(local_files_only=True)`. Record exact `config.json` and `model.safetensors` hashes; subsequent experiment commands set Hugging Face and Transformers offline.

- [ ] **Step 3: Run Windows tests and CUDA preflight**

Set `PYTHONPATH` to the detached worktree and CPU thread environment variables to `12`. Run the focused tests, then:

```powershell
python -m experiments.resnet_repair_handoff_sentinel preflight `
  --output C:\arw-results\resnet-repair-handoff-<commit>\RESNET_CUDA_SENTINEL.json
```

Expected: `status=passed`, finite loss, exact restore, peak VRAM below 8 GB, and GPU temperature within the machine's normal operating range. If this fails, stop before the matrix.

- [ ] **Step 4: Dispatch the exact serial matrix**

Run with CPU libraries limited to 12 threads, offline model loading, and the fixed seed-to-noise mapping `601=1501`, `602=1502`, `603=1503`. The runner must resume valid summaries and fail on partial or mismatched artifacts.

- [ ] **Step 5: Verify completion and mirror evidence**

Require 12/12 checksum-valid summaries and one atomic decision. Mirror only the contract, preflight, decision, summaries, and SHA manifests to the local result root. Recompute file hashes locally and report `GO/NO-GO`; do not launch any later matrix.
