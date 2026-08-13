# ResNet Repair Curve Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the existing ResNet discovery curve with only the missing doses 1 and 1250, combine them with the immutable dose-8/64 evidence, and stop or advance using a frozen continuous-trend gate.

**Architecture:** Preserve `experiments/resnet_repair_handoff_sentinel.py` and its `NO-GO` artifact unchanged. Add a separate curve-completion runner that executes 12 new bundles, validates and imports the old decision by hash, reconstructs all four doses, and writes one atomic decision. Reuse the pinned model, sealed stores, CUDA preflight, and `causal_transport_bundle.run_bundle`.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, Ruff, uv, PowerShell, Windows RTX 5080.

## Global Constraints

- Run only seeds `601, 602, 603`, new doses `1, 1250`, continuations `clean, noisy`, and the two frozen primary outcomes.
- Keep AdamW, the pinned ResNet revision, batch size 32, warm-up 500, sealed symmetric-noise mapping, and test isolation unchanged.
- Limit CPU libraries and PyTorch to 12 threads; execute GPU bundles serially; apply no GPU power limit.
- Existing dose-8/64 evidence remains immutable external-discovery evidence and is accepted only by exact SHA-256 and schema/provenance validation.
- A scientific miss stops the adaptive-policy mainline. Never add seeds, rerun finite outcomes, or tune the gate after observing results.

---

### Task 1: Add the frozen curve-completion contract and trend gate

**Files:**
- Create: `experiments/resnet_repair_curve.py`
- Create: `tests/test_resnet_repair_curve.py`

**Interfaces:**
- Consumes: `resnet_repair_handoff_sentinel.SentinelJob`, `write_contract`, `validate_preflight`, and immutable `REPAIR_HANDOFF_DECISION.json`.
- Produces: `build_new_jobs(output: Path) -> list[SentinelJob]`, `evaluate_curve(prior: dict[str, object], new: dict[str, dict[str, object]], *, contract_sha256: str, prior_decision_sha256: str) -> dict[str, object]`, and CLI commands `preflight`/`run`.

- [ ] **Step 1: Write failing matrix and provenance tests**

Add tests asserting the new matrix is exactly:

```python
assert [job.key for job in curve.build_new_jobs(tmp_path)] == [
    f"seed{seed}-dose{dose}-{continuation}"
    for seed in (601, 602, 603)
    for dose in (1, 1250)
    for continuation in ("clean", "noisy")
]
```

Also assert that evaluation rejects a changed prior SHA, non-`NO-GO` prior decision, missing old observation, wrong request dose/seed/continuation, nonfinite endpoint, `test_loaded != false`, duplicate key, or source commit not equal to the curve contract.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
uv run --locked pytest tests/test_resnet_repair_curve.py -q
```

Expected: collection fails because `experiments.resnet_repair_curve` does not exist.

- [ ] **Step 3: Implement the exact new-only job matrix and immutable contract**

Define:

```python
TRAINING_SEEDS = (601, 602, 603)
NEW_DOSES = (1, 1250)
ALL_DOSES = (1, 8, 64, 1250)
CONTINUATIONS = ("clean", "noisy")
```

The contract schema is `resnet-repair-curve-contract/1`. It binds the current source commit, `uv.lock`, pinned model artifacts, sealed stores, the old decision SHA-256, the old contract SHA-256 recorded in that decision, new matrix, optimizer configuration, outcomes, and `test_loaded=false`. Existing differing contracts fail closed.

- [ ] **Step 4: Implement the prespecified continuous-trend gate**

For each of the 12 seed × continuation × outcome strata, reconstruct margins at all four doses. Define:

```python
low = (margin_d1 + margin_d8) / 2.0
high = (margin_d64 + margin_d1250) / 2.0
contrast = high - low
```

Return `GO` only when every integrity check passes and:

```python
sum(contrast > 0 for all 12 strata) >= 10
sum(margin_d1250 > 0 for all 12 strata) >= 10
```

For both conditions, require at least 3/4 positives within every seed and at least 5/6 within every outcome and every continuation. Report every margin, contrast, grouping count, mean, median, minimum, and maximum. Binary carrier labels are diagnostic only.

- [ ] **Step 5: Add positive and scientific-failure tests**

Construct deterministic summaries whose low-dose margins are `(-2, -1)` and high-dose margins are `(2, 3)` and assert `GO`. Parameterize failures for overall count, one seed, one endpoint, one continuation, and dose-1250 positivity; each must return `NO-GO` without raising. Integrity failures must raise or make `exact_valid_matrix=false` and never produce `GO`.

- [ ] **Step 6: Run focused and regression gates**

Run:

```bash
uv run --locked pytest \
  tests/test_resnet_repair_curve.py \
  tests/test_resnet_repair_handoff_sentinel.py \
  tests/test_causal_transport_bundle.py -q
uv run --locked ruff check \
  experiments/resnet_repair_curve.py \
  tests/test_resnet_repair_curve.py
git diff --check
```

Expected: all tests pass, Ruff reports `All checks passed!`, and diff check is empty.

- [ ] **Step 7: Commit the runner and gate**

```bash
git add experiments/resnet_repair_curve.py tests/test_resnet_repair_curve.py
git commit -m "feat: add ResNet repair curve completion gate"
```

### Task 2: Dispatch only the 12 missing bundles on Windows

**Files:**
- Runtime only: `C:\arw-curve-resnet-<commit>\`
- Runtime only: `C:\arw-results\resnet-repair-curve-<commit>\`
- Mirror only: `results/resnet-repair-curve-<commit>/`

**Interfaces:**
- Consumes: exact Task-1 commit, existing pinned model cache, sealed CIFAR stores/noise bundles, and the immutable old ResNet decision.
- Produces: one contract, one validated CUDA preflight, 12 new summaries, and `REPAIR_CURVE_DECISION.json`.

- [ ] **Step 1: Push and create a detached Windows worktree**

Fetch the exact Task-1 commit into a new detached worktree. Require `git status --short --untracked-files=no` to be empty and verify the old result SHA before starting.

- [ ] **Step 2: Run tests and CUDA preflight offline**

Set `OMP_NUM_THREADS=12`, `MKL_NUM_THREADS=12`, `OPENBLAS_NUM_THREADS=12`, `NUMEXPR_NUM_THREADS=12`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`. Run the focused tests, then:

```powershell
python -m experiments.resnet_repair_curve preflight `
  --output C:\arw-results\resnet-repair-curve-<commit>\RESNET_CUDA_SENTINEL.json
```

Expected: exact current commit binding, finite loss, nonempty optimizer state, exact restoration, and peak VRAM below 8 GB.

- [ ] **Step 3: Launch the new-only serial matrix**

Invoke `python -m experiments.resnet_repair_curve run` with the sealed data paths, fixed noise mapping `601=1501`, `602=1502`, `603=1503`, old decision path, new output root, `--device cuda`, and `--workers 4`. Redirect stdout/stderr to a persistent log and start it independently of the SSH session. Confirm the first bundle reaches active GPU computation before disconnecting.

- [ ] **Step 4: Verify and mirror the decision**

Require exactly 12 succeeded new summaries, all provenance checks, `test_loaded=false`, and one atomic curve decision. Mirror only the contract, sentinel, decision, summaries, logs, and checksums. Report `GO` or `NO-GO`; do not start policy development automatically.

