# Protect-M Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen Protect-M/Protect-MV recurrence, validate it against AdamW and the approved state contract, then dispatch the fail-closed 24-run M1 Pilot on Windows.

**Architecture:** A focused optimizer module owns Protect-AdamW and the paper-faithful CAdamW control. Existing clean/noisy runners gain a method selector while retaining their data isolation, checkpoint, and artifact behavior. A small Pilot orchestrator runs the frozen four-method matrix and emits a mechanical gate decision; a separate sentinel must pass before the matrix can start.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing CIFAR-100/ViT-LoRA runners, PowerShell scheduled task on Windows RTX 5080.

## Global Constraints

- Protect-M and Protect-MV use `(alpha_fast, alpha_slow, tau)=(0.90, 0.99, 0.10)` with no detector tuning.
- Pilot methods are `adamw`, `cadam`, `protect-m`, and `protect-mv`; conditions are clean and frozen 40% instance-dependent noise; paired train seeds are `301,302,303`.
- All methods use FP32 with autocast and GradScaler disabled, one forward, one backward, `lr=3e-4`, `weight_decay=0.1`, and 20 epochs.
- CPU tests and the GPU functional/performance sentinel must pass before Pilot dispatch; each candidate's median step-time ratio must be at most `1.05`.
- No development, confirmation, or official-test artifact may be constructed or loaded; every output must record `test_loaded=false`.
- Pilot passes only if one candidate has all three noisy paired differences versus CAdam positive, mean noisy difference at least `+1.0pp`, mean clean drop versus AdamW at most `0.5pp`, every clean drop at most `1.0pp`, positive fixed-wall-clock noisy improvement, and mean wall-clock/VRAM overhead at most `5%`.

---

### Task 1: Protect-AdamW optimizer and conformance tests

**Files:**
- Create: `experiments/m1_optimizers.py`
- Create: `tests/test_m1_optimizers.py`

**Interfaces:**
- Produces: `ProtectAdamW(params, *, lr, betas, eps, weight_decay, protection_target, protection_enabled, alpha_fast, alpha_slow, tau)` and `last_diagnostics: dict[str, object]`.
- Produces: `CAdamW(params, *, lr, betas, eps, weight_decay)` implementing arXiv:2411.19647v2 Algorithm 1 with an unconditional AdamW decay followed by the elementwise `m_t * g_t > 0` adaptive-update mask.

- [ ] **Step 1: Write failing optimizer tests**

```python
def test_protection_disabled_is_bit_exact_adamw():
    reference, candidate = paired_parameters()
    adam = torch.optim.AdamW(reference, lr=3e-4, weight_decay=0.1, foreach=False)
    protect = ProtectAdamW(candidate, lr=3e-4, weight_decay=0.1,
                           protection_target="m", protection_enabled=False)
    for gradients in five_step_fixture():
        apply_gradients(reference, gradients)
        apply_gradients(candidate, gradients)
        adam.step(); protect.step()
        assert_optimizer_state_bit_exact(reference, candidate, adam, protect)

def test_rejected_step_only_changes_the_registered_carriers():
    m = forced_rejection("m")
    mv = forced_rejection("mv")
    assert torch.equal(m.before_exp_avg, m.after_exp_avg)
    assert torch.equal(m.expected_exp_avg_sq, m.after_exp_avg_sq)
    assert torch.equal(mv.before_exp_avg, mv.after_exp_avg)
    assert torch.equal(mv.before_exp_avg_sq, mv.after_exp_avg_sq)
    assert torch.equal(m.parameter, mv.parameter)

def test_default_reachability_witness_switches_off_and_on():
    detector = default_detector()
    states = [detector.observe(q) for q in [-1.0] * 10 + [1.0] * 36]
    assert states[1].admit is False
    assert states[45].admit is True
```

- [ ] **Step 2: Run tests and verify the expected import failure**

Run: `uv run --locked pytest tests/test_m1_optimizers.py -q`
Expected: failure because `experiments.m1_optimizers` does not exist.

- [ ] **Step 3: Implement the staged, fail-closed recurrence**

Implement option/dtype/finiteness validation before mutation; stream FP32 `dot`, `gg`, and `rr`; update the two-crossing detector; form ephemeral AdamW moments and parameter updates; commit `m` or `m+v` according to the frozen target; serialize `protect_state` schema version 1; expose detached step diagnostics. `protection_enabled=False` must commit both moments on every step and match the locked PyTorch for-loop order.

- [ ] **Step 4: Add failure and resume coverage**

Cover `grad=None`, zero gradients, multiple groups, nonzero decay, invalid/sparse/nonfinite gradients, unsupported options, rejected/admitted state, state-dict mismatch, atomic failed load, and bit-exact resumed execution.

- [ ] **Step 5: Verify Task 1**

Run: `uv run --locked pytest tests/test_m1_optimizers.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add experiments/m1_optimizers.py tests/test_m1_optimizers.py
git commit -m "research: implement protected AdamW candidates"
```

### Task 2: Frozen method integration in clean/noisy runners

**Files:**
- Modify: `experiments/m1_calibration_clean.py`
- Modify: `experiments/m1_calibration_noisy.py`
- Modify: `experiments/m1_noisy_data.py`
- Create: `tests/test_m1_pilot_runner.py`

**Interfaces:**
- Consumes: `ProtectAdamW` and `CAdamW` from Task 1.
- Produces: `Config.method` in `{"adamw","cadam","protect-m","protect-mv"}` and method-aware `_build_optimizer`.
- Produces: noisy public bundles for seeds `1301,1302,1303` and runner train seeds `301,302,303`.

- [ ] **Step 1: Write failing runner-contract tests**

```python
@pytest.mark.parametrize("method", ["adamw", "cadam", "protect-m", "protect-mv"])
def test_build_optimizer_uses_frozen_method(method, tiny_model):
    config = Config(seed=301, augmentation_seed=10301, epochs=20,
                    learning_rate=3e-4, weight_decay=0.1, method=method)
    optimizer = _build_optimizer(tiny_model, config)
    assert resolved_method(optimizer) == method

def test_pilot_seed_contract():
    assert PILOT_BINDINGS == {301: (1301, 10301), 302: (1302, 10302), 303: (1303, 10303)}
```

- [ ] **Step 2: Extend config and CLIs without changing old defaults**

Add `method: str = "adamw"`; accept seeds `301–303`; accept `--method`; keep prior seeds and AdamW behavior valid. Append `optimizer_diagnostics.jsonl` only for optimizers that expose diagnostics. Include the method and frozen candidate fields in `config.json`, `summary.json`, and checkpoint state.

- [ ] **Step 3: Extend public noise generation seed admission**

Allow only the existing seeds plus `1301,1302,1303`; retain exact 16,000 corruptions, 160 per class, seal validation, and private/public separation.

- [ ] **Step 4: Verify Task 2**

Run: `uv run --locked pytest tests/test_m1_pilot_runner.py tests/test_m1_noisy_data.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add experiments/m1_calibration_clean.py experiments/m1_calibration_noisy.py \
  experiments/m1_noisy_data.py tests/test_m1_pilot_runner.py
git commit -m "research: add frozen M1 pilot methods"
```

### Task 3: Sentinel and mechanical Pilot gate

**Files:**
- Create: `experiments/m1_candidate_sentinel.py`
- Create: `experiments/m1_pilot.py`
- Extend: `tests/test_m1_pilot_runner.py`

**Interfaces:**
- Produces: `SENTINEL_PASS.json` only when functional parity, state transitions, resume, information isolation, and both candidate timing ratios pass.
- Produces: `M1_PILOT_DECISION.json` with `NO-GO`, `INCONCLUSIVE`, or `PILOT-GO` from complete artifacts.

- [ ] **Step 1: Write failing gate tests**

```python
def test_matrix_is_exactly_24_cells():
    assert len(build_matrix()) == 24

def test_no_candidate_can_pass_with_one_nonpositive_seed():
    result = evaluate_candidate(candidate_rows(one_noisy_delta=0.0))
    assert result["eligible"] is False

def test_dispatch_requires_sentinel(tmp_path):
    with pytest.raises(RuntimeError, match="sentinel"):
        run_pilot(output=tmp_path, sentinel=tmp_path / "missing.json")
```

- [ ] **Step 2: Implement the GPU sentinel**

Use the real ViT-LoRA trainable set. Check five-step disabled-protection parity, the frozen reachability state hashes, skipped-nonfinite atomicity, checkpoint restart, one-forward/one-backward counters, and 10 warm-up plus 50 timed steps repeated five times. Synchronize CUDA at timing boundaries and reject median candidate/AdamW ratios over `1.05`.

- [ ] **Step 3: Implement serial resumable Pilot orchestration**

Run methods in the fixed order `adamw,cadam,protect-m,protect-mv`, then conditions `clean,noisy`, then seeds `301,302,303`. Skip only sealed successful cells with matching configuration/source hashes. Stop on a failed cell and emit `INCONCLUSIVE`; do not replace or rerun outcome-bearing cells under a changed configuration.

- [ ] **Step 4: Implement the frozen decision function**

Read final-epoch tuning accuracy, paired elapsed time, peak VRAM, and candidate diagnostics. Apply every Global Constraint mechanically, choose Protect-M unless Protect-MV exceeds it by at least `0.5pp` on mean noisy accuracy with no worse seed and no extra limit violation, and record `test_loaded=false`.

- [ ] **Step 5: Verify Task 3 and the whole local suite**

Run: `uv run --locked pytest tests/test_m1_optimizers.py tests/test_m1_pilot_runner.py -q`
Run: `uv run --locked ruff check experiments/m1_optimizers.py experiments/m1_candidate_sentinel.py experiments/m1_pilot.py experiments/m1_calibration_clean.py experiments/m1_calibration_noisy.py tests/test_m1_optimizers.py tests/test_m1_pilot_runner.py`
Expected: all tests and lint checks pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add experiments/m1_candidate_sentinel.py experiments/m1_pilot.py tests/test_m1_pilot_runner.py
git commit -m "research: gate and orchestrate Protect-M pilot"
```

### Task 4: Windows deployment, sentinel, and conditional dispatch

**Files:**
- No repository files beyond Tasks 1–3.
- Remote outputs: `C:\arw-results\m1-candidate-sentinel` and `C:\arw-results\m1-pilot`.

**Interfaces:**
- Consumes: the committed source hash from Task 3 and existing sealed CIFAR/noise stores.
- Produces: a scheduled task that continuously resumes the 24 cells only after the sentinel passes.

- [ ] **Step 1: Push and update the Windows worktree to the exact source commit**

Verify clean target code paths, `git rev-parse HEAD`, `uv sync --locked --group dev`, CUDA availability, 12-core process affinity, and no competing experiment runner.

- [ ] **Step 2: Generate and validate Pilot noise bundles**

Generate `1301–1303`, run public seal validation, and verify each private audit reports exactly 16,000 corruptions and 160 per class without exposing private fields to training.

- [ ] **Step 3: Run the sentinel synchronously**

Do not create the Pilot task unless `SENTINEL_PASS.json` exists, source/config hashes match, candidate ratios are each at most `1.05`, and `test_loaded=false`.

- [ ] **Step 4: Create and start the resumable Pilot scheduled task**

Use one serial process, no GPU power cap, and CPU affinity limited to 12 logical cores. The task starts at logon/boot and resumes sealed incomplete cells, so closing SSH does not stop the experiment.

- [ ] **Step 5: Verify live execution**

Confirm the scheduled task reports running/success, exactly one Pilot Python process exists, GPU utilization becomes nonzero during forward/backward work, the first cell writes metrics, and no test artifact is loaded.
