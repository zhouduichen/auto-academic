# Protect-M1.1 Bounded Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run one fail-closed, seed-301, three-epoch rescue test for bounded first-moment anomaly protection.

**Architecture:** Add a separate `ProtectM11AdamW` optimizer so the failed frozen M1 implementation and its 24-cell evidence remain unchanged. Reuse the clean/noisy runners through one extra method name, add a candidate-only CUDA sentinel, and add a rescue orchestrator that compares the two new cells against existing epoch-3 baselines and writes one mechanical GO/NO-GO artifact.

**Tech Stack:** Python 3.12, PyTorch 2.11, pytest, Ruff, Windows PowerShell, CUDA.

## Global Constraints

- Run only seed 301 for three epochs in clean and opaque-noisy conditions.
- Reuse existing AdamW and CAdam seed-301 trajectories; do not rerun baselines.
- Warm up for exactly 1,250 successful steps with exact AdamW moment writes.
- Use `alpha=0.99`, `threshold=3.0`, and scale floor `1e-6` without a tuning grid.
- Hold only the persistent first moment; always commit the second moment and advance parameters and clock.
- Consecutive rejected writes are impossible.
- Load no test data and no private noise-audit fields; every artifact records `test_loaded=false`.
- Stop permanently on any failed acceptance check; do not dispatch extra seeds or epochs.
- Windows uses at most 12 CPU cores and has no GPU power cap.

---

### Task 1: Bounded-Anomaly Optimizer

**Files:**
- Modify: `experiments/m1_optimizers.py`
- Create: `tests/test_m11_optimizer.py`

**Interfaces:**
- Consumes: existing AdamW parameter grouping and `ProtectAdamW.step` numerical update order.
- Produces: `ProtectM11AdamW(params, *, lr, betas, eps, weight_decay, protection_enabled=True, alpha=0.99, threshold=3.0, warmup_steps=1250)` and `diagnostics_summary(reset=False)`.

- [ ] **Step 1: Write failing warm-up and bounded-rejection tests**

```python
def test_m11_warmup_matches_adamw_and_rejection_is_single_step() -> None:
    reference_p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    candidate_p = torch.nn.Parameter(reference_p.detach().clone())
    reference = torch.optim.AdamW([reference_p], lr=1e-2, foreach=False, fused=False)
    candidate = ProtectM11AdamW([candidate_p], lr=1e-2, warmup_steps=2)
    for gradient in (torch.tensor([0.2, -0.1]), torch.tensor([-0.1, 0.3])):
        reference_p.grad = gradient.clone()
        candidate_p.grad = gradient.clone()
        reference.step()
        candidate.step()
    _assert_adam_state_equal(reference, candidate, [reference_p], [candidate_p])
    candidate.set_detector_state_for_test(mu=0.0, scale=0.1, previous_rejected=False, steps=2)
    states = [candidate.observe_score_for_test(-1.0), candidate.observe_score_for_test(-1.0)]
    assert states[0]["reject"] is True
    assert states[1]["reject"] is False


def test_m11_rejection_holds_only_first_moment() -> None:
    optimizer, parameter = _forced_m11_rejection()
    before = copy.deepcopy(optimizer.state[parameter])
    parameter.grad = torch.tensor([-1.0, 0.5])
    optimizer.step()
    after = optimizer.state[parameter]
    assert torch.equal(after["exp_avg"], before["exp_avg"])
    assert not torch.equal(after["exp_avg_sq"], before["exp_avg_sq"])
    assert after["step"].item() == before["step"].item() + 1
```

- [ ] **Step 2: Run the focused tests and confirm import failure**

Run: `uv run --locked pytest tests/test_m11_optimizer.py -q`

Expected: FAIL because `ProtectM11AdamW` does not exist.

- [ ] **Step 3: Implement the separate optimizer and serialized detector state**

Add a subclass with a distinct mechanism ID and device-state transition:

```python
class ProtectM11AdamW(ProtectAdamW):
    mechanism_id = "protect_m11_bounded_anomaly_v1"
    schema_version = 1

    def __init__(self, params: ParamsT, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, *,
                 protection_enabled: bool = True, alpha: float = 0.99,
                 threshold: float = 3.0, warmup_steps: int = 1250) -> None:
        super().__init__(params, lr, betas, eps, weight_decay,
                         protection_target="m", protection_enabled=protection_enabled)
        if not 0 <= alpha < 1 or threshold <= 0 or warmup_steps < 0:
            raise ValueError("invalid M1.1 detector configuration")
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        self.mu = 0.0
        self.scale = 1.0
        self.previous_rejected = False
        self.successful_steps = 0
        self._device_detector = None

    def _next_detector_device(self, score: Tensor) -> dict[str, Tensor]:
        state = self._ensure_m11_device_detector(score.device)
        next_steps = state["successful_steps"] + 1
        z = (score - state["mu"]) / torch.clamp(state["scale"], min=1e-6)
        eligible = next_steps > self.warmup_steps
        reject = eligible & (z <= -self.threshold) & (~state["previous_rejected"])
        mu = self.alpha * state["mu"] + (1 - self.alpha) * score
        scale = self.alpha * state["scale"] + (1 - self.alpha) * torch.abs(
            score - state["mu"]
        )
        return {
            "mu": mu,
            "scale": scale,
            "successful_steps": next_steps,
            "previous_rejected": reject,
            "admit": ~reject,
            "d": z,
            "transition_off": reject,
            "transition_on": state["previous_rejected"],
        }
```

Override detector commit/materialization and checkpoint load so the protected state contains exactly `schema_version`, `mechanism_id`, `protection_enabled`, `alpha`, `threshold`, `warmup_steps`, `mu`, `scale`, `previous_rejected`, and `successful_steps`. Reuse the inherited parameter update, nonfinite atomicity, device diagnostics, and first-moment-only commit path.

Use these exact state boundaries:

```python
def _commit_device_detector(self, state: dict[str, Tensor]) -> None:
    self._device_detector = {
        key: state[key]
        for key in ("mu", "scale", "successful_steps", "previous_rejected")
    }

def protect_state_dict(self) -> dict[str, object]:
    self._sync_detector_to_host()
    return {
        "schema_version": 1,
        "mechanism_id": self.mechanism_id,
        "protection_enabled": self.protection_enabled,
        "alpha": self.alpha,
        "threshold": self.threshold,
        "warmup_steps": self.warmup_steps,
        "mu": self.mu,
        "scale": self.scale,
        "previous_rejected": self.previous_rejected,
        "successful_steps": self.successful_steps,
    }
```

`load_state_dict` must validate exact keys and constants before calling
`Optimizer.load_state_dict`; on any exception restore both the base optimizer payload and
the prior M1.1 detector payload. Validate `mu` as finite, `scale` as finite and positive,
`previous_rejected` as bool, and `successful_steps` as a nonnegative integer.

- [ ] **Step 4: Add resume, nonfinite, diagnostics, and no-host-sync tests**

```python
def test_m11_resume_is_bit_exact() -> None:
    optimizer, parameter = _m11_after_fixture_steps()
    payload = copy.deepcopy(optimizer.state_dict())
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = ProtectM11AdamW([resumed_parameter])
    resumed.load_state_dict(payload)
    for gradient in (torch.tensor([-0.4]), torch.tensor([0.3])):
        parameter.grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()
        optimizer.step()
        resumed.step()
    assert torch.equal(parameter, resumed_parameter)
    assert optimizer.state_dict()["protect_state"] == resumed.state_dict()["protect_state"]


def test_m11_cuda_step_has_no_direct_item() -> None:
    assert ".item()" not in inspect.getsource(ProtectM11AdamW._next_detector_device)
```

- [ ] **Step 5: Run optimizer tests and lint**

Run: `uv run --locked pytest tests/test_m1_optimizers.py tests/test_m11_optimizer.py -q`

Expected: PASS.

Run: `uv run --locked ruff check experiments/m1_optimizers.py tests/test_m11_optimizer.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit the optimizer**

```bash
git add experiments/m1_optimizers.py tests/test_m11_optimizer.py
git commit -m "feat: add bounded M1.1 moment protection"
```

### Task 2: Runner Integration Without Expanding the Frozen Pilot

**Files:**
- Modify: `experiments/m1_calibration_clean.py`
- Modify: `experiments/m1_calibration_noisy.py`
- Modify: `tests/test_m1_pilot_runner.py`

**Interfaces:**
- Consumes: `ProtectM11AdamW` from Task 1.
- Produces: `RUNNER_METHODS = (*PILOT_METHODS, "protect-m11")`; `_build_optimizer(..., method="protect-m11")`.

- [ ] **Step 1: Write a failing builder/frozen-matrix test**

```python
def test_m11_is_runner_only_and_does_not_expand_frozen_pilot() -> None:
    assert "protect-m11" in RUNNER_METHODS
    assert "protect-m11" not in PILOT_METHODS
    assert len(build_matrix()) == 24
    config = Config(seed=301, augmentation_seed=10_301, epochs=3,
                    method="protect-m11", device="cpu")
    assert isinstance(_build_optimizer(TinyModel(), config), ProtectM11AdamW)
```

- [ ] **Step 2: Confirm the test fails**

Run: `uv run --locked pytest tests/test_m1_pilot_runner.py::test_m11_is_runner_only_and_does_not_expand_frozen_pilot -q`

Expected: FAIL because `RUNNER_METHODS` is undefined.

- [ ] **Step 3: Add the runner-only method**

```python
PILOT_METHODS = ("adamw", "cadam", "protect-m", "protect-mv")
RUNNER_METHODS = (*PILOT_METHODS, "protect-m11")

# inside _build_optimizer
if config.method == "protect-m11":
    return ProtectM11AdamW(groups, **common, alpha=0.99,
                           threshold=3.0, warmup_steps=1250)
```

Use `RUNNER_METHODS` only for the two CLI `--method` choices. Keep `PILOT_METHODS` in `m1_pilot.build_matrix()`.

- [ ] **Step 4: Run runner tests and lint**

Run: `uv run --locked pytest tests/test_m1_calibration_clean.py tests/test_m1_pilot_runner.py -q`

Expected: PASS, including the 24-cell invariant.

Run: `uv run --locked ruff check experiments/m1_calibration_clean.py experiments/m1_calibration_noisy.py tests/test_m1_pilot_runner.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit runner integration**

```bash
git add experiments/m1_calibration_clean.py experiments/m1_calibration_noisy.py tests/test_m1_pilot_runner.py
git commit -m "feat: expose M1.1 to calibration runners"
```

### Task 3: Candidate Sentinel and Mechanical Rescue Decision

**Files:**
- Create: `experiments/m11_candidate_sentinel.py`
- Create: `experiments/m11_rescue.py`
- Create: `tests/test_m11_rescue.py`

**Interfaces:**
- Consumes: clean/noisy runners, `ProtectM11AdamW`, existing baseline result root, and the opaque seed-1301 public bundle.
- Produces: `M11_SENTINEL_RESULT.json`, `M11_SENTINEL_PASS.json`, and `M11_RESCUE_DECISION.json`.

- [ ] **Step 1: Write failing pure decision tests**

```python
def test_rescue_decision_requires_every_gate() -> None:
    record = {
        "clean_difference_pp": -1.0,
        "noisy_difference_pp": 0.2,
        "noisy_rejected_steps": 10,
        "post_warmup_steps": 2500,
        "max_consecutive_rejections": 1,
        "wallclock_ratios": [1.02, 1.03],
        "vram_ratios": [1.0, 1.0],
        "test_loaded": False,
    }
    assert evaluate_rescue(record)["decision"] == "GO"
    record["noisy_rejected_steps"] = 0
    assert evaluate_rescue(record)["decision"] == "NO-GO"
```

- [ ] **Step 2: Confirm the decision test fails**

Run: `uv run --locked pytest tests/test_m11_rescue.py -q`

Expected: FAIL because `experiments.m11_rescue` is absent.

- [ ] **Step 3: Implement the mechanical evaluator**

```python
def evaluate_rescue(record: dict[str, object]) -> dict[str, object]:
    rejected = int(record["noisy_rejected_steps"])
    post = int(record["post_warmup_steps"])
    checks = {
        "clean_drop_at_most_2pp": float(record["clean_difference_pp"]) >= -2.0,
        "noisy_at_least_cadam": float(record["noisy_difference_pp"]) >= 0.0,
        "noisy_has_rejection": rejected >= 1,
        "noisy_rejection_rate_at_most_5pct": rejected / post <= 0.05,
        "no_consecutive_rejections": int(record["max_consecutive_rejections"]) <= 1,
        "wallclock_overhead_at_most_5pct": all(
            float(value) <= 1.05 for value in record["wallclock_ratios"]
        ),
        "vram_overhead_at_most_5pct": all(
            float(value) <= 1.05 for value in record["vram_ratios"]
        ),
        "test_not_loaded": record["test_loaded"] is False,
    }
    return {**record, "checks": checks,
            "decision": "GO" if all(checks.values()) else "NO-GO"}
```

- [ ] **Step 4: Implement the candidate-only CUDA sentinel**

Reuse the exact fixed-batch and synchronized timer functions by import, then apply this gate:

```python
from experiments.m1_candidate_sentinel import _fixed_batch, _time_method

functional = _functional_checks(images, labels, warmup_steps=2)
timings = {
    method: _time_method(method, images, labels)
    for method in ("adamw", "protect-m11")
}
time_ratio = (
    float(timings["protect-m11"]["median_step_seconds"])
    / float(timings["adamw"]["median_step_seconds"])
)
vram_ratio = (
    float(timings["protect-m11"]["peak_vram_gb"])
    / float(timings["adamw"]["peak_vram_gb"])
)
passed = bool(functional["passed"]) and time_ratio <= 1.05 and vram_ratio <= 1.05
result = {
    "status": "passed" if passed else "failed",
    "source_commit": source_commit(),
    "functional": functional,
    "timings": timings,
    "time_ratio": time_ratio,
    "vram_ratio": vram_ratio,
    "test_loaded": False,
}
clean._write_json(args.output / "M11_SENTINEL_RESULT.json", result)
if passed:
    clean._write_json(args.output / "M11_SENTINEL_PASS.json", result)
if not passed:
    raise SystemExit(2)
```

`_functional_checks` runs five disabled-protection steps against for-loop AdamW, forces one
post-warm-up anomaly followed by another identical score and requires reject/accept, saves
and resumes the detector plus optimizer state, and injects one NaN gradient before mutation.

- [ ] **Step 5: Implement the two-cell rescue orchestrator**

The CLI accepts `--data-dir`, `--image-store`, `--tuning-store`, `--bundle-301`, `--sentinel`, `--baseline-root`, and `--output`. Validate the sentinel against the current commit, then run:

```python
config = clean.Config(seed=301, augmentation_seed=10_301, epochs=3,
                      learning_rate=3e-4, weight_decay=0.1,
                      workers=4, device="cuda", method="protect-m11")
clean.run(config, args.data_dir, args.output / "clean-seed301")
noisy.run(config, args.bundle_301, args.image_store, args.tuning_store,
          args.output / "noisy-seed301")
```

Read epoch 3 from candidate clean/noisy metrics, existing AdamW clean/noisy metrics, and existing CAdam noisy metrics. Sum candidate noisy rejection counts after warm-up, compute the two elapsed-time and peak-VRAM ratios against AdamW, call `evaluate_rescue`, and atomically write `M11_RESCUE_DECISION.json`. Exit 2 on `NO-GO`; never dispatch another cell.

- [ ] **Step 6: Run unit tests and lint**

Run: `uv run --locked pytest tests/test_m11_rescue.py tests/test_m11_optimizer.py tests/test_m1_pilot_runner.py -q`

Expected: PASS.

Run: `uv run --locked ruff check experiments/m11_candidate_sentinel.py experiments/m11_rescue.py tests/test_m11_rescue.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit sentinel and rescue runner**

```bash
git add experiments/m11_candidate_sentinel.py experiments/m11_rescue.py tests/test_m11_rescue.py
git commit -m "feat: add bounded M1.1 rescue gate"
```

### Task 4: Regression Gate and Windows Execution

**Files:**
- Verify: `experiments/m1_optimizers.py`
- Verify: `experiments/m1_calibration_clean.py`
- Verify: `experiments/m1_calibration_noisy.py`
- Verify: `experiments/m11_candidate_sentinel.py`
- Verify: `experiments/m11_rescue.py`

**Interfaces:**
- Consumes: all Task 1–3 deliverables and the Windows CIFAR/public-noise stores.
- Produces: one provenance-bound Windows decision artifact; no follow-on dispatch.

- [ ] **Step 1: Run the complete relevant local gate**

Run:

```bash
uv run --locked pytest \
  tests/test_m1_optimizers.py tests/test_m11_optimizer.py \
  tests/test_m1_calibration_clean.py tests/test_m1_noisy_data.py \
  tests/test_m1_pilot_runner.py tests/test_m11_rescue.py -q
uv run --locked ruff check \
  experiments/m1_optimizers.py experiments/m1_calibration_clean.py \
  experiments/m1_calibration_noisy.py experiments/m11_candidate_sentinel.py \
  experiments/m11_rescue.py tests/test_m11_optimizer.py tests/test_m11_rescue.py
```

Expected: all tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 2: Push and deploy the exact commit**

```bash
git push origin HEAD:codex/m05-direct-experiment
```

On Windows, fetch with proxy bypass and check out the exact full SHA. Use only
`C:\arw-m0\.venv\Scripts\python.exe`; do not run `uv run --locked` on Windows.

- [ ] **Step 3: Generate one provenance-bound noisy bundle**

Resolve `$shortSha = (git rev-parse --short=7 HEAD)` after checkout. Prepare
`C:\arw-data\m11-rescue-$shortSha`, generate only noise seed 1301 with opaque ID
`pilot-1301`, and validate it. Require `sample_count=40000`, the exact source commit, and
`test_loaded=false`.

- [ ] **Step 4: Run the sentinel under the hardware contract**

Set process affinity to `0xFFF` (12 cores), leave GPU power unrestricted, and run:

```powershell
C:\arw-m0\.venv\Scripts\python.exe -m experiments.m11_candidate_sentinel `
  --data-dir C:\arw-data\cifar100 `
  --output C:\arw-results\m11-candidate-sentinel
```

Do not run rescue cells unless `M11_SENTINEL_PASS.json` exists and matches the deployed commit.

- [ ] **Step 5: Run exactly two rescue cells and stop**

```powershell
C:\arw-m0\.venv\Scripts\python.exe -m experiments.m11_rescue `
  --data-dir C:\arw-data\cifar100 `
  --image-store C:\arw-data\m11-rescue-$shortSha\image-store `
  --tuning-store C:\arw-data\m11-rescue-$shortSha\tuning-store `
  --bundle-301 C:\arw-data\m11-rescue-$shortSha\training\pilot-1301 `
  --sentinel C:\arw-results\m11-candidate-sentinel\M11_SENTINEL_PASS.json `
  --baseline-root C:\arw-results\m1-pilot `
  --output C:\arw-results\m11-rescue
```

Expected: `M11_RESCUE_DECISION.json` is written after the clean and noisy three-epoch cells. Report GO or NO-GO and dispatch nothing else.

- [ ] **Step 6: Commit any verification-only correction separately**

If Windows exposes a correctness defect, reproduce it with a failing local test, make the minimum correction, rerun the complete gate, and commit with `fix: correct M1.1 Windows validation`. Performance or scientific gate failures are results, not reasons to relax a threshold.
