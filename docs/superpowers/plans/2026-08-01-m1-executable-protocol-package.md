# M1 Executable Protocol Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the non-GPU parts of M1 as executable, tested protocol code.

**Architecture:** One strict model module, one numerical protocol module, two committed specs, generated schemas, and one focused test file. No trainer, CLI platform, data download, candidate recurrence, or Windows mutation.

**Tech Stack:** Python 3.12, Pydantic 2.13.4, NumPy 2.5.1, SciPy 1.18.0, pytest, `uv run --locked`.

## Global Constraints

- Preserve M1 split, seeds, gates, and confirmation rules.
- Calibration uses the fixed AdamW anchor before tuning.
- Baseline grids contain exactly 12 configurations per method.
- Formal noise is 160/400 per class and 16,000/40,000 total.
- Power uses separate noisy and clean paired differences from exactly 8 seeds.
- No GPU, Windows, CIFAR/model download, candidate implementation, or `arw` integration.
- Do not touch user-owned untracked files.

---

### Task 1: Fix the calibration order

**Files:**
- Modify: `docs/superpowers/specs/2026-07-31-m06-m1-quality-first-experiment-design.md`

- [ ] Replace `Calibration uses tuned-default AdamW` with the fixed anchor
  `lr=3e-4, wd=0.01, betas=(0.9,0.999), eps=1e-8, amsgrad=false`.
- [ ] State the only valid order: calibration -> fixed steps/`G_ref` -> AdamW tuning -> other grids.
- [ ] Run `git diff --check` and commit `docs: remove M1 calibration tuning cycle`.

### Task 2: Implement specs, grids, and schemas

**Files:**
- Create: `experiments/m1_protocol/__init__.py`
- Create: `experiments/m1_protocol/models.py`
- Create: `experiments/m1_protocol/protocol.py`
- Create: `experiments/m1_protocol/specs/baseline_grids.json`
- Create: `experiments/m1_protocol/specs/noise_protocol.json`
- Create: `experiments/m1_protocol/schemas/baseline_grids.schema.json`
- Create: `experiments/m1_protocol/schemas/noise_protocol.schema.json`
- Test: `tests/test_m1_protocol.py`

**Interfaces:**

```python
load_baseline_spec(path: Path) -> BaselineSpec
load_noise_spec(path: Path) -> NoiseSpec
resolve_grids(spec: BaselineSpec, selected_adamw_id: str, g_ref: float) -> dict[str, tuple[dict[str, object], ...]]
write_schemas(directory: Path) -> None
```

- [ ] Write tests asserting strict extra-field rejection, exact grid values/counts, inherited LR/WD,
  Clip `G_ref * 2**e`, and schema byte regeneration.
- [ ] Run `uv run --locked pytest tests/test_m1_protocol.py -q`; expect import failure.
- [ ] Implement the two strict Pydantic specs, canonical JSON/hash helpers, deterministic resolver, and
  schema writer with no candidate JSON field.
- [ ] Generate schemas, rerun the test, run Ruff, and commit `feat: freeze M1 protocol grids`.

### Task 3: Implement exact-rate noise and bundle separation

**Files:**
- Modify: `experiments/m1_protocol/protocol.py`
- Modify: `tests/test_m1_protocol.py`

**Interfaces:**

```python
derive_rng(noise_seed: int, domain: str) -> np.random.Generator
calibrate_pi(q: np.ndarray, target: int) -> np.ndarray
systematic_pps(pi: np.ndarray, order: np.ndarray, start: float) -> np.ndarray
destination_probabilities(feature: np.ndarray, matrix: np.ndarray, clean_label: int) -> np.ndarray
generate_noise(images: np.ndarray, labels: np.ndarray, ids: np.ndarray, noise_seed: int, spec: NoiseSpec) -> NoiseArrays
write_noise_bundles(arrays: NoiseArrays, training_dir: Path, audit_dir: Path, metadata: dict[str, str]) -> None
validate_noise_bundle(training_dir: Path, audit_dir: Path) -> None
```

- [ ] Write tests for RNG domain separation, exact target sum, exact-k PPS, excluded clean destination,
  exact per-class counts, public/private field separation, and hash tampering.
- [ ] Implement SHA-256/PCG64DXSM domains, 128-step logit-shift bisection, systematic PPS,
  standardized/L2 pixel features, Xia-style destination matrices, and fail-closed array checks.
- [ ] Store only IDs/noisy labels publicly; store clean labels, masks, q/pi, destination/transition,
  channel statistics, seed, and digests privately. Use `.npy`, `allow_pickle=False`, and SHA-256.
- [ ] Run the focused tests and Ruff; commit `feat: add exact-rate M1 noise protocol`.

### Task 4: Implement paired power and finish verification

**Files:**
- Modify: `experiments/m1_protocol/models.py`
- Modify: `experiments/m1_protocol/protocol.py`
- Modify: `tests/test_m1_protocol.py`
- Create: `docs/research-direction/17_m1_executable_protocol.md`

**Interfaces:**

```python
superiority_power(n: int, sigma_pp: float, delta_pp: float = 1.0) -> float
noninferiority_power(n: int, sigma_pp: float, margin_pp: float = 0.5) -> float
plan_post_m1(value: PostM1PowerInput) -> PostM1PowerResult
```

- [ ] Write tests for reference sample sizes `(5,11)`, `(13,36)`, `(44,139)`, separate noisy/clean
  variance, GO-only eight-seed input, floor/inflation, and one 200,000-draw Monte Carlo check per test.
- [ ] Implement exact SciPy chi-square/noncentral-t formulas and integer search from `n=5`; omit `n_cap`.
- [ ] Document the four-stage artifact order, validation commands, bundle boundary, and power gate.
- [ ] Run:

```bash
uv run --locked pytest tests/test_m1_protocol.py -q
uv run --locked ruff format --check experiments/m1_protocol tests/test_m1_protocol.py
uv run --locked ruff check experiments/m1_protocol tests/test_m1_protocol.py
uv run --locked pytest -q -m "not stage_a2_integration"
git diff --check
```

- [ ] Commit `research: add executable M1 protocol package` and report exact checks. Do not launch M1.
