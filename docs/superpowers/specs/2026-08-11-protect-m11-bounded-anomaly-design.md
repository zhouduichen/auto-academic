# M1.1 Bounded Anomaly Protection Design

## Decision

The completed M1 Pilot is a `NO-GO` for Protect-M and Protect-MV. Across every candidate
cell, the detector admitted only the first three optimizer steps, switched off once, and
never switched on again. Protect-M therefore held the first moment for almost the entire
run; Protect-MV held both moments. M1.1 is one bounded rescue test, not an expansion of the
failed Pilot.

## Scope

M1.1 tests one candidate, `protect-m11`, on seed 301 for three epochs in the clean and
existing opaque-noisy conditions. Existing AdamW and CAdam seed-301 trajectories are reused;
they are not rerun. The test never loads the test set or private noise audit fields. Failure
of any acceptance check permanently closes this M1 line.

## Optimizer

The AdamW parameter update and second-moment write are unchanged. Only the persistent
first-moment write may be held.

For each successful optimizer step, compute the existing global FP32 alignment score `q`.
Maintain device-resident detector state:

```text
mu_0 = 0
scale_0 = 1
alpha = 0.99
warmup_steps = 1250
threshold = 3.0

z_t = (q_t - mu_(t-1)) / max(scale_(t-1), 1e-6)
mu_t = alpha * mu_(t-1) + (1 - alpha) * q_t
scale_t = alpha * scale_(t-1) + (1 - alpha) * abs(q_t - mu_(t-1))
reject_t = (t > warmup_steps) and (z_t <= -threshold) and not reject_(t-1)
```

During warm-up, both moments commit exactly as AdamW while detector statistics accumulate.
After warm-up, a rejected step holds only the persistent first moment. The second moment,
parameter update, and optimizer clock always advance using the ephemeral AdamW candidates.
`not reject_(t-1)` makes consecutive holds impossible. Detector statistics update whether
the step is admitted or rejected.

Checkpoint state records the schema/mechanism ID, `mu`, `scale`, previous rejection flag,
successful-step count, and fixed constants. Loading remains fail-closed. CUDA `step` performs
no host scalar materialization; diagnostics are materialized only at epoch boundaries.

## Acceptance and Stop Rule

The two three-epoch cells must satisfy all of the following:

- clean tuning accuracy at epoch 3 is no more than 2.0 percentage points below the existing
  AdamW seed-301 trajectory;
- noisy tuning accuracy at epoch 3 is at least the existing CAdam seed-301 trajectory;
- the noisy cell records at least one rejection, rejects at most 5% of post-warm-up steps,
  and never rejects consecutive steps;
- whole-step wall-clock overhead versus AdamW is at most 5%, and peak VRAM overhead is at
  most 5%;
- functional parity when protection is disabled, nonfinite atomic rejection, checkpoint
  resume, deterministic replay, and `test_loaded=false` all pass.

If any check fails, no threshold sweep, extra seed, longer run, or Protect-MV variant is
allowed. If all checks pass, the result authorizes a separate review before any full Pilot;
it does not automatically dispatch more computation.

## Verification

CPU tests cover detector arithmetic, warm-up parity, isolated rejection, the no-consecutive
hold invariant, first-moment-only persistence, serialization, and nonfinite atomicity. A
short Windows sentinel verifies CUDA functionality and the 5% resource gates before the two
three-epoch cells run under the existing 12-core CPU limit and unrestricted GPU power.
