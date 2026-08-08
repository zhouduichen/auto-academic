# Protect-M / Protect-MV Candidate Mechanism Design

> Date: 2026-08-01
> Status: user-approved candidate recurrence and single Pilot default; implementation authorized
> Scope: exact optimizer recurrence, state contract, precision semantics, and implementation sentinels
> Working names: Protect-M and Protect-MV are experimental labels, not proposed paper names

## 1. Decision and Evidence Boundary

M0.6 has formally passed its preregistered confirmation rule. That result authorizes writing
and reviewing a candidate mechanism, but it does not select detector constants or the eventual
Protect-M versus Protect-MV structure. No M0.6 effect estimate, trajectory, confidence interval,
or admission-rate simulation was used to choose the Pilot default in this document.

This specification instantiates the Candidate Freeze Gate in
[`2026-07-31-m06-m1-quality-first-experiment-design.md`](2026-07-31-m06-m1-quality-first-experiment-design.md)
and the candidate interface in
[`2026-08-01-m1-executable-protocol-package-design.md`](2026-08-01-m1-executable-protocol-package-design.md).
It freezes one common detector and one common ephemeral parameter-update path. The only
algorithmic difference between the two candidates is the persistent state commit target:

| Candidate | Rejected step: persistent `m` | Rejected step: persistent `v` |
|---|---|---|
| Protect-M | hold | commit the ephemeral candidate |
| Protect-MV | hold | hold |

An admitted step commits both moment candidates for both methods. The current batch always
affects the current parameter update through the complete ephemeral AdamW candidate,
independently of the admission decision.

User approval on 2026-08-08 authorizes the minimum candidate implementation, CPU tests, GPU
sentinel, and—only if both sentinels pass—the frozen 24-run M1 Pilot. It does not authorize a
candidate grid, candidate tuning, development-gate access, confirmation-gate access, or
official-test access.

## 2. Alternatives Considered and Frozen Choice

Three intervention points were considered:

1. **Whole-buffer hold/commit after an ephemeral AdamW step — selected.** The current
   gradient forms complete first- and second-moment candidates and the current parameter
   update. Admission controls only which candidate moment tensors persist into the next
   optimizer step.
2. **Decay-only rejection — rejected.** Updating `m <- beta1*m` or `v <- beta2*v` on a
   rejected step would still let that step shrink persistent memory. It would not implement
   the state-level hold contract required here.
3. **Parameter-step masking or attenuation — rejected.** This would remove or weaken the
   immediate effect of a difficult batch and violate the Candidate Freeze Gate.

The selected mechanism is a scalar, optimizer-wide admission gate. It does not make a
per-layer, per-tensor, per-coordinate, or per-example admission decision. Such variants are
different mechanisms and require a new reviewed design.

## 3. Symbols and Supported AdamW Surface

For optimizer call `t`, let `A_t` be the set of parameters with a dense gradient. For each
`i in A_t`:

- `theta_{i,t-1}` is the FP32 trainable/master parameter before the step;
- `g_{i,t}` is its accumulated, unscaled FP32 gradient;
- `m_{i,t-1}` and `v_{i,t-1}` are persistent FP32 `exp_avg` and `exp_avg_sq`;
- `s_{i,t-1}` is the parameter's standard AdamW `step` counter;
- `eta_{i,t}`, `lambda_i`, `beta1_i`, `beta2_i`, and `epsilon_i` are the active parameter
  group's standard AdamW values.

The first implementation surface is intentionally narrow:

- dense FP32 trainable/master parameters and FP32 moment state;
- one or more parameter groups, including different learning rates, weight decay, betas,
  and epsilon values;
- `amsgrad=False`, `maximize=False`, `foreach=False`, `fused=False`,
  `capturable=False`, and `differentiable=False`;
- no sparse gradients.

Unsupported options or a sparse gradient fail before any parameter, optimizer, detector,
or scheduler state changes. Parameters whose gradient is `None` are omitted from `A_t` and
receive no step increment, moment initialization, moment change, parameter update, or weight
decay, matching for-loop AdamW semantics.

## 4. Persistent State

### 4.1 Per-parameter AdamW state

Each parameter uses the standard keys:

```text
step        : integer-valued scalar tensor
exp_avg     : FP32 tensor with parameter shape
exp_avg_sq  : FP32 tensor with parameter shape
```

`step` is a successful optimizer-call clock, not an admission counter. For every
`i in A_t`, it increments exactly once on every finite, non-skipped optimizer step whether
the gate admits or rejects the persistent moment candidates. This is deliberate: M0.6 held
the step counter common while replacing moment carriers, and this mechanism protects only
`m` or `m+v`, not the optimizer clock.

### 4.2 Optimizer-wide detector state

Both candidates have exactly the same detector state:

```text
fast_ema       : FP32 scalar F
slow_ema       : FP32 scalar S
admit           : boolean a
bad_streak      : integer in {0, 1}
good_streak     : integer in {0, 1}
```

Initialization before the first successful gradient-bearing step is:

```text
F = 1
S = 1
a = true
bad_streak = 0
good_streak = 0
```

The fixed two-observation transition rule in Section 6 is part of the mechanism, not a
tunable hyperparameter. A single batch therefore cannot change `a` by itself.

## 5. Shared Consistency Detector

### 5.1 Observed signal

The detector uses only the current optimizer gradient and the pre-step persistent first
moment. It receives no labels, sample IDs, losses, logits, clean reference, validation
metric, known or estimated noise rate, synthetic-noise audit fields, or future gradient.

For each active parameter, define its pre-step bias-corrected first-moment reference:

```text
if s_{i,t-1} = 0:
    r_{i,t-1} = 0
else:
    r_{i,t-1} = m_{i,t-1} / (1 - beta1_i ^ s_{i,t-1})
```

Concatenate the active gradients conceptually as `g_t` and their references as `r_t`.
The implementation must stream three FP32 reductions and must not allocate a flattened
gradient or reference vector. Iteration order is parameter-group list order followed by
parameter list order; each tensor reduction uses its storage order, and scalar accumulators
are updated in that same order:

```text
dot_t = sum_i sum_j g_{i,t}[j] * r_{i,t-1}[j]
gg_t  = sum_i sum_j g_{i,t}[j] ^ 2
rr_t  = sum_i sum_j r_{i,t-1}[j] ^ 2
```

The scalar consistency score is:

```text
if rr_t = 0:
    q_t = 1
else if gg_t = 0:
    q_t = 0
else:
    q_t = clip(dot_t / sqrt(gg_t * rr_t), -1, 1)
```

Thus the first gradient-bearing step is admitted and seeds standard AdamW state. A later exact
zero gradient does not manufacture agreement with a nonzero history. Reduction order,
accumulator dtype, and clipping are part of the frozen numerical contract.

### 5.2 Two-timescale history

The only history update is:

```text
F_t = alpha_fast * F_{t-1} + (1 - alpha_fast) * q_t
S_t = alpha_slow * S_{t-1} + (1 - alpha_slow) * q_t
d_t = F_t - S_t
```

with the structural constraint:

```text
0 <= alpha_fast < alpha_slow < 1
0 < tau < 2
```

`alpha_fast`, `alpha_slow`, and `tau` are the complete set of nonstandard configuration
fields. The 24-run Pilot freezes the same single triple for both candidates:

```text
alpha_fast = 0.90
alpha_slow = 0.99
tau        = 0.10
```

The Pilot performs no detector tuning or grid search. No condition-, seed-, class-, layer-,
parameter-group-, or epoch-specific override is permitted.

The frozen default has a mechanically generated FP32 reachability witness. Starting from the
specified initial state, ten consecutive `q=-1` observations switch admission off at step 2;
following them with 36 consecutive `q=+1` observations switches it back on at step 46. The
test must also assert the two inclusive threshold crossings that complete each transition.
These fixtures are used only for conformance tests. Empirical training behavior cannot change
the default.

## 6. Shared Admission State Machine

The history update in Section 5 occurs before the transition check. Comparisons are
inclusive. The current `q_t` may complete a transition only when the preceding successful
step already established the same-direction streak:

```text
if a_{t-1} is true:
    good_streak_t = 0
    if d_t <= -tau:
        if bad_streak_{t-1} = 1:
            a_t = false
            bad_streak_t = 0
        else:
            a_t = true
            bad_streak_t = 1
    else:
        a_t = true
        bad_streak_t = 0
else:
    bad_streak_t = 0
    if d_t >= tau:
        if good_streak_{t-1} = 1:
            a_t = true
            good_streak_t = 0
        else:
            a_t = false
            good_streak_t = 1
    else:
        a_t = false
        good_streak_t = 0
```

Only consecutive successful optimizer steps can form a streak. A skipped overflow, an
optimizer call with no gradients, evaluation, checkpoint save/load, or scheduler-only event
does not update `q`, `F`, `S`, either streak, or `a`.

The gate is deterministic. It draws no random number and has no stochastic admission mode.

## 7. Shared Ephemeral AdamW Parameter Update

For every `i in A_t`, compute without first mutating persistent moments:

```text
s'_i = s_{i,t-1} + 1

m~_{i,t} = beta1_i * m_{i,t-1}
             + (1 - beta1_i) * g_{i,t}

v~_{i,t} = beta2_i * v_{i,t-1}
             + (1 - beta2_i) * (g_{i,t} * g_{i,t})

mhat~_{i,t} = m~_{i,t} / (1 - beta1_i ^ s'_i)
vhat~_{i,t} = v~_{i,t} / (1 - beta2_i ^ s'_i)

theta'_{i,t} = (1 - eta_{i,t} * lambda_i) * theta_{i,t-1}
theta_{i,t}  = theta'_{i,t}
               - eta_{i,t} * mhat~_{i,t}
                 / (sqrt(vhat~_{i,t}) + epsilon_i)
```

The epsilon placement and decoupled weight-decay order are exact. Weight decay is applied
once on every successful active-parameter step, before the adaptive add, regardless of
`a_t` and `protection_target`. It is never admitted, rejected, accumulated, or rescaled by
the detector.

The admission value is not an input to any equation in this section. For fixed pre-step
state, group values, and gradient:

```text
theta_{i,t}(a_t = false) = theta_{i,t}(a_t = true)
```

This is the operational meaning of preserving the current batch's immediate parameter
effect. It does not claim that Protect-M and Protect-MV have identical parameter trajectories
after their persistent `v` histories diverge.

The implementation may use temporary tensors, expression scheduling within the supported
for-loop surface, or reusable scratch storage, but it must produce these equations.
Ephemeral candidates are not serialized as a second persistent moment pair. Any temporary
allocation counts toward measured peak VRAM.

For bit-exact PyTorch parity, the implementation evaluates the algebraically equivalent
adaptive term in the locked for-loop AdamW operation order:

```text
bias_correction1 = 1 - beta1_i ^ s'_i
bias_correction2 = 1 - beta2_i ^ s'_i
step_size = eta_{i,t} / bias_correction1
denominator = sqrt(v~_{i,t}) / sqrt(bias_correction2) + epsilon_i
theta_{i,t} = theta'_{i,t} - step_size * m~_{i,t} / denominator
```

The first equations define the mathematical recurrence; these latter equations freeze its
floating-point evaluation order. Algebraic reassociation, foreach, and fused AdamW kernels
are outside this version.

## 8. The Only Candidate Difference: Persistent Commit

The detector history, admission transition, ephemeral moments, denominators, and parameter
updates are staged logically before persistent state commit. All option, shape, dtype,
gradient-finiteness, detector-finiteness, and state validation completes before the first
mutation. After the common parameter update succeeds, commit the per-parameter and detector
state for that successful step.

For both candidates:

```text
step_{i,t} = s'_i

if a_t:
    m_{i,t} = m~_{i,t}
else:
    m_{i,t} = m_{i,t-1}       # exact hold
```

For Protect-M:

```text
v_{i,t} = v~_{i,t}            # unconditional commit
```

For Protect-MV:

```text
if a_t:
    v_{i,t} = v~_{i,t}
else:
    v_{i,t} = v_{i,t-1}       # exact hold
```

An exact hold means the moment tensor does not decay, grow, shrink, or change dtype. The
step counter still advances as specified in Section 4. Protect-M and Protect-MV may not
differ in detector arithmetic, history, thresholds, state transitions, initialization,
ephemeral moment construction, bias correction, parameter update, weight decay, supported
optimizer options, precision, logging, or failure behavior.

## 9. Bias Correction and AdamW Equivalence

Bias correction always uses the standard per-parameter successful-step clock, including
steps on which a moment was held. It never uses the number of admissions. Separate admission
counters, effective beta products, or per-moment clocks are forbidden by this version.

The implementation shall expose a test-only `protection_enabled=false` path. It forces every
moment candidate to commit and does not appear in any resolved M1 method configuration. With
protection disabled, all supported parameters, per-parameter state, weight
decay, and updates must be bit-exact to PyTorch for-loop AdamW in the same locked environment.
The detector's extra optimizer-wide metadata is outside the AdamW state equivalence comparison.

Changing the bias-correction clock when a write is rejected would define a different method
and requires a new candidate design.

## 10. Mixed Precision, Accumulation, and Overflow

The runner order for a training optimizer step is:

1. execute the single allowed forward pass under the common M1 autocast policy;
2. scale the loss when the common policy enables loss scaling;
3. execute the single allowed backward pass;
4. finish all preregistered gradient accumulation for that optimizer step;
5. unscale gradients exactly once;
6. determine finite/overflow status before calling candidate `step`;
7. on a finite step, call the candidate once, update the scaler, then advance the scheduler;
8. on overflow, do not call the candidate, update only the scaler, and do not advance the
   scheduler.

The consistency score sees accumulated, unscaled gradients before any method-specific
gradient transform. Protect-M and Protect-MV add no clipping or other transform. The
global-norm clipping baseline remains a separate method.

Autocast does not change optimizer arithmetic: trainable/master parameters, the streamed
detector reductions, ephemeral moment arithmetic, persistent moments, bias correction, and
parameter updates are FP32. Model outputs and saved evaluation values follow the common M1
precision protocol, not a candidate-specific override.

Overflow handling is atomic. On a skipped step, all of the following remain exactly
unchanged:

- trainable/master parameters;
- `step`, `exp_avg`, and `exp_avg_sq` for every parameter;
- `F`, `S`, `a`, `bad_streak`, and `good_streak`;
- scheduler step and learning rate;
- successful-optimizer-step counters.

The already-consumed batch still increments the external data-access and optimizer-attempt
counters. Those audit counters and a logged overflow-attempt counter never enter the
algorithm.

If candidate `step` directly receives a NaN or infinity despite the runner guard, it fails
closed before any mutation. An anticipated validation failure cannot partially commit
parameters, moments, the step clock, or detector state. An unexpected device/backend failure
during a validated step invalidates the run; execution must terminate and restart from the
last verified checkpoint rather than reuse the in-memory optimizer.

The exact common autocast dtype and complete scaler policy—enablement, initial scale, growth
factor, backoff factor, and growth interval—are M1 run-protocol values, not candidate
hyperparameters. They remain a user-review blocker in Section 15 and must be identical
across methods.

## 11. State Dictionary and Resume Contract

`state_dict()` contains standard `state` and `param_groups` plus a versioned
`protect_state` object:

```text
protect_state:
  schema_version: 1
  mechanism_id: protect_persistent_adamw_v1
  protection_target: m | mv
  protection_enabled: boolean
  alpha_fast: finite FP32 scalar
  alpha_slow: finite FP32 scalar
  tau: finite FP32 scalar
  fast_ema: finite FP32 scalar
  slow_ema: finite FP32 scalar
  admit: boolean
  bad_streak: integer in {0, 1}
  good_streak: integer in {0, 1}
```

The state dictionary contains no ephemeral candidate tensors, gradients, labels, losses,
noise metadata, or validation data. `protection_target` is mandatory and distinguishes the
two methods. `mechanism_id`, `schema_version`, target, enabled state, and all three
nonstandard hyperparameters are included in the canonical configuration hash.

Loading is fail-closed:

- the serialized version and mechanism ID must be supported exactly; there is no implicit
  migration;
- the runtime protection target, enabled state, and nonstandard hyperparameters must equal
  the checkpoint configuration; a mismatch is an error rather than a silent override;
- parameter count/order, group structure, state keys, tensor shapes, state dtypes, and step
  scalar validity must pass before mutation;
- detector scalars must be finite, `fast_ema` and `slow_ema` must lie in `[-1,1]`,
  hyperparameter and reachability constraints must hold, and streak values must be valid;
- a load failure leaves the live optimizer unchanged.

A successful save/load followed by the same batches, gradients, AMP decisions, and scheduler
events must reproduce uninterrupted parameters, optimizer state, detector state, admission
decisions, metrics, and canonical state hashes at every subsequent step.

## 12. Logging and Information Isolation

Per successful optimizer step, the candidate may log only detached diagnostics derived from
information already available to the optimizer:

```text
q, F, S, d, admit, transition, active_parameter_count,
gradient_norm, overflow_skipped=false, protection_target
```

An overflow attempt logs `overflow_skipped=true` outside the optimizer recurrence. Logging
must not alter reductions, synchronize a second copy of gradients, or feed a value back into
training. Aggregate clean/noisy admission summaries may be computed offline only after
training by joining logs to private synthetic-noise audit fields. The optimizer API and its
training bundle do not receive that join key or those private fields.

The candidate adds no model, prediction head, trainable parameter, clean reference set,
noise-rate estimate, sample deletion, forward pass, backward pass, training stage, or
validation query.

## 13. CPU Acceptance Tests

All CPU tests use explicit small tensors and do not read CIFAR, model weights, M1 gates,
Windows state, or experiment results.

### 13.1 AdamW and numerical-order tests

1. With protection disabled, compare at least five steps against locked PyTorch for-loop
   AdamW for multiple parameter groups, nonzero weight decay, different betas/epsilon,
   zero-valued gradients, and `grad=None`. Parameters, `step`, `exp_avg`, and `exp_avg_sq`
   must be bit-exact after every step.
2. Verify the hand-computed one-step and multi-step recurrence, including epsilon placement,
   bias correction, and decay-before-adaptive-update order.
3. Verify a held moment still uses the incremented standard step on the next ephemeral
   update; no admission counter exists.

### 13.2 Detector and transition tests

1. Verify exact score cases for first state, perfect agreement, orthogonality, opposition,
   `m=0`, `g=0`, and cosine clipping.
2. Verify streamed multi-group reduction equals explicit conceptual concatenation.
3. Verify one threshold crossing cannot change `a`, two consecutive crossings can, an
   interrupted streak resets, equality at `+/-tau` counts, and recovery is symmetric.
4. Verify a no-gradient call and an overflow attempt cannot advance either EMA or streak.
5. Verify the detector result is invariant to protection target when given identical
   pre-step state and gradients.

### 13.3 State-admission tests

1. On the second bad crossing, compare the current parameter against the explicit ephemeral
   AdamW equation for both `a=true` and `a=false`; it must be identical.
2. For Protect-M rejection, `m` must be bit-exact to its pre-step value, `v` must equal the
   ephemeral candidate, and `step` must increment once.
3. For Protect-MV rejection, both `m` and `v` must be bit-exact to their pre-step values and
   `step` must increment once.
4. For admission, both candidates must commit both moment candidates exactly.
5. Given identical inputs, the complete state-dictionary diff between candidates after one
   rejected step may contain only `protection_target` and the protected `v` tensor. All
   detector fields, parameters, first moments, step counters, and group values must match.
6. The next accepted clean step must demonstrate that any candidate difference arises from
   persistent pre-step state, not from a hidden mask on the previous parameter update.

### 13.4 Failure, precision, and serialization tests

1. A simulated AMP overflow leaves the complete optimizer/model/scheduler state tree
   unchanged; only external scaler and overflow-attempt logging may change.
2. Direct nonfinite gradients, sparse gradients, unsupported AdamW options, invalid detector
   values, and invalid hyperparameter constraints fail before mutation.
3. Save/load at admitted, rejected, and one-crossing streak states; uninterrupted and resumed
   runs must be bit-exact at every later step.
4. Reject wrong mechanism version, wrong protection target, changed detector triple, malformed
   moment shape/dtype, and invalid streak without partially modifying the live optimizer.
5. Inspect optimizer inputs to prove that labels, losses, noise flags, sample IDs, and
   validation values cannot enter the candidate API.

## 14. Windows GPU Sentinel Gate

GPU sentinels run serially on the experiment GPU before the Pilot. They use no development,
confirmation, or official-test partition and cannot be used to change the candidate default.

### 14.1 Functional sentinel

On the real ViT-LoRA trainable parameter set and the locked common AMP policy:

1. run five deterministic finite steps with protection disabled and the same prerecorded
   batches through locked PyTorch AdamW and the candidate;
2. require bit-exact trainable parameters, `step`, `exp_avg`, and `exp_avg_sq` after every
   step;
3. preload the frozen default's reviewed reachability fixtures, realize its frozen scores
   with synthetic gradients after backward without reading any label-quality field, and
   verify the off/on transitions plus every Protect-M/Protect-MV state commit against
   CPU-generated expected state hashes;
4. when loss scaling is enabled, force one GradScaler overflow between finite steps; when it
   is disabled, inject one nonfinite-gradient guard event instead. In either case, require
   complete candidate/model/scheduler state hashes to remain unchanged across the skipped
   step;
5. checkpoint after a one-crossing streak, restart the process, and require subsequent
   parameters, state, decisions, and metrics to be bit-exact to uninterrupted execution.

Any nondeterminism or backend difference that prevents bit-exact equality fails closed. A
numerical tolerance requires a new reviewed revision written before the sentinel rerun.

### 14.2 Compute and isolation sentinel

Profiler and runner counters must show, per batch:

- exactly one model forward and one backward;
- exactly one candidate optimizer call on a finite optimizer step;
- no candidate optimizer call on an overflow skip;
- identical batch IDs, augmentations, data accesses, accumulation count, and evaluation
  cadence across AdamW, Protect-M, and Protect-MV;
- no candidate read of tuning/development/confirmation/test artifacts or private noise-audit
  fields;
- no persistent second copy of all moment tensors.

For a paired performance sentinel, use ten untimed warm-up training steps followed by fifty
timed complete forward/backward/optimizer training steps, repeat the same prerecorded
sequence five times per method, synchronize CUDA at the timing boundaries, and compare the
median candidate/AdamW training-step ratio. Each candidate must be at most `1.05`, matching
the M1 tuning wall-clock cap. Record optimizer-only profiler time plus peak allocated and
reserved VRAM; any OOM fails. Sentinel timing is an implementation rejection check, not a
source of detector values or method selection evidence.

## 15. User-Review Register and Remaining Freeze Gates

There are no hidden default values. User approval on 2026-08-08 resolved the scientific and
precision choices needed for implementation:

| Review ID | Resolution | Remaining gate |
|---|---|---|
| `CAND-DEFAULT-001` | One Pilot triple: `(0.90, 0.99, 0.10)` with the frozen off/on witness above; no candidate tuning grid | CPU conformance tests must reproduce the witness exactly |
| `M1-AMP-001` | FP32 model/optimizer arithmetic with autocast and GradScaler disabled for every Pilot method, inherited from calibration | GPU sentinel must verify the common path |
| `CAND-CODE-001` | Implementation authorized from this approved specification | Candidate source commit and canonical mechanism/config hashes must be recorded before the GPU sentinel |

The three detector quantities are the only nonstandard configuration fields and are fixed for
the Pilot. The fixed
two-crossing rule, initialization, global gate granularity, cosine definition, hold semantics,
step clock, supported AdamW surface, serialization version, and sentinel sizes are frozen
mechanism or verification constants and may not be promoted into tuning axes. A future tuning
study would require a separately approved design and unused data gate.

Standard AdamW learning rate, weight decay, betas, epsilon, parameter groups, scheduler, and
training budget are inherited from the frozen M1 selection/protocol lifecycle. They are not
candidate hyperparameters and cannot vary between Protect-M and Protect-MV or between clean
and noisy conditions for a selected configuration.

## 16. Fail-Closed Conditions and Scope Boundary

Candidate implementation or execution stops on any of the following:

- the two methods differ anywhere except the rejected-step persistent `v` commit and their
  serialized `protection_target`;
- admission enters the current parameter equation or weight decay;
- a rejected moment decays or otherwise changes instead of holding exactly;
- bias correction uses admission count or separate per-moment clocks;
- any detector field is changed or tuned during the Pilot;
- a hidden default overrides an item in Section 15;
- an overflow, validation failure, or load failure partially mutates state;
- protection-disabled parity, state-level CPU tests, restart tests, or either GPU sentinel
  fails;
- the candidate receives forbidden training information or adds compute outside the stated
  one-forward/one-backward optimizer contract;
- the implementation cannot stay within the frozen wall-clock cap.

Passing this specification and its sentinels establishes implementation conformance only.
It does not establish noisy-label detection accuracy, robustness to systematic bias,
generalization benefit, novelty, Protect-M versus Protect-MV superiority, or M1 GO. Those
claims remain governed by the independent M1 development and confirmation gates.
