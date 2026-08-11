# Causal Transport Audit: Quality-First Confirmation Design

**Date:** 2026-08-11

**Priority:** paper quality and acceptance probability before compute minimization

**Compute:** one Windows RTX 5080; CPU affinity capped at 12 logical cores; no GPU power cap

## 1. Problem anchor

The project no longer tries to rescue Protect-M or propose another optimizer. It asks:

> Does the causal carrier identified by a local noisy-batch intervention remain the
> dominant carrier after noisy training accumulates over longer exposure scales?

Existing results are exploratory evidence. M0.6 found a larger short-horizon effect from
optimizer-state transplantation after a local corrupted pulse. The later crossed AdamW
experiment found parameter displacement dominant in both seeds at all three checkpoints.
Both are causal interventions; the discrepancy is therefore a local-to-global causal
transport problem, not a correlation-versus-causation problem.

## 2. Claim boundary

The paper has at most two main contributions:

1. **Scale-dependent carrier reversal:** optimizer moments may carry the immediate memory
   of a corrupted update, while persistent downstream damage becomes encoded primarily in
   trainable PEFT parameters as exposure accumulates.
2. **Causal Transport Audit:** a paired, multi-dose intervention protocol that tests whether
   a local causal attribution transports to accumulated training damage.

The work will not claim that optimizer memory, state reset, parameter transplantation, or
noisy-label PEFT is itself new. It will not claim universality beyond settings that pass the
confirmation protocol.

## 3. Discovery and confirmation separation

All M0.6, M1, M1.1, and commit `2eee6e1301e24bac3f984f5e243bd4c5189a8f11`
causal-attribution artifacts are immutable discovery data. They may define the hypothesis,
metrics, doses, and anticipated rejection arguments, but they cannot count as independent
confirmation.

Confirmation uses new seeds or previously unexecuted intervention outcomes. Its contract is
hashed before the first confirmation replay. Test data remain sealed throughout method and
claim selection.

## 4. Measurement

For matched clean/noisy parameters and optimizer state, each dose uses the complete factorial
`CC`, `CN`, `NC`, and `NN` under a bitwise-matched continuation. For outcome `Y`:

- parameter effect: the balanced main effect of changing clean to noisy parameters;
- state effect: the balanced main effect of changing clean to noisy optimizer state;
- interaction: the factorial residual.

The carrier ratio at dose `d` and horizon `h` is:

`R(d,h) = log2((abs(state_effect) + 1e-12) / (abs(parameter_effect) + 1e-12))`

`R >= 1` means at least twofold state dominance; `R <= -1` means at least twofold parameter
dominance. Interaction is reported separately and invalidates a main-effect classification when
its magnitude is at least the larger main effect.

Primary outcomes are clean-loss-excess AUC through horizon 128 and fixed-horizon tuning loss.
Accuracy, prediction JS divergence, parameter distance, moment distances, and gradient cosine
are secondary explanatory outcomes. Checkpoints and horizons are repeated measurements; the
paired training seed is the experimental unit.

## 5. Staged experiment program

### Gate A: inexpensive held-out mechanism check

Use the existing, not-yet-transplanted late CAdam checkpoints for two held-out seeds. Add the
matching short-pulse CAdam intervention needed to compare local and accumulated effects; a late
checkpoint replay alone cannot establish reversal. No CAdam base retraining is allowed.

- estimated GPU time: 1--2 hours;
- pass: for both seeds, the local anchor has `R >= 1`, the accumulated anchor has `R <= -1`,
  both primary outcome families agree, and interaction is not dominant at either anchor;
- fail: stop before new multi-dose training.

### Gate B: primary confirmatory carrier curve

Use CIFAR-100, ViT-B/16, rank-8 LoRA, AdamW, and the frozen 40% instance-dependent-noise
construction. Run three new paired training seeds, disjoint from all discovery seeds. From a
common starting state, capture clean/noisy exposure at exactly `1`, `8`, `64`, and `1250`
optimizer steps, then run the full factorial replay at each dose.

The local and accumulated interventions use the same corruption process, data binding, model,
optimizer, scheduler, and continuation generator. This removes the alternative explanation that
the reversal was caused by comparing a label-flip pulse with a different long-run noise model.

- estimated GPU time: 3--5 hours;
- pass: for at least two of three seeds, the dose-1 anchor has `R >= 1` and the dose-1250 anchor
  has `R <= -1`; the third seed must not show the opposite twofold reversal; interaction must not
  dominate either anchor; and both primary outcome families must agree. Doses 8 and 64 locate and
  visualize the transition but are not used to move the pass boundary after execution;
- fail: stop the route; do not add seeds, change doses, move endpoints, or sweep thresholds.

### Gate C: external confirmation

Only after Gate B passes, run three new paired seeds at the two anchor doses (`1` and `1250`) on
CIFAR-100N human-noise labels using the same backbone, PEFT scope, and audit rules. Dataset
version, label set, split, and archive hashes are frozen before loading outcomes.

- estimated GPU time: 5--8 hours;
- purpose: rule out a synthetic-noise-only explanation;
- pass: at least two of three seeds satisfy the same anchor-crossing and interaction rules as
  Gate B, and the third seed does not show the opposite twofold reversal;
- fail: retain the controlled CIFAR-100 result but do not make a cross-noise generalization claim.

The projected total is 9--14 new GPU-hours and four to six days for experiments, integrity
checks, analysis, and adversarial review. A submission-oriented evidence package is targeted
within seven to ten days if all gates pass.

## 6. ARIS workflow

ARIS supplies the research and assurance chain, not remote execution:

1. refresh `research-lit` and `novelty-check` around causal transport, state transplantation,
   training memory, noisy-label PEFT, and state-aware unlearning;
2. freeze the problem anchor, claims, nearest prior work, and strongest rejection memo;
3. run local same-family adversarial review as provisional evidence where independent reviewer
   routing is unavailable;
4. convert the approved claims into a claim-driven experiment plan;
5. after results, run experiment integrity, result-to-claim, paper-claim, citation, and
   kill-argument audits before treating the paper as submission-ready.

The activated ARIS profile blocks its own SSH and experiment-dispatch skills. Windows execution
therefore continues through the already validated project runner and scheduled-task mechanism.

## 7. Integrity, recovery, and hardware

- Exact source commit, lockfile hash, data binding, noise manifest, split, seed, checkpoint schema,
  optimizer clock, and RNG state are validated before each transplant.
- Confirmation contracts and terminal decisions are atomic and fail closed.
- Every long stage saves dose-indexed checkpoints and SHA-256 manifests so interruption resumes
  from the last valid dose rather than restarting completed work.
- Test loaders are never constructed. Validation data are used only at frozen endpoints.
- Windows uses one task at a time, a 12-logical-core affinity mask, no GPU power cap, and records
  wall-clock, GPU temperature/utilization, peak VRAM, and exit status.
- Non-finite values, provenance mismatch, missing branches, step-clock mismatch, or accidental
  test access invalidate the affected gate rather than being silently excluded.

## 8. Explicitly rejected scope

- no Protect-M rescue or new optimizer proposal;
- no EuroSAT pilot reuse as confirmation because it loaded test data and lacks compatible
  optimizer-state checkpoints;
- no favorable dataset, checkpoint, dose, seed, or endpoint selection after results;
- no large baseline catalogue unrelated to the two paper claims;
- no paper drafting before Gate B establishes a confirmatory signal.
