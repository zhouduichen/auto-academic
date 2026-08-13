# JMLR Training-Attribution Transport Long-Paper Design

**Date:** 2026-08-13

**Status:** direction approved; awaiting written-spec review

**Target:** CCF-A journal regular full-length article, JMLR first and Artificial Intelligence second

## 1. Problem anchor

The completed causal-transport Gate B is a provenance-valid scientific `NO-GO` for the
preregistered claim that a stable optimizer-state carrier at dose 1 reverses to a parameter
carrier at dose 1250. All 24 bundles succeeded, but no seed met the required crossing because
the local carrier was continuation- and outcome-sensitive. At dose 64 and especially dose
1250, parameter dominance was stable across seeds and continuations.

The new paper must not relabel that failure as a success. It asks a broader question:

> When does a causal attribution of training damage transport across exposure scales, and can
> attribution transportability predict whether repair should target optimizer state or learned
> parameters?

Existing M0.6, M1, M1.1, causal-attribution, and Gate A/B artifacts remain immutable discovery
evidence. Any new general claim requires a frozen contract and disjoint confirmation evidence.

## 2. Paper thesis and contribution boundary

The dominant contribution is **training-attribution transportability**: a formal problem,
measurement protocol, and predictive test for whether a carrier attribution remains valid as
training continues.

The paper may make at most two primary claims:

1. Local parameter/state attribution can be unstable across continuation, outcome, and exposure
   scale, while long-exposure damage can converge to a stable parameter carrier.
2. A transport-aware attribution can predict the correct class of causal repair better than a
   fixed optimizer-reset policy or simple parameter/moment/gradient proxies.

A compact Adam/PEFT dynamics analysis may support these claims by deriving testable decay and
accumulation predictions. It is not a claim of complete deep-network theory. A new optimizer is
not required and Protect-M is not revived.

## 3. Long-paper evidence shape

The full article needs four linked evidence blocks:

- **Formalization:** transport gap, continuation sensitivity, carrier handoff time, and exact
  parameter-by-optimizer-state factorial interventions.
- **Discovery:** reanalysis of the existing 24 Gate B bundles plus compatible immutable earlier
  evidence, with the original `NO-GO` preserved.
- **Independent confirmation:** at least one new noise source and one genuinely different model
  or modality, using new seeds and a frozen decision contract.
- **Predictive utility:** causal repair experiments testing whether the audit selects state repair
  early and parameter repair after handoff; proxy and fixed-policy baselines must use matched
  budgets.

The final empirical scope is decided only after a low-cost external sentinel. A JMLR submission
must ultimately interest a broad machine-learning audience rather than remain a single
CIFAR-100/ViT case study.

## 4. ARIS workflow

Reuse the existing ARIS research and assurance chain instead of creating a parallel process.
Because the project already has a visible problem and substantial evidence, enter in the middle
of the pipeline rather than restarting broad idea discovery.

1. **Evidence lock:** run the logic of `experiment-audit` and `result-to-claim` on the 24 Gate B
   artifacts. Mechanically verify cited values before any model judgment. Produce one canonical
   result-to-claim verdict.
2. **Thesis refinement:** use `research-lit`, `novelty-check`, and
   `research-refine-pipeline`. Freeze the problem anchor, keep one dominant contribution, and
   require a top-venue review verdict before experiment planning is accepted.
3. **Claim-driven plan:** use `experiment-plan` with no more than two primary claims, five core
   experiment blocks, and three baseline families. Every run must change a reviewer belief.
4. **Execution:** use the approved AutoAcademic/Windows runner path. ARIS native SSH and remote
   shell routes remain blocked. Valid checkpoints resume; scientific thresholds never change on
   retry.
5. **Post-result gates:** after each milestone, run experiment integrity and result-to-claim
   review once. Use `kill-argument` and the bounded `auto-review-loop`; failure after the bounded
   rounds stops or narrows the route rather than creating an endless rescue loop.
6. **Writing:** only validated claims enter `paper-plan` and `paper-writing`, followed by
   paper-claim and citation audits. Use the official JMLR style because the current ARIS venue
   formatter has no JMLR preset.

To respect the user's documentation constraint, retain only canonical final artifacts: the final
proposal, experiment plan/tracker, integrity and claim verdicts, narrative handoff, and paper.
Round logs required for auditability may exist under `refine-logs/` but are not expanded into
additional reports.

## 5. Staged compute and stop rules

### Stage 0: zero-GPU evidence gate

Reconstruct the dose curves, continuation agreement, effect classifications, uncertainty, and
repair contrasts from existing artifacts. Verify that the proposed transport metrics and repair
prediction are computable without changing endpoints. If the discovery evidence cannot support
a precise new hypothesis, stop before new training.

### Stage 1: external sentinel

Freeze one new external condition and run only the minimum long-exposure anchors needed to test
replication of stable parameter dominance. Budget: approximately 2--3 GPU-hours. The sentinel is
scientific pilot evidence and cannot be reused as independent confirmation unless it was included
in the frozen confirmatory contract.

- Replicated signal: authorize the full confirmation plan.
- Finite but non-replicated signal: stop the JMLR route and retain the scoped negative result.
- Infrastructure failure: repair and resume from validated checkpoints without changing the
  scientific contract.

### Stage 2: full confirmation and predictive utility

Only after Stage 1 passes, run the disjoint multi-setting matrix, strong baselines, decisive
ablations, and repair-selection tests. Current planning range is 50--90 additional GPU-hours,
released in milestones rather than as one batch. Any scope expansion must defend a frozen paper
claim and pass the preceding result-to-claim gate.

## 6. Integrity and statistical rules

- Paired training seed is the experimental unit; factorial branches, horizons, and checkpoints
  are repeated measurements, not independent samples.
- Discovery data define hypotheses and effect sizes but do not count as independent confirmation.
- Test data and human-noise outcomes remain sealed until their predeclared stage.
- Source commit, contract, dataset, split, noise, model, optimizer clock, continuation RNG, and
  artifact hashes are validated fail-closed.
- Negative or mixed results remain in the analysis. No favorable endpoint, seed, dose, model, or
  noise-source selection after outcomes are visible.
- Claims cite machine-verifiable source artifacts; absent evidence is rejected before reviewer
  judgment.

## 7. Success and terminal outcomes

**JMLR-ready** requires a novel and stable transportability thesis, independent multi-setting
confirmation, predictive repair utility, strong alternative explanations ruled out, complete
integrity, and adversarial review with no blocking claim gap.

**Narrow journal route** applies if long-exposure parameter dominance replicates but transport or
repair utility does not. The claims and venue must be narrowed; this is not rounded up to JMLR
readiness.

**Stop** applies if the external sentinel fails, evidence integrity fails, novelty is preempted,
or bounded review repeatedly finds the dominant claim unsupported. Existing results may still be
released as a scoped negative result, but no further rescue experiment is authorized by this
design.

## 8. Explicit non-goals

- changing or superseding the Gate B `NO-GO`;
- reviving Protect-M/MV or promising a new optimizer before evidence supports it;
- padding a long paper with unrelated datasets, baselines, or documentation;
- claiming model-, modality-, or noise-general conclusions from one setting;
- drafting persuasive prose before result-to-claim and integrity gates pass.
