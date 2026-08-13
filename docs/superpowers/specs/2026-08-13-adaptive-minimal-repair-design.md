# Adaptive Minimal Repair Research Design

## Objective

Build a long-form journal paper around a causal, actionable question: after a bounded harmful training episode, when is resetting optimizer state sufficient, and when must model parameters be rolled back?

Paper quality takes priority over reusing prior compute. Existing results are reused only when their protocol and provenance match the claim being tested. Exploratory evidence must never be relabeled as independent confirmation.

## Revised Claim

The rejected universal-threshold claim is retired. The new hypothesis is that damage progressively migrates from transient optimizer state into persistent model parameters as harmful exposure grows, while the location and sharpness of this repair handoff depend on architecture, optimizer, task, and endpoint.

The intended contribution has three parts:

1. A matched counterfactual decomposition that separates optimizer-state repair from parameter repair.
2. An architecture-conditioned repair-handoff curve rather than a brittle universal dose threshold.
3. An adaptive minimal-repair policy that selects the least disruptive effective intervention from short, test-free diagnostic branches.

## Evidence Policy

Existing ViT-LoRA, CIFAR-100 instance-dependent-noise results at doses 1/8/64/1250 are discovery evidence. Existing ViT-LoRA symmetric-noise and ResNet-18 symmetric-noise sentinels are external-discovery evidence because their designs were influenced by earlier findings. Their `NO-GO` decisions remain immutable and are reported as evidence against a universal threshold.

The existing data may be used for effect estimation, endpoint design, power estimation, ablations, and hypothesis generation. It may not be used as the final independent confirmation set. Test labels remain unavailable to all selection and stopping logic.

## Experimental Sequence

### Stage 1: Complete the discovery curve

Add only the missing ResNet-18 anchors at doses 1 and 1250 for seeds 601-603, clean/noisy continuation, and the two frozen endpoints. This is 12 bundles. Analyze signed repair margin continuously, including effect size and uncertainty, rather than requiring all observations to have the same carrier label.

Stage 1 advances only if the exposure effect is directionally coherent under a prespecified trend statistic and is not driven by a single seed, endpoint, or continuation. Otherwise the adaptive-policy route stops and the evidence is retained only for a measurement/limitations paper.

### Stage 2: Build the minimal-repair policy

Construct a low-cost, test-free diagnostic from short matched branches. It must choose among no intervention, optimizer-state reset, and parameter rollback. The policy is calibrated only on discovery data and compared against fixed reset, fixed rollback, oracle branch selection, and no intervention.

The method advances only if it improves recovery while reducing unnecessary parameter rollbacks, with gains replicated across both model families. A complex learned controller is out of scope unless a simple threshold or calibrated score is demonstrably inadequate.

### Stage 3: Independent confirmation

Freeze code, endpoints, seeds, decision rules, and exclusions before choosing one disjoint confirmation setting. The confirmation must introduce a meaningful shift such as a new dataset or optimizer and use untouched seeds. Confirmation is run once; failed confirmation is reported, not tuned away.

Additional breadth experiments are authorized only when they resolve a specific reviewer-critical uncertainty. There is no blanket matrix expansion.

## Analysis

The primary estimand is repair margin: parameter-repair benefit minus optimizer-state-repair benefit. For the frozen lower-is-better endpoints this is `NC - CN`, so positive values favor parameter rollback. Analyses report continuous margins, paired uncertainty, trend across log exposure, and interactions with model family and endpoint. Binary carrier labels are secondary summaries.

A compact dynamical account will model optimizer-state corruption as decaying memory and parameter displacement as accumulated damage. Its role is to explain the possibility and conditioning of a crossover, not to claim a universal closed-form threshold.

## Integrity and Failure Handling

Every bundle must validate source commit, model revision, seed, dose, continuation, checkpoint schema, and `test_loaded=false`. Decisions are written atomically and fail closed on missing, stale, duplicated, or nonfinite evidence. Interrupted runs resume from validated checkpoints without changing frozen inputs.

Any reuse decision is recorded by role: discovery, external discovery, ablation, power estimate, or excluded. Results are never discarded for contradicting the preferred narrative.

## Success Criterion

The mainline is viable when it supports both a reproducible architecture-conditioned handoff phenomenon and an adaptive policy with practical recovery benefit under independent confirmation. A curve without a useful policy may still support a measurement paper, but is not treated as sufficient evidence for the targeted high-quality long journal paper.
