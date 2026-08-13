# Full-Model-State Repair Validation Design

## Objective

Repair the ResNet causal intervention so that a model-state intervention includes parameters and persistent buffers, then test the smallest cross-architecture claim supported by the valid ViT discovery evidence: the preferred repair can depend on the future continuation distribution and the evaluation horizon.

Paper validity and contribution quality take priority over compute reuse. Reuse is allowed only when it preserves the causal estimand, independence role, and provenance. A smaller matrix is not considered a benefit if it weakens the claim or makes reviewer-critical uncertainty unidentifiable.

## Evidence roles

- The 30 ViT bundles remain discovery evidence because LayerNorm has no mutable running-statistic buffers affected by the identified defect.
- All 24 prior ResNet bundles are excluded from scientific estimates. They remain audit evidence showing why full-state restoration is required.
- Corrected ResNet runs are still discovery/repair evidence, not untouched final confirmation, because their design follows inspection of prior outcomes.

## Intervention definition

Replace the old trainable-parameter snapshot with an exact model-state snapshot containing every entry returned by `model.state_dict()`, including BatchNorm `running_mean`, `running_var`, and `num_batches_tracked`. The optimizer state remains a separately crossed factor.

The four branches therefore mean:

- `CC`: clean complete model state + clean optimizer state;
- `CN`: clean complete model state + noisy optimizer state;
- `NC`: noisy complete model state + clean optimizer state;
- `NN`: noisy complete model state + noisy optimizer state.

Model state and optimizer state must be restored before every branch. No buffer may carry across branches.

## Stage 1: two-bundle correctness sentinel

Run only ResNet seed 601, dose 1250, clean and noisy continuation. Before accepting either scientific result, require:

1. Exact snapshot schema, tensor keys, shapes, dtypes, finiteness, and model revision binding.
2. At horizon 0, all four branch measurements must be invariant to branch execution order and the clean/noisy continuation bundles must match exactly for every deterministic metric.
3. The standard branch order and a reversed-order diagnostic must produce numerically identical endpoints under the project's deterministic CUDA configuration.
4. Interrupt/resume must reproduce the uninterrupted summary and hashes.
5. `test_loaded=false`, one source commit, one immutable contract, at most 12 CPU threads, and serial GPU execution.

Any correctness failure stops without using the scientific values.

## Stage 2: bounded high-exposure completion

Only if Stage 1 passes, run the remaining seed/dose/continuation combinations for seeds 601-603 and doses 64/1250, for a maximum matrix of 12 corrected bundles total. The two Stage-1 bundles are reused, leaving at most 10 additional bundles.

The analysis treats seed as the independent unit and reports the pointwise repair margin across the frozen horizons `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512`. Its primary context contrast is, for each seed and dose, the mean of `margin_noisy - margin_clean` over horizons `32, 64, 128, 256`. Its primary action-switch diagnostic is a positive clean-continuation margin and negative noisy-continuation margin at the same horizon. Endpoint AUC and horizon-512 tuning loss are secondary summaries. The analysis does not revive a universal dose threshold or claim that every endpoint must select the same action.

## Decision

- `GO`: all correctness checks pass; at dose 1250 the primary context contrast is negative for all three seeds; and at least two of three seeds show the same clean-positive/noisy-negative repair-action switch at two or more of the four primary horizons. Dose 64 is reported as a frozen secondary replication and cannot rescue a miss at dose 1250. A `GO` authorizes a separately frozen, genuinely independent confirmation study; it is not itself confirmation.
- `NO-GO`: correctness passes but the context effect is absent or inconsistent. Stop the adaptive-repair methods mainline and retain a measurement/limitations paper option.
- Infrastructure failure: repair and exact resume only; never change seeds, doses, continuations, endpoints, or thresholds after results are visible.

No experiment beyond the 12 corrected bundles is authorized by this design.
