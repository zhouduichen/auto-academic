# ResNet Repair-Handoff Sentinel Design

**Date:** 2026-08-13

**Status:** approved design; implementation pending

**Purpose:** Test whether the observed change from optimizer-state repair at short exposure to parameter repair at longer exposure transfers from ViT-LoRA to a genuinely different model family, without changing the dataset or noise source.

## Scientific scope

The frozen ViT discovery evidence shows parameter repair beating optimizer-state repair in 2/12 observations at dose 8 and 12/12 at dose 64. The symmetric-noise external sentinel independently shows parameter repair winning 12/12 times at dose 1250, while one discrete carrier classification remains reproducibly `mixed`.

The new primary quantity is therefore the continuous repair margin:

```text
parameter_repair = NN - CN
state_repair     = NN - NC
repair_margin    = parameter_repair - state_repair
```

For the two frozen loss outcomes, a positive margin means parameter rollback lowers loss more than optimizer-state reset. Discrete carrier classes remain diagnostics and do not determine the gate.

## Frozen condition

- Dataset and split: existing sealed CIFAR-100 train/tuning split; official test remains inaccessible.
- Noise: existing sealed exact 40% class-balanced symmetric-uniform bundles.
- Model: ImageNet-pretrained `microsoft/resnet-18` with a 100-class head and full-parameter AdamW fine-tuning.
- Input pipeline: the existing deterministic 224-pixel CIFAR transform and paired batch plans.
- Optimizer: AdamW with learning rate `3e-4`, weight decay `0.1`, betas `(0.9, 0.999)`, epsilon `1e-8`, batch size 32, warm-up 500 steps, and no scheduler.
- Training seeds: `601, 602, 603`; augmentation seeds: `10601, 10602, 10603`.
- Noise bindings: reuse sealed symmetric bundles `1501, 1502, 1503` in fixed seed order.
- Exposure doses: `8, 64`.
- Continuations: `clean, noisy`.
- Primary outcomes: `clean_loss_excess_auc_128` and `tuning_loss_h512`.
- Matrix: 3 seeds × 2 doses × 2 continuations = 12 factorial bundles.

The Hugging Face ResNet checkpoint revision `65a5785d9156231087c481e0c7dd33a5ff6f7e3e` and downloaded file hashes must be pinned in the contract before the CUDA sentinel. No fallback model or online revision substitution is allowed. The earlier design draft named revision `b84c5cd73e9544fa1b67d690748d13a4bdb29267`; implementation audit showed that revision predates `model.safetensors`, so it was replaced before any ResNet result existed.

## Exact decision rule

The sentinel is `GO` only if all of the following hold:

1. Exactly 12 bundles complete from one clean source commit and one immutable contract, with finite endpoints, matching sealed inputs, and `test_loaded=false`.
2. At dose 64, all 12 seed-by-continuation-by-outcome repair margins are strictly positive.
3. At dose 8, at most 3 of the 12 repair margins are strictly positive, and each seed has at most 1 positive margin among its four repeated observations.

Any finite scientific miss is `NO-GO`. A provenance, checkpoint, resource, or infrastructure failure may be repaired and resumed only under the identical contract. No seed, outcome, continuation, threshold, or dose may be replaced after results are visible.

## Implementation boundary

Add a model-family adapter rather than altering the existing ViT path. It must provide audited model construction, logits extraction, trainable-state enumeration, and optimizer creation while reusing the existing paired exposure, factorial branch, replay, endpoint, checkpoint, and atomic-artifact logic.

The implementation requires:

- CPU tests for pretrained-load audit, 100-class head replacement, full trainable-state coverage, AdamW state compatibility, deterministic job construction, contract validation, exact decision logic, non-finite rejection, and resume behavior;
- one short CUDA functional/resource sentinel before the 12-bundle matrix;
- at most 12 CPU threads and serial GPU execution;
- expected wall time of 3–5 RTX 5080 GPU-hours, subject to the measured short sentinel.

## Terminal outcomes

- `GO`: authorize planning a disjoint multi-setting confirmation matrix for the repair-handoff thesis.
- `NO-GO`: stop the JMLR expansion and retain the current evidence as a scoped negative/diagnostic journal result.
- Infrastructure failure: exact resume only; it never changes the scientific rule.

This sentinel does not overwrite the prior Gate B or symmetric-noise `NO-GO`, does not revive Protect-M, and does not authorize the later 50–90 hour confirmation matrix.
