# Experiment Plan: External Sentinel Only

**Contract status:** frozen proposal; GPU not dispatched by this artifact

**Budget:** 2–3 RTX 5080 GPU-hours, serial execution

**Purpose:** decide whether the JMLR route deserves full confirmation compute

## Condition and matrix

- Dataset/split: existing sealed CIFAR-100 train/tuning split; test data remains inaccessible.
- Model/training: `google/vit-base-patch16-224-in21k`, LoRA rank 8, AdamW, learning rate `3e-4`, weight decay `0.1`, betas `(0.9, 0.999)`, epsilon `1e-8`, batch size 32, warm-up 500 steps.
- New noise source: exact 40% class-balanced symmetric-uniform label noise. Exactly 160 of 400 training examples in each class are corrupted; every destination is sampled uniformly from the other 99 labels.
- New paired training seeds: `501, 502, 503`; paired noise seeds: `1501, 1502, 1503`; augmentation seeds: `10501, 10502, 10503`.
- Exposure dose: `1250` only.
- Continuations: `clean, noisy`.
- Matrix size: 3 seeds × 1 dose × 2 continuations = **6 factorial bundles**.
- Frozen outcomes: `clean_loss_excess_auc_128` and `tuning_loss_h512`.
- Frozen carrier thresholds: state ratio `>= 1`, parameter ratio `<= -1`; no threshold retuning.

Each bundle uses the existing four parameter-by-state branches `CC, CN, NC, NN`. Positive repair values retain the existing definitions:

- state repair = `NN - NC`;
- parameter repair = `NN - CN`.

## Exact gate

The sentinel is `GO` only if all conditions hold:

1. All six bundles succeed from one clean source commit and one contract; all sealed input hashes match; all numeric endpoints are finite; no test artifact is constructed or loaded.
2. All 12 seed-by-continuation-by-outcome carrier classifications are `parameter`.
3. For every one of those 12 observations, `parameter_repair - state_repair > 0`.

Any finite scientific miss is `NO-GO` for the JMLR expansion. Only a provenance, resource, or infrastructure failure may be repaired and resumed without changing the matrix or gate. No failed seed, continuation, or outcome may be replaced.

## Why these runs change reviewer belief

- The symmetric-uniform channel is structurally different from the instance-dependent discovery channel, so replication weakens the explanation that long parameter dominance was caused by that specific corruption generator.
- Three paired seeds and both continuations test stability rather than a favorable single trajectory.
- The repair inequality directly connects carrier attribution to action selection; it is more informative than repeating a classification alone.
- Dose 1 and intermediate doses are excluded because the current decision is only whether the stable long-exposure/repair signal transports to a new noise source.

## Expected artifacts

- one immutable sentinel contract and its SHA-256;
- three sealed public noise bundles and separate private audit manifests;
- six bundle summaries plus checkpoint/provenance hashes;
- one atomic `SENTINEL_DECISION.json` containing the 12 classifications and 12 repair margins;
- one deterministic evidence precheck and one post-result audit/claim verdict.

## Time estimate

The completed dose-1250 bundles averaged 1,275 seconds each. Six serial bundles therefore project to 2.13 GPU-hours; noise preparation, validation, and gate evaluation bring expected wall time to about **2.3–2.7 hours**. No full 50–90 hour matrix is authorized here.
