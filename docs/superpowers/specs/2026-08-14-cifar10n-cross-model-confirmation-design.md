# CIFAR-10N Cross-Model Repair Confirmation Design

**Status:** Approved design, frozen before CIFAR-10N labels are downloaded or read.

## Purpose and evidence role

Test, once, whether continuation-dependent repair transfers to a disjoint image set with
real human label noise, and whether a diagnostic calibrated only on discovery evidence can
choose between parameter rollback and optimizer-state repair.

The existing CIFAR-100 ViT-LoRA and corrected ResNet-18 results remain discovery evidence.
They may calibrate the diagnostic and estimate cost, but they do not count toward confirmation.
All old ResNet results produced before complete `model.state_dict()` restoration remain excluded.

## Alternatives considered

- CIFAR-100N changes the noise labels but reuses the original CIFAR-100 images. It is controlled
  but does not answer the dataset-reuse objection.
- ANIMAL-10N, Food-101N, and Clothing1M use new images, but they do not provide a clean label for
  every noisy training image. They cannot support the exact paired clean/noisy exposure used by
  the factorial intervention.
- CIFAR-10N provides a disjoint 50,000-image training set, original clean labels, and human noisy
  labels for the same image IDs. It therefore changes images, class structure, and noise source
  without breaking paired causal identification.

The selected setting is the official CIFAR-10N `worse_label` vector. No other CIFAR-10N label
set may be inspected or run. Selecting the approximately 40% label set before download matches
the discovery corruption scale and prevents favorable label-set selection.

## Frozen data protocol

Use only the official CIFAR-10 training images and the official CIFAR-10N `worse_label` vector.
Before parsing labels, record the source URL, archive SHA-256, vector key, vector SHA-256, image-ID
order, clean-label SHA-256, and license/terms record. Reject duplicates, missing IDs, out-of-range
labels, unexpected class counts, or any mismatch between the clean and noisy vectors.

Create one deterministic class-stratified split with seed `20260801`. For each of ten classes,
allocate 4,000 images to training, 250 to tuning, 250 to development, and 500 to a sealed
confirmation partition. Training uses only the training partition; primary replay measurement
uses only tuning. Development, sealed confirmation, and the official CIFAR-10 test set are not
loaded by selection, execution, stopping, or primary analysis. Every artifact records
`test_loaded=false`, `development_loaded=false`, and `confirmation_loaded=false`.

## Models and factorial intervention

Run both existing, pinned model families with ten output classes:

- Microsoft ResNet-18, full-parameter AdamW fine-tuning, using the already pinned revision and
  model hashes;
- Google ViT-B/16 IN21K, rank-8 LoRA plus classifier, using the already pinned revision and
  model hashes.

Both models use `lr=3e-4`, `weight_decay=0.1`, batch size 32, 500 warm-up updates, dose 1,250,
no scheduler, and complete model-state checkpoints. The optimizer betas and epsilon remain the
existing AdamW defaults. Restore and validate every parameter and persistent buffer exactly.

Use untouched paired seeds `701, 702, 703, 704, 705`, augmentation seeds `10701` through
`10705`, and the two continuations `clean` and `noisy`. Each model therefore produces ten
scientific bundles, for an exact total of 20. Every bundle evaluates `CC`, `CN`, `NC`, and `NN`
at horizons `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512`. GPU execution is serial. Interim
scientific values cannot stop the remaining matrix.

## Frozen pre-confirmation diagnostic

At the noisy-exposure checkpoint, buffer the first eight continuation batches without updating
the model. Compute the mean cross-entropy against the observed stream labels and divide by
`log(number_of_classes)`. These 256 examples are a diagnostic prefix and are excluded from every
replay endpoint. The policy chooses parameter rollback (`CN`) below its threshold and
optimizer-state repair (`NC`) at or above its threshold.

Fit one threshold per model family using only valid CIFAR-100 discovery snapshots and the same
eight-batch computation: corrected ResNet seeds 601--603 and valid ViT seeds 501--503. Candidate
thresholds are midpoints between adjacent observed discovery scores; ties select the smallest
threshold. Leave-one-seed-out balanced accuracy must be at least 5/6 in each model family.
Thresholds, discovery hashes, scores, and the resulting policy decision are written before the
CIFAR-10N label vector is parsed.

If either threshold fails this calibration gate, the diagnostic is marked `NO-GO` and is not
changed. The 20-bundle mechanism confirmation still runs, but the paper cannot claim predictive
repair utility from this diagnostic.

## Primary estimands

For each model, seed, continuation, and horizon, the lower-is-better repair margin is
`NC - CN`, so positive values favor parameter rollback. For each model and seed, the primary
context contrast is the mean of `margin_noisy - margin_clean` over horizons 32, 64, 128, and 256.
The action-switch diagnostic requires positive clean-continuation margin and negative
noisy-continuation margin at the same primary horizon.

Seed is the independent unit. Models are repeated settings within seed, not ten independent
replicates. The cross-model seed contrast is the equal-weight mean of the ResNet and ViT context
contrasts. Endpoint AUC, horizon-512 tuning loss, and individual horizon curves are secondary.

## Decisions

### Mechanism confirmation

`GO` requires all of the following:

1. all correctness, provenance, isolation, finiteness, and exact-matrix checks pass;
2. all five cross-model seed contrasts are negative, giving an exact one-sided sign-test
   probability of `1/32 = 0.03125` under equiprobable signs;
3. each model family has a negative context contrast in at least four of five seeds;
4. each model family has an action switch at two or more primary horizons in at least four of
   five seeds.

Otherwise the mechanism result is `NO-GO`. No seed, label set, horizon, endpoint, or threshold
may be replaced.

### Predictive repair utility

This decision is eligible only if the discovery calibration gate passed. For each model and seed,
compute trapezoidal clean-loss-excess AUC at horizons 32, 64, 128, and 256, then average it equally
over the clean and noisy continuations. Compare the frozen policy with always-`CN`, always-`NC`,
no repair (`NN`), and the per-context oracle.

`GO` requires the policy to have lower mean loss than both fixed repair policies in each model
family, beat the better fixed repair within at least eight of ten model-by-seed units, and have
pooled normalized oracle regret no greater than 0.25. A zero oracle gap is scored as zero regret
only when the policy ties the oracle; otherwise it fails closed.

Mechanism `GO` with policy `NO-GO` supports a transport-measurement paper but not a deployable
repair-selection claim. Both `GO` decisions authorize the stronger long-journal mainline.

## Integrity, recovery, and resource limits

The immutable contract binds source commit, lockfile, model revisions and hashes, dataset and
label hashes, split, seeds, optimizer, dose, diagnostic prefix, thresholds, horizons, decision
rules, and all discovery inputs. The runner removes stale decisions before validation and writes
new decisions atomically. Completed bundles are reused only when every recorded hash validates;
interrupted bundles resume without changing scientific inputs.

Run on the Windows RTX 5080 with no GPU power cap and at most 12 CPU library threads. Record
wall-clock time, peak VRAM, GPU identity, and temperature telemetry. Estimated GPU time is
5--7 hours. Infrastructure failures may be repaired and exactly resumed; scientific failures
may not be tuned away. This design authorizes no additional confirmation dataset or label set.
