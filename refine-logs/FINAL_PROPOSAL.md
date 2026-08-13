# Final Proposal: Transport-Aware Training Attribution

**Target:** JMLR regular full-length article  
**Status:** `CONDITIONAL_PASS_FOR_EXTERNAL_SENTINEL`  
**Evidence status:** discovery evidence verified; independent ARIS reviewer unavailable

## Dominant contribution

Define and test **training-attribution transportability**: whether a causal attribution of training damage to learned parameters versus optimizer state remains valid across exposure and subsequent continuation. Use checkpoint-matched factorial interchange of parameters and optimizer state to estimate the carrier, its transport gap, and its handoff time. Then test whether the inferred carrier predicts which intervention—parameter rollback or optimizer-state reset—actually repairs the damaged trajectory.

The paper has one dominant claim and one supporting claim:

1. Carrier attribution can be locally unstable yet converge to a stable carrier after continued exposure; therefore a single-time attribution is not generally transportable.
2. A transport-aware carrier estimate predicts the better matched-budget repair action more reliably than fixed reset policies and simple training proxies.

The completed CIFAR-100/ViT-LoRA run is discovery evidence only. Its frozen Gate B remains `NO-GO`; it supplies the counterexample that motivates transportability and the prediction that long-exposure damage is parameter-carried.

## Novelty decision

`CONDITIONAL_PASS`: the literature refresh found close components but no exact end-to-end match.

- [Process-Tensor Tomography of SGD](https://arxiv.org/abs/2601.16563) measures non-Markovian training memory and uses optimizer reset as a causal break, but does not factorially separate parameter and optimizer-state damage across exposure or use the attribution to choose a repair.
- [Accountability Attribution](https://openreview.net/forum?id=hn1n9QjZfb) attributes behavior to training stages while modeling optimization dynamics, but does not identify a parameter-versus-state carrier through checkpoint interchange or study transport of that carrier.
- [Resetting the Optimizer in Deep RL](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html) evaluates a fixed optimizer-reset intervention in non-stationary RL, not an attribution-selected repair in supervised noisy-label training.
- [MAGIC](https://openreview.net/pdf/ce14195844d930682afeacdf97da9eaa36bcd599.pdf) includes optimizer state in optimizer-aware data attribution, but targets data value rather than the temporal carrier and repair question.
- [In-Run Data Shapley for Adam](https://openreview.net/pdf/f58723802dc13f12eecfe0992a50763c663c3d80.pdf) strengthens the case that attribution is optimizer-dependent, but does not perform the proposed state/parameter causal transport test.

The novelty claim must be re-audited before submission because the independent ARIS reviewer is currently unavailable. The paper must not be framed merely as another optimizer-reset or noisy-label method.

## Falsifiable prediction

Under a new exact-rate symmetric-uniform label-noise channel, the long-exposure anchor will again be parameter-carried across all new paired seeds and both continuations, and parameter rollback will reduce both frozen loss outcomes more than optimizer-state reset. Failure ends the JMLR route before the larger matrix.

## Conditional full-paper evidence

Only after the external sentinel passes: add a genuinely different model family, at least one additional dataset or modality, disjoint confirmation seeds, matched-budget fixed-reset and proxy baselines, decisive component interventions, uncertainty at the paired-seed level, and a held-out repair-selection evaluation. Protect-M is excluded.
