# M0 Fixed Runner Implementation Plan

**Goal:** Execute the frozen four-trajectory optimizer-state contamination audit on the
Windows RTX 5080 without exposing raw commands or loading CIFAR-100 test data.

**Architecture:** A deterministic experiment entry point owns data splits, batch/augmentation
manifests, causal branch construction, trajectory replay, metrics, and SHA-256 artifacts. The
Mac submits only a typed matrix to a separately registered `optimizer-state-m0` Windows project.

## Completed locally

- [x] Implement trainable-parameter and optimizer-state snapshot/restore.
- [x] Construct Control, Parameter-only, State-only, and Full branches.
- [x] Norm-match corrupt gradients to the clean gradient.
- [x] Restore Python, NumPy, CPU, and CUDA RNG state across candidates and branches.
- [x] Implement deterministic CIFAR-100 45k/5k split and replay manifests.
- [x] Implement ViT-B/16 + LoRA-r8 AdamW and no-momentum SGD bundles.
- [x] Emit loss-excess AUC, half-life, `D_m`, `D_v`, `D_theta`, gradient cosine, runtime,
      VRAM, and file-level SHA-256.
- [x] Add analytical and small end-to-end tests, including exact SGD State-only control.
- [x] Extend the Mac API contract and CLI with typed M0 matrix fields.

## Windows gate

- [ ] Register a fixed `optimizer-state-m0` project and runner profile on Windows.
- [ ] Deploy the reviewed source commit and reject any non-allowlisted recipe patch.
- [ ] Run one bounded AdamW/label-flip/seed-0 sentinel.
- [ ] Audit its state chain, replay checksum, norm ratio, artifacts, and numerical invariants.
- [ ] Release the remaining eleven bundles only after the sentinel passes.

The current server advertises only `karpathy-autoresearch` and `reliablepeft-phase1`; submitting
M0 to either would run the wrong executor and is forbidden.
