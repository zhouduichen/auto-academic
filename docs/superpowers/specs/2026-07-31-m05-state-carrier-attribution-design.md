# M0.5 Optimizer-State Carrier Attribution and M1 Gate Design

> Date: 2026-07-31  
> Compute: one desktop RTX 5080; at most 10 logical CPU cores  
> Status: approved design; training has not started

## 1. Purpose

M0 established a causal result for label noise: replacing clean AdamW state with the
state produced by one mislabeled pulse changes the subsequent clean trajectory, while
the no-momentum SGD state-only control is exactly null. M0 did not identify whether the
causal carrier is the first moment `m`, the second moment `v`, or their interaction.

M0.5 answers only that attribution question. It is a gate before algorithm design, not
an additional benchmark and not publication-level evidence.

## 2. Scope

- Dataset/model: the same CIFAR-100, pretrained ViT-B/16, and rank-8 LoRA setup as the
  valid M0 runs.
- Pulse: `label_flip` only. The M0 input-degradation result was directionally unstable
  and is excluded from method selection.
- Optimizer: AdamW only. The existing no-momentum SGD runs remain the negative control.
- Seeds: the same three paired seeds as M0.
- Replay horizons: `0, 1, 2, 4, 8, 16, 32, 64, 128`.
- Test set: never loaded.

## 3. Factorial State Intervention

For each seed, execute one clean pulse and one label-flip pulse from the same warmed
checkpoint. Let `(theta_c, m_c, v_c)` and `(theta_d, m_d, v_d)` denote their post-pulse
snapshots. Construct five branches:

| Branch | Parameters | First moment | Second moment | Role |
|---|---|---|---|---|
| control | `theta_c` | `m_c` | `v_c` | paired reference |
| m-only | `theta_c` | `m_d` | `v_c` | first-moment carrier |
| v-only | `theta_c` | `m_c` | `v_d` | second-moment carrier |
| state-both | `theta_c` | `m_d` | `v_d` | reproduce M0 state-only effect |
| parameter-only | `theta_d` | `m_c` | `v_c` | audit separation from parameter damage |

All branches replay identical clean batches, augmentations, sample IDs, and RNG states.
The three bundles are the experimental units; horizon rows are repeated measurements,
not independent samples.

## 4. Outcomes and Attribution Rule

The primary outcome is signed AUC of clean-loss excess over 128 replay steps relative to
control. Secondary outcomes are final clean-loss excess, recovery half-life, parameter
distance, gradient cosine, and state distance.

For each seed define `A_m`, `A_v`, and `A_both`. Define the interaction contrast as:

`A_interaction = A_both - A_m - A_v`.

A component is an eligible carrier only when its AUC is positive in all three seeds and
its median magnitude is at least 25% of the median `A_both` magnitude.

- **m-dominant:** only `m` is eligible, or median `A_m` is at least twice median `A_v`
  and the interaction is below 25% of `A_both`.
- **v-dominant:** the symmetric rule for `v`.
- **coupled:** both are eligible without 2x dominance, or the median interaction is at
  least 25% of median `A_both`.
- **inconclusive:** `A_both` is not positive in all seeds, signs are unstable, or the
  classification changes under leave-one-seed-out analysis.

The inconclusive outcome stops optimizer development. Extra datasets must not be used to
rescue an ambiguous carrier result.

## 5. Candidate Method Decision

The eventual method controls admission into persistent optimizer memory, not whether the
current example is allowed to affect parameters.

- If `m` dominates, protect only first-moment admission and retain magnitude-faithful
  standard second-moment updates.
- If `v` dominates, retain standard first-moment updates and use a second-moment hold or
  floor so rejected observations neither inflate nor shrink the denominator memory.
- If the effect is coupled, use an ephemeral AdamW candidate state for the current
  parameter update and a separately maintained persistent state. Admission uses a
  two-timescale consistency score with hysteresis; one batch cannot change admission
  state.

The candidate must not add a model, clean reference data, a known noise rate, or an extra
forward/backward pass. A difficult but correct batch keeps its immediate update. Only its
ability to influence future steps is restricted.

Simple norm thresholding, unconditional moment resets, and one-step adjacent-gradient
agreement are rejected: they are respectively too close to clipping, erase useful
history, or admit recurring systematic label errors too easily.

## 6. M1 Sequential Gate

M1 starts only after M0.5 yields a non-ambiguous carrier and the corresponding method is
implemented with state-level unit tests.

### M1-A: 12-run early rejection gate

- Methods: AdamW, tuned global-norm Clip-AdamW, candidate.
- Conditions: clean and 40% deterministic instance-dependent label noise.
- Seeds: two strictly paired seeds.
- Total: `3 methods x 2 conditions x 2 seeds = 12` runs.

Continue only if the candidate improves the noisy primary metric by at least 1.0
percentage point over the strongest simple baseline, loses no more than 0.5 points on
clean data, and adds no more than 5% wall-clock or peak VRAM.

### M1-B: bounded direct-prior check

Only after M1-A passes, add the frozen strongest compatible alignment/selective-update
baseline, the 20% noise condition, and any missing paired comparison. M1-A plus M1-B may
not exceed 22 runs. Hyperparameter trial counts must be identical across methods and are
reported separately from evaluated runs.

The clean validation set supplies M1 decisions. The test set remains sealed until the
algorithm and hyperparameter rule are frozen for formal evidence.

## 7. CPU and Thermal Policy

The process may be affinitized to at most 10 logical CPU cores, but routine work is kept
well below that ceiling:

- `torch.set_num_threads(4)` and one inter-op thread;
- `DataLoader(num_workers=2, pin_memory=True, persistent_workers=True,
  prefetch_factor=2)`;
- one experiment at a time, Windows priority `BelowNormal`;
- asynchronous host-to-device copies and a GPU-fitting batch size;
- no parallel seed execution and no CPU-heavy online corruption generation.

CIFAR-100 is memory-resident and this configuration should feed one RTX 5080 without
requiring high CPU utilization. GPU utilization is not itself a scientific objective;
throughput and wall-clock are. Worker count must not be raised merely to make the GPU
utilization graph look higher.

## 8. Audit and Failure Handling

Each bundle must persist the run configuration, source commit, split/noise/train seeds,
sample IDs, corruption IDs, replay manifest, branch construction manifest, trajectory
metrics, stdout, and SHA-256 manifest. The runner fails closed on pretrained-weight
loading anomalies, non-finite metrics, state-schema mismatch, RNG mismatch, missing
horizons, or accidental test loading.

Required tests before submission:

1. exact tensor-level construction of all five branches;
2. `m-only` changes no parameter or `v` tensor at horizon zero;
3. `v-only` changes no parameter or `m` tensor at horizon zero;
4. branch norm and RNG invariants;
5. deterministic tiny-model replay;
6. artifact and source-commit validation.

## 9. Stop Conditions

Stop the method line if any of the following occurs:

- M0.5 carrier attribution is inconclusive;
- the candidate cannot beat clipping under equal tuning and compute;
- gains require additional passes, a clean reference set, or knowledge of noise rate;
- the clean non-inferiority or 5% efficiency limits fail;
- the mechanism is claim-equivalent to a verified direct prior.

