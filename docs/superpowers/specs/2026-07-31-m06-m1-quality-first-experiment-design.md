# M0.6–M1 Quality-First Experiment Design

> Date: 2026-07-31  
> Compute: one desktop RTX 5080; one experiment at a time  
> Scope: independent mechanism confirmation, candidate-structure development, and M1 confirmation  
> Planning boundary: this document stops at the M1 decision and does not authorize F1, new datasets, or cross-modal experiments

## 1. Decision Context

The completed M0.5 pilot used three training seeds and the five paired branches
`control`, `m-only`, `v-only`, `state-both`, and `parameter-only`. Its archived evidence
is `tmp/m05-full-20260731.zip`, with SHA-256:

`e60234600fec472f50881ee1d7fd81e468a55b95bd3b43eb15adb169a6b0b8a8`.

The pilot AUCs were:

| Seed | `A_m` | `A_v` | `A_both` |
|---:|---:|---:|---:|
| 0 | 0.339711 | -0.049936 | 0.227512 |
| 1 | 0.920292 | 0.073930 | 1.192836 |
| 2 | 0.667919 | 0.261654 | 0.542449 |

The full three-seed rule returned `m-dominant`, but leave-one-seed-out decisions were
`m-dominant`, `coupled`, and `m-dominant`. Under the frozen M0.5 rule, the final result is
therefore `inconclusive`.

This design does not relabel that result as confirmatory. Seeds 0–2 remain pilot evidence
used only to choose an effect-size threshold and plan sample size. M0.6 uses a new cohort.
The original M0.5 stop rule is superseded only prospectively by the user-approved M0.6
design; no result from the pilot is added to M0.6 confidence intervals or hypothesis tests.

## 2. Research Questions and Claims

### M0.6 mechanism question

Does first-moment contamination have a reproducible, practically nontrivial causal effect
on future clean loss across independent warm-up trajectories and independent mislabeled
pulses?

M0.6 confirms only the presence of a first-moment effect. It does not try to prove that the
second moment or interaction is exactly zero, and it does not select the final algorithm
structure.

### M1 development questions

1. Can admission control over persistent optimizer memory beat tuned optimizer-level
   alternatives under sustained instance-dependent label noise?
2. Does protecting both first and second moments provide enough incremental benefit to
   justify its additional mechanism and state, or should the method protect the first
   moment only?

### M1 confirmation claim

After method structure, code, training budget, and hyperparameters are frozen, the chosen
candidate must improve noisy-condition accuracy over the strongest tuned simple baseline,
remain non-inferior on clean data, stay within the efficiency budget, and remain competitive
with a direct noisy-label baseline on an accuracy–cost Pareto comparison.

## 3. Global Experimental Constraints

- Use CIFAR-100 and `google/vit-base-patch16-224-in21k` with rank-8 LoRA on query/value.
- Keep classification-head and LoRA training rules identical across compared methods.
- Use one model, one training stage, and one forward/backward pass per batch for the
  candidate.
- The candidate may not access a clean reference set, the true noise rate, clean labels for
  corrupted examples, validation examples during training, or test examples.
- Keep batch size, precision, model, trainable parameter set, augmentation family, and
  optimizer-step budget paired across methods.
- Run one experiment at a time. Do not parallelize seeds.
- Official CIFAR-100 test data must not be constructed or loaded anywhere in M0.6 or M1.
- Every comparison is reported at fixed optimizer steps and fixed wall-clock.
- Failed runs, retries, timing, peak VRAM, data accesses, configuration hashes, and source
  commits are retained.
- Results may not authorize a new dataset, F1 matrix, audio experiment, or paper claim.
  M1 ends with `GO`, `NO-GO`, or `INCONCLUSIVE`; any later stage requires a new design based
  on the completed M1 evidence.

## 4. M0.6 Independent Mechanism Confirmation

### 4.1 Design and randomization

M0.6 uses ten new training seeds, `3` through `12`. Each training seed creates one warm-up
checkpoint and is paired with two independent pulse seeds, `271828` and `161803`. The two
pulses share the same warm-up checkpoint, validation probe, replay batches, replay
augmentations, and replay order for that training seed.

Randomness is separated as follows:

| Source | Value or rule |
|---|---|
| Split seed | `20260731` |
| Train seeds | `3, 4, 5, 6, 7, 8, 9, 10, 11, 12` |
| Pulse seeds | `271828, 161803` per train seed |
| Replay seed | `420000 + train_seed` |
| Probe seed | `20260806` |

Each train-seed/pulse-seed pair produces one bundle with five branches:

| Branch | Parameters | First moment | Second moment |
|---|---|---|---|
| Control | clean | clean | clean |
| M-only | clean | corrupt | clean |
| V-only | clean | clean | corrupt |
| State-both | clean | corrupt | corrupt |
| Parameter-only | corrupt | clean | clean |

The corrupt gradient is norm-matched to the clean gradient. All branches replay the same
128 clean batches and are evaluated at horizons `0, 1, 2, 4, 8, 16, 32, 64, 128`.

There are 20 evidence bundles. The two pulses are nested repeats within a training seed;
the statistical sample size is ten, not twenty.

### 4.2 Factorial estimands

For each bundle, let `A_m`, `A_v`, and `A_both` be signed clean-loss-excess AUC through
horizon 128 relative to Control. Define the 2×2 factorial effects:

```text
M = 0.5 * (A_m + A_both - A_v)
V = 0.5 * (A_v + A_both - A_m)
I = A_both - A_m - A_v
```

`M` is the first-moment main effect, `V` the second-moment main effect, and `I` their
interaction. The two pulse-level estimates are averaged within each training seed before
cross-seed inference.

The pilot first-moment effects were `0.308580`, `1.019599`, and `0.474357`, with mean
`0.600845` and standard deviation `0.372003`. The frozen minimum meaningful effect follows
the previously approved 25% carrier threshold:

```text
delta_m = 0.25 * median(A_both in pilot) = 0.13561234064400196
```

The pilot standardized excess above `delta_m` is approximately `1.25`. Eight independent
training seeds were estimated to approach 90% one-sided power under the pilot variance;
ten are used to protect against pilot variance underestimation, pulse heterogeneity, and
the stricter two-sided interval rule.

### 4.3 Confirmation rule

M0.6 passes only when all conditions hold:

1. The lower endpoint of the two-sided 95% t interval for the ten train-seed mean `M`
   values is greater than `0.13561234064400196`.
2. At least nine of ten train-seed mean `M` values are positive.
3. The lower endpoint of the two-sided 95% t interval for the ten train-seed mean
   `A_both` values is greater than zero.
4. At least nine of ten train-seed mean `A_both` values are positive.
5. For both `M` and `A_both`, each pulse-seed subgroup has a positive mean, and neither
   pulse-seed subgroup alone accounts for more than 75% of the pooled absolute mean
   effect.

Condition 5 is a heterogeneity safeguard, not an additional independent hypothesis test.
For effect `E` in `{M, A_both}`, its pulse-subgroup shares are defined as
`abs(mean(E_p)) / (abs(mean(E_271828)) + abs(mean(E_161803)))`; a zero denominator fails
the safeguard. Both shares must be at most `0.75`.
The primary inference is the train-seed-level `M` interval. `V` and `I` receive effect
estimates, train-seed intervals, raw trajectories, and pulse-stratified summaries but no
binary absence claim.

If any confirmation condition fails, the optimizer-admission algorithm line stops. M0.6
does not permit adding seeds, changing the effect threshold, substituting a pulse, or using
another dataset to rescue the result.

### 4.4 Bridge and artifact audit

Before M0.6, a non-evidence bridge bundle reruns pilot train seed `0` and pulse seed
`314159` with the audited runner. It must match the archived trajectory at every horizon,
including parameter, first-moment, second-moment, optimizer-step, RNG, batch-ID, and
augmentation hashes.

Bit-exact equality is required by default. If a documented CUDA or dependency change makes
bit-exact floating-point equality impossible, the bridge fails closed. A numerical
tolerance may be proposed only in a new reviewed design revision written before any M0.6
evidence bundle runs.

Each bridge and evidence bundle must contain:

- run configuration and exact source commit;
- environment and dependency lock digest;
- split, pulse, replay, augmentation, and probe manifests;
- horizon-zero branch-construction manifest;
- trajectory metrics and summary;
- stdout/stderr and timing breakdown;
- test-loaded audit set to false;
- file-level SHA-256 manifest.

## 5. M1 Data Partition and Noise Isolation

M1 uses a new class-stratified split with split seed `20260801`:

| Partition | Size | Role |
|---|---:|---|
| Train | 40,000 | All M1 training |
| Tuning validation | 2,500 | Budget and hyperparameter selection |
| Development gate | 2,500 | Protect-M versus Protect-MV selection |
| Confirmation gate | 5,000 | Independent M1 confirmation |
| Official test | 10,000 | Never loaded |

Partition IDs and labels are hashed before training. Tuning validation is the only
validation partition visible during calibration and hyperparameter search. Development
gate is opened once after all six M1-Dev method configurations are frozen. Confirmation
gate is opened once after the candidate structure, candidate code, candidate parameters,
and direct baseline are frozen.

M1 compares clean training with exactly 40% instance-dependent label noise. The noise
generator must be frozen in a separately reviewed annex before calibration. The annex
must define its feature extractor, transition formula, exact-40% sampling algorithm,
temperature or concentration parameters, RNG implementation, and primary-source
provenance. It must satisfy all of the following binding requirements:

- construct noise only from the 40,000 training examples and their synthetic-generation
  inputs;
- make both corruption propensity or wrong-class destination depend on the individual
  example;
- corrupt exactly 16,000 examples;
- guarantee every corrupted label differs from its clean label;
- use neither tuning, development-gate, confirmation-gate, nor test information;
- save noisy IDs, clean labels, noisy labels, propensities, transition probabilities,
  feature-extractor digest, and generator digest;
- give all methods the identical manifest for a paired noise seed.

The candidate never receives clean labels from the manifest. They exist only for offline
synthetic-noise diagnostics.

## 6. Candidate Freeze Gate

M1 does not begin merely because M0.6 passes. A separate candidate-mechanism specification
must be reviewed and committed first. It must define the exact recurrence for the
ephemeral parameter-update path, persistent state, consistency score, history update,
admission rule, initialization, bias correction, weight decay, and mixed-precision
behavior.

The two candidates must:

- share the same detector, consistency score, history, initialization, and parameter
  update;
- differ only in whether admission control applies to `m` alone or to both `m` and `v`;
- use at most three nonstandard tunable hyperparameters;
- use one configuration across clean and noisy conditions;
- add no forward/backward pass, model, stage, clean reference, noise-rate estimate, or
  sample deletion;
- preserve the immediate parameter effect of the current batch;
- expose state-level tests showing exactly which tensors are admitted or held.

Protect-M and Protect-MV are working experimental labels, not proposed paper names.

## 7. M1 Budget Calibration

Calibration uses a preregistered AdamW anchor and is excluded from method comparisons. The anchor
is independent of every M1 tuning result: `lr=3e-4`, `weight_decay=0.01`,
`betas=(0.9,0.999)`, `eps=1e-8`, and `amsgrad=false`. All remaining model, LoRA, batch,
augmentation, AMP, scheduler, and parameter-group rules are the common M1 rules.

The binding order is: freeze split/noise/anchor/grid specs; run anchor calibration; freeze the exact
optimizer-step budget and `G_ref`; tune AdamW under that budget; freeze AdamW; resolve the other
method grids; tune the other methods. Calibration is not repeated after AdamW selection.

| Condition | Train seeds | Noise seeds | Maximum budget |
|---|---|---|---:|
| Clean | `101, 102` | not applicable | 50 epochs |
| Noisy | `101, 102` | `1101, 1102` | 50 epochs |

Validation occurs once per epoch on tuning validation. For each condition, choose the
earliest epoch at which the two-seed smoothed tuning accuracy reaches 99% of that
condition's maximum smoothed accuracy and remains within 0.2 percentage points for the
next five evaluations. The fixed M1 budget is the larger clean/noisy epoch, clipped to
`[20, 50]`, converted to an exact optimizer-step count.

All methods then use the same batch size, data accesses, optimizer steps, evaluation
cadence, and final-checkpoint endpoint. There is no early stopping and no best-checkpoint
reporting. Timestamped trajectories also support fixed-wall-clock evaluation using the
median paired runtime of tuned AdamW as the wall-clock budget. For every other method and
seed, the fixed-wall-clock endpoint is the last preregistered evaluation checkpoint whose
elapsed training time does not exceed that budget; metrics are not interpolated between
checkpoints.

## 8. Equal-Compute Hyperparameter Tuning

### 8.1 Seed isolation

| Stage | Train seeds | Noise seeds | Augmentation seeds |
|---|---|---|---|
| Tuning | `201, 202` | `1201, 1202` | `10201, 10202` |
| M1-Dev evaluation | `301, 302, 303` | `1301, 1302, 1303` | `10301, 10302, 10303` |
| M1-Confirm | `401–408` | `1401–1408` | `10401–10408` |

Initialization follows train seed and is paired across methods. Each method receives the
same batch, augmentation, and noise manifests for a given seed and condition.

### 8.2 Methods and search budget

M1-Dev includes six methods:

1. AdamW;
2. global-norm Clip-AdamW;
3. beta-retuned AdamW;
4. the pre-frozen compatible choice of PNM or AdaPNM;
5. Protect-M;
6. Protect-MV.

Each method receives exactly twelve preregistered configurations and the same
successive-halving schedule:

| Rung | Configurations | Step fraction | Train seeds | Conditions |
|---|---:|---:|---|---|
| 1 | 12 | 25% | `201` | clean and noisy |
| 2 | 4 | 50% | `202` | clean and noisy |
| 3 | 2 | 100% | `201, 202` | clean and noisy |

This is 18 full-run equivalents per method. Each method is capped both by identical
optimizer-step/data-access budget and by 105% of the corresponding AdamW tuning
wall-clock. Protect-M and Protect-MV use identical candidate grids. A selected
configuration applies unchanged to clean and noisy runs.

After anchor calibration has frozen the common optimizer-step budget, AdamW is the first method
tuned and supplies the inherited public optimizer configuration and the clean reference. At each rung, configurations for
another method are ranked lexicographically:

1. clean tuning accuracy no more than 0.5 percentage points below AdamW at that rung;
2. higher noisy tuning accuracy;
3. higher fixed-wall-clock noisy accuracy;
4. lower peak VRAM;
5. fewer nondefault parameters;
6. lexicographically lower preregistered configuration ID.

If no configuration satisfies clean non-inferiority, that method has a tuning failure.
To preserve the complete 36-run matrix, it still enters M1-Dev with the configuration
having the highest clean tuning accuracy, followed by the remaining lexicographic
tie-breakers. It is marked clean-ineligible before gate access, cannot become the
strongest simple baseline, and cannot be selected as the candidate. Search ranges may not
be expanded after seeing results. An optimum at a search boundary is recorded as a design
warning; changing the range requires a new design version and unused gate split.

Before development-gate access, the strongest simple baseline is frozen from
Clip-AdamW, beta-retuned AdamW, and PNM/AdaPNM using the same clean constraint and the
two-seed noisy tuning mean. AdamW remains a separate anchor. The strongest simple baseline
cannot be reselected after gate access.

## 9. M1-Dev Evaluation and Structure Selection

M1-Dev evaluates six frozen methods, two conditions, and three paired evaluation seeds:

`6 methods × 2 conditions × 3 seeds = 36 runs`.

The primary endpoint is final-checkpoint top-1 accuracy on development gate at fixed
optimizer steps. Common constraints are fixed-wall-clock top-1, clean non-inferiority,
wall-clock, and peak VRAM. Secondary metrics are balanced accuracy, macro-F1, NLL, ECE,
worst-ten-class mean accuracy, accuracy AUC over epochs, synthetic-noise memorization,
clean-hard/noisy admission summaries, throughput, and data accesses.

For candidate `c` and evaluation seed `s`, define:

```text
Delta_noisy[c,s] = Acc_noisy[c,s] - Acc_noisy[strongest_simple,s]
```

A candidate is eligible only when:

- its mean `Delta_noisy` is at least +1.0 percentage point;
- all three `Delta_noisy` values are positive;
- the worst `Delta_noisy` is at least +0.25 percentage points;
- its mean clean drop relative to AdamW is no more than 0.5 percentage points;
- no seed has a clean drop greater than 1.0 percentage point;
- its mean fixed-wall-clock noisy improvement is positive;
- mean wall-clock and peak-VRAM overhead relative to AdamW are each at most 5%;
- every artifact, finite-value, pairing, and test-isolation audit passes.

M1-Dev does not use p-values. Its three seeds are a strict development gate, not
confirmatory evidence.

Structure selection is deterministic:

- if neither candidate is eligible, stop with M1 `NO-GO`;
- if one candidate is eligible, freeze it;
- if both are eligible, freeze Protect-M unless Protect-MV beats Protect-M by at least
  0.5 percentage points in mean noisy accuracy, is non-worse in all three noisy seeds,
  loses no more than an additional 0.25 percentage points on clean data, and adds no more
  than two percentage points of wall-clock or VRAM overhead relative to Protect-M.

After selection, candidate source, recurrence, hyperparameters, training budget, and all
manifests are frozen. M1-Confirm may not modify them.

## 10. Direct Noisy-Label Baseline Freeze

M1-Confirm adds one direct noisy-label baseline. The pre-ranked choices are:

1. DSS, because its single-network form is the closest recent direct method in the local
   verified claim matrix;
2. ELR, only if a pre-result compatibility audit finds that DSS cannot run without
   changing its core mechanism or violating reproducibility requirements.

The choice is based only on primary-source algorithm requirements, code availability,
license, and compatibility with the frozen data/model protocol. It cannot depend on
candidate performance. The audit and chosen source commit are committed before direct
baseline tuning.

If neither DSS nor ELR can pass the compatibility, source-integrity, and license audit,
M1-Confirm does not start. A different direct baseline requires a reviewed design revision;
it may not be chosen after viewing confirmation data.

The direct baseline receives the same twelve-configuration, 18-full-run-equivalent tuning
budget on tuning validation. Candidate, AdamW, and strongest-simple configurations remain
frozen and are not retuned. Extra forward passes, stages, states, or data accesses used by
the direct method are allowed only when required by its published algorithm and are fully
charged to fixed-wall-clock and efficiency comparisons.

## 11. M1-Confirm

M1-Confirm includes four frozen methods:

1. tuned AdamW;
2. frozen strongest simple baseline;
3. frozen candidate;
4. frozen direct noisy-label baseline.

It evaluates clean and 40% instance-dependent noise with train seeds `401–408`, paired
noise seeds `1401–1408`, and paired augmentation seeds `10401–10408`:

`4 methods × 2 conditions × 8 seeds = 64 runs`.

There is no interim access, optional stopping, or adaptive seed addition. Confirmation
gate is evaluated only after all 64 runs and their manifests validate.

### 11.1 Primary comparisons

For seed `s`:

```text
Delta_noisy[s] = Acc_noisy[candidate,s] - Acc_noisy[strongest_simple,s]
Delta_clean[s] = Acc_clean[candidate,s] - Acc_clean[AdamW,s]
```

The noisy superiority and clean non-inferiority requirements are co-primary intersection
conditions. The experimental unit is seed. Epochs, checkpoints, validation examples, and
individual predictions are not counted as independent experimental replicates.

### 11.2 M1 GO rule

All conditions must hold:

**Noisy superiority**

- mean `Delta_noisy` is at least +1.0 percentage point;
- the lower endpoint of its two-sided 95% paired t interval is above zero;
- at least seven of eight `Delta_noisy` values are positive;
- mean fixed-wall-clock noisy improvement is positive;
- balanced accuracy and macro-F1 are not both lower than the strongest simple baseline.

**Clean non-inferiority**

- mean clean drop is no more than 0.5 percentage points;
- the lower endpoint of the one-sided 95% interval for `Delta_clean` is above -0.5
  percentage points;
- no seed loses more than 1.0 percentage point;
- worst-ten-class mean accuracy loses no more than 1.0 percentage point.

**Efficiency and method contract**

- mean fixed-step wall-clock overhead relative to AdamW is at most 5%;
- peak-VRAM overhead relative to AdamW is at most 5%;
- the candidate uses no extra model, stage, forward/backward, clean reference, true noise
  rate, or sample deletion;
- all artifact, source, pairing, finite-value, and test-isolation audits pass.

**Direct-baseline Pareto check**

At least one condition must hold:

- the lower endpoint of the candidate-minus-direct-baseline two-sided 95% paired interval
  at fixed steps is above -0.5 percentage points; or
- at fixed wall-clock, the lower endpoint is above zero and the mean advantage is at
  least +0.5 percentage points.

If the direct baseline has higher accuracy, lower wall-clock, and lower peak VRAM, the
candidate is strictly dominated and receives `NO-GO`.

### 11.3 NO-GO and INCONCLUSIVE

`NO-GO` applies when any of the following occurs:

- mean noisy improvement is below +1.0 percentage point;
- clean non-inferiority fails;
- the efficiency or candidate-method contract fails;
- fewer than six of eight noisy paired effects are positive;
- exactly six of eight noisy paired effects are positive and either negative paired
  effect is at most -1.0 percentage point;
- clipping, beta retuning, PNM/AdaPNM, or the direct baseline strictly dominates the
  candidate;
- gains require known noise rate, extra reference data, or an unreported compute path.

`INCONCLUSIVE` is reserved for an otherwise valid complete matrix where:

- mean noisy improvement reaches +1.0 percentage point but its confidence interval
  crosses zero;
- exactly six of eight noisy paired effects are positive without a large reverse effect;
- the direct-baseline comparison lies on the predeclared ±0.5-point boundary; or
- a non-algorithmic system failure prevents completion after the one allowed exact retry.

`INCONCLUSIVE` does not authorize more seeds. It requires a result review and a new user-
approved design before any additional GPU experiment.

## 12. Failure, Retry, and Missing-Data Policy

- A verified infrastructure failure may be retried once with the exact same source,
  configuration, seed, manifest, and idempotency key. The failed artifact is retained.
- OOM, numerical divergence, algorithm timeout, and non-finite state are method failures,
  not infrastructure retries.
- An unfavorable stochastic result may not be rerun.
- No outlier is removed. No missing metric or seed is imputed.
- A method with a missing formal matrix cell cannot receive `GO`.
- A split, noise, pairing, source, or test-isolation defect invalidates the entire affected
  gate. Repair requires a new design version and entirely unused gate seeds.

## 13. M1 Deliverables and Planning Stop

M1 ends with:

- calibration, tuning, M1-Dev, direct-baseline, and M1-Confirm manifests;
- all successful, failed, and retried artifacts;
- frozen source commits, dependency digests, and configuration hashes;
- per-seed paired effects, t intervals, sign summaries, secondary metrics, and raw
  trajectories;
- fixed-step and fixed-wall-clock accuracy–cost tables and Pareto plot;
- admission and synthetic-noise diagnostic summaries;
- a machine-readable and human-readable `GO`, `NO-GO`, or `INCONCLUSIVE` report;
- a claim table stating which mechanism and performance claims are supported, partially
  supported, or rejected.

No F1 experiment, new dataset, real-noise benchmark, audio task, theoretical extension,
or paper experiment is planned by this document. The next decision is made only after the
complete M1 evidence is reviewed.
