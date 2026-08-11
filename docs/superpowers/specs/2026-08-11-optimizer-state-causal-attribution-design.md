# Optimizer-State Causal Attribution: Ten-Day Evidence Gate

**Date:** 2026-08-11

**Target:** a decisive ten-day A/T1 paper-viability gate, not a completed paper

**Compute boundary:** one Windows RTX 5080, CPU affinity capped at 12 cores, at most eight new GPU-hours

## 1. Research question and claim boundary

The project will no longer optimize or rescue Protect-M. The new question is:

> In noisy-label parameter-efficient fine-tuning, how much of the downstream performance gap is carried by model-parameter displacement, how much is carried by AdamW moment state, and how much requires their interaction?

This is a causal-attribution study, not a new optimizer proposal. It will not claim that optimizer memory is newly discovered or that resetting optimizer state is new. Those claims are already threatened by prior work including *Resetting the Optimizer in Deep RL* and *Process-Tensor Tomography of SGD*. The intended difference is a crossed parameter/state intervention that attributes the noisy-label robustness gap under a common future training sequence.

The ten-day gate succeeds only if it finds a stable, falsifiable effect hierarchy or stage-dependent crossover. An unstable or null decomposition is `NO-GO` for this paper route.

## 2. Existing evidence reused

The analysis treats the following as immutable prior evidence:

- 20 M0.6 causal pulse bundles with `parameter_only`, `m_only`, `v_only`, `state_both`, and control branches;
- 24 M1 Pilot cells: AdamW, CAdam, Protect-M, and Protect-MV across clean/noisy conditions and three paired seeds;
- two M1.1 bounded-rescue cells plus its hardware sentinel and mechanical `NO-GO` decision;
- AdamW, clipping, and beta-control calibration results as exploratory negative controls, not as independent confirmatory samples.

These artifacts supply the short-horizon mechanism evidence and long-horizon failure evidence. They are reanalyzed locally before any new GPU task starts. No existing failed candidate is relabeled as successful, and no test split is opened.

Before any replay result exists, the zero-GPU reanalysis writes and hashes one legacy-evidence prediction: `parameter`, `state`, or `interaction` dominance at each of the three checkpoints. The prediction cannot be changed after the CUDA sentinel starts.

## 3. Counterfactual intervention

At a matched training checkpoint, let `theta_C`, `z_C` denote model parameters and AdamW optimizer state from clean-label training, and `theta_N`, `z_N` the corresponding objects from noisy-label training. Construct the complete 2x2 factorial:

| Branch | Parameters | Optimizer state | Interpretation |
|---|---|---|---|
| CC | `theta_C` | `z_C` | clean reference |
| CN | `theta_C` | `z_N` | state-only contamination |
| NC | `theta_N` | `z_C` | parameter-only displacement |
| NN | `theta_N` | `z_N` | observed noisy state |

Every branch receives bitwise-identical continuation batches, augmentation draws, scheduler state, and RNG stream. Optimizer clocks and parameter-group schemas must match exactly before a transplant is allowed. A mismatch fails closed rather than coercing state.

The primary continuation uses the frozen noisy-label training sequence and measures tuning loss at horizon 512. A secondary clean continuation measures clean-loss-excess AUC through horizon 128 and bridges directly to M0.6. Horizons 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512 are recorded where applicable.

## 4. New experimental units

### Mandatory gate

1. **Four checkpoint-capture runs:** two paired seeds (`301`, `302`) x clean/noisy AdamW. Each runs through epoch 3 and saves exact epoch-1, epoch-2, and epoch-3 states. Existing M1 epoch metrics are reused, but its single retained checkpoint is epoch 20 and therefore cannot substitute for the missing early checkpoints.
2. **Twelve factorial replay bundles:** two seeds x three checkpoints x two continuation regimes. Each bundle contains the four CC/CN/NC/NN branches under one shared replay stream.

This is 16 new GPU jobs and 48 short replay branches. The replay branches are not counted as independent statistical samples; the paired seed is the experimental unit.

### Conditional confirmation

Only if the AdamW gate passes, run four late-stage CAdam replay bundles: two seeds x two continuation regimes. Existing paired clean/noisy epoch-20 CAdam checkpoints are reused, so no new CAdam base training is allowed during the ten-day gate.

The hard maximum is therefore 20 new GPU jobs and 64 replay branches. Expected GPU use is under four hours; eight hours is the fail-closed ceiling including sentinel and one infrastructure retry.

## 5. Outcomes and analysis

For each replay outcome `Y`, report the balanced factorial contrasts:

- parameter main effect: average change from `theta_C` to `theta_N` across both optimizer states;
- state main effect: average change from `z_C` to `z_N` across both parameter states;
- interaction: the residual non-additivity of parameter and state changes.

The primary outcome is tuning-loss difference at noisy-continuation horizon 512. Secondary outcomes are clean-loss-excess AUC through 128, tuning accuracy, prediction-distribution divergence, `d_theta`, `d_m`, `d_v`, and gradient cosine to CC.

Uncertainty uses paired seed-level contrasts and reports both seeds individually. No branch-level pseudo-replication is permitted. With only two seeds, the ten-day result is a mechanism gate, not a final population-level significance claim.

## 6. Mechanical GO/NO-GO rule

The route is `GO` only when all of the following hold:

1. all mandatory artifacts are complete, finite, provenance-valid, and `test_loaded=false`;
2. the dominant parameter/state contrast has the same direction in both seeds at no fewer than two of the three checkpoints;
3. at those checkpoints, the magnitude of the dominant main effect is at least twice the other main effect, or a directionally replicated parameter-state interaction is at least as large as both main effects;
4. the dominant component agrees with the frozen legacy-evidence prediction in both seeds at no fewer than two checkpoints, without changing endpoints or excluding inconvenient runs;
5. the final novelty audit finds no prior work performing the same crossed clean/noisy parameter-state attribution in noisy-label PEFT.

If the conditional CAdam confirmation runs, its final-checkpoint dominant component must match the frozen AdamW hierarchy in both seeds; otherwise the overall route becomes `NO-GO`.

Otherwise the route is `NO-GO`. A failure cannot trigger a threshold sweep, extra seed, longer continuation, new optimizer, or favorable-noise search inside the ten-day budget.

## 7. Data flow, integrity, and artifacts

The local reanalysis reads immutable Windows artifacts and their SHA-256 manifests. New capture checkpoints go to a new result root keyed by the implementation commit. The replay runner validates model keys, tensor shapes/dtypes, optimizer parameter ordering, step counters, scheduler state, source commit, data binding, and RNG payload before constructing any branch.

The gate writes only the minimum necessary artifacts:

- one machine-readable inventory of reused evidence;
- one pre-GPU `LEGACY_EVIDENCE_PREDICTION.json`;
- one JSONL replay table containing all branch outcomes;
- one mechanical `CAUSAL_ATTRIBUTION_DECISION.json`;
- figure-ready tables for the factorial decomposition and cross-stage comparison.

Raw checkpoints and trajectories remain on Windows; compact evidence and hashes are mirrored to the Mac. No paper draft or documentation expansion is part of this gate.

## 8. Ten-day schedule

- Days 1-2: immutable inventory, zero-GPU reanalysis, and novelty refresh;
- Days 3-4: tested capture/transplant/replay implementation and CUDA sentinel;
- Days 5-6: four capture runs and twelve mandatory replay bundles;
- Days 7-8: mechanical analysis, integrity audit, and conditional CAdam confirmation if eligible;
- Day 9: robustness checks using only preregistered endpoints;
- Day 10: final figure-ready tables and `GO/NO-GO` paper-viability decision.

No other Windows experiment is authorized by this design.
