# Idea Discovery Report

> Direction: fixed-budget single-GPU training under unknown or mixed data defects
>
> Pipeline: `research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline`
>
> Current checkpoint: Phase 1 literature landscape in progress
>
> This is the single composed ARIS report. Per-paper evidence remains in the CSV matrix and verification artifacts; no separate literature-review Markdown file is created.

## Executive Status

No method candidate is authorized yet. The current broad formulation —
“estimate sample training value and allocate compute under a fixed budget” — is
not itself novel enough to enter experiments.

Two recent works directly occupy most of that formulation:

- [Compute-Constrained Data Selection (ICLR 2025)](https://openreview.net/forum?id=4es2oO9tw1) jointly budgets selection and fine-tuning cost and finds that expensive selectors are often not compute-optimal.
- [Computational Budget Should Be Considered in Data Selection (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/22c799f287fd05e7174fd65a3ce134af-Abstract-Conference.html) makes the compute budget part of the selection objective through bilevel optimization.

The research must therefore identify a mechanism that those methods fail to
express under low-quality or mixed-defect training data. “Single GPU” is a
constraint for that investigation, not the missing contribution.

## Search Execution

### Sources that contributed

- ARIS arXiv adapter: contributed structured title, abstract, author, category, and date metadata.
- Primary web sources: contributed official ACL Anthology, PMLR, OpenReview, CVF, NeurIPS proceedings, IEEE, DCASE, and dataset pages.
- Existing project matrix: 98 previously collected records were screened for reuse.

### Degraded sources

- Semantic Scholar adapter returned HTTP 429 and did not contribute.
- OpenAlex adapter was unavailable because `requests` is absent from the current project runtime.
- No local `papers/` or `literature/` PDF collection and no `research-wiki/` were present.

The aggregate remains valid because arXiv and primary web sources contributed.
Missing sources are an audit limitation, not a reason to label unresolved papers
as verified.

### Current corpus state

- Eight focused arXiv query families produced 113 de-duplicated candidates from 2019–2026 before relevance screening.
- Forty high-priority candidates entered ARIS paper-existence verification.
- The old 98-row matrix is a discovery cache, not claim-level evidence; surveys, workshops, unrelated PEFT papers, and duplicate preprint/publication records do not count toward the direct-prior set.

## Literature Landscape

### 1. Robust PEFT under noisy labels is already an established subproblem

| Work | Status | Mechanism that constrains us | Risk |
|---|---|---|---|
| [CleaR](https://aclanthology.org/2024.acl-long.322/) | ACL 2024 long | routes likely clean examples through PEFT modules while bypassing suspected noise | high against clean routing or PEFT gating |
| [TURN](https://www.ijcai.org/proceedings/2024/403) | IJCAI 2024 | tunes the classifier first, reduces the noisy subset, then fine-tunes the pretrained model | high against simple two-stage filtering |
| [Delora](https://aclanthology.org/2025.findings-acl.792/) | ACL Findings 2025 | separates clean/noisy memorization with dual LoRA modules for detection and later training | high against dual-adapter detectors |
| [RACT](https://arxiv.org/abs/2602.00084) | 2026 preprint | connects LoRA capacity/rank to noise memorization and uses rank discrepancy plus temporal separation | high against rank scheduling or two-rank detection |
| [Robustness to Noisy Labels in Parameter Efficient Fine-tuning](https://openreview.net/forum?id=-JjxqH-CH14) | ACL ARR short submission record | reports LoRA noise/imbalance behavior and forgetting dynamics | medium; empirical conclusions must be compared, publication status needs care |

Consequence: a proposal cannot claim novelty from “using LoRA under label
noise,” restricting adapter exposure, separating two adapters, changing rank
with noise, or merely observing PEFT memorization.

### 2. “Difficult but useful” versus “mislabeled” is already a central sample-selection question

| Work | Status | Mechanism that constrains us | Risk |
|---|---|---|---|
| [Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e826f682178d9830a3b704141b4989e-Abstract-Conference.html) | NeurIPS 2025 | identifies particularly harmful mislabeled-easy examples through later-state recalibration | high |
| [Handling Label Noise via Instance-Level Difficulty Modeling and Dynamic Optimization](https://proceedings.neurips.cc/paper_files/paper/2025/hash/429e7b31625a8b7839f9e4d6e2aa9bb9-Abstract-Conference.html) | NeurIPS 2025 | uses wrong-event histories to model cleanliness and difficulty, then dynamically weights instances | high |
| [Debiased Sample Selection for Learning with Noisy Labels](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.html) | CVPR 2026 | targets class- and instance-level confirmation bias and explicitly preserves hard samples | high |
| [Mitigating Label Noise on Graphs via Topological Sample Selection](https://proceedings.mlr.press/v235/wu24ae.html) | ICML 2024 | preserves informative boundary samples using topology rather than generic small-loss filtering | medium |
| [Revisiting sample weights based method for noisy-label detection and classification](https://openaccess.thecvf.com/content/ACCV2024/html/Hoang_Revisiting_sample_weights_based_method_for_noisy-label_detection_and_classification_ACCV_2024_paper.html) | ACCV 2024 | learns usefulness weights from a small clean reference set | high if a clean reference set is assumed |

Consequence: “do not discard difficult clean samples” and “reweight by training
dynamics” are not sufficient differences. A surviving mechanism must specify
what information it uses that wrong events, temporal consistency, class
calibration, feature neighborhoods, and clean-reference meta-weighting cannot
recover.

### 3. Compute-aware data selection directly overlaps the current broad thesis

| Work | Status | Mechanism that constrains us | Risk |
|---|---|---|---|
| [Compute-Constrained Data Selection](https://openreview.net/forum?id=4es2oO9tw1) | ICLR 2025 | jointly accounts for selector cost and downstream training gain across budgets | critical |
| [Computational Budget Should Be Considered in Data Selection](https://proceedings.neurips.cc/paper_files/paper/2025/hash/22c799f287fd05e7174fd65a3ce134af-Abstract-Conference.html) | NeurIPS 2025 | budget-conditioned bilevel subset selection | critical |
| [MATES](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c4bec0d2fd217e6c2c3eafeced432582-Abstract-Conference.html) | NeurIPS 2024 | continually adapts a small influence model to the learner's changing data preference | high |
| [QuRating](https://arxiv.org/abs/2402.09739) | 2024 preprint/venue pending verification | combines learned quality ratings, diversity, selection, and curriculum | medium |
| [CoIDO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f3f2ff9579ba6deeb89caa2fe1f0b99c-Abstract-Conference.html) | NeurIPS 2025 | couples importance and diversity with a lightweight scorer for multimodal instruction tuning | medium |

Consequence: the project cannot present “selection cost must count,” “value
changes over training,” “importance plus diversity,” or “budget-conditioned
selection” as new by themselves.

### 4. Scalable data valuation is strong but has unresolved reliability costs

| Work | Status | Relevant result or limitation | Risk |
|---|---|---|---|
| [What is Your Data Worth to GPT?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6d26053b977f8c589669fd201615119-Abstract-Conference.html) | NeurIPS 2025 | LoGra makes influence estimation much cheaper through gradient projection | high against generic efficient influence scoring |
| [Distributional Training Data Attribution](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html) | NeurIPS 2025 | treats training randomness as part of attribution and applies it to ViT data pruning | high against deterministic single-run “value” claims |
| [Taming Hyperparameter Sensitivity in Data Attribution](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f7dee2af577ab06037891f264157d718-Abstract-Conference.html) | NeurIPS 2025 | shows attribution methods are sensitive and expensive to tune because evaluation requires retraining | critical evaluation requirement |
| [DATE-LM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e1ebda145808ca45774993fb67314894-Abstract-Datasets_and_Benchmarks_Track.html) | NeurIPS 2025 Datasets & Benchmarks | reports that no attribution method dominates and simple baselines remain competitive | critical baseline requirement |

Consequence: any “sample training value” must be defined over training
randomness, include the cost of estimating and tuning it, and beat cheap
selectors. A single trajectory score cannot be called expected marginal value
without evidence.

### 5. Mixed defects and open-set contamination are not an empty space

High-priority works include AEON (joint in-distribution and out-of-distribution
noise-rate estimation), semantic-contamination methods, Extended T,
EvidentialMix, open-set noisy-label learning, and early-learning methods for
joint localization/categorization noise. These papers mean that merely combining
closed-set label flips with OOD samples is not a contribution.

The remaining question is narrower: whether a compute-aware policy can remain
useful when defect type and rate are unknown and when the same observable
training signal conflates hard-clean, mislabeled, OOD, duplicate, and degraded
inputs. That question is not yet a validated gap; full method-level extraction is
still required.

### 6. Audio is useful as real-noise evidence, not as an independent research branch

| Work/resource | Role |
|---|---|
| [Learning Sound Event Classifiers from Web Audio with Noisy Labels](https://arxiv.org/abs/1901.01189) | introduces FSDnoisy18k and shows that abundant noisy web audio can outperform a smaller clean set |
| [Model-agnostic Approaches to Handling Noisy Labels When Training Sound Event Classifiers](https://arxiv.org/abs/1910.12004) | establishes low-overhead label smoothing, mixup, and robust-loss baselines |
| [Learning with Out-of-Distribution Data for Audio Classification](https://arxiv.org/abs/2002.04683) | treats OOD clips as a major real audio defect and studies detection/relabeling |
| [Audio Tagging by Cross Filtering Noisy Labels](https://arxiv.org/abs/2007.08165) | provides an audio-specific cross-filtering baseline |
| [ARCA23K](https://arxiv.org/abs/2109.09227) | supplies a real open-set-noise audio dataset with a verified companion set |

Consequence: audio can test whether a visual mechanism survives real open-set
and web-label noise. It should not be added to the mechanism probe before the
visual mechanism and compute accounting pass.

## Current Gap Statement

The only gap formulation still worth auditing is:

> Under a fixed total budget that includes scoring overhead, can one online
> training intervention distinguish marginally useful hard-clean examples from
> harmful or redundant examples when defect type and rate are unknown, in a way
> that existing budget-aware selection and noisy-label dynamics methods cannot
> express?

This is a search anchor, not a novelty claim. It will fail if the detailed reading
shows that CADS/MATES plus IDO/Early Cutting/DSS can express the same
intervention.

## Candidate Status

No candidate has passed `/idea-creator` filtering. Candidate generation is
intentionally deferred until the following direct-prior cluster is extracted at
method, cost, assumption, and experiment level:

1. Compute-Constrained Data Selection;
2. CADS;
3. MATES;
4. IDO;
5. Early Cutting;
6. Debiased Sample Selection;
7. CleaR;
8. TURN;
9. Delora;
10. RACT;
11. LoGra;
12. Distributional Training Data Attribution.

## Immediate Next Actions

1. Finish existence/status verification for the 40 high-priority candidates.
2. Populate `docs/research-direction/09_adjacent_work_claim_matrix.csv` with the 12 direct killers first, then adjacent baselines.
3. Read methods, ablations, compute accounting, clean-reference assumptions, and limitations for the 12-paper set.
4. Only then run `/idea-creator`; reject any proposal equivalent to a known selector, router, loss, dual adapter, rank schedule, influence scorer, or bilevel budget allocator.
5. Run `/novelty-check` and `/research-review` before authorizing a 12–24-run mechanism probe.

## Decision

PENDING_LITERATURE_AUDIT
