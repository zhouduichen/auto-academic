# Noisy-Data Novelty Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-backed go/no-go decision on one A/T1-caliber research mechanism for single-GPU training with low-quality data, before designing or running new experiments.

**Architecture:** The work is an evidence-first funnel. It freezes an audit protocol, extracts claim-level evidence from the closest papers, expands the search to adjacent fields, audits the existing EuroSAT pilot, subjects candidate mechanisms to kill arguments, and permits at most one candidate to enter a separate experiment-design cycle. Every conclusion is stored as a reviewable artifact; no training code or GPU job is authorized by this plan.

**Tech Stack:** Markdown, CSV, primary-source web research, existing Phase 1 JSON/NumPy artifacts, Git, ARIS-style claim and kill-argument review.

## Global Constraints

- The publication floor is CCF-A Full/Regular paper or a venue in the user's formal institutional T1 list.
- CCF-B/T2, CCF-C, Findings, Short, Demo, Workshop, and preprint-only outcomes are not project goals.
- The research object is low-quality-data training under a fixed single-desktop-RTX-5080 budget.
- AutoResearch is an experiment-planning reference; ARIS is a research-assurance reference. Neither is a paper contribution.
- ReliablePEFT, selection reliability, `/selection-audit`, LCB/Bootstrap packaging, and Agent orchestration are rejected main topics.
- Existing Phase 1 outputs are development pilot evidence only and remain untracked.
- Do not edit training code, install training dependencies, launch GPU work, name a proposed method, or draft a paper under this plan.
- Use only primary sources for technical claims; label preprints, Findings papers, and unpublished submissions accurately.
- Record contradictory and negative evidence with the same detail as supporting evidence.
- If no candidate passes the novelty gate, the correct result is `NO_GO`; do not lower the venue target.

---

## File Map

- Create `docs/research-direction/08_novelty_audit_protocol.md`: frozen search, extraction, inclusion, exclusion, and decision rules.
- Create `docs/research-direction/09_adjacent_work_claim_matrix.csv`: one row per work with claim-level overlap fields.
- Create `docs/research-direction/10_phase1_pilot_audit.md`: salvageable evidence, invalid comparisons, and prohibited reuse of the 48-run pilot.
- Create `docs/research-direction/11_candidate_mechanism_dossiers.md`: at most three candidate mechanisms, each with falsifiable predictions and irreducible differences.
- Create `docs/research-direction/12_kill_argument_review.md`: strongest rejection case and independent verdict for each candidate.
- Create `docs/research-direction/13_novelty_gate_decision.md`: exactly one `GO`, or `NO_GO`.
- Modify `docs/research-direction/04_gap_analysis.md`: replace provisional gaps only after the gate decision.
- Modify `docs/research-direction/05_candidate_topics.md`: promote the selected candidate or record `NO_GO`.
- Modify `docs/research-direction/07_recommendation.md`: point to the gate decision and next approved action.

---

### Task 1: Freeze the Novelty-Audit Protocol

**Files:**
- Create: `docs/research-direction/08_novelty_audit_protocol.md`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-31-high-quality-noisy-data-research-design.md`
- Produces: Frozen definitions and decision rules used by Tasks 2, 3, 5, and 6.

- [ ] **Step 1: Write the protocol header and scope**

Create the document with these exact top-level sections:

```markdown
# 08 — Novelty Audit Protocol

> Frozen date: 2026-07-31
> Publication floor: CCF-A Full/Regular paper or formal institutional T1
> Authorized activity: literature and existing-artifact analysis only

## 1. Research domain
## 2. Definitions
## 3. Search sources and query families
## 4. Inclusion criteria
## 5. Exclusion criteria
## 6. Evidence extraction schema
## 7. Candidate gate
## 8. Stop rules
## 9. Audit trail
```

Define these terms unambiguously in Section 2:

- `data defect`: label error, input degradation, OOD sample, duplicate/low-information sample, or coupled imbalance;
- `sample reliability`: probability that the observed sample/label is valid;
- `sample training value`: expected marginal clean-evaluation improvement per unit compute;
- `compute budget`: GPU wall-clock, optimizer steps, data accesses, peak VRAM, and extra stored state;
- `direct prior`: a work that can implement the candidate claim by changing only dataset, backbone, or parameter values;
- `irreducible difference`: a structural mechanism that disappears if replaced with the nearest prior method.

- [ ] **Step 2: Freeze search query families**

Record these query families, each combined with `2024`, `2025`, and `2026`:

```text
robust parameter-efficient fine-tuning noisy labels
LoRA label noise memorization rank curriculum
sample valuation training dynamics noisy data
compute-aware curriculum learning fixed budget
single-model noisy-label learning efficient
mixed corruption label noise OOD input degradation
hard clean samples versus mislabeled sample selection
audio noisy labels parameter-efficient fine-tuning
cross-modal sample quality estimation
data cleaning AutoML budget allocation
```

Require searches in official proceedings or primary repositories: PMLR, ACL Anthology, CVF Open Access, OpenReview, AAAI, IJCAI, ACM Digital Library, IEEE Xplore, JMLR, and author repositories linked by the paper.

- [ ] **Step 3: Freeze inclusion, exclusion, and gate rules**

Use these inclusion rules:

1. The work changes training, data use, sample selection, routing, weighting, labeling, or compute allocation under low-quality data.
2. The work studies pretrained models or a mechanism plausibly transferable to PEFT.
3. The work provides enough method and experiment detail for claim-level comparison.
4. 2024—2026 works are mandatory; earlier work is included when it defines a strong baseline or core mechanism.

Use these exclusion rules:

1. Secondary summaries without a primary source.
2. Pure inference-time robustness with no training-data-quality component.
3. Hardware noise unrelated to training-data quality.
4. Papers whose only connection is the word “noise.”

Use this candidate gate:

```text
PASS only if:
1. no direct prior implements the same mechanism;
2. the mechanism makes at least one unique falsifiable prediction;
3. a strong simple baseline cannot express the same intervention;
4. the claim can be tested fairly on one RTX 5080;
5. a credible A/T1 evidence package is feasible;
6. the strongest kill argument has a concrete answer.
Otherwise: FAIL.
```

- [ ] **Step 4: Validate and commit**

Run:

```bash
rg -n "TBD|TODO|fill in|later" docs/research-direction/08_novelty_audit_protocol.md
git diff --check
```

Expected:

- `rg` exits with status 1 and prints nothing.
- `git diff --check` exits with status 0.

Commit:

```bash
git add docs/research-direction/08_novelty_audit_protocol.md
git commit -m "docs: freeze noisy-data novelty audit protocol"
```

---

### Task 2: Extract the Anchor-Work Claims

**Files:**
- Create: `docs/research-direction/09_adjacent_work_claim_matrix.csv`

**Interfaces:**
- Consumes: Definitions and gate rules from Task 1.
- Produces: Claim-level prior-art matrix used to reject duplicates in Tasks 3 and 5.

- [ ] **Step 1: Create the matrix schema**

Use this exact CSV header:

```csv
work_id,title,year,status,venue,primary_url,code_url,task_modalities,backbone_family,peft_method,data_defects,noise_assumed_known,clean_reference_required,extra_models,extra_training_stages,core_mechanism,main_claim,mechanism_evidence,efficiency_reported,strongest_result,reported_limitations,overlap_with_candidate_A,overlap_with_candidate_B,direct_prior_risk,audit_notes
```

Use RFC 4180-compatible quoting. Unknown values must be the literal `not_reported`, not an empty cell.

- [ ] **Step 2: Add the five mandatory anchor works**

Read the full method, experiments, ablations, limitations, and supplementary material for:

1. CleaR — [ACL 2024](https://aclanthology.org/2024.acl-long.322/)
2. Delora — [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.792/)
3. RACT — [arXiv 2602.00084](https://arxiv.org/abs/2602.00084), marked `preprint`
4. Normalized Loss Functions — [ICML 2020](https://proceedings.mlr.press/v119/ma20c.html)
5. Fine-tuning Pre-trained Models for Robustness under Noisy Labels — [IJCAI 2024](https://www.ijcai.org/proceedings/2024/403)

For every row:

- describe the mechanism, not the paper's marketing name;
- distinguish label detection from final predictive performance;
- record whether noise rate is assumed known;
- record clean validation/reference data requirements;
- count auxiliary models and training stages;
- record whether wall-clock, GPU hours, VRAM, or only trainable parameters are reported;
- state the most damaging overlap with candidate A/B.

- [ ] **Step 3: Validate row completeness**

Run:

```bash
uv run python - <<'PY'
import csv
from pathlib import Path

path = Path("docs/research-direction/09_adjacent_work_claim_matrix.csv")
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

required = {
    "work_id", "title", "year", "status", "venue", "primary_url",
    "core_mechanism", "main_claim", "reported_limitations",
    "direct_prior_risk", "audit_notes",
}
assert len(rows) == 5, f"expected 5 anchor rows, got {len(rows)}"
assert required <= set(rows[0]), "missing required columns"
for index, row in enumerate(rows, start=2):
    for field in required:
        assert row[field].strip(), f"row {index}: empty {field}"
print("anchor matrix: 5 complete rows")
PY
```

Expected: `anchor matrix: 5 complete rows`

- [ ] **Step 4: Commit**

```bash
git add docs/research-direction/09_adjacent_work_claim_matrix.csv
git commit -m "docs: map anchor noisy-data training claims"
```

---

### Task 3: Expand the 2024–2026 Adjacent-Work Audit

**Files:**
- Modify: `docs/research-direction/09_adjacent_work_claim_matrix.csv`

**Interfaces:**
- Consumes: Task 1 query families and Task 2 schema.
- Produces: A minimum 25-work primary-source corpus spanning all mechanisms that could kill the candidates.

- [ ] **Step 1: Search all ten query families**

For each query family:

1. search official proceedings and primary repositories;
2. follow backward references to the mechanism-defining baseline;
3. follow forward/related work through 2026-07-31;
4. record the search date and rejected false positives in `audit_notes`;
5. never infer acceptance status from arXiv alone.

The final corpus must include at least:

- 5 robust PEFT works;
- 5 training-dynamics/sample-selection works;
- 4 data-valuation or influence works;
- 4 compute-aware/curriculum works;
- 3 mixed-defect or OOD-contamination works;
- 2 real-noise audio works;
- the five anchors from Task 2.

A work may satisfy multiple categories, but the CSV must contain at least 25 distinct works.

- [ ] **Step 2: Mark direct-prior risk**

Use only these values:

```text
high: same problem and structurally equivalent mechanism
medium: same problem or mechanism, but not both
low: useful baseline or adjacent evidence
```

For every `high` row, write a one-sentence explanation in both overlap columns. Do not protect a preferred candidate by calling a direct overlap “medium.”

- [ ] **Step 3: Validate coverage**

Run:

```bash
uv run python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path("docs/research-direction/09_adjacent_work_claim_matrix.csv")
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

assert len(rows) >= 25, f"expected at least 25 works, got {len(rows)}"
status = Counter(row["status"] for row in rows)
risk = Counter(row["direct_prior_risk"].split(":", 1)[0] for row in rows)
assert set(risk) <= {"high", "medium", "low"}, risk
assert risk["high"] > 0, "an audit with no high-risk prior is not credible"
assert all(row["primary_url"].startswith("https://") for row in rows)
print(f"expanded matrix: {len(rows)} works; status={dict(status)}; risk={dict(risk)}")
PY
```

Expected: at least 25 works, at least one `high` risk, and no assertion failure.

- [ ] **Step 4: Commit**

```bash
git add docs/research-direction/09_adjacent_work_claim_matrix.csv
git commit -m "docs: expand recent noisy-data prior-art audit"
```

---

### Task 4: Audit the Existing EuroSAT Pilot

**Files:**
- Create: `docs/research-direction/10_phase1_pilot_audit.md`

**Interfaces:**
- Consumes: `experiments/phase1_eurosat/phase1_train.py`, `experiments/phase1_eurosat/phase1_analyze.py`, and untracked `experiments/phase1_eurosat/outputs/`.
- Produces: A binding salvage/prohibition table for later experiment design.

- [ ] **Step 1: Record artifact inventory without modifying outputs**

Run:

```bash
find experiments/phase1_eurosat/outputs -type f | sort
jq -s '{
  runs:length,
  epochs:(map(.epochs)|unique),
  seeds:(map(.seed)|unique),
  configs:(map(.config_id)|unique|length)
}' experiments/phase1_eurosat/outputs/config_*/seed_*/metrics.json
```

Expected:

- 48 metrics files and corresponding logits/labels;
- epochs include `1` and `10`;
- seeds are `0`, `1`, `2`;
- 16 config IDs.

- [ ] **Step 2: Trace the five known validity issues to code or artifacts**

Document exact evidence for:

1. one 1-epoch run among mostly 10-epoch runs;
2. `seed` controls both dataset permutation and model RNG;
3. analysis selects a `config × seed` cell rather than a configuration-level independent estimate;
4. repeated strategy calls recreate the RNG with the same seed;
5. training data were clean and cannot answer noisy-training claims.

For each issue, classify:

```text
salvageable descriptive evidence
requires reanalysis
requires selective rerun
prohibited for formal claims
```

- [ ] **Step 3: Freeze permitted reuse**

The conclusion must state:

- outputs remain immutable and untracked;
- the pilot may inform feasibility, logging format, runtime, VRAM, and a development-only LoRA region;
- no old test result may select the formal method or tune thresholds;
- no 48-run selection-reliability conclusion carries into the new paper;
- formal experiments require independent `split_seed`, `train_seed`, and `noise_seed`, plus saved sample IDs.

- [ ] **Step 4: Validate and commit**

Run:

```bash
rg -n "epoch|split_seed|train_seed|noise_seed|config × seed|prohibited" docs/research-direction/10_phase1_pilot_audit.md
git diff --check
```

Expected: all six terms are present; diff check passes.

Commit:

```bash
git add docs/research-direction/10_phase1_pilot_audit.md
git commit -m "docs: audit EuroSAT pilot evidence"
```

---

### Task 5: Write at Most Three Candidate Mechanism Dossiers

**Files:**
- Create: `docs/research-direction/11_candidate_mechanism_dossiers.md`

**Interfaces:**
- Consumes: Tasks 2–4 evidence.
- Produces: Candidate mechanisms that can be independently killed in Task 6.

- [ ] **Step 1: Apply a fixed dossier schema**

For each candidate, use these exact subsections:

```markdown
## Candidate [A-C]: [descriptive mechanism, not a branded method name]

### One-sentence research claim
### User problem addressed
### Closest three prior works
### Irreducible structural difference
### Mechanism hypothesis
### Unique falsifiable predictions
### Strongest simple baseline
### Strongest recent baseline
### Minimal 12–24-run probe
### A/T1 evidence path
### Compute feasibility on one RTX 5080
### Kill argument
### Current verdict
```

Do not create a third candidate merely to fill the alphabet. One or two candidates are acceptable; four are prohibited.

- [ ] **Step 2: Enforce falsifiability**

Every candidate must include:

- at least two predictions that distinguish it from its closest prior;
- an observation that would falsify the mechanism;
- an observation that would show the same performance came only from extra compute;
- an observation that would collapse the contribution to an existing baseline.

Reject a candidate immediately if its irreducible difference is only:

- another weighted sum of known signals;
- a modality change;
- a LoRA-rank schedule already covered by RACT;
- dual adapters already covered by Delora;
- clean routing already covered by CleaR;
- a new name for small-loss filtering.

- [ ] **Step 3: Validate no unsupported novelty language**

Run:

```bash
rg -n "first|首次|novel|全新|从未" docs/research-direction/11_candidate_mechanism_dossiers.md
rg -n "Closest three prior works|Irreducible structural difference|Unique falsifiable predictions|Kill argument|Current verdict" docs/research-direction/11_candidate_mechanism_dossiers.md
```

Expected:

- The first command prints nothing and exits 1 unless the word appears inside an explicitly attributed source claim.
- The second command prints every required subsection for every candidate.

- [ ] **Step 4: Commit**

```bash
git add docs/research-direction/11_candidate_mechanism_dossiers.md
git commit -m "docs: propose falsifiable noisy-data mechanisms"
```

---

### Task 6: Run Kill-Argument Review and Make the Gate Decision

**Files:**
- Create: `docs/research-direction/12_kill_argument_review.md`
- Create: `docs/research-direction/13_novelty_gate_decision.md`

**Interfaces:**
- Consumes: Claim matrix, pilot audit, and candidate dossiers.
- Produces: Exactly one authorized candidate or a binding `NO_GO`.

- [ ] **Step 1: Review each candidate from the rejection position**

For each candidate, answer:

1. Which prior work makes this look incremental?
2. Can the same intervention be expressed by an existing loss, selector, router, or rank schedule?
3. Is “single GPU” only an author constraint?
4. Does the proposed cross-modal evidence prove anything beyond code reuse?
5. Could the gain come from more compute, more tuning, or cleaner validation?
6. Is a 12–24-run probe capable of distinguishing the mechanism?
7. What full A/T1 evidence would still be missing after a successful probe?

Assign one verdict:

```text
KILL
REVISE_AND_REAUDIT
SURVIVES_FOR_PROBE
```

- [ ] **Step 2: Obtain an independent review**

Give the reviewer only:

- `08_novelty_audit_protocol.md`;
- `09_adjacent_work_claim_matrix.csv`;
- `10_phase1_pilot_audit.md`;
- `11_candidate_mechanism_dossiers.md`;
- the seven rejection questions above.

Record the reviewer identity/backend, complete prompt, raw verdict, disagreements, and resolution in `12_kill_argument_review.md`. Do not let the candidate author silently override a `KILL`.

- [ ] **Step 3: Write the binding decision**

`13_novelty_gate_decision.md` must contain:

```markdown
# 13 — Novelty Gate Decision

## Decision
GO: Candidate X
```

or:

```markdown
# 13 — Novelty Gate Decision

## Decision
NO_GO
```

For `GO`, include:

- the exact one-sentence claim;
- the irreducible difference;
- two falsifiable predictions;
- the closest prior and why it is not equivalent;
- the maximum authorized probe budget;
- an explicit statement that formal experiments remain unauthorized.

For `NO_GO`, list the overlap that killed each candidate and the next search domain. Do not promote a runner-up automatically.

- [ ] **Step 4: Validate decision cardinality**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path

text = Path("docs/research-direction/13_novelty_gate_decision.md").read_text(encoding="utf-8")
go_lines = [line for line in text.splitlines() if line.startswith("GO: ")]
no_go_lines = [line for line in text.splitlines() if line == "NO_GO"]
assert len(go_lines) + len(no_go_lines) == 1, (go_lines, no_go_lines)
assert not (go_lines and no_go_lines)
print("novelty decision: exactly one outcome")
PY
```

Expected: `novelty decision: exactly one outcome`

- [ ] **Step 5: Commit**

```bash
git add docs/research-direction/12_kill_argument_review.md docs/research-direction/13_novelty_gate_decision.md
git commit -m "docs: decide noisy-data novelty gate"
```

---

### Task 7: Reconcile the Research Direction and Hand Off to Experiment Design

**Files:**
- Modify: `docs/research-direction/04_gap_analysis.md`
- Modify: `docs/research-direction/05_candidate_topics.md`
- Modify: `docs/research-direction/07_recommendation.md`

**Interfaces:**
- Consumes: Binding decision from Task 6.
- Produces: A contradiction-free research state and the authorization boundary for the next design cycle.

- [ ] **Step 1: Update the three current-state documents**

If the decision is `GO`:

- replace provisional gap language with the exact audited claim;
- link the closest-prior matrix and kill review;
- mark only the selected candidate as `authorized for mechanism-probe design`;
- state that no GPU run is authorized until a separate experiment design is reviewed.

If the decision is `NO_GO`:

- mark all candidates rejected;
- keep the broad user problem and A/T1 target;
- identify the next literature-search domain;
- state that experiment design remains blocked by novelty, not by compute.

- [ ] **Step 2: Run consistency checks**

Run:

```bash
rg -n "ReliablePEFT|selection-audit|CCF-B.*目标|T2.*目标" \
  docs/research-direction/04_gap_analysis.md \
  docs/research-direction/05_candidate_topics.md \
  docs/research-direction/07_recommendation.md
git diff --check
git status --short
```

Expected:

- `rg` exits 1 with no conflicting active recommendation;
- `git diff --check` exits 0;
- untracked `experiments/phase1_eurosat/outputs/` remains untracked and unstaged.

- [ ] **Step 3: Commit**

```bash
git add \
  docs/research-direction/04_gap_analysis.md \
  docs/research-direction/05_candidate_topics.md \
  docs/research-direction/07_recommendation.md
git commit -m "docs: reconcile research direction after novelty gate"
```

- [ ] **Step 4: Stop before experiment design**

Report:

- the gate outcome;
- the surviving claim or `NO_GO`;
- the closest prior;
- the strongest unresolved risk;
- the exact next authorized action.

Do not edit `experiments/`, submit jobs, or invoke an implementation plan. A `GO` starts a new brainstorming cycle for the mechanism-probe experimental design.
