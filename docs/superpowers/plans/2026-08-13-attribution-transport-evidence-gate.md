# Attribution Transport Evidence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the completed 24-bundle Gate B run into a locally verified discovery-evidence package, apply the available ARIS integrity/refinement gates, and stop with an approved external-sentinel plan without launching new GPU work.

**Architecture:** A small read-only evidence module validates the immutable Gate B root and derives transport/repair summaries from the existing factorial endpoints. Deterministic checks run before any reviewer judgment. ARIS then refines the JMLR thesis and generates the first conditional sentinel plan; the old Gate A/B implementation plan remains historical and is not rerun.

**Tech Stack:** Python 3.12, existing causal-transport modules, JSON/CSV, pytest, Ruff, SHA-256, project-local ARIS skills.

## Global Constraints

- Preserve `GATE_B_DECISION.json` as `NO-GO`; no threshold, endpoint, seed, dose, or continuation changes.
- Consume exactly seeds `(401, 402, 403)`, doses `(1, 8, 64, 1250)`, continuations `("clean", "noisy")`, and the two frozen primary outcomes.
- Treat paired training seed as the experimental unit; branches and horizons are repeated measurements.
- Existing artifacts are discovery evidence, not independent confirmation.
- Do not launch GPU work in this plan. The terminal output is a frozen, conditional external-sentinel plan.
- Keep ARIS native `ssh`, `scp`, `experiment-bridge`, `experiment-queue`, `monitor-experiment`, `run-experiment`, and `result-to-claim` blocked under the current safe profile.
- If no independent reviewer backend is callable, write a provisional verdict and do not mark a supportive claim accepted.
- Do not touch unrelated untracked workspace files.

---

## File structure

- Create `experiments/attribution_transport_evidence.py`: validate one immutable Gate B result root and derive discovery-only transport and repair records.
- Create `tests/test_attribution_transport_evidence.py`: exact-matrix, fail-closed validation, handoff, continuation-agreement, and repair-contrast tests.
- Runtime-only `results/causal-transport-b-e34cca9/`: mirrored decision, contract, 24 summaries, and SHA-256 manifest; never edited by the analysis module.
- Runtime-only `results/causal-transport-b-e34cca9/DISCOVERY_EVIDENCE.json`: deterministic derived evidence with source hashes.
- Runtime-only `.aris/claims.json` and `.aris/evidence_precheck.json`: machine-checkable claim citations.
- Runtime-only `EXPERIMENT_AUDIT.md`, `EXPERIMENT_AUDIT.json`, and `CLAIMS_FROM_RESULTS.md`: integrity and claim verdicts; supportive verdicts remain provisional without an independent backend.
- ARIS outputs `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/EXPERIMENT_PLAN.md`, and `refine-logs/EXPERIMENT_TRACKER.md`: canonical refinement outputs only.

### Task 1: Validate and summarize immutable Gate B evidence

**Files:**
- Create: `experiments/attribution_transport_evidence.py`
- Create: `tests/test_attribution_transport_evidence.py`

**Interfaces:**
- Consumes: a result root containing `CAUSAL_TRANSPORT_CONTRACT.json`, `GATE_B_DECISION.json`, and `gate-b/<bundle>/summary.json`.
- Produces: `load_gate_b(root: Path) -> dict[tuple[int, int, str], dict[str, object]]`.
- Produces: `build_discovery_evidence(root: Path) -> dict[str, object]`.
- Produces: `write_discovery_evidence(root: Path, evidence: dict[str, object]) -> Path`.

- [ ] **Step 1: Write failing exact-matrix and provenance tests**

```python
def test_load_gate_b_requires_exact_matrix(tmp_path: Path) -> None:
    root = _gate_b_root(tmp_path)
    summaries = evidence.load_gate_b(root)
    assert set(summaries) == {
        (seed, dose, continuation)
        for seed in (401, 402, 403)
        for dose in (1, 8, 64, 1250)
        for continuation in ("clean", "noisy")
    }


def test_load_gate_b_rejects_go_or_mismatched_contract(tmp_path: Path) -> None:
    root = _gate_b_root(tmp_path)
    _rewrite_json(root / "GATE_B_DECISION.json", {"decision": "GO"})
    with pytest.raises(RuntimeError, match="NO-GO"):
        evidence.load_gate_b(root)
```

The fixture writes finite summaries with the exact effect/endpoint schema, one source commit, one contract hash, and `test_loaded=false`. Additional tests remove one bundle, duplicate one key, set a non-finite effect, change a source commit, change a contract, and set `test_loaded=true`; each must fail closed.

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `uv run --locked pytest tests/test_attribution_transport_evidence.py -q`

Expected: collection fails because `experiments.attribution_transport_evidence` does not exist.

- [ ] **Step 3: Implement exact read-only validation**

```python
SEEDS = (401, 402, 403)
DOSES = (1, 8, 64, 1250)
CONTINUATIONS = ("clean", "noisy")


def load_gate_b(root: Path) -> dict[tuple[int, int, str], dict[str, object]]:
    decision = _read_json(root / "GATE_B_DECISION.json")
    if decision.get("decision") != "NO-GO" or decision.get("bundle_count") != 24:
        raise RuntimeError("Gate B must be the complete frozen NO-GO")
    contract_hash = decision.get("contract_sha256")
    source_commit = decision.get("source_commit")
    summaries: dict[tuple[int, int, str], dict[str, object]] = {}
    for path in sorted((root / "gate-b").glob("*/summary.json")):
        summary = _read_json(path)
        request = _mapping(summary, "request")
        key = (int(request["seed"]), int(request["dose"]), str(request["continuation"]))
        _validate_summary(summary, contract_hash=contract_hash, source_commit=source_commit)
        if key in summaries:
            raise RuntimeError("duplicate Gate B bundle")
        summaries[key] = summary
    expected = {
        (seed, dose, continuation)
        for seed in SEEDS
        for dose in DOSES
        for continuation in CONTINUATIONS
    }
    if set(summaries) != expected:
        raise RuntimeError("Gate B bundle matrix is incomplete")
    return summaries
```

Validation reuses `causal_transport_gate.PRIMARY_OUTCOMES` and `classify_effect`; it never changes their thresholds.

- [ ] **Step 4: Add failing derived-evidence tests**

```python
def test_discovery_evidence_reports_agreement_handoff_and_repair(tmp_path: Path) -> None:
    root = _gate_b_root(tmp_path)
    record = evidence.build_discovery_evidence(root)
    assert set(record["by_dose"]) == {"1", "8", "64", "1250"}
    assert record["cohort_stable_parameter_handoff"] == 1250
    assert record["by_dose"]["1250"]["continuation_agreement"] == 1.0
    repair = record["by_bundle"]["seed401-dose1250-clean"]["repair"]
    assert set(repair) == {
        "clean_loss_excess_auc_128",
        "tuning_loss_h512",
    }
```

- [ ] **Step 5: Implement transport and repair derivations**

For each summary and primary outcome, record the unchanged carrier class and carrier ratio. Define continuation agreement at one dose as the fraction of six seed-by-outcome pairs whose clean/noisy continuation classes match. Define a seed handoff as the earliest frozen dose for which both outcomes and both continuations are `parameter` at that dose and every later frozen dose. Define the cohort handoff by the same rule across all three seeds.

For loss-like endpoints, derive direct repair benefits from the observed contaminated branch `NN`:

```python
state_repair = endpoints["NN"][outcome] - endpoints["NC"][outcome]
parameter_repair = endpoints["NN"][outcome] - endpoints["CN"][outcome]
```

Positive values mean the clean component lowers the loss. Store raw values and do not convert them into a supportive paper claim in this task.

- [ ] **Step 6: Write atomically with source hashes**

`write_discovery_evidence` adds SHA-256 values for the contract, decision, and every consumed summary, writes `DISCOVERY_EVIDENCE.json.tmp`, flushes and `fsync`s, then replaces only `DISCOVERY_EVIDENCE.json`. It refuses to overwrite an artifact whose source-hash map differs.

Add the exact module entry point used by Task 2:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    record = build_discovery_evidence(args.root)
    output = write_discovery_evidence(args.root, record)
    print(json.dumps({"bundles": record["bundle_count"], "output": str(output)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run tests and lint**

Run: `uv run --locked pytest tests/test_attribution_transport_evidence.py tests/test_causal_transport_gate.py tests/test_causal_transport_orchestrator.py -q`

Expected: all tests pass.

Run: `uv run --locked ruff check experiments/attribution_transport_evidence.py tests/test_attribution_transport_evidence.py`

Expected: `All checks passed!`

- [ ] **Step 8: Commit the evidence module only**

```bash
git add experiments/attribution_transport_evidence.py tests/test_attribution_transport_evidence.py
git commit -m "feat: derive attribution transport discovery evidence"
```

### Task 2: Mirror and mechanically verify the completed Windows run

**Files:**
- Create runtime artifacts only under: `results/causal-transport-b-e34cca9/`
- Do not commit raw result artifacts.

**Interfaces:**
- Consumes: `C:\arw-results\causal-transport-b-e34cca9` from the Windows evidence source.
- Produces: exact local copies plus `ARCHIVE_PROVENANCE.json` and `DISCOVERY_EVIDENCE.json`.

- [ ] **Step 1: Inventory before transfer**

Record relative path, byte size, and SHA-256 for the contract, decision, and exactly 24 summaries on Windows. Reject extra/missing summary keys before copying. Do not copy checkpoints, cached datasets, or logs unrelated to the decision.

- [ ] **Step 2: Transfer through the approved operator evidence path**

The ARIS safe profile remains unable to call SSH/SCP. Use the existing operator-controlled Windows evidence channel, then compare every local byte size and SHA-256 with the remote inventory. A mismatch deletes only the incomplete local staging directory and retries once; it never changes the source.

- [ ] **Step 3: Run deterministic analysis**

Run:

```bash
uv run --locked python -m experiments.attribution_transport_evidence \
  --root results/causal-transport-b-e34cca9
```

Expected: writes one `DISCOVERY_EVIDENCE.json`, reports `24/24`, preserves decision `NO-GO`, and exits `0`.

- [ ] **Step 4: Verify the known frozen facts**

Mechanically assert:

- 24 summaries and one decision are present;
- the decision is `NO-GO` and `required_crossings=false`;
- dose 1250 has parameter classification for all three seeds, two outcomes, and two continuations;
- no test data were loaded;
- the aggregate source commit and contract match the terminal decision.

Any mismatch blocks Task 3 and is reported as evidence integrity failure.

### Task 3: Apply ARIS integrity and result-to-claim gates

**Files:**
- Create runtime: `.aris/claims.json`
- Create runtime: `.aris/evidence_precheck.json`
- Create runtime: `EXPERIMENT_AUDIT.md`
- Create runtime: `EXPERIMENT_AUDIT.json`
- Create runtime: `CLAIMS_FROM_RESULTS.md`

**Interfaces:**
- Consumes: source code, tests, frozen decision/contract, 24 summaries, and `DISCOVERY_EVIDENCE.json`.
- Produces: one integrity verdict and one claim-support verdict.

- [ ] **Step 1: Define only discovery claims**

Write exactly these candidate claims with numeric citations to `DISCOVERY_EVIDENCE.json`:

1. Local carrier classification is continuation/outcome sensitive in the completed setting.
2. Dose-1250 parameter dominance is stable within the completed three-seed setting.
3. The original stable state-to-parameter crossing claim failed its frozen gate.

Do not claim cross-model, cross-modality, human-noise, or deployable repair generality.

- [ ] **Step 2: Run deterministic evidence precheck**

Resolve `evidence_check.py` using the ARIS canonical helper chain and write `.aris/evidence_precheck.json`. A missing path or value terminally rejects that claim before review. If the helper is unavailable, label the precheck skipped instead of fabricating success.

- [ ] **Step 3: Run the available `experiment-audit` workflow once**

Provide paths, not executor summaries, to the independent reviewer backend. If no independent backend is callable, write `integrity_status: unavailable` and a provisional report; do not self-acquit.

- [ ] **Step 4: Respect the blocked `result-to-claim` adapter**

The current activation profile intentionally returns `skill_not_activated` for `result-to-claim`. Do not bypass the block. If no approved independent adapter is available, write `CLAIMS_FROM_RESULTS.md` beginning with:

```text
verdict: REVIEW_UNAVAILABLE
```

The deterministic `NO-GO` and evidence-existence facts remain valid; supportive paper claims remain provisional and cannot authorize GPU work by themselves.

- [ ] **Step 5: Record the gate outcome**

If integrity fails or any cited number is absent, stop. Otherwise hand the constrained discovery statements, limitations, and reviewer availability status to Task 4.

### Task 4: Run ARIS thesis refinement and create the conditional sentinel plan

**Files:**
- Produce canonical: `refine-logs/FINAL_PROPOSAL.md`
- Produce canonical: `refine-logs/EXPERIMENT_PLAN.md`
- Produce canonical: `refine-logs/EXPERIMENT_TRACKER.md`
- Do not modify experiment runners in this task.

**Interfaces:**
- Consumes: the approved design spec, frozen discovery evidence, integrity report, claim verdict, current literature, one RTX 5080, and the JMLR full-article target.
- Produces: a reviewed thesis and a no-GPU-yet external-sentinel plan.

- [ ] **Step 1: Run novelty refresh before method refinement**

Use ARIS `research-lit` and `novelty-check` around temporal transport of training attribution, optimizer-state/parameter interventions, optimizer memory, training-stage attribution, noisy-label PEFT, and repair selection. Preserve exact nearest-work differences and reject a thesis already performed by prior work.

- [ ] **Step 2: Run `research-refine-pipeline` with the frozen anchor**

The immutable anchor is the failed Gate B crossing plus stable long-exposure parameter carrier. Require one dominant contribution, at most one supporting claim, no Protect-M revival, and no experiment matrix until the thesis review is stable.

- [ ] **Step 3: Generate only the first conditional experiment milestone**

The plan may authorize no more than 2--3 GPU-hours for an external sentinel. It must name the new condition, seeds, anchors, endpoints, exact pass/fail rule, expected artifact set, and the reason each run changes a reviewer belief. Full 50--90-hour confirmation remains conditional and unsubmitted.

- [ ] **Step 4: Run plan self-checks**

Verify no placeholders, no threshold changes, no reuse of discovery seeds as confirmation, no test access, no unsupported general claims, and no ARIS native remote-execution path.

- [ ] **Step 5: Stop before dispatch**

Present the final thesis, novelty verdict, evidence status, sentinel matrix, estimated time, and go/no-go rule. GPU dispatch requires a separate approval of that exact frozen sentinel matrix.

## Completion condition

This incremental plan is complete when the old 24-bundle run has a local hash-verified evidence package, deterministic discovery summary, honest integrity/claim status, and an ARIS-refined external-sentinel plan. It deliberately does not implement or dispatch the sentinel.
