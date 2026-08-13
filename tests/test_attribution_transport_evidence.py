from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import attribution_transport_evidence as evidence

CONTRACT = "a" * 64
COMMIT = "b" * 40
OUTCOMES = ("clean_loss_excess_auc_128", "tuning_loss_h512")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _effects(kind: str) -> dict[str, float]:
    return {
        "state": {"parameter": 1.0, "state": 4.0, "interaction": 0.1},
        "parameter": {"parameter": 4.0, "state": 1.0, "interaction": 0.1},
        "mixed": {"parameter": 1.0, "state": 1.0, "interaction": 0.1},
    }[kind]


def _summary(seed: int, dose: int, continuation: str) -> dict[str, object]:
    kind = "state" if dose == 1 else "mixed" if dose == 8 else "parameter"
    effects = {outcome: dict(_effects(kind)) for outcome in OUTCOMES}
    if seed == 401 and dose == 64 and continuation == "noisy":
        effects["tuning_loss_h512"] = dict(_effects("mixed"))
    endpoints = {
        "CC": {OUTCOMES[0]: 0.0, OUTCOMES[1]: 0.0},
        "CN": {OUTCOMES[0]: 1.0, OUTCOMES[1]: 1.0},
        "NC": {OUTCOMES[0]: 2.0, OUTCOMES[1]: 2.0},
        "NN": {OUTCOMES[0]: 4.0, OUTCOMES[1]: 4.0},
    }
    return {
        "schema": "causal-transport-bundle/1",
        "status": "succeeded",
        "source_commit": COMMIT,
        "contract_sha256": CONTRACT,
        "request": {
            "seed": seed,
            "dose": dose,
            "continuation": continuation,
            "source_commit": COMMIT,
            "contract_sha256": CONTRACT,
        },
        "effects": effects,
        "endpoints": endpoints,
        "test_loaded": False,
    }


def _gate_b_root(root: Path) -> Path:
    _write_json(
        root / "CAUSAL_TRANSPORT_CONTRACT.json",
        {
            "schema": "causal-transport-contract/1",
            "contract_sha256": CONTRACT,
            "source_commit": COMMIT,
            "seeds": {"B": [401, 402, 403]},
            "doses": {"B": [1, 8, 64, 1250]},
            "continuations": ["clean", "noisy"],
            "primary_outcomes": list(OUTCOMES),
            "test_loaded": False,
        },
    )
    _write_json(
        root / "GATE_B_DECISION.json",
        {
            "schema": "causal-transport-stage-decision/1",
            "stage": "B",
            "decision": "NO-GO",
            "bundle_count": 24,
            "source_commit": COMMIT,
            "contract_sha256": CONTRACT,
            "checks": {"required_crossings": False},
            "test_loaded": False,
        },
    )
    for seed in (401, 402, 403):
        for dose in (1, 8, 64, 1250):
            for continuation in ("clean", "noisy"):
                bundle = f"seed{seed}-dose{dose}-{continuation}"
                _write_json(
                    root / "gate-b" / bundle / "summary.json",
                    _summary(seed, dose, continuation),
                )
    return root


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
    decision_path = root / "GATE_B_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["decision"] = "GO"
    _write_json(decision_path, decision)
    with pytest.raises(RuntimeError, match="NO-GO"):
        evidence.load_gate_b(root)

    root = _gate_b_root(tmp_path / "mismatch")
    summary_path = root / "gate-b" / "seed401-dose1-clean" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["contract_sha256"] = "c" * 64
    _write_json(summary_path, summary)
    with pytest.raises(RuntimeError, match="contract"):
        evidence.load_gate_b(root)


@pytest.mark.parametrize("failure", ["missing", "nonfinite", "test_access"])
def test_load_gate_b_fails_closed_on_invalid_bundle(
    tmp_path: Path, failure: str
) -> None:
    root = _gate_b_root(tmp_path)
    path = root / "gate-b" / "seed403-dose1250-noisy" / "summary.json"
    if failure == "missing":
        path.unlink()
    else:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if failure == "nonfinite":
            summary["effects"][OUTCOMES[0]]["parameter"] = float("nan")
        else:
            summary["test_loaded"] = True
        _write_json(path, summary)
    with pytest.raises(RuntimeError, match=r"matrix|finite|test isolation"):
        evidence.load_gate_b(root)


def test_discovery_evidence_reports_agreement_handoff_and_repair(
    tmp_path: Path,
) -> None:
    root = _gate_b_root(tmp_path)
    record = evidence.build_discovery_evidence(root)
    assert record["bundle_count"] == 24
    assert set(record["by_dose"]) == {"1", "8", "64", "1250"}
    assert record["cohort_stable_parameter_handoff"] == 1250
    assert record["by_dose"]["1250"]["continuation_agreement"] == 1.0
    repair = record["by_bundle"]["seed401-dose1250-clean"]["repair"]
    assert repair[OUTCOMES[0]] == {
        "state_repair": 2.0,
        "parameter_repair": 3.0,
    }


def test_discovery_evidence_write_is_atomic_and_source_bound(tmp_path: Path) -> None:
    root = _gate_b_root(tmp_path)
    record = evidence.build_discovery_evidence(root)
    path = evidence.write_discovery_evidence(root, record)
    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert evidence.write_discovery_evidence(root, record) == path
    changed = {**record, "consumed_sha256": {"different": "d" * 64}}
    with pytest.raises(RuntimeError, match="different source"):
        evidence.write_discovery_evidence(root, changed)
