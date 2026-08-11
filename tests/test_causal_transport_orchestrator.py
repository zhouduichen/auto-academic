from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import causal_transport_orchestrator as orchestrator


def _summary(commit: str, carrier: str, contract: str = "a" * 64) -> dict[str, object]:
    effects = {
        "state": {"parameter": 1.0, "state": 4.0, "interaction": 0.1},
        "parameter": {"parameter": 4.0, "state": 1.0, "interaction": 0.1},
        "reverse": {"parameter": 4.0, "state": 1.0, "interaction": 5.0},
    }[carrier]
    return {
        "status": "succeeded",
        "source_commit": commit,
        "contract_sha256": contract,
        "effects": {
            "clean_loss_excess_auc_128": dict(effects),
            "tuning_loss_h512": dict(effects),
        },
        "wall_clock_seconds": 1.0,
        "test_loaded": False,
    }


def test_gate_a_matrix_contains_only_local_and_legacy_jobs(tmp_path: Path) -> None:
    jobs = orchestrator.build_gate_a_matrix(tmp_path, tmp_path / "legacy")
    assert len(jobs) == 8
    assert {(job.kind, job.seed, job.dose) for job in jobs} == {
        ("bundle", 301, 1),
        ("bundle", 302, 1),
        ("legacy-replay", 301, 1250),
        ("legacy-replay", 302, 1250),
    }
    assert {job.continuation for job in jobs} == {"clean", "noisy"}
    assert all("capture" not in str(job.output) for job in jobs)


def test_gate_b_matrix_is_exact_and_ordered(tmp_path: Path) -> None:
    jobs = orchestrator.build_gate_b_matrix(tmp_path)
    assert len(jobs) == 24
    assert [(job.seed, job.dose, job.continuation) for job in jobs] == [
        (seed, dose, continuation)
        for seed in (401, 402, 403)
        for dose in (1, 8, 64, 1250)
        for continuation in ("clean", "noisy")
    ]


def test_combine_continuations_is_conservative() -> None:
    combined = orchestrator.combine_continuations(
        _summary("1" * 40, "state"), _summary("1" * 40, "parameter")
    )
    assert combined["effects"]["tuning_loss_h512"]["interaction"] == 1.0
    assert combined["continuations_agree"] is False


def test_collect_gate_b_requires_all_24_jobs(tmp_path: Path) -> None:
    jobs = orchestrator.build_gate_b_matrix(tmp_path)
    summaries = {
        job.key: _summary("1" * 40, "state" if job.dose == 1 else "parameter")
        for job in jobs[:-1]
    }
    with pytest.raises(RuntimeError, match="24"):
        orchestrator.collect_gate_b(jobs, summaries, contract_sha256="a" * 64)


def test_collect_gate_b_passes_two_of_three_crossings(tmp_path: Path) -> None:
    jobs = orchestrator.build_gate_b_matrix(tmp_path)
    summaries: dict[str, dict[str, object]] = {}
    for job in jobs:
        carrier = "state" if job.dose == 1 else "parameter"
        if job.seed == 403 and job.dose == 1:
            carrier = "reverse"
        summaries[job.key] = _summary("1" * 40, carrier)
    decision = orchestrator.collect_gate_b(
        jobs, summaries, contract_sha256="a" * 64
    )
    assert decision["decision"] == "GO"
    assert decision["crossing_seeds"] == [401, 402]
    assert decision["bundle_count"] == 24


def test_contract_refuses_overwrite(tmp_path: Path) -> None:
    first = orchestrator.write_contract(tmp_path, {"source_commit": "1" * 40})
    assert orchestrator.write_contract(
        tmp_path, {"source_commit": "1" * 40}
    ) == first
    with pytest.raises(RuntimeError, match="contract"):
        orchestrator.write_contract(tmp_path, {"source_commit": "2" * 40})
    stored = json.loads((tmp_path / "CAUSAL_TRANSPORT_CONTRACT.json").read_text())
    assert stored == first
