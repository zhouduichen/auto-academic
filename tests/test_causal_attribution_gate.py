from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments import causal_attribution_gate as gate


def passing_attribution_record(dominant: str) -> dict[str, object]:
    if dominant == "parameter":
        values = {"parameter": 4.0, "state": 1.0, "interaction": 0.5}
    elif dominant == "state":
        values = {"parameter": 1.0, "state": 4.0, "interaction": 0.5}
    else:
        values = {"parameter": 1.0, "state": 1.0, "interaction": 2.0}
    return {
        "mandatory_artifacts": 12,
        "effects": {
            str(seed): {str(epoch): dict(values) for epoch in (1, 2, 3)}
            for seed in (301, 302)
        },
        "prediction": {
            "checkpoint_predictions": {str(epoch): dominant for epoch in (1, 2, 3)}
        },
        "cadam_confirmation": {"301": dominant, "302": dominant},
        "novelty_audit_passed": True,
        "gpu_hours": 2.0,
        "test_loaded": False,
    }


def test_mandatory_matrix_is_exact() -> None:
    cells = gate.build_mandatory_matrix()
    assert len(cells) == 12
    assert {(cell.seed, cell.epoch, cell.continuation) for cell in cells} == {
        (seed, epoch, continuation)
        for seed in (301, 302)
        for epoch in (1, 2, 3)
        for continuation in ("clean", "noisy")
    }


def test_gate_requires_replicated_twofold_dominance_and_prediction_match() -> None:
    record = passing_attribution_record(dominant="parameter")
    result = gate.evaluate_attribution(record, require_cadam=True)
    assert result["decision"] == "GO"
    assert all(result["checks"].values())

    for epoch in (2, 3):
        record["effects"]["302"][str(epoch)] = {
            "parameter": 0.5,
            "state": 1.0,
            "interaction": 0.2,
        }
    assert gate.evaluate_attribution(record, require_cadam=True)["decision"] == "NO-GO"


def test_gate_rejects_direction_disagreement() -> None:
    record = passing_attribution_record(dominant="state")
    for epoch in (1, 2, 3):
        record["effects"]["302"][str(epoch)]["state"] = -4.0
    assert gate.evaluate_attribution(record, require_cadam=True)["decision"] == "NO-GO"


def test_gate_rejects_nonfinite_effect() -> None:
    record = passing_attribution_record(dominant="state")
    record["effects"]["301"]["1"]["state"] = float("nan")
    result = gate.evaluate_attribution(record, require_cadam=True)
    assert result["decision"] == "NO-GO"
    assert result["checks"]["finite_and_isolated"] is False


def test_cadam_confirmation_is_required_after_adamw_go() -> None:
    record = passing_attribution_record(dominant="state")
    record["cadam_confirmation"] = {"301": "parameter", "302": "state"}
    assert gate.evaluate_attribution(record, require_cadam=True)["decision"] == "NO-GO"


def test_orchestrator_runs_exact_jobs_and_removes_stale_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    output = tmp_path / "output"
    args = argparse.Namespace(
        data_dir=tmp_path / "data",
        image_store=tmp_path / "images",
        tuning_store=tmp_path / "tuning",
        bundle_301=tmp_path / "bundle301",
        bundle_302=tmp_path / "bundle302",
        legacy_m06_root=tmp_path / "m06",
        legacy_m1_root=tmp_path / "m1",
        legacy_m11_decision=tmp_path / "m11.json",
        sentinel=tmp_path / "sentinel.json",
        output=output,
    )
    monkeypatch.setattr(gate, "source_commit", lambda: "c" * 40)

    def fake_validate(*_: object) -> dict[str, object]:
        calls.append("validate")
        return {"source_commit": "c" * 40}

    monkeypatch.setattr(gate, "validate_inputs", fake_validate)
    monkeypatch.setattr(gate, "run_capture_cell", lambda *_: calls.append("capture"))

    def fake_replay(cell: object, *_: object) -> None:
        assert isinstance(cell, gate.AttributionCell)
        calls.append("cadam-replay" if cell.method == "cadam" else "adamw-replay")

    monkeypatch.setattr(gate, "run_replay_cell", fake_replay)
    monkeypatch.setattr(
        gate,
        "collect_record",
        lambda *args, **kwargs: passing_attribution_record("parameter"),
    )
    stale = args.output / "CAUSAL_ATTRIBUTION_DECISION.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"decision":"GO"}', encoding="utf-8")
    result = gate.run_gate(args)
    assert result["decision"] == "GO"
    assert calls[0] == "validate"
    assert calls[1:5] == ["capture"] * 4
    assert calls[5:17] == ["adamw-replay"] * 12
    assert calls[17:] == ["cadam-replay"] * 4
    assert json.loads(stale.read_text(encoding="utf-8"))["source_commit"] == "c" * 40


def test_adamw_no_go_does_not_run_cadam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    args = argparse.Namespace(output=tmp_path)
    monkeypatch.setattr(gate, "source_commit", lambda: "d" * 40)
    monkeypatch.setattr(gate, "validate_inputs", lambda *_: {})
    monkeypatch.setattr(gate, "run_capture_cell", lambda *_: None)
    monkeypatch.setattr(gate, "run_replay_cell", lambda cell, *_: calls.append(cell.method))
    failing = passing_attribution_record("parameter")
    failing["novelty_audit_passed"] = False
    monkeypatch.setattr(gate, "collect_record", lambda *args, **kwargs: failing)
    assert gate.run_gate(args)["decision"] == "NO-GO"
    assert calls == ["adamw"] * 12
