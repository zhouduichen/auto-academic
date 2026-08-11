from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import m11_candidate_sentinel as sentinel
from experiments import m11_rescue as rescue


def _passing_record() -> dict[str, object]:
    return {
        "clean_difference_pp": -1.0,
        "noisy_difference_pp": 0.2,
        "noisy_rejected_steps": 10,
        "post_warmup_steps": 2500,
        "max_consecutive_rejections": 1,
        "wallclock_ratios": [1.02, 1.03],
        "vram_ratios": [1.0, 1.0],
        "test_loaded": False,
    }


def test_rescue_decision_requires_every_gate() -> None:
    record = _passing_record()
    result = rescue.evaluate_rescue(record)
    assert result["decision"] == "GO"
    assert all(result["checks"].values())
    assert record == _passing_record()

    failures = {
        "clean_difference_pp": -2.01,
        "noisy_difference_pp": -0.01,
        "noisy_rejected_steps": 0,
        "max_consecutive_rejections": 2,
        "wallclock_ratios": [1.0, 1.051],
        "vram_ratios": [1.051, 1.0],
        "test_loaded": True,
    }
    for key, value in failures.items():
        failed = _passing_record()
        failed[key] = value
        assert rescue.evaluate_rescue(failed)["decision"] == "NO-GO"


@pytest.mark.parametrize("post_warmup_steps", [0, -1])
def test_rescue_decision_fails_closed_on_invalid_denominator(
    post_warmup_steps: int,
) -> None:
    record = _passing_record()
    record["post_warmup_steps"] = post_warmup_steps
    result = rescue.evaluate_rescue(record)
    assert result["decision"] == "NO-GO"
    assert result["checks"]["noisy_rejection_rate_at_most_5pct"] is False


def _write_sentinel(path: Path, commit: str, **overrides: object) -> None:
    value: dict[str, object] = {
        "status": "passed",
        "source_commit": commit,
        "functional": {
            "disabled_protection_parity": True,
            "isolated_rejection": True,
            "resume": True,
            "deterministic_replay": True,
            "nonfinite_rejected_atomically": True,
            "passed": True,
        },
        "time_ratio": 1.02,
        "vram_ratio": 1.01,
        "test_loaded": False,
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_sentinel_validation_is_provenance_bound_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "M11_SENTINEL_PASS.json"
    _write_sentinel(path, "a" * 40)
    assert rescue.validate_m11_sentinel(path, "a" * 40)["status"] == "passed"

    _write_sentinel(path, "b" * 40)
    with pytest.raises(RuntimeError, match="source commit"):
        rescue.validate_m11_sentinel(path, "a" * 40)
    _write_sentinel(path, "a" * 40, time_ratio=1.051)
    with pytest.raises(RuntimeError, match="resource"):
        rescue.validate_m11_sentinel(path, "a" * 40)
    _write_sentinel(path, "a" * 40, functional={"passed": False})
    with pytest.raises(RuntimeError, match="functional"):
        rescue.validate_m11_sentinel(path, "a" * 40)


def test_sentinel_functional_checks_cover_the_candidate_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sentinel, "_disabled_protection_parity", lambda *_: True)
    result = sentinel._functional_checks(torch.zeros(1), torch.zeros(1), warmup_steps=2)
    assert result == {
        "disabled_protection_parity": True,
        "isolated_rejection": True,
        "resume": True,
        "deterministic_replay": True,
        "nonfinite_rejected_atomically": True,
        "passed": True,
    }


def _write_metrics(path: Path, *, accuracy: float, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "epoch": epoch,
            "optimizer_steps": epoch * 1250,
            "tuning_accuracy": accuracy - (3 - epoch) * 0.01,
            "elapsed_training_seconds": seconds * epoch / 3,
            "test_loaded": False,
        }
        for epoch in (1, 2, 3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_cell(
    cell: Path,
    *,
    method: str,
    condition: str,
    accuracy: float,
    seconds: float,
    peak_vram_gb: float,
    commit: str,
    epochs: int,
) -> None:
    _write_metrics(cell / "epoch_metrics.jsonl", accuracy=accuracy, seconds=seconds)
    (cell / "summary.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "condition": condition,
                "seed": 301,
                "epochs": epochs,
                "optimizer_steps": 3750 if epochs == 3 else epochs * 1250,
                "peak_vram_gb": peak_vram_gb,
                "source_commit": commit,
                "method": method,
                "test_loaded": False,
            }
        ),
        encoding="utf-8",
    )


def _rescue_args(tmp_path: Path) -> SimpleNamespace:
    sentinel_path = tmp_path / "M11_SENTINEL_PASS.json"
    _write_sentinel(sentinel_path, "c" * 40)
    baseline = tmp_path / "baseline"
    _write_cell(
        baseline / "adamw" / "clean-seed301",
        method="adamw",
        condition="clean",
        accuracy=0.70,
        seconds=100.0,
        peak_vram_gb=4.0,
        commit="b" * 40,
        epochs=20,
    )
    _write_cell(
        baseline / "adamw" / "noisy-seed301",
        method="adamw",
        condition="noisy",
        accuracy=0.60,
        seconds=100.0,
        peak_vram_gb=4.0,
        commit="b" * 40,
        epochs=20,
    )
    _write_cell(
        baseline / "cadam" / "noisy-seed301",
        method="cadam",
        condition="noisy",
        accuracy=0.61,
        seconds=101.0,
        peak_vram_gb=4.0,
        commit="b" * 40,
        epochs=20,
    )
    return SimpleNamespace(
        data_dir=tmp_path / "data",
        image_store=tmp_path / "images",
        tuning_store=tmp_path / "tuning",
        bundle_301=tmp_path / "bundle",
        sentinel=sentinel_path,
        baseline_root=baseline,
        output=tmp_path / "rescue",
    )


def test_orchestrator_runs_exactly_two_cells_and_atomically_decides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _rescue_args(tmp_path)
    calls: list[tuple[str, object, tuple[Path, ...]]] = []

    def fake_clean_run(config: object, data_dir: Path, output: Path) -> dict[str, object]:
        calls.append(("clean", config, (data_dir, output)))
        _write_cell(
            output,
            method="protect-m11",
            condition="clean",
            accuracy=0.69,
            seconds=102.0,
            peak_vram_gb=4.0,
            commit="c" * 40,
            epochs=3,
        )
        return {"status": "succeeded"}

    def fake_noisy_run(
        config: object,
        bundle: Path,
        image_store: Path,
        tuning_store: Path,
        output: Path,
    ) -> dict[str, object]:
        calls.append(("noisy", config, (bundle, image_store, tuning_store, output)))
        _write_cell(
            output,
            method="protect-m11",
            condition="noisy",
            accuracy=0.612,
            seconds=103.0,
            peak_vram_gb=4.0,
            commit="c" * 40,
            epochs=3,
        )
        diagnostics = [
            {
                "epoch": 1,
                "successful_steps": 1250,
                "rejected_steps": 0,
                "transitions_off": 0,
                "transitions_on": 0,
                "previous_rejected": False,
            },
            {
                "epoch": 2,
                "successful_steps": 1250,
                "rejected_steps": 4,
                "transitions_off": 4,
                "transitions_on": 4,
                "previous_rejected": False,
            },
            {
                "epoch": 3,
                "successful_steps": 1250,
                "rejected_steps": 6,
                "transitions_off": 6,
                "transitions_on": 6,
                "previous_rejected": False,
            },
        ]
        (output / "optimizer_diagnostics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in diagnostics), encoding="utf-8"
        )
        return {"status": "succeeded"}

    monkeypatch.setattr(rescue, "source_commit", lambda: "c" * 40)
    monkeypatch.setattr(rescue.clean, "run", fake_clean_run)
    monkeypatch.setattr(rescue.noisy, "run", fake_noisy_run)

    result = rescue.run_rescue(args)

    assert [call[0] for call in calls] == ["clean", "noisy"]
    assert calls[0][1] is calls[1][1]
    config = calls[0][1]
    assert (config.seed, config.augmentation_seed, config.epochs) == (301, 10_301, 3)
    assert (config.method, config.workers, config.device) == ("protect-m11", 4, "cuda")
    assert result["decision"] == "GO"
    assert result["source_commit"] == "c" * 40
    assert result["baseline_source_commits"] == ["b" * 40]
    assert result["noisy_rejected_steps"] == 10
    assert result["post_warmup_steps"] == 2500
    decision = args.output / "M11_RESCUE_DECISION.json"
    assert json.loads(decision.read_text(encoding="utf-8"))["decision"] == "GO"
    assert not decision.with_suffix(".json.tmp").exists()


def test_orchestrator_rejects_candidate_from_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _rescue_args(tmp_path)

    def fake_run(config: object, *paths: Path) -> dict[str, object]:
        output = paths[-1]
        condition = "clean" if len(paths) == 2 else "noisy"
        _write_cell(
            output,
            method="protect-m11",
            condition=condition,
            accuracy=0.7,
            seconds=100.0,
            peak_vram_gb=4.0,
            commit="d" * 40,
            epochs=3,
        )
        if condition == "noisy":
            (output / "optimizer_diagnostics.jsonl").write_text(
                json.dumps({"epoch": 1, "successful_steps": 1250, "rejected_steps": 0})
                + "\n",
                encoding="utf-8",
            )
        return {"status": "succeeded"}

    monkeypatch.setattr(rescue, "source_commit", lambda: "c" * 40)
    monkeypatch.setattr(rescue.clean, "run", fake_run)
    monkeypatch.setattr(rescue.noisy, "run", fake_run)
    with pytest.raises(RuntimeError, match="source commit"):
        rescue.run_rescue(args)
    assert not (args.output / "M11_RESCUE_DECISION.json").exists()


def test_failed_sentinel_removes_stale_pass_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "sentinel"
    output.mkdir()
    stale = output / "M11_SENTINEL_PASS.json"
    stale.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sentinel, "_fixed_batch", lambda *_: (torch.zeros(1), torch.zeros(1)))
    monkeypatch.setattr(
        sentinel, "_functional_checks", lambda *_, **__: {"passed": False}
    )
    monkeypatch.setattr(
        sentinel,
        "_time_method",
        lambda method, *_: {
            "median_step_seconds": 1.0 if method == "adamw" else 1.01,
            "peak_vram_gb": 1.0,
        },
    )
    monkeypatch.setattr(sentinel, "source_commit", lambda: "c" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        ["m11_candidate_sentinel", "--data-dir", str(tmp_path), "--output", str(output)],
    )
    with pytest.raises(SystemExit) as raised:
        sentinel.main()
    assert raised.value.code == 2
    assert not stale.exists()
    result = json.loads((output / "M11_SENTINEL_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["test_loaded"] is False
