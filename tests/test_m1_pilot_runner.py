from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.m1_calibration_clean import (
    PILOT_METHODS,
    RUNNER_METHODS,
    Config,
    _build_optimizer,
    _optimizer_epoch_diagnostic,
    _write_optimizer_diagnostic,
)
from experiments.m1_noisy_data import PILOT_NOISE_SEEDS
from experiments.m1_optimizers import CAdamW, ProtectAdamW, ProtectM11AdamW
from experiments.m1_pilot import build_matrix, evaluate_candidate, validate_sentinel


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


@pytest.mark.parametrize("method", PILOT_METHODS)
def test_build_optimizer_uses_frozen_method(method: str) -> None:
    model = TinyModel()
    config = Config(
        seed=301,
        augmentation_seed=10_301,
        epochs=20,
        learning_rate=3e-4,
        weight_decay=0.1,
        method=method,
        device="cpu",
    )
    optimizer = _build_optimizer(model, config)
    if method == "adamw":
        assert type(optimizer) is torch.optim.AdamW
    elif method == "cadam":
        assert isinstance(optimizer, CAdamW)
    else:
        assert isinstance(optimizer, ProtectAdamW)
        assert optimizer.protection_target == method.removeprefix("protect-")
        assert (optimizer.alpha_fast, optimizer.alpha_slow, optimizer.tau) == (0.9, 0.99, 0.1)


def test_m11_is_runner_only_and_does_not_expand_frozen_pilot() -> None:
    assert "protect-m11" in RUNNER_METHODS
    assert "protect-m11" not in PILOT_METHODS
    assert len(build_matrix()) == 24
    config = Config(
        seed=301,
        augmentation_seed=10_301,
        epochs=3,
        method="protect-m11",
        device="cpu",
    )
    optimizer = _build_optimizer(TinyModel(), config)
    assert isinstance(optimizer, ProtectM11AdamW)
    assert (optimizer.alpha, optimizer.threshold, optimizer.warmup_steps) == (
        0.99,
        3.0,
        1250,
    )


def test_invalid_method_is_rejected() -> None:
    config = Config(seed=301, augmentation_seed=10_301, method="unknown", device="cpu")
    with pytest.raises(ValueError, match="unsupported M1 method"):
        _build_optimizer(TinyModel(), config)


def test_pilot_seed_contract_is_frozen() -> None:
    assert PILOT_NOISE_SEEDS == (1301, 1302, 1303)


def test_optimizer_diagnostics_are_jsonl(tmp_path: Path) -> None:
    optimizer = CAdamW(TinyModel().parameters())
    optimizer.last_diagnostics = {"alignment_ratio": 0.75, "active_parameter_count": 2}
    path = tmp_path / "optimizer_diagnostics.jsonl"
    _write_optimizer_diagnostic(optimizer, path, step=7)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "active_parameter_count": 2,
        "alignment_ratio": 0.75,
        "step": 7,
    }


def test_protect_diagnostics_are_deferred_to_epoch_boundary() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = ProtectAdamW([parameter], protection_target="m")
    for gradient in (0.2, -0.1, 0.3):
        parameter.grad = torch.tensor([gradient])
        optimizer.step()
    row = _optimizer_epoch_diagnostic(optimizer, epoch=1, optimizer_steps=3)
    assert row is not None
    assert row["epoch"] == 1
    assert row["optimizer_steps"] == 3
    assert row["successful_steps"] == 3
    assert optimizer.diagnostics_summary()["successful_steps"] == 0


def test_matrix_is_exactly_24_cells() -> None:
    matrix = build_matrix()
    assert len(matrix) == 24
    assert {(cell.method, cell.condition, cell.seed) for cell in matrix} == {
        (method, condition, seed)
        for method in PILOT_METHODS
        for condition in ("clean", "noisy")
        for seed in (301, 302, 303)
    }


def _candidate_rows(noisy_deltas: tuple[float, float, float]) -> dict[str, object]:
    return {
        "candidate": "protect-m",
        "noisy_differences_pp": list(noisy_deltas),
        "clean_differences_pp": [-0.1, -0.2, 0.0],
        "fixed_wallclock_noisy_differences_pp": [0.8, 0.9, 1.0],
        "wallclock_ratios": [1.01] * 6,
        "vram_ratios": [1.0] * 6,
    }


def test_candidate_gate_requires_three_positive_one_point_mean() -> None:
    assert evaluate_candidate(_candidate_rows((1.1, 1.2, 1.0)))["eligible"] is True
    assert evaluate_candidate(_candidate_rows((1.5, 1.5, 0.0)))["eligible"] is False
    assert evaluate_candidate(_candidate_rows((0.8, 0.9, 1.0)))["eligible"] is False


def test_candidate_gate_enforces_clean_and_efficiency() -> None:
    clean_failure = _candidate_rows((1.1, 1.2, 1.0))
    clean_failure["clean_differences_pp"] = [-1.1, 0.0, 0.0]
    assert evaluate_candidate(clean_failure)["eligible"] is False
    timing_failure = _candidate_rows((1.1, 1.2, 1.0))
    timing_failure["wallclock_ratios"] = [1.06] * 6
    assert evaluate_candidate(timing_failure)["eligible"] is False


def test_dispatch_requires_matching_passed_sentinel(tmp_path: Path) -> None:
    path = tmp_path / "sentinel.json"
    with pytest.raises(RuntimeError, match="sentinel"):
        validate_sentinel(path, source_commit="a" * 40)
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "source_commit": "b" * 40,
                "test_loaded": False,
                "candidate_ratios": {"protect-m": 1.01, "protect-mv": 1.02},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source"):
        validate_sentinel(path, source_commit="a" * 40)
