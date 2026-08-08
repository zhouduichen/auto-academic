from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.m1_calibration_clean import (
    PILOT_METHODS,
    Config,
    _build_optimizer,
    _write_optimizer_diagnostic,
)
from experiments.m1_noisy_data import PILOT_NOISE_SEEDS
from experiments.m1_optimizers import CAdamW, ProtectAdamW


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
