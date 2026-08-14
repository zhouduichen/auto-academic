from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments import repair_handoff_models as models
from experiments.m1_calibration_clean import Config


class _FakeResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


def _config(model_id: str) -> Config:
    return Config(seed=601, augmentation_seed=10_601, model_id=model_id)


def _allowed_loading() -> dict[str, object]:
    return {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [
            ("classifier.1.weight", (1000, 512), (100, 512)),
            ("classifier.1.bias", (1000,), (100,)),
        ],
        "error_msgs": [],
    }


def test_default_transport_model_preserves_vit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = nn.Linear(1, 1)
    monkeypatch.setattr(models.clean, "_build_model", lambda config: expected)
    assert models.build_transport_model(_config(models.VIT_MODEL_ID)) is expected


def test_resnet_load_is_pinned_and_allows_only_replaced_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _FakeResNet()
    captured: dict[str, object] = {}

    def fake_load(*args: object, **kwargs: object) -> tuple[nn.Module, dict[str, object]]:
        captured.update(kwargs)
        return expected, _allowed_loading()

    monkeypatch.setattr(
        models.ResNetForImageClassification, "from_pretrained", fake_load
    )
    actual = models.build_transport_model(_config(models.RESNET_MODEL_ID))
    assert actual is expected
    assert captured == {
        "revision": models.RESNET_REVISION,
        "num_labels": 100,
        "ignore_mismatched_sizes": True,
        "output_loading_info": True,
        "use_safetensors": True,
        "local_files_only": True,
    }
    assert all(parameter.requires_grad for parameter in actual.parameters())


def test_resnet_accepts_ten_class_confirmation_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load(*args: object, **kwargs: object) -> tuple[nn.Module, dict[str, object]]:
        captured.update(kwargs)
        loading = _allowed_loading()
        loading["mismatched_keys"] = [
            ("classifier.1.weight", (1000, 512), (10, 512)),
            ("classifier.1.bias", (1000,), (10,)),
        ]
        return _FakeResNet(), loading

    monkeypatch.setattr(models.ResNetForImageClassification, "from_pretrained", fake_load)
    models.build_transport_model(_config(models.RESNET_MODEL_ID), num_labels=10)
    assert captured["num_labels"] == 10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("missing_keys", ["resnet.encoder.stages.0.layers.0.layer.0.weight"]),
        ("unexpected_keys", ["unexpected.weight"]),
        (
            "mismatched_keys",
            [("resnet.embedder.embedder.convolution.weight", (1,), (2,))],
        ),
        ("error_msgs", ["broken checkpoint"]),
    ],
)
def test_resnet_load_rejects_every_other_anomaly(
    field: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    loading = _allowed_loading()
    loading[field] = value
    monkeypatch.setattr(
        models.ResNetForImageClassification,
        "from_pretrained",
        lambda *args, **kwargs: (_FakeResNet(), loading),
    )
    with pytest.raises(RuntimeError, match="load audit"):
        models.build_transport_model(_config(models.RESNET_MODEL_ID))


def test_transport_model_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        models.build_transport_model(_config("unknown/model"))


def test_resnet_binding_hashes_exact_local_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    weights = tmp_path / "model.safetensors"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")

    def fake_cached_file(
        model_id: str,
        filename: str,
        *,
        revision: str,
        local_files_only: bool,
    ) -> str:
        assert model_id == models.RESNET_MODEL_ID
        assert revision == models.RESNET_REVISION
        assert local_files_only is True
        return str(config if filename == "config.json" else weights)

    monkeypatch.setattr(models, "cached_file", fake_cached_file)
    assert models.resnet_artifact_binding(local_files_only=True) == {
        "model_id": models.RESNET_MODEL_ID,
        "revision": models.RESNET_REVISION,
        "config_sha256": hashlib.sha256(b"config").hexdigest(),
        "model_safetensors_sha256": hashlib.sha256(b"weights").hexdigest(),
    }
