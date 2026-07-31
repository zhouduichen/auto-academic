import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTModel

import experiments.m0_optimizer_state.m0_run as m0


class FakeCIFAR100:
    def __init__(self) -> None:
        self.targets = [class_id for class_id in range(100) for _ in range(500)]
        self.image = Image.new("RGB", (32, 32), color=(64, 128, 192))

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        return self.image, self.targets[index]


class TinyVisionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(3, 100)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        pooled = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(logits=self.classifier(pooled))


def test_small_sgd_bundle_is_reproducible_and_writes_audit_artifacts(
    tmp_path: object, monkeypatch: object
) -> None:
    monkeypatch.setattr(m0, "CIFAR100", lambda **_: FakeCIFAR100())
    monkeypatch.setattr(m0, "build_model", lambda _: TinyVisionModel())
    config = m0.RunConfig(
        optimizer="sgd",
        pulse="label_flip",
        seed=0,
        split_seed=123,
        pulse_seed=456,
        warmup_steps=1,
        replay_steps=2,
        batch_size=2,
        probe_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        model_id="unused",
        lora_rank=1,
        device="cpu",
    )
    output_dir = tmp_path / "run"

    summary = m0.run(config, tmp_path / "data", output_dir)

    assert summary["test_loaded"] is False
    assert summary["train_steps"] == 11
    assert summary["primary_endpoint"]["max_abs_clean_loss_excess"] == 0.0
    expected = {
        "run_config.json",
        "split_ids.json",
        "replay_manifest.json",
        "checkpoint_manifest.json",
        "provenance.json",
        "trajectory_metrics.jsonl",
        "summary.json",
        "sha256_manifest.json",
    }
    assert expected == {path.name for path in output_dir.iterdir()}


def test_small_m05_bundle_writes_five_state_carrier_trajectories(
    tmp_path: object, monkeypatch: object
) -> None:
    monkeypatch.setattr(m0, "CIFAR100", lambda **_: FakeCIFAR100())
    monkeypatch.setattr(m0, "build_model", lambda _: TinyVisionModel())
    config = m0.RunConfig(
        optimizer="adamw",
        pulse="label_flip",
        seed=0,
        split_seed=123,
        pulse_seed=456,
        warmup_steps=1,
        replay_steps=2,
        batch_size=2,
        probe_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        model_id="unused",
        lora_rank=1,
        device="cpu",
        state_attribution=True,
    )

    summary = m0.run(config, tmp_path / "data", tmp_path / "m05")

    assert summary["bundle_id"] == "m05-adamw-label_flip-seed0"
    assert tuple(summary["branches"]) == (
        "control",
        "m_only",
        "v_only",
        "state_both",
        "parameter_only",
    )
    assert summary["train_steps"] == 13
    assert summary["test_loaded"] is False
    assert (tmp_path / "m05" / "branch_construction_manifest.json").is_file()
    assert (tmp_path / "m05" / "provenance.json").is_file()


def test_explicit_m06_seeds_split_replay_probe_and_pulse_randomness(
    tmp_path: object, monkeypatch: object
) -> None:
    monkeypatch.setattr(m0, "CIFAR100", lambda **_: FakeCIFAR100())
    monkeypatch.setattr(m0, "build_model", lambda _: TinyVisionModel())
    plan_calls: list[tuple[int, int]] = []
    original_build_batch_plans = m0.build_batch_plans
    original_corrupt_pulse = m0.corrupt_pulse
    pulse_seeds: list[int] = []

    def capture_plans(
        train_indices: list[int], count: int, batch_size: int, seed: int
    ) -> list[m0.BatchPlan]:
        plan_calls.append((count, seed))
        return original_build_batch_plans(train_indices, count, batch_size, seed)

    def capture_pulse(
        pulse: str, images: torch.Tensor, labels: torch.Tensor, pulse_seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pulse_seeds.append(pulse_seed)
        return original_corrupt_pulse(pulse, images, labels, pulse_seed)

    monkeypatch.setattr(m0, "build_batch_plans", capture_plans)
    monkeypatch.setattr(m0, "corrupt_pulse", capture_pulse)
    config = m0.RunConfig(
        optimizer="adamw",
        pulse="label_flip",
        seed=3,
        split_seed=123,
        pulse_seed=271_828,
        warmup_steps=1,
        replay_steps=2,
        batch_size=2,
        probe_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        model_id="unused",
        lora_rank=1,
        device="cpu",
        state_attribution=True,
        replay_seed=420_003,
        probe_seed=20260806,
    )
    output_dir = tmp_path / "m06"

    summary = m0.run(config, tmp_path / "data", output_dir)

    assert plan_calls == [(2, 3), (2, 420_003)]
    assert pulse_seeds == [271_828]
    split = json.loads((output_dir / "split_ids.json").read_text(encoding="utf-8"))
    expected_probe = (
        np.random.default_rng(20260806)
        .choice(split["validation"], config.probe_size, replace=False)
        .tolist()
    )
    assert split["probe"] == expected_probe
    assert summary["bundle_id"] == "m06-adamw-label_flip-train3-pulse271828"


def test_label_flip_changes_every_label() -> None:
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 98, 99])
    _, corrupt = m0.corrupt_pulse("label_flip", images, labels, pulse_seed=17)
    assert torch.all(corrupt != labels)


def _tiny_vit_config() -> ViTConfig:
    return ViTConfig(
        image_size=32,
        patch_size=16,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=100,
    )


def _adamw_config() -> m0.RunConfig:
    return m0.RunConfig(
        optimizer="adamw",
        pulse="label_flip",
        seed=0,
        split_seed=1,
        pulse_seed=2,
        warmup_steps=1,
        replay_steps=1,
        batch_size=2,
        probe_size=32,
        learning_rate=3e-4,
        weight_decay=0.01,
        model_id="unused",
        lora_rank=2,
        device="cpu",
    )


def test_build_model_loads_audited_backbone_then_adds_classifier(
    monkeypatch: object,
) -> None:
    tiny_backbone = ViTModel(_tiny_vit_config())
    monkeypatch.setattr(
        ViTModel,
        "from_pretrained",
        lambda *_args, **_kwargs: (
            tiny_backbone,
            {
                "missing_keys": set(),
                "unexpected_keys": set(),
                "mismatched_keys": set(),
                "error_msgs": [],
            },
        ),
    )
    monkeypatch.setattr(
        m0.ViTForImageClassification,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail(
            "classifier wrapper must not load pretrained backbone weights"
        ),
    )

    model = m0.build_model(_adamw_config())
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert any("lora_" in name for name in trainable_names)
    assert any("classifier" in name for name in trainable_names)
    assert all("lora_" in name or "classifier" in name for name in trainable_names)


@pytest.mark.parametrize("field", ["missing_keys", "unexpected_keys"])
def test_build_model_fails_closed_on_backbone_load_anomaly(field: str, monkeypatch: object) -> None:
    loading_info = {
        "missing_keys": set(),
        "unexpected_keys": set(),
        "mismatched_keys": set(),
        "error_msgs": [],
    }
    loading_info[field] = {"layers.0.attention.q_proj.weight"}
    monkeypatch.setattr(
        ViTModel,
        "from_pretrained",
        lambda *_args, **_kwargs: (ViTModel(_tiny_vit_config()), loading_info),
    )
    monkeypatch.setattr(
        m0.ViTForImageClassification,
        "from_pretrained",
        lambda *_args, **_kwargs: ViTForImageClassification(_tiny_vit_config()),
    )

    with pytest.raises(RuntimeError, match=field):
        m0.build_model(_adamw_config())
