from typing import ClassVar

import numpy as np
import torch
from PIL import Image

import experiments.m1_calibration_clean as m1


def test_m1_split_is_exact_stratified_and_reproducible() -> None:
    labels = [class_id for class_id in range(100) for _ in range(500)]
    left = m1.m1_split(labels, 20_260_801)
    right = m1.m1_split(labels, 20_260_801)

    assert left == right
    assert {key: len(value) for key, value in left.items()} == {
        "train": 40_000,
        "tuning": 2_500,
        "development": 2_500,
        "confirmation": 5_000,
    }
    assert len(set().union(*(set(value) for value in left.values()))) == 50_000
    label_array = np.asarray(labels)
    assert np.bincount(label_array[left["train"]], minlength=100).tolist() == [400] * 100
    assert np.bincount(label_array[left["tuning"]], minlength=100).tolist() == [25] * 100


def test_optimizer_keeps_bias_out_of_weight_decay() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    optimizer = m1._build_optimizer(model, m1.Config(seed=101, augmentation_seed=10_101))
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01, 0.0]


def test_optimizer_uses_retuned_betas() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    config = m1.Config(seed=101, augmentation_seed=10_101, betas=(0.7, 0.95))
    optimizer = m1._build_optimizer(model, config)
    assert all(group["betas"] == (0.7, 0.95) for group in optimizer.param_groups)


def test_small_run_only_constructs_training_dataset(tmp_path: object, monkeypatch: object) -> None:
    calls: list[dict[str, object]] = []

    class FakeDataset:
        targets: ClassVar[list[int]] = [0, 0, 1, 1, 2, 2, 3, 3]
        image = Image.new("RGB", (32, 32), color=(64, 128, 192))

        def __getitem__(self, index: int) -> tuple[Image.Image, int]:
            return self.image, self.targets[index]

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = torch.nn.Linear(3, 4)

        def forward(self, pixel_values: torch.Tensor) -> object:
            logits = self.classifier(pixel_values.mean(dim=(2, 3)))
            return type("Output", (), {"logits": logits})()

    def fake_cifar(**kwargs: object) -> FakeDataset:
        calls.append(kwargs)
        return FakeDataset()

    monkeypatch.setattr(m1, "CIFAR100", fake_cifar)
    monkeypatch.setattr(
        m1,
        "m1_split",
        lambda *_: {
            "train": [0, 2, 4, 6],
            "tuning": [1, 3],
            "development": [5],
            "confirmation": [7],
        },
    )
    monkeypatch.setattr(m1, "_build_model", lambda _: TinyModel())
    monkeypatch.setattr(
        m1.m0,
        "build_provenance",
        lambda _: {"source_commit": "test", "source_dirty": True, "uv_lock_sha256": "test"},
    )
    config = m1.Config(
        seed=101,
        augmentation_seed=10_101,
        epochs=1,
        batch_size=2,
        workers=0,
        device="cpu",
    )

    summary = m1.run(config, tmp_path / "data", tmp_path / "output")

    assert calls == [
        {"root": tmp_path / "data", "train": True, "download": False, "transform": None}
    ]
    assert summary["test_loaded"] is False
