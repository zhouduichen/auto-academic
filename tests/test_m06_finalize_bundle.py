from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import experiments.m0_optimizer_state.m0_run as m0
from experiments.m0_optimizer_state.finalize_bundle import finalize_bundle


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
        return SimpleNamespace(logits=self.classifier(pixel_values.mean(dim=(2, 3))))


def _write_tiny_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(m0, "CIFAR100", lambda **_: FakeCIFAR100())
    monkeypatch.setattr(m0, "build_model", lambda _: TinyVisionModel())
    output_dir = tmp_path / "bundle"
    m0.run(
        m0.RunConfig(
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
        ),
        tmp_path / "data",
        output_dir,
    )
    (output_dir / "stdout.log").write_text("completed\n", encoding="utf-8")
    return output_dir


def test_finalize_bundle_adds_stdout_to_verified_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = _write_tiny_bundle(tmp_path, monkeypatch)

    finalize_bundle(output_dir)

    manifest = json.loads((output_dir / "sha256_manifest.json").read_text(encoding="utf-8"))
    assert "stdout.log" in {item["filename"] for item in manifest}


def test_finalize_bundle_rejects_incomplete_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = _write_tiny_bundle(tmp_path, monkeypatch)
    trajectory = output_dir / "trajectory_metrics.jsonl"
    lines = trajectory.read_text(encoding="utf-8").splitlines()
    trajectory.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        finalize_bundle(output_dir)


def test_finalize_bundle_rejects_empty_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = _write_tiny_bundle(tmp_path, monkeypatch)
    (output_dir / "stdout.log").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=r"stdout\.log is empty"):
        finalize_bundle(output_dir)
