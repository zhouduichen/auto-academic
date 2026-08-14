from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments import causal_transport_bundle as bundle


def _stores(root: Path) -> tuple[Path, Path]:
    image = root / "image"
    training = root / "training"
    image.mkdir()
    training.mkdir()
    ids = np.asarray([b"cifar10-train-00010", b"cifar10-train-00020"])
    np.save(image / "sample_ids.npy", ids, allow_pickle=False)
    np.save(image / "images.npy", np.zeros((2, 32, 32, 3), dtype=np.uint8))
    np.save(image / "train_order.npy", np.asarray([20, 10]), allow_pickle=False)
    np.save(training / "sample_ids.npy", ids, allow_pickle=False)
    np.save(training / "noisy_labels.npy", np.asarray([7, 8]), allow_pickle=False)
    return image, training


def test_cifar10n_request_binds_labels_and_prefix(tmp_path: Path) -> None:
    request = bundle.TransportRequest.testing(tmp_path, dose=64)
    request = bundle.TransportRequest(
        **{
            **request.__dict__,
            "dataset_name": "cifar10n",
            "num_labels": 10,
            "diagnostic_prefix_steps": 8,
        }
    )
    bundle.validate_request(request, allow_test_values=True)
    with pytest.raises(ValueError, match="num_labels"):
        bundle.validate_request(
            bundle.TransportRequest(**{**request.__dict__, "num_labels": 100}),
            allow_test_values=True,
        )


def test_aligned_clean_targets_do_not_expose_full_dataset(tmp_path: Path) -> None:
    image, training = _stores(tmp_path)
    dataset = bundle.PairedTransportDataset(
        image,
        training,
        clean_targets=np.asarray([3, 4]),
        augmentation_seed=10_701,
        clean_targets_aligned=True,
    )
    _, clean_label, noisy_label = dataset[0]
    assert (clean_label, noisy_label) == (4, 8)


class _LogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.linear(self.bn(pixel_values)))


def test_stream_label_nll_is_normalized_finite_and_nonmutating() -> None:
    torch.manual_seed(3)
    model = _LogitModel()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    score = bundle.stream_label_nll(
        model,
        [(torch.tensor([[1.0, 2.0], [2.0, 1.0]]), torch.tensor([0, 1]))],
        torch.device("cpu"),
        num_labels=2,
    )
    assert np.isfinite(score) and score > 0.0
    assert model.training is True
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
