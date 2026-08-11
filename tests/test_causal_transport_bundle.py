from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import causal_transport_bundle as bundle
from experiments.m0_optimizer_state import m0_run as m0
from src.arw import m0_core


def _stores(root: Path) -> tuple[Path, Path]:
    image = root / "image"
    training = root / "training"
    image.mkdir()
    training.mkdir()
    ids = np.asarray([b"cifar100-train-00000", b"cifar100-train-00001"])
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    images[0, :, :, 0] = 255
    images[1, :, :, 1] = 255
    np.save(image / "sample_ids.npy", ids, allow_pickle=False)
    np.save(image / "images.npy", images, allow_pickle=False)
    np.save(image / "train_order.npy", np.asarray([1, 0]), allow_pickle=False)
    np.save(training / "sample_ids.npy", ids, allow_pickle=False)
    np.save(training / "noisy_labels.npy", np.asarray([9, 8]), allow_pickle=False)
    return image, training


def _state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> m0_core.BranchState:
    return m0_core.BranchState(
        deepcopy(m0_core.capture_trainable_state(model)),
        deepcopy(m0_core.capture_optimizer_state(optimizer)),
        0.0,
        0.0,
    )


def test_paired_dataset_changes_only_labels(tmp_path: Path) -> None:
    image, training = _stores(tmp_path)
    data = bundle.PairedTransportDataset(
        image, training, clean_targets=np.asarray([3, 4]), augmentation_seed=10_401
    )
    data.epoch = 2
    first_image, first_clean, first_noisy = data[0]
    repeated_image, repeated_clean, repeated_noisy = data[0]
    assert torch.equal(first_image, repeated_image)
    assert (first_clean, first_noisy) == (4, 8)
    assert (repeated_clean, repeated_noisy) == (4, 8)


def test_paired_dataset_rejects_id_mismatch(tmp_path: Path) -> None:
    image, training = _stores(tmp_path)
    np.save(
        training / "sample_ids.npy",
        np.asarray([b"cifar100-train-00001", b"cifar100-train-00000"]),
        allow_pickle=False,
    )
    with pytest.raises(RuntimeError, match="IDs"):
        bundle.PairedTransportDataset(
            image, training, clean_targets=np.asarray([3, 4]), augmentation_seed=10_401
        )


def test_clean_and_noisy_exposure_share_clock_but_change_carriers() -> None:
    torch.manual_seed(4)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.05, foreach=False, fused=False
    )
    criterion = torch.nn.CrossEntropyLoss()
    images = torch.tensor([[1.0, -1.0]])
    clean_labels = torch.tensor([0])
    noisy_labels = torch.tensor([1])
    optimizer.zero_grad(set_to_none=True)
    criterion(model(images), clean_labels).backward()
    optimizer.step()
    common = _state(model, optimizer)
    common_rng = m0.capture_rng_state()
    batches = [(images, clean_labels, noisy_labels)]
    clean = bundle.run_paired_exposure(
        model,
        optimizer,
        criterion,
        common,
        common_rng,
        batches,
        noisy=False,
        device=torch.device("cpu"),
    )
    noisy = bundle.run_paired_exposure(
        model,
        optimizer,
        criterion,
        common,
        common_rng,
        batches,
        noisy=True,
        device=torch.device("cpu"),
    )
    assert clean.next_epoch == noisy.next_epoch == 0
    bundle.replay.validate_checkpoint_pair(clean, noisy)
    assert any(
        not torch.equal(clean.parameters[name], noisy.parameters[name])
        for name in clean.parameters
    )


@pytest.mark.parametrize("dose", [0, 2, 7, 65, 1249, 1251])
def test_request_rejects_unfrozen_dose(dose: int, tmp_path: Path) -> None:
    request = bundle.TransportRequest.testing(tmp_path, dose=dose)
    with pytest.raises(ValueError, match="dose"):
        bundle.validate_request(request, allow_test_values=True)


def test_snapshot_pair_is_atomic(tmp_path: Path) -> None:
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = bundle.replay.CheckpointState(
        deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), 0
    )
    bundle.write_snapshot_pair(tmp_path, state, state, contract_sha256="a" * 64)
    clean, noisy = bundle.load_snapshot_pair(tmp_path, contract_sha256="a" * 64)
    bundle.replay.validate_checkpoint_pair(clean, noisy)
    (tmp_path / "noisy-exposure.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="snapshot"):
        bundle.load_snapshot_pair(tmp_path, contract_sha256="a" * 64)
