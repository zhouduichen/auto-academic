from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import cifar10n_transport_data as data


def test_cifar10_split_is_exact_disjoint_and_deterministic() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 5_000)
    left = data.cifar10_split(labels)
    right = data.cifar10_split(labels)
    assert {name: len(ids) for name, ids in left.items()} == {
        "train": 40_000,
        "tuning": 2_500,
        "development": 2_500,
        "confirmation": 5_000,
    }
    assert all(np.array_equal(left[name], right[name]) for name in left)
    combined = np.concatenate(list(left.values()))
    assert len(np.unique(combined)) == 50_000
    for class_id in range(10):
        assert np.sum(labels[left["train"]] == class_id) == 4_000
        assert np.sum(labels[left["tuning"]] == class_id) == 250


@pytest.mark.parametrize(
    "labels",
    [
        np.zeros(49_999, dtype=np.int64),
        np.zeros(50_000, dtype=np.float32),
        np.full(50_000, 10, dtype=np.int64),
        np.zeros(50_000, dtype=np.int64),
    ],
)
def test_cifar10_split_rejects_malformed_labels(labels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        data.cifar10_split(labels)


def _write_store(path: Path, files: dict[str, np.ndarray], schema: str) -> None:
    path.mkdir()
    records = {}
    for name, value in files.items():
        np.save(path / name, value, allow_pickle=False)
        target = path / f"{name}.npy"
        records[target.name] = {
            "bytes": target.stat().st_size,
            "sha256": data._sha(target),
        }
    manifest = {
        "schema": schema,
        "dataset": "cifar10n",
        "label_set": "worse_label",
        **data.ACCESS_FLAGS,
        "files": records,
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_validate_cifar10n_detects_corruption_and_access_flags(tmp_path: Path) -> None:
    ids = np.asarray(
        [f"cifar10-train-{index:05d}".encode() for index in range(40_000)], dtype="S19"
    )
    _write_store(
        tmp_path / "training",
        {"sample_ids": ids, "noisy_labels": np.zeros(40_000, dtype=np.int64)},
        "cifar10n-training/1",
    )
    _write_store(
        tmp_path / "image-store",
        {
            "sample_ids": ids,
            "clean_labels": np.zeros(40_000, dtype=np.int64),
            "images": np.zeros((40_000, 1, 1, 3), dtype=np.uint8),
            "train_order": np.arange(40_000),
        },
        "cifar10n-image-store/1",
    )
    _write_store(
        tmp_path / "tuning-store",
        {
            "images": np.zeros((2_500, 1, 1, 3), dtype=np.uint8),
            "clean_labels": np.zeros(2_500, dtype=np.int64),
        },
        "cifar10n-tuning-store/1",
    )
    (tmp_path / "SPLIT_MANIFEST.json").write_text(
        json.dumps({**data.ACCESS_FLAGS}), encoding="utf-8"
    )
    assert data.validate_cifar10n(tmp_path)["test_loaded"] is False
    with (tmp_path / "training" / "noisy_labels.npy").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(RuntimeError, match="sealed file mismatch"):
        data.validate_cifar10n(tmp_path)


def test_validate_cifar10n_rejects_loaded_partition(tmp_path: Path) -> None:
    path = tmp_path / "training"
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "cifar10n",
                "label_set": "worse_label",
                **{**data.ACCESS_FLAGS, "test_loaded": True},
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="access isolation"):
        data._validate_manifest(path)
