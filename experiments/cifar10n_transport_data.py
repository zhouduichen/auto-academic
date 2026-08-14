"""Seal CIFAR-10N paired clean/human-noisy training inputs without test access."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from experiments.m1_pilot import source_commit

DATASET = "cifar10n"
LABEL_SET = "worse_label"
SPLIT_SEED = 20_260_801
OFFICIAL_SOURCE = "https://github.com/UCSC-REAL/cifar-10-100n"
ACCESS_FLAGS = {
    "test_loaded": False,
    "development_loaded": False,
    "confirmation_loaded": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def cifar10_split(
    clean_labels: np.ndarray, split_seed: int = SPLIT_SEED
) -> dict[str, np.ndarray]:
    labels = np.asarray(clean_labels)
    if labels.shape != (50_000,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("CIFAR-10 clean labels must be an integer vector of length 50000")
    if labels.min(initial=0) < 0 or labels.max(initial=0) > 9:
        raise ValueError("CIFAR-10 clean labels are out of range")
    rng = np.random.default_rng(split_seed)
    result: dict[str, list[int]] = {
        "train": [],
        "tuning": [],
        "development": [],
        "confirmation": [],
    }
    for class_id in range(10):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) != 5_000:
            raise ValueError(f"CIFAR-10 class {class_id} must contain exactly 5000 images")
        rng.shuffle(indices)
        result["train"].extend(indices[:4_000].tolist())
        result["tuning"].extend(indices[4_000:4_250].tolist())
        result["development"].extend(indices[4_250:4_500].tolist())
        result["confirmation"].extend(indices[4_500:].tolist())
    frozen: dict[str, np.ndarray] = {}
    for name, indices in result.items():
        values = np.asarray(indices, dtype=np.int64)
        rng.shuffle(values)
        frozen[name] = values
    expected = {"train": 40_000, "tuning": 2_500, "development": 2_500, "confirmation": 5_000}
    if {name: len(values) for name, values in frozen.items()} != expected:
        raise RuntimeError("CIFAR-10 split size invariant failed")
    combined = np.concatenate(list(frozen.values()))
    if len(np.unique(combined)) != 50_000 or not np.array_equal(
        np.sort(combined), np.arange(50_000)
    ):
        raise RuntimeError("CIFAR-10 split isolation invariant failed")
    return frozen


def _manifest(root: Path, metadata: dict[str, object]) -> dict[str, object]:
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    record = {**metadata, "files": files}
    _atomic_json(root / "manifest.json", record)
    return record


def _stable_ids(indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"cifar10-train-{int(index):05d}".encode("ascii") for index in indices],
        dtype="S19",
    )


def seal_cifar10n(
    cifar_root: Path,
    label_file: Path,
    output: Path,
    *,
    expected_label_sha256: str,
) -> dict[str, object]:
    if len(expected_label_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_label_sha256
    ):
        raise ValueError("expected_label_sha256 must be lowercase SHA-256")
    actual_label_sha256 = _sha(label_file)
    if actual_label_sha256 != expected_label_sha256:
        raise RuntimeError("CIFAR-10N label-file checksum mismatch")
    raw = torch.load(label_file, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or LABEL_SET not in raw:
        raise RuntimeError("official CIFAR-10N worse_label is unavailable")
    noisy = np.asarray(raw[LABEL_SET])
    if noisy.shape != (50_000,) or not np.issubdtype(noisy.dtype, np.integer):
        raise RuntimeError("CIFAR-10N worse_label must be an integer vector of length 50000")
    noisy = noisy.astype(np.int64, copy=False)
    if noisy.min(initial=0) < 0 or noisy.max(initial=0) > 9:
        raise RuntimeError("CIFAR-10N worse_label is out of range")

    dataset = CIFAR10(root=cifar_root, train=True, download=False, transform=None)
    clean = np.asarray(dataset.targets, dtype=np.int64)
    images = np.asarray(dataset.data)
    if images.shape != (50_000, 32, 32, 3) or images.dtype != np.uint8:
        raise RuntimeError("CIFAR-10 training image payload is malformed")
    split = cifar10_split(clean)
    if output.exists():
        raise FileExistsError("CIFAR-10N sealed output already exists")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    common = {
        "dataset": DATASET,
        "label_set": LABEL_SET,
        "split_seed": SPLIT_SEED,
        "official_source": OFFICIAL_SOURCE,
        "label_file_sha256": actual_label_sha256,
        "clean_labels_sha256": _array_sha(clean),
        "image_id_order_sha256": _array_sha(np.arange(50_000, dtype=np.int64)),
        "source_commit": source_commit(),
        **ACCESS_FLAGS,
    }
    train_ids = split["train"]
    tuning_ids = split["tuning"]
    training = temporary / "training"
    image_store = temporary / "image-store"
    tuning_store = temporary / "tuning-store"
    for path in (training, image_store, tuning_store):
        path.mkdir()

    stable_train_ids = _stable_ids(train_ids)
    np.save(training / "sample_ids.npy", stable_train_ids, allow_pickle=False)
    np.save(training / "noisy_labels.npy", noisy[train_ids], allow_pickle=False)
    np.save(image_store / "sample_ids.npy", stable_train_ids, allow_pickle=False)
    np.save(image_store / "images.npy", images[train_ids], allow_pickle=False)
    np.save(image_store / "clean_labels.npy", clean[train_ids], allow_pickle=False)
    np.save(image_store / "train_order.npy", train_ids, allow_pickle=False)
    np.save(tuning_store / "images.npy", images[tuning_ids], allow_pickle=False)
    np.save(tuning_store / "clean_labels.npy", clean[tuning_ids], allow_pickle=False)
    _manifest(training, {**common, "schema": "cifar10n-training/1", "sample_count": 40_000})
    _manifest(image_store, {**common, "schema": "cifar10n-image-store/1", "sample_count": 40_000})
    _manifest(tuning_store, {**common, "schema": "cifar10n-tuning-store/1", "sample_count": 2_500})
    _atomic_json(
        temporary / "SPLIT_MANIFEST.json",
        {
            **common,
            "schema": "cifar10n-split/1",
            "counts": {name: len(values) for name, values in split.items()},
            "index_sha256": {name: _array_sha(values) for name, values in split.items()},
        },
    )
    os.replace(temporary, output)
    return validate_cifar10n(output)


def _validate_manifest(path: Path) -> dict[str, object]:
    try:
        record = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("CIFAR-10N manifest cannot be loaded") from error
    if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
        raise RuntimeError("CIFAR-10N manifest is malformed")
    for flag, expected in ACCESS_FLAGS.items():
        if record.get(flag) is not expected:
            raise RuntimeError("CIFAR-10N access isolation failed")
    for name, raw in record["files"].items():
        file_path = (path / str(name)).resolve()
        if file_path.parent != path.resolve() or not isinstance(raw, dict):
            raise RuntimeError("CIFAR-10N manifest path is invalid")
        if (
            not file_path.is_file()
            or file_path.stat().st_size != raw.get("bytes")
            or _sha(file_path) != raw.get("sha256")
        ):
            raise RuntimeError("CIFAR-10N sealed file mismatch")
    return record


def validate_cifar10n(root: Path) -> dict[str, object]:
    training = root / "training"
    image_store = root / "image-store"
    tuning_store = root / "tuning-store"
    records = {
        "training": _validate_manifest(training),
        "image_store": _validate_manifest(image_store),
        "tuning_store": _validate_manifest(tuning_store),
    }
    if any(
        record.get("dataset") != DATASET or record.get("label_set") != LABEL_SET
        for record in records.values()
    ):
        raise RuntimeError("CIFAR-10N store identity mismatch")
    training_ids = np.load(training / "sample_ids.npy", allow_pickle=False)
    image_ids = np.load(image_store / "sample_ids.npy", allow_pickle=False)
    if not np.array_equal(training_ids, image_ids) or len(training_ids) != 40_000:
        raise RuntimeError("CIFAR-10N public IDs mismatch")
    if len(np.load(training / "noisy_labels.npy", allow_pickle=False)) != 40_000:
        raise RuntimeError("CIFAR-10N noisy-label length mismatch")
    if len(np.load(image_store / "clean_labels.npy", allow_pickle=False)) != 40_000:
        raise RuntimeError("CIFAR-10N clean-label length mismatch")
    split = json.loads((root / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    if any(split.get(flag) is not expected for flag, expected in ACCESS_FLAGS.items()):
        raise RuntimeError("CIFAR-10N split isolation failed")
    return {
        "dataset": DATASET,
        "label_set": LABEL_SET,
        "training": training,
        "image_store": image_store,
        "tuning_store": tuning_store,
        "training_manifest_sha256": _sha(training / "manifest.json"),
        "image_store_manifest_sha256": _sha(image_store / "manifest.json"),
        "tuning_store_manifest_sha256": _sha(tuning_store / "manifest.json"),
        "split_manifest_sha256": _sha(root / "SPLIT_MANIFEST.json"),
        **ACCESS_FLAGS,
    }
