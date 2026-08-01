from pathlib import Path

import numpy as np
import pytest

from experiments import m1_noisy_data as noise


def test_rng_domains_and_exact_pps_are_deterministic() -> None:
    left = noise.derive_rng(1101, "base-rate/0").random(8)
    replay = noise.derive_rng(1101, "base-rate/0").random(8)
    other = noise.derive_rng(1101, "matrix/0").random(8)
    np.testing.assert_array_equal(left, replay)
    assert not np.array_equal(left, other)
    q = np.linspace(0.05, 0.95, 400, dtype=np.float64)
    pi = noise.calibrate_pi(q, 160)
    assert pi.sum() == pytest.approx(160.0, abs=1e-10)
    order = noise.derive_rng(1101, "pps-order/0").permutation(400)
    selected = noise.systematic_pps(pi, order, 0.25)
    assert len(selected) == len(np.unique(selected)) == 160


def _tiny_store(root: Path, role: str, source_commit: str) -> None:
    root.mkdir()
    ids = np.asarray([b"cifar100-python-train-v1/00000"], dtype="S30")
    np.save(root / "sample_ids.npy", ids, allow_pickle=False)
    names = ["sample_ids.npy"]
    if role == "train-images-only":
        np.save(root / "images.npy", np.zeros((1, 32, 32, 3), dtype=np.uint8))
        np.save(root / "train_order.npy", np.asarray([0], dtype=np.int64))
        names += ["images.npy", "train_order.npy"]
    elif role == "tuning-evaluation":
        np.save(root / "images.npy", np.zeros((1, 32, 32, 3), dtype=np.uint8))
        np.save(root / "clean_labels.npy", np.asarray([0], dtype=np.uint8))
        names += ["images.npy", "clean_labels.npy"]
    else:
        np.save(root / "noisy_labels.npy", np.asarray([1], dtype=np.uint8))
        names += ["noisy_labels.npy"]
    metadata: dict[str, object] = {
        "schema": "test/1",
        "role": role,
        "source_commit": source_commit,
        "test_loaded": False,
    }
    if role == "public-noise":
        metadata["train_ids_sha256"] = noise._sha(root / "sample_ids.npy")
    noise._seal(root, names, metadata)


def test_public_validation_rejects_private_fields(tmp_path: Path) -> None:
    public, images, tuning = tmp_path / "public", tmp_path / "images", tmp_path / "tuning"
    _tiny_store(public, "public-noise", "a" * 40)
    _tiny_store(images, "train-images-only", "a" * 40)
    _tiny_store(tuning, "tuning-evaluation", "a" * 40)
    assert noise.validate_public(public, images, tuning)["test_loaded"] is False
    np.save(public / "clean_labels.npy", np.asarray([0], dtype=np.uint8))
    manifest = noise._validate_seal(public)
    manifest["files"]["clean_labels.npy"] = {
        "bytes": (public / "clean_labels.npy").stat().st_size,
        "sha256": noise._sha(public / "clean_labels.npy"),
    }
    noise._write_json(public / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="private"):
        noise.validate_public(public, images, tuning)
