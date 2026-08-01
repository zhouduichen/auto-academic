"""Prepare and seal exact-rate M1 instance-dependent-noise inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, ndtr, ndtri

from experiments.m1_calibration_clean import m1_split

ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
PRIVATE_NAMES = {
    "clean_labels.npy",
    "corruption_mask.npy",
    "q_base_raw.npy",
    "q_base_used.npy",
    "inclusion_probability.npy",
    "destination_probabilities.npy",
    "transition_probabilities.npy",
    "channel_statistics.npy",
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is unavailable")
    status = subprocess.run(  # noqa: S603 - resolved git with fixed arguments
        [git, "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("tracked worktree must be clean")
    return subprocess.run(  # noqa: S603 - resolved git with fixed arguments
        [git, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _uv_lock_sha256() -> str:
    return _sha(Path(__file__).resolve().parents[1] / "uv.lock")


def _seal(root: Path, names: list[str], metadata: dict[str, object]) -> None:
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": _sha(root / name)}
        for name in sorted(names)
    }
    _write_json(root / "manifest.json", {**metadata, "files": files})


def _validate_seal(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        if path.parent != root.resolve() or not path.is_file():
            raise RuntimeError(f"invalid sealed path: {name}")
        if path.stat().st_size != record["bytes"] or _sha(path) != record["sha256"]:
            raise RuntimeError(f"sealed file mismatch: {name}")
    return manifest


def derive_rng(seed: int, domain: str) -> np.random.Generator:
    payload = b"m1-idn-v1\0" + seed.to_bytes(8, "big") + b"\0" + domain.encode()
    entropy = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    return np.random.Generator(np.random.PCG64DXSM(entropy))


def calibrate_pi(q: np.ndarray, target: int) -> np.ndarray:
    lower, upper = -64.0, 64.0
    values = np.asarray(q, dtype=np.float64)
    if expit(logit(values) + lower).sum() >= target:
        raise ValueError("invalid lower calibration bracket")
    if expit(logit(values) + upper).sum() <= target:
        raise ValueError("invalid upper calibration bracket")
    for _ in range(128):
        midpoint = np.float64((lower + upper) / 2.0)
        if expit(logit(values) + midpoint).sum() < target:
            lower = float(midpoint)
        else:
            upper = float(midpoint)
    return expit(logit(values) + np.float64((lower + upper) / 2.0))


def systematic_pps(pi: np.ndarray, order: np.ndarray, start: float) -> np.ndarray:
    cumulative = np.cumsum(pi[order], dtype=np.float64)
    target = round(float(pi.sum()))
    cumulative[-1] = float(target)
    selected = np.searchsorted(
        cumulative, start + np.arange(target, dtype=np.float64), side="right"
    )
    result = order[selected]
    if len(result) != len(np.unique(result)):
        raise RuntimeError("systematic PPS selected duplicate positions")
    return result


def _stable_ids(indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"cifar100-python-train-v1/{int(index):05d}".encode() for index in indices],
        dtype="S30",
    )


def prepare_stores(archive: Path, train_payload: Path, output: Path) -> None:
    if hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest() != ARCHIVE_MD5:
        raise RuntimeError("CIFAR-100 archive MD5 mismatch")
    with train_payload.open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")  # noqa: S301 - pinned CIFAR archive
    raw = payload["data"] if "data" in payload else payload[b"data"]
    label_key = "fine_labels" if "fine_labels" in payload else b"fine_labels"
    labels = np.asarray(payload[label_key], dtype=np.uint8)
    images = np.asarray(raw, dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    split = m1_split(labels.tolist(), 20_260_801)
    train_order = np.asarray(split["train"], dtype=np.int64)
    train_sorted = np.sort(train_order)
    tuning = np.asarray(split["tuning"], dtype=np.int64)
    source = output / "private-train-source"
    image_store = output / "image-store"
    tuning_store = output / "tuning-store"
    for directory in (source, image_store, tuning_store):
        directory.mkdir(parents=True, exist_ok=False)
    train_ids = _stable_ids(train_sorted)
    np.save(source / "sample_ids.npy", train_ids, allow_pickle=False)
    np.save(source / "images.npy", images[train_sorted], allow_pickle=False)
    np.save(source / "clean_labels.npy", labels[train_sorted], allow_pickle=False)
    np.save(source / "train_order.npy", train_order, allow_pickle=False)
    np.save(image_store / "sample_ids.npy", train_ids, allow_pickle=False)
    np.save(image_store / "images.npy", images[train_sorted], allow_pickle=False)
    np.save(image_store / "train_order.npy", train_order, allow_pickle=False)
    np.save(tuning_store / "sample_ids.npy", _stable_ids(tuning), allow_pickle=False)
    np.save(tuning_store / "images.npy", images[tuning], allow_pickle=False)
    np.save(tuning_store / "clean_labels.npy", labels[tuning], allow_pickle=False)
    common = {
        "schema": "m1-image-store/1",
        "source_commit": _source_commit(),
        "uv_lock_sha256": _uv_lock_sha256(),
        "split_seed": 20_260_801,
        "archive_sha256": _sha(archive),
        "train_payload_sha256": _sha(train_payload),
        "test_constructed": False,
        "test_loaded": False,
        "test_access_count": 0,
    }
    _seal(
        source,
        ["sample_ids.npy", "images.npy", "clean_labels.npy", "train_order.npy"],
        {**common, "role": "private-train-source"},
    )
    _seal(
        image_store,
        ["sample_ids.npy", "images.npy", "train_order.npy"],
        {**common, "role": "train-images-only"},
    )
    _seal(
        tuning_store,
        ["sample_ids.npy", "images.npy", "clean_labels.npy"],
        {**common, "role": "tuning-evaluation"},
    )


def generate_noise(
    source: Path, training: Path, audit: Path, noise_seed: int, opaque_id: str
) -> None:
    source_manifest = _validate_seal(source)
    if source_manifest["role"] != "private-train-source":
        raise RuntimeError("noise generator requires the private train role")
    if source_manifest["source_commit"] != _source_commit():
        raise RuntimeError("generator/source-store commit mismatch")
    ids = np.load(source / "sample_ids.npy", allow_pickle=False)
    images = np.load(source / "images.npy", allow_pickle=False)
    clean = np.load(source / "clean_labels.npy", allow_pickle=False)
    if ids.shape != (40_000,) or images.shape != (40_000, 32, 32, 3):
        raise RuntimeError("unexpected train source shape")
    if clean.shape != (40_000,) or len(np.unique(ids)) != 40_000:
        raise RuntimeError("invalid train IDs or labels")
    if not np.array_equal(ids, np.sort(ids)):
        raise RuntimeError("train IDs must be stable-ID sorted")
    if not np.array_equal(np.bincount(clean, minlength=100), np.full(100, 400)):
        raise RuntimeError("train source must contain 400 examples per class")

    z = images.astype(np.float64) / 255.0
    channel_mean = z.mean(axis=(0, 1, 2), dtype=np.float64)
    channel_std = z.std(axis=(0, 1, 2), dtype=np.float64, ddof=0)
    features = ((z - channel_mean) / channel_std).reshape(40_000, -1)
    norms = np.linalg.norm(features, axis=1)
    features /= norms[:, None]
    if not np.isfinite(features).all() or np.any(norms <= 0):
        raise RuntimeError("invalid feature matrix")

    q_raw = np.empty(40_000, dtype=np.float64)
    q_used = np.empty_like(q_raw)
    inclusion = np.empty_like(q_raw)
    destination = np.zeros((40_000, 100), dtype=np.float64)
    transition = np.zeros_like(destination)
    noisy = clean.copy()
    mask = np.zeros(40_000, dtype=bool)
    matrix_hashes: dict[str, str] = {}
    a, b = -4.0, 6.0
    low_cdf, span = ndtr(a), ndtr(b) - ndtr(a)
    for class_id in range(100):
        positions = np.flatnonzero(clean == class_id)
        u = derive_rng(noise_seed, f"base-rate/{class_id}").random(400)
        raw = 0.4 + 0.1 * ndtri(low_cdf + u * span)
        used = np.clip(raw, 1e-12, 1.0 - 1e-12)
        pi = calibrate_pi(used, 160)
        q_raw[positions], q_used[positions], inclusion[positions] = raw, used, pi
        order = derive_rng(noise_seed, f"pps-order/{class_id}").permutation(400)
        start = float(derive_rng(noise_seed, f"pps-start/{class_id}").random())
        selected_local = systematic_pps(pi, order, start)
        selected = positions[selected_local]
        mask[selected] = True
        matrix = derive_rng(noise_seed, f"matrix/{class_id}").standard_normal(
            (3072, 100), dtype=np.float64
        )
        matrix_hashes[str(class_id)] = hashlib.sha256(matrix.tobytes()).hexdigest()
        logits = features[positions] @ matrix
        logits[:, class_id] = -np.inf
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        destination[positions] = probabilities
        transition[positions] = pi[:, None] * probabilities
        transition[positions, class_id] = 1.0 - pi
        for position in selected:
            stable_id = ids[position].decode("ascii")
            noisy[position] = derive_rng(noise_seed, f"destination/{stable_id}").choice(
                100, p=destination[position]
            )

    counts = np.bincount(clean[mask], minlength=100)
    if mask.sum() != 16_000 or not np.array_equal(counts, np.full(100, 160)):
        raise RuntimeError("exact corruption count validation failed")
    if np.any(noisy[mask] == clean[mask]) or np.any(noisy[~mask] != clean[~mask]):
        raise RuntimeError("noisy-label inequality validation failed")
    if not np.isfinite(transition).all() or not np.allclose(
        transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("transition validation failed")

    training.mkdir(parents=True, exist_ok=False)
    audit.mkdir(parents=True, exist_ok=False)
    np.save(training / "sample_ids.npy", ids, allow_pickle=False)
    np.save(training / "noisy_labels.npy", noisy, allow_pickle=False)
    private_arrays = {
        "clean_labels.npy": clean,
        "corruption_mask.npy": mask,
        "q_base_raw.npy": q_raw,
        "q_base_used.npy": q_used,
        "inclusion_probability.npy": inclusion,
        "destination_probabilities.npy": destination,
        "transition_probabilities.npy": transition,
        "channel_statistics.npy": np.stack([channel_mean, channel_std]),
    }
    for name, value in private_arrays.items():
        np.save(audit / name, value, allow_pickle=False)
    common = {
        "source_commit": source_manifest["source_commit"],
        "uv_lock_sha256": source_manifest["uv_lock_sha256"],
        "train_ids_sha256": _sha(source / "sample_ids.npy"),
        "test_constructed": False,
        "test_loaded": False,
        "test_access_count": 0,
    }
    _seal(
        training,
        ["sample_ids.npy", "noisy_labels.npy"],
        {
            **common,
            "schema": "m1-noise-training/1",
            "opaque_artifact_id": opaque_id,
            "sample_count": 40_000,
            "protocol_handle": "opaque-m1-idn-v1",
        },
    )
    _seal(
        audit,
        list(private_arrays),
        {
            **common,
            "schema": "m1-noise-audit/1",
            "noise_seed": noise_seed,
            "total_corruptions": 16_000,
            "corruptions_per_class": counts.tolist(),
            "feature_matrix_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "matrix_sha256": matrix_hashes,
        },
    )


def validate_public(training: Path, image_store: Path, tuning_store: Path) -> dict[str, object]:
    public = _validate_seal(training)
    images = _validate_seal(image_store)
    tuning = _validate_seal(tuning_store)
    if any(name in PRIVATE_NAMES for name in public["files"]):
        raise RuntimeError("public bundle exposes private arrays")
    if public["train_ids_sha256"] != images["files"]["sample_ids.npy"]["sha256"]:
        raise RuntimeError("noise/image sample IDs differ")
    if images["source_commit"] != public["source_commit"] or tuning["source_commit"] != public[
        "source_commit"
    ]:
        raise RuntimeError("source commit binding mismatch")
    if any(manifest.get("test_loaded") for manifest in (public, images, tuning)):
        raise RuntimeError("test isolation flag failed")
    return public


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--train-payload", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--source", type=Path, required=True)
    generate.add_argument("--training", type=Path, required=True)
    generate.add_argument("--audit", type=Path, required=True)
    generate.add_argument("--noise-seed", type=int, choices=(1101, 1102), required=True)
    generate.add_argument("--opaque-id", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--training", type=Path, required=True)
    validate.add_argument("--image-store", type=Path, required=True)
    validate.add_argument("--tuning-store", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_stores(args.archive, args.train_payload, args.output)
    elif args.command == "generate":
        generate_noise(args.source, args.training, args.audit, args.noise_seed, args.opaque_id)
    else:
        print(json.dumps(validate_public(args.training, args.image_store, args.tuning_store)))


if __name__ == "__main__":
    main()
