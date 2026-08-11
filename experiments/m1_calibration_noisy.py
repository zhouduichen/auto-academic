"""M1 noisy calibration runner consuming only sealed public data stores."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from experiments import m1_calibration_clean as clean
from experiments.m0_optimizer_state import m0_run as m0
from experiments.m1_noisy_data import _sha, validate_public


class NoisyTrainDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, image_store: Path, training: Path, augmentation_seed: int) -> None:
        ids = np.load(image_store / "sample_ids.npy", allow_pickle=False)
        public_ids = np.load(training / "sample_ids.npy", allow_pickle=False)
        if not np.array_equal(ids, public_ids):
            raise RuntimeError("public training/image IDs differ")
        self.images = np.load(image_store / "images.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(training / "noisy_labels.npy", mmap_mode="r", allow_pickle=False)
        self.order = np.load(image_store / "train_order.npy", allow_pickle=False)
        originals = np.asarray([int(value.decode("ascii")[-5:]) for value in ids])
        self.row_by_original = {int(value): row for row, value in enumerate(originals)}
        self.augmentation_seed = augmentation_seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, position: int) -> tuple[Tensor, int]:
        original = int(self.order[position])
        row = self.row_by_original[original]
        rng = np.random.default_rng(
            np.random.SeedSequence([self.augmentation_seed, self.epoch, original])
        )
        image = Image.fromarray(np.asarray(self.images[row]))
        tensor = m0._transform(
            image,
            int(rng.integers(0, 9)),
            int(rng.integers(0, 9)),
            bool(rng.integers(0, 2)),
        )
        return tensor, int(self.labels[row])


class TuningDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, tuning_store: Path) -> None:
        self.images = np.load(tuning_store / "images.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(tuning_store / "clean_labels.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, position: int) -> tuple[Tensor, int]:
        image = Image.fromarray(np.asarray(self.images[position]))
        return m0._transform(image, 4, 4, False), int(self.labels[position])


def _validate_input_provenance(
    public: dict[str, object],
    current: dict[str, object],
    actual_binding: dict[str, object],
    expected_input_binding: dict[str, object] | None,
) -> dict[str, object]:
    if expected_input_binding is None:
        if current["source_commit"] != public["source_commit"]:
            raise RuntimeError("runner/noise source commit mismatch")
        if current["uv_lock_sha256"] != public["uv_lock_sha256"]:
            raise RuntimeError("runner/noise lock mismatch")
        return {
            "runner_source_commit": current["source_commit"],
            "input_source_commit": public["source_commit"],
            "reused_sealed_input": False,
        }
    required = {
        "noise_bundle_sha256",
        "image_store_sha256",
        "tuning_store_sha256",
        "source_commit",
        "test_loaded",
    }
    if set(expected_input_binding) != required or expected_input_binding != actual_binding:
        raise RuntimeError("frozen input binding mismatch")
    if expected_input_binding["source_commit"] != public["source_commit"]:
        raise RuntimeError("frozen input binding source mismatch")
    if expected_input_binding["test_loaded"] is not False or public["test_loaded"] is not False:
        raise RuntimeError("test isolation flag failed")
    return {
        "runner_source_commit": current["source_commit"],
        "input_source_commit": public["source_commit"],
        "reused_sealed_input": True,
    }


def run(
    config: clean.Config,
    training: Path,
    image_store: Path,
    tuning_store: Path,
    output_dir: Path,
    *,
    expected_input_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    checkpoint_epochs = clean._validate_checkpoint_epochs(config)
    public = validate_public(training, image_store, tuning_store)
    actual_binding = {
        "noise_bundle_sha256": _sha(training / "manifest.json"),
        "image_store_sha256": _sha(image_store / "manifest.json"),
        "tuning_store_sha256": _sha(tuning_store / "manifest.json"),
        "source_commit": public["source_commit"],
        "test_loaded": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    m0.seed_everything(config.seed)
    device = torch.device(config.device)
    provenance = m0.build_provenance(device)
    input_provenance = _validate_input_provenance(
        public, provenance, actual_binding, expected_input_binding
    )
    written_binding = {**actual_binding, **input_provenance}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        binding_path = output_dir / "input_binding.json"
        if summary.get("status") == "succeeded":
            if not binding_path.is_file() or json.loads(
                binding_path.read_text(encoding="utf-8")
            ) != written_binding:
                raise RuntimeError("completed run input binding mismatch")
            return summary
    clean._write_json(output_dir / "config.json", asdict(config))
    clean._write_json(output_dir / "input_binding.json", written_binding)
    clean._write_json(output_dir / "provenance.json", provenance)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_data = NoisyTrainDataset(image_store, training, config.augmentation_seed)
    tuning_data = TuningDataset(tuning_store)
    model = clean._build_model(config).to(device)
    optimizer = clean._build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = output_dir / "checkpoint.pt"
    start_epoch = 0
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["next_epoch"])
    metrics_path = output_dir / "epoch_metrics.jsonl"
    gradient_path = output_dir / "gradient_norms.jsonl"
    diagnostic_path = output_dir / "optimizer_diagnostics.jsonl"
    existing_metrics = (
        [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if metrics_path.is_file()
        else []
    )
    steps_per_epoch = len(train_data) // config.batch_size
    if len(existing_metrics) < start_epoch:
        raise RuntimeError("checkpoint is ahead of epoch metrics")
    metrics_path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in existing_metrics[:start_epoch]
        ),
        encoding="utf-8",
    )
    existing_norms = gradient_path.read_text().splitlines() if gradient_path.is_file() else []
    completed_steps = start_epoch * steps_per_epoch
    if len(existing_norms) < completed_steps:
        raise RuntimeError("checkpoint is ahead of gradient metrics")
    gradient_path.write_text(
        "\n".join(existing_norms[:completed_steps]) + ("\n" if completed_steps else ""),
        encoding="utf-8",
    )
    existing_diagnostics = (
        diagnostic_path.read_text(encoding="utf-8").splitlines()
        if diagnostic_path.is_file()
        else []
    )
    if len(existing_diagnostics) < start_epoch and config.method != "adamw":
        raise RuntimeError("checkpoint is ahead of optimizer diagnostics")
    diagnostic_path.write_text(
        "\n".join(existing_diagnostics[:start_epoch]) + ("\n" if start_epoch else ""),
        encoding="utf-8",
    )
    elapsed_offset = (
        float(existing_metrics[start_epoch - 1]["elapsed_training_seconds"]) if start_epoch else 0.0
    )
    started = time.monotonic()
    optimizer_steps = completed_steps
    for epoch in range(start_epoch, config.epochs):
        train_data.epoch = epoch
        train_loader = clean._loader(train_data, config, epoch)
        tuning_loader = DataLoader(
            tuning_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=device.type == "cuda",
        )
        model.train()
        train_loss, seen = 0.0, 0
        norms: list[float] = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(m0.forward_logits(model, images), labels)
            loss.backward()
            norm = clean._global_gradient_norm(model)
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer_steps += 1
            norms.append(norm)
            train_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        tuning_loss, tuning_accuracy = clean._evaluate(model, tuning_loader, device)
        row = {
            "epoch": epoch + 1,
            "optimizer_steps": optimizer_steps,
            "train_loss": train_loss / seen,
            "tuning_loss": tuning_loss,
            "tuning_accuracy": tuning_accuracy,
            "elapsed_training_seconds": elapsed_offset + time.monotonic() - started,
            "test_loaded": False,
        }
        with gradient_path.open("a", encoding="utf-8") as stream:
            first = optimizer_steps - len(norms) + 1
            for step, norm in enumerate(norms, start=first):
                payload = json.dumps({"step": step, "global_l2": norm}, separators=(",", ":"))
                stream.write(payload + "\n")
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        epoch_diagnostic = clean._optimizer_epoch_diagnostic(
            optimizer, epoch + 1, optimizer_steps
        )
        if epoch_diagnostic is not None:
            with diagnostic_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(epoch_diagnostic, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
        clean._write_epoch_checkpoints(
            output_dir, model, optimizer, epoch + 1, checkpoint_epochs
        )
        print(json.dumps(row, sort_keys=True), flush=True)
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    summary = {
        "status": "succeeded",
        "condition": "noisy",
        "seed": config.seed,
        "epochs": config.epochs,
        "optimizer_steps": optimizer_steps,
        "peak_vram_gb": peak,
        "source_commit": provenance["source_commit"],
        "noise_bundle_sha256": _sha(training / "manifest.json"),
        "method": config.method,
        "test_loaded": False,
    }
    clean._write_json(summary_path, summary)
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"sha256_manifest.json", "stdout.log"}
    )
    clean._write_json(
        output_dir / "sha256_manifest.json", {path.name: _sha(path) for path in files}
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, choices=(101, 102, 201, 202, 301, 302, 303), required=True
    )
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--image-store", type=Path, required=True)
    parser.add_argument("--tuning-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--augmentation-seed", type=int)
    parser.add_argument("--method", choices=clean.RUNNER_METHODS, default="adamw")
    parser.add_argument("--checkpoint-epoch", type=int, action="append", default=[])
    parser.add_argument("--expected-input-binding", type=Path)
    args = parser.parse_args()
    config = clean.Config(
        seed=args.seed,
        augmentation_seed=(
            args.augmentation_seed if args.augmentation_seed is not None else 10_000 + args.seed
        ),
        epochs=args.epochs,
        workers=args.workers,
        device="cuda",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        max_grad_norm=args.max_grad_norm,
        method=args.method,
        checkpoint_epochs=tuple(args.checkpoint_epoch),
    )
    expected_input_binding = (
        json.loads(args.expected_input_binding.read_text(encoding="utf-8"))
        if args.expected_input_binding is not None
        else None
    )
    print(
        json.dumps(
            run(
                config,
                args.training,
                args.image_store,
                args.tuning_store,
                args.output_dir,
                expected_input_binding=expected_input_binding,
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
