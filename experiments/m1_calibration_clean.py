"""Minimal clean-only M1 AdamW budget-calibration runner.

Only CIFAR-100's training split is constructed. The tuning partition is used
for epoch evaluation; development, confirmation, and official test data are
never accessed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100

from experiments.m0_optimizer_state import m0_run as m0
from experiments.m1_optimizers import CAdamW, ProtectAdamW

PILOT_METHODS = ("adamw", "cadam", "protect-m", "protect-mv")


@dataclass(frozen=True)
class Config:
    seed: int
    split_seed: int = 20_260_801
    augmentation_seed: int = 0
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float | None = None
    model_id: str = "google/vit-base-patch16-224-in21k"
    lora_rank: int = 8
    workers: int = 4
    device: str = "cuda"
    method: str = "adamw"


class IndexedCIFAR100(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        dataset: Any,
        indices: list[int],
        *,
        augmentation_seed: int | None,
        epoch: int = 0,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.augmentation_seed = augmentation_seed
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Tensor, int]:
        index = self.indices[position]
        image, label = self.dataset[index]
        if self.augmentation_seed is None:
            top, left, flip = 4, 4, False
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.augmentation_seed, self.epoch, index])
            )
            top = int(rng.integers(0, 9))
            left = int(rng.integers(0, 9))
            flip = bool(rng.integers(0, 2))
        return m0._transform(image, top, left, flip), int(label)


def m1_split(labels: list[int], split_seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(split_seed)
    label_array = np.asarray(labels)
    split = {"train": [], "tuning": [], "development": [], "confirmation": []}
    for class_id in range(100):
        indices = np.flatnonzero(label_array == class_id)
        if len(indices) != 500:
            raise RuntimeError(f"class {class_id} has {len(indices)} examples, expected 500")
        rng.shuffle(indices)
        split["train"].extend(indices[:400].tolist())
        split["tuning"].extend(indices[400:425].tolist())
        split["development"].extend(indices[425:450].tolist())
        split["confirmation"].extend(indices[450:].tolist())
    for values in split.values():
        rng.shuffle(values)
    expected = {"train": 40_000, "tuning": 2_500, "development": 2_500, "confirmation": 5_000}
    if {key: len(value) for key, value in split.items()} != expected:
        raise RuntimeError("unexpected M1 split sizes")
    return split


def _write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _sha_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _build_model(config: Config) -> nn.Module:
    return m0.build_model(
        m0.RunConfig(
            optimizer="adamw",
            pulse="label_flip",
            seed=config.seed,
            split_seed=config.split_seed,
            pulse_seed=0,
            warmup_steps=0,
            replay_steps=0,
            batch_size=config.batch_size,
            probe_size=1,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            model_id=config.model_id,
            lora_rank=config.lora_rank,
            device=config.device,
        )
    )


def _build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith(".bias") or "norm" in name.lower() else decay).append(parameter)
    if not decay or not no_decay:
        raise RuntimeError("AdamW parameter-group audit failed")
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    common = {
        "lr": config.learning_rate,
        "betas": config.betas,
        "eps": config.eps,
    }
    if config.method == "adamw":
        return torch.optim.AdamW(groups, **common, amsgrad=False, foreach=False, fused=False)
    if config.method == "cadam":
        return CAdamW(groups, **common)
    if config.method in {"protect-m", "protect-mv"}:
        return ProtectAdamW(
            groups,
            **common,
            protection_target=config.method.removeprefix("protect-"),
            alpha_fast=0.9,
            alpha_slow=0.99,
            tau=0.1,
        )
    raise ValueError(f"unsupported M1 method: {config.method}")


def _optimizer_diagnostic(
    optimizer: torch.optim.Optimizer, step: int
) -> dict[str, object] | None:
    value = getattr(optimizer, "last_diagnostics", None)
    if not isinstance(value, dict) or not value:
        return None
    return {"step": step, **value}


def _write_optimizer_diagnostic(
    optimizer: torch.optim.Optimizer, path: Path, step: int
) -> None:
    row = _optimizer_diagnostic(optimizer, step)
    if row is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _optimizer_epoch_diagnostic(
    optimizer: torch.optim.Optimizer, epoch: int, optimizer_steps: int
) -> dict[str, object] | None:
    summary = getattr(optimizer, "diagnostics_summary", None)
    if callable(summary):
        value = summary(reset=True)
    else:
        value = getattr(optimizer, "last_diagnostics", None)
    if not isinstance(value, dict) or not value:
        return None
    return {"epoch": epoch, "optimizer_steps": optimizer_steps, **value}


def _global_gradient_norm(model: nn.Module) -> float:
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("trainable parameters have no gradients")
    value = float(
        torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(gradient.float()) for gradient in gradients])
        )
    )
    if not math.isfinite(value):
        raise RuntimeError("non-finite global gradient norm")
    return value


def _loader(dataset: Dataset[tuple[Tensor, int]], config: Config, epoch: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed * 1_000_000 + epoch)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.workers,
        pin_memory=config.device == "cuda",
        drop_last=False,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = m0.forward_logits(model, images)
            total_loss += float(criterion(logits, labels))
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
    return total_loss / count, correct / count


def _checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, next_epoch: int
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "next_epoch": next_epoch,
        },
        temporary,
    )
    temporary.replace(path)


def run(config: Config, data_dir: Path, output_dir: Path) -> dict[str, object]:
    if config.augmentation_seed != 10_000 + config.seed:
        raise ValueError("augmentation_seed must equal 10000 + train seed")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "succeeded":
            return summary

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    m0.seed_everything(config.seed)
    device = torch.device(config.device)
    provenance = m0.build_provenance(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # The official test constructor is deliberately absent from this module.
    dataset = CIFAR100(root=data_dir, train=True, download=False, transform=None)
    split = m1_split(dataset.targets, config.split_seed)
    split_payload = {
        **split,
        "labels_sha256": sha256(np.asarray(dataset.targets, dtype=np.uint8).tobytes()).hexdigest(),
        "test_loaded": False,
    }
    _write_json(output_dir / "config.json", asdict(config))
    _write_json(output_dir / "split_manifest.json", split_payload)
    _write_json(output_dir / "provenance.json", {**provenance, "test_loaded": False})

    model = _build_model(config).to(device)
    optimizer = _build_optimizer(model, config)
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
    elapsed_offset = 0.0
    if metrics_path.is_file():
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) < start_epoch:
            raise RuntimeError("checkpoint is ahead of epoch metrics")
        if start_epoch:
            elapsed_offset = float(rows[start_epoch - 1]["elapsed_training_seconds"])
        metrics_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows[:start_epoch]),
            encoding="utf-8",
        )
    if gradient_path.is_file():
        rows = [json.loads(line) for line in gradient_path.read_text(encoding="utf-8").splitlines()]
        completed_steps = start_epoch * (len(split["train"]) // config.batch_size)
        if len(rows) < completed_steps:
            raise RuntimeError("checkpoint is ahead of gradient metrics")
        gradient_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n" for row in rows[:completed_steps]
            ),
            encoding="utf-8",
        )
    if diagnostic_path.is_file():
        rows = diagnostic_path.read_text(encoding="utf-8").splitlines()
        if len(rows) < start_epoch and config.method != "adamw":
            raise RuntimeError("checkpoint is ahead of optimizer diagnostics")
        diagnostic_path.write_text(
            "\n".join(rows[:start_epoch]) + ("\n" if start_epoch else ""),
            encoding="utf-8",
        )
    started = time.monotonic()
    optimizer_steps = start_epoch * (len(split["train"]) // config.batch_size)
    for epoch in range(start_epoch, config.epochs):
        train_data = IndexedCIFAR100(
            dataset, split["train"], augmentation_seed=config.augmentation_seed, epoch=epoch
        )
        validation_data = IndexedCIFAR100(dataset, split["tuning"], augmentation_seed=None)
        train_loader = _loader(train_data, config, epoch)
        validation_loader = DataLoader(
            validation_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=device.type == "cuda",
        )
        model.train()
        train_loss = 0.0
        seen = 0
        epoch_norms: list[float] = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(m0.forward_logits(model, images), labels)
            loss.backward()
            norm = _global_gradient_norm(model)
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer_steps += 1
            epoch_norms.append(norm)
            train_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        validation_loss, validation_accuracy = _evaluate(model, validation_loader, device)
        elapsed = elapsed_offset + time.monotonic() - started
        row = {
            "epoch": epoch + 1,
            "optimizer_steps": optimizer_steps,
            "train_loss": train_loss / seen,
            "tuning_loss": validation_loss,
            "tuning_accuracy": validation_accuracy,
            "elapsed_training_seconds": elapsed,
            "test_loaded": False,
        }
        with gradient_path.open("a", encoding="utf-8") as stream:
            for step, norm in enumerate(epoch_norms, start=optimizer_steps - len(epoch_norms) + 1):
                payload = json.dumps({"step": step, "global_l2": norm}, separators=(",", ":"))
                stream.write(payload + "\n")
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        epoch_diagnostic = _optimizer_epoch_diagnostic(optimizer, epoch + 1, optimizer_steps)
        if epoch_diagnostic is not None:
            with diagnostic_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(epoch_diagnostic, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
        _checkpoint(checkpoint_path, model, optimizer, epoch + 1)
        print(json.dumps(row, sort_keys=True), flush=True)

    peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    summary = {
        "status": "succeeded",
        "condition": "clean",
        "seed": config.seed,
        "epochs": config.epochs,
        "optimizer_steps": optimizer_steps,
        "peak_vram_gb": peak_vram,
        "source_commit": provenance["source_commit"],
        "method": config.method,
        "test_loaded": False,
    }
    _write_json(summary_path, summary)
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"sha256_manifest.json", "stdout.log"}
    )
    _write_json(output_dir / "sha256_manifest.json", {path.name: _sha_file(path) for path in files})
    return summary


def parse_args() -> tuple[Config, Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, choices=(101, 102, 201, 202, 301, 302, 303), required=True
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data/cifar100"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--augmentation-seed", type=int)
    parser.add_argument("--method", choices=PILOT_METHODS, default="adamw")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 50:
        parser.error("epochs must be in [1, 50]")
    config = Config(
        seed=args.seed,
        augmentation_seed=(
            args.augmentation_seed if args.augmentation_seed is not None else 10_000 + args.seed
        ),
        epochs=args.epochs,
        workers=args.workers,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        max_grad_norm=args.max_grad_norm,
        method=args.method,
    )
    return config, args.data_dir, args.output_dir


def main() -> None:
    config, data_dir, output_dir = parse_args()
    print(json.dumps(run(config, data_dir, output_dir), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
