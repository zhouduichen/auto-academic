"""Cross clean/noisy parameters and optimizer state under common continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

from experiments import m1_calibration_clean as clean
from experiments import m1_calibration_noisy as noisy
from experiments.m0_optimizer_state import m0_run as m0
from experiments.m1_pilot import source_commit as current_source_commit
from src.arw import m0_core

HORIZONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
BRANCH_NAMES = ("CC", "CN", "NC", "NN")


@dataclass(frozen=True)
class CheckpointState:
    parameters: dict[str, Tensor]
    optimizer: dict[str, Any]
    next_epoch: int


@dataclass(frozen=True)
class ReplayRequest:
    clean_checkpoint: Path
    noisy_checkpoint: Path
    continuation: Literal["clean", "noisy"]
    seed: Literal[301, 302]
    checkpoint_epoch: Literal[1, 2, 3, 20]
    method: Literal["adamw", "cadam"]
    data_dir: Path
    image_store: Path
    tuning_store: Path
    noisy_bundle: Path
    output: Path
    source_commit: str
    device: str = "cuda"
    probe_size: int = 256


def _load_payload(path: Path, device: torch.device) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"checkpoint cannot be loaded: {path}") from error
    if not isinstance(payload, dict) or set(payload) != {"model", "optimizer", "next_epoch"}:
        raise ValueError("checkpoint top-level schema mismatch")
    if not isinstance(payload["model"], dict) or not isinstance(payload["optimizer"], dict):
        raise ValueError("checkpoint model or optimizer is malformed")
    next_epoch = payload["next_epoch"]
    if isinstance(next_epoch, bool) or not isinstance(next_epoch, int) or not 1 <= next_epoch <= 50:
        raise ValueError("checkpoint next_epoch is malformed")
    return payload


def _state_from_payload(model: nn.Module, payload: dict[str, object]) -> CheckpointState:
    model_state = payload["model"]
    optimizer_state = payload["optimizer"]
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint payload is malformed")
    model.load_state_dict(model_state, strict=True)
    return CheckpointState(
        parameters=m0_core.capture_trainable_state(model),
        optimizer=deepcopy(optimizer_state),
        next_epoch=int(payload["next_epoch"]),
    )


def load_checkpoint_state(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> CheckpointState:
    return _state_from_payload(model, _load_payload(checkpoint_path, device))


def _validate_finite(value: object, label: str) -> None:
    if isinstance(value, Tensor):
        if value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite or empty {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite {label}")


def _step_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.numel() != 1 or right.numel() != 1 or not torch.equal(left, right):
            return False
        value = float(left)
        return math.isfinite(value) and value >= 0 and value.is_integer()
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, int | float) or not isinstance(right, int | float) or left != right:
        return False
    value = float(left)
    return math.isfinite(value) and value >= 0 and value.is_integer()


def validate_checkpoint_pair(clean_state: CheckpointState, noisy_state: CheckpointState) -> None:
    if clean_state.next_epoch != noisy_state.next_epoch:
        raise ValueError("checkpoint next_epoch values do not match")
    if not clean_state.parameters or set(clean_state.parameters) != set(noisy_state.parameters):
        raise ValueError("checkpoint parameter names do not match")
    for name in sorted(clean_state.parameters):
        left = clean_state.parameters[name]
        right = noisy_state.parameters[name]
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            raise ValueError("checkpoint parameters must be tensors")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("checkpoint parameter shape or dtype mismatch")
        _validate_finite(left, f"clean parameter {name}")
        _validate_finite(right, f"noisy parameter {name}")

    for label, optimizer in (("clean", clean_state.optimizer), ("noisy", noisy_state.optimizer)):
        if set(optimizer) != {"state", "param_groups"}:
            raise ValueError(f"{label} optimizer schema is malformed")
        _validate_finite(optimizer, f"{label} optimizer")
    if clean_state.optimizer["param_groups"] != noisy_state.optimizer["param_groups"]:
        raise ValueError("optimizer parameter groups do not match")
    clean_raw = clean_state.optimizer["state"]
    noisy_raw = noisy_state.optimizer["state"]
    if not isinstance(clean_raw, dict) or not isinstance(noisy_raw, dict) or not clean_raw:
        raise ValueError("optimizer state is empty or malformed")
    if set(clean_raw) != set(noisy_raw):
        raise ValueError("optimizer parameter-state IDs do not match")
    for parameter_id in clean_raw:
        left = clean_raw[parameter_id]
        right = noisy_raw[parameter_id]
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("optimizer parameter-state keys do not match")
        if set(left) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("optimizer parameter-state schema is unsupported")
        if not _step_equal(left["step"], right["step"]):
            raise ValueError("optimizer step counters do not match")
        for moment in ("exp_avg", "exp_avg_sq"):
            left_moment = left[moment]
            right_moment = right[moment]
            if not isinstance(left_moment, Tensor) or not isinstance(right_moment, Tensor):
                raise ValueError(f"optimizer {moment} is not a tensor")
            if left_moment.shape != right_moment.shape or left_moment.dtype != right_moment.dtype:
                raise ValueError(f"optimizer {moment} shape or dtype mismatch")


def build_factorial_branches(
    clean_state: CheckpointState, noisy_state: CheckpointState
) -> dict[str, m0_core.BranchState]:
    validate_checkpoint_pair(clean_state, noisy_state)
    return {
        "CC": m0_core.BranchState(
            deepcopy(clean_state.parameters), deepcopy(clean_state.optimizer), 0.0, 0.0
        ),
        "CN": m0_core.BranchState(
            deepcopy(clean_state.parameters), deepcopy(noisy_state.optimizer), 0.0, 0.0
        ),
        "NC": m0_core.BranchState(
            deepcopy(noisy_state.parameters), deepcopy(clean_state.optimizer), 0.0, 0.0
        ),
        "NN": m0_core.BranchState(
            deepcopy(noisy_state.parameters), deepcopy(noisy_state.optimizer), 0.0, 0.0
        ),
    }


def factorial_effects(values: dict[str, float]) -> dict[str, float]:
    if set(values) != set(BRANCH_NAMES) or any(
        isinstance(value, bool) or not math.isfinite(value) for value in values.values()
    ):
        raise ValueError("factorial values must contain four finite branches")
    return {
        "parameter": 0.5 * ((values["NC"] - values["CC"]) + (values["NN"] - values["CN"])),
        "state": 0.5 * ((values["CN"] - values["CC"]) + (values["NN"] - values["NC"])),
        "interaction": values["NN"] - values["NC"] - values["CN"] + values["CC"],
    }


def capture_common_rng() -> dict[str, Any]:
    return m0.capture_rng_state()


def _forward_logits(model: nn.Module, images: Tensor) -> Tensor:
    try:
        output = model(pixel_values=images)
    except TypeError:
        output = model(images)
    if isinstance(output, Tensor):
        return output
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("model output does not expose logits")
    return logits


def _training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    images: Tensor,
    labels: Tensor,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(_forward_logits(model, images), labels)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite replay loss")
    loss.backward()
    optimizer.step()


def run_cached_branches(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    branches: dict[str, m0_core.BranchState],
    cached_batches: list[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    common_rng: dict[str, Any],
    horizons: tuple[int, ...],
    measure: Callable[[str, int], dict[str, object]],
) -> list[dict[str, object]]:
    if tuple(branches) != BRANCH_NAMES:
        raise ValueError("factorial branches must be ordered CC, CN, NC, NN")
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("replay horizons must be unique and increasing")
    if horizons[0] < 0 or horizons[-1] > len(cached_batches):
        raise ValueError("replay horizons exceed cached batches")
    device = next(model.parameters()).device
    selected = set(horizons)
    rows: list[dict[str, object]] = []
    for branch_name in BRANCH_NAMES:
        m0_core.restore_branch(model, optimizer, branches[branch_name])
        m0.restore_rng_state(common_rng)
        for horizon in range(len(cached_batches) + 1):
            if horizon in selected:
                row = measure(branch_name, horizon)
                if row.get("branch") != branch_name or row.get("horizon") != horizon:
                    raise RuntimeError("measurement callback returned wrong branch or horizon")
                if row.get("test_loaded") is not False:
                    raise RuntimeError("measurement callback violated test isolation")
                rows.append(row)
            if horizon < len(cached_batches):
                cpu_images, cpu_labels = cached_batches[horizon]
                _training_step(
                    model,
                    optimizer,
                    criterion,
                    cpu_images.to(device, non_blocking=True),
                    cpu_labels.to(device, non_blocking=True),
                )
        print(f"replay branch={branch_name} complete", flush=True)
    return rows


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid replay input JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"replay input JSON must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_from_json(value: dict[str, object], device: str) -> clean.Config:
    fields = {field.name for field in clean.Config.__dataclass_fields__.values()}
    unknown = set(value) - fields
    if unknown:
        raise RuntimeError(f"unsupported capture config keys: {sorted(unknown)}")
    payload = dict(value)
    payload["device"] = device
    payload["betas"] = tuple(payload.get("betas", (0.9, 0.999)))
    payload["checkpoint_epochs"] = tuple(payload.get("checkpoint_epochs", ()))
    return clean.Config(**payload)


def _validate_request(request: ReplayRequest) -> tuple[clean.Config, dict[str, object]]:
    if request.continuation not in {"clean", "noisy"}:
        raise ValueError("continuation must be clean or noisy")
    if request.seed not in {301, 302} or request.checkpoint_epoch not in {1, 2, 3, 20}:
        raise ValueError("unsupported replay seed or checkpoint")
    if request.method not in {"adamw", "cadam"}:
        raise ValueError("unsupported replay method")
    if request.method == "cadam" and request.checkpoint_epoch != 20:
        raise ValueError("CAdam confirmation is restricted to epoch 20")
    if request.method == "adamw" and request.checkpoint_epoch == 20:
        raise ValueError("mandatory AdamW replay is restricted to epochs 1, 2, and 3")
    if request.source_commit != current_source_commit():
        raise RuntimeError("replay request source commit mismatch")
    clean_config_value = _read_json(request.clean_checkpoint.parent / "config.json")
    noisy_config_value = _read_json(request.noisy_checkpoint.parent / "config.json")
    if clean_config_value != noisy_config_value:
        raise RuntimeError("clean/noisy capture configurations differ")
    config = _config_from_json(clean_config_value, request.device)
    if config.seed != request.seed or config.method != request.method:
        raise RuntimeError("capture config seed or method mismatch")
    for condition, checkpoint in (
        ("clean", request.clean_checkpoint),
        ("noisy", request.noisy_checkpoint),
    ):
        summary = _read_json(checkpoint.parent / "summary.json")
        if (
            summary.get("status") != "succeeded"
            or summary.get("condition") != condition
            or summary.get("seed") != request.seed
            or summary.get("method") != request.method
            or summary.get("test_loaded") is not False
        ):
            raise RuntimeError(f"{condition} capture summary is invalid")
    binding = _read_json(request.noisy_checkpoint.parent / "input_binding.json")
    if binding.get("test_loaded") is not False:
        raise RuntimeError("noisy capture input binding violated test isolation")
    return config, binding


def _compare_full_model_states(
    clean_payload: dict[str, object], noisy_payload: dict[str, object], trainable: set[str]
) -> None:
    clean_model = clean_payload["model"]
    noisy_model = noisy_payload["model"]
    if not isinstance(clean_model, dict) or not isinstance(noisy_model, dict):
        raise ValueError("checkpoint model state is malformed")
    if set(clean_model) != set(noisy_model):
        raise ValueError("full model state keys do not match")
    for name in clean_model:
        left = clean_model[name]
        right = noisy_model[name]
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            raise ValueError("full model state contains non-tensors")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("full model tensor shape or dtype mismatch")
        _validate_finite(left, f"clean model {name}")
        _validate_finite(right, f"noisy model {name}")
        if name not in trainable and not torch.equal(left, right):
            raise ValueError("frozen model tensors differ across checkpoints")


def _cached_continuation(
    request: ReplayRequest, config: clean.Config, next_epoch: int
) -> list[tuple[Tensor, Tensor]]:
    if request.continuation == "clean":
        dataset = CIFAR100(root=request.data_dir, train=True, download=False, transform=None)
        split = clean.m1_split(dataset.targets, config.split_seed)
        training = clean.IndexedCIFAR100(
            dataset,
            split["train"],
            augmentation_seed=config.augmentation_seed,
            epoch=next_epoch,
        )
    else:
        training = noisy.NoisyTrainDataset(
            request.image_store, request.noisy_bundle, config.augmentation_seed
        )
        training.epoch = next_epoch
    loader = clean._loader(training, config, next_epoch)
    cached: list[tuple[Tensor, Tensor]] = []
    for images, labels in loader:
        cached.append((images.detach().cpu().clone(), labels.detach().cpu().clone()))
        if len(cached) == 512:
            break
    if len(cached) != 512:
        raise RuntimeError("continuation loader produced fewer than 512 batches")
    return cached


def _probe_and_tuning_loaders(
    request: ReplayRequest, config: clean.Config
) -> tuple[list[tuple[Tensor, Tensor]], DataLoader]:
    tuning = noisy.TuningDataset(request.tuning_store)
    if not 1 <= request.probe_size <= len(tuning):
        raise ValueError("probe_size is outside the tuning store")
    rng = np.random.default_rng(
        np.random.SeedSequence([20_260_811, request.seed, request.checkpoint_epoch])
    )
    indices = sorted(rng.choice(len(tuning), request.probe_size, replace=False).tolist())
    probe_loader = DataLoader(
        Subset(tuning, indices), batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    probe = [(images.clone(), labels.clone()) for images, labels in probe_loader]
    tuning_loader = DataLoader(
        tuning,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=request.device == "cuda",
    )
    return probe, tuning_loader


def _probe_metrics(
    model: nn.Module,
    criterion: nn.Module,
    probe: list[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[float, Tensor]:
    model.eval()
    loss_sum = 0.0
    count = 0
    probabilities: list[Tensor] = []
    with torch.no_grad():
        for cpu_images, cpu_labels in probe:
            images = cpu_images.to(device, non_blocking=True)
            labels = cpu_labels.to(device, non_blocking=True)
            logits = _forward_logits(model, images)
            loss_sum += float(criterion(logits, labels)) * len(labels)
            count += len(labels)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
    return loss_sum / count, torch.cat(probabilities)


def _diagnostic_gradient(
    model: nn.Module,
    criterion: nn.Module,
    batch: tuple[Tensor, Tensor],
    device: torch.device,
) -> Tensor:
    model.eval()
    model.zero_grad(set_to_none=True)
    images, labels = batch
    loss = criterion(
        _forward_logits(model, images.to(device, non_blocking=True)),
        labels.to(device, non_blocking=True),
    )
    loss.backward()
    chunks = [
        parameter.grad.detach().float().flatten().cpu()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    model.zero_grad(set_to_none=True)
    if not chunks:
        raise RuntimeError("diagnostic gradient is empty")
    return torch.cat(chunks)


def _jensen_shannon(left: Tensor, right: Tensor) -> float:
    epsilon = torch.finfo(torch.float32).eps
    left = left.clamp_min(epsilon)
    right = right.clamp_min(epsilon)
    midpoint = 0.5 * (left + right)
    value = 0.5 * (
        (left * (left.log() - midpoint.log())).sum(dim=1)
        + (right * (right.log() - midpoint.log())).sum(dim=1)
    ).mean()
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite Jensen-Shannon divergence")
    return result


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _request_record(request: ReplayRequest) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(request).items()
    }


def _validate_existing(request: ReplayRequest) -> dict[str, object] | None:
    summary_path = request.output / "summary.json"
    manifest_path = request.output / "sha256_manifest.json"
    existing_files = list(request.output.iterdir()) if request.output.is_dir() else []
    if not summary_path.is_file() and not manifest_path.is_file():
        if existing_files:
            raise RuntimeError("partial replay bundle exists without a valid summary")
        return None
    if not summary_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("partial replay bundle exists without a valid summary")
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    if (
        summary.get("status") != "succeeded"
        or summary.get("source_commit") != request.source_commit
        or summary.get("request") != _request_record(request)
        or summary.get("test_loaded") is not False
    ):
        raise RuntimeError("completed replay summary does not match request")
    for name, expected in manifest.items():
        path = request.output / name
        if not isinstance(expected, str) or not path.is_file() or _sha256(path) != expected:
            raise RuntimeError("completed replay manifest validation failed")
    return summary


def run_replay(request: ReplayRequest) -> dict[str, object]:
    existing = _validate_existing(request)
    if existing is not None:
        return existing
    started = time.monotonic()
    config, input_binding = _validate_request(request)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    m0.seed_everything(request.seed)
    device = torch.device(request.device)
    provenance = m0.build_provenance(device)
    if provenance["source_commit"] != request.source_commit:
        raise RuntimeError("runtime provenance does not match replay request")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for evidence replay")
        torch.cuda.reset_peak_memory_stats(device)

    model = clean._build_model(config).to(device)
    optimizer = clean._build_optimizer(model, config)
    clean_payload = _load_payload(request.clean_checkpoint, device)
    noisy_payload = _load_payload(request.noisy_checkpoint, device)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    _compare_full_model_states(clean_payload, noisy_payload, trainable_names)
    clean_state = _state_from_payload(model, clean_payload)
    noisy_state = _state_from_payload(model, noisy_payload)
    validate_checkpoint_pair(clean_state, noisy_state)
    if clean_state.next_epoch != request.checkpoint_epoch:
        raise RuntimeError("checkpoint epoch does not match replay request")
    branches = build_factorial_branches(clean_state, noisy_state)
    cached = _cached_continuation(request, config, clean_state.next_epoch)
    probe, tuning_loader = _probe_and_tuning_loaders(request, config)
    criterion = nn.CrossEntropyLoss()
    common_rng = capture_common_rng()
    references: dict[int, tuple[m0_core.BranchState, Tensor, float, Tensor]] = {}

    def measure_branch(branch_name: str, horizon: int) -> dict[str, object]:
        state = m0_core.BranchState(
            m0_core.capture_trainable_state(model),
            m0_core.capture_optimizer_state(optimizer),
            0.0,
            0.0,
        )
        gradient = _diagnostic_gradient(model, criterion, probe[0], device)
        probe_loss, probabilities = _probe_metrics(model, criterion, probe, device)
        if branch_name == "CC":
            references[horizon] = (state, gradient, probe_loss, probabilities)
            d_theta = d_m = d_v = loss_excess = prediction_js = 0.0
            gradient_cosine = 1.0
        else:
            reference_state, reference_gradient, reference_loss, reference_probabilities = (
                references[horizon]
            )
            d_theta = m0_core.parameter_state_distance(
                state.parameters, reference_state.parameters
            )
            d_m = m0_core.optimizer_moment_distance(
                state.optimizer, reference_state.optimizer, "exp_avg"
            )
            d_v = m0_core.optimizer_moment_distance(
                state.optimizer, reference_state.optimizer, "exp_avg_sq"
            )
            gradient_cosine = m0.gradient_cosine(gradient, reference_gradient)
            loss_excess = probe_loss - reference_loss
            prediction_js = _jensen_shannon(probabilities, reference_probabilities)
        row: dict[str, object] = {
            "branch": branch_name,
            "horizon": horizon,
            "probe_loss": probe_loss,
            "clean_loss_excess": loss_excess,
            "d_theta": d_theta,
            "d_m": d_m,
            "d_v": d_v,
            "gradient_cosine_to_cc": gradient_cosine,
            "prediction_js_to_cc": prediction_js,
            "test_loaded": False,
        }
        if horizon == 512:
            tuning_loss, tuning_accuracy = clean._evaluate(model, tuning_loader, device)
            row["tuning_loss"] = tuning_loss
            row["tuning_accuracy"] = tuning_accuracy
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f"non-finite replay metric: {key}")
        return row

    rows = run_cached_branches(
        model, optimizer, branches, cached, criterion, common_rng, HORIZONS, measure_branch
    )
    by_branch = {
        branch: [row for row in rows if row["branch"] == branch] for branch in BRANCH_NAMES
    }
    endpoints: dict[str, dict[str, float]] = {}
    for branch, branch_rows in by_branch.items():
        clean_rows = [row for row in branch_rows if int(row["horizon"]) <= 128]
        endpoints[branch] = {
            "clean_loss_excess_auc_128": m0_core.trapezoid_auc(
                [int(row["horizon"]) for row in clean_rows],
                [float(row["clean_loss_excess"]) for row in clean_rows],
            ),
            "tuning_loss_h512": float(branch_rows[-1]["tuning_loss"]),
            "tuning_accuracy_h512": float(branch_rows[-1]["tuning_accuracy"]),
        }
    effects = {
        "tuning_loss_h512": factorial_effects(
            {branch: endpoints[branch]["tuning_loss_h512"] for branch in BRANCH_NAMES}
        ),
        "clean_loss_excess_auc_128": factorial_effects(
            {
                branch: endpoints[branch]["clean_loss_excess_auc_128"]
                for branch in BRANCH_NAMES
            }
        ),
    }
    elapsed = time.monotonic() - started
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    summary: dict[str, object] = {
        "schema": "causal-attribution-replay/1",
        "status": "succeeded",
        "request": _request_record(request),
        "source_commit": request.source_commit,
        "input_source_commit": input_binding.get(
            "input_source_commit", input_binding.get("source_commit")
        ),
        "uv_lock_sha256": provenance["uv_lock_sha256"],
        "endpoints": endpoints,
        "effects": effects,
        "wall_clock_seconds": elapsed,
        "peak_vram_gb": peak_vram,
        "test_loaded": False,
    }
    request.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(request.output / "trajectory_metrics.jsonl", rows)
    clean._write_json(request.output / "factorial_effects.json", effects)
    clean._write_json(
        request.output / "replay_manifest.json",
        {
            "source_commit": request.source_commit,
            "clean_checkpoint_sha256": _sha256(request.clean_checkpoint),
            "noisy_checkpoint_sha256": _sha256(request.noisy_checkpoint),
            "noisy_bundle_manifest_sha256": _sha256(request.noisy_bundle / "manifest.json"),
            "common_rng_sha256": m0._state_sha(common_rng),
            "horizons": list(HORIZONS),
            "scheduler": "none",
            "test_loaded": False,
        },
    )
    clean._write_json(request.output / "summary.json", summary)
    manifest = {
        path.name: _sha256(path)
        for path in sorted(request.output.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    clean._write_json(request.output / "sha256_manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--noisy-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation", choices=("clean", "noisy"), required=True)
    parser.add_argument("--seed", type=int, choices=(301, 302), required=True)
    parser.add_argument("--checkpoint-epoch", type=int, choices=(1, 2, 3, 20), required=True)
    parser.add_argument("--method", choices=("adamw", "cadam"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-store", type=Path, required=True)
    parser.add_argument("--tuning-store", type=Path, required=True)
    parser.add_argument("--noisy-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    request = ReplayRequest(
        clean_checkpoint=args.clean_checkpoint,
        noisy_checkpoint=args.noisy_checkpoint,
        continuation=args.continuation,
        seed=args.seed,
        checkpoint_epoch=args.checkpoint_epoch,
        method=args.method,
        data_dir=args.data_dir,
        image_store=args.image_store,
        tuning_store=args.tuning_store,
        noisy_bundle=args.noisy_bundle,
        output=args.output,
        source_commit=current_source_commit(),
        device=args.device,
    )
    print(json.dumps(run_replay(request), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
