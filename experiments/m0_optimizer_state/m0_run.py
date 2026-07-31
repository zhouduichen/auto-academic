"""M0/M0.5: causal audit of persistent optimizer-state contamination.

One invocation runs one optimizer x pulse x seed bundle and produces four
paired trajectories from a shared checkpoint. CIFAR-100 test data is never
constructed or loaded.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from torchvision.datasets import CIFAR100
from torchvision.transforms import functional as vision
from transformers import ViTForImageClassification, ViTModel

from arw.m0_core import (
    BranchState,
    build_causal_branches,
    build_state_carrier_branches,
    capture_optimizer_state,
    capture_trainable_state,
    optimizer_moment_distance,
    parameter_state_distance,
    recovery_half_life,
    restore_branch,
    take_candidate_step,
    trapezoid_auc,
)

HORIZONS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
BRANCHES = ("control", "parameter_only", "state_only", "full")
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RunConfig:
    optimizer: str
    pulse: str
    seed: int
    split_seed: int
    pulse_seed: int
    warmup_steps: int
    replay_steps: int
    batch_size: int
    probe_size: int
    learning_rate: float
    weight_decay: float
    model_id: str
    lora_rank: int
    device: str
    state_attribution: bool = False
    replay_seed: int | None = None
    probe_seed: int | None = None


@dataclass(frozen=True)
class BatchPlan:
    indices: tuple[int, ...]
    crop_top: tuple[int, ...]
    crop_left: tuple[int, ...]
    horizontal_flip: tuple[bool, ...]


RngState = dict[str, object]


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _state_sha(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return sha256(buffer.getvalue()).hexdigest()


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_provenance(device: torch.device) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    git_executable = shutil.which("git")
    try:
        if git_executable is None:
            raise OSError("git executable is unavailable")
        source_commit = subprocess.run(  # noqa: S603 - fixed git arguments
            [git_executable, "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_dirty = bool(
            subprocess.run(  # noqa: S603 - fixed git arguments
                [git_executable, "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        if device.type == "cuda":
            raise RuntimeError("source provenance is unavailable") from error
        source_commit = "unavailable"
        source_dirty = True
    lock_path = repo_root / "uv.lock"
    if not lock_path.is_file():
        if device.type == "cuda":
            raise RuntimeError("uv.lock is unavailable")
        lock_sha256 = "unavailable"
    else:
        lock_sha256 = _file_sha(lock_path)
    if device.type == "cuda" and source_dirty:
        raise RuntimeError("CUDA evidence runs require a clean tracked worktree")
    return {
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "uv_lock_sha256": lock_sha256,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _optimizer_projection(optimizer: dict[str, object], key: str) -> dict[str, object]:
    raw_state = optimizer.get("state")
    if not isinstance(raw_state, dict):
        raise ValueError("optimizer state is malformed")
    projection: dict[str, object] = {}
    for parameter_id in sorted(raw_state, key=str):
        parameter_state = raw_state[parameter_id]
        if not isinstance(parameter_state, dict) or key not in parameter_state:
            raise ValueError(f"optimizer state is missing {key}")
        projection[str(parameter_id)] = parameter_state[key]
    return projection


def build_branch_construction_manifest(
    branches: dict[str, BranchState], branch_rng: RngState
) -> dict[str, object]:
    rng_sha256 = _state_sha(branch_rng)
    return {
        branch_name: {
            "parameters_sha256": _state_sha(branch.parameters),
            "exp_avg_sha256": _state_sha(_optimizer_projection(branch.optimizer, "exp_avg")),
            "exp_avg_sq_sha256": _state_sha(_optimizer_projection(branch.optimizer, "exp_avg_sq")),
            "optimizer_step_sha256": _state_sha(_optimizer_projection(branch.optimizer, "step")),
            "rng_sha256": rng_sha256,
        }
        for branch_name, branch in branches.items()
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def capture_rng_state() -> RngState:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: RngState) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def stratified_split(labels: list[int], split_seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(split_seed)
    train: list[int] = []
    validation: list[int] = []
    label_array = np.asarray(labels)
    for class_id in range(100):
        class_indices = np.flatnonzero(label_array == class_id)
        rng.shuffle(class_indices)
        train.extend(class_indices[:450].tolist())
        validation.extend(class_indices[450:].tolist())
    rng.shuffle(train)
    rng.shuffle(validation)
    if len(train) != 45_000 or len(validation) != 5_000:
        raise RuntimeError("unexpected CIFAR-100 stratified split size")
    return train, validation


def build_batch_plans(
    train_indices: list[int], count: int, batch_size: int, seed: int
) -> list[BatchPlan]:
    rng = np.random.default_rng(seed)
    plans: list[BatchPlan] = []
    for _ in range(count):
        indices = tuple(
            int(value) for value in rng.choice(train_indices, batch_size, replace=False)
        )
        plans.append(
            BatchPlan(
                indices=indices,
                crop_top=tuple(int(value) for value in rng.integers(0, 9, batch_size)),
                crop_left=tuple(int(value) for value in rng.integers(0, 9, batch_size)),
                horizontal_flip=tuple(bool(value) for value in rng.integers(0, 2, batch_size)),
            )
        )
    return plans


def _transform(image: Any, top: int, left: int, flip: bool) -> Tensor:
    tensor = vision.pil_to_tensor(image).float().div_(255)
    tensor = vision.pad(tensor, [4, 4, 4, 4], padding_mode="reflect")
    tensor = vision.crop(tensor, top, left, 32, 32)
    if flip:
        tensor = vision.hflip(tensor)
    tensor = vision.resize(tensor, [224, 224], antialias=True)
    return vision.normalize(tensor, NORMALIZE_MEAN, NORMALIZE_STD)


def materialize_batch(
    dataset: CIFAR100, plan: BatchPlan, device: torch.device
) -> tuple[Tensor, Tensor]:
    images: list[Tensor] = []
    labels: list[int] = []
    for index, top, left, flip in zip(
        plan.indices,
        plan.crop_top,
        plan.crop_left,
        plan.horizontal_flip,
        strict=True,
    ):
        image, label = dataset[index]
        images.append(_transform(image, top, left, flip))
        labels.append(int(label))
    return (
        torch.stack(images).to(device, non_blocking=True),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def materialize_batch_cpu(dataset: CIFAR100, plan: BatchPlan) -> tuple[Tensor, Tensor]:
    return materialize_batch(dataset, plan, torch.device("cpu"))


def materialize_probe(
    dataset: CIFAR100, indices: list[int], batch_size: int, device: torch.device
) -> list[tuple[Tensor, Tensor]]:
    batches: list[tuple[Tensor, Tensor]] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        plan = BatchPlan(
            indices=tuple(batch_indices),
            crop_top=(4,) * len(batch_indices),
            crop_left=(4,) * len(batch_indices),
            horizontal_flip=(False,) * len(batch_indices),
        )
        batches.append(materialize_batch(dataset, plan, device))
    return batches


def build_model(config: RunConfig) -> nn.Module:
    backbone, loading_info = ViTModel.from_pretrained(
        config.model_id,
        output_loading_info=True,
    )
    anomaly_fields = (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    )
    anomalies = {
        field: loading_info.get(field, []) for field in anomaly_fields if loading_info.get(field)
    }
    if anomalies:
        raise RuntimeError(f"ViT backbone load audit failed: {anomalies}")

    backbone.config.num_labels = 100
    base = ViTForImageClassification(backbone.config)
    base.vit = backbone
    lora = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.lora_rank,
        lora_alpha=config.lora_rank * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["classifier"],
    )
    return get_peft_model(base, lora)


def build_optimizer(
    model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, momentum=0.0, weight_decay=weight_decay
        )
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def forward_logits(model: nn.Module, images: Tensor) -> Tensor:
    output = model(pixel_values=images)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("ViT output does not expose logits")
    return logits


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    images: Tensor,
    labels: Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(forward_logits(model, images), labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def evaluate_loss(
    model: nn.Module, criterion: nn.Module, probe: list[tuple[Tensor, Tensor]]
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for images, labels in probe:
            loss = criterion(forward_logits(model, images), labels)
            total_loss += float(loss) * len(labels)
            total_examples += len(labels)
    return total_loss / total_examples


def diagnostic_gradient(
    model: nn.Module, criterion: nn.Module, batch: tuple[Tensor, Tensor]
) -> Tensor:
    model.eval()
    model.zero_grad(set_to_none=True)
    images, labels = batch
    loss = criterion(forward_logits(model, images), labels)
    loss.backward()
    chunks = [
        parameter.grad.detach().float().flatten().cpu()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    model.zero_grad(set_to_none=True)
    return torch.cat(chunks)


def gradient_cosine(left: Tensor, right: Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    if denominator == 0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def corrupt_pulse(
    pulse: str, images: Tensor, labels: Tensor, pulse_seed: int
) -> tuple[Tensor, Tensor]:
    if pulse == "label_flip":
        offset = pulse_seed % 99 + 1
        return images, (labels + offset) % 100
    if pulse == "input_degradation":
        generator = torch.Generator(device=images.device)
        generator.manual_seed(pulse_seed)
        noise = torch.randn(
            images.shape, generator=generator, device=images.device, dtype=images.dtype
        )
        degraded = vision.gaussian_blur(images, kernel_size=[5, 5], sigma=[2.0, 2.0])
        degraded = degraded + 0.25 * noise
        return degraded, labels
    raise ValueError(f"unsupported pulse: {pulse}")


def snapshot_branch(
    model: nn.Module, optimizer: torch.optim.Optimizer, template: BranchState
) -> BranchState:
    return BranchState(
        parameters=capture_trainable_state(model),
        optimizer=capture_optimizer_state(optimizer),
        pulse_loss=template.pulse_loss,
        gradient_norm=template.gradient_norm,
    )


def run_trajectory(
    branch_name: str,
    initial: BranchState,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataset: CIFAR100,
    replay_plans: list[BatchPlan],
    probe: list[tuple[Tensor, Tensor]],
    device: torch.device,
    control_references: dict[int, tuple[BranchState, Tensor]] | None,
    control_losses: dict[int, float] | None,
    branch_rng: RngState,
    replay_batches_cpu: list[tuple[Tensor, Tensor]] | None = None,
) -> tuple[list[dict[str, object]], dict[int, tuple[BranchState, Tensor]]]:
    restore_branch(model, optimizer, initial)
    restore_rng_state(branch_rng)
    references: dict[int, tuple[BranchState, Tensor]] = {}
    rows: list[dict[str, object]] = []
    horizons = set(HORIZONS)
    for horizon in range(len(replay_plans) + 1):
        if horizon in horizons:
            state = snapshot_branch(model, optimizer, initial)
            gradient = diagnostic_gradient(model, criterion, probe[0])
            clean_loss = evaluate_loss(model, criterion, probe)
            if control_references is None:
                references[horizon] = (state, gradient)
                d_theta = 0.0
                d_m = 0.0
                d_v = 0.0
                cosine = 1.0
                loss_excess = 0.0
            else:
                if control_losses is None:
                    raise RuntimeError("control losses are required for a non-control trajectory")
                control_state, control_gradient = control_references[horizon]
                d_theta = parameter_state_distance(state.parameters, control_state.parameters)
                d_m = optimizer_moment_distance(state.optimizer, control_state.optimizer, "exp_avg")
                d_v = optimizer_moment_distance(
                    state.optimizer, control_state.optimizer, "exp_avg_sq"
                )
                cosine = gradient_cosine(gradient, control_gradient)
                loss_excess = clean_loss - control_losses[horizon]
            rows.append(
                {
                    "branch": branch_name,
                    "horizon": horizon,
                    "clean_loss": clean_loss,
                    "clean_loss_excess": loss_excess,
                    "d_theta": d_theta,
                    "d_m": d_m,
                    "d_v": d_v,
                    "gradient_cosine_to_control": cosine,
                }
            )
        if horizon < len(replay_plans):
            if replay_batches_cpu is None:
                images, labels = materialize_batch(dataset, replay_plans[horizon], device)
            else:
                cpu_images, cpu_labels = replay_batches_cpu[horizon]
                images = cpu_images.to(device, non_blocking=True)
                labels = cpu_labels.to(device, non_blocking=True)
            training_step(model, optimizer, criterion, images, labels)
    return rows, references


def write_artifacts(
    output_dir: Path,
    config: RunConfig,
    train_indices: list[int],
    validation_indices: list[int],
    probe_indices: list[int],
    warmup_plans: list[BatchPlan],
    pulse_plan: BatchPlan,
    replay_plans: list[BatchPlan],
    checkpoint_manifest: dict[str, object],
    branch_construction_manifest: dict[str, object] | None,
    provenance: dict[str, object],
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(_stable_json(asdict(config)), encoding="utf-8")
    (output_dir / "split_ids.json").write_text(
        _stable_json(
            {
                "train": train_indices,
                "validation": validation_indices,
                "probe": probe_indices,
                "test_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    schedules = {
        "warmup": [asdict(plan) for plan in warmup_plans],
        "pulse": asdict(pulse_plan),
        "replay": [asdict(plan) for plan in replay_plans],
    }
    (output_dir / "replay_manifest.json").write_text(
        _stable_json({"sha256": _sha_json(schedules), **schedules}), encoding="utf-8"
    )
    (output_dir / "checkpoint_manifest.json").write_text(
        _stable_json(checkpoint_manifest), encoding="utf-8"
    )
    if branch_construction_manifest is not None:
        (output_dir / "branch_construction_manifest.json").write_text(
            _stable_json(branch_construction_manifest), encoding="utf-8"
        )
    (output_dir / "provenance.json").write_text(_stable_json(provenance), encoding="utf-8")
    (output_dir / "trajectory_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(_stable_json(summary), encoding="utf-8")

    manifest = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "sha256_manifest.json":
            payload = path.read_bytes()
            manifest.append(
                {
                    "filename": path.name,
                    "byte_size": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                }
            )
    (output_dir / "sha256_manifest.json").write_text(_stable_json(manifest), encoding="utf-8")


def run(config: RunConfig, data_dir: Path, output_dir: Path) -> dict[str, object]:
    started = time.time()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed_everything(config.seed)
    if config.state_attribution and (config.optimizer != "adamw" or config.pulse != "label_flip"):
        raise ValueError("state attribution requires AdamW with label_flip")
    device = torch.device(config.device)
    provenance = build_provenance(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset = CIFAR100(root=data_dir, train=True, download=True, transform=None)
    train_indices, validation_indices = stratified_split(dataset.targets, config.split_seed)
    effective_probe_seed = (
        config.probe_seed if config.probe_seed is not None else config.split_seed + config.seed
    )
    probe_rng = np.random.default_rng(effective_probe_seed)
    probe_indices = probe_rng.choice(validation_indices, config.probe_size, replace=False).tolist()
    if config.replay_seed is None:
        plan_count = config.warmup_steps + 1 + config.replay_steps
        plans = build_batch_plans(train_indices, plan_count, config.batch_size, config.seed)
        warmup_plans = plans[: config.warmup_steps]
        pulse_plan = plans[config.warmup_steps]
        replay_plans = plans[config.warmup_steps + 1 :]
    else:
        pre_replay_plans = build_batch_plans(
            train_indices, config.warmup_steps + 1, config.batch_size, config.seed
        )
        warmup_plans = pre_replay_plans[: config.warmup_steps]
        pulse_plan = pre_replay_plans[config.warmup_steps]
        replay_plans = build_batch_plans(
            train_indices, config.replay_steps, config.batch_size, config.replay_seed
        )
    if len(replay_plans) != config.replay_steps:
        raise RuntimeError("replay schedule length mismatch")

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    probe = materialize_probe(dataset, probe_indices, config.batch_size, device)

    for step, plan in enumerate(warmup_plans, start=1):
        images, labels = materialize_batch(dataset, plan, device)
        loss = training_step(model, optimizer, criterion, images, labels)
        if step % 50 == 0 or step == config.warmup_steps:
            print(f"warmup step={step}/{config.warmup_steps} loss={loss:.6f}", flush=True)

    pre_parameters = capture_trainable_state(model)
    pre_optimizer = capture_optimizer_state(optimizer)
    clean_images, clean_labels = materialize_batch(dataset, pulse_plan, device)
    corrupt_images, corrupt_labels = corrupt_pulse(
        config.pulse, clean_images, clean_labels, config.pulse_seed
    )
    model.train()
    pre_candidate_rng = capture_rng_state()
    restore_rng_state(pre_candidate_rng)
    clean = take_candidate_step(
        model,
        optimizer,
        criterion,
        clean_images,
        clean_labels,
        pre_parameters,
        pre_optimizer,
        forward_fn=lambda images: model(pixel_values=images),
    )
    clean_post_rng = capture_rng_state()
    restore_rng_state(pre_candidate_rng)
    corrupt = take_candidate_step(
        model,
        optimizer,
        criterion,
        corrupt_images,
        corrupt_labels,
        pre_parameters,
        pre_optimizer,
        target_gradient_norm=clean.gradient_norm,
        forward_fn=lambda images: model(pixel_values=images),
    )
    corrupt_post_rng = capture_rng_state()
    if _state_sha(clean_post_rng) != _state_sha(corrupt_post_rng):
        raise RuntimeError("clean/corrupt candidate steps consumed different RNG state")
    branches = (
        build_state_carrier_branches(clean, corrupt)
        if config.state_attribution
        else build_causal_branches(clean, corrupt)
    )
    branch_construction_manifest = (
        build_branch_construction_manifest(branches, clean_post_rng)
        if config.state_attribution
        else None
    )
    branch_names = tuple(branches)
    replay_batches_cpu = None
    if config.state_attribution:
        cache_started = time.time()
        replay_batches_cpu = [materialize_batch_cpu(dataset, plan) for plan in replay_plans]
        print(
            f"replay cache complete batches={len(replay_batches_cpu)} "
            f"seconds={time.time() - cache_started:.2f}",
            flush=True,
        )
    checkpoint_manifest = {
        "pre_parameters_sha256": _state_sha(pre_parameters),
        "pre_optimizer_sha256": _state_sha(pre_optimizer),
        "clean_candidate_sha256": _state_sha(clean),
        "corrupt_candidate_sha256": _state_sha(corrupt),
        "pre_candidate_rng_sha256": _state_sha(pre_candidate_rng),
        "post_candidate_rng_sha256": _state_sha(clean_post_rng),
        "clean_gradient_norm": clean.gradient_norm,
        "corrupt_gradient_norm_after_matching": corrupt.gradient_norm,
        "norm_ratio": corrupt.gradient_norm / clean.gradient_norm,
    }

    rows_by_branch: dict[str, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []
    control_references: dict[int, tuple[BranchState, Tensor]] | None = None
    control_losses: dict[int, float] | None = None
    for branch_name in branch_names:
        branch_rows, references = run_trajectory(
            branch_name,
            branches[branch_name],
            model,
            optimizer,
            criterion,
            dataset,
            replay_plans,
            probe,
            device,
            control_references,
            control_losses,
            clean_post_rng,
            replay_batches_cpu,
        )
        rows_by_branch[branch_name] = branch_rows
        all_rows.extend(branch_rows)
        if branch_name == "control":
            control_references = references
            control_losses = {int(row["horizon"]): float(row["clean_loss"]) for row in branch_rows}
        print(f"trajectory={branch_name} complete", flush=True)

    branch_summary: dict[str, object] = {}
    for branch_name in branch_names:
        branch_rows = rows_by_branch[branch_name]
        horizons = [int(row["horizon"]) for row in branch_rows]
        excess = [float(row["clean_loss_excess"]) for row in branch_rows]
        branch_summary[branch_name] = {
            "clean_loss_excess_auc_128": trapezoid_auc(horizons, excess),
            "recovery_half_life": recovery_half_life(horizons, excess),
            "final_clean_loss_excess": excess[-1],
            "max_abs_clean_loss_excess": max(abs(value) for value in excess),
        }

    if config.optimizer == "sgd":
        state_only = branch_summary["state_only"]
        if (
            not isinstance(state_only, dict)
            or float(state_only["max_abs_clean_loss_excess"]) > 1e-10
        ):
            raise RuntimeError("SGD no-momentum State-only negative control failed")

    elapsed = time.time() - started
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    primary_branch = "state_both" if config.state_attribution else "state_only"
    summary = {
        "status": "succeeded",
        "bundle_id": (
            f"m06-{config.optimizer}-{config.pulse}-train{config.seed}-pulse{config.pulse_seed}"
            if config.state_attribution
            and config.replay_seed is not None
            and config.probe_seed is not None
            else (
                f"m05-{config.optimizer}-{config.pulse}-seed{config.seed}"
                if config.state_attribution
                else f"{config.optimizer}-{config.pulse}-seed{config.seed}"
            )
        ),
        "primary_endpoint": branch_summary[primary_branch],
        "branches": branch_summary,
        "wall_clock_seconds": elapsed,
        "peak_vram_gb": peak_vram,
        "train_steps": config.warmup_steps + 2 + len(branch_names) * config.replay_steps,
        "test_loaded": False,
        "source_commit": provenance["source_commit"],
    }
    write_artifacts(
        output_dir,
        config,
        train_indices,
        validation_indices,
        probe_indices,
        warmup_plans,
        pulse_plan,
        replay_plans,
        checkpoint_manifest,
        branch_construction_manifest,
        provenance,
        all_rows,
        summary,
    )
    return summary


def parse_args() -> tuple[RunConfig, Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), required=True)
    parser.add_argument("--pulse", choices=("label_flip", "input_degradation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=20_260_731)
    parser.add_argument("--pulse-seed", type=int, default=314_159)
    parser.add_argument("--replay-seed", type=int)
    parser.add_argument("--probe-seed", type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/cifar100"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--replay-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-attribution", action="store_true")
    args = parser.parse_args()
    seed_values = (args.seed, args.split_seed, args.pulse_seed)
    if args.replay_seed is not None:
        seed_values += (args.replay_seed,)
    if args.probe_seed is not None:
        seed_values += (args.probe_seed,)
    if any(value < 0 for value in seed_values):
        parser.error("all seeds must be nonnegative")
    learning_rate = 3e-4 if args.optimizer == "adamw" else 1e-2
    config = RunConfig(
        optimizer=args.optimizer,
        pulse=args.pulse,
        seed=args.seed,
        split_seed=args.split_seed,
        pulse_seed=args.pulse_seed,
        warmup_steps=args.warmup_steps,
        replay_steps=args.replay_steps,
        batch_size=args.batch_size,
        probe_size=args.probe_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        model_id="google/vit-base-patch16-224-in21k",
        lora_rank=8,
        device=args.device,
        state_attribution=args.state_attribution,
        replay_seed=args.replay_seed,
        probe_seed=args.probe_seed,
    )
    return config, args.data_dir, args.output_dir


def main() -> None:
    config, data_dir, output_dir = parse_args()
    summary = run(config, data_dir, output_dir)
    print(_stable_json(summary), end="", flush=True)


if __name__ == "__main__":
    main()
