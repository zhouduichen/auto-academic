"""Fail-closed GPU sentinel for the frozen Protect-M Pilot."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.datasets import CIFAR100

from experiments import m1_calibration_clean as clean
from experiments.m0_optimizer_state import m0_run as m0
from experiments.m1_optimizers import ProtectAdamW
from experiments.m1_pilot import source_commit

SENTINEL_SEED = 901


def _fixed_batch(data_dir: Path, device: torch.device) -> tuple[Tensor, Tensor]:
    dataset = CIFAR100(root=data_dir, train=True, download=False, transform=None)
    images = torch.stack([m0._transform(dataset[index][0], 4, 4, False) for index in range(32)])
    labels = torch.tensor([int(dataset[index][1]) for index in range(32)], dtype=torch.long)
    return images.to(device), labels.to(device)


def _config(method: str) -> clean.Config:
    return clean.Config(
        seed=SENTINEL_SEED,
        augmentation_seed=10_000 + SENTINEL_SEED,
        epochs=1,
        learning_rate=3e-4,
        weight_decay=0.1,
        workers=0,
        device="cuda",
        method=method,
    )


def _build(method: str, device: torch.device) -> tuple[nn.Module, torch.optim.Optimizer]:
    m0.seed_everything(SENTINEL_SEED)
    model = clean._build_model(_config(method)).to(device)
    optimizer = clean._build_optimizer(model, _config(method))
    return model, optimizer


def _training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: Tensor,
    labels: Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(m0.forward_logits(model, images), labels)
    loss.backward()
    optimizer.step()


def _tensor_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return torch.equal(left, right)
    return left == right


def _model_optimizer_equal(
    left_model: nn.Module,
    left_optimizer: torch.optim.Optimizer,
    right_model: nn.Module,
    right_optimizer: torch.optim.Optimizer,
) -> bool:
    left_parameters = list(left_model.named_parameters())
    right_parameters = list(right_model.named_parameters())
    if [name for name, _ in left_parameters] != [name for name, _ in right_parameters]:
        return False
    for (_, left_parameter), (_, right_parameter) in zip(
        left_parameters, right_parameters, strict=True
    ):
        if not torch.equal(left_parameter, right_parameter):
            return False
        left_state = left_optimizer.state.get(left_parameter, {})
        right_state = right_optimizer.state.get(right_parameter, {})
        if set(left_state) != set(right_state):
            return False
        for key in left_state:
            if not _tensor_equal(left_state[key], right_state[key]):
                return False
    return True


def _parity_check(images: Tensor, labels: Tensor) -> bool:
    reference_model, reference_optimizer = _build("adamw", images.device)
    m_model, m_optimizer = _build("protect-m", images.device)
    mv_model, mv_optimizer = _build("protect-mv", images.device)
    for optimizer in (m_optimizer, mv_optimizer):
        if not isinstance(optimizer, ProtectAdamW):
            raise TypeError("candidate optimizer construction returned wrong type")
        optimizer.protection_enabled = False
    parity = True
    for _ in range(5):
        _training_step(reference_model, reference_optimizer, images, labels)
        _training_step(m_model, m_optimizer, images, labels)
        _training_step(mv_model, mv_optimizer, images, labels)
        parity = parity and _model_optimizer_equal(
            reference_model, reference_optimizer, m_model, m_optimizer
        )
        parity = parity and _model_optimizer_equal(
            reference_model, reference_optimizer, mv_model, mv_optimizer
        )
    del reference_optimizer, reference_model, m_optimizer, m_model, mv_optimizer, mv_model
    torch.cuda.synchronize(images.device)
    torch.cuda.empty_cache()
    return parity


def _functional_checks(images: Tensor, labels: Tensor) -> dict[str, object]:
    parity = _parity_check(images, labels)

    parameter = nn.Parameter(torch.tensor([1.0], device=images.device))
    optimizer = ProtectAdamW([parameter], protection_target="m")
    states = [optimizer.observe_score_for_test(score) for score in [-1.0] * 10 + [1.0] * 36]
    reachability = states[1]["admit"] is False and states[45]["admit"] is True

    parameter.grad = torch.tensor([0.5], device=images.device)
    optimizer.step()
    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    resumed_parameter = nn.Parameter(saved_parameter.clone())
    resumed = ProtectAdamW([resumed_parameter], protection_target="m")
    resumed.load_state_dict(saved_optimizer)
    for gradient in (-0.4, 0.3):
        parameter.grad = torch.tensor([gradient], device=images.device)
        resumed_parameter.grad = torch.tensor([gradient], device=images.device)
        optimizer.step()
        resumed.step()
    resume = torch.equal(parameter, resumed_parameter) and (
        optimizer.protect_state_dict() == resumed.protect_state_dict()
    )

    before_parameter = parameter.detach().clone()
    before_state = copy.deepcopy(optimizer.state_dict())
    parameter.grad = torch.tensor([float("nan")], device=images.device)
    nonfinite_rejected = False
    try:
        optimizer.step()
    except FloatingPointError:
        nonfinite_rejected = torch.equal(parameter, before_parameter) and (
            optimizer.protect_state_dict() == before_state["protect_state"]
        )
    return {
        "disabled_protection_parity": parity,
        "reachability": reachability,
        "resume": resume,
        "nonfinite_rejected_atomically": nonfinite_rejected,
        "passed": parity and reachability and resume and nonfinite_rejected,
    }


def _time_method(method: str, images: Tensor, labels: Tensor) -> dict[str, object]:
    model, optimizer = _build(method, images.device)
    torch.cuda.reset_peak_memory_stats(images.device)
    for _ in range(10):
        _training_step(model, optimizer, images, labels)
    timings = []
    for _ in range(5):
        torch.cuda.synchronize(images.device)
        started = time.perf_counter()
        for _ in range(50):
            _training_step(model, optimizer, images, labels)
        torch.cuda.synchronize(images.device)
        timings.append((time.perf_counter() - started) / 50)
    peak = torch.cuda.max_memory_allocated(images.device) / 1024**3
    result = {
        "median_step_seconds": statistics.median(timings),
        "repeats": timings,
        "peak_vram_gb": peak,
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the candidate sentinel")
    device = torch.device("cuda")
    images, labels = _fixed_batch(args.data_dir, device)
    functional = _functional_checks(images, labels)
    timings = {
        method: _time_method(method, images, labels)
        for method in ("adamw", "protect-m", "protect-mv")
    }
    adamw_seconds = float(timings["adamw"]["median_step_seconds"])
    ratios = {
        method: float(timings[method]["median_step_seconds"]) / adamw_seconds
        for method in ("protect-m", "protect-mv")
    }
    passed = bool(functional["passed"]) and all(ratio <= 1.05 for ratio in ratios.values())
    result = {
        "status": "passed" if passed else "failed",
        "source_commit": source_commit(),
        "functional": functional,
        "timings": timings,
        "candidate_ratios": ratios,
        "test_loaded": False,
    }
    clean._write_json(args.output / "SENTINEL_RESULT.json", result)
    if passed:
        clean._write_json(args.output / "SENTINEL_PASS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
