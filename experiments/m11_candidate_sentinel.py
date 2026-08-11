"""Fail-closed CUDA sentinel for the bounded M1.1 rescue candidate."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from experiments import m1_calibration_clean as clean
from experiments.m1_candidate_sentinel import (
    _build,
    _fixed_batch,
    _model_optimizer_equal,
    _time_method,
    _training_step,
)
from experiments.m1_optimizers import ProtectM11AdamW
from experiments.m1_pilot import source_commit


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _disabled_protection_parity(images: Tensor, labels: Tensor) -> bool:
    reference_model, reference_optimizer = _build("adamw", images.device)
    candidate_model, candidate_optimizer = _build("protect-m11", images.device)
    if not isinstance(candidate_optimizer, ProtectM11AdamW):
        raise TypeError("candidate optimizer construction returned wrong type")
    candidate_optimizer.protection_enabled = False
    parity = True
    for _ in range(5):
        _training_step(reference_model, reference_optimizer, images, labels)
        _training_step(candidate_model, candidate_optimizer, images, labels)
        parity = parity and _model_optimizer_equal(
            reference_model,
            reference_optimizer,
            candidate_model,
            candidate_optimizer,
        )
    del reference_optimizer, reference_model, candidate_optimizer, candidate_model
    torch.cuda.synchronize(images.device)
    torch.cuda.empty_cache()
    return parity


def _functional_checks(
    images: Tensor, labels: Tensor, *, warmup_steps: int = 2
) -> dict[str, object]:
    parity = _disabled_protection_parity(images, labels)

    anomaly_parameter = nn.Parameter(torch.tensor([1.0], device=images.device))
    anomaly_optimizer = ProtectM11AdamW(
        [anomaly_parameter], warmup_steps=warmup_steps
    )
    anomaly_optimizer.set_detector_state_for_test(
        mu=0.0,
        scale=0.1,
        previous_rejected=False,
        steps=warmup_steps,
    )
    anomaly_states = [
        anomaly_optimizer.observe_score_for_test(-1.0),
        anomaly_optimizer.observe_score_for_test(-1.0),
    ]
    isolated_rejection = (
        anomaly_states[0]["reject"] is True
        and anomaly_states[1]["reject"] is False
    )

    parameter = nn.Parameter(torch.tensor([1.0], device=images.device))
    optimizer = ProtectM11AdamW([parameter], warmup_steps=warmup_steps)
    parameter.grad = torch.tensor([0.5], device=images.device)
    optimizer.step()
    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    resumed_parameter = nn.Parameter(saved_parameter.clone())
    resumed = ProtectM11AdamW([resumed_parameter], warmup_steps=warmup_steps)
    resumed.load_state_dict(saved_optimizer)
    resume = _nested_equal(optimizer.state_dict(), resumed.state_dict())
    for gradient in (-0.4, 0.3):
        parameter.grad = torch.tensor([gradient], device=images.device)
        resumed_parameter.grad = torch.tensor([gradient], device=images.device)
        optimizer.step()
        resumed.step()
    deterministic_replay = torch.equal(parameter, resumed_parameter) and _nested_equal(
        optimizer.state_dict(), resumed.state_dict()
    )

    before_parameter = parameter.detach().clone()
    before_state = copy.deepcopy(optimizer.state_dict())
    parameter.grad = torch.tensor([float("nan")], device=images.device)
    nonfinite_rejected = False
    try:
        optimizer.step()
    except FloatingPointError:
        nonfinite_rejected = torch.equal(parameter, before_parameter) and _nested_equal(
            optimizer.state_dict(), before_state
        )

    checks = {
        "disabled_protection_parity": parity,
        "isolated_rejection": isolated_rejection,
        "resume": resume,
        "deterministic_replay": deterministic_replay,
        "nonfinite_rejected_atomically": nonfinite_rejected,
    }
    return {**checks, "passed": all(checks.values())}


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    try:
        top = float(numerator)
        bottom = float(denominator)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(top) or not math.isfinite(bottom) or top < 0 or bottom <= 0:
        return None
    return top / bottom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pass_path = args.output / "M11_SENTINEL_PASS.json"
    pass_path.unlink(missing_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M1.1 candidate sentinel")
    device = torch.device("cuda")
    images, labels = _fixed_batch(args.data_dir, device)
    functional = _functional_checks(images, labels, warmup_steps=2)
    timings = {
        method: _time_method(method, images, labels)
        for method in ("adamw", "protect-m11")
    }
    time_ratio = _safe_ratio(
        timings["protect-m11"].get("median_step_seconds"),
        timings["adamw"].get("median_step_seconds"),
    )
    vram_ratio = _safe_ratio(
        timings["protect-m11"].get("peak_vram_gb"),
        timings["adamw"].get("peak_vram_gb"),
    )
    resources_pass = (
        time_ratio is not None
        and vram_ratio is not None
        and time_ratio > 0
        and vram_ratio > 0
        and time_ratio <= 1.05
        and vram_ratio <= 1.05
    )
    passed = bool(functional["passed"]) and resources_pass
    result: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "source_commit": source_commit(),
        "functional": functional,
        "timings": timings,
        "time_ratio": time_ratio,
        "vram_ratio": vram_ratio,
        "test_loaded": False,
    }
    clean._write_json(args.output / "M11_SENTINEL_RESULT.json", result)
    if passed:
        clean._write_json(pass_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
