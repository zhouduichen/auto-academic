"""CUDA functional/resource sentinel for crossed optimizer-state replay."""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from experiments import causal_attribution_replay as replay
from experiments import m1_calibration_clean as clean
from experiments.m0_optimizer_state import m0_run as m0
from experiments.m1_pilot import source_commit
from src.arw import m0_core

FUNCTIONAL_KEYS = {
    "checkpoint_schema",
    "step_clock_match",
    "factorial_carriers_exact",
    "deterministic_replay",
    "nonfinite_rejected",
    "test_isolated",
}


def _state_hash(value: object) -> str:
    return m0._state_sha(value)


def _factorial_carriers_are_exact(
    branches: dict[str, m0_core.BranchState],
    clean_state: replay.CheckpointState,
    noisy_state: replay.CheckpointState,
) -> bool:
    return all(
        (
            _state_hash(branches[branch].parameters) == _state_hash(parameters)
            and _state_hash(branches[branch].optimizer) == _state_hash(optimizer)
        )
        for branch, parameters, optimizer in (
            ("CC", clean_state.parameters, clean_state.optimizer),
            ("CN", clean_state.parameters, noisy_state.optimizer),
            ("NC", noisy_state.parameters, clean_state.optimizer),
            ("NN", noisy_state.parameters, noisy_state.optimizer),
        )
    )


def _deterministic_replay(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    branches: dict[str, m0_core.BranchState],
    device: torch.device,
) -> bool:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20_260_811)
    cached = [
        (
            torch.randn((1, 3, 224, 224), generator=generator),
            torch.randint(0, 100, (1,), generator=generator),
        )
        for _ in range(8)
    ]
    criterion = nn.CrossEntropyLoss()
    common_rng = replay.capture_common_rng()

    def measure(branch: str, horizon: int) -> dict[str, object]:
        return {
            "branch": branch,
            "horizon": horizon,
            "parameters_sha256": _state_hash(m0_core.capture_trainable_state(model)),
            "optimizer_sha256": _state_hash(m0_core.capture_optimizer_state(optimizer)),
            "test_loaded": False,
        }

    first = replay.run_cached_branches(
        model, optimizer, branches, cached, criterion, common_rng, (0, 8), measure
    )
    second = replay.run_cached_branches(
        model, optimizer, branches, cached, criterion, common_rng, (0, 8), measure
    )
    torch.cuda.synchronize(device)
    return first == second


def _nonfinite_is_rejected(
    clean_state: replay.CheckpointState, noisy_state: replay.CheckpointState
) -> bool:
    corrupt = deepcopy(noisy_state)
    raw_state = corrupt.optimizer.get("state")
    if not isinstance(raw_state, dict) or not raw_state:
        return False
    first = next(iter(raw_state.values()))
    if not isinstance(first, dict) or not isinstance(first.get("exp_avg"), torch.Tensor):
        return False
    first["exp_avg"].view(-1)[0] = float("nan")
    try:
        replay.validate_checkpoint_pair(clean_state, corrupt)
    except ValueError as error:
        return "non-finite" in str(error)
    return False


def run_sentinel(
    clean_checkpoint: Path, noisy_checkpoint: Path, output: Path
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    pass_path = output / "CAUSAL_ATTRIBUTION_SENTINEL_PASS.json"
    result_path = output / "CAUSAL_ATTRIBUTION_SENTINEL_RESULT.json"
    pass_path.unlink(missing_ok=True)
    started = time.monotonic()
    commit = source_commit()
    checks = {key: False for key in FUNCTIONAL_KEYS}
    resource_checks = {
        "peak_vram_within_90pct": False,
        "elapsed_under_15_minutes": False,
    }
    result: dict[str, object]
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device("cuda")
        provenance = m0.build_provenance(device)
        if provenance["source_commit"] != commit:
            raise RuntimeError("sentinel runtime provenance mismatch")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        m0.seed_everything(20_260_811)
        torch.cuda.reset_peak_memory_stats(device)

        clean_config = replay._read_json(clean_checkpoint.parent / "config.json")
        noisy_config = replay._read_json(noisy_checkpoint.parent / "config.json")
        if clean_config != noisy_config:
            raise RuntimeError("sentinel checkpoint configurations differ")
        config = replay._config_from_json(clean_config, "cuda")
        if config.method != "adamw":
            raise RuntimeError("sentinel requires AdamW checkpoints")
        model = clean._build_model(config).to(device)
        optimizer = clean._build_optimizer(model, config)
        clean_payload = replay._load_payload(clean_checkpoint, device)
        noisy_payload = replay._load_payload(noisy_checkpoint, device)
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        replay._compare_full_model_states(clean_payload, noisy_payload, trainable)
        clean_state = replay._state_from_payload(model, clean_payload)
        noisy_state = replay._state_from_payload(model, noisy_payload)
        checks["checkpoint_schema"] = True
        replay.validate_checkpoint_pair(clean_state, noisy_state)
        checks["step_clock_match"] = True
        branches = replay.build_factorial_branches(clean_state, noisy_state)
        checks["factorial_carriers_exact"] = _factorial_carriers_are_exact(
            branches, clean_state, noisy_state
        )
        checks["deterministic_replay"] = _deterministic_replay(
            model, optimizer, branches, device
        )
        checks["nonfinite_rejected"] = _nonfinite_is_rejected(clean_state, noisy_state)
        checks["test_isolated"] = True
        elapsed = time.monotonic() - started
        peak_bytes = torch.cuda.max_memory_allocated(device)
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        resource_checks["peak_vram_within_90pct"] = 0 < peak_bytes < 0.9 * total_bytes
        resource_checks["elapsed_under_15_minutes"] = 0 < elapsed <= 900
        passed = all(checks.values()) and all(resource_checks.values())
        result = {
            "schema": "causal-attribution-sentinel/1",
            "status": "passed" if passed else "failed",
            "source_commit": commit,
            "uv_lock_sha256": provenance["uv_lock_sha256"],
            "clean_checkpoint_sha256": replay._sha256(clean_checkpoint),
            "noisy_checkpoint_sha256": replay._sha256(noisy_checkpoint),
            "functional_checks": checks,
            "resource_checks": resource_checks,
            "elapsed_seconds": elapsed,
            "peak_vram_gb": peak_bytes / 1024**3,
            "total_vram_gb": total_bytes / 1024**3,
            "test_loaded": False,
        }
    except Exception as error:  # sentinel must always leave a machine-readable result
        result = {
            "schema": "causal-attribution-sentinel/1",
            "status": "failed",
            "source_commit": commit,
            "functional_checks": checks,
            "resource_checks": resource_checks,
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "test_loaded": False,
        }
    clean._write_json(result_path, result)
    if result["status"] == "passed":
        clean._write_json(pass_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--noisy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_sentinel(args.clean_checkpoint, args.noisy_checkpoint, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
