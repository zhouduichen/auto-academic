"""CUDA functional/resource sentinel for causal transport bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from experiments import causal_attribution_replay as replay
from experiments import causal_transport_bundle as bundle
from experiments import m1_calibration_clean as clean
from experiments.m0_optimizer_state import m0_run as m0
from experiments.m1_pilot import source_commit as current_source_commit
from src.arw import m0_core

REQUIRED_CHECKS = (
    "paired_images_exact",
    "labels_only_difference",
    "optimizer_clock_match",
    "factorial_carriers_exact",
    "deterministic_replay",
    "resume_atomic",
    "nonfinite_rejected",
    "test_isolated",
)
ARTIFACT_NAME = "CAUSAL_TRANSPORT_SENTINEL_PASS.json"


def evaluate_sentinel(record: dict[str, object]) -> dict[str, object]:
    functional = record.get("functional")
    peak = record.get("peak_vram_gb")
    elapsed = record.get("elapsed_seconds")
    checks = {
        "cuda": record.get("cuda") is True,
        "functional": isinstance(functional, dict)
        and set(functional) == set(REQUIRED_CHECKS)
        and all(functional.get(name) is True for name in REQUIRED_CHECKS),
        "peak_vram_below_8_gb": isinstance(peak, int | float)
        and not isinstance(peak, bool)
        and math.isfinite(float(peak))
        and 0 < float(peak) < 8.0,
        "elapsed_below_300_seconds": isinstance(elapsed, int | float)
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and 0 < float(elapsed) < 300.0,
        "test_isolated": record.get("test_loaded") is False,
    }
    return {
        **record,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def _state(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> replay.FactorialBranchState:
    return replay.FactorialBranchState(
        deepcopy(replay.capture_model_state(model)),
        deepcopy(m0_core.capture_optimizer_state(optimizer)),
    )


def _checkpoint(state: replay.FactorialBranchState) -> replay.CheckpointState:
    return replay.CheckpointState(
        deepcopy(state.model_state), deepcopy(state.optimizer), 0
    )


def _tensor_hash(value: object) -> str:
    return m0._state_sha(value)


def _run_once(device: torch.device, seed: int) -> tuple[dict[str, bool], str]:
    m0.seed_everything(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0.01, foreach=False, fused=False
    )
    criterion = nn.CrossEntropyLoss()
    warm_images = torch.tensor(
        [[1.0, 0.0, -1.0, 0.5], [-0.5, 1.0, 0.0, 1.0]], device=device
    )
    warm_labels = torch.tensor([0, 1], device=device)
    for _ in range(2):
        replay._training_step(model, optimizer, criterion, warm_images, warm_labels)
    common = _state(model, optimizer)
    common_rng = m0.capture_rng_state()
    paired_images = warm_images.detach().cpu().clone()
    clean_labels = torch.tensor([0, 1])
    noisy_labels = torch.tensor([1, 0])
    batches = [(paired_images, clean_labels, noisy_labels)]
    clean_state = bundle.run_paired_exposure(
        model,
        optimizer,
        criterion,
        common,
        common_rng,
        batches,
        noisy=False,
        device=device,
    )
    noisy_state = bundle.run_paired_exposure(
        model,
        optimizer,
        criterion,
        common,
        common_rng,
        batches,
        noisy=True,
        device=device,
    )
    replay.validate_checkpoint_pair(clean_state, noisy_state)
    branches = replay.build_factorial_branches(clean_state, noisy_state)
    clean_parameters = _tensor_hash(clean_state.parameters)
    noisy_parameters = _tensor_hash(noisy_state.parameters)
    clean_optimizer = _tensor_hash(clean_state.optimizer)
    noisy_optimizer = _tensor_hash(noisy_state.optimizer)
    actual = {
        name: (_tensor_hash(state.parameters), _tensor_hash(state.optimizer))
        for name, state in branches.items()
    }
    expected = {
        "CC": (clean_parameters, clean_optimizer),
        "CN": (clean_parameters, noisy_optimizer),
        "NC": (noisy_parameters, clean_optimizer),
        "NN": (noisy_parameters, noisy_optimizer),
    }
    clock_match = bundle._optimizer_clock(clean_state) == bundle._optimizer_clock(
        noisy_state
    ) == 3
    digest = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        {
            "paired_images_exact": torch.equal(paired_images, batches[0][0]),
            "labels_only_difference": not torch.equal(clean_labels, noisy_labels),
            "optimizer_clock_match": clock_match,
            "factorial_carriers_exact": actual == expected,
            "test_isolated": True,
        },
        digest,
    )


def _atomic_checks(output: Path, device: torch.device) -> tuple[bool, bool]:
    model = nn.Linear(2, 2, bias=False).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, foreach=False, fused=False
    )
    replay._training_step(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.ones(1, 2, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
    )
    state = _checkpoint(_state(model, optimizer))
    atomic_root = output / "sentinel-snapshots"
    bundle.write_snapshot_pair(atomic_root, state, state, contract_sha256="a" * 64)
    loaded = bundle.load_snapshot_pair(atomic_root, contract_sha256="a" * 64)
    replay.validate_checkpoint_pair(*loaded)
    resume_atomic = True

    bad_parameters = deepcopy(state.parameters)
    first = next(iter(bad_parameters))
    bad_parameters[first].flatten()[0] = float("nan")
    bad = replay.CheckpointState(bad_parameters, deepcopy(state.optimizer), 0)
    bad_root = output / "sentinel-nonfinite"
    try:
        bundle.write_snapshot_pair(bad_root, bad, state, contract_sha256="a" * 64)
    except ValueError:
        nonfinite_rejected = not (bad_root / bundle.SNAPSHOT_MANIFEST).exists()
    else:
        nonfinite_rejected = False
    return resume_atomic, nonfinite_rejected


def run_sentinel(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    pass_path = output / ARTIFACT_NAME
    pass_path.unlink(missing_ok=True)
    started = time.monotonic()
    source_commit = current_source_commit()
    if not torch.cuda.is_available():
        return evaluate_sentinel(
            {
                "schema": "causal-transport-sentinel/1",
                "cuda": False,
                "functional": {name: False for name in REQUIRED_CHECKS},
                "peak_vram_gb": 0.0,
                "elapsed_seconds": time.monotonic() - started,
                "source_commit": source_commit,
                "test_loaded": False,
            }
        )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    first_checks, first_digest = _run_once(device, 20_260_811)
    second_checks, second_digest = _run_once(device, 20_260_811)
    resume_atomic, nonfinite_rejected = _atomic_checks(output, device)
    functional = {
        **first_checks,
        "deterministic_replay": first_digest == second_digest
        and first_checks == second_checks,
        "resume_atomic": resume_atomic,
        "nonfinite_rejected": nonfinite_rejected,
    }
    record = evaluate_sentinel(
        {
            "schema": "causal-transport-sentinel/1",
            "cuda": True,
            "functional": functional,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "elapsed_seconds": time.monotonic() - started,
            "device_name": torch.cuda.get_device_name(device),
            "total_vram_gb": torch.cuda.get_device_properties(device).total_memory / 1024**3,
            "source_commit": source_commit,
            "uv_lock_sha256": clean._sha_file(Path(__file__).resolve().parents[1] / "uv.lock"),
            "test_loaded": False,
        }
    )
    if record["status"] == "passed":
        clean._write_json(pass_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_sentinel(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
