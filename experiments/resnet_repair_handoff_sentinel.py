"""Frozen cross-model sentinel for temporal repair-action handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from experiments import causal_attribution_replay as replay
from experiments import causal_transport_bundle as bundle
from experiments import causal_transport_gate as gate
from experiments import m1_calibration_clean as clean
from experiments import repair_handoff_models as repair_models
from experiments.m1_noisy_data import _sha, validate_public
from experiments.m1_pilot import source_commit as current_source_commit
from src.arw import m0_core

TRAINING_SEEDS = (601, 602, 603)
DOSES = (8, 64)
CONTINUATIONS = ("clean", "noisy")
EXPECTED_NOISE_SHA256 = {
    601: "0da2b1ff8158463f5722c36ec8349398acfc802ee2c6e7974bfe5817637c3b0f",
    602: "539b94efdf99cb4747ac36743a66c7ee1373fbe58eb14f675a811f928a01c093",
    603: "8525d2a921eef11d34acc632476ccdf7f224ba4e04cefb1a5727f36561527c90",
}


@dataclass(frozen=True)
class SentinelJob:
    seed: int
    dose: int
    continuation: Literal["clean", "noisy"]
    output: Path

    @property
    def key(self) -> str:
        return f"seed{self.seed}-dose{self.dose}-{self.continuation}"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_jobs(output: Path) -> list[SentinelJob]:
    return [
        SentinelJob(
            seed=seed,
            dose=dose,
            continuation=continuation,
            output=output / "sentinel" / f"seed{seed}-dose{dose}-{continuation}",
        )
        for seed in TRAINING_SEEDS
        for dose in DOSES
        for continuation in CONTINUATIONS
    ]


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_model_binding(binding: dict[str, str]) -> None:
    if set(binding) != {
        "model_id",
        "revision",
        "config_sha256",
        "model_safetensors_sha256",
    }:
        raise ValueError("model binding schema is malformed")
    if (
        binding["model_id"] != repair_models.RESNET_MODEL_ID
        or binding["revision"] != repair_models.RESNET_REVISION
        or not _valid_sha256(binding["config_sha256"])
        or not _valid_sha256(binding["model_safetensors_sha256"])
    ):
        raise ValueError("model binding is not the frozen ResNet revision")


def _validate_store_bindings(bindings: dict[str, dict[str, object]]) -> None:
    if set(bindings) != {str(seed) for seed in TRAINING_SEEDS}:
        raise ValueError("store bindings do not match the frozen seed set")
    required = {
        "noise_bundle_sha256",
        "image_store_sha256",
        "tuning_store_sha256",
        "source_commit",
        "test_loaded",
    }
    for seed in TRAINING_SEEDS:
        binding = bindings[str(seed)]
        if (
            set(binding) != required
            or binding.get("test_loaded") is not False
            or binding.get("noise_bundle_sha256") != EXPECTED_NOISE_SHA256[seed]
        ):
            raise ValueError("store binding is not the frozen symmetric-noise input")


def write_contract(
    output: Path,
    *,
    source_commit: str,
    uv_lock_sha256: str,
    discovery_decision_sha256: str,
    prior_sentinel_decision_sha256: str,
    model_binding: dict[str, str],
    store_bindings: dict[str, dict[str, object]],
) -> dict[str, object]:
    _validate_model_binding(model_binding)
    _validate_store_bindings(store_bindings)
    hashes = (
        uv_lock_sha256,
        discovery_decision_sha256,
        prior_sentinel_decision_sha256,
    )
    if len(source_commit) != 40 or any(not _valid_sha256(value) for value in hashes):
        raise ValueError("contract provenance is malformed")
    core: dict[str, object] = {
        "schema": "resnet-repair-handoff-contract/1",
        "source_commit": source_commit,
        "uv_lock_sha256": uv_lock_sha256,
        "discovery_decision_sha256": discovery_decision_sha256,
        "prior_sentinel_decision_sha256": prior_sentinel_decision_sha256,
        "model_binding": model_binding,
        "store_bindings": store_bindings,
        "training_seeds": list(TRAINING_SEEDS),
        "doses": list(DOSES),
        "continuations": list(CONTINUATIONS),
        "warmup_steps": 500,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "scheduler": "none",
        },
        "batch_size": 32,
        "primary_outcomes": list(gate.PRIMARY_OUTCOMES),
        "test_loaded": False,
    }
    record = {**core, "contract_sha256": hashlib.sha256(_canonical(core)).hexdigest()}
    path = output / "REPAIR_HANDOFF_CONTRACT.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing repair-handoff contract is unreadable") from error
        if existing != record:
            raise RuntimeError("existing repair-handoff contract differs")
        return record
    _atomic_json(path, record)
    return record


def _finite_endpoint_values(endpoints: object) -> dict[str, dict[str, float]]:
    if not isinstance(endpoints, dict) or set(endpoints) != {"CC", "CN", "NC", "NN"}:
        raise RuntimeError("sentinel endpoints are malformed")
    converted: dict[str, dict[str, float]] = {}
    for branch in ("CC", "CN", "NC", "NN"):
        raw = endpoints[branch]
        if not isinstance(raw, dict):
            raise RuntimeError("sentinel endpoints are malformed")
        converted[branch] = {}
        for name, value in raw.items():
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("sentinel endpoint is non-finite")
            converted[branch][str(name)] = number
        if not set(gate.PRIMARY_OUTCOMES).issubset(converted[branch]):
            raise RuntimeError("sentinel primary endpoint is missing")
    return converted


def evaluate_sentinel(
    summaries: dict[str, dict[str, object]], *, contract_sha256: str
) -> dict[str, object]:
    jobs = build_jobs(Path("unused"))
    jobs_by_key = {job.key: job for job in jobs}
    if set(summaries) != set(jobs_by_key):
        raise RuntimeError("sentinel requires the exact 12-bundle matrix")
    valid = True
    commits: set[str] = set()
    observations: dict[str, dict[str, object]] = {}
    positive_by_dose = {8: 0, 64: 0}
    dose8_positive_by_seed = {seed: 0 for seed in TRAINING_SEEDS}
    for key in sorted(jobs_by_key):
        job = jobs_by_key[key]
        summary = summaries[key]
        request = summary.get("request")
        config = request.get("config") if isinstance(request, dict) else None
        commit = summary.get("source_commit")
        if isinstance(commit, str):
            commits.add(commit)
        if (
            summary.get("status") != "succeeded"
            or summary.get("contract_sha256") != contract_sha256
            or summary.get("test_loaded") is not False
            or not isinstance(request, dict)
            or request.get("seed") != job.seed
            or request.get("dose") != job.dose
            or request.get("continuation") != job.continuation
            or not isinstance(config, dict)
            or config.get("model_id") != repair_models.RESNET_MODEL_ID
        ):
            valid = False
        endpoints = _finite_endpoint_values(summary.get("endpoints"))
        raw_effects = summary.get("effects")
        if not isinstance(raw_effects, dict):
            raise RuntimeError("sentinel effects are malformed")
        per_outcome: dict[str, object] = {}
        for outcome in gate.PRIMARY_OUTCOMES:
            effect = raw_effects.get(outcome)
            if not isinstance(effect, dict):
                raise RuntimeError("sentinel effect is missing")
            carrier = gate.classify_effect(effect)  # type: ignore[arg-type]
            parameter_repair = endpoints["NN"][outcome] - endpoints["CN"][outcome]
            state_repair = endpoints["NN"][outcome] - endpoints["NC"][outcome]
            margin = parameter_repair - state_repair
            positive = margin > 0.0
            positive_by_dose[job.dose] += int(positive)
            if job.dose == 8:
                dose8_positive_by_seed[job.seed] += int(positive)
            per_outcome[outcome] = {
                "carrier": carrier,
                "parameter_repair": parameter_repair,
                "state_repair": state_repair,
                "repair_margin": margin,
            }
        observations[key] = per_outcome
    valid = valid and len(commits) == 1 and all(len(value) == 40 for value in commits)
    checks = {
        "exact_valid_matrix": valid,
        "dose64_all_12_parameter_repair": positive_by_dose[64] == 12,
        "dose8_at_most_3_parameter_repair": positive_by_dose[8] <= 3,
        "dose8_each_seed_at_most_1": all(
            count <= 1 for count in dose8_positive_by_seed.values()
        ),
    }
    return {
        "schema": "resnet-repair-handoff-decision/1",
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "bundle_count": 12,
        "contract_sha256": contract_sha256,
        "source_commit": commits.pop() if len(commits) == 1 else None,
        "positive_by_dose": {str(key): value for key, value in positive_by_dose.items()},
        "dose8_positive_by_seed": {
            str(key): value for key, value in dose8_positive_by_seed.items()
        },
        "observations": observations,
        "test_loaded": False,
    }


def evaluate_preflight(record: dict[str, object]) -> dict[str, object]:
    required_true = (
        "cuda",
        "finite_losses",
        "optimizer_state_nonempty",
        "restore_exact",
    )
    checks = {
        **{name: record.get(name) is True for name in required_true},
        "vram_below_8gb": isinstance(record.get("peak_vram_gb"), int | float)
        and float(record["peak_vram_gb"]) < 8.0,
        "test_isolated": record.get("test_loaded") is False,
    }
    return {**record, "checks": checks, "status": "passed" if all(checks.values()) else "failed"}


def validate_preflight(
    record: dict[str, object], *, source_commit: str, model_binding: dict[str, str]
) -> None:
    if (
        record.get("status") != "passed"
        or record.get("source_commit") != source_commit
        or record.get("model_binding") != model_binding
    ):
        raise RuntimeError("CUDA preflight binding or status mismatch")
    if evaluate_preflight(record).get("status") != "passed":
        raise RuntimeError("CUDA preflight functional checks failed")


def _recursive_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _recursive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _recursive_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def run_preflight(output: Path) -> dict[str, object]:
    source_commit = current_source_commit()
    model_binding = repair_models.resnet_artifact_binding(local_files_only=True)
    started = time.monotonic()
    cuda = torch.cuda.is_available()
    record: dict[str, object] = {
        "schema": "resnet-repair-handoff-cuda-sentinel/1",
        "source_commit": source_commit,
        "model_binding": model_binding,
        "cuda": cuda,
        "finite_losses": False,
        "optimizer_state_nonempty": False,
        "restore_exact": False,
        "peak_vram_gb": 0.0,
        "elapsed_seconds": 0.0,
        "test_loaded": False,
    }
    if cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(min(12, os.cpu_count() or 1))
        torch.manual_seed(20_260_813)
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        config = clean.Config(
            seed=601,
            augmentation_seed=10_601,
            epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            weight_decay=0.1,
            workers=0,
            device="cuda",
            method="adamw",
            model_id=repair_models.RESNET_MODEL_ID,
            lora_rank=0,
        )
        model = repair_models.build_transport_model(config).to(device)
        optimizer = clean._build_optimizer(model, config)
        criterion = nn.CrossEntropyLoss()
        images = torch.randn(2, 3, 224, 224, device=device)
        labels = torch.tensor([0, 1], device=device)
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        first_loss = criterion(replay._forward_logits(model, images), labels)
        first_loss.backward()
        optimizer.step()
        losses.append(float(first_loss.detach().cpu()))
        parameters = m0_core.capture_trainable_state(model)
        optimizer_state = m0_core.capture_optimizer_state(optimizer)
        optimizer.zero_grad(set_to_none=True)
        second_loss = criterion(replay._forward_logits(model, images), labels)
        second_loss.backward()
        optimizer.step()
        losses.append(float(second_loss.detach().cpu()))
        m0_core.restore_trainable_state(model, parameters)
        m0_core.restore_optimizer_state(optimizer, optimizer_state)
        record.update(
            {
                "finite_losses": all(math.isfinite(loss) for loss in losses),
                "optimizer_state_nonempty": bool(optimizer_state.get("state")),
                "restore_exact": _recursive_equal(
                    parameters, m0_core.capture_trainable_state(model)
                )
                and _recursive_equal(
                    optimizer_state, m0_core.capture_optimizer_state(optimizer)
                ),
                "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                "losses": losses,
            }
        )
    record["elapsed_seconds"] = time.monotonic() - started
    decision = evaluate_preflight(record)
    _atomic_json(output, decision)
    return decision


def _binding(
    noise_bundle: Path, image_store: Path, tuning_store: Path
) -> dict[str, object]:
    public = validate_public(noise_bundle, image_store, tuning_store)
    return {
        "noise_bundle_sha256": _sha(noise_bundle / "manifest.json"),
        "image_store_sha256": _sha(image_store / "manifest.json"),
        "tuning_store_sha256": _sha(tuning_store / "manifest.json"),
        "source_commit": public["source_commit"],
        "test_loaded": False,
    }


def _parse_noise_bundles(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text.isdecimal() or not path_text:
            raise ValueError("noise bundle must use SEED=PATH")
        seed = int(seed_text)
        if seed in result:
            raise ValueError("duplicate noise bundle seed")
        result[seed] = Path(path_text)
    if set(result) != set(TRAINING_SEEDS):
        raise ValueError(f"noise bundles must contain exact seeds {TRAINING_SEEDS}")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    source_commit = current_source_commit()
    model_binding = repair_models.resnet_artifact_binding(local_files_only=True)
    try:
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("CUDA preflight cannot be loaded") from error
    if not isinstance(preflight, dict):
        raise RuntimeError("CUDA preflight is malformed")
    validate_preflight(
        preflight, source_commit=source_commit, model_binding=model_binding
    )
    noise_bundles = _parse_noise_bundles(args.noise_bundle)
    store_bindings = {
        str(seed): _binding(path, args.image_store, args.tuning_store)
        for seed, path in sorted(noise_bundles.items())
    }
    contract = write_contract(
        args.output,
        source_commit=source_commit,
        uv_lock_sha256=_sha(Path(__file__).resolve().parents[1] / "uv.lock"),
        discovery_decision_sha256=_sha(args.discovery_decision),
        prior_sentinel_decision_sha256=_sha(args.prior_sentinel_decision),
        model_binding=model_binding,
        store_bindings=store_bindings,
    )
    contract_sha256 = str(contract["contract_sha256"])
    decision_path = args.output / "REPAIR_HANDOFF_DECISION.json"
    decision_path.unlink(missing_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    for job in build_jobs(args.output):
        config = clean.Config(
            seed=job.seed,
            augmentation_seed=10_000 + job.seed,
            epochs=1,
            batch_size=32,
            learning_rate=3e-4,
            weight_decay=0.1,
            workers=args.workers,
            device=args.device,
            method="adamw",
            model_id=repair_models.RESNET_MODEL_ID,
            lora_rank=0,
        )
        summaries[job.key] = bundle.run_bundle(
            bundle.TransportRequest(
                method="adamw",
                seed=job.seed,
                dose=job.dose,
                warmup_steps=500,
                continuation=job.continuation,
                config=config,
                data_dir=args.data_dir,
                image_store=args.image_store,
                tuning_store=args.tuning_store,
                noisy_bundle=noise_bundles[job.seed],
                output=job.output,
                source_commit=source_commit,
                contract_sha256=contract_sha256,
                device=args.device,
                expected_input_binding=store_bindings[str(job.seed)],
            )
        )
    decision = evaluate_sentinel(summaries, contract_sha256=contract_sha256)
    _atomic_json(decision_path, decision)
    return {"contract": contract, "repair_handoff": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--discovery-decision", type=Path, required=True)
    execute.add_argument("--prior-sentinel-decision", type=Path, required=True)
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--image-store", type=Path, required=True)
    execute.add_argument("--tuning-store", type=Path, required=True)
    execute.add_argument("--noise-bundle", action="append", default=[])
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(run_preflight(args.output), indent=2, sort_keys=True))
    else:
        print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
