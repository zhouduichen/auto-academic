"""Fail-closed ResNet sentinel for complete-model-state repair interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments import causal_attribution_replay as replay
from experiments import causal_transport_bundle as bundle
from experiments import m1_calibration_clean as clean
from experiments import repair_handoff_models as repair_models
from experiments import resnet_repair_handoff_sentinel as handoff
from experiments.m1_noisy_data import _sha
from experiments.m1_pilot import source_commit as current_source_commit

SEED = 601
DOSE = 1250
CONTINUATIONS = ("clean", "noisy")
STANDARD_ORDER = replay.BRANCH_NAMES
REVERSED_ORDER = tuple(reversed(replay.BRANCH_NAMES))


@dataclass(frozen=True)
class FullStateJob:
    continuation: str
    branch_order: tuple[str, ...]
    output: Path

    @property
    def key(self) -> str:
        suffix = "standard" if self.branch_order == STANDARD_ORDER else "reversed"
        return f"seed{SEED}-dose{DOSE}-{self.continuation}-{suffix}"


def build_jobs(output: Path) -> list[FullStateJob]:
    return [
        FullStateJob(
            continuation,
            order,
            output
            / ("sentinel" if order == STANDARD_ORDER else "diagnostic")
            / f"seed{SEED}-dose{DOSE}-{continuation}",
        )
        for continuation in CONTINUATIONS
        for order in (STANDARD_ORDER, REVERSED_ORDER)
    ]


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} cannot be loaded") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is malformed")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("trajectory cannot be loaded") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("trajectory is malformed")
    return rows


def _finite(value: object) -> bool:
    if isinstance(value, bool | str) or value is None:
        return True
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_finite(item) for item in value)
    return False


def _horizon_zero(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("horizon") == 0]


def write_contract(
    output: Path,
    *,
    source_commit: str,
    uv_lock_sha256: str,
    model_binding: dict[str, str],
    store_binding: dict[str, object],
    excluded_artifact_sha256: dict[str, str],
) -> dict[str, object]:
    handoff._validate_model_binding(model_binding)
    if (
        set(store_binding)
        != {
            "noise_bundle_sha256",
            "image_store_sha256",
            "tuning_store_sha256",
            "source_commit",
            "test_loaded",
        }
        or store_binding.get("noise_bundle_sha256")
        != handoff.EXPECTED_NOISE_SHA256[SEED]
        or store_binding.get("test_loaded") is not False
    ):
        raise ValueError("full-state store binding is malformed")
    if (
        len(source_commit) != 40
        or not handoff._valid_sha256(uv_lock_sha256)
        or set(excluded_artifact_sha256) != {"handoff", "curve"}
        or any(
            not handoff._valid_sha256(value)
            for value in excluded_artifact_sha256.values()
        )
    ):
        raise ValueError("full-state contract provenance is malformed")
    core: dict[str, object] = {
        "schema": "resnet-full-state-contract/1",
        "source_commit": source_commit,
        "uv_lock_sha256": uv_lock_sha256,
        "model_binding": model_binding,
        "store_binding": store_binding,
        "excluded_artifact_sha256": excluded_artifact_sha256,
        "excluded_resnet_bundle_count": 24,
        "checkpoint_schema": bundle.SNAPSHOT_SCHEMA,
        "seed": SEED,
        "dose": DOSE,
        "continuations": list(CONTINUATIONS),
        "branch_orders": [list(STANDARD_ORDER), list(REVERSED_ORDER)],
        "horizons": list(replay.HORIZONS),
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
        "test_loaded": False,
    }
    record = {
        **core,
        "contract_sha256": hashlib.sha256(handoff._canonical(core)).hexdigest(),
    }
    path = output / "FULL_STATE_CONTRACT.json"
    if path.is_file():
        if _read_json(path, "existing full-state contract") != record:
            raise RuntimeError("existing full-state contract differs")
        return record
    handoff._atomic_json(path, record)
    return record


def evaluate_sentinel(
    summaries: dict[str, dict[str, object]],
    trajectories: dict[str, list[dict[str, object]]],
    *,
    contract_sha256: str,
    source_commit: str,
    snapshot_schema_and_buffers_exact: bool,
    resume_hashes_exact: bool,
) -> dict[str, object]:
    expected = {job.key: job for job in build_jobs(Path("unused"))}
    exact_matrix = set(summaries) == set(expected) == set(trajectories)
    bindings_valid = exact_matrix
    finite = exact_matrix
    if exact_matrix:
        for key, job in expected.items():
            summary = summaries[key]
            request = summary.get("request")
            config = request.get("config") if isinstance(request, dict) else None
            bindings_valid = bindings_valid and bool(
                summary.get("status") == "succeeded"
                and summary.get("source_commit") == source_commit
                and summary.get("contract_sha256") == contract_sha256
                and summary.get("branch_order") == list(job.branch_order)
                and summary.get("test_loaded") is False
                and isinstance(request, dict)
                and request.get("seed") == SEED
                and request.get("dose") == DOSE
                and request.get("continuation") == job.continuation
                and isinstance(config, dict)
                and config.get("model_id") == repair_models.RESNET_MODEL_ID
            )
            finite = finite and _finite(summary) and _finite(trajectories[key])

    order_exact = exact_matrix and all(
        trajectories[f"seed{SEED}-dose{DOSE}-{continuation}-standard"]
        == trajectories[f"seed{SEED}-dose{DOSE}-{continuation}-reversed"]
        and summaries[f"seed{SEED}-dose{DOSE}-{continuation}-standard"].get(
            "endpoints"
        )
        == summaries[f"seed{SEED}-dose{DOSE}-{continuation}-reversed"].get(
            "endpoints"
        )
        and summaries[f"seed{SEED}-dose{DOSE}-{continuation}-standard"].get(
            "effects"
        )
        == summaries[f"seed{SEED}-dose{DOSE}-{continuation}-reversed"].get(
            "effects"
        )
        for continuation in CONTINUATIONS
    )
    horizon_zero_exact = exact_matrix and _horizon_zero(
        trajectories[f"seed{SEED}-dose{DOSE}-clean-standard"]
    ) == _horizon_zero(trajectories[f"seed{SEED}-dose{DOSE}-noisy-standard"])
    checks = {
        "exact_valid_matrix": exact_matrix and bindings_valid and finite,
        "full_model_state_schema_and_buffers_exact": snapshot_schema_and_buffers_exact,
        "horizon_zero_continuations_exact": horizon_zero_exact,
        "branch_order_invariant": order_exact,
        "resume_hashes_exact": resume_hashes_exact,
    }
    return {
        "schema": "resnet-full-state-sentinel-decision/1",
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "bundle_count": 2,
        "diagnostic_count": 2,
        "contract_sha256": contract_sha256,
        "source_commit": source_commit,
        "test_loaded": False,
    }


def _snapshot_has_full_buffers(path: Path, contract_sha256: str) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return False
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != bundle.SNAPSHOT_SCHEMA
        or payload.get("contract_sha256") != contract_sha256
        or not isinstance(payload.get("model_state"), dict)
    ):
        return False
    names = set(payload["model_state"])
    required_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    return all(any(name.endswith(suffix) for name in names) for suffix in required_suffixes)


def run(args: argparse.Namespace) -> dict[str, object]:
    source_commit = current_source_commit()
    model_binding = repair_models.resnet_artifact_binding(local_files_only=True)
    preflight = _read_json(args.preflight, "CUDA preflight")
    handoff.validate_preflight(
        preflight, source_commit=source_commit, model_binding=model_binding
    )
    noise_bundle = args.noise_bundle
    store_binding = handoff._binding(
        noise_bundle, args.image_store, args.tuning_store
    )
    excluded = {
        "handoff": _sha(args.excluded_handoff_decision),
        "curve": _sha(args.excluded_curve_decision),
    }
    contract = write_contract(
        args.output,
        source_commit=source_commit,
        uv_lock_sha256=_sha(Path(__file__).resolve().parents[1] / "uv.lock"),
        model_binding=model_binding,
        store_binding=store_binding,
        excluded_artifact_sha256=excluded,
    )
    contract_sha256 = str(contract["contract_sha256"])
    decision_path = args.output / "FULL_STATE_SENTINEL_DECISION.json"
    decision_path.unlink(missing_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    trajectories: dict[str, list[dict[str, object]]] = {}
    resume_hashes_exact = True
    snapshot_checks: list[bool] = []
    for job in build_jobs(args.output):
        config = clean.Config(
            seed=SEED,
            augmentation_seed=10_000 + SEED,
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
        request = bundle.TransportRequest(
            method="adamw",
            seed=SEED,
            dose=DOSE,
            warmup_steps=500,
            continuation=job.continuation,  # type: ignore[arg-type]
            config=config,
            data_dir=args.data_dir,
            image_store=args.image_store,
            tuning_store=args.tuning_store,
            noisy_bundle=noise_bundle,
            output=job.output,
            source_commit=source_commit,
            contract_sha256=contract_sha256,
            device=args.device,
            expected_input_binding=store_binding,
        )
        summaries[job.key] = bundle.run_bundle(request, branch_order=job.branch_order)
        manifest_path = job.output / "sha256_manifest.json"
        before = manifest_path.read_bytes()
        bundle.run_bundle(request, branch_order=job.branch_order)
        resume_hashes_exact = resume_hashes_exact and before == manifest_path.read_bytes()
        trajectories[job.key] = _read_jsonl(job.output / "trajectory_metrics.jsonl")
        snapshot_checks.append(
            _snapshot_has_full_buffers(
                job.output / "clean-exposure.pt", contract_sha256
            )
            and _snapshot_has_full_buffers(
                job.output / "noisy-exposure.pt", contract_sha256
            )
        )
    decision = evaluate_sentinel(
        summaries,
        trajectories,
        contract_sha256=contract_sha256,
        source_commit=source_commit,
        snapshot_schema_and_buffers_exact=all(snapshot_checks),
        resume_hashes_exact=resume_hashes_exact,
    )
    handoff._atomic_json(decision_path, decision)
    return {"contract": contract, "full_state_sentinel": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--excluded-handoff-decision", type=Path, required=True)
    execute.add_argument("--excluded-curve-decision", type=Path, required=True)
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--image-store", type=Path, required=True)
    execute.add_argument("--tuning-store", type=Path, required=True)
    execute.add_argument("--noise-bundle", type=Path, required=True)
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(handoff.run_preflight(args.output), indent=2, sort_keys=True))
    else:
        result = run(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["full_state_sentinel"]["decision"] != "GO":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
