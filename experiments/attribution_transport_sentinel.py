"""Run the frozen external attribution-transport sentinel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from experiments import causal_transport_bundle as bundle
from experiments import causal_transport_gate as gate
from experiments import m1_calibration_clean as clean
from experiments.m1_noisy_data import (
    _authorize_sealed_source_reuse,
    _seal,
    _sha,
    _source_commit,
    _validate_seal,
    derive_rng,
    validate_public,
)
from experiments.m1_pilot import source_commit as current_source_commit

TRAINING_SEEDS = (501, 502, 503)
NOISE_SEEDS = (1501, 1502, 1503)
SEED_TO_NOISE = dict(zip(TRAINING_SEEDS, NOISE_SEEDS, strict=True))
DOSE = 1250
CONTINUATIONS = ("clean", "noisy")
CORRUPTIONS_PER_CLASS = 160


@dataclass(frozen=True)
class SentinelJob:
    seed: int
    continuation: Literal["clean", "noisy"]
    output: Path

    @property
    def key(self) -> str:
        return f"seed{self.seed}-dose{DOSE}-{self.continuation}"


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


def symmetric_uniform_labels(
    clean_labels: np.ndarray,
    sample_ids: np.ndarray,
    *,
    noise_seed: int,
    class_count: int,
    corruptions_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact class-balanced symmetric-uniform corruptions."""
    clean_values = np.asarray(clean_labels)
    ids = np.asarray(sample_ids)
    if clean_values.ndim != 1 or ids.shape != clean_values.shape:
        raise ValueError("labels and sample IDs must be aligned vectors")
    if class_count < 2 or not 0 < corruptions_per_class:
        raise ValueError("noise dimensions must be positive")
    noisy = clean_values.copy()
    mask = np.zeros(len(clean_values), dtype=bool)
    for class_id in range(class_count):
        positions = np.flatnonzero(clean_values == class_id)
        if len(positions) < corruptions_per_class:
            raise ValueError("class has too few examples for the frozen corruption rate")
        order = derive_rng(noise_seed, f"symmetric/select/{class_id}").permutation(
            len(positions)
        )
        selected = positions[order[:corruptions_per_class]]
        mask[selected] = True
        for position in selected:
            stable_id = ids[position]
            if isinstance(stable_id, np.bytes_):
                stable_text = bytes(stable_id).decode("ascii")
            else:
                stable_text = str(stable_id)
            destination = int(
                derive_rng(noise_seed, f"symmetric/destination/{stable_text}").integers(
                    class_count - 1
                )
            )
            noisy[position] = destination + (destination >= class_id)
    if np.any(noisy[mask] == clean_values[mask]) or np.any(
        noisy[~mask] != clean_values[~mask]
    ):
        raise RuntimeError("symmetric corruption invariant failed")
    return noisy, mask


def generate_symmetric_noise(
    source: Path,
    training: Path,
    audit: Path,
    *,
    noise_seed: int,
    opaque_id: str,
    expected_source_manifest_sha256: str,
) -> None:
    if noise_seed not in NOISE_SEEDS:
        raise ValueError(f"noise_seed must be one of {NOISE_SEEDS}")
    source_manifest = _validate_seal(source)
    if source_manifest.get("role") != "private-train-source":
        raise RuntimeError("noise generator requires the private train role")
    _authorize_sealed_source_reuse(
        source_manifest,
        current_commit=_source_commit(),
        actual_manifest_sha256=_sha(source / "manifest.json"),
        expected_manifest_sha256=expected_source_manifest_sha256,
    )
    ids = np.load(source / "sample_ids.npy", allow_pickle=False)
    clean_labels = np.load(source / "clean_labels.npy", allow_pickle=False)
    if ids.shape != (40_000,) or clean_labels.shape != (40_000,):
        raise RuntimeError("unexpected CIFAR-100 train source shape")
    if len(np.unique(ids)) != 40_000 or not np.array_equal(ids, np.sort(ids)):
        raise RuntimeError("train source IDs are invalid")
    expected_counts = np.full(100, 400)
    if not np.array_equal(np.bincount(clean_labels, minlength=100), expected_counts):
        raise RuntimeError("train source is not class balanced")
    noisy_labels, mask = symmetric_uniform_labels(
        clean_labels,
        ids,
        noise_seed=noise_seed,
        class_count=100,
        corruptions_per_class=CORRUPTIONS_PER_CLASS,
    )
    counts = np.bincount(clean_labels[mask], minlength=100)
    if mask.sum() != 16_000 or not np.array_equal(
        counts, np.full(100, CORRUPTIONS_PER_CLASS)
    ):
        raise RuntimeError("exact corruption count validation failed")

    training.mkdir(parents=True, exist_ok=False)
    audit.mkdir(parents=True, exist_ok=False)
    np.save(training / "sample_ids.npy", ids, allow_pickle=False)
    np.save(training / "noisy_labels.npy", noisy_labels, allow_pickle=False)
    np.save(audit / "clean_labels.npy", clean_labels, allow_pickle=False)
    np.save(audit / "corruption_mask.npy", mask, allow_pickle=False)
    common = {
        "source_commit": source_manifest["source_commit"],
        "uv_lock_sha256": source_manifest["uv_lock_sha256"],
        "train_ids_sha256": _sha(source / "sample_ids.npy"),
        "test_constructed": False,
        "test_loaded": False,
        "test_access_count": 0,
    }
    _seal(
        training,
        ["sample_ids.npy", "noisy_labels.npy"],
        {
            **common,
            "schema": "m1-noise-training/1",
            "opaque_artifact_id": opaque_id,
            "sample_count": 40_000,
            "protocol_handle": "opaque-attribution-symmetric-v1",
        },
    )
    _seal(
        audit,
        ["clean_labels.npy", "corruption_mask.npy"],
        {
            **common,
            "schema": "attribution-symmetric-noise-audit/1",
            "noise_seed": noise_seed,
            "noise_kind": "class-balanced-symmetric-uniform",
            "total_corruptions": 16_000,
            "corruptions_per_class": counts.tolist(),
        },
    )


def build_jobs(output: Path) -> list[SentinelJob]:
    return [
        SentinelJob(
            seed=seed,
            continuation=continuation,
            output=output / "sentinel" / f"seed{seed}-dose{DOSE}-{continuation}",
        )
        for seed in TRAINING_SEEDS
        for continuation in CONTINUATIONS
    ]


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


def write_contract(
    output: Path,
    *,
    source_commit: str,
    uv_lock_sha256: str,
    discovery_decision_sha256: str,
    store_bindings: dict[str, dict[str, object]],
) -> dict[str, object]:
    if set(store_bindings) != {str(seed) for seed in TRAINING_SEEDS} or any(
        binding.get("test_loaded") is not False for binding in store_bindings.values()
    ):
        raise ValueError("store bindings must match the frozen seed matrix")
    core: dict[str, object] = {
        "schema": "attribution-transport-sentinel-contract/1",
        "source_commit": source_commit,
        "uv_lock_sha256": uv_lock_sha256,
        "discovery_decision_sha256": discovery_decision_sha256,
        "training_seeds": list(TRAINING_SEEDS),
        "noise_seeds": [SEED_TO_NOISE[seed] for seed in TRAINING_SEEDS],
        "dose": DOSE,
        "continuations": list(CONTINUATIONS),
        "noise_kind": "class-balanced-symmetric-uniform-40pct",
        "warmup_steps": 500,
        "primary_outcomes": list(gate.PRIMARY_OUTCOMES),
        "ratio_thresholds": {"state": 1.0, "parameter": -1.0},
        "store_bindings": store_bindings,
        "test_loaded": False,
    }
    record = {**core, "contract_sha256": hashlib.sha256(_canonical(core)).hexdigest()}
    path = output / "SENTINEL_CONTRACT.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing sentinel contract is unreadable") from error
        if existing != record:
            raise RuntimeError("existing sentinel contract differs")
        return record
    _atomic_json(path, record)
    return record


def evaluate_sentinel(
    summaries: dict[str, dict[str, object]], *, contract_sha256: str
) -> dict[str, object]:
    jobs = build_jobs(Path("unused"))
    jobs_by_key = {job.key: job for job in jobs}
    expected = set(jobs_by_key)
    if set(summaries) != expected:
        raise RuntimeError("sentinel requires the exact six-bundle matrix")
    classifications: dict[str, dict[str, object]] = {}
    valid = True
    all_parameter = True
    all_parameter_repair_better = True
    for key in sorted(expected):
        summary = summaries[key]
        request = summary.get("request")
        expected_job = jobs_by_key[key]
        if (
            summary.get("status") != "succeeded"
            or summary.get("contract_sha256") != contract_sha256
            or summary.get("test_loaded") is not False
            or not isinstance(request, dict)
            or request.get("seed") != expected_job.seed
            or request.get("dose") != DOSE
            or request.get("continuation") != expected_job.continuation
        ):
            valid = False
        effects = summary.get("effects")
        endpoints = summary.get("endpoints")
        if not isinstance(effects, dict) or not isinstance(endpoints, dict):
            raise RuntimeError("sentinel bundle outcomes are missing")
        try:
            endpoint_values = [
                float(value)
                for branch in ("CC", "CN", "NC", "NN")
                for value in endpoints[branch].values()
            ]
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError("sentinel endpoints are malformed") from error
        if not endpoint_values or not all(math.isfinite(value) for value in endpoint_values):
            raise RuntimeError("sentinel endpoints are non-finite")
        per_outcome: dict[str, object] = {}
        for outcome in gate.PRIMARY_OUTCOMES:
            raw_effect = effects.get(outcome)
            if not isinstance(raw_effect, dict):
                raise RuntimeError("sentinel effect is missing")
            values = {name: float(raw_effect[name]) for name in raw_effect}
            if set(values) != {"parameter", "state", "interaction"} or not all(
                math.isfinite(value) for value in values.values()
            ):
                raise RuntimeError("sentinel effect is malformed")
            carrier = gate.classify_effect(values)
            try:
                nn_value = float(endpoints["NN"][outcome])
                nc_value = float(endpoints["NC"][outcome])
                cn_value = float(endpoints["CN"][outcome])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("sentinel endpoints are malformed") from error
            repair_margin = (nn_value - cn_value) - (nn_value - nc_value)
            if not math.isfinite(repair_margin):
                raise RuntimeError("sentinel repair margin is non-finite")
            all_parameter = all_parameter and carrier == "parameter"
            all_parameter_repair_better = (
                all_parameter_repair_better and repair_margin > 0.0
            )
            per_outcome[outcome] = {
                "carrier": carrier,
                "repair_margin": repair_margin,
            }
        classifications[key] = per_outcome
    checks = {
        "exact_valid_matrix": valid,
        "all_12_parameter": all_parameter,
        "all_12_parameter_repair_better": all_parameter_repair_better,
    }
    return {
        "schema": "attribution-transport-sentinel-decision/1",
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "bundle_count": 6,
        "contract_sha256": contract_sha256,
        "classifications": classifications,
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
    noise_bundles = _parse_noise_bundles(args.noise_bundle)
    bindings = {
        str(seed): _binding(path, args.image_store, args.tuning_store)
        for seed, path in sorted(noise_bundles.items())
    }
    contract = write_contract(
        args.output,
        source_commit=source_commit,
        uv_lock_sha256=_sha(Path(__file__).resolve().parents[1] / "uv.lock"),
        discovery_decision_sha256=_sha(args.discovery_decision),
        store_bindings=bindings,
    )
    contract_sha256 = str(contract["contract_sha256"])
    decision_path = args.output / "SENTINEL_DECISION.json"
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
        )
        summaries[job.key] = bundle.run_bundle(
            bundle.TransportRequest(
                method="adamw",
                seed=job.seed,
                dose=DOSE,
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
                expected_input_binding=bindings[str(job.seed)],
            )
        )
    decision = evaluate_sentinel(summaries, contract_sha256=contract_sha256)
    _atomic_json(decision_path, decision)
    return {"contract": contract, "sentinel": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-noise")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--training", type=Path, required=True)
    prepare.add_argument("--audit", type=Path, required=True)
    prepare.add_argument("--noise-seed", type=int, choices=NOISE_SEEDS, required=True)
    prepare.add_argument("--opaque-id", required=True)
    prepare.add_argument("--expected-source-manifest-sha256", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--discovery-decision", type=Path, required=True)
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--image-store", type=Path, required=True)
    execute.add_argument("--tuning-store", type=Path, required=True)
    execute.add_argument("--noise-bundle", action="append", default=[])
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare-noise":
        generate_symmetric_noise(
            args.source,
            args.training,
            args.audit,
            noise_seed=args.noise_seed,
            opaque_id=args.opaque_id,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        )
        print(json.dumps({"status": "prepared", "noise_seed": args.noise_seed}))
    else:
        print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
