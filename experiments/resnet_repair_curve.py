"""Complete the ResNet repair-handoff discovery curve at doses 1 and 1250."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from experiments import causal_transport_bundle as bundle
from experiments import causal_transport_gate as gate
from experiments import m1_calibration_clean as clean
from experiments import repair_handoff_models as repair_models
from experiments import resnet_repair_handoff_sentinel as handoff
from experiments.m1_noisy_data import _sha
from experiments.m1_pilot import source_commit as current_source_commit

TRAINING_SEEDS = handoff.TRAINING_SEEDS
NEW_DOSES = (1, 1250)
ALL_DOSES = (1, 8, 64, 1250)
CONTINUATIONS = handoff.CONTINUATIONS


def build_new_jobs(output: Path) -> list[handoff.SentinelJob]:
    return [
        handoff.SentinelJob(
            seed=seed,
            dose=dose,
            continuation=continuation,
            output=output / "curve" / f"seed{seed}-dose{dose}-{continuation}",
        )
        for seed in TRAINING_SEEDS
        for dose in NEW_DOSES
        for continuation in CONTINUATIONS
    ]


def _expected_old_keys() -> set[str]:
    return {
        f"seed{seed}-dose{dose}-{continuation}"
        for seed in TRAINING_SEEDS
        for dose in handoff.DOSES
        for continuation in CONTINUATIONS
    }


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} cannot be loaded") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is malformed")
    return value


def _validate_prior_decision(
    prior: dict[str, object], *, prior_decision_sha256: str
) -> dict[str, dict[str, object]]:
    observations = prior.get("observations")
    if (
        prior.get("schema") != "resnet-repair-handoff-decision/1"
        or prior.get("decision") != "NO-GO"
        or prior.get("bundle_count") != 12
        or prior.get("test_loaded") is not False
        or not handoff._valid_sha256(prior.get("contract_sha256"))
        or not handoff._valid_sha256(prior_decision_sha256)
        or not isinstance(prior.get("source_commit"), str)
        or len(str(prior["source_commit"])) != 40
        or not isinstance(observations, dict)
        or set(observations) != _expected_old_keys()
    ):
        raise RuntimeError("prior handoff decision is malformed")
    converted: dict[str, dict[str, object]] = {}
    for key, raw in observations.items():
        if not isinstance(raw, dict) or set(raw) != set(gate.PRIMARY_OUTCOMES):
            raise RuntimeError("prior handoff observation is malformed")
        converted[key] = raw
        for outcome in gate.PRIMARY_OUTCOMES:
            item = raw[outcome]
            if not isinstance(item, dict):
                raise RuntimeError("prior handoff observation is malformed")
            margin = float(item.get("repair_margin", math.nan))
            if not math.isfinite(margin):
                raise RuntimeError("prior handoff margin is non-finite")
    return converted


def write_contract(
    output: Path,
    *,
    source_commit: str,
    uv_lock_sha256: str,
    prior_decision_sha256: str,
    prior_contract_sha256: str,
    model_binding: dict[str, str],
    store_bindings: dict[str, dict[str, object]],
) -> dict[str, object]:
    handoff._validate_model_binding(model_binding)
    handoff._validate_store_bindings(store_bindings)
    hashes = (
        uv_lock_sha256,
        prior_decision_sha256,
        prior_contract_sha256,
    )
    if len(source_commit) != 40 or any(
        not handoff._valid_sha256(value) for value in hashes
    ):
        raise ValueError("curve contract provenance is malformed")
    core: dict[str, object] = {
        "schema": "resnet-repair-curve-contract/1",
        "source_commit": source_commit,
        "uv_lock_sha256": uv_lock_sha256,
        "prior_decision_sha256": prior_decision_sha256,
        "prior_contract_sha256": prior_contract_sha256,
        "model_binding": model_binding,
        "store_bindings": store_bindings,
        "training_seeds": list(TRAINING_SEEDS),
        "new_doses": list(NEW_DOSES),
        "all_doses": list(ALL_DOSES),
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
        "trend_gate": {
            "overall_min_positive": 10,
            "per_seed_min_positive": 3,
            "per_outcome_min_positive": 5,
            "per_continuation_min_positive": 5,
        },
        "test_loaded": False,
    }
    record = {
        **core,
        "contract_sha256": hashlib.sha256(handoff._canonical(core)).hexdigest(),
    }
    path = output / "REPAIR_CURVE_CONTRACT.json"
    if path.is_file():
        existing = _read_json(path, label="existing curve contract")
        if existing != record:
            raise RuntimeError("existing curve contract differs")
        return record
    handoff._atomic_json(path, record)
    return record


def _new_margins(
    summaries: dict[str, dict[str, object]],
    *,
    contract_sha256: str,
    source_commit: str,
) -> dict[tuple[int, int, str, str], float]:
    jobs = build_new_jobs(Path("unused"))
    by_key = {job.key: job for job in jobs}
    if set(summaries) != set(by_key):
        raise RuntimeError("curve completion requires the exact 12-bundle matrix")
    result: dict[tuple[int, int, str, str], float] = {}
    for key in sorted(by_key):
        job = by_key[key]
        summary = summaries[key]
        request = summary.get("request")
        config = request.get("config") if isinstance(request, dict) else None
        if (
            summary.get("status") != "succeeded"
            or summary.get("contract_sha256") != contract_sha256
            or summary.get("source_commit") != source_commit
            or summary.get("test_loaded") is not False
            or not isinstance(request, dict)
            or request.get("seed") != job.seed
            or request.get("dose") != job.dose
            or request.get("continuation") != job.continuation
            or not isinstance(config, dict)
            or config.get("model_id") != repair_models.RESNET_MODEL_ID
        ):
            raise RuntimeError("curve completion bundle binding mismatch")
        endpoints = handoff._finite_endpoint_values(summary.get("endpoints"))
        for outcome in gate.PRIMARY_OUTCOMES:
            result[(job.seed, job.dose, job.continuation, outcome)] = (
                endpoints["NC"][outcome] - endpoints["CN"][outcome]
            )
    return result


def _describe(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _positive_counts(
    observations: list[dict[str, object]], *, field: str
) -> dict[str, object]:
    positive = [row for row in observations if float(row[field]) > 0.0]
    by_seed = {
        str(seed): sum(row["seed"] == seed for row in positive)
        for seed in TRAINING_SEEDS
    }
    by_outcome = {
        outcome: sum(row["outcome"] == outcome for row in positive)
        for outcome in gate.PRIMARY_OUTCOMES
    }
    by_continuation = {
        continuation: sum(row["continuation"] == continuation for row in positive)
        for continuation in CONTINUATIONS
    }
    return {
        "overall": len(positive),
        "by_seed": by_seed,
        "by_outcome": by_outcome,
        "by_continuation": by_continuation,
    }


def _counts_pass(counts: dict[str, object]) -> bool:
    by_seed = counts["by_seed"]
    by_outcome = counts["by_outcome"]
    by_continuation = counts["by_continuation"]
    return bool(
        int(counts["overall"]) >= 10
        and isinstance(by_seed, dict)
        and all(int(value) >= 3 for value in by_seed.values())
        and isinstance(by_outcome, dict)
        and all(int(value) >= 5 for value in by_outcome.values())
        and isinstance(by_continuation, dict)
        and all(int(value) >= 5 for value in by_continuation.values())
    )


def evaluate_curve(
    prior: dict[str, object],
    summaries: dict[str, dict[str, object]],
    *,
    contract_sha256: str,
    prior_decision_sha256: str,
    source_commit: str,
) -> dict[str, object]:
    old = _validate_prior_decision(
        prior, prior_decision_sha256=prior_decision_sha256
    )
    margins = _new_margins(
        summaries,
        contract_sha256=contract_sha256,
        source_commit=source_commit,
    )
    for seed in TRAINING_SEEDS:
        for dose in handoff.DOSES:
            for continuation in CONTINUATIONS:
                key = f"seed{seed}-dose{dose}-{continuation}"
                for outcome in gate.PRIMARY_OUTCOMES:
                    raw = old[key][outcome]
                    if not isinstance(raw, dict):
                        raise RuntimeError("prior handoff observation is malformed")
                    margins[(seed, dose, continuation, outcome)] = float(
                        raw["repair_margin"]
                    )

    observations: list[dict[str, object]] = []
    for seed in TRAINING_SEEDS:
        for continuation in CONTINUATIONS:
            for outcome in gate.PRIMARY_OUTCOMES:
                by_dose = {
                    str(dose): margins[(seed, dose, continuation, outcome)]
                    for dose in ALL_DOSES
                }
                low = (by_dose["1"] + by_dose["8"]) / 2.0
                high = (by_dose["64"] + by_dose["1250"]) / 2.0
                observations.append(
                    {
                        "seed": seed,
                        "continuation": continuation,
                        "outcome": outcome,
                        "repair_margin_by_dose": by_dose,
                        "low_exposure_mean": low,
                        "high_exposure_mean": high,
                        "high_minus_low": high - low,
                        "dose1250_margin": by_dose["1250"],
                    }
                )

    contrast_counts = _positive_counts(observations, field="high_minus_low")
    accumulated_counts = _positive_counts(observations, field="dose1250_margin")
    checks = {
        "exact_valid_matrix": True,
        "high_exposure_shift_is_robust": _counts_pass(contrast_counts),
        "dose1250_parameter_repair_is_robust": _counts_pass(accumulated_counts),
    }
    return {
        "schema": "resnet-repair-curve-decision/1",
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "new_bundle_count": 12,
        "combined_bundle_count": 24,
        "contract_sha256": contract_sha256,
        "prior_decision_sha256": prior_decision_sha256,
        "source_commit": source_commit,
        "contrast_positive_counts": contrast_counts,
        "dose1250_positive_counts": accumulated_counts,
        "contrast_summary": _describe(
            [float(row["high_minus_low"]) for row in observations]
        ),
        "dose1250_summary": _describe(
            [float(row["dose1250_margin"]) for row in observations]
        ),
        "observations": observations,
        "test_loaded": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source_commit = current_source_commit()
    model_binding = repair_models.resnet_artifact_binding(local_files_only=True)
    preflight = _read_json(args.preflight, label="CUDA preflight")
    handoff.validate_preflight(
        preflight, source_commit=source_commit, model_binding=model_binding
    )
    prior = _read_json(args.prior_decision, label="prior handoff decision")
    prior_decision_sha256 = _sha(args.prior_decision)
    _validate_prior_decision(prior, prior_decision_sha256=prior_decision_sha256)
    noise_bundles = handoff._parse_noise_bundles(args.noise_bundle)
    store_bindings = {
        str(seed): handoff._binding(path, args.image_store, args.tuning_store)
        for seed, path in sorted(noise_bundles.items())
    }
    contract = write_contract(
        args.output,
        source_commit=source_commit,
        uv_lock_sha256=_sha(Path(__file__).resolve().parents[1] / "uv.lock"),
        prior_decision_sha256=prior_decision_sha256,
        prior_contract_sha256=str(prior["contract_sha256"]),
        model_binding=model_binding,
        store_bindings=store_bindings,
    )
    contract_sha256 = str(contract["contract_sha256"])
    decision_path = args.output / "REPAIR_CURVE_DECISION.json"
    decision_path.unlink(missing_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    for job in build_new_jobs(args.output):
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
    decision = evaluate_curve(
        prior,
        summaries,
        contract_sha256=contract_sha256,
        prior_decision_sha256=prior_decision_sha256,
        source_commit=source_commit,
    )
    handoff._atomic_json(decision_path, decision)
    return {"contract": contract, "repair_curve": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--prior-decision", type=Path, required=True)
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--image-store", type=Path, required=True)
    execute.add_argument("--tuning-store", type=Path, required=True)
    execute.add_argument("--noise-bundle", action="append", default=[])
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(handoff.run_preflight(args.output), indent=2, sort_keys=True))
    else:
        print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
