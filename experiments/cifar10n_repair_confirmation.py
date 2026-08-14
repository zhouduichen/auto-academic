"""One-shot CIFAR-10N cross-model repair confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from experiments import causal_attribution_replay as replay
from experiments import causal_transport_bundle as transport
from experiments import cifar10n_transport_data as data
from experiments import m1_calibration_clean as clean
from experiments import repair_handoff_models as models
from experiments.m1_pilot import source_commit
from src.arw import m0_core

SEEDS = (701, 702, 703, 704, 705)
MODELS = ("resnet18", "vit_lora")
CONTINUATIONS = ("clean", "noisy")
PRIMARY_HORIZONS = (32, 64, 128, 256)
DOSE = 1250
WARMUP_STEPS = 500
PREFIX_STEPS = 8


@dataclass(frozen=True)
class ConfirmationJob:
    model: str
    seed: int
    continuation: str
    output: Path

    @property
    def key(self) -> str:
        return f"{self.model}-seed{self.seed}-dose{DOSE}-{self.continuation}"


def build_jobs(output: Path) -> list[ConfirmationJob]:
    return [
        ConfirmationJob(
            model,
            seed,
            continuation,
            output / "matrix" / model / f"seed{seed}-dose{DOSE}-{continuation}",
        )
        for model in MODELS
        for seed in SEEDS
        for continuation in CONTINUATIONS
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise RuntimeError("confirmation trajectory cannot be loaded") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("confirmation trajectory is malformed")
    return rows


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_contract(
    output: Path,
    *,
    design_sha256: str,
    uv_lock_sha256: str,
    sealed: dict[str, object],
    model_bindings: dict[str, dict[str, str]],
    policy_decision_sha256: str,
) -> dict[str, object]:
    if (
        any(len(value) != 64 for value in (design_sha256, uv_lock_sha256, policy_decision_sha256))
        or set(model_bindings) != set(MODELS)
        or sealed.get("dataset") != "cifar10n"
        or sealed.get("label_set") != "worse_label"
        or any(
            sealed.get(flag) is not False
            for flag in ("test_loaded", "development_loaded", "confirmation_loaded")
        )
    ):
        raise ValueError("confirmation contract inputs are malformed")
    core: dict[str, object] = {
        "schema": "cifar10n-repair-confirmation-contract/1",
        "source_commit": source_commit(),
        "design_sha256": design_sha256,
        "uv_lock_sha256": uv_lock_sha256,
        "sealed_input": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in sealed.items()
        },
        "model_bindings": model_bindings,
        "policy_decision_sha256": policy_decision_sha256,
        "jobs": [job.key for job in build_jobs(Path("unused"))],
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "continuations": list(CONTINUATIONS),
        "dose": DOSE,
        "warmup_steps": WARMUP_STEPS,
        "diagnostic_prefix_steps": PREFIX_STEPS,
        "horizons": list(replay.HORIZONS),
        "primary_horizons": list(PRIMARY_HORIZONS),
        "mechanism_gate": {
            "negative_cross_model_seeds": 5,
            "negative_per_model_seeds": 4,
            "switch_per_model_seeds": 4,
            "minimum_switch_horizons": 2,
        },
        "policy_gate": {
            "unit_wins": 8,
            "maximum_normalized_oracle_regret": 0.25,
        },
        "test_loaded": False,
        "development_loaded": False,
        "confirmation_loaded": False,
    }
    record = {
        **core,
        "contract_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    path = output / "CIFAR10N_CONFIRMATION_CONTRACT.json"
    if path.is_file():
        if _read_json(path, "existing confirmation contract") != record:
            raise RuntimeError("existing confirmation contract differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, record)
    return record


def _trajectory_values(rows: list[dict[str, object]]) -> dict[str, dict[int, float]]:
    expected = {(branch, horizon) for branch in replay.BRANCH_NAMES for horizon in replay.HORIZONS}
    indexed: dict[tuple[str, int], float] = {}
    for row in rows:
        try:
            key = (str(row["branch"]), int(row["horizon"]))
            value = float(row["clean_loss_excess"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("confirmation trajectory row is malformed") from error
        if key in indexed or not math.isfinite(value):
            raise RuntimeError("confirmation trajectory row is invalid")
        indexed[key] = value
    if set(indexed) != expected:
        raise RuntimeError("confirmation trajectory matrix is incomplete")
    return {
        branch: {horizon: indexed[(branch, horizon)] for horizon in replay.HORIZONS}
        for branch in replay.BRANCH_NAMES
    }


def _auc(values: dict[int, float]) -> float:
    horizons = list(PRIMARY_HORIZONS)
    return m0_core.trapezoid_auc(horizons, [values[horizon] for horizon in horizons])


def evaluate_confirmation(
    summaries: dict[str, dict[str, object]],
    trajectories: dict[str, list[dict[str, object]]],
    *,
    contract_sha256: str,
    expected_source_commit: str,
    policy_decision: dict[str, object],
    resume_hashes_exact: bool,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    jobs = {job.key: job for job in build_jobs(Path("unused"))}
    exact = set(summaries) == set(trajectories) == set(jobs)
    valid = exact and resume_hashes_exact
    values: dict[str, dict[str, dict[int, float]]] = {}
    diagnostics: dict[str, float] = {}
    if exact:
        for key, job in jobs.items():
            summary = summaries[key]
            request = summary.get("request")
            config = request.get("config") if isinstance(request, dict) else None
            diagnostic = summary.get("diagnostic")
            expected_model_id = (
                models.RESNET_MODEL_ID if job.model == "resnet18" else models.VIT_MODEL_ID
            )
            valid = valid and bool(
                summary.get("status") == "succeeded"
                and summary.get("source_commit") == expected_source_commit
                and summary.get("contract_sha256") == contract_sha256
                and summary.get("test_loaded") is False
                and isinstance(request, dict)
                and request.get("dataset_name") == "cifar10n"
                and request.get("num_labels") == 10
                and request.get("diagnostic_prefix_steps") == PREFIX_STEPS
                and request.get("seed") == job.seed
                and request.get("dose") == DOSE
                and request.get("continuation") == job.continuation
                and isinstance(config, dict)
                and config.get("model_id") == expected_model_id
                and isinstance(diagnostic, dict)
                and diagnostic.get("prefix_steps") == PREFIX_STEPS
            )
            try:
                score = float(diagnostic["normalized_stream_label_nll"])  # type: ignore[index]
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("confirmation diagnostic is malformed") from error
            if not math.isfinite(score):
                raise RuntimeError("confirmation diagnostic is non-finite")
            diagnostics[key] = score
            values[key] = _trajectory_values(trajectories[key])

    contrasts: dict[str, float] = {}
    switches: dict[str, dict[str, object]] = {}
    if exact:
        for model in MODELS:
            for seed in SEEDS:
                clean_key = f"{model}-seed{seed}-dose{DOSE}-clean"
                noisy_key = f"{model}-seed{seed}-dose{DOSE}-noisy"
                clean_margin = {
                    horizon: values[clean_key]["NC"][horizon]
                    - values[clean_key]["CN"][horizon]
                    for horizon in replay.HORIZONS
                }
                noisy_margin = {
                    horizon: values[noisy_key]["NC"][horizon]
                    - values[noisy_key]["CN"][horizon]
                    for horizon in replay.HORIZONS
                }
                contrasts[f"{model}-seed{seed}"] = sum(
                    noisy_margin[horizon] - clean_margin[horizon]
                    for horizon in PRIMARY_HORIZONS
                ) / len(PRIMARY_HORIZONS)
                switch_horizons = [
                    horizon
                    for horizon in PRIMARY_HORIZONS
                    if clean_margin[horizon] > 0.0 and noisy_margin[horizon] < 0.0
                ]
                switches[f"{model}-seed{seed}"] = {
                    "count": len(switch_horizons),
                    "horizons": switch_horizons,
                }
    cross_model = {
        str(seed): sum(contrasts[f"{model}-seed{seed}"] for model in MODELS)
        / len(MODELS)
        for seed in SEEDS
    } if exact else {}
    negative_per_model = {
        model: sum(contrasts[f"{model}-seed{seed}"] < 0.0 for seed in SEEDS)
        for model in MODELS
    } if exact else {}
    switch_per_model = {
        model: sum(
            int(switches[f"{model}-seed{seed}"]["count"]) >= 2  # type: ignore[index]
            for seed in SEEDS
        )
        for model in MODELS
    } if exact else {}
    mechanism_checks = {
        "exact_valid_matrix": valid,
        "all_cross_model_seed_contrasts_negative": exact
        and all(value < 0.0 for value in cross_model.values()),
        "at_least_four_negative_seeds_per_model": exact
        and all(value >= 4 for value in negative_per_model.values()),
        "at_least_four_switch_seeds_per_model": exact
        and all(value >= 4 for value in switch_per_model.values()),
    }
    mechanism = {
        "schema": "cifar10n-mechanism-decision/1",
        "decision": "GO" if all(mechanism_checks.values()) else "NO-GO",
        "checks": mechanism_checks,
        "cross_model_seed_contrasts": cross_model,
        "model_seed_contrasts": contrasts,
        "action_switches": switches,
        "negative_per_model": negative_per_model,
        "switch_per_model": switch_per_model,
        "contract_sha256": contract_sha256,
        "test_loaded": False,
    }

    eligible = policy_decision.get("decision") == "GO"
    unit_rows: list[dict[str, object]] = []
    if exact and eligible:
        raw_models = policy_decision.get("models")
        if not isinstance(raw_models, dict) or set(raw_models) != set(MODELS):
            raise RuntimeError("policy thresholds are malformed")
        for model in MODELS:
            threshold = float(raw_models[model]["threshold"])  # type: ignore[index]
            if not math.isfinite(threshold):
                raise RuntimeError("policy threshold is non-finite")
            for seed in SEEDS:
                aggregated = {branch: 0.0 for branch in ("CN", "NC", "NN")}
                selected = 0.0
                oracle = 0.0
                choices: dict[str, str] = {}
                for continuation in CONTINUATIONS:
                    key = f"{model}-seed{seed}-dose{DOSE}-{continuation}"
                    branch_auc = {
                        branch: _auc(values[key][branch]) for branch in aggregated
                    }
                    for branch, value in branch_auc.items():
                        aggregated[branch] += value / len(CONTINUATIONS)
                    choice = "CN" if diagnostics[key] < threshold else "NC"
                    choices[continuation] = choice
                    selected += branch_auc[choice] / len(CONTINUATIONS)
                    oracle += min(branch_auc["CN"], branch_auc["NC"]) / len(CONTINUATIONS)
                best_fixed = min(aggregated["CN"], aggregated["NC"])
                unit_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "policy": selected,
                        "always_CN": aggregated["CN"],
                        "always_NC": aggregated["NC"],
                        "no_repair_NN": aggregated["NN"],
                        "oracle": oracle,
                        "best_fixed": best_fixed,
                        "beats_best_fixed": selected < best_fixed,
                        "choices": choices,
                    }
                )
    model_improvement = {
        model: bool(
            sum(float(row["policy"]) for row in unit_rows if row["model"] == model)
            < sum(float(row["always_CN"]) for row in unit_rows if row["model"] == model)
            and sum(float(row["policy"]) for row in unit_rows if row["model"] == model)
            < sum(float(row["always_NC"]) for row in unit_rows if row["model"] == model)
        )
        for model in MODELS
    } if unit_rows else {}
    wins = sum(bool(row["beats_best_fixed"]) for row in unit_rows)
    if unit_rows:
        pooled_policy = sum(float(row["policy"]) for row in unit_rows) / len(unit_rows)
        pooled_oracle = sum(float(row["oracle"]) for row in unit_rows) / len(unit_rows)
        pooled_fixed = sum(float(row["best_fixed"]) for row in unit_rows) / len(unit_rows)
        gap = pooled_fixed - pooled_oracle
        if gap == 0.0:
            normalized_regret = 0.0 if pooled_policy == pooled_oracle else math.inf
        else:
            normalized_regret = (pooled_policy - pooled_oracle) / gap
    else:
        normalized_regret = math.inf
    policy_checks = {
        "calibration_eligible": eligible,
        "lower_than_both_fixed_in_each_model": eligible
        and all(model_improvement.values()),
        "beats_best_fixed_in_eight_units": eligible and wins >= 8,
        "normalized_oracle_regret_at_most_quarter": eligible
        and math.isfinite(normalized_regret)
        and normalized_regret <= 0.25,
    }
    policy = {
        "schema": "cifar10n-policy-decision/1",
        "decision": "GO" if all(policy_checks.values()) else "NO-GO",
        "checks": policy_checks,
        "unit_results": unit_rows,
        "model_improvement": model_improvement,
        "unit_wins": wins,
        "normalized_oracle_regret": normalized_regret
        if math.isfinite(normalized_regret)
        else None,
        "contract_sha256": contract_sha256,
        "test_loaded": False,
    }
    report = {
        "schema": "cifar10n-confirmation-report/1",
        "mechanism_decision": mechanism["decision"],
        "policy_decision": policy["decision"],
        "bundle_count": len(summaries),
        "contract_sha256": contract_sha256,
        "source_commit": expected_source_commit,
        "test_loaded": False,
    }
    return mechanism, policy, report


def run_preflight(output: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    bindings = {
        "resnet18": models.resnet_artifact_binding(local_files_only=True),
        "vit_lora": models.vit_artifact_binding(local_files_only=True),
    }
    checks: dict[str, bool] = {}
    peaks: dict[str, float] = {}
    for model_key, model_id in (
        ("resnet18", models.RESNET_MODEL_ID),
        ("vit_lora", models.VIT_MODEL_ID),
    ):
        config = clean.Config(
            seed=701,
            augmentation_seed=10_701,
            batch_size=2,
            model_id=model_id,
            lora_rank=0 if model_key == "resnet18" else 8,
            device="cuda",
        )
        torch.cuda.reset_peak_memory_stats(device)
        model = models.build_transport_model(config, num_labels=10).to(device)
        optimizer = clean._build_optimizer(model, config)
        images = torch.randn(2, 3, 224, 224, device=device)
        labels = torch.tensor([0, 1], device=device)
        replay._training_step(model, optimizer, nn.CrossEntropyLoss(), images, labels)
        state = replay.capture_model_state(model)
        optimizer_state = m0_core.capture_optimizer_state(optimizer)
        replay._training_step(model, optimizer, nn.CrossEntropyLoss(), images, labels)
        replay.restore_model_state(model, state)
        m0_core.restore_optimizer_state(optimizer, optimizer_state)
        model_exact = all(
            torch.equal(value, state[name]) for name, value in model.state_dict().items()
        )
        optimizer_exact = (
            m0_core.optimizer_state_distance(
                m0_core.capture_optimizer_state(optimizer), optimizer_state
            )
            == 0.0
        )
        checks[model_key] = model_exact and optimizer_exact
        peaks[model_key] = torch.cuda.max_memory_allocated(device) / 1024**3
        del optimizer, model, images, labels
        torch.cuda.empty_cache()
    record = {
        "schema": "cifar10n-confirmation-preflight/1",
        "status": "passed" if all(checks.values()) else "failed",
        "source_commit": source_commit(),
        "model_bindings": bindings,
        "checks": checks,
        "peak_vram_gb": peaks,
        "cuda": True,
        "test_loaded": False,
    }
    _atomic_json(output, record)
    return record


def run(args: argparse.Namespace) -> dict[str, object]:
    current_commit = source_commit()
    sealed = data.validate_cifar10n(args.sealed_root)
    policy_decision = _read_json(args.policy_decision, "policy decision")
    preflight = _read_json(args.preflight, "CUDA preflight")
    model_bindings = {
        "resnet18": models.resnet_artifact_binding(local_files_only=True),
        "vit_lora": models.vit_artifact_binding(local_files_only=True),
    }
    if (
        preflight.get("status") != "passed"
        or preflight.get("source_commit") != current_commit
        or preflight.get("model_bindings") != model_bindings
        or preflight.get("test_loaded") is not False
    ):
        raise RuntimeError("confirmation preflight mismatch")
    root = Path(__file__).resolve().parents[1]
    contract = write_contract(
        args.output,
        design_sha256=_sha(
            root / "docs/superpowers/specs/2026-08-14-cifar10n-cross-model-confirmation-design.md"
        ),
        uv_lock_sha256=_sha(root / "uv.lock"),
        sealed=sealed,
        model_bindings=model_bindings,
        policy_decision_sha256=_sha(args.policy_decision),
    )
    contract_sha256 = str(contract["contract_sha256"])
    for name in (
        "CIFAR10N_MECHANISM_DECISION.json",
        "CIFAR10N_POLICY_DECISION.json",
        "CIFAR10N_CONFIRMATION_REPORT.json",
    ):
        (args.output / name).unlink(missing_ok=True)
    training = Path(sealed["training"])
    image_store = Path(sealed["image_store"])
    tuning_store = Path(sealed["tuning_store"])
    training_manifest = _read_json(training / "manifest.json", "training manifest")
    binding = {
        "noise_bundle_sha256": _sha(training / "manifest.json"),
        "image_store_sha256": _sha(image_store / "manifest.json"),
        "tuning_store_sha256": _sha(tuning_store / "manifest.json"),
        "source_commit": training_manifest["source_commit"],
        "test_loaded": False,
    }
    summaries: dict[str, dict[str, object]] = {}
    trajectories: dict[str, list[dict[str, object]]] = {}
    resume_exact = True
    for job in build_jobs(args.output):
        model_id = models.RESNET_MODEL_ID if job.model == "resnet18" else models.VIT_MODEL_ID
        config = clean.Config(
            seed=job.seed,
            split_seed=data.SPLIT_SEED,
            augmentation_seed=10_000 + job.seed,
            epochs=1,
            batch_size=32,
            learning_rate=3e-4,
            weight_decay=0.1,
            workers=args.workers,
            device=args.device,
            method="adamw",
            model_id=model_id,
            lora_rank=0 if job.model == "resnet18" else 8,
        )
        request = transport.TransportRequest(
            method="adamw",
            seed=job.seed,
            dose=DOSE,
            warmup_steps=WARMUP_STEPS,
            continuation=job.continuation,  # type: ignore[arg-type]
            config=config,
            data_dir=args.cifar_root,
            image_store=image_store,
            tuning_store=tuning_store,
            noisy_bundle=training,
            output=job.output,
            source_commit=current_commit,
            contract_sha256=contract_sha256,
            device=args.device,
            expected_input_binding=binding,
            dataset_name="cifar10n",
            num_labels=10,
            diagnostic_prefix_steps=PREFIX_STEPS,
        )
        summaries[job.key] = transport.run_bundle(request)
        manifest = job.output / "sha256_manifest.json"
        before = manifest.read_bytes()
        resume_exact = resume_exact and transport.run_bundle(request) == summaries[job.key]
        resume_exact = resume_exact and before == manifest.read_bytes()
        trajectories[job.key] = _read_jsonl(job.output / "trajectory_metrics.jsonl")
    mechanism, policy, report = evaluate_confirmation(
        summaries,
        trajectories,
        contract_sha256=contract_sha256,
        expected_source_commit=current_commit,
        policy_decision=policy_decision,
        resume_hashes_exact=resume_exact,
    )
    _atomic_json(args.output / "CIFAR10N_MECHANISM_DECISION.json", mechanism)
    _atomic_json(args.output / "CIFAR10N_POLICY_DECISION.json", policy)
    _atomic_json(args.output / "CIFAR10N_CONFIRMATION_REPORT.json", report)
    return {"mechanism": mechanism, "policy": policy, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--sealed-root", type=Path, required=True)
    execute.add_argument("--policy-decision", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--cifar-root", type=Path, required=True)
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(run_preflight(args.output), indent=2, sort_keys=True))
    else:
        result = run(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["mechanism"]["decision"] != "GO":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
