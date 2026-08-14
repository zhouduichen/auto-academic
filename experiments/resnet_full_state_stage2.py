"""Bounded Stage-2 completion for the full-state ResNet repair study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from experiments import causal_attribution_replay as replay
from experiments import causal_transport_bundle as bundle
from experiments import m1_calibration_clean as clean
from experiments import repair_handoff_models as repair_models
from experiments import resnet_full_state_repair as stage1
from experiments import resnet_repair_handoff_sentinel as handoff
from experiments.m1_noisy_data import _sha
from experiments.m1_pilot import source_commit as current_source_commit

SEEDS = (601, 602, 603)
DOSES = (64, 1250)
CONTINUATIONS = ("clean", "noisy")
PRIMARY_HORIZONS = (32, 64, 128, 256)
REUSED_KEYS = {"seed601-dose1250-clean", "seed601-dose1250-noisy"}


@dataclass(frozen=True)
class Stage2Job:
    seed: int
    dose: int
    continuation: str
    output: Path

    @property
    def key(self) -> str:
        return f"seed{self.seed}-dose{self.dose}-{self.continuation}"


def build_jobs(output: Path) -> list[Stage2Job]:
    return [
        Stage2Job(
            seed,
            dose,
            continuation,
            output / "matrix" / f"seed{seed}-dose{dose}-{continuation}",
        )
        for seed in SEEDS
        for dose in DOSES
        for continuation in CONTINUATIONS
    ]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_contract(
    output: Path,
    *,
    source_commit: str,
    uv_lock_sha256: str,
    model_binding: dict[str, str],
    store_bindings: dict[str, dict[str, object]],
    stage1_decision_sha256: str,
    stage1_contract_sha256: str,
    stage1_source_commit: str,
    transport_core_sha256: dict[str, str],
) -> dict[str, object]:
    handoff._validate_model_binding(model_binding)
    if (
        len(source_commit) != 40
        or len(stage1_source_commit) != 40
        or not handoff._valid_sha256(uv_lock_sha256)
        or not handoff._valid_sha256(stage1_decision_sha256)
        or not handoff._valid_sha256(stage1_contract_sha256)
        or set(store_bindings) != {str(seed) for seed in SEEDS}
        or set(transport_core_sha256)
        != {"causal_attribution_replay.py", "causal_transport_bundle.py"}
        or any(not handoff._valid_sha256(value) for value in transport_core_sha256.values())
    ):
        raise ValueError("Stage-2 provenance is malformed")
    for seed in SEEDS:
        binding = store_bindings[str(seed)]
        if (
            binding.get("noise_bundle_sha256")
            != handoff.EXPECTED_NOISE_SHA256[seed]
            or binding.get("test_loaded") is not False
        ):
            raise ValueError("Stage-2 store binding is malformed")
    core: dict[str, object] = {
        "schema": "resnet-full-state-stage2-contract/1",
        "source_commit": source_commit,
        "stage1_source_commit": stage1_source_commit,
        "stage1_contract_sha256": stage1_contract_sha256,
        "stage1_decision_sha256": stage1_decision_sha256,
        "uv_lock_sha256": uv_lock_sha256,
        "model_binding": model_binding,
        "store_bindings": store_bindings,
        "transport_core_sha256": transport_core_sha256,
        "checkpoint_schema": bundle.SNAPSHOT_SCHEMA,
        "seeds": list(SEEDS),
        "doses": list(DOSES),
        "continuations": list(CONTINUATIONS),
        "reused_keys": sorted(REUSED_KEYS),
        "new_bundle_count": 10,
        "horizons": list(replay.HORIZONS),
        "primary_horizons": list(PRIMARY_HORIZONS),
        "warmup_steps": 500,
        "batch_size": 32,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "scheduler": "none",
        },
        "test_loaded": False,
    }
    record = {
        **core,
        "contract_sha256": hashlib.sha256(_canonical(core)).hexdigest(),
    }
    path = output / "FULL_STATE_STAGE2_CONTRACT.json"
    if path.is_file():
        if stage1._read_json(path, "existing Stage-2 contract") != record:
            raise RuntimeError("existing Stage-2 contract differs")
    else:
        handoff._atomic_json(path, record)
    return record


def _margin_by_horizon(rows: list[dict[str, object]]) -> dict[int, float]:
    expected = {
        (branch, horizon) for branch in replay.BRANCH_NAMES for horizon in replay.HORIZONS
    }
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        try:
            key = (str(row["branch"]), int(row["horizon"]))
            value = float(row["clean_loss_excess"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Stage-2 trajectory is malformed") from error
        if key in indexed or not math.isfinite(value):
            raise RuntimeError("Stage-2 trajectory is malformed")
        indexed[key] = row
    if set(indexed) != expected:
        raise RuntimeError("Stage-2 trajectory matrix is incomplete")
    return {
        horizon: float(indexed[("NC", horizon)]["clean_loss_excess"])
        - float(indexed[("CN", horizon)]["clean_loss_excess"])
        for horizon in replay.HORIZONS
    }


def evaluate_stage2(
    summaries: dict[str, dict[str, object]],
    trajectories: dict[str, list[dict[str, object]]],
    *,
    stage2_contract_sha256: str,
    stage1_contract_sha256: str,
    source_commit: str,
    stage1_source_commit: str,
    resume_hashes_exact: bool,
) -> dict[str, object]:
    jobs = {job.key: job for job in build_jobs(Path("unused"))}
    exact = set(summaries) == set(trajectories) == set(jobs)
    valid = exact and resume_hashes_exact
    margins: dict[str, dict[int, float]] = {}
    if exact:
        for key, job in jobs.items():
            summary = summaries[key]
            request = summary.get("request")
            config = request.get("config") if isinstance(request, dict) else None
            reused = key in REUSED_KEYS
            valid = valid and bool(
                summary.get("status") == "succeeded"
                and summary.get("source_commit")
                == (stage1_source_commit if reused else source_commit)
                and summary.get("contract_sha256")
                == (stage1_contract_sha256 if reused else stage2_contract_sha256)
                and summary.get("branch_order") == list(replay.BRANCH_NAMES)
                and summary.get("test_loaded") is False
                and isinstance(request, dict)
                and request.get("seed") == job.seed
                and request.get("dose") == job.dose
                and request.get("continuation") == job.continuation
                and isinstance(config, dict)
                and config.get("model_id") == repair_models.RESNET_MODEL_ID
                and stage1._finite(summary)
                and stage1._finite(trajectories[key])
            )
            margins[key] = _margin_by_horizon(trajectories[key])

    contrasts: dict[str, float] = {}
    switches: dict[str, object] = {}
    if exact:
        for seed in SEEDS:
            for dose in DOSES:
                clean_key = f"seed{seed}-dose{dose}-clean"
                noisy_key = f"seed{seed}-dose{dose}-noisy"
                contrasts[f"seed{seed}-dose{dose}"] = sum(
                    margins[noisy_key][horizon] - margins[clean_key][horizon]
                    for horizon in PRIMARY_HORIZONS
                ) / len(PRIMARY_HORIZONS)
            switch_horizons = [
                horizon
                for horizon in PRIMARY_HORIZONS
                if margins[f"seed{seed}-dose1250-clean"][horizon] > 0.0
                and margins[f"seed{seed}-dose1250-noisy"][horizon] < 0.0
            ]
            switches[str(seed)] = {
                "count": len(switch_horizons),
                "horizons": switch_horizons,
            }
    context_pass = exact and all(
        contrasts[f"seed{seed}-dose1250"] < 0.0 for seed in SEEDS
    )
    switch_pass = exact and sum(
        int(switches[str(seed)]["count"]) >= 2  # type: ignore[index]
        for seed in SEEDS
    ) >= 2
    checks = {
        "exact_valid_matrix": valid,
        "dose1250_context_contrast_negative_all_seeds": context_pass,
        "dose1250_action_switch_at_least_two_seeds": switch_pass,
    }
    return {
        "schema": "resnet-full-state-stage2-decision/1",
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "bundle_count": 12,
        "reused_bundle_count": 2,
        "new_bundle_count": 10,
        "primary_context_contrasts": contrasts,
        "primary_action_switches": switches,
        "pointwise_repair_margins": {
            key: {str(horizon): value for horizon, value in values.items()}
            for key, values in margins.items()
        },
        "stage2_contract_sha256": stage2_contract_sha256,
        "stage1_contract_sha256": stage1_contract_sha256,
        "source_commit": source_commit,
        "stage1_source_commit": stage1_source_commit,
        "test_loaded": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    analysis_source_commit = current_source_commit()
    model_binding = repair_models.resnet_artifact_binding(local_files_only=True)
    stage1_decision_path = args.stage1_output / "FULL_STATE_SENTINEL_DECISION.json"
    stage1_contract_path = args.stage1_output / "FULL_STATE_CONTRACT.json"
    stage1_decision = stage1._read_json(stage1_decision_path, "Stage-1 decision")
    stage1_contract = stage1._read_json(stage1_contract_path, "Stage-1 contract")
    if (
        stage1_decision.get("decision") != "GO"
        or not all(stage1_decision.get("checks", {}).values())
        or stage1_decision.get("contract_sha256") != stage1_contract.get("contract_sha256")
    ):
        raise RuntimeError("Stage 1 did not pass exactly")
    stage1_source_commit = str(stage1_decision["source_commit"])
    stage1_contract_sha256 = str(stage1_contract["contract_sha256"])
    noise_bundles = handoff._parse_noise_bundles(args.noise_bundle)
    store_bindings = {
        str(seed): handoff._binding(path, args.image_store, args.tuning_store)
        for seed, path in sorted(noise_bundles.items())
    }
    existing_contract_path = args.output / "FULL_STATE_STAGE2_CONTRACT.json"
    scientific_source_commit = analysis_source_commit
    if existing_contract_path.is_file():
        existing_contract = stage1._read_json(
            existing_contract_path, "existing Stage-2 contract"
        )
        scientific_source_commit = str(existing_contract.get("source_commit", ""))
    preflight = stage1._read_json(args.preflight, "CUDA preflight")
    handoff.validate_preflight(
        preflight,
        source_commit=scientific_source_commit,
        model_binding=model_binding,
    )
    contract = write_contract(
        args.output,
        source_commit=scientific_source_commit,
        uv_lock_sha256=_sha(Path(__file__).resolve().parents[1] / "uv.lock"),
        model_binding=model_binding,
        store_bindings=store_bindings,
        stage1_decision_sha256=_sha(stage1_decision_path),
        stage1_contract_sha256=stage1_contract_sha256,
        stage1_source_commit=stage1_source_commit,
        transport_core_sha256={
            name: _sha(Path(__file__).with_name(name))
            for name in ("causal_attribution_replay.py", "causal_transport_bundle.py")
        },
    )
    contract_sha256 = str(contract["contract_sha256"])
    summaries: dict[str, dict[str, object]] = {}
    trajectories: dict[str, list[dict[str, object]]] = {}
    resume_exact = True
    for job in build_jobs(args.output):
        if job.key in REUSED_KEYS:
            path = args.stage1_output / "sentinel" / job.key
            summaries[job.key] = stage1._read_json(path / "summary.json", "Stage-1 summary")
            trajectories[job.key] = stage1._read_jsonl(path / "trajectory_metrics.jsonl")
            continue
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
        request = bundle.TransportRequest(
            method="adamw",
            seed=job.seed,
            dose=job.dose,
            warmup_steps=500,
            continuation=job.continuation,  # type: ignore[arg-type]
            config=config,
            data_dir=args.data_dir,
            image_store=args.image_store,
            tuning_store=args.tuning_store,
            noisy_bundle=noise_bundles[job.seed],
            output=job.output,
            source_commit=scientific_source_commit,
            contract_sha256=contract_sha256,
            device=args.device,
            expected_input_binding=store_bindings[str(job.seed)],
        )
        completed = bundle._validate_existing_bundle(request, replay.BRANCH_NAMES)
        if completed is None:
            if scientific_source_commit != analysis_source_commit:
                raise RuntimeError("a contracted Stage-2 bundle is incomplete")
            completed = bundle.run_bundle(request)
        summaries[job.key] = completed
        manifest = job.output / "sha256_manifest.json"
        before = manifest.read_bytes()
        validated = bundle._validate_existing_bundle(request, replay.BRANCH_NAMES)
        resume_exact = resume_exact and validated == completed and before == manifest.read_bytes()
        trajectories[job.key] = stage1._read_jsonl(job.output / "trajectory_metrics.jsonl")
    decision = evaluate_stage2(
        summaries,
        trajectories,
        stage2_contract_sha256=contract_sha256,
        stage1_contract_sha256=stage1_contract_sha256,
        source_commit=scientific_source_commit,
        stage1_source_commit=stage1_source_commit,
        resume_hashes_exact=resume_exact,
    )
    decision["analysis_source_commit"] = analysis_source_commit
    handoff._atomic_json(args.output / "FULL_STATE_STAGE2_DECISION.json", decision)
    return {"contract": contract, "stage2": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--stage1-output", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--image-store", type=Path, required=True)
    execute.add_argument("--tuning-store", type=Path, required=True)
    execute.add_argument("--noise-bundle", action="append", default=[])
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(handoff.run_preflight(args.output), indent=2, sort_keys=True))
        return
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["stage2"]["decision"] != "GO":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
