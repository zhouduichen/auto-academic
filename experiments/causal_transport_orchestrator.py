"""Run the frozen Gate A/B causal transport matrices in fail-closed order."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from experiments import causal_attribution_replay as replay
from experiments import causal_transport_bundle as bundle
from experiments import causal_transport_gate as gate
from experiments import m1_calibration_clean as clean
from experiments.m1_noisy_data import _sha, validate_public
from experiments.m1_pilot import source_commit as current_source_commit


@dataclass(frozen=True)
class TransportJob:
    kind: Literal["bundle", "legacy-replay"]
    stage: Literal["A", "B"]
    seed: int
    dose: int
    continuation: Literal["clean", "noisy"]
    method: Literal["adamw", "cadam"]
    output: Path
    clean_checkpoint: Path | None = None
    noisy_checkpoint: Path | None = None

    @property
    def key(self) -> str:
        return (
            f"{self.stage}:{self.kind}:{self.method}:seed{self.seed}:"
            f"dose{self.dose}:{self.continuation}"
        )


def build_gate_a_matrix(output: Path, legacy_m1_root: Path) -> list[TransportJob]:
    jobs: list[TransportJob] = []
    for seed in (301, 302):
        for continuation in ("clean", "noisy"):
            jobs.append(
                TransportJob(
                    kind="bundle",
                    stage="A",
                    seed=seed,
                    dose=1,
                    continuation=continuation,
                    method="cadam",
                    output=output
                    / "gate-a"
                    / f"local-seed{seed}-dose1-{continuation}",
                )
            )
        for continuation in ("clean", "noisy"):
            jobs.append(
                TransportJob(
                    kind="legacy-replay",
                    stage="A",
                    seed=seed,
                    dose=1250,
                    continuation=continuation,
                    method="cadam",
                    output=output
                    / "gate-a"
                    / f"accumulated-seed{seed}-epoch20-{continuation}",
                    clean_checkpoint=legacy_m1_root
                    / "cadam"
                    / f"clean-seed{seed}"
                    / "checkpoint.pt",
                    noisy_checkpoint=legacy_m1_root
                    / "cadam"
                    / f"noisy-seed{seed}"
                    / "checkpoint.pt",
                )
            )
    return jobs


def build_gate_b_matrix(output: Path) -> list[TransportJob]:
    return [
        TransportJob(
            kind="bundle",
            stage="B",
            seed=seed,
            dose=dose,
            continuation=continuation,
            method="adamw",
            output=output / "gate-b" / f"seed{seed}-dose{dose}-{continuation}",
        )
        for seed in (401, 402, 403)
        for dose in bundle.FROZEN_DOSES
        for continuation in ("clean", "noisy")
    ]


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def write_contract(output: Path, payload: dict[str, object]) -> dict[str, object]:
    core = {
        "schema": "causal-transport-contract/1",
        **payload,
        "seeds": {"A": [301, 302], "B": [401, 402, 403]},
        "doses": {"A": [1, 1250], "B": list(bundle.FROZEN_DOSES)},
        "continuations": ["clean", "noisy"],
        "warmup_steps": 500,
        "horizons": list(replay.HORIZONS),
        "primary_outcomes": list(gate.PRIMARY_OUTCOMES),
        "ratio_thresholds": {"state": 1.0, "parameter": -1.0},
        "test_loaded": False,
    }
    record = {
        **core,
        "contract_sha256": hashlib.sha256(_canonical(core)).hexdigest(),
    }
    path = output / "CAUSAL_TRANSPORT_CONTRACT.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing causal transport contract is unreadable") from error
        if existing != record:
            raise RuntimeError("existing causal transport contract differs")
        return record
    output.mkdir(parents=True, exist_ok=True)
    clean._write_json(path, record)
    return record


def _effect_dict(summary: dict[str, object], outcome: str) -> dict[str, float]:
    effects = summary.get("effects")
    if not isinstance(effects, dict) or not isinstance(effects.get(outcome), dict):
        raise RuntimeError("continuation summary effects are incomplete")
    raw = effects[outcome]
    if set(raw) != {"parameter", "state", "interaction"}:
        raise RuntimeError("continuation effect schema is malformed")
    return {key: float(raw[key]) for key in ("parameter", "state", "interaction")}


def combine_continuations(
    clean_summary: dict[str, object], noisy_summary: dict[str, object]
) -> dict[str, object]:
    if any(
        summary.get("status") != "succeeded" or summary.get("test_loaded") is not False
        for summary in (clean_summary, noisy_summary)
    ):
        raise RuntimeError("continuation summary is not a valid success")
    if clean_summary.get("source_commit") != noisy_summary.get("source_commit"):
        raise RuntimeError("continuation source commits differ")
    clean_contract = clean_summary.get("contract_sha256")
    noisy_contract = noisy_summary.get("contract_sha256")
    if clean_contract is not None and clean_contract != noisy_contract:
        raise RuntimeError("continuation contracts differ")
    combined_effects: dict[str, dict[str, float]] = {}
    classifications: dict[str, dict[str, str]] = {}
    agree = True
    for outcome in gate.PRIMARY_OUTCOMES:
        left = _effect_dict(clean_summary, outcome)
        right = _effect_dict(noisy_summary, outcome)
        left_class = gate.classify_effect(left)
        right_class = gate.classify_effect(right)
        classifications[outcome] = {"clean": left_class, "noisy": right_class}
        if left_class != right_class:
            agree = False
            combined_effects[outcome] = {
                "parameter": 1.0,
                "state": 1.0,
                "interaction": 1.0,
            }
        else:
            # Retain the continuation closest to the classification boundary.
            left_ratio = abs(gate.carrier_ratio(left))
            right_ratio = abs(gate.carrier_ratio(right))
            combined_effects[outcome] = left if left_ratio <= right_ratio else right
    return {
        "status": "succeeded",
        "source_commit": clean_summary["source_commit"],
        "contract_sha256": clean_contract,
        "effects": combined_effects,
        "continuation_classes": classifications,
        "continuations_agree": agree,
        "wall_clock_seconds": float(clean_summary.get("wall_clock_seconds", 0.0))
        + float(noisy_summary.get("wall_clock_seconds", 0.0)),
        "test_loaded": False,
    }


def _validate_summary_set(
    jobs: list[TransportJob],
    summaries: dict[str, dict[str, object]],
    *,
    expected_count: int,
    contract_sha256: str,
) -> None:
    if len(jobs) != expected_count or set(summaries) != {job.key for job in jobs}:
        raise RuntimeError(f"stage requires exactly {expected_count} completed bundles")
    for job in jobs:
        summary = summaries[job.key]
        if summary.get("status") != "succeeded" or summary.get("test_loaded") is not False:
            raise RuntimeError("stage contains an invalid bundle summary")
        if job.kind == "bundle" and summary.get("contract_sha256") != contract_sha256:
            raise RuntimeError("bundle summary contract mismatch")


def _two_continuations(
    jobs: list[TransportJob],
    summaries: dict[str, dict[str, object]],
    *,
    seed: int,
    dose: int,
    kind: str,
) -> dict[str, object]:
    selected = {
        job.continuation: summaries[job.key]
        for job in jobs
        if job.seed == seed and job.dose == dose and job.kind == kind
    }
    if set(selected) != {"clean", "noisy"}:
        raise RuntimeError("anchor is missing a continuation")
    return combine_continuations(selected["clean"], selected["noisy"])


def collect_gate_a(
    jobs: list[TransportJob],
    summaries: dict[str, dict[str, object]],
    *,
    contract_sha256: str,
) -> dict[str, object]:
    _validate_summary_set(
        jobs, summaries, expected_count=8, contract_sha256=contract_sha256
    )
    records = {
        str(seed): {
            "local": _two_continuations(
                jobs, summaries, seed=seed, dose=1, kind="bundle"
            ),
            "accumulated": _two_continuations(
                jobs, summaries, seed=seed, dose=1250, kind="legacy-replay"
            ),
        }
        for seed in (301, 302)
    }
    decision = gate.evaluate_stage(records, (301, 302), "A")
    return {
        **decision,
        "contract_sha256": contract_sha256,
        "bundle_count": 8,
        "wall_clock_seconds": sum(
            float(summary.get("wall_clock_seconds", 0.0)) for summary in summaries.values()
        ),
    }


def collect_gate_b(
    jobs: list[TransportJob],
    summaries: dict[str, dict[str, object]],
    *,
    contract_sha256: str,
) -> dict[str, object]:
    _validate_summary_set(
        jobs, summaries, expected_count=24, contract_sha256=contract_sha256
    )
    records = {
        str(seed): {
            "local": _two_continuations(
                jobs, summaries, seed=seed, dose=1, kind="bundle"
            ),
            "accumulated": _two_continuations(
                jobs, summaries, seed=seed, dose=1250, kind="bundle"
            ),
        }
        for seed in (401, 402, 403)
    }
    decision = gate.evaluate_stage(records, (401, 402, 403), "B")
    return {
        **decision,
        "contract_sha256": contract_sha256,
        "bundle_count": 24,
        "wall_clock_seconds": sum(
            float(summary.get("wall_clock_seconds", 0.0)) for summary in summaries.values()
        ),
    }


def _binding(noise_bundle: Path, image_store: Path, tuning_store: Path) -> dict[str, object]:
    public = validate_public(noise_bundle, image_store, tuning_store)
    return {
        "noise_bundle_sha256": _sha(noise_bundle / "manifest.json"),
        "image_store_sha256": _sha(image_store / "manifest.json"),
        "tuning_store_sha256": _sha(tuning_store / "manifest.json"),
        "source_commit": public["source_commit"],
        "test_loaded": False,
    }


def _run_job(
    job: TransportJob,
    args: argparse.Namespace,
    noise_bundles: dict[int, Path],
    contract_sha256: str,
    source_commit: str,
) -> dict[str, object]:
    noise_bundle = noise_bundles[job.seed]
    if job.kind == "legacy-replay":
        if job.clean_checkpoint is None or job.noisy_checkpoint is None:
            raise RuntimeError("legacy replay checkpoint path is missing")
        summary = replay.run_replay(
            replay.ReplayRequest(
                clean_checkpoint=job.clean_checkpoint,
                noisy_checkpoint=job.noisy_checkpoint,
                continuation=job.continuation,
                seed=job.seed,  # type: ignore[arg-type]
                checkpoint_epoch=20,
                method="cadam",
                data_dir=args.data_dir,
                image_store=args.image_store,
                tuning_store=args.tuning_store,
                noisy_bundle=noise_bundle,
                output=job.output,
                source_commit=source_commit,
                device=args.device,
            )
        )
        return {**summary, "contract_sha256": contract_sha256}
    config = clean.Config(
        seed=job.seed,
        augmentation_seed=10_000 + job.seed,
        epochs=1,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.1,
        workers=args.workers,
        device=args.device,
        method=job.method,
    )
    return bundle.run_bundle(
        bundle.TransportRequest(
            method=job.method,
            seed=job.seed,
            dose=job.dose,
            warmup_steps=500,
            continuation=job.continuation,
            config=config,
            data_dir=args.data_dir,
            image_store=args.image_store,
            tuning_store=args.tuning_store,
            noisy_bundle=noise_bundle,
            output=job.output,
            source_commit=source_commit,
            contract_sha256=contract_sha256,
            device=args.device,
            expected_input_binding=_binding(
                noise_bundle, args.image_store, args.tuning_store
            ),
        )
    )


def run_stage(
    jobs: list[TransportJob],
    args: argparse.Namespace,
    noise_bundles: dict[int, Path],
    contract_sha256: str,
    source_commit: str,
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for job in jobs:
        summaries[job.key] = _run_job(
            job, args, noise_bundles, contract_sha256, source_commit
        )
    return summaries


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
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    source_commit = current_source_commit()
    noise_bundles = _parse_noise_bundles(args.noise_bundle)
    needed = {301, 302} if args.stage == "A" else {401, 402, 403}
    if args.stage == "all":
        needed = {301, 302, 401, 402, 403}
    if set(noise_bundles) != needed:
        raise RuntimeError(f"noise bundles must contain exact seeds {sorted(needed)}")
    store_bindings = {
        str(seed): _binding(path, args.image_store, args.tuning_store)
        for seed, path in sorted(noise_bundles.items())
    }
    contract = write_contract(
        args.output,
        {
            "source_commit": source_commit,
            "uv_lock_sha256": _sha(Path(__file__).resolve().parents[1] / "uv.lock"),
            "store_bindings": store_bindings,
            "legacy_discovery_decision_sha256": _sha(args.discovery_decision),
        },
    )
    contract_sha256 = str(contract["contract_sha256"])
    result: dict[str, object] = {"contract": contract}
    if args.stage in {"A", "all"}:
        decision_path = args.output / "GATE_A_DECISION.json"
        decision_path.unlink(missing_ok=True)
        jobs_a = build_gate_a_matrix(args.output, args.legacy_m1_root)
        summaries_a = run_stage(
            jobs_a, args, noise_bundles, contract_sha256, source_commit
        )
        decision_a = collect_gate_a(
            jobs_a, summaries_a, contract_sha256=contract_sha256
        )
        gate.write_decision(decision_path, decision_a)
        result["gate_a"] = decision_a
        if decision_a["decision"] != "GO":
            return result
    if args.stage in {"B", "all"}:
        decision_path = args.output / "GATE_B_DECISION.json"
        decision_path.unlink(missing_ok=True)
        jobs_b = build_gate_b_matrix(args.output)
        summaries_b = run_stage(
            jobs_b, args, noise_bundles, contract_sha256, source_commit
        )
        decision_b = collect_gate_b(
            jobs_b, summaries_b, contract_sha256=contract_sha256
        )
        gate.write_decision(decision_path, decision_b)
        result["gate_b"] = decision_b
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("A", "B", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-m1-root", type=Path, required=True)
    parser.add_argument("--discovery-decision", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-store", type=Path, required=True)
    parser.add_argument("--tuning-store", type=Path, required=True)
    parser.add_argument("--noise-bundle", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
