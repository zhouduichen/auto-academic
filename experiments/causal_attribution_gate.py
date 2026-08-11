"""Orchestrate and mechanically gate optimizer-state causal attribution."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from experiments import causal_attribution_evidence as evidence
from experiments import causal_attribution_replay as replay
from experiments import m1_calibration_clean as clean
from experiments import m1_calibration_noisy as noisy
from experiments import m1_noisy_data as noise_data
from experiments.m0_optimizer_state import m0_run as m0
from experiments.m1_pilot import source_commit


@dataclass(frozen=True)
class CaptureCell:
    seed: Literal[301, 302]
    condition: Literal["clean", "noisy"]


@dataclass(frozen=True)
class AttributionCell:
    method: Literal["adamw", "cadam"]
    seed: Literal[301, 302]
    epoch: Literal[1, 2, 3, 20]
    continuation: Literal["clean", "noisy"]


def build_capture_matrix() -> tuple[CaptureCell, ...]:
    return tuple(
        CaptureCell(seed, condition)
        for seed in (301, 302)
        for condition in ("clean", "noisy")
    )


def build_mandatory_matrix() -> tuple[AttributionCell, ...]:
    return tuple(
        AttributionCell("adamw", seed, epoch, continuation)
        for seed in (301, 302)
        for epoch in (1, 2, 3)
        for continuation in ("clean", "noisy")
    )


def build_cadam_matrix() -> tuple[AttributionCell, ...]:
    return tuple(
        AttributionCell("cadam", seed, 20, continuation)
        for seed in (301, 302)
        for continuation in ("clean", "noisy")
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid causal attribution JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"causal attribution JSON must be an object: {path}")
    return value


def _actual_input_binding(
    training: Path, image_store: Path, tuning_store: Path, public: dict[str, object]
) -> dict[str, object]:
    return {
        "noise_bundle_sha256": noise_data._sha(training / "manifest.json"),
        "image_store_sha256": noise_data._sha(image_store / "manifest.json"),
        "tuning_store_sha256": noise_data._sha(tuning_store / "manifest.json"),
        "source_commit": public["source_commit"],
        "test_loaded": False,
    }


def _validate_novelty_audit(path: Path, commit: str) -> dict[str, object]:
    audit = _read_json(path)
    sources = audit.get("primary_sources")
    if (
        audit.get("status") != "passed"
        or audit.get("source_commit") != commit
        or audit.get("same_crossed_noisy_peft_method_found") is not False
        or audit.get("test_loaded") is not False
        or not isinstance(sources, list)
        or len(sources) < 3
        or any(not isinstance(item, str) or not item.startswith("https://") for item in sources)
    ):
        raise RuntimeError("novelty audit is missing, stale, or malformed")
    return audit


def validate_inputs(args: argparse.Namespace, commit: str) -> dict[str, object]:
    provenance = m0.build_provenance(torch.device("cuda"))
    if provenance["source_commit"] != commit:
        raise RuntimeError("gate runtime provenance mismatch")
    _, prediction = evidence.write_legacy_evidence(
        args.legacy_m06_root,
        args.legacy_m1_root,
        args.legacy_m11_decision,
        args.output / "evidence",
    )
    sentinel = _read_json(args.sentinel)
    required = sentinel.get("functional_checks")
    resources = sentinel.get("resource_checks")
    required_keys = {
        "checkpoint_schema",
        "step_clock_match",
        "factorial_carriers_exact",
        "deterministic_replay",
        "nonfinite_rejected",
        "test_isolated",
    }
    if (
        sentinel.get("status") != "passed"
        or sentinel.get("source_commit") != commit
        or sentinel.get("test_loaded") is not False
        or not isinstance(required, dict)
        or set(required) != required_keys
        or not all(value is True for value in required.values())
        or not isinstance(resources, dict)
        or not resources
        or not all(value is True for value in resources.values())
    ):
        raise RuntimeError("causal attribution sentinel is invalid")
    bindings: dict[str, dict[str, object]] = {}
    for seed, training in ((301, args.bundle_301), (302, args.bundle_302)):
        public = noise_data.validate_public(training, args.image_store, args.tuning_store)
        actual = _actual_input_binding(training, args.image_store, args.tuning_store, public)
        frozen = _read_json(
            args.legacy_m1_root / "adamw" / f"noisy-seed{seed}" / "input_binding.json"
        )
        noisy._validate_input_provenance(public, provenance, actual, frozen)
        bindings[str(seed)] = frozen
    novelty = _validate_novelty_audit(args.output / "NOVELTY_AUDIT.json", commit)
    return {
        "source_commit": commit,
        "provenance": provenance,
        "prediction": prediction,
        "input_bindings": bindings,
        "sentinel": sentinel,
        "novelty_audit": novelty,
        "test_loaded": False,
    }


def _capture_config(cell: CaptureCell) -> clean.Config:
    return clean.Config(
        seed=cell.seed,
        split_seed=20_260_801,
        augmentation_seed=10_000 + cell.seed,
        epochs=3,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        max_grad_norm=None,
        model_id="google/vit-base-patch16-224-in21k",
        lora_rank=8,
        workers=4,
        device="cuda",
        method="adamw",
        checkpoint_epochs=(1, 2, 3),
    )


def _validate_capture_result(
    summary: dict[str, object], output: Path, config: clean.Config, condition: str, commit: str
) -> None:
    expected_config = json.loads(json.dumps(asdict(config)))
    if (
        summary.get("status") != "succeeded"
        or summary.get("condition") != condition
        or summary.get("seed") != config.seed
        or summary.get("source_commit") != commit
        or summary.get("test_loaded") is not False
        or _read_json(output / "config.json") != expected_config
    ):
        raise RuntimeError("capture result failed provenance validation")
    for epoch in (1, 2, 3):
        if not (output / f"checkpoint-epoch{epoch}.pt").is_file():
            raise RuntimeError("capture result is missing an archived checkpoint")


def run_capture_cell(
    cell: CaptureCell, args: argparse.Namespace, context: dict[str, object]
) -> dict[str, object]:
    commit = str(context["source_commit"])
    config = _capture_config(cell)
    output = args.output / "captures" / "adamw" / f"{cell.condition}-seed{cell.seed}"
    if cell.condition == "clean":
        summary = clean.run(config, args.data_dir, output)
    else:
        binding = context["input_bindings"]
        if not isinstance(binding, dict) or not isinstance(binding.get(str(cell.seed)), dict):
            raise RuntimeError("validated noisy input binding is unavailable")
        bundle = args.bundle_301 if cell.seed == 301 else args.bundle_302
        summary = noisy.run(
            config,
            bundle,
            args.image_store,
            args.tuning_store,
            output,
            expected_input_binding=binding[str(cell.seed)],
        )
    _validate_capture_result(summary, output, config, cell.condition, commit)
    return summary


def _cell_output(root: Path, cell: AttributionCell) -> Path:
    return (
        root
        / "replays"
        / cell.method
        / f"seed{cell.seed}"
        / f"epoch{cell.epoch}"
        / cell.continuation
    )


def run_replay_cell(
    cell: AttributionCell, args: argparse.Namespace, context: dict[str, object]
) -> dict[str, object]:
    if cell.method == "adamw":
        root = args.output / "captures" / "adamw"
        clean_checkpoint = root / f"clean-seed{cell.seed}" / f"checkpoint-epoch{cell.epoch}.pt"
        noisy_checkpoint = root / f"noisy-seed{cell.seed}" / f"checkpoint-epoch{cell.epoch}.pt"
    else:
        root = args.legacy_m1_root / "cadam"
        clean_checkpoint = root / f"clean-seed{cell.seed}" / "checkpoint.pt"
        noisy_checkpoint = root / f"noisy-seed{cell.seed}" / "checkpoint.pt"
    bundle = args.bundle_301 if cell.seed == 301 else args.bundle_302
    request = replay.ReplayRequest(
        clean_checkpoint=clean_checkpoint,
        noisy_checkpoint=noisy_checkpoint,
        continuation=cell.continuation,
        seed=cell.seed,
        checkpoint_epoch=cell.epoch,
        method=cell.method,
        data_dir=args.data_dir,
        image_store=args.image_store,
        tuning_store=args.tuning_store,
        noisy_bundle=bundle,
        output=_cell_output(args.output, cell),
        source_commit=str(context["source_commit"]),
    )
    return replay.run_replay(request)


def _dominant(values: dict[str, float], *, require_effect_rule: bool) -> tuple[str, int] | None:
    if set(values) != {"parameter", "state", "interaction"} or any(
        isinstance(value, bool) or not math.isfinite(value) for value in values.values()
    ):
        return None
    magnitudes = {key: abs(value) for key, value in values.items()}
    maximum = max(magnitudes.values())
    winners = [key for key, value in magnitudes.items() if value == maximum]
    if len(winners) != 1 or maximum == 0:
        return None
    component = winners[0]
    if require_effect_rule:
        if component == "interaction":
            eligible = maximum >= magnitudes["parameter"] and maximum >= magnitudes["state"]
        else:
            other = "state" if component == "parameter" else "parameter"
            eligible = maximum >= 2 * magnitudes[other]
        if not eligible:
            return None
    sign = 1 if values[component] > 0 else -1
    return component, sign


def _effect_table(record: dict[str, object]) -> dict[str, dict[str, dict[str, float]]] | None:
    effects = record.get("effects")
    if not isinstance(effects, dict) or set(effects) != {"301", "302"}:
        return None
    result: dict[str, dict[str, dict[str, float]]] = {}
    for seed in ("301", "302"):
        epochs = effects.get(seed)
        if not isinstance(epochs, dict) or set(epochs) != {"1", "2", "3"}:
            return None
        result[seed] = {}
        for epoch in ("1", "2", "3"):
            values = epochs.get(epoch)
            if not isinstance(values, dict) or set(values) != {
                "parameter",
                "state",
                "interaction",
            }:
                return None
            converted: dict[str, float] = {}
            for key, value in values.items():
                if isinstance(value, bool) or not isinstance(value, int | float):
                    return None
                converted_value = float(value)
                if not math.isfinite(converted_value):
                    return None
                converted[key] = converted_value
            result[seed][epoch] = converted
    return result


def evaluate_attribution(
    record: dict[str, object], *, require_cadam: bool
) -> dict[str, object]:
    result = deepcopy(record)
    effects = _effect_table(record)
    finite_and_isolated = effects is not None and record.get("test_loaded") is False
    prediction = record.get("prediction")
    checkpoint_predictions = (
        prediction.get("checkpoint_predictions") if isinstance(prediction, dict) else None
    )
    if not isinstance(checkpoint_predictions, dict):
        checkpoint_predictions = {}

    raw_matches: dict[str, list[int]] = {key: [] for key in ("parameter", "state", "interaction")}
    eligible_matches: dict[str, list[int]] = {
        key: [] for key in ("parameter", "state", "interaction")
    }
    predicted_matches: dict[str, list[int]] = {
        key: [] for key in ("parameter", "state", "interaction")
    }
    if effects is not None:
        for epoch in (1, 2, 3):
            raw = [
                _dominant(effects[seed][str(epoch)], require_effect_rule=False)
                for seed in ("301", "302")
            ]
            eligible = [
                _dominant(effects[seed][str(epoch)], require_effect_rule=True)
                for seed in ("301", "302")
            ]
            if raw[0] is not None and raw[0] == raw[1]:
                raw_matches[raw[0][0]].append(epoch)
            if eligible[0] is not None and eligible[0] == eligible[1]:
                component = eligible[0][0]
                eligible_matches[component].append(epoch)
                if checkpoint_predictions.get(str(epoch)) == component:
                    predicted_matches[component].append(epoch)
    selected_component = next(
        (
            component
            for component in ("parameter", "state", "interaction")
            if len(predicted_matches[component]) >= 2
        ),
        None,
    )
    replicated = any(len(values) >= 2 for values in raw_matches.values())
    effect_rule = any(len(values) >= 2 for values in eligible_matches.values())
    prediction_matches = selected_component is not None
    cadam = record.get("cadam_confirmation")
    cadam_matches = not require_cadam
    if require_cadam and selected_component is not None and isinstance(cadam, dict):
        cadam_matches = set(cadam) == {"301", "302"} and all(
            cadam[seed] == selected_component for seed in ("301", "302")
        )
    gpu_hours = record.get("gpu_hours")
    gpu_ok = (
        not isinstance(gpu_hours, bool)
        and isinstance(gpu_hours, int | float)
        and math.isfinite(float(gpu_hours))
        and 0 < float(gpu_hours) <= 8.0
    )
    checks = {
        "mandatory_complete": record.get("mandatory_artifacts") == 12,
        "finite_and_isolated": finite_and_isolated,
        "replicated_at_two_checkpoints": replicated,
        "twofold_or_interaction_rule": effect_rule,
        "legacy_prediction_matches": prediction_matches,
        "novelty_audit_passed": record.get("novelty_audit_passed") is True,
        "cadam_matches_when_required": cadam_matches,
        "gpu_hours_at_most_8": gpu_ok,
    }
    result["selected_component"] = selected_component
    result["qualifying_checkpoints"] = (
        predicted_matches[selected_component] if selected_component is not None else []
    )
    result["checks"] = checks
    result["decision"] = "GO" if all(checks.values()) else "NO-GO"
    return result


def _capture_elapsed(output: Path, seed: int, condition: str) -> float:
    path = output / "captures" / "adamw" / f"{condition}-seed{seed}" / "epoch_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 3 or rows[-1].get("test_loaded") is not False:
        raise RuntimeError("capture timing rows are incomplete")
    return float(rows[-1]["elapsed_training_seconds"])


def collect_record(
    args: argparse.Namespace,
    context: dict[str, object],
    *,
    include_cadam: bool,
) -> dict[str, object]:
    summaries: dict[tuple[str, int, int, str], dict[str, object]] = {}
    wall_seconds = sum(
        _capture_elapsed(args.output, seed, condition)
        for seed in (301, 302)
        for condition in ("clean", "noisy")
    )
    cells = list(build_mandatory_matrix())
    if include_cadam:
        cells.extend(build_cadam_matrix())
    for cell in cells:
        summary = _read_json(_cell_output(args.output, cell) / "summary.json")
        if (
            summary.get("status") != "succeeded"
            or summary.get("source_commit") != context["source_commit"]
            or summary.get("test_loaded") is not False
        ):
            raise RuntimeError("replay summary is incomplete or stale")
        wall_seconds += float(summary["wall_clock_seconds"])
        summaries[(cell.method, cell.seed, cell.epoch, cell.continuation)] = summary
    sentinel = context.get("sentinel")
    if isinstance(sentinel, dict):
        wall_seconds += float(sentinel.get("elapsed_seconds", 0.0))

    effects: dict[str, dict[str, dict[str, float]]] = {"301": {}, "302": {}}
    for seed in (301, 302):
        for epoch in (1, 2, 3):
            summary = summaries[("adamw", seed, epoch, "noisy")]
            raw_effects = summary.get("effects")
            primary = raw_effects.get("tuning_loss_h512") if isinstance(raw_effects, dict) else None
            if not isinstance(primary, dict):
                raise RuntimeError("primary factorial effects are missing")
            effects[str(seed)][str(epoch)] = {
                component: float(primary[component])
                for component in ("parameter", "state", "interaction")
            }
    cadam_confirmation: dict[str, str | None] = {}
    if include_cadam:
        for seed in (301, 302):
            summary = summaries[("cadam", seed, 20, "noisy")]
            raw_effects = summary.get("effects")
            primary = raw_effects.get("tuning_loss_h512") if isinstance(raw_effects, dict) else None
            dominant = _dominant(
                {key: float(primary[key]) for key in ("parameter", "state", "interaction")}
                if isinstance(primary, dict)
                else {},
                require_effect_rule=True,
            )
            cadam_confirmation[str(seed)] = dominant[0] if dominant is not None else None
    novelty = context.get("novelty_audit")
    return {
        "schema": "causal-attribution-decision-record/1",
        "source_commit": context["source_commit"],
        "mandatory_artifacts": len(build_mandatory_matrix()),
        "effects": effects,
        "prediction": context["prediction"],
        "cadam_confirmation": cadam_confirmation,
        "novelty_audit_passed": isinstance(novelty, dict) and novelty.get("status") == "passed",
        "gpu_hours": wall_seconds / 3600,
        "test_loaded": False,
    }


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    final_path = args.output / "CAUSAL_ATTRIBUTION_DECISION.json"
    final_path.unlink(missing_ok=True)
    commit = source_commit()
    context = validate_inputs(args, commit)
    for cell in build_capture_matrix():
        run_capture_cell(cell, args, context)
    for cell in build_mandatory_matrix():
        run_replay_cell(cell, args, context)
    adamw_record = collect_record(args, context, include_cadam=False)
    adamw_decision = evaluate_attribution(adamw_record, require_cadam=False)
    adamw_decision["source_commit"] = commit
    clean._write_json(args.output / "ADAMW_ATTRIBUTION_DECISION.json", adamw_decision)
    if adamw_decision["decision"] == "GO":
        for cell in build_cadam_matrix():
            run_replay_cell(cell, args, context)
        final_record = collect_record(args, context, include_cadam=True)
        final = evaluate_attribution(final_record, require_cadam=True)
    else:
        final = adamw_decision
    final["source_commit"] = commit
    clean._write_json(final_path, final)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-store", type=Path, required=True)
    parser.add_argument("--tuning-store", type=Path, required=True)
    parser.add_argument("--bundle-301", type=Path, required=True)
    parser.add_argument("--bundle-302", type=Path, required=True)
    parser.add_argument("--legacy-m06-root", type=Path, required=True)
    parser.add_argument("--legacy-m1-root", type=Path, required=True)
    parser.add_argument("--legacy-m11-decision", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_gate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
