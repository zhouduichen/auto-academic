"""Run exactly two M1.1 cells and write one mechanical rescue decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments import m1_calibration_clean as clean
from experiments import m1_calibration_noisy as noisy
from experiments.m1_pilot import source_commit

SEED = 301
EPOCHS = 3
STEPS_PER_EPOCH = 1250
WARMUP_STEPS = 1250


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _two_resource_ratios(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    ratios = [_finite_float(item) for item in value]
    if any(item is None or item <= 0 for item in ratios):
        return None
    return [float(item) for item in ratios if item is not None]


def evaluate_rescue(record: dict[str, object]) -> dict[str, object]:
    """Return a deterministic GO/NO-GO decision without mutating ``record``."""
    clean_difference = _finite_float(record.get("clean_difference_pp"))
    noisy_difference = _finite_float(record.get("noisy_difference_pp"))
    rejected = _nonnegative_int(record.get("noisy_rejected_steps"))
    post = _nonnegative_int(record.get("post_warmup_steps"))
    max_consecutive = _nonnegative_int(record.get("max_consecutive_rejections"))
    wallclock = _two_resource_ratios(record.get("wallclock_ratios"))
    vram = _two_resource_ratios(record.get("vram_ratios"))
    valid_rate = rejected is not None and post is not None and post > 0
    checks = {
        "clean_drop_at_most_2pp": clean_difference is not None
        and clean_difference >= -2.0,
        "noisy_at_least_cadam": noisy_difference is not None
        and noisy_difference >= 0.0,
        "noisy_has_rejection": rejected is not None and rejected >= 1,
        "noisy_rejection_rate_at_most_5pct": valid_rate
        and rejected / post <= 0.05,
        "no_consecutive_rejections": max_consecutive is not None
        and max_consecutive <= 1,
        "wallclock_overhead_at_most_5pct": wallclock is not None
        and all(value <= 1.05 for value in wallclock),
        "vram_overhead_at_most_5pct": vram is not None
        and all(value <= 1.05 for value in vram),
        "test_not_loaded": record.get("test_loaded") is False,
    }
    return {
        **record,
        "checks": checks,
        "decision": "GO" if all(checks.values()) else "NO-GO",
    }


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{description} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object")
    return value


def validate_m11_sentinel(path: Path, current_commit: str) -> dict[str, object]:
    value = _read_json_object(path, "M1.1 sentinel pass artifact")
    if value.get("status") != "passed":
        raise RuntimeError("M1.1 sentinel did not pass")
    if value.get("source_commit") != current_commit:
        raise RuntimeError("M1.1 sentinel source commit mismatch")
    if value.get("test_loaded") is not False:
        raise RuntimeError("M1.1 sentinel test isolation failed")
    functional = value.get("functional")
    required_functional = {
        "disabled_protection_parity",
        "isolated_rejection",
        "resume",
        "deterministic_replay",
        "nonfinite_rejected_atomically",
        "passed",
    }
    if not isinstance(functional, dict) or any(
        functional.get(key) is not True for key in required_functional
    ):
        raise RuntimeError("M1.1 sentinel functional gate failed")
    time_ratio = _finite_float(value.get("time_ratio"))
    vram_ratio = _finite_float(value.get("vram_ratio"))
    if (
        time_ratio is None
        or vram_ratio is None
        or time_ratio <= 0
        or vram_ratio <= 0
        or time_ratio > 1.05
        or vram_ratio > 1.05
    ):
        raise RuntimeError("M1.1 sentinel resource gate failed")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"{description} is missing: {path}")
    rows: list[dict[str, object]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{description} contains a non-object row")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{description} is unreadable: {path}") from error
    if not rows:
        raise RuntimeError(f"{description} is empty")
    return rows


def _cell_epoch3(
    cell: Path,
    *,
    method: str,
    condition: str,
    current_commit: str | None = None,
) -> dict[str, object]:
    summary = _read_json_object(cell / "summary.json", "cell summary")
    metrics = _read_jsonl(cell / "epoch_metrics.jsonl", "cell epoch metrics")
    expected_epochs = EPOCHS if current_commit is not None else None
    if (
        summary.get("status") != "succeeded"
        or summary.get("method") != method
        or summary.get("condition") != condition
        or summary.get("seed") != SEED
        or summary.get("test_loaded") is not False
        or _nonnegative_int(summary.get("epochs")) is None
        or int(summary["epochs"]) < EPOCHS
        or (expected_epochs is not None and summary.get("epochs") != expected_epochs)
    ):
        raise RuntimeError(f"invalid {method} {condition} cell summary")
    commit = summary.get("source_commit")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError(f"invalid {method} {condition} source commit")
    if current_commit is not None and commit != current_commit:
        raise RuntimeError(f"{method} {condition} source commit mismatch")
    if current_commit is not None and len(metrics) != EPOCHS:
        raise RuntimeError(f"invalid {method} {condition} metric count")
    if any(row.get("test_loaded") is not False for row in metrics):
        raise RuntimeError(f"{method} {condition} metric isolation failed")
    epoch3 = [row for row in metrics if row.get("epoch") == EPOCHS]
    if len(epoch3) != 1:
        raise RuntimeError(f"invalid {method} {condition} epoch-3 metrics")
    accuracy = _finite_float(epoch3[0].get("tuning_accuracy"))
    seconds = _finite_float(epoch3[0].get("elapsed_training_seconds"))
    peak = _finite_float(summary.get("peak_vram_gb"))
    steps = _nonnegative_int(epoch3[0].get("optimizer_steps"))
    if (
        accuracy is None
        or not 0 <= accuracy <= 1
        or seconds is None
        or seconds <= 0
        or peak is None
        or peak <= 0
        or steps != EPOCHS * STEPS_PER_EPOCH
        or (
            current_commit is not None
            and summary.get("optimizer_steps") != EPOCHS * STEPS_PER_EPOCH
        )
    ):
        raise RuntimeError(f"invalid {method} {condition} measurements")
    return {
        "accuracy": accuracy,
        "seconds": seconds,
        "peak_vram_gb": peak,
        "optimizer_steps": steps,
        "source_commit": commit,
    }


def _noisy_rejection_summary(cell: Path) -> tuple[int, int, int]:
    rows = _read_jsonl(
        cell / "optimizer_diagnostics.jsonl", "candidate noisy diagnostics"
    )
    if len(rows) != EPOCHS or [row.get("epoch") for row in rows] != [1, 2, 3]:
        raise RuntimeError("candidate noisy diagnostics must contain exactly three epochs")
    rejected = 0
    transition_off = 0
    transition_on = 0
    for row in rows:
        successful = _nonnegative_int(row.get("successful_steps"))
        row_rejected = _nonnegative_int(row.get("rejected_steps"))
        row_off = _nonnegative_int(row.get("transitions_off"))
        row_on = _nonnegative_int(row.get("transitions_on"))
        if (
            successful != STEPS_PER_EPOCH
            or row_rejected is None
            or row_rejected > successful
            or row_off != row_rejected
            or row_on is None
            or not isinstance(row.get("previous_rejected"), bool)
        ):
            raise RuntimeError("candidate noisy diagnostics are malformed")
        rejected += row_rejected
        transition_off += row_off
        transition_on += row_on
    if _nonnegative_int(rows[0].get("rejected_steps")) != 0:
        raise RuntimeError("candidate rejected during the fixed warm-up")
    ending_rejection = int(rows[-1]["previous_rejected"] is True)
    if transition_off != rejected or transition_on != rejected - ending_rejection:
        raise RuntimeError("candidate rejection isolation evidence is inconsistent")
    post_warmup = EPOCHS * STEPS_PER_EPOCH - WARMUP_STEPS
    max_consecutive = 1 if rejected else 0
    return rejected, post_warmup, max_consecutive


def run_rescue(args: argparse.Namespace) -> dict[str, object]:
    commit = source_commit()
    sentinel = validate_m11_sentinel(args.sentinel, commit)
    config = clean.Config(
        seed=SEED,
        augmentation_seed=10_000 + SEED,
        epochs=EPOCHS,
        learning_rate=3e-4,
        weight_decay=0.1,
        workers=4,
        device="cuda",
        method="protect-m11",
    )
    clean_cell = args.output / "clean-seed301"
    noisy_cell = args.output / "noisy-seed301"
    clean.run(config, args.data_dir, clean_cell)
    noisy.run(
        config,
        args.bundle_301,
        args.image_store,
        args.tuning_store,
        noisy_cell,
    )

    candidate_clean = _cell_epoch3(
        clean_cell,
        method="protect-m11",
        condition="clean",
        current_commit=commit,
    )
    candidate_noisy = _cell_epoch3(
        noisy_cell,
        method="protect-m11",
        condition="noisy",
        current_commit=commit,
    )
    adamw_clean = _cell_epoch3(
        args.baseline_root / "adamw" / "clean-seed301",
        method="adamw",
        condition="clean",
    )
    adamw_noisy = _cell_epoch3(
        args.baseline_root / "adamw" / "noisy-seed301",
        method="adamw",
        condition="noisy",
    )
    cadam_noisy = _cell_epoch3(
        args.baseline_root / "cadam" / "noisy-seed301",
        method="cadam",
        condition="noisy",
    )
    rejected, post_warmup, max_consecutive = _noisy_rejection_summary(noisy_cell)
    baseline_commits = sorted(
        {
            str(adamw_clean["source_commit"]),
            str(adamw_noisy["source_commit"]),
            str(cadam_noisy["source_commit"]),
        }
    )
    record: dict[str, object] = {
        "status": "succeeded",
        "candidate": "protect-m11",
        "seed": SEED,
        "epochs": EPOCHS,
        "clean_difference_pp": 100
        * (float(candidate_clean["accuracy"]) - float(adamw_clean["accuracy"])),
        "noisy_difference_pp": 100
        * (float(candidate_noisy["accuracy"]) - float(cadam_noisy["accuracy"])),
        "noisy_rejected_steps": rejected,
        "post_warmup_steps": post_warmup,
        "max_consecutive_rejections": max_consecutive,
        "wallclock_ratios": [
            float(candidate_clean["seconds"]) / float(adamw_clean["seconds"]),
            float(candidate_noisy["seconds"]) / float(adamw_noisy["seconds"]),
        ],
        "vram_ratios": [
            float(candidate_clean["peak_vram_gb"])
            / float(adamw_clean["peak_vram_gb"]),
            float(candidate_noisy["peak_vram_gb"])
            / float(adamw_noisy["peak_vram_gb"]),
        ],
        "source_commit": commit,
        "baseline_source_commits": baseline_commits,
        "sentinel_time_ratio": sentinel["time_ratio"],
        "sentinel_vram_ratio": sentinel["vram_ratio"],
        "test_loaded": False,
    }
    result = evaluate_rescue(record)
    args.output.mkdir(parents=True, exist_ok=True)
    clean._write_json(args.output / "M11_RESCUE_DECISION.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-store", type=Path, required=True)
    parser.add_argument("--tuning-store", type=Path, required=True)
    parser.add_argument("--bundle-301", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_rescue(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
