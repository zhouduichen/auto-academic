"""Run and evaluate the frozen 24-cell Protect-M M1 Pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.m1_calibration_clean import PILOT_METHODS, _write_json

PILOT_SEEDS = (301, 302, 303)
PILOT_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1


@dataclass(frozen=True)
class PilotCell:
    method: str
    condition: str
    seed: int

    @property
    def relative_path(self) -> Path:
        return Path(self.method) / f"{self.condition}-seed{self.seed}"


def build_matrix() -> tuple[PilotCell, ...]:
    return tuple(
        PilotCell(method, condition, seed)
        for method in PILOT_METHODS
        for condition in ("clean", "noisy")
        for seed in PILOT_SEEDS
    )


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return statistics.fmean(values)


def evaluate_candidate(record: dict[str, object]) -> dict[str, object]:
    noisy = [float(value) for value in record["noisy_differences_pp"]]
    clean = [float(value) for value in record["clean_differences_pp"]]
    fixed = [float(value) for value in record["fixed_wallclock_noisy_differences_pp"]]
    wallclock = [float(value) for value in record["wallclock_ratios"]]
    vram = [float(value) for value in record["vram_ratios"]]
    checks = {
        "three_positive_noisy_seeds": len(noisy) == 3 and all(value > 0 for value in noisy),
        "mean_noisy_at_least_1pp": len(noisy) == 3 and _mean(noisy) >= 1.0,
        "mean_clean_drop_at_most_0_5pp": len(clean) == 3 and _mean(clean) >= -0.5,
        "each_clean_drop_at_most_1pp": len(clean) == 3 and all(value >= -1.0 for value in clean),
        "fixed_wallclock_noisy_positive": len(fixed) == 3 and _mean(fixed) > 0,
        "mean_wallclock_overhead_at_most_5pct": len(wallclock) == 6
        and _mean(wallclock) <= 1.05,
        "mean_vram_overhead_at_most_5pct": len(vram) == 6 and _mean(vram) <= 1.05,
    }
    return {
        **record,
        "mean_noisy_difference_pp": _mean(noisy),
        "mean_clean_difference_pp": _mean(clean),
        "mean_fixed_wallclock_noisy_difference_pp": _mean(fixed),
        "mean_wallclock_ratio": _mean(wallclock),
        "mean_vram_ratio": _mean(vram),
        "checks": checks,
        "eligible": all(checks.values()),
    }


def source_commit() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is unavailable")
    completed = subprocess.run(  # noqa: S603 - resolved git executable and fixed arguments
        [git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_sentinel(path: Path, source_commit: str) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError("sentinel pass artifact is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise RuntimeError("sentinel did not pass")
    if value.get("source_commit") != source_commit:
        raise RuntimeError("sentinel source commit mismatch")
    if value.get("test_loaded") is not False:
        raise RuntimeError("sentinel test isolation failed")
    ratios = value.get("candidate_ratios")
    if not isinstance(ratios, dict) or set(ratios) != {"protect-m", "protect-mv"}:
        raise RuntimeError("sentinel candidate ratios are malformed")
    if any(float(ratio) > 1.05 for ratio in ratios.values()):
        raise RuntimeError("sentinel timing cap failed")
    return value


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixed_wallclock_accuracy(path: Path, seconds: float) -> float:
    eligible = [
        row for row in _rows(path) if float(row["elapsed_training_seconds"]) <= seconds
    ]
    return float(eligible[-1]["tuning_accuracy"]) if eligible else -1.0


def _cell_summary(root: Path, method: str, condition: str, seed: int) -> dict[str, Any]:
    cell = root / method / f"{condition}-seed{seed}"
    summary_path = cell / "summary.json"
    metrics_path = cell / "epoch_metrics.jsonl"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"incomplete Pilot cell: {cell}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = _rows(metrics_path)
    if (
        summary.get("status") != "succeeded"
        or summary.get("test_loaded") is not False
        or summary.get("method") != method
        or int(summary.get("epochs", -1)) != PILOT_EPOCHS
        or len(metrics) != PILOT_EPOCHS
    ):
        raise RuntimeError(f"invalid Pilot cell: {cell}")
    return {
        "accuracy": float(metrics[-1]["tuning_accuracy"]),
        "seconds": float(metrics[-1]["elapsed_training_seconds"]),
        "peak_vram_gb": float(summary["peak_vram_gb"]),
        "metrics_path": metrics_path,
    }


def evaluate_pilot(root: Path) -> dict[str, object]:
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    try:
        for cell in build_matrix():
            cells[(cell.method, cell.condition, cell.seed)] = _cell_summary(
                root, cell.method, cell.condition, cell.seed
            )
    except (FileNotFoundError, RuntimeError) as error:
        return {
            "status": "succeeded",
            "decision": "INCONCLUSIVE",
            "reason": str(error),
            "test_loaded": False,
        }
    results: dict[str, dict[str, object]] = {}
    for candidate in ("protect-m", "protect-mv"):
        noisy_differences = []
        clean_differences = []
        fixed_differences = []
        wallclock_ratios = []
        vram_ratios = []
        for seed in PILOT_SEEDS:
            candidate_noisy = cells[(candidate, "noisy", seed)]
            cadam_noisy = cells[("cadam", "noisy", seed)]
            candidate_clean = cells[(candidate, "clean", seed)]
            adamw_clean = cells[("adamw", "clean", seed)]
            noisy_differences.append(
                100 * (candidate_noisy["accuracy"] - cadam_noisy["accuracy"])
            )
            clean_differences.append(
                100 * (candidate_clean["accuracy"] - adamw_clean["accuracy"])
            )
            fixed_accuracy = _fixed_wallclock_accuracy(
                candidate_noisy["metrics_path"], cadam_noisy["seconds"]
            )
            fixed_differences.append(100 * (fixed_accuracy - cadam_noisy["accuracy"]))
            for condition in ("clean", "noisy"):
                candidate_cell = cells[(candidate, condition, seed)]
                adamw_cell = cells[("adamw", condition, seed)]
                wallclock_ratios.append(candidate_cell["seconds"] / adamw_cell["seconds"])
                vram_ratios.append(
                    candidate_cell["peak_vram_gb"] / adamw_cell["peak_vram_gb"]
                )
        results[candidate] = evaluate_candidate(
            {
                "candidate": candidate,
                "noisy_differences_pp": noisy_differences,
                "clean_differences_pp": clean_differences,
                "fixed_wallclock_noisy_differences_pp": fixed_differences,
                "wallclock_ratios": wallclock_ratios,
                "vram_ratios": vram_ratios,
            }
        )
    eligible = [name for name, result in results.items() if bool(result["eligible"])]
    selected: str | None = None
    if eligible:
        selected = "protect-m" if "protect-m" in eligible else "protect-mv"
        if set(eligible) == {"protect-m", "protect-mv"}:
            m_differences = [
                float(value) for value in results["protect-m"]["noisy_differences_pp"]
            ]
            mv_differences = [
                float(value) for value in results["protect-mv"]["noisy_differences_pp"]
            ]
            if (
                _mean(mv_differences) - _mean(m_differences) >= 0.5
                and all(mv >= m for mv, m in zip(mv_differences, m_differences, strict=True))
                and float(results["protect-mv"]["mean_wallclock_ratio"])
                <= float(results["protect-m"]["mean_wallclock_ratio"])
                and float(results["protect-mv"]["mean_vram_ratio"])
                <= float(results["protect-m"]["mean_vram_ratio"])
            ):
                selected = "protect-mv"
    return {
        "status": "succeeded",
        "decision": "PILOT-GO" if selected is not None else "NO-GO",
        "selected": selected,
        "candidates": results,
        "test_loaded": False,
    }


def _run_cell(cell: PilotCell, args: argparse.Namespace) -> None:
    output = args.output / cell.relative_path
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "succeeded" and summary.get("method") == cell.method:
            return
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        f"experiments.m1_calibration_{cell.condition}",
        "--seed",
        str(cell.seed),
        "--epochs",
        str(PILOT_EPOCHS),
        "--learning-rate",
        str(LEARNING_RATE),
        "--weight-decay",
        str(WEIGHT_DECAY),
        "--method",
        cell.method,
        "--augmentation-seed",
        str(10_000 + cell.seed),
        "--output-dir",
        str(output),
    ]
    if cell.condition == "clean":
        command += ["--data-dir", str(args.data_dir), "--device", "cuda"]
    else:
        command += [
            "--training",
            str(getattr(args, f"bundle_{cell.seed}")),
            "--image-store",
            str(args.image_store),
            "--tuning-store",
            str(args.tuning_store),
        ]
    with (output / "stdout.log").open("a", encoding="utf-8") as stream:
        completed = subprocess.run(  # noqa: S603 - fixed module entrypoints
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        raise RuntimeError(f"Pilot cell failed: {cell.relative_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-store", type=Path, required=True)
    parser.add_argument("--tuning-store", type=Path, required=True)
    for seed in PILOT_SEEDS:
        parser.add_argument(f"--bundle-{seed}", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = source_commit()
    validate_sentinel(args.sentinel, commit)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for cell in build_matrix():
            print(f"start {cell.method} {cell.condition} seed={cell.seed}", flush=True)
            _run_cell(cell, args)
    except RuntimeError as error:
        _write_json(
            args.output / "M1_PILOT_DECISION.json",
            {
                "status": "succeeded",
                "decision": "INCONCLUSIVE",
                "reason": str(error),
                "source_commit": commit,
                "test_loaded": False,
            },
        )
        raise
    result = {**evaluate_pilot(args.output), "source_commit": commit}
    _write_json(args.output / "M1_PILOT_DECISION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
