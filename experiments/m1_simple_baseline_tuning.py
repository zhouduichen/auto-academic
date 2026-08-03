"""Tune frozen Clip-AdamW and beta-retuned AdamW grids after AdamW selection."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from experiments.m1_adamw_tuning import _run_command
from experiments.m1_calibration_clean import _write_json


@dataclass(frozen=True)
class BaselineConfig:
    config_id: str
    beta1: float = 0.9
    beta2: float = 0.999
    max_grad_norm: float | None = None
    nondefault_count: int = 1


def clip_grid(g_ref: float) -> tuple[BaselineConfig, ...]:
    exponents = (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    return tuple(
        BaselineConfig(
            config_id=f"clip-e{index:02d}",
            max_grad_norm=g_ref * 2.0**exponent,
        )
        for index, exponent in enumerate(exponents)
    )


def beta_grid() -> tuple[BaselineConfig, ...]:
    return tuple(
        BaselineConfig(
            config_id=f"beta-b1-{i:02d}-b2-{j:02d}",
            beta1=beta1,
            beta2=beta2,
            nondefault_count=int((beta1, beta2) != (0.9, 0.999)),
        )
        for i, beta1 in enumerate((0.5, 0.7, 0.9, 0.95))
        for j, beta2 in enumerate((0.95, 0.99, 0.999))
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixed_wallclock_accuracy(path: Path, seconds: float) -> float:
    eligible = [row for row in _rows(path) if float(row["elapsed_training_seconds"]) <= seconds]
    return float(eligible[-1]["tuning_accuracy"]) if eligible else -1.0


def _run_cell(
    config: BaselineConfig,
    seed: int,
    epochs: int,
    condition: str,
    output: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        __import__("sys").executable,
        "-m",
        f"experiments.m1_calibration_{condition}",
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--beta1",
        str(config.beta1),
        "--beta2",
        str(config.beta2),
        "--augmentation-seed",
        str(10_000 + seed),
        "--output-dir",
        str(output),
    ]
    if config.max_grad_norm is not None:
        command += ["--max-grad-norm", str(config.max_grad_norm)]
    if condition == "clean":
        command += ["--data-dir", str(args.data_dir), "--device", "cuda"]
    else:
        bundle = args.bundle_201 if seed == 201 else args.bundle_202
        command += [
            "--training",
            str(bundle),
            "--image-store",
            str(args.image_store),
            "--tuning-store",
            str(args.tuning_store),
        ]
    _run_command(command, output)


def _rank(
    configs: list[BaselineConfig],
    rung: Path,
    seeds: tuple[int, ...],
    adamw_rung: dict[str, object],
    adamw_root: Path,
) -> list[dict[str, object]]:
    adamw_ranking = adamw_rung["ranking"]
    if not isinstance(adamw_ranking, list) or not adamw_ranking:
        raise RuntimeError("AdamW rung ranking is empty")
    clean_floor = float(adamw_ranking[0]["clean_accuracy"]) - 0.005
    rung_name = rung.name
    records: list[dict[str, object]] = []
    for config in configs:
        clean_values = []
        noisy_values = []
        fixed_wallclock_values = []
        peak_values = []
        wallclock_ok = True
        for seed in seeds:
            clean_metrics = _rows(
                rung / config.config_id / f"clean-seed{seed}" / "epoch_metrics.jsonl"
            )
            noisy_path = rung / config.config_id / f"noisy-seed{seed}"
            noisy_metrics = _rows(noisy_path / "epoch_metrics.jsonl")
            clean_values.append(float(clean_metrics[-1]["tuning_accuracy"]))
            noisy_values.append(float(noisy_metrics[-1]["tuning_accuracy"]))
            adamw_noisy_path = (
                adamw_root / rung_name / adamw_ranking[0]["config_id"] / f"noisy-seed{seed}"
            )
            adamw_clean_path = (
                adamw_root / rung_name / adamw_ranking[0]["config_id"] / f"clean-seed{seed}"
            )
            adamw_noisy_rows = _rows(adamw_noisy_path / "epoch_metrics.jsonl")
            adamw_clean_rows = _rows(adamw_clean_path / "epoch_metrics.jsonl")
            budget = float(adamw_noisy_rows[-1]["elapsed_training_seconds"])
            noisy_elapsed = float(noisy_metrics[-1]["elapsed_training_seconds"])
            clean_elapsed = float(clean_metrics[-1]["elapsed_training_seconds"])
            clean_budget = float(adamw_clean_rows[-1]["elapsed_training_seconds"])
            wallclock_ok = (
                wallclock_ok
                and noisy_elapsed <= 1.05 * budget
                and clean_elapsed <= 1.05 * clean_budget
            )
            fixed_wallclock_values.append(
                _fixed_wallclock_accuracy(noisy_path / "epoch_metrics.jsonl", budget)
            )
            noisy_summary = json.loads((noisy_path / "summary.json").read_text(encoding="utf-8"))
            clean_summary = json.loads(
                (rung / config.config_id / f"clean-seed{seed}" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            peak_values.extend(
                [float(noisy_summary["peak_vram_gb"]), float(clean_summary["peak_vram_gb"])]
            )
        records.append(
            {
                "config_id": config.config_id,
                "beta1": config.beta1,
                "beta2": config.beta2,
                "max_grad_norm": config.max_grad_norm,
                "clean_accuracy": sum(clean_values) / len(clean_values),
                "noisy_accuracy": sum(noisy_values) / len(noisy_values),
                "fixed_wallclock_noisy_accuracy": sum(fixed_wallclock_values)
                / len(fixed_wallclock_values),
                "peak_vram_gb": max(peak_values),
                "wallclock_eligible": wallclock_ok,
                "nondefault_count": config.nondefault_count,
            }
        )
    for record in records:
        record["clean_eligible"] = float(record["clean_accuracy"]) >= clean_floor
    return sorted(
        records,
        key=lambda record: (
            not bool(record["clean_eligible"]),
            not bool(record["wallclock_eligible"]),
            -float(record["noisy_accuracy"]),
            -float(record["fixed_wallclock_noisy_accuracy"]),
            float(record["peak_vram_gb"]),
            int(record["nondefault_count"]),
            str(record["config_id"]),
        ),
    )


def _run_method(
    method: str,
    grid: tuple[BaselineConfig, ...],
    adamw: dict[str, object],
    args: argparse.Namespace,
) -> None:
    output = args.output / method
    stages = (("rung1", (201,), 5, 4), ("rung2", (202,), 10, 2), ("rung3", (201, 202), 20, 1))
    active = list(grid)
    history: dict[str, object] = {}
    for name, seeds, epochs, keep in stages:
        rung = output / name
        for config in active:
            for seed in seeds:
                for condition in ("clean", "noisy"):
                    cell = rung / config.config_id / f"{condition}-seed{seed}"
                    print(
                        f"start {method} {name} {config.config_id} {condition} seed={seed}",
                        flush=True,
                    )
                    _run_cell(config, seed, epochs, condition, cell, args)
        adamw_rung = adamw["history"][name]
        ranking = _rank(active, rung, seeds, adamw_rung, args.adamw_root)
        history[name] = {"epochs": epochs, "seeds": seeds, "ranking": ranking}
        selected_ids = {str(record["config_id"]) for record in ranking[:keep]}
        active = [config for config in active if config.config_id in selected_ids]
        _write_json(output / f"{name}_result.json", history[name])
    final = history["rung3"]
    ranking = final["ranking"]
    _write_json(
        output / "SELECTION.json",
        {
            "status": "succeeded",
            "method": method,
            "selected": ranking[0],
            "history": history,
            "test_loaded": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-store", type=Path, required=True)
    parser.add_argument("--tuning-store", type=Path, required=True)
    parser.add_argument("--bundle-201", type=Path, required=True)
    parser.add_argument("--bundle-202", type=Path, required=True)
    parser.add_argument("--adamw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g-ref", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    args = parser.parse_args()
    if not math.isfinite(args.g_ref) or args.g_ref <= 0:
        parser.error("g-ref must be finite and positive")
    adamw = json.loads((args.adamw_root / "ADAMW_SELECTION.json").read_text(encoding="utf-8"))
    if adamw.get("status") != "succeeded" or adamw.get("test_loaded") is not False:
        raise RuntimeError("AdamW selection is not a sealed successful artifact")
    args.output.mkdir(parents=True, exist_ok=True)
    _run_method("clip-adamw", clip_grid(args.g_ref), adamw, args)
    _run_method("beta-adamw", beta_grid(), adamw, args)
    _write_json(
        args.output / "SIMPLE_BASELINES_COMPLETE.json",
        {"status": "succeeded", "methods": ["clip-adamw", "beta-adamw"], "test_loaded": False},
    )


if __name__ == "__main__":
    main()
