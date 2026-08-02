"""Run the frozen 12 -> 4 -> 2 AdamW successive-halving matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from experiments.m1_calibration_clean import _write_json

LEARNING_RATES = (3e-5, 1e-4, 3e-4, 1e-3)
WEIGHT_DECAYS = (0.0, 0.01, 0.1)


@dataclass(frozen=True)
class AdamWConfig:
    config_id: str
    learning_rate: float
    weight_decay: float

    @property
    def nondefault_count(self) -> int:
        return int(self.learning_rate != 3e-4) + int(self.weight_decay != 0.01)


GRID = tuple(
    AdamWConfig(f"adamw-lr{lr_index:02d}-wd{wd_index:02d}", learning_rate, weight_decay)
    for lr_index, learning_rate in enumerate(LEARNING_RATES)
    for wd_index, weight_decay in enumerate(WEIGHT_DECAYS)
)


def _last_accuracy(path: Path) -> float:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return float(rows[-1]["tuning_accuracy"])


def _run_command(command: list[str], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.json"
    if summary.is_file() and json.loads(summary.read_text())["status"] == "succeeded":
        return
    with (output / "stdout.log").open("a", encoding="utf-8") as log:
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and module entrypoints
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        raise RuntimeError(f"tuning cell failed: {output} exit={completed.returncode}")


def _run_cell(
    config: AdamWConfig,
    seed: int,
    epochs: int,
    condition: str,
    output: Path,
    args: argparse.Namespace,
) -> None:
    common = [
        sys.executable,
        "-m",
        f"experiments.m1_calibration_{condition}",
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--augmentation-seed",
        str(10_000 + seed),
        "--output-dir",
        str(output),
    ]
    if condition == "clean":
        common += ["--data-dir", str(args.data_dir), "--device", "cuda"]
    else:
        bundle = args.bundle_201 if seed == 201 else args.bundle_202
        common += [
            "--training",
            str(bundle),
            "--image-store",
            str(args.image_store),
            "--tuning-store",
            str(args.tuning_store),
        ]
    _run_command(common, output)


def _rank(
    configs: list[AdamWConfig], rung: Path, seeds: tuple[int, ...]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for config in configs:
        clean_values = [
            _last_accuracy(rung / config.config_id / f"clean-seed{seed}" / "epoch_metrics.jsonl")
            for seed in seeds
        ]
        noisy_values = [
            _last_accuracy(rung / config.config_id / f"noisy-seed{seed}" / "epoch_metrics.jsonl")
            for seed in seeds
        ]
        records.append(
            {
                "config_id": config.config_id,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "clean_accuracy": sum(clean_values) / len(clean_values),
                "noisy_accuracy": sum(noisy_values) / len(noisy_values),
                "nondefault_count": config.nondefault_count,
            }
        )
    best_clean = max(float(record["clean_accuracy"]) for record in records)
    for record in records:
        record["clean_eligible"] = float(record["clean_accuracy"]) >= best_clean - 0.005
    return sorted(
        records,
        key=lambda record: (
            not bool(record["clean_eligible"]),
            -float(record["noisy_accuracy"]),
            int(record["nondefault_count"]),
            str(record["config_id"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-store", type=Path, required=True)
    parser.add_argument("--tuning-store", type=Path, required=True)
    parser.add_argument("--bundle-201", type=Path, required=True)
    parser.add_argument("--bundle-202", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = (
        ("rung1", list(GRID), (201,), 5, 4),
        ("rung2", None, (202,), 10, 2),
        ("rung3", None, (201, 202), 20, 1),
    )
    active = list(GRID)
    history: dict[str, object] = {}
    for name, fixed, seeds, epochs, keep in stages:
        if fixed is not None:
            active = fixed
        rung = args.output / name
        for config in active:
            for seed in seeds:
                for condition in ("clean", "noisy"):
                    cell = rung / config.config_id / f"{condition}-seed{seed}"
                    print(f"start {name} {config.config_id} {condition} seed={seed}", flush=True)
                    _run_cell(config, seed, epochs, condition, cell, args)
        ranking = _rank(active, rung, seeds)
        history[name] = {"epochs": epochs, "seeds": seeds, "ranking": ranking}
        selected_ids = {str(record["config_id"]) for record in ranking[:keep]}
        active = [config for config in active if config.config_id in selected_ids]
        _write_json(args.output / f"{name}_result.json", history[name])
    final = history["rung3"]
    if not isinstance(final, dict):
        raise RuntimeError("rung3 result is malformed")
    ranking = final["ranking"]
    if not isinstance(ranking, list) or not ranking:
        raise RuntimeError("rung3 ranking is empty")
    _write_json(
        args.output / "ADAMW_SELECTION.json",
        {"status": "succeeded", "selected": ranking[0], "history": history, "test_loaded": False},
    )


if __name__ == "__main__":
    main()
