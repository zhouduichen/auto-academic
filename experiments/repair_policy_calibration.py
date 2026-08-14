"""Freeze a discovery-only stream diagnostic for choosing repair carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR100

from experiments import causal_attribution_replay as replay
from experiments import causal_transport_bundle as transport
from experiments import m1_calibration_clean as clean
from experiments import repair_handoff_models as repair_models
from src.arw import m0_core

DISCOVERY_SEEDS = {"resnet18": (601, 602, 603), "vit_lora": (501, 502, 503)}
CONTINUATIONS = ("clean", "noisy")
PREFIX_STEPS = 8


@dataclass(frozen=True)
class DiscoveryScore:
    model: str
    seed: int
    continuation: str
    score: float
    bundle_sha256: str = ""


def candidate_thresholds(scores: list[float]) -> list[float]:
    if len(scores) < 2 or any(not math.isfinite(value) for value in scores):
        raise ValueError("calibration scores must contain finite values")
    ordered = sorted(set(scores))
    if len(ordered) < 2:
        raise ValueError("calibration scores do not define a threshold")
    return [(left + right) / 2.0 for left, right in pairwise(ordered)]


def _accuracy(rows: list[DiscoveryScore], threshold: float) -> float:
    if not rows:
        raise ValueError("calibration fold is empty")
    correct = sum(
        (row.score < threshold) == (row.continuation == "clean") for row in rows
    )
    return correct / len(rows)


def _fit(rows: list[DiscoveryScore]) -> tuple[float, list[float], float]:
    thresholds = candidate_thresholds([row.score for row in rows])
    by_score = [(threshold, _accuracy(rows, threshold)) for threshold in thresholds]
    best = max(score for _, score in by_score)
    maximizing = [threshold for threshold, score in by_score if score == best]
    return min(maximizing), maximizing, best


def leave_one_seed_out(scores: list[DiscoveryScore]) -> dict[str, object]:
    if not scores:
        raise ValueError("discovery scores are empty")
    models = {row.model for row in scores}
    if len(models) != 1:
        raise ValueError("calibration must fit one model family at a time")
    model = next(iter(models))
    expected_seeds = set(DISCOVERY_SEEDS.get(model, ()))
    expected = {(seed, continuation) for seed in expected_seeds for continuation in CONTINUATIONS}
    actual = {(row.seed, row.continuation) for row in scores}
    if actual != expected or len(scores) != len(expected):
        raise ValueError("calibration scores do not match the frozen paired seed matrix")
    if any(not math.isfinite(row.score) for row in scores):
        raise ValueError("calibration score is non-finite")
    predictions: list[dict[str, object]] = []
    for held_seed in sorted(expected_seeds):
        training = [row for row in scores if row.seed != held_seed]
        held = [row for row in scores if row.seed == held_seed]
        threshold, _, _ = _fit(training)
        for row in held:
            prediction = "clean" if row.score < threshold else "noisy"
            predictions.append(
                {
                    "seed": row.seed,
                    "continuation": row.continuation,
                    "prediction": prediction,
                    "threshold": threshold,
                    "correct": prediction == row.continuation,
                }
            )
    balanced_accuracy = sum(bool(row["correct"]) for row in predictions) / len(
        predictions
    )
    threshold, maximizing, fit_accuracy = _fit(scores)
    return {
        "model": model,
        "decision": "GO" if balanced_accuracy >= 5 / 6 else "NO-GO",
        "rule": "lt:CN,ge:NC",
        "threshold": threshold,
        "maximizing_thresholds": maximizing,
        "fit_accuracy": fit_accuracy,
        "leave_one_seed_out_balanced_accuracy": balanced_accuracy,
        "predictions": predictions,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_bundle(path: Path) -> dict[str, object]:
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (path / "sha256_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("discovery bundle cannot be loaded") from error
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "succeeded"
        or summary.get("test_loaded") is not False
        or not isinstance(manifest, dict)
    ):
        raise RuntimeError("discovery bundle is invalid")
    for name, expected in manifest.items():
        target = path / str(name)
        if not isinstance(expected, str) or not target.is_file() or _sha(target) != expected:
            raise RuntimeError("discovery bundle checksum failed")
    return summary


def _restore_discovery_snapshot(
    model: nn.Module, path: Path, contract_sha256: str, model_key: str
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract_sha256") != contract_sha256:
        raise RuntimeError("discovery snapshot contract mismatch")
    if payload.get("schema") == transport.SNAPSHOT_SCHEMA:
        state = payload.get("model_state")
        if not isinstance(state, dict):
            raise RuntimeError("complete discovery model state is missing")
        replay.restore_model_state(model, state)
        return
    if payload.get("schema") != "causal-transport-exposure/1" or model_key != "vit_lora":
        raise RuntimeError("legacy discovery snapshot is not eligible")
    if list(model.named_buffers()):
        raise RuntimeError("legacy ViT snapshot omits persistent buffers")
    parameters = payload.get("parameters")
    expected = {name for name, value in model.named_parameters() if value.requires_grad}
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise RuntimeError("legacy ViT trainable state is incomplete")
    m0_core.restore_trainable_state(model, parameters)


def score_discovery_bundle(
    path: Path, *, model_key: str, device: torch.device
) -> DiscoveryScore:
    summary = _validate_bundle(path)
    request = summary.get("request")
    if not isinstance(request, dict) or not isinstance(request.get("config"), dict):
        raise RuntimeError("discovery request is malformed")
    config = clean.Config(**request["config"])
    seed = int(request.get("seed", -1))
    continuation = str(request.get("continuation", ""))
    if seed not in DISCOVERY_SEEDS.get(model_key, ()) or continuation not in CONTINUATIONS:
        raise RuntimeError("discovery bundle is outside the frozen calibration matrix")
    expected_model = (
        repair_models.RESNET_MODEL_ID
        if model_key == "resnet18"
        else repair_models.VIT_MODEL_ID
    )
    if config.model_id != expected_model or int(request.get("dose", -1)) != 1250:
        raise RuntimeError("discovery model or dose mismatch")
    model = repair_models.build_transport_model(config, num_labels=100).to(device)
    _restore_discovery_snapshot(
        model,
        path / "noisy-exposure.pt",
        str(summary.get("contract_sha256")),
        model_key,
    )
    data_dir = Path(str(request["data_dir"]))
    image_store = Path(str(request["image_store"]))
    noisy_bundle = Path(str(request["noisy_bundle"]))
    cifar = CIFAR100(root=data_dir, train=True, download=False, transform=None)
    dataset = transport.PairedTransportDataset(
        image_store,
        noisy_bundle,
        clean_targets=np.asarray(cifar.targets),
        augmentation_seed=config.augmentation_seed,
    )
    start = int(request["warmup_steps"]) + int(request["dose"])
    plans = transport._batch_plans(
        len(dataset), config.batch_size, seed, start + PREFIX_STEPS
    )[start:]
    batches = []
    for plan in plans:
        images, clean_labels, noisy_labels = transport._materialize_plan(dataset, plan)
        batches.append(
            (images, noisy_labels if continuation == "noisy" else clean_labels)
        )
    score = transport.stream_label_nll(model, batches, device, num_labels=100)
    return DiscoveryScore(
        model=model_key,
        seed=seed,
        continuation=continuation,
        score=score,
        bundle_sha256=_sha(path / "sha256_manifest.json"),
    )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def calibrate_policy(
    bundles: dict[str, list[Path]], output: Path, *, device: str = "cuda"
) -> dict[str, object]:
    decision_path = output / "REPAIR_POLICY_DECISION.json"
    output.mkdir(parents=True, exist_ok=True)
    decision_path.unlink(missing_ok=True)
    expected_models = set(DISCOVERY_SEEDS)
    if set(bundles) != expected_models or any(len(paths) != 6 for paths in bundles.values()):
        raise ValueError("policy calibration requires exact two-model 12-bundle inputs")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    scores = {
        model: [
            score_discovery_bundle(path, model_key=model, device=torch_device)
            for path in paths
        ]
        for model, paths in bundles.items()
    }
    models = {model: leave_one_seed_out(rows) for model, rows in scores.items()}
    record = {
        "schema": "repair-policy-calibration/1",
        "decision": "GO"
        if all(value["decision"] == "GO" for value in models.values())
        else "NO-GO",
        "prefix_steps": PREFIX_STEPS,
        "score": "mean_stream_label_nll/log(num_labels)",
        "models": models,
        "scores": {
            model: [row.__dict__ for row in rows] for model, rows in scores.items()
        },
        "test_loaded": False,
        "development_loaded": False,
        "confirmation_loaded": False,
    }
    _atomic_json(decision_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet-bundle", type=Path, action="append", default=[])
    parser.add_argument("--vit-bundle", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = calibrate_policy(
        {"resnet18": args.resnet_bundle, "vit_lora": args.vit_bundle},
        args.output,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
