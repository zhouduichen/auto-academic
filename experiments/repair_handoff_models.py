"""Pinned model adapters for the repair-handoff transport experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

from torch import nn
from transformers import ResNetForImageClassification
from transformers.utils.hub import cached_file

from experiments import m1_calibration_clean as clean

VIT_MODEL_ID = "google/vit-base-patch16-224-in21k"
RESNET_MODEL_ID = "microsoft/resnet-18"
RESNET_REVISION = "b84c5cd73e9544fa1b67d690748d13a4bdb29267"
ALLOWED_HEAD_MISMATCH = {"classifier.1.weight", "classifier.1.bias"}


def _mismatch_names(raw: object) -> set[str]:
    if not isinstance(raw, list):
        raise RuntimeError("ResNet load audit mismatch record is malformed")
    names: set[str] = set()
    for record in raw:
        if isinstance(record, (tuple, list)) and record and isinstance(record[0], str):
            names.add(record[0])
        elif isinstance(record, dict) and isinstance(record.get("key"), str):
            names.add(str(record["key"]))
        else:
            raise RuntimeError("ResNet load audit mismatch record is malformed")
    return names


def _validate_loading_info(loading: object) -> None:
    if not isinstance(loading, dict):
        raise RuntimeError("ResNet load audit is unavailable")
    for field in ("missing_keys", "unexpected_keys", "error_msgs"):
        value = loading.get(field, [])
        if not isinstance(value, list) or value:
            raise RuntimeError(f"ResNet load audit failed: {field}")
    if _mismatch_names(loading.get("mismatched_keys", [])) != ALLOWED_HEAD_MISMATCH:
        raise RuntimeError("ResNet load audit failed: mismatched_keys")


def build_transport_model(config: clean.Config) -> nn.Module:
    """Build only one of the two frozen transport model families."""
    if config.model_id == VIT_MODEL_ID:
        return clean._build_model(config)
    if config.model_id != RESNET_MODEL_ID:
        raise ValueError("unsupported transport model_id")
    model, loading = ResNetForImageClassification.from_pretrained(
        RESNET_MODEL_ID,
        revision=RESNET_REVISION,
        num_labels=100,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        use_safetensors=True,
        local_files_only=True,
    )
    _validate_loading_info(loading)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("ResNet sentinel requires full-parameter fine-tuning")
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resnet_artifact_binding(*, local_files_only: bool) -> dict[str, str]:
    """Resolve and hash the two pinned ResNet files used by the experiment."""
    resolved: dict[str, Path] = {}
    for filename in ("config.json", "model.safetensors"):
        raw = cached_file(
            RESNET_MODEL_ID,
            filename,
            revision=RESNET_REVISION,
            local_files_only=local_files_only,
        )
        if not isinstance(raw, str) or not Path(raw).is_file():
            raise RuntimeError(f"pinned ResNet artifact is unavailable: {filename}")
        resolved[filename] = Path(raw)
    return {
        "model_id": RESNET_MODEL_ID,
        "revision": RESNET_REVISION,
        "config_sha256": _sha256(resolved["config.json"]),
        "model_safetensors_sha256": _sha256(resolved["model.safetensors"]),
    }
