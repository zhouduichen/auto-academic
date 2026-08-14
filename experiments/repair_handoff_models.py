"""Pinned model adapters for the repair-handoff transport experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import ResNetForImageClassification, ViTForImageClassification, ViTModel
from transformers.utils.hub import cached_file

from experiments import m1_calibration_clean as clean

VIT_MODEL_ID = "google/vit-base-patch16-224-in21k"
VIT_REVISION = "b4569560a39a0f1af58e3ddaf17facf20ab919b0"
RESNET_MODEL_ID = "microsoft/resnet-18"
RESNET_REVISION = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
ALLOWED_HEAD_MISMATCH = {"classifier.1.weight", "classifier.1.bias"}


def _mismatch_names(raw: object) -> set[str]:
    if not isinstance(raw, (list, set, tuple)):
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
        if not isinstance(value, (list, set, tuple)) or value:
            raise RuntimeError(f"ResNet load audit failed: {field}")
    if _mismatch_names(loading.get("mismatched_keys", [])) != ALLOWED_HEAD_MISMATCH:
        raise RuntimeError("ResNet load audit failed: mismatched_keys")


def build_transport_model(config: clean.Config, *, num_labels: int = 100) -> nn.Module:
    """Build only one of the two frozen transport model families."""
    if num_labels not in {10, 100}:
        raise ValueError("transport num_labels must be 10 or 100")
    if config.model_id == VIT_MODEL_ID:
        if num_labels == 100:
            return clean._build_model(config)
        backbone, loading = ViTModel.from_pretrained(
            VIT_MODEL_ID,
            revision=VIT_REVISION,
            output_loading_info=True,
            local_files_only=True,
        )
        anomalies = {
            name: loading.get(name, [])
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading.get(name)
        }
        if anomalies:
            raise RuntimeError(f"ViT backbone load audit failed: {anomalies}")
        backbone.config.num_labels = num_labels
        base = ViTForImageClassification(backbone.config)
        base.vit = backbone
        return get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_rank,
                lora_alpha=config.lora_rank * 2,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                modules_to_save=["classifier"],
            ),
        )
    if config.model_id != RESNET_MODEL_ID:
        raise ValueError("unsupported transport model_id")
    model, loading = ResNetForImageClassification.from_pretrained(
        RESNET_MODEL_ID,
        revision=RESNET_REVISION,
        num_labels=num_labels,
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


def vit_artifact_binding(*, local_files_only: bool) -> dict[str, str]:
    """Resolve and hash the pinned ViT backbone files."""
    resolved: dict[str, Path] = {}
    for filename in ("config.json", "model.safetensors"):
        raw = cached_file(
            VIT_MODEL_ID,
            filename,
            revision=VIT_REVISION,
            local_files_only=local_files_only,
        )
        if not isinstance(raw, str) or not Path(raw).is_file():
            raise RuntimeError(f"pinned ViT artifact is unavailable: {filename}")
        resolved[filename] = Path(raw)
    return {
        "model_id": VIT_MODEL_ID,
        "revision": VIT_REVISION,
        "config_sha256": _sha256(resolved["config.json"]),
        "model_safetensors_sha256": _sha256(resolved["model.safetensors"]),
    }
