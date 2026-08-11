"""Validate immutable legacy evidence and freeze a pre-GPU causal prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from experiments.m1_calibration_clean import _write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json_value(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid legacy JSON: {path}") from error


def _read_json(path: Path) -> dict[str, object]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"legacy JSON must be an object: {path}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def _validate_decision(path: Path, label: str) -> dict[str, object]:
    value = _read_json(path)
    if value.get("status") != "succeeded" or value.get("decision") != "NO-GO":
        raise RuntimeError(f"{label} must be a succeeded NO-GO decision")
    if value.get("test_loaded") is not False:
        raise RuntimeError(f"{label} test isolation failed")
    source_commit = value.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError(f"{label} source commit is malformed")
    return {
        "status": "succeeded",
        "decision": "NO-GO",
        "source_commit": source_commit,
        "sha256": _sha256(path),
        "test_loaded": False,
    }


def inventory_legacy(
    m06_root: Path, m1_root: Path, m11_decision: Path
) -> dict[str, object]:
    complete_path = m06_root / "COMPLETE.json"
    complete = _read_json(complete_path)
    summary_paths = sorted(m06_root.glob("train*-pulse*/summary.json"))
    if (
        complete.get("status") != "succeeded"
        or complete.get("completed_bundles") != 20
        or len(summary_paths) != 20
    ):
        raise RuntimeError("expected exactly 20 complete M0.6 bundles")

    parameter_auc: list[float] = []
    state_auc: list[float] = []
    bundle_ids: list[str] = []
    consumed = {
        "m06/COMPLETE.json": _sha256(complete_path),
    }
    for summary_path in summary_paths:
        manifest_path = summary_path.with_name("sha256_manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError("expected exactly 20 complete M0.6 bundles")
        manifest = _read_json_value(manifest_path)
        if not isinstance(manifest, list | dict):
            raise RuntimeError("M0.6 SHA manifest must be a JSON array or object")
        summary = _read_json(summary_path)
        if summary.get("status") != "succeeded" or summary.get("test_loaded") is not False:
            raise RuntimeError("M0.6 summary status or test isolation failed")
        bundle_id = summary.get("bundle_id")
        source_commit = summary.get("source_commit")
        branches = summary.get("branches")
        if not isinstance(bundle_id, str) or not bundle_id:
            raise RuntimeError("M0.6 bundle ID is malformed")
        if not isinstance(source_commit, str) or len(source_commit) != 40:
            raise RuntimeError("M0.6 source commit is malformed")
        if not isinstance(branches, dict):
            raise RuntimeError("M0.6 branch summary is malformed")
        parameter_branch = branches.get("parameter_only")
        state_branch = branches.get("state_both")
        if not isinstance(parameter_branch, dict) or not isinstance(state_branch, dict):
            raise RuntimeError("M0.6 causal branches are missing")
        parameter_auc.append(
            _finite_number(
                parameter_branch.get("clean_loss_excess_auc_128"), "parameter AUC"
            )
        )
        state_auc.append(
            _finite_number(state_branch.get("clean_loss_excess_auc_128"), "state AUC")
        )
        bundle_ids.append(bundle_id)
        cell = summary_path.parent.name
        consumed[f"m06/{cell}/summary.json"] = _sha256(summary_path)
        consumed[f"m06/{cell}/sha256_manifest.json"] = _sha256(manifest_path)
    if len(set(bundle_ids)) != 20:
        raise RuntimeError("M0.6 bundle IDs must be unique")

    m1_path = m1_root / "M1_PILOT_DECISION.json"
    m1 = _validate_decision(m1_path, "M1")
    m11 = _validate_decision(m11_decision, "M1.1")
    consumed["m1/M1_PILOT_DECISION.json"] = _sha256(m1_path)
    consumed[f"m11/{m11_decision.name}"] = _sha256(m11_decision)
    inventory: dict[str, object] = {
        "schema": "causal-attribution-legacy-evidence/1",
        "m06": {
            "bundle_count": 20,
            "bundle_ids": bundle_ids,
            "parameter_auc_128": parameter_auc,
            "state_auc_128": state_auc,
        },
        "m1": m1,
        "m11": m11,
        "consumed_sha256": dict(sorted(consumed.items())),
        "test_loaded": False,
    }
    inventory["inventory_sha256"] = _canonical_sha256(inventory)
    return inventory


def derive_prediction(inventory: dict[str, object]) -> dict[str, object]:
    m06 = inventory.get("m06")
    if not isinstance(m06, dict):
        raise RuntimeError("M0.6 inventory is malformed")
    raw_parameter = m06.get("parameter_auc_128")
    raw_state = m06.get("state_auc_128")
    if not isinstance(raw_parameter, list) or not isinstance(raw_state, list):
        raise RuntimeError("M0.6 AUC inventory is malformed")
    if len(raw_parameter) != 20 or len(raw_state) != 20:
        raise RuntimeError("M0.6 AUC inventory must contain 20 bundles")
    parameter = statistics.median(
        abs(_finite_number(value, "parameter AUC")) for value in raw_parameter
    )
    state = statistics.median(abs(_finite_number(value, "state AUC")) for value in raw_state)
    if parameter >= 2 * state:
        dominant = "parameter"
    elif state >= 2 * parameter:
        dominant = "state"
    else:
        dominant = "interaction"
    inventory_sha256 = inventory.get("inventory_sha256")
    if not isinstance(inventory_sha256, str) or len(inventory_sha256) != 64:
        raise RuntimeError("legacy inventory hash is malformed")
    return {
        "schema": "causal-attribution-legacy-prediction/1",
        "rule": "median-absolute-auc-twofold-v1",
        "parameter_median_abs_auc_128": parameter,
        "state_median_abs_auc_128": state,
        "checkpoint_predictions": {str(epoch): dominant for epoch in (1, 2, 3)},
        "source_artifact_sha256": inventory_sha256,
        "test_loaded": False,
    }


def write_legacy_evidence(
    m06_root: Path, m1_root: Path, m11_decision: Path, output: Path
) -> tuple[dict[str, object], dict[str, object]]:
    inventory = inventory_legacy(m06_root, m1_root, m11_decision)
    prediction = derive_prediction(inventory)
    prediction_path = output / "LEGACY_EVIDENCE_PREDICTION.json"
    if prediction_path.is_file():
        existing = _read_json(prediction_path)
        if existing.get("source_artifact_sha256") != prediction["source_artifact_sha256"]:
            raise RuntimeError("existing prediction belongs to a different inventory")
        if existing != prediction:
            raise RuntimeError("existing prediction content differs")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "LEGACY_EVIDENCE_INVENTORY.json", inventory)
    _write_json(prediction_path, prediction)
    return inventory, prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m06-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--m11-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, prediction = write_legacy_evidence(
        args.m06_root, args.m1_root, args.m11_decision, args.output
    )
    print(json.dumps(prediction, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
