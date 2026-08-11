from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import causal_attribution_evidence as evidence


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _legacy_fixture(
    root: Path, parameter_auc: float, state_auc: float
) -> tuple[Path, Path, Path]:
    m06 = root / "m06"
    for index in range(20):
        cell = m06 / f"train{index + 3}-pulse161803"
        _write_json(
            cell / "summary.json",
            {
                "status": "succeeded",
                "bundle_id": f"m06-{index}",
                "branches": {
                    "parameter_only": {"clean_loss_excess_auc_128": parameter_auc},
                    "state_both": {"clean_loss_excess_auc_128": state_auc},
                },
                "source_commit": "a" * 40,
                "test_loaded": False,
            },
        )
        _write_json(cell / "sha256_manifest.json", {})
    _write_json(m06 / "COMPLETE.json", {"status": "succeeded", "completed_bundles": 20})
    m1 = root / "m1"
    _write_json(
        m1 / "M1_PILOT_DECISION.json",
        {
            "status": "succeeded",
            "decision": "NO-GO",
            "source_commit": "b" * 40,
            "test_loaded": False,
        },
    )
    m11 = root / "M11_RESCUE_DECISION.json"
    _write_json(
        m11,
        {
            "status": "succeeded",
            "decision": "NO-GO",
            "source_commit": "c" * 40,
            "test_loaded": False,
        },
    )
    return m06, m1, m11


@pytest.mark.parametrize(
    ("parameter_auc", "state_auc", "dominant"),
    [(4.0, 1.0, "parameter"), (1.0, 4.0, "state"), (1.0, 1.5, "interaction")],
)
def test_prediction_uses_frozen_twofold_rule(
    tmp_path: Path, parameter_auc: float, state_auc: float, dominant: str
) -> None:
    m06, m1, m11 = _legacy_fixture(tmp_path, parameter_auc, state_auc)
    inventory = evidence.inventory_legacy(m06, m1, m11)
    prediction = evidence.derive_prediction(inventory)
    assert prediction["checkpoint_predictions"] == {
        "1": dominant,
        "2": dominant,
        "3": dominant,
    }
    assert prediction["test_loaded"] is False


def test_inventory_fails_closed_on_wrong_bundle_count(tmp_path: Path) -> None:
    m06, m1, m11 = _legacy_fixture(tmp_path, 2.0, 1.0)
    (m06 / "train3-pulse161803" / "summary.json").unlink()
    with pytest.raises(RuntimeError, match=r"20 complete M0\.6 bundles"):
        evidence.inventory_legacy(m06, m1, m11)


def test_existing_prediction_cannot_be_rebound(tmp_path: Path) -> None:
    m06, m1, m11 = _legacy_fixture(tmp_path, 2.0, 1.0)
    output = tmp_path / "output"
    evidence.write_legacy_evidence(m06, m1, m11, output)
    prediction_path = output / "LEGACY_EVIDENCE_PREDICTION.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["source_artifact_sha256"] = "f" * 64
    _write_json(prediction_path, prediction)
    with pytest.raises(RuntimeError, match="different inventory"):
        evidence.write_legacy_evidence(m06, m1, m11, output)
