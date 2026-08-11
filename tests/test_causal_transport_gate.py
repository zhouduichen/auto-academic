from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import causal_transport_gate as gate


def _summary(kind: str, *, commit: str = "a" * 40) -> dict[str, object]:
    values = {
        "state": {"parameter": 1.0, "state": 4.0, "interaction": 0.5},
        "parameter": {"parameter": 4.0, "state": 1.0, "interaction": 0.5},
        "mixed": {"parameter": 1.0, "state": 1.0, "interaction": 0.5},
        "interaction": {"parameter": 1.0, "state": 2.0, "interaction": 3.0},
    }[kind]
    return {
        "effects": {
            "clean_loss_excess_auc_128": dict(values),
            "tuning_loss_h512": dict(values),
        },
        "source_commit": commit,
        "test_loaded": False,
    }


def _pair(local: str, accumulated: str) -> dict[str, object]:
    return {"local": _summary(local), "accumulated": _summary(accumulated)}


def test_carrier_ratio_and_interaction_override() -> None:
    assert gate.carrier_ratio(
        {"parameter": 1.0, "state": 4.0, "interaction": 0.5}
    ) == pytest.approx(2.0)
    assert (
        gate.classify_effect(
            {"parameter": 1.0, "state": 4.0, "interaction": 0.5}
        )
        == "state"
    )
    assert (
        gate.classify_effect(
            {"parameter": 4.0, "state": 1.0, "interaction": 0.5}
        )
        == "parameter"
    )
    assert (
        gate.classify_effect(
            {"parameter": 1.0, "state": 2.0, "interaction": 3.0}
        )
        == "interaction"
    )


@pytest.mark.parametrize(
    "effects",
    [
        {"parameter": float("nan"), "state": 1.0, "interaction": 0.0},
        {"parameter": True, "state": 1.0, "interaction": 0.0},
        {"parameter": 1.0, "state": 1.0},
    ],
)
def test_carrier_ratio_rejects_malformed_effects(effects: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="effects"):
        gate.carrier_ratio(effects)  # type: ignore[arg-type]


def test_gate_a_requires_both_seed_crossings() -> None:
    records = {"301": _pair("state", "parameter"), "302": _pair("state", "parameter")}
    result = gate.evaluate_stage(records, (301, 302), "A")
    assert result["decision"] == "GO"
    records["302"] = _pair("mixed", "parameter")
    assert gate.evaluate_stage(records, (301, 302), "A")["decision"] == "NO-GO"


def test_gate_b_requires_two_of_three_without_reverse() -> None:
    passing = _pair("state", "parameter")
    neutral = _pair("mixed", "mixed")
    reverse = _pair("parameter", "state")
    result = gate.evaluate_stage(
        {"401": passing, "402": passing, "403": neutral}, (401, 402, 403), "B"
    )
    assert result["decision"] == "GO"
    assert result["crossing_seeds"] == [401, 402]
    assert (
        gate.evaluate_stage(
            {"401": passing, "402": passing, "403": reverse},
            (401, 402, 403),
            "B",
        )["decision"]
        == "NO-GO"
    )


def test_stage_fails_closed_on_interaction_or_outcome_disagreement() -> None:
    interaction = _pair("interaction", "parameter")
    mismatch = _pair("state", "parameter")
    mismatch["local"]["effects"]["tuning_loss_h512"] = {  # type: ignore[index]
        "parameter": 4.0,
        "state": 1.0,
        "interaction": 0.5,
    }
    result = gate.evaluate_stage(
        {"401": interaction, "402": mismatch, "403": _pair("state", "parameter")},
        (401, 402, 403),
        "B",
    )
    assert result["decision"] == "NO-GO"


def test_stage_rejects_wrong_seed_set_commit_and_test_access() -> None:
    with pytest.raises(ValueError, match="seed set"):
        gate.evaluate_stage({"401": _pair("state", "parameter")}, (401, 402, 403), "B")
    wrong = _pair("state", "parameter")
    wrong["accumulated"] = _summary("parameter", commit="b" * 40)
    with pytest.raises(ValueError, match="source commits"):
        gate.evaluate_stage(
            {"401": wrong, "402": _pair("state", "parameter"), "403": _pair("mixed", "mixed")},
            (401, 402, 403),
            "B",
        )
    accessed = _pair("state", "parameter")
    accessed["local"]["test_loaded"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="test isolation"):
        gate.evaluate_stage(
            {"401": accessed, "402": _pair("state", "parameter"), "403": _pair("mixed", "mixed")},
            (401, 402, 403),
            "B",
        )


def test_atomic_decision_replaces_only_same_contract(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "DECISION.json"
    first = {"contract_sha256": "a" * 64, "decision": "NO-GO"}
    gate.write_decision(path, first)
    assert json.loads(path.read_text(encoding="utf-8")) == first
    gate.write_decision(path, {"contract_sha256": "a" * 64, "decision": "GO"})
    assert json.loads(path.read_text(encoding="utf-8"))["decision"] == "GO"
    with pytest.raises(RuntimeError, match="different contract"):
        gate.write_decision(
            path, {"contract_sha256": "b" * 64, "decision": "NO-GO"}
        )
