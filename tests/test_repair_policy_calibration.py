from __future__ import annotations

from copy import deepcopy

import pytest

from experiments import repair_policy_calibration as calibration


def _scores(*, overlap: bool = False) -> list[calibration.DiscoveryScore]:
    clean = [0.20, 0.25, 0.30]
    noisy = [0.70, 0.75, 0.80]
    if overlap:
        noisy = [0.21, 0.22, 0.23]
    rows = []
    for seed, left, right in zip((601, 602, 603), clean, noisy, strict=True):
        rows.extend(
            [
                calibration.DiscoveryScore("resnet18", seed, "clean", left),
                calibration.DiscoveryScore("resnet18", seed, "noisy", right),
            ]
        )
    return rows


def test_thresholds_are_adjacent_midpoints() -> None:
    assert calibration.candidate_thresholds([0.1, 0.4, 0.9]) == [0.25, 0.65]


def test_leave_one_seed_out_freezes_direction_and_smallest_tie() -> None:
    result = calibration.leave_one_seed_out(_scores())
    assert result["decision"] == "GO"
    assert result["rule"] == "lt:CN,ge:NC"
    assert result["threshold"] == min(result["maximizing_thresholds"])
    assert result["leave_one_seed_out_balanced_accuracy"] == 1.0


def test_overlap_fails_calibration() -> None:
    assert calibration.leave_one_seed_out(_scores(overlap=True))["decision"] == "NO-GO"


def test_calibration_rejects_missing_duplicate_nonfinite_or_mixed_model() -> None:
    rows = _scores()
    with pytest.raises(ValueError, match="matrix"):
        calibration.leave_one_seed_out(rows[:-1])
    duplicate = deepcopy(rows)
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="matrix"):
        calibration.leave_one_seed_out(duplicate)
    nonfinite = deepcopy(rows)
    nonfinite[0] = calibration.DiscoveryScore(
        "resnet18", 601, "clean", float("nan")
    )
    with pytest.raises(ValueError, match="non-finite"):
        calibration.leave_one_seed_out(nonfinite)
    mixed = deepcopy(rows)
    mixed[0] = calibration.DiscoveryScore("vit_lora", 501, "clean", 0.1)
    with pytest.raises(ValueError, match="one model"):
        calibration.leave_one_seed_out(mixed)
