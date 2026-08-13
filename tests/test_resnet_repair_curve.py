from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments import causal_transport_gate as gate
from experiments import resnet_repair_curve as curve
from experiments.repair_handoff_models import RESNET_MODEL_ID


def test_build_new_jobs_is_exact_missing_matrix(tmp_path: Path) -> None:
    assert [job.key for job in curve.build_new_jobs(tmp_path)] == [
        f"seed{seed}-dose{dose}-{continuation}"
        for seed in (601, 602, 603)
        for dose in (1, 1250)
        for continuation in ("clean", "noisy")
    ]


def _old_decision(margins: dict[int, float] | None = None) -> dict[str, object]:
    values = margins or {8: -1.0, 64: 2.0}
    observations = {}
    for seed in curve.TRAINING_SEEDS:
        for dose in (8, 64):
            for continuation in curve.CONTINUATIONS:
                observations[f"seed{seed}-dose{dose}-{continuation}"] = {
                    outcome: {
                        "carrier": "parameter" if values[dose] > 0 else "state",
                        "parameter_repair": values[dose],
                        "state_repair": 0.0,
                        "repair_margin": values[dose],
                    }
                    for outcome in gate.PRIMARY_OUTCOMES
                }
    return {
        "schema": "resnet-repair-handoff-decision/1",
        "decision": "NO-GO",
        "bundle_count": 12,
        "contract_sha256": "b" * 64,
        "source_commit": "9" * 40,
        "observations": observations,
        "test_loaded": False,
    }


def _summary(
    *, seed: int, dose: int, continuation: str, margin: float
) -> dict[str, object]:
    endpoints = {branch: {} for branch in ("CC", "CN", "NC", "NN")}
    for outcome in gate.PRIMARY_OUTCOMES:
        endpoints["CC"][outcome] = 0.0
        endpoints["CN"][outcome] = 1.0
        endpoints["NC"][outcome] = 1.0 + margin
        endpoints["NN"][outcome] = 3.0
    return {
        "status": "succeeded",
        "source_commit": "f" * 40,
        "contract_sha256": "a" * 64,
        "request": {
            "seed": seed,
            "dose": dose,
            "continuation": continuation,
            "config": {"model_id": RESNET_MODEL_ID},
        },
        "endpoints": endpoints,
        "test_loaded": False,
    }


def _new_summaries() -> dict[str, dict[str, object]]:
    return {
        job.key: _summary(
            seed=job.seed,
            dose=job.dose,
            continuation=job.continuation,
            margin=-2.0 if job.dose == 1 else 3.0,
        )
        for job in curve.build_new_jobs(Path("unused"))
    }


def _evaluate(
    prior: dict[str, object] | None = None,
    summaries: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    return curve.evaluate_curve(
        prior or _old_decision(),
        summaries or _new_summaries(),
        contract_sha256="a" * 64,
        prior_decision_sha256="c" * 64,
        source_commit="f" * 40,
    )


def test_curve_gate_accepts_robust_high_exposure_shift() -> None:
    record = _evaluate()
    assert record["decision"] == "GO"
    assert record["checks"] == {
        "exact_valid_matrix": True,
        "high_exposure_shift_is_robust": True,
        "dose1250_parameter_repair_is_robust": True,
    }
    assert record["contrast_positive_counts"]["overall"] == 12
    assert record["dose1250_positive_counts"]["overall"] == 12


@pytest.mark.parametrize(
    ("keys", "dose"),
    [
        (["seed601-dose1250-clean"], 1250),
        (
            [
                "seed601-dose1250-clean",
                "seed601-dose1250-noisy",
            ],
            1250,
        ),
    ],
)
def test_curve_gate_rejects_scientific_misses(keys: list[str], dose: int) -> None:
    summaries = _new_summaries()
    for key in keys:
        summary = summaries[key]
        for outcome in gate.PRIMARY_OUTCOMES:
            summary["endpoints"]["NC"][outcome] = -9.0
    record = _evaluate(summaries=summaries)
    assert record["decision"] == "NO-GO"
    assert dose == 1250


def test_curve_gate_rejects_missing_mismatched_and_nonfinite_new_bundle() -> None:
    missing = _new_summaries()
    missing.pop("seed603-dose1250-noisy")
    with pytest.raises(RuntimeError, match="exact 12-bundle"):
        _evaluate(summaries=missing)

    mismatched = _new_summaries()
    mismatched["seed601-dose1-clean"]["request"]["dose"] = 8
    with pytest.raises(RuntimeError, match="binding mismatch"):
        _evaluate(summaries=mismatched)

    nonfinite = _new_summaries()
    nonfinite["seed601-dose1-clean"]["endpoints"]["NC"][
        gate.PRIMARY_OUTCOMES[0]
    ] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        _evaluate(summaries=nonfinite)


def test_curve_gate_rejects_malformed_prior_decision() -> None:
    prior = _old_decision()
    prior["decision"] = "GO"
    with pytest.raises(RuntimeError, match="prior handoff"):
        _evaluate(prior=prior)

    prior = _old_decision()
    prior["observations"].pop("seed603-dose64-noisy")
    with pytest.raises(RuntimeError, match="prior handoff"):
        _evaluate(prior=prior)

    prior = _old_decision()
    prior["observations"]["seed601-dose8-clean"][gate.PRIMARY_OUTCOMES[0]][
        "repair_margin"
    ] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        _evaluate(prior=prior)


def _binding(character: str) -> dict[str, str]:
    return {
        "model_id": RESNET_MODEL_ID,
        "revision": "65a5785d9156231087c481e0c7dd33a5ff6f7e3e",
        "config_sha256": character * 64,
        "model_safetensors_sha256": character * 64,
    }


def _stores() -> dict[str, dict[str, object]]:
    return {
        str(seed): {
            "noise_bundle_sha256": curve.handoff.EXPECTED_NOISE_SHA256[seed],
            "image_store_sha256": "2" * 64,
            "tuning_store_sha256": "3" * 64,
            "source_commit": "4" * 40,
            "test_loaded": False,
        }
        for seed in curve.TRAINING_SEEDS
    }


def test_curve_contract_is_content_bound_and_fail_closed(tmp_path: Path) -> None:
    record = curve.write_contract(
        tmp_path,
        source_commit="f" * 40,
        uv_lock_sha256="a" * 64,
        prior_decision_sha256="b" * 64,
        prior_contract_sha256="c" * 64,
        model_binding=_binding("d"),
        store_bindings=_stores(),
    )
    assert record["new_doses"] == [1, 1250]
    assert record["trend_gate"]["overall_min_positive"] == 10
    with pytest.raises(RuntimeError, match="differs"):
        curve.write_contract(
            tmp_path,
            source_commit="f" * 40,
            uv_lock_sha256="a" * 64,
            prior_decision_sha256="e" * 64,
            prior_contract_sha256="c" * 64,
            model_binding=_binding("d"),
            store_bindings=_stores(),
        )


def test_curve_gate_rejects_wrong_source_commit() -> None:
    summaries = deepcopy(_new_summaries())
    summaries["seed601-dose1-clean"]["source_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="binding mismatch"):
        _evaluate(summaries=summaries)
