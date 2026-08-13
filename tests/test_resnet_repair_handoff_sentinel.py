from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import causal_transport_gate as gate
from experiments import resnet_repair_handoff_sentinel as sentinel
from experiments.repair_handoff_models import RESNET_MODEL_ID, RESNET_REVISION


def test_build_jobs_is_exact_matrix(tmp_path: Path) -> None:
    assert [job.key for job in sentinel.build_jobs(tmp_path)] == [
        f"seed{seed}-dose{dose}-{continuation}"
        for seed in (601, 602, 603)
        for dose in (8, 64)
        for continuation in ("clean", "noisy")
    ]


def _summary(
    *, seed: int, dose: int, continuation: str, parameter_better: bool
) -> dict[str, object]:
    endpoints: dict[str, dict[str, float]] = {
        branch: {} for branch in ("CC", "CN", "NC", "NN")
    }
    effects: dict[str, dict[str, float]] = {}
    for outcome in gate.PRIMARY_OUTCOMES:
        endpoints["CC"][outcome] = 0.0
        endpoints["NN"][outcome] = 10.0
        endpoints["CN"][outcome] = 2.0 if parameter_better else 9.0
        endpoints["NC"][outcome] = 8.0 if parameter_better else 1.0
        effects[outcome] = {
            "parameter": 2.0 if parameter_better else 0.1,
            "state": 0.1 if parameter_better else 2.0,
            "interaction": 0.0,
        }
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
        "effects": effects,
        "endpoints": endpoints,
        "test_loaded": False,
    }


def _passing_summaries() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for job in sentinel.build_jobs(Path("unused")):
        result[job.key] = _summary(
            seed=job.seed,
            dose=job.dose,
            continuation=job.continuation,
            parameter_better=job.dose == 64,
        )
    # The frozen short-dose rule permits one positive observation per seed.
    for seed in sentinel.TRAINING_SEEDS:
        summary = result[f"seed{seed}-dose8-clean"]
        outcome = gate.PRIMARY_OUTCOMES[0]
        summary["endpoints"]["CN"][outcome] = 2.0
        summary["endpoints"]["NC"][outcome] = 8.0
    return result


def test_gate_requires_short_state_and_long_parameter_repairs() -> None:
    record = sentinel.evaluate_sentinel(
        _passing_summaries(), contract_sha256="a" * 64
    )
    assert record["decision"] == "GO"
    assert record["checks"] == {
        "exact_valid_matrix": True,
        "dose64_all_12_parameter_repair": True,
        "dose8_at_most_3_parameter_repair": True,
        "dose8_each_seed_at_most_1": True,
    }


def test_gate_rejects_each_scientific_failure() -> None:
    long_failure = _passing_summaries()
    long_failure["seed601-dose64-clean"] = _summary(
        seed=601, dose=64, continuation="clean", parameter_better=False
    )
    assert (
        sentinel.evaluate_sentinel(long_failure, contract_sha256="a" * 64)[
            "decision"
        ]
        == "NO-GO"
    )

    too_many_short = _passing_summaries()
    summary = too_many_short["seed601-dose8-noisy"]
    outcome = gate.PRIMARY_OUTCOMES[0]
    summary["endpoints"]["CN"][outcome] = 2.0
    summary["endpoints"]["NC"][outcome] = 8.0
    decision = sentinel.evaluate_sentinel(
        too_many_short, contract_sha256="a" * 64
    )
    assert decision["decision"] == "NO-GO"
    assert decision["checks"]["dose8_each_seed_at_most_1"] is False


def test_gate_rejects_missing_mismatched_or_nonfinite_bundle() -> None:
    missing = _passing_summaries()
    missing.pop("seed603-dose64-noisy")
    with pytest.raises(RuntimeError, match="exact 12-bundle"):
        sentinel.evaluate_sentinel(missing, contract_sha256="a" * 64)

    mismatch = _passing_summaries()
    mismatch["seed601-dose8-clean"]["request"]["dose"] = 64
    assert (
        sentinel.evaluate_sentinel(mismatch, contract_sha256="a" * 64)[
            "decision"
        ]
        == "NO-GO"
    )

    nonfinite = _passing_summaries()
    nonfinite["seed601-dose8-clean"]["endpoints"]["NN"][
        gate.PRIMARY_OUTCOMES[0]
    ] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        sentinel.evaluate_sentinel(nonfinite, contract_sha256="a" * 64)


def _binding(character: str) -> dict[str, str]:
    return {
        "model_id": RESNET_MODEL_ID,
        "revision": RESNET_REVISION,
        "config_sha256": character * 64,
        "model_safetensors_sha256": character * 64,
    }


def _stores() -> dict[str, dict[str, object]]:
    return {
        str(seed): {
            "noise_bundle_sha256": sentinel.EXPECTED_NOISE_SHA256[seed],
            "image_store_sha256": "2" * 64,
            "tuning_store_sha256": "3" * 64,
            "source_commit": "4" * 40,
            "test_loaded": False,
        }
        for seed in sentinel.TRAINING_SEEDS
    }


def test_contract_is_content_bound_and_fail_closed(tmp_path: Path) -> None:
    record = sentinel.write_contract(
        tmp_path,
        source_commit="f" * 40,
        uv_lock_sha256="a" * 64,
        discovery_decision_sha256="b" * 64,
        prior_sentinel_decision_sha256="c" * 64,
        model_binding=_binding("d"),
        store_bindings=_stores(),
    )
    assert json.loads((tmp_path / "REPAIR_HANDOFF_CONTRACT.json").read_text()) == record
    with pytest.raises(RuntimeError, match="differs"):
        sentinel.write_contract(
            tmp_path,
            source_commit="f" * 40,
            uv_lock_sha256="a" * 64,
            discovery_decision_sha256="b" * 64,
            prior_sentinel_decision_sha256="c" * 64,
            model_binding=_binding("e"),
            store_bindings=_stores(),
        )


def test_preflight_evaluation_and_binding() -> None:
    model_binding = _binding("d")
    record = {
        "source_commit": "f" * 40,
        "model_binding": model_binding,
        "cuda": True,
        "finite_losses": True,
        "optimizer_state_nonempty": True,
        "restore_exact": True,
        "peak_vram_gb": 2.5,
        "test_loaded": False,
    }
    assert sentinel.evaluate_preflight(record)["status"] == "passed"
    sentinel.validate_preflight(
        {**record, "status": "passed"},
        source_commit="f" * 40,
        model_binding=model_binding,
    )
    record["restore_exact"] = False
    assert sentinel.evaluate_preflight(record)["status"] == "failed"
    with pytest.raises(RuntimeError, match="binding"):
        sentinel.validate_preflight(
            {**record, "status": "passed"},
            source_commit="f" * 40,
            model_binding=_binding("e"),
        )
