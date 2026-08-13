from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import attribution_transport_sentinel as sentinel
from experiments import causal_transport_gate as gate


def test_symmetric_uniform_labels_are_exact_deterministic_and_wrong() -> None:
    clean = np.repeat(np.arange(3), 5)
    ids = np.asarray([f"sample-{index}".encode() for index in range(15)])
    first, first_mask = sentinel.symmetric_uniform_labels(
        clean,
        ids,
        noise_seed=1501,
        class_count=3,
        corruptions_per_class=2,
    )
    second, second_mask = sentinel.symmetric_uniform_labels(
        clean,
        ids,
        noise_seed=1501,
        class_count=3,
        corruptions_per_class=2,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    assert np.array_equal(np.bincount(clean[first_mask], minlength=3), [2, 2, 2])
    assert np.all(first[first_mask] != clean[first_mask])
    assert np.array_equal(first[~first_mask], clean[~first_mask])


def test_build_jobs_is_exact_frozen_matrix(tmp_path: Path) -> None:
    jobs = sentinel.build_jobs(tmp_path)
    assert [job.key for job in jobs] == [
        f"seed{seed}-dose1250-{continuation}"
        for seed in (501, 502, 503)
        for continuation in ("clean", "noisy")
    ]


def _summary(*, parameter: bool = True, repair_better: bool = True) -> dict[str, object]:
    effects = (
        {"parameter": 2.0, "state": 0.1, "interaction": 0.0}
        if parameter
        else {"parameter": 0.1, "state": 2.0, "interaction": 0.0}
    )
    endpoints: dict[str, dict[str, float]] = {
        branch: {} for branch in ("CC", "CN", "NC", "NN")
    }
    for outcome in gate.PRIMARY_OUTCOMES:
        endpoints["CC"][outcome] = 0.0
        endpoints["NN"][outcome] = 10.0
        endpoints["CN"][outcome] = 2.0 if repair_better else 9.0
        endpoints["NC"][outcome] = 8.0 if repair_better else 1.0
    return {
        "status": "succeeded",
        "contract_sha256": "a" * 64,
        "request": {"seed": 501, "dose": 1250, "continuation": "clean"},
        "effects": {outcome: effects for outcome in gate.PRIMARY_OUTCOMES},
        "endpoints": endpoints,
        "test_loaded": False,
    }


def _summaries() -> dict[str, dict[str, object]]:
    summaries = {}
    for job in sentinel.build_jobs(Path("unused")):
        summary = _summary()
        summary["request"] = {
            "seed": job.seed,
            "dose": 1250,
            "continuation": job.continuation,
        }
        summaries[job.key] = summary
    return summaries


def test_evaluate_sentinel_requires_all_carriers_and_repairs() -> None:
    passing = sentinel.evaluate_sentinel(_summaries(), contract_sha256="a" * 64)
    assert passing["decision"] == "GO"
    assert passing["checks"] == {
        "exact_valid_matrix": True,
        "all_12_parameter": True,
        "all_12_parameter_repair_better": True,
    }

    carrier_failure = _summaries()
    carrier_failure["seed501-dose1250-clean"]["effects"] = _summary(
        parameter=False
    )["effects"]
    assert (
        sentinel.evaluate_sentinel(carrier_failure, contract_sha256="a" * 64)[
            "decision"
        ]
        == "NO-GO"
    )

    repair_failure = _summaries()
    repair_failure["seed503-dose1250-noisy"]["endpoints"] = _summary(
        repair_better=False
    )["endpoints"]
    assert (
        sentinel.evaluate_sentinel(repair_failure, contract_sha256="a" * 64)[
            "decision"
        ]
        == "NO-GO"
    )


def test_evaluate_sentinel_rejects_missing_or_nonfinite_bundle() -> None:
    missing = _summaries()
    missing.pop("seed503-dose1250-noisy")
    with pytest.raises(RuntimeError, match="exact six-bundle"):
        sentinel.evaluate_sentinel(missing, contract_sha256="a" * 64)

    nonfinite = _summaries()
    raw = nonfinite["seed501-dose1250-clean"]["effects"]
    assert isinstance(raw, dict)
    raw[gate.PRIMARY_OUTCOMES[0]]["parameter"] = float("nan")
    with pytest.raises(RuntimeError, match="malformed"):
        sentinel.evaluate_sentinel(nonfinite, contract_sha256="a" * 64)


def test_write_contract_is_content_bound_and_fail_closed(tmp_path: Path) -> None:
    binding = {
        "noise_bundle_sha256": "b" * 64,
        "image_store_sha256": "c" * 64,
        "tuning_store_sha256": "d" * 64,
        "source_commit": "e" * 40,
        "test_loaded": False,
    }
    bindings = {str(seed): binding for seed in sentinel.TRAINING_SEEDS}
    record = sentinel.write_contract(
        tmp_path,
        source_commit="f" * 40,
        uv_lock_sha256="1" * 64,
        discovery_decision_sha256="2" * 64,
        store_bindings=bindings,
    )
    assert json.loads((tmp_path / "SENTINEL_CONTRACT.json").read_text()) == record
    assert sentinel.write_contract(
        tmp_path,
        source_commit="f" * 40,
        uv_lock_sha256="1" * 64,
        discovery_decision_sha256="2" * 64,
        store_bindings=bindings,
    ) == record
    with pytest.raises(RuntimeError, match="differs"):
        sentinel.write_contract(
            tmp_path,
            source_commit="f" * 40,
            uv_lock_sha256="1" * 64,
            discovery_decision_sha256="3" * 64,
            store_bindings=bindings,
        )
