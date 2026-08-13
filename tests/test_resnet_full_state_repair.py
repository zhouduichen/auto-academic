from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments import causal_attribution_replay as replay
from experiments import resnet_full_state_repair as sentinel
from experiments.repair_handoff_models import RESNET_MODEL_ID, RESNET_REVISION


def test_jobs_are_exact_two_conditions_and_two_orders(tmp_path: Path) -> None:
    assert [job.key for job in sentinel.build_jobs(tmp_path)] == [
        "seed601-dose1250-clean-standard",
        "seed601-dose1250-clean-reversed",
        "seed601-dose1250-noisy-standard",
        "seed601-dose1250-noisy-reversed",
    ]


def _summary(continuation: str, order: tuple[str, ...]) -> dict[str, object]:
    return {
        "status": "succeeded",
        "source_commit": "f" * 40,
        "contract_sha256": "a" * 64,
        "branch_order": list(order),
        "request": {
            "seed": 601,
            "dose": 1250,
            "continuation": continuation,
            "config": {"model_id": RESNET_MODEL_ID},
        },
        "endpoints": {"value": 1.0},
        "effects": {"value": 2.0},
        "test_loaded": False,
    }


def _row(branch: str, horizon: int, value: float) -> dict[str, object]:
    return {
        "branch": branch,
        "horizon": horizon,
        "probe_loss": value,
        "clean_loss_excess": value,
        "test_loaded": False,
    }


def _passing() -> tuple[
    dict[str, dict[str, object]], dict[str, list[dict[str, object]]]
]:
    summaries = {}
    trajectories = {}
    base = [
        _row(branch, horizon, float(index + horizon))
        for index, branch in enumerate(replay.BRANCH_NAMES)
        for horizon in (0, 1)
    ]
    for job in sentinel.build_jobs(Path("unused")):
        summaries[job.key] = _summary(job.continuation, job.branch_order)
        trajectories[job.key] = deepcopy(base)
    return summaries, trajectories


def _evaluate(
    summaries: dict[str, dict[str, object]],
    trajectories: dict[str, list[dict[str, object]]],
    *,
    buffers: bool = True,
    resume: bool = True,
) -> dict[str, object]:
    return sentinel.evaluate_sentinel(
        summaries,
        trajectories,
        contract_sha256="a" * 64,
        source_commit="f" * 40,
        snapshot_schema_and_buffers_exact=buffers,
        resume_hashes_exact=resume,
    )


def test_gate_passes_only_exact_full_state_invariant_matrix() -> None:
    summaries, trajectories = _passing()
    record = _evaluate(summaries, trajectories)
    assert record["decision"] == "GO"
    assert all(record["checks"].values())


@pytest.mark.parametrize("failure", ["buffers", "resume", "order", "horizon0"])
def test_gate_fails_each_correctness_violation(failure: str) -> None:
    summaries, trajectories = _passing()
    buffers = resume = True
    if failure == "buffers":
        buffers = False
    elif failure == "resume":
        resume = False
    elif failure == "order":
        trajectories["seed601-dose1250-clean-reversed"][4]["probe_loss"] = 99.0
    else:
        trajectories["seed601-dose1250-noisy-standard"][0]["probe_loss"] = 99.0
    assert (
        _evaluate(
            summaries, trajectories, buffers=buffers, resume=resume
        )["decision"]
        == "NO-GO"
    )


def test_gate_rejects_missing_mismatched_or_nonfinite_bundle() -> None:
    summaries, trajectories = _passing()
    summaries.pop("seed601-dose1250-noisy-reversed")
    assert _evaluate(summaries, trajectories)["decision"] == "NO-GO"

    summaries, trajectories = _passing()
    summaries["seed601-dose1250-clean-standard"]["source_commit"] = "0" * 40
    assert _evaluate(summaries, trajectories)["decision"] == "NO-GO"

    summaries, trajectories = _passing()
    trajectories["seed601-dose1250-clean-standard"][0]["probe_loss"] = float("nan")
    assert _evaluate(summaries, trajectories)["decision"] == "NO-GO"


def _binding(character: str) -> dict[str, str]:
    return {
        "model_id": RESNET_MODEL_ID,
        "revision": RESNET_REVISION,
        "config_sha256": character * 64,
        "model_safetensors_sha256": character * 64,
    }


def _store() -> dict[str, object]:
    return {
        "noise_bundle_sha256": sentinel.handoff.EXPECTED_NOISE_SHA256[601],
        "image_store_sha256": "2" * 64,
        "tuning_store_sha256": "3" * 64,
        "source_commit": "4" * 40,
        "test_loaded": False,
    }


def test_contract_binds_excluded_artifacts_and_fails_closed(tmp_path: Path) -> None:
    record = sentinel.write_contract(
        tmp_path,
        source_commit="f" * 40,
        uv_lock_sha256="a" * 64,
        model_binding=_binding("b"),
        store_binding=_store(),
        excluded_artifact_sha256={"handoff": "c" * 64, "curve": "d" * 64},
    )
    assert record["checkpoint_schema"] == "causal-transport-exposure/2"
    assert record["excluded_resnet_bundle_count"] == 24
    with pytest.raises(RuntimeError, match="differs"):
        sentinel.write_contract(
            tmp_path,
            source_commit="f" * 40,
            uv_lock_sha256="a" * 64,
            model_binding=_binding("b"),
            store_binding=_store(),
            excluded_artifact_sha256={"handoff": "e" * 64, "curve": "d" * 64},
        )
