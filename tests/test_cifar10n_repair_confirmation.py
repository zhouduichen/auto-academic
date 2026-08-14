from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments import causal_attribution_replay as replay
from experiments import cifar10n_repair_confirmation as confirmation
from experiments import repair_handoff_models as models


def _summary(job: confirmation.ConfirmationJob) -> dict[str, object]:
    model_id = models.RESNET_MODEL_ID if job.model == "resnet18" else models.VIT_MODEL_ID
    score = 0.2 if job.continuation == "clean" else 0.8
    return {
        "status": "succeeded",
        "source_commit": "1" * 40,
        "contract_sha256": "a" * 64,
        "request": {
            "dataset_name": "cifar10n",
            "num_labels": 10,
            "diagnostic_prefix_steps": 8,
            "seed": job.seed,
            "dose": 1250,
            "continuation": job.continuation,
            "config": {"model_id": model_id},
        },
        "diagnostic": {
            "prefix_steps": 8,
            "normalized_stream_label_nll": score,
        },
        "test_loaded": False,
    }


def _trajectory(continuation: str) -> list[dict[str, object]]:
    rows = []
    for branch in replay.BRANCH_NAMES:
        for horizon in replay.HORIZONS:
            value = 0.0
            if branch == "CN":
                value = -1.0 if continuation == "clean" else 1.0
            elif branch == "NC":
                value = 1.0 if continuation == "clean" else -1.0
            elif branch == "NN":
                value = 2.0
            rows.append(
                {"branch": branch, "horizon": horizon, "clean_loss_excess": value}
            )
    return rows


def _passing():
    summaries = {}
    trajectories = {}
    for job in confirmation.build_jobs(Path("unused")):
        summaries[job.key] = _summary(job)
        trajectories[job.key] = _trajectory(job.continuation)
    policy = {
        "decision": "GO",
        "models": {
            "resnet18": {"threshold": 0.5},
            "vit_lora": {"threshold": 0.5},
        },
    }
    return summaries, trajectories, policy


def _evaluate(summaries, trajectories, policy, *, resume: bool = True):
    return confirmation.evaluate_confirmation(
        summaries,
        trajectories,
        contract_sha256="a" * 64,
        expected_source_commit="1" * 40,
        policy_decision=policy,
        resume_hashes_exact=resume,
    )


def test_jobs_are_exact_twenty_bundle_matrix() -> None:
    jobs = confirmation.build_jobs(Path("unused"))
    assert len(jobs) == 20
    assert len({job.key for job in jobs}) == 20


def test_mechanism_and_policy_gates_pass_frozen_rules() -> None:
    mechanism, policy, report = _evaluate(*_passing())
    assert mechanism["decision"] == "GO"
    assert policy["decision"] == "GO"
    assert policy["unit_wins"] == 10
    assert policy["normalized_oracle_regret"] == 0.0
    assert report["bundle_count"] == 20


def test_gate_fails_matrix_provenance_resume_and_science() -> None:
    summaries, trajectories, policy = _passing()
    summaries.pop(next(iter(summaries)))
    assert _evaluate(summaries, trajectories, policy)[0]["decision"] == "NO-GO"

    summaries, trajectories, policy = _passing()
    summaries["resnet18-seed701-dose1250-clean"]["source_commit"] = "0" * 40
    assert _evaluate(summaries, trajectories, policy)[0]["decision"] == "NO-GO"

    summaries, trajectories, policy = _passing()
    assert _evaluate(summaries, trajectories, policy, resume=False)[0]["decision"] == "NO-GO"

    summaries, trajectories, policy = _passing()
    for seed in (701, 702):
        key = f"vit_lora-seed{seed}-dose1250-noisy"
        broken = deepcopy(trajectories[key])
        for row in broken:
            if row["branch"] == "NC":
                row["clean_loss_excess"] = 2.0
        trajectories[key] = broken
    assert _evaluate(summaries, trajectories, policy)[0]["decision"] == "NO-GO"


def test_policy_is_ineligible_after_calibration_no_go() -> None:
    summaries, trajectories, policy = _passing()
    policy["decision"] = "NO-GO"
    mechanism, policy_result, _ = _evaluate(summaries, trajectories, policy)
    assert mechanism["decision"] == "GO"
    assert policy_result["decision"] == "NO-GO"
    assert policy_result["checks"]["calibration_eligible"] is False
