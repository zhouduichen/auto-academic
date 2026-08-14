from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments import causal_attribution_replay as replay
from experiments import resnet_full_state_stage2 as stage2
from experiments.repair_handoff_models import RESNET_MODEL_ID


def _summary(seed: int, dose: int, continuation: str, reused: bool) -> dict[str, object]:
    return {
        "status": "succeeded",
        "source_commit": ("1" if reused else "2") * 40,
        "contract_sha256": ("a" if reused else "b") * 64,
        "branch_order": list(replay.BRANCH_NAMES),
        "request": {
            "seed": seed,
            "dose": dose,
            "continuation": continuation,
            "config": {"model_id": RESNET_MODEL_ID},
        },
        "test_loaded": False,
    }


def _rows(clean: bool, seed: int) -> list[dict[str, object]]:
    rows = []
    for branch in replay.BRANCH_NAMES:
        for horizon in replay.HORIZONS:
            value = 1.0
            if branch == "NC":
                value = 1.2 if clean else 0.8
                if seed == 603 and horizon in stage2.PRIMARY_HORIZONS[:3]:
                    value = 1.2 if clean else 0.8
            elif branch == "CN":
                value = 1.0
            rows.append(
                {
                    "branch": branch,
                    "horizon": horizon,
                    "clean_loss_excess": value,
                    "test_loaded": False,
                }
            )
    return rows


def _passing() -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    summaries = {}
    trajectories = {}
    for job in stage2.build_jobs(Path("unused")):
        reused = job.key in stage2.REUSED_KEYS
        summaries[job.key] = _summary(job.seed, job.dose, job.continuation, reused)
        trajectories[job.key] = _rows(job.continuation == "clean", job.seed)
    return summaries, trajectories


def _evaluate(summaries, trajectories, *, resume: bool = True):
    return stage2.evaluate_stage2(
        summaries,
        trajectories,
        stage2_contract_sha256="b" * 64,
        stage1_contract_sha256="a" * 64,
        source_commit="2" * 40,
        stage1_source_commit="1" * 40,
        resume_hashes_exact=resume,
    )


def test_exact_matrix_has_ten_new_and_two_reused_jobs() -> None:
    jobs = stage2.build_jobs(Path("unused"))
    assert len(jobs) == 12
    assert sum(job.key not in stage2.REUSED_KEYS for job in jobs) == 10


def test_stage2_gate_passes_frozen_context_and_switch_rules() -> None:
    summaries, trajectories = _passing()
    decision = _evaluate(summaries, trajectories)
    assert decision["decision"] == "GO"
    assert all(decision["checks"].values())


def test_stage2_gate_fails_closed_on_matrix_provenance_or_science() -> None:
    summaries, trajectories = _passing()
    summaries.pop("seed603-dose64-noisy")
    assert _evaluate(summaries, trajectories)["decision"] == "NO-GO"

    summaries, trajectories = _passing()
    summaries["seed602-dose1250-clean"]["source_commit"] = "0" * 40
    assert _evaluate(summaries, trajectories)["decision"] == "NO-GO"

    summaries, trajectories = _passing()
    broken = deepcopy(trajectories["seed602-dose1250-noisy"])
    for row in broken:
        if row["branch"] == "NC":
            row["clean_loss_excess"] = 1.3
    trajectories["seed602-dose1250-noisy"] = broken
    assert _evaluate(summaries, trajectories)["decision"] == "NO-GO"

    summaries, trajectories = _passing()
    assert _evaluate(summaries, trajectories, resume=False)["decision"] == "NO-GO"
