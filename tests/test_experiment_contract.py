from hashlib import sha256
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from arw.models import (
    CandidatePatch,
    ExperimentCancelRequest,
    ExperimentDetail,
    ExperimentSubmitRequest,
)

CONTRACT = Path(__file__).parents[1] / "contracts" / "openapi.yaml"


def test_contract_has_stage_b_paths_and_write_guards() -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    paths = document["paths"]
    assert {
        "/api/v1/experiments",
        "/api/v1/experiments/{experiment_id}",
        "/api/v1/experiments/{experiment_id}/events",
        "/api/v1/experiments/{experiment_id}/cancel",
        "/api/v1/experiments/{experiment_id}/artifacts",
        "/api/v1/experiments/{experiment_id}/artifacts/{artifact_id}/content",
    } <= set(paths)
    for operation in (
        paths["/api/v1/experiments"]["post"],
        paths["/api/v1/experiments/{experiment_id}/cancel"]["post"],
    ):
        assert operation["security"] == [{"bearerAuth": []}]
        assert any(
            parameter.get("name") == "Idempotency-Key" and parameter["required"]
            for parameter in operation["parameters"]
        )


def test_contract_write_examples_match_runtime_models() -> None:
    paths = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["paths"]
    submit = paths["/api/v1/experiments"]["post"]["requestBody"]["content"]["application/json"][
        "example"
    ]
    cancel = paths["/api/v1/experiments/{experiment_id}/cancel"]["post"]["requestBody"]["content"][
        "application/json"
    ]["example"]
    ExperimentSubmitRequest.model_validate(submit)
    ExperimentCancelRequest.model_validate(cancel)


def test_submit_is_strict_and_hash_bound() -> None:
    patch = "diff --git a/train.py b/train.py\n"
    payload = {
        "project_id": "autoresearch",
        "source_commit": "a" * 40,
        "title": "try lower lr",
        "plan_id": "plan-1",
        "candidate": {"patch_sha256": sha256(patch.encode()).hexdigest(), "patch": patch},
        "matrix": {"seeds": [1, 2], "time_budget_seconds": 300, "max_parallel": 2},
    }
    ExperimentSubmitRequest.model_validate(payload)
    payload["candidate"]["patch_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="patch_sha256"):
        ExperimentSubmitRequest.model_validate(payload)
    assert not (
        {"source_url", "adapter_type", "command", "argv", "host", "env"}
        & ExperimentSubmitRequest.model_fields.keys()
    )


def test_matrix_seeds_are_unique() -> None:
    patch = "x"
    with pytest.raises(ValidationError, match="unique"):
        ExperimentSubmitRequest.model_validate(
            {
                "project_id": "p",
                "source_commit": "a" * 40,
                "title": "t",
                "plan_id": "p",
                "candidate": {"patch_sha256": sha256(patch.encode()).hexdigest(), "patch": patch},
                "matrix": {"seeds": [1, 1], "time_budget_seconds": 30, "max_parallel": 1},
            }
        )


def test_experiment_terminal_invariants() -> None:
    base = {
        "experiment_id": "exp-1",
        "project_id": "p",
        "source_commit": "a" * 40,
        "title": "t",
        "plan_id": "plan",
        "state": "succeeded",
        "version": 2,
        "candidate_patch_sha256": "b" * 64,
        "matrix": {"seeds": [1], "time_budget_seconds": 30, "max_parallel": 1},
        "error": None,
    }
    with pytest.raises(ValidationError, match="result"):
        ExperimentDetail.model_validate({**base, "result": None})
    ExperimentDetail.model_validate({**base, "result": {"summary": "ok", "artifact_count": 1}})


def test_candidate_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CandidatePatch.model_validate({"patch_sha256": "a" * 64, "patch": "x", "command": "bad"})
