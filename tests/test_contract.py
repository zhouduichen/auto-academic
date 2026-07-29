from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from arw.models import (
    HealthResponse,
    MetaResponse,
    ProjectListResponse,
    StatusResponse,
    TaskListResponse,
)

CONTRACT = Path(__file__).parents[1] / "contracts" / "openapi.yaml"


def test_contract_defines_read_only_paths() -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert document["openapi"] == "3.1.0"
    assert {
        "/healthz",
        "/api/v1/meta",
        "/api/v1/status",
        "/api/v1/projects",
        "/api/v1/tasks",
    } <= set(document["paths"])


def test_contract_forbids_unevaluated_and_additional_properties() -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    schemas = document["components"]["schemas"]

    for name in (
        "HealthResponse",
        "MetaResponse",
        "StatusResponse",
        "ProjectListResponse",
        "TaskListResponse",
    ):
        assert schemas[name]["unevaluatedProperties"] is False

    for name in ("StatusData", "ProjectSummary", "TaskSummary"):
        assert schemas[name]["additionalProperties"] is False


def test_models_accept_contract_examples() -> None:
    HealthResponse.model_validate({"api_version": "1.0", "request_id": "req_1", "status": "ok"})
    MetaResponse.model_validate(
        {
            "api_version": "1.0",
            "request_id": "req_2",
            "server_version": "0.1.0",
            "minimum_client_version": "0.1.0",
            "features": ["projects.read", "tasks.read"],
        }
    )
    StatusResponse.model_validate(
        {
            "api_version": "1.0",
            "request_id": "req_3",
            "data": {
                "node_id": "autoresearch-5080",
                "state": "ready",
                "active_tasks": 1,
                "waiting_approval": 2,
            },
        }
    )
    ProjectListResponse.model_validate(
        {
            "api_version": "1.0",
            "request_id": "req_4",
            "items": [
                {
                    "project_id": "karpathy-autoresearch",
                    "name": "Karpathy AutoResearch",
                    "source_url": "https://github.com/karpathy/autoresearch",
                    "source_commit": "228791fb499afffb54b46200aca536f79142f117",
                    "execution_platform": "windows_cuda",
                }
            ],
            "next_cursor": None,
        }
    )
    TaskListResponse.model_validate(
        {
            "api_version": "1.0",
            "request_id": "req_5",
            "items": [
                {
                    "task_id": "task_1",
                    "project_id": "karpathy-autoresearch",
                    "title": "baseline",
                    "status": "waiting-approval",
                    "version": 3,
                }
            ],
            "next_cursor": None,
        }
    )


def test_list_models_require_next_cursor() -> None:
    with pytest.raises(ValidationError, match="next_cursor"):
        ProjectListResponse.model_validate(
            {"api_version": "1.0", "request_id": "req_projects", "items": []}
        )

    with pytest.raises(ValidationError, match="next_cursor"):
        TaskListResponse.model_validate(
            {"api_version": "1.0", "request_id": "req_tasks", "items": []}
        )


def test_models_reject_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        HealthResponse.model_validate(
            {
                "api_version": "1.0",
                "request_id": "req_health",
                "status": "ok",
                "secret": "must-not-pass",
            }
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        StatusResponse.model_validate(
            {
                "api_version": "1.0",
                "request_id": "req_status",
                "data": {
                    "node_id": "autoresearch-5080",
                    "state": "ready",
                    "active_tasks": 0,
                    "waiting_approval": 0,
                    "secret": "must-not-pass",
                },
            }
        )
