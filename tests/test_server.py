"""Comprehensive server-level tests for the ARW Windows coordinator."""

from __future__ import annotations

import sqlite3
import time
import uuid
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arw.models import (
    CandidatePatch,
    ExperimentCancelRequest,
    ExperimentMatrix,
    ExperimentSubmitRequest,
)
from arw_server.db import now
from arw_server.executor import FakeExecutor
from arw_server.main import create_app
from arw_server.state import (
    ALL_STATES,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    can_transition,
    is_terminal,
)

pytestmark = pytest.mark.stage_c

FIXED_PATCH = "diff --git a/train.py b/train.py\n@@ -1 +1 @@\n-old\n+new\n"
FIXED_COMMIT = "228791fb499afffb54b46200aca536f79142f117"


def _make_submit_body(
    project_id: str = "karpathy-autoresearch",
    title: str = "test experiment",
    plan_id: str = "plan-1",
    patch: str | None = None,
    seeds: list[int] | None = None,
) -> ExperimentSubmitRequest:
    p = patch or FIXED_PATCH
    return ExperimentSubmitRequest(
        project_id=project_id,
        source_commit=FIXED_COMMIT,
        title=title,
        plan_id=plan_id,
        candidate=CandidatePatch(
            patch_sha256=sha256(p.encode()).hexdigest(),
            patch=p,
        ),
        matrix=ExperimentMatrix(
            seeds=seeds or [1],
            time_budget_seconds=60,
            max_parallel=1,
        ),
    )


def _set_state(db_path: Path, experiment_id: str, state: str, version: int) -> None:
    """Directly set experiment state and version for test setup."""
    db = sqlite3.connect(str(db_path))
    db.execute(
        "UPDATE experiments SET state = ?, version = ?, updated_at = ? WHERE experiment_id = ?",
        (state, version, now(), experiment_id),
    )
    db.commit()
    db.close()


def _wait_for_terminal(client: TestClient, experiment_id: str, timeout: float = 10.0) -> dict:
    """Poll until an experiment reaches a terminal state, then return the response JSON."""
    deadline = time.time() + timeout
    state: str = ""
    body: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/api/v1/experiments/{experiment_id}")
        body = resp.json()
        state = body["data"]["state"]
        if state in ("succeeded", "failed", "cancelled", "policy-rejected"):
            return body
        time.sleep(0.1)
    raise TimeoutError(
        f"experiment {experiment_id} did not reach terminal state "
        f"within {timeout}s (last state: {state})"
    )


# ── Fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    app = create_app(
        db_path=db_path,
        api_token="test-token",
        worktree_root=tmp_path / "worktrees",
    )
    with TestClient(app) as c:
        c.headers = {"Authorization": "Bearer test-token"}
        c._test_db_path = db_path  # type: ignore[attr-defined]
        yield c


# ── 1. /healthz ────────────────────────────────────────────────────────────────


def test_healthz_returns_ok(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["api_version"] == "1.0"
    assert len(body["request_id"]) > 0


# ── 2. /api/v1/meta requires auth ──────────────────────────────────────────────


def test_meta_requires_auth(tmp_path: Path) -> None:
    db_path = tmp_path / "meta_auth.db"
    app = create_app(db_path=db_path, api_token="test-token", worktree_root=tmp_path / "w")
    with TestClient(app) as c:
        resp = c.get("/api/v1/meta")
        assert resp.status_code == 401


# ── 3. /api/v1/meta with valid token ───────────────────────────────────────────


def test_meta_with_valid_token(client: TestClient) -> None:
    resp = client.get("/api/v1/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert body["server_version"] == "0.1.0"
    assert body["minimum_client_version"] == "0.1.0"
    assert "experiments.read" in body["features"]
    assert "experiments.submit" in body["features"]


# ── 4. /api/v1/projects ───────────────────────────────────────────────────────


def test_list_projects_returns_karpathy(client: TestClient) -> None:
    resp = client.get("/api/v1/projects")
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_version"] == "1.0"
    assert body["next_cursor"] is None
    items = body["items"]
    assert len(items) == 1
    assert items[0]["project_id"] == "karpathy-autoresearch"
    assert items[0]["name"] == "Karpathy AutoResearch"
    assert items[0]["source_commit"] == FIXED_COMMIT
    assert items[0]["execution_platform"] == "windows_cuda"


# ── 5. POST /api/v1/experiments ───────────────────────────────────────────────


def test_submit_experiment_creates_and_returns_201(client: TestClient) -> None:
    body = _make_submit_body(title="integration-test")
    resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    assert resp.status_code == 201
    data = resp.json()
    assert data["api_version"] == "1.0"
    detail = data["data"]
    assert detail["experiment_id"]
    assert len(detail["experiment_id"]) == 16
    # After auto-validation the experiment reaches "queued"
    assert detail["state"] == "queued"
    assert detail["version"] == 4
    assert detail["project_id"] == "karpathy-autoresearch"
    assert detail["title"] == "integration-test"
    assert detail["plan_id"] == "plan-1"
    assert detail["source_commit"] == FIXED_COMMIT


# ── 6. POST /api/v1/experiments requires auth ──────────────────────────────────


def test_submit_experiment_requires_auth(tmp_path: Path) -> None:
    db_path = tmp_path / "submit_auth.db"
    app = create_app(db_path=db_path, api_token="test-token", worktree_root=tmp_path / "w")
    body = _make_submit_body()
    with TestClient(app) as c:
        resp = c.post("/api/v1/experiments", json=body.model_dump(mode="json"))
        assert resp.status_code == 401


# ── 7. POST rejects unknown project ────────────────────────────────────────────


def test_submit_experiment_rejects_unknown_project(client: TestClient) -> None:
    body = _make_submit_body(project_id="unknown-project")
    resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "unknown_project"


# ── 8. Idempotency-Key ─────────────────────────────────────────────────────────


def test_submit_experiment_idempotency(client: TestClient) -> None:
    body = _make_submit_body(title="idempotent")
    key = uuid.uuid4().hex
    merged = {**client.headers, "Idempotency-Key": key}

    resp1 = client.post("/api/v1/experiments", json=body.model_dump(mode="json"), headers=merged)
    assert resp1.status_code == 201
    exp_id1 = resp1.json()["data"]["experiment_id"]

    resp2 = client.post("/api/v1/experiments", json=body.model_dump(mode="json"), headers=merged)
    assert resp2.status_code == 201
    exp_id2 = resp2.json()["data"]["experiment_id"]
    assert exp_id1 == exp_id2

    # A different key creates a new experiment
    merged2 = {**client.headers, "Idempotency-Key": uuid.uuid4().hex}
    resp3 = client.post("/api/v1/experiments", json=body.model_dump(mode="json"), headers=merged2)
    assert resp3.status_code == 201
    assert resp3.json()["data"]["experiment_id"] != exp_id1


# ── 9. GET experiment after submit ─────────────────────────────────────────────


def test_get_experiment_after_submit(client: TestClient) -> None:
    body = _make_submit_body(title="get-after-submit")
    submit_resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    exp_id = submit_resp.json()["data"]["experiment_id"]

    get_resp = client.get(f"/api/v1/experiments/{exp_id}")
    assert get_resp.status_code == 200
    detail = get_resp.json()["data"]
    assert detail["experiment_id"] == exp_id
    assert detail["title"] == "get-after-submit"
    assert detail["state"] == "queued"


# ── 10. GET /api/v1/experiments ────────────────────────────────────────────────


def test_list_experiments(client: TestClient) -> None:
    body = _make_submit_body(title="list-test")
    submit_resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    exp_id = submit_resp.json()["data"]["experiment_id"]

    resp = client.get("/api/v1/experiments")
    assert resp.status_code == 200
    body_json = resp.json()
    assert body_json["api_version"] == "1.0"
    assert body_json["next_cursor"] is None
    items = body_json["items"]
    assert len(items) >= 1
    ids = [item["experiment_id"] for item in items]
    assert exp_id in ids


# ── 11. GET experiment events ──────────────────────────────────────────────────


def test_get_experiment_events(client: TestClient) -> None:
    body = _make_submit_body(title="events-test")
    submit_resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    exp_id = submit_resp.json()["data"]["experiment_id"]

    resp = client.get(f"/api/v1/experiments/{exp_id}/events")
    assert resp.status_code == 200
    body_json = resp.json()
    assert body_json["api_version"] == "1.0"
    assert body_json["next_cursor"] is None
    events = body_json["items"]
    event_types = [e["event_type"] for e in events]
    # Auto-validation produces: submitted → policy-validating → waiting-approval → queued
    assert "submitted" in event_types
    assert "policy-validating" in event_types
    assert "waiting-approval" in event_types
    assert "queued" in event_types
    # Events are ordered by occurred_at
    for event in events:
        assert event["experiment_id"] == exp_id
        assert event["event_id"]
        assert event["version"] >= 1


# ── 12. POST cancel ────────────────────────────────────────────────────────────


def test_cancel_experiment(client: TestClient) -> None:
    body = _make_submit_body(title="cancel-me")
    submit_resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    exp_id = submit_resp.json()["data"]["experiment_id"]

    # Move experiment to "running" so cancelling is a valid transition
    _set_state(client._test_db_path, exp_id, "running", 5)  # type: ignore[attr-defined]

    cancel = ExperimentCancelRequest(reason="no longer needed", expected_version=5)
    cancel_resp = client.post(
        f"/api/v1/experiments/{exp_id}/cancel", json=cancel.model_dump(mode="json")
    )
    assert cancel_resp.status_code == 200
    detail = cancel_resp.json()["data"]
    assert detail["state"] == "cancelled"
    assert detail["version"] == 6


# ── 13. Cancel with wrong expected_version ─────────────────────────────────────


def test_cancel_with_wrong_version_returns_409(client: TestClient) -> None:
    body = _make_submit_body(title="version-conflict")
    submit_resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    exp_id = submit_resp.json()["data"]["experiment_id"]

    _set_state(client._test_db_path, exp_id, "running", 5)  # type: ignore[attr-defined]

    cancel = ExperimentCancelRequest(reason="stop", expected_version=99)
    resp = client.post(f"/api/v1/experiments/{exp_id}/cancel", json=cancel.model_dump(mode="json"))
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"]["code"] == "version_conflict"


# ── 14. Cancel already terminal ────────────────────────────────────────────────


def test_cancel_already_terminal_returns_409(client: TestClient) -> None:
    body = _make_submit_body(title="already-done")
    submit_resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    exp_id = submit_resp.json()["data"]["experiment_id"]

    # Place experiment in a terminal state
    _set_state(client._test_db_path, exp_id, "succeeded", 6)  # type: ignore[attr-defined]

    cancel = ExperimentCancelRequest(reason="too late", expected_version=6)
    resp = client.post(f"/api/v1/experiments/{exp_id}/cancel", json=cancel.model_dump(mode="json"))
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"]["code"] == "already_terminal"


# ── 15. GET nonexistent experiment ─────────────────────────────────────────────


def test_get_nonexistent_experiment_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/experiments/nonexistent-id")
    assert resp.status_code == 404
    error_body = resp.json()
    assert error_body["detail"]["error"]["code"] == "not_found"


# ── 16. GET /api/v1/status ─────────────────────────────────────────────────────


def test_status_returns_counts(client: TestClient) -> None:
    resp = client.get("/api/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_version"] == "1.0"
    data = body["data"]
    assert data["node_id"] == "autoresearch-5080"
    assert data["state"] == "ready"
    assert isinstance(data["active_tasks"], int)
    assert data["active_tasks"] >= 0
    assert isinstance(data["waiting_approval"], int)
    assert data["waiting_approval"] >= 0


# ── 17. Worker picks up queued experiment ──────────────────────────────────────


def test_worker_picks_up_queued_experiment(client: TestClient) -> None:
    body = _make_submit_body(title="worker-pickup")
    submit_resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    assert submit_resp.json()["data"]["state"] == "queued"
    exp_id = submit_resp.json()["data"]["experiment_id"]

    # The startup worker stopped because the queue was empty; restart it.
    client.app.state.worker.start()

    detail = _wait_for_terminal(client, exp_id)
    assert detail["data"]["state"] == "succeeded"
    assert detail["data"]["result"] is not None
    assert len(detail["data"]["result"]["summary"]) > 0
    assert detail["data"]["result"]["artifact_count"] == 2


# ── 18. Artifact manifest after worker completion ──────────────────────────────


def test_artifact_manifest_after_completion(client: TestClient) -> None:
    body = _make_submit_body(title="artifact-check")
    submit_resp = client.post("/api/v1/experiments", json=body.model_dump(mode="json"))
    exp_id = submit_resp.json()["data"]["experiment_id"]

    client.app.state.worker.start()
    _wait_for_terminal(client, exp_id)

    manifest_resp = client.get(f"/api/v1/experiments/{exp_id}/artifacts")
    assert manifest_resp.status_code == 200
    items = manifest_resp.json()["items"]
    assert len(items) == 2
    filenames = {entry["filename"] for entry in items}
    assert "run.log" in filenames
    assert "run.json" in filenames
    for entry in items:
        assert entry["artifact_id"]
        assert entry["byte_size"] > 0
        assert len(entry["sha256"]) == 64
        assert entry["media_type"] in ("text/plain", "application/json")


# ── 19. State machine validation ───────────────────────────────────────────────


def test_state_machine_validation() -> None:
    # Every declared valid transition returns True
    for current, targets in VALID_TRANSITIONS.items():
        for target in targets:
            assert can_transition(current, target), (
                f"expected can_transition({current!r}, {target!r}) to be True"
            )

        # Targets not declared for this state return False
        declared = targets | {current}
        for other in ALL_STATES:
            if other not in declared:
                assert not can_transition(current, other), (
                    f"expected can_transition({current!r}, {other!r}) to be False"
                )

    # Unrecognised states are not valid sources
    assert not can_transition("nonexistent", "submitted")
    assert not can_transition("submitted", "nonexistent")

    # Terminal state classification
    for state in TERMINAL_STATES:
        assert is_terminal(state), f"{state} should be terminal"
    non_terminal = set(ALL_STATES) - TERMINAL_STATES
    for state in non_terminal:
        assert not is_terminal(state), f"{state} should not be terminal"

    # Unknown states are not terminal
    assert not is_terminal("nonexistent")


# ── 20. FakeExecutor deterministic output ──────────────────────────────────────


def test_fake_executor_produces_deterministic_results(tmp_path: Path) -> None:
    executor = FakeExecutor(base_delay=0.0)
    patch = FIXED_PATCH
    patch_sha256 = sha256(patch.encode()).hexdigest()

    run1 = executor.execute("exp-1", tmp_path / "run1", patch, patch_sha256, 60)

    assert run1.success is True
    assert "val_bpb=1.1000" in run1.summary
    assert len(run1.artifacts) == 2

    filenames = {a[0] for a in run1.artifacts}
    assert filenames == {"run.log", "run.json"}

    for filename, content, media_type in run1.artifacts:
        assert isinstance(filename, str)
        assert isinstance(content, bytes)
        assert len(content) > 0
        assert isinstance(media_type, str)
        if filename == "run.json":
            assert media_type == "application/json"
            assert b"experiment_id" in content
        elif filename == "run.log":
            assert media_type == "text/plain"
            assert b"val_bpb" in content

    # A second execution with identical parameters produces exactly the same bytes
    run2 = executor.execute("exp-1", tmp_path / "run2", patch, patch_sha256, 60)

    artifacts1 = {a[0]: a[1] for a in run1.artifacts}
    artifacts2 = {a[0]: a[1] for a in run2.artifacts}
    for name in ("run.log", "run.json"):
        assert artifacts1[name] == artifacts2[name], f"{name} should be byte-identical across runs"
        assert sha256(artifacts1[name]).hexdigest() == sha256(artifacts2[name]).hexdigest()
