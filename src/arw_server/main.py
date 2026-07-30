"""ARW Server — FastAPI application for the Windows Coordinator node."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import HttpUrl

from arw.models import (
    ArtifactEntry,
    ArtifactManifestResponse,
    ExperimentCancelRequest,
    ExperimentDetail,
    ExperimentEvent,
    ExperimentEventListResponse,
    ExperimentListResponse,
    ExperimentResponse,
    ExperimentResult,
    ExperimentSubmitRequest,
    ExperimentSummary,
    HealthResponse,
    MetaResponse,
    ProjectListResponse,
    ProjectSummary,
    SafeError,
    StatusData,
    StatusResponse,
)
from arw_server.auth import (
    ScopedToken,  # used in route annotations
    require_scope,
)
from arw_server.db import connect as db_connect
from arw_server.db import now
from arw_server.executor import FakeExecutor, ReliablePEFTExecutor
from arw_server.state import can_transition, is_terminal
from arw_server.worker import Worker

SERVER_VERSION = "0.1.0"
MINIMUM_CLIENT_VERSION = "0.1.0"
FEATURES: set[str] = {
    "experiments.submit",
    "experiments.cancel",
    "experiments.read",
    "projects.read",
    "tasks.read",
    "artifacts.read",
}

KARPATHY_PROJECT = ProjectSummary(
    project_id="karpathy-autoresearch",
    name="Karpathy AutoResearch",
    source_url=HttpUrl("https://github.com/karpathy/autoresearch"),
    source_commit="228791fb499afffb54b46200aca536f79142f117",
    execution_platform="windows_cuda",
)

RELIABLEPEFT_PROJECT = ProjectSummary(
    project_id="reliablepeft-phase1",
    name="ReliablePEFT Phase 1: EuroSAT LoRA",
    source_url=HttpUrl("https://github.com/zhouduichen/auto-academic"),
    source_commit="847dcced0a55bf9fca29f663b0a9c9a3e9d115a8",
    execution_platform="windows_cuda",
)

FIXED_PROJECTS = (KARPATHY_PROJECT, RELIABLEPEFT_PROJECT)

RELIABLEPEFT_SCRIPT_DIR = Path("experiments/phase1_eurosat")


def _request_id(request: Request) -> str:
    rid = request.headers.get("X-Request-ID", "")
    return rid if rid else uuid.uuid4().hex[:12]


def _error(code: str, message: str, status: int, request_id: str = "") -> HTTPException:
    from arw.models import ErrorData, ErrorResponse

    rid = request_id or uuid.uuid4().hex[:12]
    return HTTPException(
        status_code=status,
        detail=ErrorResponse(
            api_version="1.0",
            request_id=rid,
            error=ErrorData(code=code, message=message),
        ).model_dump(mode="json"),
    )


def _build_detail(row: sqlite3.Row) -> ExperimentDetail:
    matrix = json.loads(row["matrix_json"])
    result: ExperimentResult | None = None
    error: SafeError | None = None
    if row["result_summary"] is not None:
        result = ExperimentResult(
            summary=row["result_summary"],
            artifact_count=row["result_artifact_count"] or 0,
        )
    if row["error_code"] is not None:
        error = SafeError(code=row["error_code"], message=row["error_message"] or "")
    return ExperimentDetail(
        experiment_id=row["experiment_id"],
        project_id=row["project_id"],
        source_commit=row["source_commit"],
        title=row["title"],
        state=row["state"],
        version=row["version"],
        plan_id=row["plan_id"],
        candidate_patch_sha256=row["candidate_patch_sha256"],
        matrix=matrix,
        result=result,
        error=error,
    )


def create_app(
    db_path: Path | None = None,
    *,
    api_token: str | None = None,
    worktree_root: Path | None = None,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        worker.start()
        yield
        worker.stop()

    app = FastAPI(
        title="AutoResearch Workbench",
        version=SERVER_VERSION,
        docs_url=None,
        lifespan=_lifespan,
    )
    db = db_connect(db_path or Path("arw_server.db"))
    db.row_factory = sqlite3.Row
    token = api_token or os.environ.get("ARW_API_TOKEN", "")
    if not token:
        raise RuntimeError("ARW_API_TOKEN is required to start the server")
    token_bytes = token.encode()
    app.state.token_verifier = lambda supplied, _ctx: supplied.encode() == token_bytes
    app.state.token_scopes = frozenset(FEATURES)

    worker = Worker(
        db,
        worktree_root or Path("worktrees"),
        executors={
            "reliablepeft-phase1": ReliablePEFTExecutor(RELIABLEPEFT_SCRIPT_DIR),
        },
        executor=FakeExecutor(),
    )
    app.state.worker = worker

    # ── route helpers (closed over `db`) ──────────────────────────────────

    def _get_experiment(experiment_id: str) -> ExperimentResponse:
        row = db.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise _error("not_found", f"experiment {experiment_id} not found", 404)
        return ExperimentResponse(
            api_version="1.0", request_id=uuid.uuid4().hex[:12], data=_build_detail(row)
        )

    def _check_project(project_id: str) -> None:
        valid = {p.project_id for p in FIXED_PROJECTS}
        if project_id not in valid:
            raise _error("unknown_project", f"project {project_id} is not supported", 400)

    def _auto_validate(experiment_id: str) -> None:
        for target in ("policy-validating", "waiting-approval", "queued"):
            row = db.execute(
                "SELECT state, version FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                return
            current, version = row["state"], row["version"]
            if not can_transition(current, target):
                return
            new_version = version + 1
            ts = now()
            db.execute(
                "UPDATE experiments SET state = ?, version = ?, updated_at = ? "
                "WHERE experiment_id = ?",
                (target, new_version, ts, experiment_id),
            )
            db.execute(
                "INSERT INTO events (event_id, experiment_id, version, event_type, "
                "state, occurred_at) VALUES (?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, experiment_id, new_version, target, target, ts),
            )
            db.commit()

    def _record_event(experiment_id: str, event_type: str, state: str, version: int) -> None:
        db.execute(
            "INSERT INTO events (event_id, experiment_id, version, event_type, "
            "state, occurred_at) VALUES (?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, experiment_id, version, event_type, state, now()),
        )
        db.commit()

    # ── /healthz ─────────────────────────────────────────────────────────

    @app.get("/healthz")
    async def health() -> HealthResponse:
        return HealthResponse(api_version="1.0", request_id=uuid.uuid4().hex[:12], status="ok")

    # ── /api/v1/meta ─────────────────────────────────────────────────────

    @app.get("/api/v1/meta")
    async def meta(
        _token: ScopedToken = Depends(require_scope("experiments.read")),
    ) -> MetaResponse:
        return MetaResponse(
            api_version="1.0",
            request_id=uuid.uuid4().hex[:12],
            server_version=SERVER_VERSION,
            minimum_client_version=MINIMUM_CLIENT_VERSION,
            features=FEATURES,
        )

    # ── /api/v1/status ───────────────────────────────────────────────────

    @app.get("/api/v1/status")
    async def status(
        _token: ScopedToken = Depends(require_scope("experiments.read")),
        request: Request = None,  # type: ignore[assignment]
    ) -> StatusResponse:
        active = db.execute("SELECT COUNT(*) FROM experiments WHERE state = 'running'").fetchone()[
            0
        ]
        waiting = db.execute(
            "SELECT COUNT(*) FROM experiments WHERE state = 'waiting-approval'"
        ).fetchone()[0]
        return StatusResponse(
            api_version="1.0",
            request_id=_request_id(request),
            data=StatusData(
                node_id="autoresearch-5080",
                state="ready",
                active_tasks=active,
                waiting_approval=waiting,
            ),
        )

    # ── /api/v1/projects ─────────────────────────────────────────────────

    @app.get("/api/v1/projects")
    async def list_projects(
        _token: ScopedToken = Depends(require_scope("projects.read")),
        request: Request = None,  # type: ignore[assignment]
    ) -> ProjectListResponse:
        return ProjectListResponse(
            api_version="1.0",
            request_id=_request_id(request),
            items=list(FIXED_PROJECTS),
            next_cursor=None,
        )

    # ── /api/v1/tasks ────────────────────────────────────────────────────

    @app.get("/api/v1/tasks")
    async def list_tasks(
        _token: ScopedToken = Depends(require_scope("tasks.read")),
        request: Request = None,  # type: ignore[assignment]
        limit: int = 100,
        cursor: str | None = None,
    ) -> JSONResponse:
        rows = db.execute(
            "SELECT experiment_id, project_id, title, state, version "
            "FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit + 1,),
        ).fetchall()
        items: list[dict[str, object]] = []
        for i, row in enumerate(rows):
            if i >= limit:
                break
            items.append(
                {
                    "task_id": row["experiment_id"],
                    "project_id": row["project_id"],
                    "title": row["title"],
                    "status": row["state"],
                    "version": row["version"],
                }
            )
        next_cursor_val: str | None = rows[limit]["experiment_id"] if len(rows) > limit else None
        return JSONResponse(
            {
                "api_version": "1.0",
                "request_id": _request_id(request),
                "items": items,
                "next_cursor": next_cursor_val,
            }
        )

    # ── GET /api/v1/experiments ──────────────────────────────────────────

    @app.get("/api/v1/experiments")
    async def list_experiments(
        _token: ScopedToken = Depends(require_scope("experiments.read")),
        request: Request = None,  # type: ignore[assignment]
        limit: int = 100,
        cursor: str | None = None,
    ) -> ExperimentListResponse:
        rows = db.execute(
            "SELECT experiment_id, project_id, source_commit, title, state, version "
            "FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit + 1,),
        ).fetchall()
        summaries: list[ExperimentSummary] = []
        for i, row in enumerate(rows):
            if i >= limit:
                break
            summaries.append(
                ExperimentSummary(
                    experiment_id=row["experiment_id"],
                    project_id=row["project_id"],
                    source_commit=row["source_commit"],
                    title=row["title"],
                    state=row["state"],
                    version=row["version"],
                )
            )
        next_cursor_val: str | None = rows[limit]["experiment_id"] if len(rows) > limit else None
        return ExperimentListResponse(
            api_version="1.0",
            request_id=_request_id(request),
            items=summaries,
            next_cursor=next_cursor_val,
        )

    # ── POST /api/v1/experiments ─────────────────────────────────────────

    @app.post("/api/v1/experiments", status_code=201)
    async def submit_experiment(
        body: ExperimentSubmitRequest,
        request: Request,
        _token: ScopedToken = Depends(require_scope("experiments.submit")),
    ) -> ExperimentResponse:
        _check_project(body.project_id)
        idempotency_key = request.headers.get("Idempotency-Key", "")
        if idempotency_key:
            existing = db.execute(
                "SELECT experiment_id FROM idempotency WHERE key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return _get_experiment(existing["experiment_id"])
        experiment_id = uuid.uuid4().hex[:16]
        ts = now()
        db.execute(
            "INSERT INTO experiments (experiment_id, project_id, source_commit, "
            "title, plan_id, candidate_patch_sha256, candidate_patch, matrix_json, "
            "state, version, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'submitted', 1, ?, ?)",
            (
                experiment_id,
                body.project_id,
                body.source_commit,
                body.title,
                body.plan_id,
                body.candidate.patch_sha256,
                body.candidate.patch,
                json.dumps(body.matrix.model_dump(mode="json")),
                ts,
                ts,
            ),
        )
        _record_event(experiment_id, "submitted", "submitted", 1)
        if idempotency_key:
            db.execute(
                "INSERT INTO idempotency (key, experiment_id, created_at) VALUES (?, ?, ?)",
                (idempotency_key, experiment_id, ts),
            )
        db.commit()
        _auto_validate(experiment_id)
        return _get_experiment(experiment_id)

    # ── GET /api/v1/experiments/{experiment_id} ──────────────────────────

    @app.get("/api/v1/experiments/{experiment_id}")
    async def get_experiment(
        experiment_id: str,
        request: Request,
        _token: ScopedToken = Depends(require_scope("experiments.read")),
    ) -> ExperimentResponse:
        resp = _get_experiment(experiment_id)
        return ExperimentResponse(
            api_version="1.0",
            request_id=_request_id(request),
            data=resp.data,
        )

    # ── GET /api/v1/experiments/{experiment_id}/events ───────────────────

    @app.get("/api/v1/experiments/{experiment_id}/events")
    async def list_experiment_events(
        experiment_id: str,
        request: Request,
        _token: ScopedToken = Depends(require_scope("experiments.read")),
        limit: int = 100,
        cursor: str | None = None,
    ) -> ExperimentEventListResponse:
        rows = db.execute(
            "SELECT event_id, experiment_id, version, event_type, state, occurred_at "
            "FROM events WHERE experiment_id = ? ORDER BY occurred_at LIMIT ?",
            (experiment_id, limit + 1),
        ).fetchall()
        if not rows:
            raise _error(
                "not_found",
                f"experiment {experiment_id} not found",
                404,
                _request_id(request),
            )
        events: list[ExperimentEvent] = []
        for i, row in enumerate(rows):
            if i >= limit:
                break
            events.append(
                ExperimentEvent(
                    event_id=row["event_id"],
                    experiment_id=row["experiment_id"],
                    version=row["version"],
                    event_type=row["event_type"],
                    occurred_at=str(row["occurred_at"]),
                    state=row["state"],
                )
            )
        next_cursor_val: str | None = rows[limit]["event_id"] if len(rows) > limit else None
        return ExperimentEventListResponse(
            api_version="1.0",
            request_id=_request_id(request),
            items=events,
            next_cursor=next_cursor_val,
        )

    # ── POST /api/v1/experiments/{experiment_id}/cancel ──────────────────

    @app.post("/api/v1/experiments/{experiment_id}/cancel")
    async def cancel_experiment(
        experiment_id: str,
        body: ExperimentCancelRequest,
        request: Request,
        _token: ScopedToken = Depends(require_scope("experiments.cancel")),
    ) -> ExperimentResponse:
        row = db.execute(
            "SELECT state, version FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise _error(
                "not_found",
                f"experiment {experiment_id} not found",
                404,
                _request_id(request),
            )
        state, version = row["state"], row["version"]
        if body.expected_version != version:
            raise _error(
                "version_conflict",
                f"expected version {body.expected_version}, actual {version}",
                409,
                _request_id(request),
            )
        if is_terminal(state):
            raise _error(
                "already_terminal",
                f"experiment is already in terminal state: {state}",
                409,
                _request_id(request),
            )
        if not can_transition(state, "cancelled"):
            raise _error(
                "invalid_transition",
                f"cannot cancel from state: {state}",
                409,
                _request_id(request),
            )
        new_version = version + 1
        db.execute(
            "UPDATE experiments SET state = 'cancelled', version = ?, updated_at = ? "
            "WHERE experiment_id = ?",
            (new_version, now(), experiment_id),
        )
        _record_event(experiment_id, "cancelled", "cancelled", new_version)
        return _get_experiment(experiment_id)

    # ── GET …/artifacts ──────────────────────────────────────────────────

    @app.get("/api/v1/experiments/{experiment_id}/artifacts")
    async def get_artifact_manifest(
        experiment_id: str,
        request: Request,
        _token: ScopedToken = Depends(require_scope("artifacts.read")),
    ) -> ArtifactManifestResponse:
        row = db.execute(
            "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise _error(
                "not_found",
                f"experiment {experiment_id} not found",
                404,
                _request_id(request),
            )
        rows = db.execute(
            "SELECT artifact_id, filename, byte_size, sha256, media_type "
            "FROM artifacts WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        entries = [
            ArtifactEntry(
                artifact_id=r["artifact_id"],
                filename=r["filename"],
                byte_size=r["byte_size"],
                sha256=r["sha256"],
                media_type=r["media_type"],
            )
            for r in rows
        ]
        return ArtifactManifestResponse(
            api_version="1.0",
            request_id=_request_id(request),
            items=entries,
        )

    # ── GET …/artifacts/{aid}/content ────────────────────────────────────

    @app.get("/api/v1/experiments/{experiment_id}/artifacts/{artifact_id}/content")
    async def download_artifact(
        experiment_id: str,
        artifact_id: str,
        request: Request,
        _token: ScopedToken = Depends(require_scope("artifacts.read")),
    ) -> Response:
        row = db.execute(
            "SELECT filename, media_type FROM artifacts "
            "WHERE artifact_id = ? AND experiment_id = ?",
            (artifact_id, experiment_id),
        ).fetchone()
        if row is None:
            raise _error(
                "not_found",
                f"artifact {artifact_id} not found",
                404,
                _request_id(request),
            )
        return Response(
            content=f"[artifact {artifact_id} for experiment {experiment_id}]\n",
            status_code=200,
            media_type=row["media_type"],
        )

    return app


def main() -> None:
    token = os.environ.get("ARW_API_TOKEN", "")
    if not token:
        raise RuntimeError("ARW_API_TOKEN environment variable is required")
    app = create_app(
        db_path=Path(os.environ.get("ARW_DB_PATH", "arw_server.db")),
        api_token=token,
        worktree_root=Path(os.environ.get("ARW_WORKTREE_ROOT", "worktrees")),
    )
    uvicorn.run(app, host="0.0.0.0", port=8443, log_level="info")


if __name__ == "__main__":
    main()
