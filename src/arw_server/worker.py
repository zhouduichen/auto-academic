"""Background worker — dequeues experiments and runs them."""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from pathlib import Path

from arw_server.db import now
from arw_server.executor import ExecutionResult, Executor, FakeExecutor
from arw_server.state import assert_transition, is_terminal

logger = logging.getLogger("arw_server.worker")

POLL_INTERVAL = 1.0
MAX_RUNNING = 1


class Worker:
    """Single-thread bounded worker that dequeues one experiment at a time."""

    def __init__(
        self,
        db: sqlite3.Connection,
        worktree_root: Path,
        *,
        executor: Executor | None = None,
        executors: dict[str, Executor] | None = None,
    ) -> None:
        self._db = db
        self._worktree_root = worktree_root
        self._executor = executor or FakeExecutor()
        self._executors = executors or {}
        self._running: bool = False
        self._current: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def current_experiment_id(self) -> str | None:
        return self._current

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="arw-worker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        import time as _time

        while self._running:
            job = self._dequeue()
            if job is None:
                _time.sleep(POLL_INTERVAL)
                continue
            self._current = job
            try:
                self._execute(job)
            except Exception:
                logger.exception("worker failed experiment %s", job)
                self._fail(job, "worker_internal_error", "worker crashed during execution")
            finally:
                self._current = None

    def _dequeue(self) -> str | None:
        row = self._db.execute(
            "SELECT experiment_id FROM experiments WHERE state = 'queued' "
            "ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def _execute(self, experiment_id: str) -> None:
        row = self._db.execute(
            "SELECT candidate_patch, candidate_patch_sha256, matrix_json, project_id "
            "FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return
        patch, patch_sha256, matrix_json, project_id = row
        import json

        matrix = json.loads(matrix_json)
        time_budget = matrix.get("time_budget_seconds", 300)
        self._transition(experiment_id, "running")
        worktree = self._worktree_root / experiment_id
        result: ExecutionResult
        try:
            executor = self._executors.get(project_id, self._executor)
            # ReliablePEFT experiments: pass config_id + seed from matrix
            if project_id == "reliablepeft-phase1":
                config_id = matrix.get("config_id")
                seeds_list = matrix.get("seeds", [0])
                seed = seeds_list[0] if seeds_list else 0
                epochs = matrix.get("epochs", 10)
                batch_size = matrix.get("batch_size", 32)
                result = executor.execute(
                    experiment_id, worktree, patch, patch_sha256, time_budget,
                    config_id=config_id, seed=seed,
                    epochs=epochs, batch_size=batch_size,
                )
            else:
                result = executor.execute(
                    experiment_id, worktree, patch, patch_sha256, time_budget
                )
        except Exception as exc:
            self._fail(experiment_id, "executor_error", str(exc))
            return
        target = "succeeded" if result.success else "failed"
        self._transition(experiment_id, target)
        if result.success:
            self._db.execute(
                "UPDATE experiments SET result_summary = ?, result_artifact_count = ?, "
                "updated_at = ? WHERE experiment_id = ?",
                (result.summary, len(result.artifacts), now(), experiment_id),
            )
        else:
            self._db.execute(
                "UPDATE experiments SET error_code = 'execution_failed', "
                "error_message = ?, updated_at = ? WHERE experiment_id = ?",
                (result.summary, now(), experiment_id),
            )
        for filename, content, media_type in result.artifacts:
            import hashlib

            artifact_id = uuid.uuid4().hex
            sha256 = hashlib.sha256(content).hexdigest()
            self._db.execute(
                "INSERT INTO artifacts (artifact_id, experiment_id, filename, "
                "byte_size, sha256, media_type) VALUES (?, ?, ?, ?, ?, ?)",
                (artifact_id, experiment_id, filename, len(content), sha256, media_type),
            )
        self._db.commit()

    def _transition(self, experiment_id: str, target: str) -> None:
        row = self._db.execute(
            "SELECT state, version FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return
        current, version = row
        assert_transition(current, target)
        new_version = version + 1
        self._db.execute(
            "UPDATE experiments SET state = ?, version = ?, updated_at = ? WHERE experiment_id = ?",
            (target, new_version, now(), experiment_id),
        )
        self._db.execute(
            "INSERT INTO events (event_id, experiment_id, version, event_type, "
            "state, occurred_at) VALUES (?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, experiment_id, new_version, target, target, now()),
        )
        self._db.commit()

    def _fail(self, experiment_id: str, code: str, message: str) -> None:
        row = self._db.execute(
            "SELECT state FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None or is_terminal(row[0]):
            return
        self._transition(experiment_id, "failed")
        self._db.execute(
            "UPDATE experiments SET error_code = ?, error_message = ?, "
            "updated_at = ? WHERE experiment_id = ?",
            (code, message, now(), experiment_id),
        )
        self._db.commit()
