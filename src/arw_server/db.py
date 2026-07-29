"""SQLite persistence for experiments, events, and idempotency."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    source_commit   TEXT NOT NULL,
    title           TEXT NOT NULL,
    plan_id         TEXT NOT NULL,
    candidate_patch_sha256 TEXT NOT NULL,
    candidate_patch TEXT NOT NULL,
    matrix_json     TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'submitted',
    version         INTEGER NOT NULL DEFAULT 1,
    result_summary  TEXT,
    result_artifact_count INTEGER,
    error_code      TEXT,
    error_message   TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id),
    version         INTEGER NOT NULL,
    event_type      TEXT NOT NULL,
    state           TEXT NOT NULL,
    occurred_at     REAL NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS idempotency (
    key             TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    created_at      REAL NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id     TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id),
    filename        TEXT NOT NULL,
    byte_size       INTEGER NOT NULL,
    sha256          TEXT NOT NULL,
    media_type      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_events_experiment
    ON events(experiment_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_experiments_state
    ON experiments(state, created_at);
CREATE INDEX IF NOT EXISTS idx_idempotency_key
    ON idempotency(key);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and foreign keys enforced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(SCHEMA)
    db.commit()
    return db


def now() -> float:
    return time.time()
