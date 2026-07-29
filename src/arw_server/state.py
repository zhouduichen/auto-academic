"""Server-side experiment state machine."""

from __future__ import annotations

from typing import Literal

VALID_TRANSITIONS: dict[str, set[str]] = {
    "submitted": {"policy-validating"},
    "policy-validating": {"waiting-approval", "policy-rejected"},
    "waiting-approval": {"queued", "policy-rejected", "cancelled"},
    "queued": {"running", "cancelled"},
    "running": {"succeeded", "failed", "cancelled"},
}

TERMINAL_STATES: set[str] = {"policy-rejected", "failed", "succeeded", "cancelled"}

ALL_STATES = (
    "submitted",
    "policy-validating",
    "waiting-approval",
    "policy-rejected",
    "queued",
    "running",
    "failed",
    "succeeded",
    "cancelled",
)

ExperimentState = Literal[
    "submitted",
    "policy-validating",
    "waiting-approval",
    "policy-rejected",
    "queued",
    "running",
    "failed",
    "succeeded",
    "cancelled",
]


def can_transition(current: str, target: str) -> bool:
    return target in VALID_TRANSITIONS.get(current, set())


def assert_transition(current: str, target: str) -> None:
    if not can_transition(current, target):
        raise ValueError(f"invalid state transition: {current} → {target}")


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES
