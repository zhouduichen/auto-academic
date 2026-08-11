"""Pure, fail-closed decisions for causal carrier transport experiments."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Literal

EPSILON = 1e-12
PRIMARY_OUTCOMES = ("clean_loss_excess_auc_128", "tuning_loss_h512")
EFFECT_KEYS = ("parameter", "state", "interaction")


def _effect_values(effects: dict[str, float]) -> dict[str, float]:
    if set(effects) != set(EFFECT_KEYS):
        raise ValueError("effects must contain parameter, state, and interaction")
    values: dict[str, float] = {}
    for key in EFFECT_KEYS:
        value = effects[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("effects must be finite numbers")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("effects must be finite numbers")
        values[key] = converted
    return values


def carrier_ratio(effects: dict[str, float]) -> float:
    """Return log2 absolute state-to-parameter causal-effect ratio."""
    values = _effect_values(effects)
    return math.log2(
        (abs(values["state"]) + EPSILON) / (abs(values["parameter"]) + EPSILON)
    )


def classify_effect(effects: dict[str, float]) -> str:
    """Classify a factorial effect using the frozen twofold/interaction rule."""
    values = _effect_values(effects)
    parameter = abs(values["parameter"])
    state = abs(values["state"])
    interaction = abs(values["interaction"])
    if interaction >= max(parameter, state):
        return "interaction"
    ratio = carrier_ratio(values)
    if ratio >= 1.0:
        return "state"
    if ratio <= -1.0:
        return "parameter"
    return "mixed"


def _summary_classes(summary: dict[str, object]) -> tuple[dict[str, str], str]:
    if summary.get("test_loaded") is not False:
        raise ValueError("test isolation failed")
    commit = summary.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("source commit is invalid")
    raw_effects = summary.get("effects")
    if not isinstance(raw_effects, dict) or set(raw_effects) != set(PRIMARY_OUTCOMES):
        raise ValueError("summary effects are incomplete")
    classes: dict[str, str] = {}
    for outcome in PRIMARY_OUTCOMES:
        effects = raw_effects[outcome]
        if not isinstance(effects, dict):
            raise ValueError("summary effects are malformed")
        classes[outcome] = classify_effect(effects)  # type: ignore[arg-type]
    return classes, commit


def evaluate_anchor_pair(
    local: dict[str, object], accumulated: dict[str, object]
) -> dict[str, object]:
    """Evaluate one seed's local and accumulated anchors."""
    local_classes, local_commit = _summary_classes(local)
    accumulated_classes, accumulated_commit = _summary_classes(accumulated)
    if local_commit != accumulated_commit:
        raise ValueError("anchor source commits differ")
    crossing = all(
        local_classes[outcome] == "state"
        and accumulated_classes[outcome] == "parameter"
        for outcome in PRIMARY_OUTCOMES
    )
    reverse = all(
        local_classes[outcome] == "parameter"
        and accumulated_classes[outcome] == "state"
        for outcome in PRIMARY_OUTCOMES
    )
    return {
        "local": local_classes,
        "accumulated": accumulated_classes,
        "crossing": crossing,
        "reverse": reverse,
        "source_commit": local_commit,
        "test_loaded": False,
    }


def evaluate_stage(
    records: dict[str, dict[str, object]],
    required_seeds: tuple[int, ...],
    stage: Literal["A", "B", "C"],
) -> dict[str, object]:
    """Apply the frozen replicated crossing rule to one stage."""
    if stage not in {"A", "B", "C"}:
        raise ValueError("stage must be A, B, or C")
    expected = {str(seed) for seed in required_seeds}
    if set(records) != expected or len(set(required_seeds)) != len(required_seeds):
        raise ValueError("stage seed set is incomplete or duplicated")
    per_seed: dict[str, dict[str, object]] = {}
    commits: set[str] = set()
    for seed in required_seeds:
        record = records[str(seed)]
        if not isinstance(record, dict) or set(record) != {"local", "accumulated"}:
            raise ValueError("stage anchor record is malformed")
        local = record["local"]
        accumulated = record["accumulated"]
        if not isinstance(local, dict) or not isinstance(accumulated, dict):
            raise ValueError("stage anchor summaries are malformed")
        evaluated = evaluate_anchor_pair(local, accumulated)
        commits.add(str(evaluated["source_commit"]))
        per_seed[str(seed)] = evaluated
    if len(commits) != 1:
        raise ValueError("stage source commits differ")
    crossing_seeds = [
        seed for seed in required_seeds if per_seed[str(seed)]["crossing"] is True
    ]
    reverse_seeds = [
        seed for seed in required_seeds if per_seed[str(seed)]["reverse"] is True
    ]
    required_crossings = len(required_seeds) if stage == "A" else 2
    checks = {
        "exact_seed_set": True,
        "source_commit_match": True,
        "test_isolated": True,
        "required_crossings": len(crossing_seeds) >= required_crossings,
        "no_opposite_reversal": not reverse_seeds,
    }
    return {
        "schema": "causal-transport-stage-decision/1",
        "stage": stage,
        "required_seeds": list(required_seeds),
        "per_seed": per_seed,
        "crossing_seeds": crossing_seeds,
        "reverse_seeds": reverse_seeds,
        "checks": checks,
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "source_commit": commits.pop(),
        "test_loaded": False,
    }


def write_decision(path: Path, decision: dict[str, object]) -> None:
    """Atomically write a decision without crossing contract identities."""
    contract = decision.get("contract_sha256")
    if not isinstance(contract, str) or len(contract) != 64:
        raise ValueError("decision contract_sha256 is invalid")
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing decision is unreadable") from error
        if not isinstance(existing, dict) or existing.get("contract_sha256") != contract:
            raise RuntimeError("existing decision belongs to a different contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
