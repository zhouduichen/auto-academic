"""Validate and summarize immutable Gate B causal-transport evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from experiments import causal_transport_gate as gate

SEEDS = (401, 402, 403)
DOSES = (1, 8, 64, 1250)
CONTINUATIONS = ("clean", "noisy")
BRANCHES = ("CC", "CN", "NC", "NN")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid Gate B JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Gate B JSON must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def _validate_contract_and_decision(
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    contract = _read_json(root / "CAUSAL_TRANSPORT_CONTRACT.json")
    decision = _read_json(root / "GATE_B_DECISION.json")
    if contract.get("schema") != "causal-transport-contract/1":
        raise RuntimeError("Gate B contract schema is invalid")
    if contract.get("test_loaded") is not False:
        raise RuntimeError("Gate B contract test isolation failed")
    if _mapping(contract.get("seeds"), "contract seeds").get("B") != list(SEEDS):
        raise RuntimeError("Gate B contract seeds differ")
    if _mapping(contract.get("doses"), "contract doses").get("B") != list(DOSES):
        raise RuntimeError("Gate B contract doses differ")
    if contract.get("continuations") != list(CONTINUATIONS):
        raise RuntimeError("Gate B contract continuations differ")
    if contract.get("primary_outcomes") != list(gate.PRIMARY_OUTCOMES):
        raise RuntimeError("Gate B contract outcomes differ")
    if decision.get("decision") != "NO-GO" or decision.get("bundle_count") != 24:
        raise RuntimeError("Gate B must be the complete frozen NO-GO")
    if decision.get("stage") != "B" or decision.get("test_loaded") is not False:
        raise RuntimeError("Gate B decision stage or test isolation failed")
    checks = _mapping(decision.get("checks"), "Gate B decision checks")
    if checks.get("required_crossings") is not False:
        raise RuntimeError("Gate B decision must preserve failed crossings")
    contract_hash = contract.get("contract_sha256")
    source_commit = contract.get("source_commit")
    if not isinstance(contract_hash, str) or len(contract_hash) != 64:
        raise RuntimeError("Gate B contract hash is invalid")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError("Gate B source commit is invalid")
    if decision.get("contract_sha256") != contract_hash:
        raise RuntimeError("Gate B decision contract mismatch")
    if decision.get("source_commit") != source_commit:
        raise RuntimeError("Gate B decision source commit mismatch")
    return contract, decision


def _validate_summary(
    summary: dict[str, object], *, contract_hash: str, source_commit: str
) -> tuple[int, int, str]:
    if summary.get("status") != "succeeded":
        raise RuntimeError("Gate B summary is not succeeded")
    if summary.get("test_loaded") is not False:
        raise RuntimeError("Gate B summary test isolation failed")
    if summary.get("contract_sha256") != contract_hash:
        raise RuntimeError("Gate B summary contract mismatch")
    if summary.get("source_commit") != source_commit:
        raise RuntimeError("Gate B summary source commit mismatch")
    request = _mapping(summary.get("request"), "Gate B request")
    if request.get("contract_sha256") != contract_hash:
        raise RuntimeError("Gate B request contract mismatch")
    if request.get("source_commit") != source_commit:
        raise RuntimeError("Gate B request source commit mismatch")
    seed = request.get("seed")
    dose = request.get("dose")
    continuation = request.get("continuation")
    if seed not in SEEDS or dose not in DOSES or continuation not in CONTINUATIONS:
        raise RuntimeError("Gate B request key is outside the frozen matrix")

    effects = _mapping(summary.get("effects"), "Gate B effects")
    if set(effects) != set(gate.PRIMARY_OUTCOMES):
        raise RuntimeError("Gate B effects are incomplete")
    for outcome in gate.PRIMARY_OUTCOMES:
        outcome_effects = _mapping(effects[outcome], f"{outcome} effects")
        if set(outcome_effects) != set(gate.EFFECT_KEYS):
            raise RuntimeError(f"{outcome} effects are malformed")
        values = {
            key: _finite(outcome_effects[key], f"{outcome} {key}")
            for key in gate.EFFECT_KEYS
        }
        gate.classify_effect(values)

    endpoints = _mapping(summary.get("endpoints"), "Gate B endpoints")
    if set(endpoints) != set(BRANCHES):
        raise RuntimeError("Gate B endpoints are incomplete")
    for branch in BRANCHES:
        metrics = _mapping(endpoints[branch], f"{branch} endpoints")
        for outcome in gate.PRIMARY_OUTCOMES:
            _finite(metrics.get(outcome), f"{branch} {outcome}")
    return int(seed), int(dose), str(continuation)


def load_gate_b(root: Path) -> dict[tuple[int, int, str], dict[str, object]]:
    """Load one exact, complete, immutable Gate B result matrix."""
    contract, _ = _validate_contract_and_decision(root)
    contract_hash = str(contract["contract_sha256"])
    source_commit = str(contract["source_commit"])
    summaries: dict[tuple[int, int, str], dict[str, object]] = {}
    for path in sorted((root / "gate-b").glob("*/summary.json")):
        summary = _read_json(path)
        key = _validate_summary(
            summary, contract_hash=contract_hash, source_commit=source_commit
        )
        if key in summaries:
            raise RuntimeError("duplicate Gate B bundle")
        summaries[key] = summary
    expected = {
        (seed, dose, continuation)
        for seed in SEEDS
        for dose in DOSES
        for continuation in CONTINUATIONS
    }
    if set(summaries) != expected:
        raise RuntimeError("Gate B bundle matrix is incomplete")
    return summaries


def _outcome_effects(summary: dict[str, object], outcome: str) -> dict[str, float]:
    effects = _mapping(summary["effects"], "Gate B effects")
    raw = _mapping(effects[outcome], f"{outcome} effects")
    return {key: float(raw[key]) for key in gate.EFFECT_KEYS}


def _repair(summary: dict[str, object], outcome: str) -> dict[str, float]:
    endpoints = _mapping(summary["endpoints"], "Gate B endpoints")
    nn = _mapping(endpoints["NN"], "NN endpoints")
    nc = _mapping(endpoints["NC"], "NC endpoints")
    cn = _mapping(endpoints["CN"], "CN endpoints")
    return {
        "state_repair": float(nn[outcome]) - float(nc[outcome]),
        "parameter_repair": float(nn[outcome]) - float(cn[outcome]),
    }


def _stable_handoff(
    classes: dict[tuple[int, int, str, str], str], seeds: tuple[int, ...]
) -> int | None:
    for index, dose in enumerate(DOSES):
        later = DOSES[index:]
        if all(
            classes[(seed, candidate, continuation, outcome)] == "parameter"
            for seed in seeds
            for candidate in later
            for continuation in CONTINUATIONS
            for outcome in gate.PRIMARY_OUTCOMES
        ):
            return dose
    return None


def build_discovery_evidence(root: Path) -> dict[str, object]:
    """Derive discovery-only transport and direct-repair summaries."""
    contract, decision = _validate_contract_and_decision(root)
    summaries = load_gate_b(root)
    classes: dict[tuple[int, int, str, str], str] = {}
    by_bundle: dict[str, object] = {}
    for (seed, dose, continuation), summary in sorted(summaries.items()):
        carrier: dict[str, object] = {}
        repair: dict[str, object] = {}
        for outcome in gate.PRIMARY_OUTCOMES:
            effects = _outcome_effects(summary, outcome)
            classification = gate.classify_effect(effects)
            classes[(seed, dose, continuation, outcome)] = classification
            carrier[outcome] = {
                "class": classification,
                "ratio": gate.carrier_ratio(effects),
                "effects": effects,
            }
            repair[outcome] = _repair(summary, outcome)
        key = f"seed{seed}-dose{dose}-{continuation}"
        by_bundle[key] = {
            "seed": seed,
            "dose": dose,
            "continuation": continuation,
            "carrier": carrier,
            "repair": repair,
        }

    by_dose: dict[str, object] = {}
    for dose in DOSES:
        agreements = [
            classes[(seed, dose, "clean", outcome)]
            == classes[(seed, dose, "noisy", outcome)]
            for seed in SEEDS
            for outcome in gate.PRIMARY_OUTCOMES
        ]
        counts = Counter(
            classes[(seed, dose, continuation, outcome)]
            for seed in SEEDS
            for continuation in CONTINUATIONS
            for outcome in gate.PRIMARY_OUTCOMES
        )
        by_dose[str(dose)] = {
            "continuation_agreement": sum(agreements) / len(agreements),
            "classification_counts": dict(sorted(counts.items())),
        }

    consumed = {
        "CAUSAL_TRANSPORT_CONTRACT.json": _sha256(
            root / "CAUSAL_TRANSPORT_CONTRACT.json"
        ),
        "GATE_B_DECISION.json": _sha256(root / "GATE_B_DECISION.json"),
    }
    for path in sorted((root / "gate-b").glob("*/summary.json")):
        consumed[str(path.relative_to(root))] = _sha256(path)
    return {
        "schema": "attribution-transport-discovery-evidence/1",
        "decision": decision["decision"],
        "bundle_count": len(summaries),
        "source_commit": contract["source_commit"],
        "contract_sha256": contract["contract_sha256"],
        "discovery_only": True,
        "by_bundle": by_bundle,
        "by_dose": by_dose,
        "seed_stable_parameter_handoff": {
            str(seed): _stable_handoff(classes, (seed,)) for seed in SEEDS
        },
        "cohort_stable_parameter_handoff": _stable_handoff(classes, SEEDS),
        "consumed_sha256": dict(sorted(consumed.items())),
        "test_loaded": False,
    }


def write_discovery_evidence(
    root: Path, evidence: dict[str, object]
) -> Path:
    """Atomically write evidence without rebinding an existing source set."""
    path = root / "DISCOVERY_EVIDENCE.json"
    if path.is_file():
        existing = _read_json(path)
        if existing.get("consumed_sha256") != evidence.get("consumed_sha256"):
            raise RuntimeError("existing discovery evidence belongs to a different source")
        if existing != evidence:
            raise RuntimeError("existing discovery evidence content differs")
        return path
    payload = (
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    record = build_discovery_evidence(args.root)
    output = write_discovery_evidence(args.root, record)
    print(json.dumps({"bundles": record["bundle_count"], "output": str(output)}))


if __name__ == "__main__":
    main()
