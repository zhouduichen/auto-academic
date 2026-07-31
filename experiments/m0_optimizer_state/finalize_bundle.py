"""Validate and finalize one M0.6 evidence bundle."""

from __future__ import annotations

import argparse
import json
import math
import os
from hashlib import sha256
from pathlib import Path

from experiments.m0_optimizer_state.m0_run import HORIZONS, _stable_json


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "filename": path.name,
        "byte_size": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _verify_existing_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "sha256_manifest.json"
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, list):
        raise ValueError("SHA-256 manifest must be a list")
    for item in manifest:
        if not isinstance(item, dict):
            raise ValueError("SHA-256 manifest item is malformed")
        filename = item.get("filename")
        if not isinstance(filename, str) or filename == "sha256_manifest.json":
            raise ValueError("SHA-256 manifest filename is invalid")
        path = output_dir / filename
        if not path.is_file() or _file_record(path) != item:
            raise ValueError(f"artifact hash mismatch: {filename}")


def _validate_trajectory(output_dir: Path, config: dict[str, object]) -> None:
    rows = [
        json.loads(line)
        for line in (output_dir / "trajectory_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    branches = (
        ("control", "m_only", "v_only", "state_both", "parameter_only")
        if config.get("state_attribution") is True
        else ("control", "parameter_only", "state_only", "full")
    )
    replay_steps = config.get("replay_steps")
    if not isinstance(replay_steps, int):
        raise ValueError("replay_steps is missing")
    horizons = tuple(horizon for horizon in HORIZONS if horizon <= replay_steps)
    observed = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("trajectory row is malformed")
        branch = row.get("branch")
        horizon = row.get("horizon")
        observed.append((branch, horizon))
        for key, value in row.items():
            if key not in {"branch", "horizon"} and (
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
            ):
                raise ValueError(f"trajectory value is non-finite: {key}")
    expected = [(branch, horizon) for branch in branches for horizon in horizons]
    if observed != expected:
        raise ValueError("trajectory branch/horizon matrix is incomplete")


def finalize_bundle(output_dir: Path) -> None:
    """Validate a completed bundle and atomically rewrite its file manifest."""

    required = {
        "run_config.json",
        "split_ids.json",
        "replay_manifest.json",
        "checkpoint_manifest.json",
        "provenance.json",
        "trajectory_metrics.jsonl",
        "summary.json",
        "stdout.log",
        "sha256_manifest.json",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise ValueError(f"bundle is missing required artifacts: {missing}")

    _verify_existing_manifest(output_dir)
    config = _load_json(output_dir / "run_config.json")
    summary = _load_json(output_dir / "summary.json")
    split_ids = _load_json(output_dir / "split_ids.json")
    provenance = _load_json(output_dir / "provenance.json")
    if not all(isinstance(item, dict) for item in (config, summary, split_ids, provenance)):
        raise ValueError("bundle metadata is malformed")
    if (
        config.get("state_attribution") is True
        and not (output_dir / "branch_construction_manifest.json").is_file()
    ):
        raise ValueError("state-attribution bundle lacks branch construction manifest")
    if summary.get("status") != "succeeded" or summary.get("test_loaded") is not False:
        raise ValueError("bundle summary is not a successful test-isolated run")
    if split_ids.get("test_loaded") is not False:
        raise ValueError("split manifest does not prove test isolation")
    if summary.get("source_commit") != provenance.get("source_commit"):
        raise ValueError("summary/provenance source commits do not match")
    if (output_dir / "stdout.log").stat().st_size == 0:
        raise ValueError("stdout.log is empty")
    _validate_trajectory(output_dir, config)

    manifest = [
        _file_record(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"sha256_manifest.json", ".sha256_manifest.json.tmp"}
    ]
    temporary_path = output_dir / ".sha256_manifest.json.tmp"
    temporary_path.write_text(_stable_json(manifest), encoding="utf-8")
    os.replace(temporary_path, output_dir / "sha256_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    finalize_bundle(args.output_dir)


if __name__ == "__main__":
    main()
