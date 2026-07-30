#!/usr/bin/env python3
"""Submit ReliablePEFT Phase 1 experiments from Mac to Windows via the ARW API.

Usage:
  # Smoke test: single config × single seed × 1 epoch
  uv run python scripts/submit_phase1.py --smoke

  # Full batch: all 16 configs × 3 seeds (48 experiments)
  uv run python scripts/submit_phase1.py --all

  # Filtered: specific configs and seeds
  uv run python scripts/submit_phase1.py --config-ids 0,1,2 --seeds 0

  # Dry run (no submission)
  uv run python scripts/submit_phase1.py --all --dry-run

Environment variables:
  ARW_SERVER    API server URL (default: https://autoresearch-5080.tail2530b8.ts.net:8443)
  ARW_API_TOKEN API token (required)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

SERVER = os.environ.get(
    "ARW_SERVER", "https://autoresearch-5080.tail2530b8.ts.net:8443"
)
TOKEN = os.environ.get("ARW_API_TOKEN", "")

# Match Stage C Windows server HEAD
SOURCE_COMMIT = "7a8bce4000000000000000000000000000000000"  # 40 hex chars

# 16 LoRA configurations (must match phase1_train.py build_16_lora_configs)
CONFIGS: list[dict[str, Any]] = [
    {"config_id": 0, "rank": 2, "lr": 1e-4, "wd": 0.0, "target": "query,value", "desc": "r=2,lr=1e-4,wd=0,qv"},
    {"config_id": 1, "rank": 2, "lr": 3e-4, "wd": 0.0, "target": "query,value", "desc": "r=2,lr=3e-4,wd=0,qv"},
    {"config_id": 2, "rank": 2, "lr": 1e-4, "wd": 0.0, "target": "query,value,output.dense", "desc": "r=2,lr=1e-4,wd=0,qvo"},
    {"config_id": 3, "rank": 2, "lr": 1e-3, "wd": 0.01, "target": "query,value", "desc": "r=2,lr=1e-3,wd=1e-2,qv"},
    {"config_id": 4, "rank": 4, "lr": 1e-4, "wd": 0.0, "target": "query,value", "desc": "r=4,lr=1e-4,wd=0,qv"},
    {"config_id": 5, "rank": 4, "lr": 3e-4, "wd": 0.0, "target": "query,value,output.dense", "desc": "r=4,lr=3e-4,wd=0,qvo"},
    {"config_id": 6, "rank": 4, "lr": 1e-4, "wd": 0.01, "target": "query,value", "desc": "r=4,lr=1e-4,wd=1e-2,qv"},
    {"config_id": 7, "rank": 4, "lr": 3e-4, "wd": 0.01, "target": "query,value,output.dense", "desc": "r=4,lr=3e-4,wd=1e-2,qvo"},
    {"config_id": 8, "rank": 8, "lr": 1e-4, "wd": 0.0, "target": "query,value", "desc": "r=8,lr=1e-4,wd=0,qv"},
    {"config_id": 9, "rank": 8, "lr": 3e-4, "wd": 0.0, "target": "query,value,output.dense", "desc": "r=8,lr=3e-4,wd=0,qvo"},
    {"config_id": 10, "rank": 8, "lr": 1e-3, "wd": 0.0, "target": "query,value", "desc": "r=8,lr=1e-3,wd=0,qv"},
    {"config_id": 11, "rank": 8, "lr": 1e-4, "wd": 0.01, "target": "query,value,output.dense", "desc": "r=8,lr=1e-4,wd=1e-2,qvo"},
    {"config_id": 12, "rank": 16, "lr": 1e-4, "wd": 0.0, "target": "query,value", "desc": "r=16,lr=1e-4,wd=0,qv"},
    {"config_id": 13, "rank": 16, "lr": 3e-4, "wd": 0.0, "target": "query,value,output.dense", "desc": "r=16,lr=3e-4,wd=0,qvo"},
    {"config_id": 14, "rank": 16, "lr": 1e-3, "wd": 0.01, "target": "query,value", "desc": "r=16,lr=1e-3,wd=1e-2,qv"},
    {"config_id": 15, "rank": 16, "lr": 1e-4, "wd": 0.01, "target": "query,value,output.dense", "desc": "r=16,lr=1e-4,wd=1e-2,qvo"},
]


def _make_dummy_patch(config_id: int, seed: int) -> tuple[str, str]:
    """Create a minimal valid unified diff to satisfy the CandidatePatch schema.
    The ReliablePEFT executor ignores this patch entirely.
    """
    patch = (
        f"--- a/train.py\n"
        f"+++ b/train.py\n"
        f"@@ -1,1 +1,1 @@\n"
        f"- # AutoResearch placeholder\n"
        f"+ # ReliablePEFT config_{config_id:02d} seed={seed}\n"
    )
    sha = hashlib.sha256(patch.encode()).hexdigest()
    return patch, sha


def submit_one(
    client: httpx.Client,
    config: dict[str, Any],
    seed: int,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    time_budget: int = 900,
    dry_run: bool = False,
) -> str | None:
    """Submit a single (config, seed) experiment. Returns experiment_id or None."""
    cid = config["config_id"]
    desc = config["desc"]
    title = f"[Phase1] config_{cid:02d}/seed_{seed} — {desc}"
    plan_id = f"phase1-c{cid:02d}-s{seed}"

    dummy_patch, patch_sha256 = _make_dummy_patch(cid, seed)

    body = {
        "project_id": "reliablepeft-phase1",
        "source_commit": SOURCE_COMMIT,
        "title": title,
        "plan_id": plan_id,
        "candidate": {
            "patch_sha256": patch_sha256,
            "patch": dummy_patch,
        },
        "matrix": {
            "seeds": [seed],
            "time_budget_seconds": time_budget,
            "max_parallel": 1,
            "config_id": cid,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    }

    if dry_run:
        print(f"  [DRY] config_{cid:02d}/seed_{seed} — {desc}")
        return None

    resp = client.post("/api/v1/experiments", json=body)
    if resp.status_code == 201:
        data = resp.json()
        exp_id = data.get("data", {}).get("experiment_id", "?")
        return exp_id
    else:
        detail = resp.text[:300]
        print(f"  [FAIL] config_{cid:02d}/seed_{seed}: HTTP {resp.status_code} — {detail}")
        return None


def main() -> None:
    if not TOKEN:
        print("ERROR: ARW_API_TOKEN environment variable is required", file=sys.stderr)
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv

    # Determine mode
    if "--smoke" in sys.argv:
        config_ids = [0]
        seeds = [0]
        epochs = 1
        batch_size = 32
        time_budget = 600  # 10 min for 1 epoch
        print("🔥 SMOKE TEST MODE: config_00 × seed=0 × epochs=1")
    elif "--all" in sys.argv:
        config_ids = list(range(16))
        seeds = [0, 1, 2]
        epochs = 10
        batch_size = 32
        time_budget = 900  # 15 min per run
        print("🚀 FULL BATCH: 16 configs × 3 seeds = 48 runs × 10 epochs")
    else:
        config_ids = list(range(16))
        seeds = [0, 1, 2]
        epochs = 10
        batch_size = 32
        time_budget = 900
        # Parse optional filters
        for arg in sys.argv:
            if arg.startswith("--config-ids="):
                config_ids = [int(x) for x in arg.split("=")[1].split(",")]
            if arg.startswith("--seeds="):
                seeds = [int(x) for x in arg.split("=")[1].split(",")]
            if arg.startswith("--epochs="):
                epochs = int(arg.split("=")[1])
            if arg.startswith("--time-budget="):
                time_budget = int(arg.split("=")[1])

    total = len(config_ids) * len(seeds)
    print(f"Server: {SERVER}")
    print(f"Configs: {config_ids} | Seeds: {seeds} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Total: {total} experiments | Time budget: {time_budget}s each")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    if dry_run:
        for c in CONFIGS:
            if c["config_id"] not in config_ids:
                continue
            for s in seeds:
                submit_one(None, c, s, epochs=epochs, dry_run=True)  # type: ignore[arg-type]
        print(f"\nWould submit {total} experiments.")
        return

    client = httpx.Client(
        base_url=SERVER,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
        },
        timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
        verify=False,  # Tailscale TLS with self-signed cert
    )

    submitted: list[str] = []
    failed = 0

    try:
        for c in CONFIGS:
            if c["config_id"] not in config_ids:
                continue
            for s in seeds:
                exp_id = submit_one(
                    client, c, s,
                    epochs=epochs, batch_size=batch_size, time_budget=time_budget,
                )
                if exp_id:
                    submitted.append(exp_id)
                    idx = len(submitted) + failed
                    print(f"  [{idx}/{total}] config_{c['config_id']:02d}/seed_{s} → {exp_id}")
                else:
                    failed += 1
                # Rate limit
                time.sleep(0.3)
    finally:
        client.close()

    print("=" * 60)
    print(f"Done: {len(submitted)} submitted, {failed} failed, {total} total")

    if submitted:
        print(f"\nExperiment IDs:")
        for eid in submitted:
            print(f"  {eid}")
        print(f"\nMonitor with:")
        print(f"  curl -sk -H 'Authorization: Bearer $ARW_API_TOKEN' \\")
        print(f"    '{SERVER}/api/v1/experiments/{submitted[0]}' | python3 -m json.tool")


if __name__ == "__main__":
    main()
