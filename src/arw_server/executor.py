"""Experiment executor — fake and real variants."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Protocol

KARPATHY_COMMIT = "228791fb499afffb54b46200aca536f79142f117"
MUTABLE_PATH = "train.py"
PROTECTED_PATHS = ("prepare.py",)
FIXED_COMMAND = ["uv", "run", "python", "train.py"]


class ExecutorError(RuntimeError):
    pass


class ExecutionResult:
    __slots__ = ("artifacts", "success", "summary")

    def __init__(
        self,
        success: bool,
        summary: str,
        artifacts: list[tuple[str, bytes, str]],  # (filename, content, media_type)
    ) -> None:
        self.success = success
        self.summary = summary
        self.artifacts = artifacts


class Executor(Protocol):
    def execute(
        self,
        experiment_id: str,
        worktree: Path,
        patch: str,
        patch_sha256: str,
        time_budget_seconds: int,
    ) -> ExecutionResult: ...


class FakeExecutor:
    """Deterministic fake executor for contract verification — no CUDA needed."""

    def __init__(self, base_delay: float = 0.05) -> None:
        self._base_delay = base_delay

    def execute(
        self,
        experiment_id: str,
        worktree: Path,
        patch: str,
        patch_sha256: str,
        time_budget_seconds: int,
    ) -> ExecutionResult:
        _validate_patch(patch, patch_sha256)
        _validate_mutable_only(patch)
        worktree.mkdir(parents=True, exist_ok=True)
        template = worktree / MUTABLE_PATH
        template.write_text(
            "# AutoResearch training script (fake worktree)\n"
            f"# Original commit: {KARPATHY_COMMIT}\n",
            encoding="utf-8",
        )
        template.chmod(0o644)
        time.sleep(self._base_delay)
        run_log = (
            "[fake executor]\n"
            "val_bpb = 1.1000\n"
            f"training_seconds = {min(time_budget_seconds, 60)}\n"
            "peak_vram_mb = 8192\n"
            "mfu_percent = 45.2\n"
            "num_steps = 100\n"
            "total_tokens_M = 16.0\n"
        ).encode()
        run_sha256 = hashlib.sha256(run_log).hexdigest()
        return ExecutionResult(
            success=True,
            summary=f"val_bpb=1.1000 training_seconds={min(time_budget_seconds, 60)} (fake)",
            artifacts=[
                ("run.log", run_log, "text/plain"),
                (
                    "run.json",
                    (
                        f'{{"experiment_id":"{experiment_id}","val_bpb":1.1,'
                        f'"sha256_run_log":"{run_sha256}"}}\n'
                    ).encode(),
                    "application/json",
                ),
            ],
        )


class RealExecutor:
    """Karpathy worktree runner with OS sandbox — requires CUDA gate to pass."""

    def __init__(self, karpathy_clone: Path, *, cuda_gate: bool = False) -> None:
        if not cuda_gate:
            raise ExecutorError("real executor requires CUDA gate to be enabled")
        if not karpathy_clone.is_dir():
            raise ExecutorError(f"Karpathy clone not found: {karpathy_clone}")
        self._clone = karpathy_clone

    def execute(
        self,
        experiment_id: str,
        worktree: Path,
        patch: str,
        patch_sha256: str,
        time_budget_seconds: int,
    ) -> ExecutionResult:
        _validate_patch(patch, patch_sha256)
        _validate_mutable_only(patch)
        worktree.mkdir(parents=True, exist_ok=True)
        _git_worktree(self._clone, worktree, KARPATHY_COMMIT)
        _apply_patch(worktree, patch)
        env = os.environ.copy()
        env.setdefault("TIME_BUDGET", str(time_budget_seconds))
        try:
            proc = subprocess.run(
                FIXED_COMMAND,
                cwd=str(worktree),
                capture_output=True,
                timeout=time_budget_seconds + 30,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, summary="timeout", artifacts=[])
        stdout = proc.stdout
        stderr = proc.stderr
        combined = stdout + b"\n" + stderr
        sha256 = hashlib.sha256(combined).hexdigest()
        success = proc.returncode == 0
        summary = "ok" if success else f"exit {proc.returncode}"
        return ExecutionResult(
            success=success,
            summary=summary,
            artifacts=[
                ("run.log", combined, "text/plain"),
                (
                    "run.json",
                    (
                        f'{{"experiment_id":"{experiment_id}","success":{str(success).lower()},'
                        f'"sha256":"{sha256}","exit_code":{proc.returncode}}}\n'
                    ).encode(),
                    "application/json",
                ),
            ],
        )


def _validate_patch(patch: str, patch_sha256: str) -> None:
    actual = hashlib.sha256(patch.encode()).hexdigest()
    if actual != patch_sha256:
        raise ExecutorError(f"patch SHA-256 mismatch: expected {patch_sha256}, got {actual}")


def _validate_mutable_only(patch: str) -> None:
    for line in patch.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line[6:].split("\t")[0].strip()
            if path != MUTABLE_PATH and path in PROTECTED_PATHS:
                raise ExecutorError(f"patch modifies protected path: {path}")


def _git_worktree(clone: Path, worktree: Path, commit: str) -> None:
    subprocess.run(
        ["git", "-C", str(clone), "worktree", "add", "--detach", str(worktree), commit],
        check=True,
        capture_output=True,
        text=True,
    )


def _apply_patch(worktree: Path, patch: str) -> None:
    proc = subprocess.run(
        ["git", "apply", "--check"],
        input=patch,
        capture_output=True,
        text=True,
        cwd=str(worktree),
    )
    if proc.returncode != 0:
        raise ExecutorError(f"patch does not apply cleanly: {proc.stderr.strip()}")
    subprocess.run(
        ["git", "apply"],
        input=patch,
        capture_output=True,
        text=True,
        check=True,
        cwd=str(worktree),
    )
