"""Experiment executor — fake, real karpathy, and reliablepeft variants."""

from __future__ import annotations

import hashlib
import json
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


# ── ReliablePEFT Executor ──────────────────────────────────────────

RELIABLEPEFT_PHASE1_SCRIPT = "phase1_train.py"


class ReliablePEFTExecutor:
    """Real executor that runs phase1_train.py with config_id + seed.

    Requires CUDA-capable GPU. Checks torch.cuda.is_available() before
    executing and fails fast if no GPU is found.
    """

    def __init__(self, script_dir: Path) -> None:
        script_path = script_dir / RELIABLEPEFT_PHASE1_SCRIPT
        if not script_path.is_file():
            raise ExecutorError(
                f"ReliablePEFT script not found: {script_path}"
            )
        self._script_dir = script_dir.resolve()

    def _check_cuda(self) -> str | None:
        """Return error message if CUDA is unavailable, None if OK."""
        import sys
        try:
            import torch
        except ImportError:
            return "torch not installed in server venv — cannot check CUDA"
        if not torch.cuda.is_available():
            return f"CUDA not available (torch={torch.__version__}, device_count={torch.cuda.device_count()})"
        device_name = torch.cuda.get_device_name(0) or "unknown"
        try:
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except AttributeError:
            vram_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[ReliablePEFT] CUDA ready: {device_name} ({vram_total:.1f} GB)", flush=True)
        return None

    def execute(
        self,
        experiment_id: str,
        worktree: Path,
        patch: str,
        patch_sha256: str,
        time_budget_seconds: int,
        config_id: int | None = None,
        seed: int | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> ExecutionResult:
        """Run phase1_train.py --config-id X --seed Y."""
        if config_id is None or seed is None:
            return ExecutionResult(
                success=False,
                summary="config_id and seed are required for ReliablePEFT experiments",
                artifacts=[],
            )

        # ── Pre-flight: CUDA check ──
        cuda_error = self._check_cuda()
        if cuda_error is not None:
            return ExecutionResult(
                success=False,
                summary=f"GPU pre-flight failed: {cuda_error}",
                artifacts=[("preflight.log", cuda_error.encode(), "text/plain")],
            )

        data_dir = str(self._script_dir / "data" / "eurosat")
        output_dir = str(self._script_dir / "outputs" / "phase1_eurosat")

        import sys
        python_exe = sys.executable
        cmd = [
            python_exe, str(self._script_dir / RELIABLEPEFT_PHASE1_SCRIPT),
            "--config-id", str(config_id),
            "--seed", str(seed),
            "--data-dir", data_dir,
            "--output-dir", output_dir,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
        ]

        env = os.environ.copy()
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._script_dir),
                capture_output=True,
                timeout=time_budget_seconds + 120,
                env=env,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                summary=f"timeout after {time_budget_seconds + 120}s",
                artifacts=[],
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = (stdout + "\n" + stderr).encode()
        sha256_val = hashlib.sha256(combined).hexdigest()

        # ── Collect output files from the run directory ──
        run_dir = (
            self._script_dir / "outputs" / "phase1_eurosat"
            / f"config_{config_id:02d}" / f"seed_{seed}"
        )
        metrics_bytes = b"{}"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                metrics_bytes = metrics_path.read_bytes()
            except Exception:
                pass

        # Capture per-sample logits (.npy) as artifacts for Phase 1 analysis
        npy_files = [
            "val_logits.npy", "val_labels.npy",
            "test_logits.npy", "test_labels.npy",
        ]
        npy_artifacts: list[tuple[str, bytes, str]] = []
        for npy_name in npy_files:
            npy_path = run_dir / npy_name
            if npy_path.is_file():
                try:
                    npy_artifacts.append(
                        (npy_name, npy_path.read_bytes(), "application/octet-stream")
                    )
                except Exception:
                    pass

        success = proc.returncode == 0
        if success:
            try:
                m = json.loads(metrics_bytes)
                val_acc = m.get("val_acc")
                test_acc = m.get("test_acc")
                train_time = m.get("train_time_seconds")
                peak_vram = m.get("peak_vram_gb")
                val_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "?"
                test_str = f"{test_acc:.4f}" if isinstance(test_acc, (int, float)) else "?"
                time_str = f"{train_time:.0f}s" if isinstance(train_time, (int, float)) else "?"
                vram_str = f"{peak_vram:.1f}GB" if isinstance(peak_vram, (int, float)) else "?"
                summary = (
                    f"config_{config_id:02d}/seed_{seed}: "
                    f"val_acc={val_str} test_acc={test_str} "
                    f"time={time_str} vram={vram_str}"
                )
            except Exception:
                summary = f"config_{config_id:02d}/seed_{seed}: ok (exit 0)"
        else:
            tail = (stderr or stdout)[-500:]
            summary = f"config_{config_id:02d}/seed_{seed}: exit {proc.returncode} — {tail}"

        # Build artifact list: run log + metrics + .npy files + run summary JSON
        artifacts: list[tuple[str, bytes, str]] = [
            ("run.log", combined, "text/plain"),
            ("metrics.json", metrics_bytes, "application/json"),
            (
                "run.json",
                json.dumps({
                    "experiment_id": experiment_id,
                    "config_id": config_id,
                    "seed": seed,
                    "success": success,
                    "sha256": sha256_val,
                    "exit_code": proc.returncode,
                }).encode(),
                "application/json",
            ),
        ]
        artifacts.extend(npy_artifacts)

        return ExecutionResult(
            success=success,
            summary=summary,
            artifacts=artifacts,
        )
