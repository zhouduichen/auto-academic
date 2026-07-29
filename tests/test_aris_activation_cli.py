import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import arw.aris_activation as activation
from arw.aris_activation import ActivationState, run_profile
from arw.cli import app

pytestmark = pytest.mark.stage_a2
runner = CliRunner()


def _state(workspace: Path) -> ActivationState:
    return ActivationState(
        schema_version=1,
        vendor_profile_sha256="a" * 64,
        capability_profile_sha256="b" * 64,
        activation_snapshot=workspace / ".aris" / "vendor" / "aris-activation" / ("c" * 64),
        activation_tree_sha256="d" * 64,
        official_manifest_sha256="e" * 64,
        deny_rules_sha256="f" * 64,
        entry_count=82,
        native_skills=("native",),
        blocked_skills=("blocked",),
        forbidden_commands=("ssh",),
    )


def test_aris_help_exposes_complete_a2_lifecycle() -> None:
    result = runner.invoke(app, ["aris", "--help"])

    assert result.exit_code == 0
    for command in ("verify", "snapshot", "plan", "install", "preflight", "uninstall", "run"):
        assert command in result.output


@pytest.mark.parametrize("command", ["install", "uninstall"])
def test_mutating_activation_commands_require_confirmation(command: str, tmp_path: Path) -> None:
    result = runner.invoke(app, ["aris", command, "--workspace", str(tmp_path)])

    assert result.exit_code == 2
    assert "--confirm" in result.output


def test_run_profile_launches_exact_hardened_local_codex_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path.resolve()
    state = _state(workspace)
    checks: list[tuple[object, object, Path]] = []
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        activation,
        "preflight",
        lambda lock, profile, root: checks.append((lock, profile, root)) or state,
    )
    monkeypatch.setattr(activation.shutil, "which", lambda command: "/usr/local/bin/codex")

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(activation.subprocess, "run", run)

    result = run_profile(object(), object(), workspace, "Analyze the current paper.")  # type: ignore[arg-type]

    assert result == 0
    assert len(checks) == 2
    argv, kwargs = calls[0]
    assert argv[:11] == [
        "codex",
        "--strict-config",
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "never",
        "-c",
        "sandbox_workspace_write.network_access=false",
        "-c",
        f'projects."{workspace}".trust_level="trusted"',
        "--cd",
    ]
    assert argv[11] == str(workspace)
    safety_prompt = argv[12]
    assert safety_prompt.endswith("Analyze the current paper.")
    for forbidden in (
        "ssh",
        "scp",
        "rsync",
        "screen",
        "tmux",
        "Modal",
        "Vast",
        "qzcli",
        "Tailscale",
        "background",
        "training",
        ".agents",
        ".codex",
        "activation snapshot",
        "manifest",
    ):
        assert forbidden in safety_prompt
    assert "--search" not in argv
    assert "--add-dir" not in argv
    assert kwargs == {"check": False}


def test_run_profile_requires_local_codex(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(activation, "preflight", lambda *_: _state(tmp_path))
    monkeypatch.setattr(activation.shutil, "which", lambda command: None)

    with pytest.raises(activation.ArisActivationError, match="Codex CLI"):
        run_profile(object(), object(), tmp_path, "Task")  # type: ignore[arg-type]
