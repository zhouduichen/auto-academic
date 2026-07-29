"""Stage A2 integration gate — real ARIS lifecycle in a temporary workspace.

Disabled by default. To enable:

    ARW_RUN_STAGE_A2_INTEGRATION=1 uv run pytest tools/stage_a2_gate.py -q

This gate uses the real pinned ARIS clone at the commit recorded in
``configs/integrations/aris.yaml`` and exercises the complete ARIS installer
lifecycle (plan → install → preflight → uninstall) with real git operations
and the unmodified official ``install_aris_codex.sh``.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from arw.aris_activation import (
    ACTIVATION_ENTRY_COUNT,
    ADAPTER_MARKER,
    BLOCKED_SKILL_COUNT,
    NATIVE_SKILL_COUNT,
    ArisActivationError,
    install_profile,
    load_capability_profile,
    plan_install,
    preflight,
    uninstall_profile,
    verify_activation_snapshot,
)
from arw.aris_vendor import ArisVendorError, load_lock

if not os.environ.get("ARW_RUN_STAGE_A2_INTEGRATION"):
    pytest.skip("ARW_RUN_STAGE_A2_INTEGRATION is not set", allow_module_level=True)

pytestmark = pytest.mark.stage_a2_integration

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "integrations" / "aris.yaml"
PROFILE = ROOT / "configs" / "integrations" / "aris-capabilities.yaml"


def _assert_workspace_clean(workspace: Path) -> None:
    """Verify no activation state, manifest, or deny rules remain after uninstall."""
    assert not (workspace / ".aris" / "autoacademic-activation.json").exists(), (
        "activation state should be removed by uninstall"
    )
    assert not (workspace / ".aris" / "installed-skills-codex.txt").exists(), (
        "official installer manifest should be removed by uninstall"
    )
    assert not (workspace / ".codex" / "rules" / "aris-deny.rules").exists(), (
        "deny rules should be removed by uninstall"
    )


def _assert_codex_execpolicy_validates(rules_path: Path) -> None:
    """Verify Codex execpolicy can parse the deny rules file.

    Skipped gracefully when the ``codex`` CLI is not available.
    """
    if shutil.which("codex") is None:
        pytest.skip("codex CLI not available — skipping execpolicy parse check")
    result = subprocess.run(
        ["codex", "execpolicy", "check", str(rules_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise AssertionError(
            f"codex execpolicy rejected deny rules (exit {result.returncode}): {detail}"
        )


def test_stage_a2_full_lifecycle() -> None:
    """Real plan → install → preflight → uninstall lifecycle with the fixed ARIS clone.

    This is the definitive Stage A2 gate. It runs the actual ARIS installer
    (unmodified ``install_aris_codex.sh``) against a content-addressed activation
    snapshot in an isolated temporary workspace and verifies:
    * plan output
    * 82-entry manifest (81 skills + shared-references)
    * 13 blocked adapters contain ``skill_not_activated``
    * deny rules are syntactically valid for Codex execpolicy
    * preflight re-validates identical state
    * uninstall removes all managed artefacts
    * activation snapshot integrity survives the full lifecycle
    """
    lock = load_lock(LOCK)
    profile = load_capability_profile(PROFILE)

    with tempfile.TemporaryDirectory(prefix="arw-stage-a2-gate-") as tmp:
        workspace = Path(tmp)

        # ── Plan ──────────────────────────────────────────────────────
        plan_output = plan_install(lock, profile, workspace)
        assert "CREATE: 82" in plan_output, (
            f"plan output should declare 82 CREATE entries, got:\n{plan_output}"
        )

        # ── Install ───────────────────────────────────────────────────
        state = install_profile(lock, profile, workspace)
        assert state.entry_count == ACTIVATION_ENTRY_COUNT, (
            f"expected {ACTIVATION_ENTRY_COUNT} entries, got {state.entry_count}"
        )
        assert len(state.native_skills) == NATIVE_SKILL_COUNT
        assert len(state.blocked_skills) == BLOCKED_SKILL_COUNT

        # ── Manifest: 82 entries ──────────────────────────────────────
        manifest_path = workspace / ".aris" / "installed-skills-codex.txt"
        assert manifest_path.is_file(), "official installer manifest is missing"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        body_lines = [
            ln
            for ln in manifest_text.splitlines()
            if ln.count("\t") == 4 and not ln.startswith("kind\t")
        ]
        assert len(body_lines) == ACTIVATION_ENTRY_COUNT, (
            f"manifest body has {len(body_lines)} rows, expected {ACTIVATION_ENTRY_COUNT}"
        )

        # ── Blocked adapters ──────────────────────────────────────────
        for name in profile.blocked_skills:
            skill_dir = workspace / ".agents" / "skills" / name
            assert skill_dir.is_symlink(), (
                f"blocked skill {name} should be a symlink into the activation snapshot"
            )
            resolved = skill_dir.resolve(strict=True)
            adapter = resolved / "SKILL.md"
            assert adapter.is_file(), f"blocked skill {name} is missing SKILL.md adapter"
            content = adapter.read_text(encoding="utf-8")
            assert ADAPTER_MARKER in content, (
                f"blocked skill {name} adapter does not contain {ADAPTER_MARKER}"
            )
            assert "allowed-tools: Read" in content, (
                f"blocked skill {name} adapter should only allow Read"
            )
            assert "Bash" not in content, f"blocked skill {name} adapter must not allow Bash"

        # ── Deny rules syntax ─────────────────────────────────────────
        rules_path = workspace / ".codex" / "rules" / "aris-deny.rules"
        assert rules_path.is_file(), "deny rules file is missing"
        rules_text = rules_path.read_text(encoding="utf-8")
        for command in ("ssh", "scp", "rsync", "screen", "tmux", "modal"):
            assert f'pattern = ["{command}"]' in rules_text, (
                f"deny rules are missing forbidden command: {command}"
            )
        assert 'decision = "forbidden"' in rules_text, "deny rules must set decision=forbidden"
        _assert_codex_execpolicy_validates(rules_path)

        # ── Preflight ─────────────────────────────────────────────────
        state2 = preflight(lock, profile, workspace)
        assert state2 == state, "preflight state must match the state returned by install_profile"

        # ── Activation snapshot integrity ─────────────────────────────
        snapshot = workspace / ".aris" / "vendor" / "aris-activation" / profile.digest()
        manifest = verify_activation_snapshot(lock, profile, workspace, snapshot)
        assert manifest.entry_count == ACTIVATION_ENTRY_COUNT

        # ── Uninstall ─────────────────────────────────────────────────
        uninstall_profile(lock, profile, workspace)
        _assert_workspace_clean(workspace)

        # ── Activation snapshot survives uninstall ────────────────────
        assert snapshot.is_dir(), (
            "activation snapshot must survive uninstall (it is a content-addressed artefact)"
        )
        assert all(
            not (path.lstat().st_mode & stat.S_IWUSR) for path in (snapshot, *snapshot.rglob("*"))
        ), "activation snapshot must remain read-only after uninstall"


def test_stage_a2_install_rejects_invalid_workspace() -> None:
    """A2 install must reject a workspace that is a symlink."""
    lock = load_lock(LOCK)
    profile = load_capability_profile(PROFILE)

    with tempfile.TemporaryDirectory(prefix="arw-stage-a2-gate-") as tmp:
        base = Path(tmp)
        real_workspace = base / "real-workspace"
        real_workspace.mkdir()
        symlink_workspace = base / "symlink-workspace"
        symlink_workspace.symlink_to(real_workspace, target_is_directory=True)

        with pytest.raises(ArisActivationError, match="symlink"):
            install_profile(lock, profile, symlink_workspace)


def test_stage_a2_uninstall_idempotent_against_clean_workspace() -> None:
    """A2 uninstall on a clean workspace must refuse (no snapshot to verify)."""
    lock = load_lock(LOCK)
    profile = load_capability_profile(PROFILE)

    with tempfile.TemporaryDirectory(prefix="arw-stage-a2-gate-") as tmp:
        workspace = Path(tmp)
        # On a clean workspace the A1 vendor snapshot does not exist, so
        # uninstall_profile fails at the first verification step.
        with pytest.raises(ArisVendorError):
            uninstall_profile(lock, profile, workspace)


def test_stage_a2_deny_rules_forbid_all_ten_commands() -> None:
    """Verify the generated deny rules block all ten forbidden commands."""
    profile = load_capability_profile(PROFILE)

    from arw.aris_activation import FORBIDDEN_COMMANDS, _rules_bytes

    rules = _rules_bytes(profile).decode()
    assert len(FORBIDDEN_COMMANDS) == 10
    for command in FORBIDDEN_COMMANDS:
        assert f'pattern = ["{command}"]' in rules, (
            f"deny rules missing forbidden command: {command}"
        )
        assert 'decision = "forbidden"' in rules, (
            f"deny rules for {command} must set decision=forbidden"
        )
