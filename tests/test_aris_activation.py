import hashlib
import json
import stat
import subprocess
from pathlib import Path

import pytest

import arw.aris_activation as activation
from arw.aris_activation import (
    ADAPTER_MARKER,
    ActivationState,
    ArisActivationError,
    CapabilityProfile,
    create_activation_snapshot,
    install_profile,
    load_capability_profile,
    plan_install,
    preflight,
    uninstall_profile,
    verify_activation_snapshot,
    verify_capability_profile,
)
from arw.aris_vendor import ArisLock, SnapshotManifest, VendorReport, load_lock, verify_repo

pytestmark = pytest.mark.stage_a2
ROOT = Path(__file__).parents[1]
LOCK = ROOT / "configs" / "integrations" / "aris.yaml"
PROFILE = ROOT / "configs" / "integrations" / "aris-capabilities.yaml"


def _report(profile: CapabilityProfile) -> VendorReport:
    return VendorReport(
        skill_names=tuple(sorted((*profile.native_skills, *profile.blocked_skills))),
        inventory_sha256=profile.inventory_sha256,
        source_tree_sha256="0" * 64,
        profile_sha256=profile.vendor_profile_sha256,
    )


def test_profile_partitions_the_exact_pinned_inventory() -> None:
    profile = load_capability_profile(PROFILE)
    report = verify_repo(load_lock(LOCK))

    verify_capability_profile(profile, report)

    assert len(profile.native_skills) == 68
    assert len(profile.blocked_skills) == 13
    assert len(set(profile.native_skills)) == 68
    assert len(set(profile.blocked_skills)) == 13
    assert set(profile.native_skills).isdisjoint(profile.blocked_skills)
    assert set(profile.native_skills) | set(profile.blocked_skills) == set(report.skill_names)
    assert profile.inventory_sha256 == report.inventory_sha256
    assert profile.vendor_profile_sha256 == report.profile_sha256


@pytest.mark.parametrize("group", ["native_skills", "blocked_skills"])
def test_profile_rejects_duplicate_names(group: str) -> None:
    profile = load_capability_profile(PROFILE)
    values = getattr(profile, group)
    changed = profile.model_copy(update={group: (*values, values[0])})

    with pytest.raises(ArisActivationError, match="duplicate"):
        verify_capability_profile(changed, _report(profile))


def test_profile_rejects_overlapping_names() -> None:
    profile = load_capability_profile(PROFILE)
    changed = profile.model_copy(
        update={"blocked_skills": (*profile.blocked_skills, profile.native_skills[0])}
    )

    with pytest.raises(ArisActivationError, match="overlap"):
        verify_capability_profile(changed, _report(profile))


@pytest.mark.parametrize("group", ["native_skills", "blocked_skills"])
def test_profile_rejects_removed_name(group: str) -> None:
    profile = load_capability_profile(PROFILE)
    changed = profile.model_copy(update={group: getattr(profile, group)[:-1]})

    with pytest.raises(ArisActivationError, match=r"count|inventory"):
        verify_capability_profile(changed, _report(profile))


def test_profile_rejects_added_name() -> None:
    profile = load_capability_profile(PROFILE)
    changed = profile.model_copy(update={"native_skills": (*profile.native_skills, "unknown")})

    with pytest.raises(ArisActivationError, match=r"count|inventory"):
        verify_capability_profile(changed, _report(profile))


def test_profile_digest_hashes_canonical_json() -> None:
    profile = load_capability_profile(PROFILE)
    canonical = json.dumps(
        profile.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()

    assert profile.digest() == hashlib.sha256(canonical).hexdigest()


def _remove_write_bits(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        path.chmod(path.stat().st_mode & ~0o222)


def _activation_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[ArisLock, CapabilityProfile, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile = load_capability_profile(PROFILE)
    lock = ArisLock(
        schema_version=1,
        repo_path=tmp_path / "unused-upstream",
        origin="https://example.invalid/aris.git",
        commit="a" * 40,
        license_spdx="MIT",
        inventory_count=81,
        shared_reference_count=1,
        inventory_sha256=profile.inventory_sha256,
        source_tree_sha256="b" * 64,
        activation_mode="disabled",
        unknown_skill_policy="block",
    )
    profile = profile.model_copy(update={"vendor_profile_sha256": lock.digest()})
    source = workspace / ".aris" / "vendor" / "aris" / lock.digest()
    for name in (*profile.native_skills, *profile.blocked_skills):
        skill = source / "skills" / "skills-codex" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\nsource\n", encoding="utf-8")
        if name in profile.blocked_skills:
            executable = skill / "execute.sh"
            executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            executable.chmod(0o755)
    for relative in (
        "skills/shared-references/reference.md",
        "skills/skills-codex/shared-references/reference.md",
        "tools/install_aris_codex.sh",
        "templates/template.md",
        "docs/guide.md",
        ".autoacademic/vendor.json",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture {relative}\n", encoding="utf-8")
    _remove_write_bits(source)
    vendor_manifest = SnapshotManifest(
        schema_version=1,
        aris_origin=lock.origin,
        aris_commit=lock.commit,
        profile_sha256=lock.digest(),
        inventory_count=lock.inventory_count,
        shared_reference_count=lock.shared_reference_count,
        inventory_sha256=lock.inventory_sha256,
        source_tree_sha256=lock.source_tree_sha256,
        activation_mode="disabled",
        unknown_skill_policy="block",
    )
    monkeypatch.setattr(activation, "create_snapshot", lambda *_: source)
    monkeypatch.setattr(activation, "verify_snapshot", lambda *_: vendor_manifest)
    return lock, profile, workspace


def test_activation_snapshot_preserves_native_content_and_blocks_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock, profile, workspace = _activation_fixture(monkeypatch, tmp_path)
    source = workspace / ".aris" / "vendor" / "aris" / lock.digest()

    snapshot = create_activation_snapshot(lock, profile, workspace)
    manifest = verify_activation_snapshot(lock, profile, workspace, snapshot)

    assert snapshot.name == profile.digest()
    assert manifest.native_skills == profile.native_skills
    assert manifest.blocked_skills == profile.blocked_skills
    assert manifest.native_count == 68
    assert manifest.blocked_count == 13
    assert manifest.entry_count == 82
    for name in profile.native_skills:
        relative = Path("skills") / "skills-codex" / name / "SKILL.md"
        assert (snapshot / relative).read_bytes() == (source / relative).read_bytes()
        assert stat.S_IMODE((snapshot / relative).stat().st_mode) == stat.S_IMODE(
            (source / relative).stat().st_mode
        )
    for name in profile.blocked_skills:
        skill = snapshot / "skills" / "skills-codex" / name
        assert [path.name for path in skill.iterdir()] == ["SKILL.md"]
        adapter = (skill / "SKILL.md").read_text(encoding="utf-8")
        assert f"name: {name}" in adapter
        assert "allowed-tools: Read" in adapter
        assert ADAPTER_MARKER in adapter
        assert "Bash" not in adapter
    for relative in (
        "skills/shared-references/reference.md",
        "tools/install_aris_codex.sh",
        "templates/template.md",
        "docs/guide.md",
    ):
        assert (snapshot / relative).is_file()
    assert all(not path.lstat().st_mode & 0o222 for path in (snapshot, *snapshot.rglob("*")))


@pytest.mark.parametrize("tamper", ["native", "adapter", "manifest", "symlink", "writable"])
def test_activation_snapshot_rejects_tampering(
    tamper: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock, profile, workspace = _activation_fixture(monkeypatch, tmp_path)
    snapshot = create_activation_snapshot(lock, profile, workspace)
    native = snapshot / "skills" / "skills-codex" / profile.native_skills[0] / "SKILL.md"
    adapter = snapshot / "skills" / "skills-codex" / profile.blocked_skills[0] / "SKILL.md"
    manifest = snapshot / ".autoacademic" / "activation.json"
    if tamper == "native":
        native.chmod(0o644)
        native.write_text("changed\n", encoding="utf-8")
    elif tamper == "adapter":
        adapter.chmod(0o644)
        adapter.write_text("changed\n", encoding="utf-8")
    elif tamper == "manifest":
        manifest.chmod(0o644)
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["entry_count"] = 81
        manifest.write_text(json.dumps(document), encoding="utf-8")
    elif tamper == "symlink":
        native.parent.chmod(0o755)
        native.unlink()
        native.symlink_to(PROFILE)
    else:
        adapter.chmod(0o644)

    with pytest.raises(ArisActivationError):
        verify_activation_snapshot(lock, profile, workspace, snapshot)


def _official_manifest(workspace: Path, repo: Path, profile: CapabilityProfile) -> bytes:
    lines = [
        "version\t1",
        f"repo_root\t{repo}",
        f"project_root\t{workspace}",
        "generated\t2026-07-29T00:00:00Z",
        "packages\tskills-codex",
        "kind\tname\tsource_rel\ttarget_rel\tmode",
    ]
    names = sorted((*profile.native_skills, *profile.blocked_skills, "shared-references"))
    for name in names:
        kind = "support" if name == "shared-references" else "skill"
        lines.append(f"{kind}\t{name}\tskills/skills-codex/{name}\t.agents/skills/{name}\tsymlink")
    return ("\n".join(lines) + "\n").encode()


def _mock_official_installer(
    monkeypatch: pytest.MonkeyPatch, profile: CapabilityProfile
) -> list[tuple[list[str], dict[str, object]]]:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        workspace = Path(argv[2])
        repo = Path(argv[argv.index("--aris-repo") + 1])
        if "--dry-run" in argv:
            return subprocess.CompletedProcess(argv, 0, "Plan summary:\n  CREATE: 82\n", "")
        manifest = workspace / ".aris" / "installed-skills-codex.txt"
        if "--uninstall" in argv:
            if manifest.exists():
                for name in (*profile.native_skills, *profile.blocked_skills, "shared-references"):
                    (workspace / ".agents" / "skills" / name).unlink(missing_ok=True)
                manifest.replace(manifest.with_suffix(".txt.prev"))
            return subprocess.CompletedProcess(argv, 0, "", "")
        skills = workspace / ".agents" / "skills"
        skills.mkdir(parents=True, exist_ok=True)
        for name in (*profile.native_skills, *profile.blocked_skills, "shared-references"):
            (skills / name).symlink_to(repo / "skills" / "skills-codex" / name)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(_official_manifest(workspace, repo, profile))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(activation.subprocess, "run", run)
    return calls


def test_plan_uses_official_dry_run_without_mutating_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock, profile, workspace = _activation_fixture(monkeypatch, tmp_path)
    source = workspace / ".aris" / "vendor" / "aris" / lock.digest()
    for path in tuple(workspace.iterdir()):
        if path.name == ".aris":
            path.rename(tmp_path / "fixture-aris")
    monkeypatch.setattr(activation, "create_activation_snapshot", lambda *_: source)
    calls = _mock_official_installer(monkeypatch, profile)

    output = plan_install(lock, profile, workspace)

    assert "CREATE: 82" in output
    assert not any(workspace.iterdir())
    argv, _ = calls[0]
    assert argv[:3] == ["bash", argv[1], str(workspace)]
    assert argv[-3:] == ["--all", "--no-doc", "--dry-run"]
    assert "--quiet" not in argv


def _installed_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[
    ArisLock, CapabilityProfile, Path, ActivationState, list[tuple[list[str], dict[str, object]]]
]:
    lock, profile, workspace = _activation_fixture(monkeypatch, tmp_path)
    calls = _mock_official_installer(monkeypatch, profile)
    state = install_profile(lock, profile, workspace)
    return lock, profile, workspace, state, calls


def test_install_and_preflight_require_exact_official_82_entry_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock, profile, workspace, state, calls = _installed_fixture(monkeypatch, tmp_path)

    assert preflight(lock, profile, workspace) == state
    assert state.entry_count == 82
    assert len(state.native_skills) == 68
    assert len(state.blocked_skills) == 13
    argv, kwargs = calls[0]
    snapshot = workspace / ".aris" / "vendor" / "aris-activation" / profile.digest()
    assert argv == [
        "bash",
        str(
            workspace
            / ".aris"
            / "vendor"
            / "aris"
            / lock.digest()
            / "tools"
            / "install_aris_codex.sh"
        ),
        str(workspace),
        "--aris-repo",
        str(snapshot),
        "--all",
        "--quiet",
        "--no-doc",
    ]
    assert Path(kwargs["env"]["HOME"]) == workspace / ".aris" / "installer-home"  # type: ignore[index]
    rules = (workspace / ".codex" / "rules" / "aris-deny.rules").read_text(encoding="utf-8")
    assert all(f'pattern = ["{command}"]' in rules for command in profile.forbidden_commands)


@pytest.mark.parametrize("tamper", ["missing", "retargeted", "real-path", "extra-manifest"])
def test_preflight_rejects_installed_entry_drift(
    tamper: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock, profile, workspace, _, _ = _installed_fixture(monkeypatch, tmp_path)
    target = workspace / ".agents" / "skills" / profile.native_skills[0]
    if tamper == "missing":
        target.unlink()
    elif tamper == "retargeted":
        target.unlink()
        target.symlink_to(workspace / ".aris" / "vendor" / "aris" / lock.digest())
    elif tamper == "real-path":
        target.unlink()
        target.mkdir()
    else:
        manifest = workspace / ".aris" / "installed-skills-codex.txt"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(
                "skill\tunexpected\tskills/skills-codex/unexpected\t"
                ".agents/skills/unexpected\tsymlink\n"
            )

    with pytest.raises(ArisActivationError):
        preflight(lock, profile, workspace)


def test_install_rolls_back_with_official_uninstall_after_postcheck_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock, profile, workspace = _activation_fixture(monkeypatch, tmp_path)
    calls = _mock_official_installer(monkeypatch, profile)
    monkeypatch.setattr(
        activation,
        "_write_rules",
        lambda *_: (_ for _ in ()).throw(ArisActivationError("forced postcheck failure")),
    )

    with pytest.raises(ArisActivationError, match="forced postcheck"):
        install_profile(lock, profile, workspace)

    assert len(calls) == 2
    assert "--uninstall" in calls[1][0]
    assert not (workspace / ".aris" / "installed-skills-codex.txt").exists()
    assert not any((workspace / ".agents" / "skills").iterdir())


def test_uninstall_uses_a1_installer_and_preserves_user_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock, profile, workspace, _, calls = _installed_fixture(monkeypatch, tmp_path)
    user_file = workspace / ".agents" / "skills" / "user-skill.txt"
    user_file.write_text("keep\n", encoding="utf-8")
    research = workspace / "research.txt"
    research.write_text("keep\n", encoding="utf-8")

    uninstall_profile(lock, profile, workspace)

    argv, _ = calls[-1]
    a1_snapshot = workspace / ".aris" / "vendor" / "aris" / lock.digest()
    assert argv[3:] == [
        "--aris-repo",
        str(a1_snapshot),
        "--uninstall",
        "--quiet",
        "--no-doc",
    ]
    assert user_file.read_text(encoding="utf-8") == "keep\n"
    assert research.read_text(encoding="utf-8") == "keep\n"
    assert a1_snapshot.is_dir()
    assert (workspace / ".aris" / "vendor" / "aris-activation" / profile.digest()).is_dir()
    assert not (workspace / ".codex" / "rules" / "aris-deny.rules").exists()
    assert not (workspace / ".aris" / "autoacademic-activation.json").exists()


def test_uninstall_preserves_modified_deny_rules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock, profile, workspace, _, _ = _installed_fixture(monkeypatch, tmp_path)
    rules = workspace / ".codex" / "rules" / "aris-deny.rules"
    rules.write_text("user replacement\n", encoding="utf-8")

    uninstall_profile(lock, profile, workspace)

    assert rules.read_text(encoding="utf-8") == "user replacement\n"
    assert not (workspace / ".aris" / "autoacademic-activation.json").exists()


@pytest.mark.parametrize(
    "relative", [".aris", ".agents", ".agents/skills", ".codex", ".codex/rules"]
)
def test_install_rejects_symlinked_managed_parents(relative: str, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    lock = load_lock(LOCK)
    profile = load_capability_profile(PROFILE)

    with pytest.raises(ArisActivationError, match="symlink"):
        install_profile(lock, profile, workspace)
