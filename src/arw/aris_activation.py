from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from arw.aris_vendor import (
    ArisLock,
    Commit,
    Sha256,
    SnapshotManifest,
    VendorReport,
    create_snapshot,
    verify_snapshot,
)

NATIVE_SKILL_COUNT = 68
BLOCKED_SKILL_COUNT = 13
TOTAL_SKILL_COUNT = NATIVE_SKILL_COUNT + BLOCKED_SKILL_COUNT
FORBIDDEN_COMMANDS = (
    "ssh",
    "scp",
    "rsync",
    "screen",
    "tmux",
    "modal",
    "vast",
    "vastai",
    "qzcli",
    "tailscale",
)
ADAPTER_MARKER = "skill_not_activated"
ACTIVATION_ENTRY_COUNT: Literal[82] = 82
TreeEntry = tuple[bytes, bytes, bytes, bytes]
PositiveInt = Annotated[int, Field(ge=1)]


class ArisActivationError(RuntimeError):
    pass


class CapabilityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    vendor_profile_sha256: Sha256
    inventory_sha256: Sha256
    unknown_skill_policy: Literal["block"]
    blocked_adapter_version: Literal[1]
    native_skills: tuple[str, ...]
    blocked_skills: tuple[str, ...]
    forbidden_commands: tuple[str, ...]

    def digest(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class ActivationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    aris_origin: str
    aris_commit: Commit
    vendor_profile_sha256: Sha256
    inventory_sha256: Sha256
    source_tree_sha256: Sha256
    capability_profile_sha256: Sha256
    activation_mode: Literal["development-safe"]
    unknown_skill_policy: Literal["block"]
    native_count: PositiveInt
    blocked_count: PositiveInt
    native_skills: tuple[str, ...]
    blocked_skills: tuple[str, ...]
    entry_count: Literal[82]
    blocked_adapter_version: Literal[1]
    forbidden_commands: tuple[str, ...]
    activation_tree_sha256: Sha256


class ActivationState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    vendor_profile_sha256: Sha256
    capability_profile_sha256: Sha256
    activation_snapshot: Path
    activation_tree_sha256: Sha256
    official_manifest_sha256: Sha256
    deny_rules_sha256: Sha256
    entry_count: Literal[82]
    native_skills: tuple[str, ...]
    blocked_skills: tuple[str, ...]
    forbidden_commands: tuple[str, ...]


def load_capability_profile(path: Path) -> CapabilityProfile:
    try:
        return CapabilityProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        raise ArisActivationError("ARIS capability profile is invalid") from exc


def _assert_unique(label: str, names: tuple[str, ...]) -> None:
    if len(names) != len(set(names)):
        raise ArisActivationError(f"{label} contains a duplicate skill name")


def verify_capability_profile(profile: CapabilityProfile, vendor_report: VendorReport) -> None:
    _assert_unique("native_skills", profile.native_skills)
    _assert_unique("blocked_skills", profile.blocked_skills)
    native = set(profile.native_skills)
    blocked = set(profile.blocked_skills)
    if native & blocked:
        raise ArisActivationError("native and blocked skill inventories overlap")
    if len(native) != NATIVE_SKILL_COUNT or len(blocked) != BLOCKED_SKILL_COUNT:
        raise ArisActivationError("capability profile skill count is not 68 native and 13 blocked")
    if native | blocked != set(vendor_report.skill_names):
        raise ArisActivationError("capability profile does not match the ARIS inventory")
    if profile.inventory_sha256 != vendor_report.inventory_sha256:
        raise ArisActivationError("capability profile inventory digest does not match A1")
    if profile.vendor_profile_sha256 != vendor_report.profile_sha256:
        raise ArisActivationError("capability profile vendor digest does not match A1")
    if profile.forbidden_commands != FORBIDDEN_COMMANDS:
        raise ArisActivationError("capability profile forbidden commands do not match Stage A2")


def _adapter_bytes(name: str) -> bytes:
    return (
        "---\n"
        f"name: {name}\n"
        "description: This execution-capable ARIS skill is disabled in Stage A2.\n"
        "allowed-tools: Read\n"
        "---\n\n"
        "# Skill not activated\n\n"
        f"`{ADAPTER_MARKER}`\n\n"
        "This skill is intentionally unavailable in the development-safe profile.\n"
    ).encode()


def _scan_tree(root: Path, *, exclude_activation_manifest: bool = False) -> dict[bytes, TreeEntry]:
    entries: dict[bytes, TreeEntry] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if exclude_activation_manifest and relative == ".autoacademic/activation.json":
            continue
        info = path.lstat()
        encoded = relative.encode()
        if stat.S_ISDIR(info.st_mode):
            entry = (encoded, b"tree", b"040000", b"")
        elif stat.S_ISREG(info.st_mode):
            mode = b"100755" if info.st_mode & 0o111 else b"100644"
            entry = (encoded, b"blob", mode, path.read_bytes())
        else:
            raise ArisActivationError(f"unsupported activation entry: {relative}")
        entries[encoded] = entry
    return entries


def _digest_tree(entries: dict[bytes, TreeEntry]) -> str:
    digest = hashlib.sha256()
    for key in sorted(entries):
        for field in entries[key]:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _expected_activation_entries(
    source: Path, profile: CapabilityProfile
) -> dict[bytes, TreeEntry]:
    entries = _scan_tree(source)
    for name in profile.blocked_skills:
        prefix = f"skills/skills-codex/{name}".encode()
        for path in tuple(entries):
            if path == prefix or path.startswith(prefix + b"/"):
                del entries[path]
        entries[prefix] = (prefix, b"tree", b"040000", b"")
        skill_path = prefix + b"/SKILL.md"
        entries[skill_path] = (skill_path, b"blob", b"100644", _adapter_bytes(name))
    return entries


def _set_owner_write(root: Path, enabled: bool) -> None:
    paths = (root, *root.rglob("*"))
    for path in paths:
        mode = path.lstat().st_mode
        path.chmod(mode | stat.S_IWUSR if enabled else mode & ~0o222)


def _assert_read_only(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        if path.lstat().st_mode & 0o222:
            relative = path.relative_to(root) if path != root else Path(".")
            raise ArisActivationError(f"activation snapshot entry is writable: {relative}")


def _activation_vendor_root(workspace: Path) -> Path:
    if not workspace.is_dir() or workspace.is_symlink():
        raise ArisActivationError(f"workspace is not a real directory: {workspace}")
    root = workspace / ".aris" / "vendor" / "aris-activation"
    for path in (workspace / ".aris", workspace / ".aris" / "vendor", root):
        if path.is_symlink():
            raise ArisActivationError(f"activation vendor path contains a symlink: {path}")
        if path.exists() and not path.is_dir():
            raise ArisActivationError(f"activation vendor path is not a directory: {path}")
    return root


def _vendor_report_from_snapshot(source: Path, manifest: SnapshotManifest) -> VendorReport:
    skill_files = tuple((source / "skills" / "skills-codex").glob("*/SKILL.md"))
    if any(path.is_symlink() or path.parent.is_symlink() for path in skill_files):
        raise ArisActivationError("A1 snapshot skill inventory contains a symlink")
    return VendorReport(
        skill_names=tuple(sorted(path.parent.name for path in skill_files)),
        inventory_sha256=manifest.inventory_sha256,
        source_tree_sha256=manifest.source_tree_sha256,
        profile_sha256=manifest.profile_sha256,
    )


def _expected_activation_manifest(
    lock: ArisLock, profile: CapabilityProfile, tree_sha256: str
) -> ActivationManifest:
    return ActivationManifest(
        schema_version=1,
        aris_origin=lock.origin,
        aris_commit=lock.commit,
        vendor_profile_sha256=lock.digest(),
        inventory_sha256=lock.inventory_sha256,
        source_tree_sha256=lock.source_tree_sha256,
        capability_profile_sha256=profile.digest(),
        activation_mode="development-safe",
        unknown_skill_policy="block",
        native_count=NATIVE_SKILL_COUNT,
        blocked_count=BLOCKED_SKILL_COUNT,
        native_skills=profile.native_skills,
        blocked_skills=profile.blocked_skills,
        entry_count=ACTIVATION_ENTRY_COUNT,
        blocked_adapter_version=profile.blocked_adapter_version,
        forbidden_commands=profile.forbidden_commands,
        activation_tree_sha256=tree_sha256,
    )


def create_activation_snapshot(lock: ArisLock, profile: CapabilityProfile, workspace: Path) -> Path:
    source = create_snapshot(lock, workspace)
    vendor_manifest = verify_snapshot(lock, source)
    verify_capability_profile(profile, _vendor_report_from_snapshot(source, vendor_manifest))
    vendor = _activation_vendor_root(workspace)
    destination = vendor / profile.digest()
    if destination.exists() or destination.is_symlink():
        verify_activation_snapshot(lock, profile, workspace, destination)
        return destination
    vendor.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".activation-", dir=vendor))
    staged = temporary / "root"
    try:
        shutil.copytree(source, staged)
        _set_owner_write(staged, True)
        for name in profile.blocked_skills:
            skill = staged / "skills" / "skills-codex" / name
            shutil.rmtree(skill)
            skill.mkdir()
            (skill / "SKILL.md").write_bytes(_adapter_bytes(name))
        expected_entries = _expected_activation_entries(source, profile)
        if _scan_tree(staged) != expected_entries:
            raise ArisActivationError("staged activation tree does not match its A1 derivation")
        tree_sha256 = _digest_tree(expected_entries)
        manifest = _expected_activation_manifest(lock, profile, tree_sha256)
        metadata = staged / ".autoacademic"
        metadata.mkdir(exist_ok=True)
        (metadata / "activation.json").write_text(
            manifest.model_dump_json() + "\n", encoding="utf-8"
        )
        _set_owner_write(staged, False)
        # macOS requires the moved directory itself to remain owner-writable for rename(2).
        staged.chmod(staged.stat().st_mode | stat.S_IWUSR)
        os.rename(staged, destination)
        destination.chmod(destination.stat().st_mode & ~0o222)
        verify_activation_snapshot(lock, profile, workspace, destination)
    finally:
        if staged.exists():
            _set_owner_write(staged, True)
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def verify_activation_snapshot(
    lock: ArisLock,
    profile: CapabilityProfile,
    workspace: Path,
    snapshot: Path,
) -> ActivationManifest:
    source = workspace / ".aris" / "vendor" / "aris" / lock.digest()
    vendor_manifest = verify_snapshot(lock, source)
    verify_capability_profile(profile, _vendor_report_from_snapshot(source, vendor_manifest))
    expected_path = _activation_vendor_root(workspace) / profile.digest()
    if snapshot != expected_path or not snapshot.is_dir() or snapshot.is_symlink():
        raise ArisActivationError("activation snapshot path does not match the profile digest")
    expected_entries = _expected_activation_entries(source, profile)
    actual_entries = _scan_tree(snapshot, exclude_activation_manifest=True)
    if actual_entries != expected_entries:
        raise ArisActivationError("activation tree does not match its A1 derivation")
    expected = _expected_activation_manifest(lock, profile, _digest_tree(expected_entries))
    try:
        actual = ActivationManifest.model_validate_json(
            (snapshot / ".autoacademic" / "activation.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ArisActivationError("activation manifest is invalid") from exc
    if actual != expected:
        raise ArisActivationError("activation manifest does not match the locked profile")
    _assert_read_only(snapshot)
    return actual


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _installer_argv(
    installer: Path,
    workspace: Path,
    aris_repo: Path,
    *options: str,
) -> list[str]:
    return [
        "bash",
        str(installer),
        str(workspace),
        "--aris-repo",
        str(aris_repo),
        *options,
    ]


def _run_installer(
    argv: list[str], *, home: Path, capture_output: bool
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    try:
        return subprocess.run(  # noqa: S603
            argv,
            check=True,
            capture_output=capture_output,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or str(exc)
        raise ArisActivationError(f"official ARIS installer failed: {str(detail).strip()}") from exc


def _assert_managed_paths_safe(workspace: Path) -> None:
    if not workspace.is_dir() or workspace.is_symlink():
        raise ArisActivationError(f"workspace is not a real directory: {workspace}")
    managed = (
        workspace / ".aris",
        workspace / ".aris" / "vendor",
        workspace / ".aris" / "installed-skills-codex.txt",
        workspace / ".aris" / "autoacademic-activation.json",
        workspace / ".agents",
        workspace / ".agents" / "skills",
        workspace / ".codex",
        workspace / ".codex" / "rules",
        workspace / ".codex" / "rules" / "aris-deny.rules",
    )
    for path in managed:
        if path.is_symlink():
            raise ArisActivationError(f"managed activation path contains a symlink: {path}")
    for path in managed:
        if path.exists() and path.suffix == "" and not path.is_dir():
            raise ArisActivationError(f"managed activation parent is not a directory: {path}")


def _rules_bytes(profile: CapabilityProfile) -> bytes:
    blocks = []
    for command in profile.forbidden_commands:
        blocks.append(
            "prefix_rule(\n"
            f'    pattern = ["{command}"],\n'
            '    decision = "forbidden",\n'
            '    justification = "Stage A2 forbids remote and external execution commands.",\n'
            f'    match = ["{command} --help"],\n'
            ")\n"
        )
    return (
        "# AutoAcademic Stage A2 managed deny rules.\n"
        "# Do not edit while ARIS activation is installed.\n\n" + "\n".join(blocks)
    ).encode()


def _check_rules_compatible(workspace: Path, profile: CapabilityProfile) -> None:
    path = workspace / ".codex" / "rules" / "aris-deny.rules"
    if path.exists() and (not path.is_file() or path.read_bytes() != _rules_bytes(profile)):
        raise ArisActivationError(f"deny rules conflict with an existing file: {path}")


def _write_rules(workspace: Path, profile: CapabilityProfile) -> Path:
    _check_rules_compatible(workspace, profile)
    path = workspace / ".codex" / "rules" / "aris-deny.rules"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(_rules_bytes(profile))
    return path


def plan_install(lock: ArisLock, profile: CapabilityProfile, workspace: Path) -> str:
    _assert_managed_paths_safe(workspace)
    _check_rules_compatible(workspace, profile)
    expected_repo = workspace / ".aris" / "vendor" / "aris-activation" / profile.digest()
    with tempfile.TemporaryDirectory(prefix="arw-aris-plan-") as temporary:
        staging_workspace = Path(temporary)
        activation = create_activation_snapshot(lock, profile, staging_workspace)
        a1_snapshot = staging_workspace / ".aris" / "vendor" / "aris" / lock.digest()
        installer = a1_snapshot / "tools" / "install_aris_codex.sh"
        result = _run_installer(
            _installer_argv(
                installer,
                workspace,
                activation,
                "--all",
                "--no-doc",
                "--dry-run",
            ),
            home=staging_workspace / "installer-home",
            capture_output=True,
        )
        return result.stdout.replace(str(activation), str(expected_repo))


def _parse_official_manifest(path: Path) -> tuple[dict[str, str], list[tuple[str, ...]]]:
    if not path.is_file() or path.is_symlink():
        raise ArisActivationError("official installer manifest is missing or not a regular file")
    metadata: dict[str, str] = {}
    rows: list[tuple[str, ...]] = []
    in_body = False
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = tuple(line.split("\t"))
        if fields == ("kind", "name", "source_rel", "target_rel", "mode"):
            if in_body:
                raise ArisActivationError("official installer manifest repeats its body header")
            in_body = True
        elif in_body:
            if len(fields) != 5:
                raise ArisActivationError("official installer manifest has an invalid body row")
            rows.append(fields)
        else:
            if len(fields) != 2 or fields[0] in metadata:
                raise ArisActivationError("official installer manifest has invalid metadata")
            metadata[fields[0]] = fields[1]
    if not in_body:
        raise ArisActivationError("official installer manifest has no body header")
    if set(metadata) != {"version", "repo_root", "project_root", "generated", "packages"}:
        raise ArisActivationError(
            "official installer manifest metadata is incomplete or unexpected"
        )
    return metadata, rows


def _validate_official_install(
    workspace: Path, activation: Path, profile: CapabilityProfile
) -> tuple[str, int]:
    manifest_path = workspace / ".aris" / "installed-skills-codex.txt"
    metadata, rows = _parse_official_manifest(manifest_path)
    expected_names = set(profile.native_skills) | set(profile.blocked_skills)
    if metadata.get("version") != "1":
        raise ArisActivationError("official installer manifest version is not 1")
    if metadata.get("repo_root") != str(activation):
        raise ArisActivationError(
            "official installer manifest repo root is not the activation snapshot"
        )
    if metadata.get("project_root") != str(workspace):
        raise ArisActivationError("official installer manifest project root is incorrect")
    if metadata.get("packages") != "skills-codex":
        raise ArisActivationError("official installer manifest package set is incorrect")
    seen: set[str] = set()
    for kind, name, source_rel, target_rel, mode in rows:
        if name in seen:
            raise ArisActivationError("official installer manifest contains a duplicate entry")
        seen.add(name)
        expected_kind = "support" if name == "shared-references" else "skill"
        if name != "shared-references" and name not in expected_names:
            raise ArisActivationError(f"official installer manifest contains unknown skill: {name}")
        expected_source = f"skills/skills-codex/{name}"
        expected_target_rel = f".agents/skills/{name}"
        if (kind, source_rel, target_rel, mode) != (
            expected_kind,
            expected_source,
            expected_target_rel,
            "symlink",
        ):
            raise ArisActivationError(f"official installer manifest row is invalid: {name}")
        target = workspace / target_rel
        expected_target = activation / source_rel
        if not target.is_symlink():
            raise ArisActivationError(f"installed ARIS entry is not a symlink: {name}")
        try:
            actual_resolved = target.resolve(strict=True)
            expected_resolved = expected_target.resolve(strict=True)
        except OSError as exc:
            raise ArisActivationError(f"installed ARIS entry has a broken target: {name}") from exc
        if actual_resolved != expected_resolved:
            raise ArisActivationError(f"installed ARIS entry targets the wrong snapshot: {name}")
    if seen != expected_names | {"shared-references"}:
        raise ArisActivationError("official installer manifest is not the exact 82-entry inventory")
    if len(rows) != ACTIVATION_ENTRY_COUNT:
        raise ArisActivationError("official installer manifest does not contain 82 entries")
    return _sha256_bytes(manifest_path.read_bytes()), len(rows)


def _state_path(workspace: Path) -> Path:
    return workspace / ".aris" / "autoacademic-activation.json"


def _expected_state(
    lock: ArisLock,
    profile: CapabilityProfile,
    snapshot: Path,
    activation_manifest: ActivationManifest,
    official_manifest_sha256: str,
    rules_sha256: str,
) -> ActivationState:
    return ActivationState(
        schema_version=1,
        vendor_profile_sha256=lock.digest(),
        capability_profile_sha256=profile.digest(),
        activation_snapshot=snapshot,
        activation_tree_sha256=activation_manifest.activation_tree_sha256,
        official_manifest_sha256=official_manifest_sha256,
        deny_rules_sha256=rules_sha256,
        entry_count=ACTIVATION_ENTRY_COUNT,
        native_skills=profile.native_skills,
        blocked_skills=profile.blocked_skills,
        forbidden_commands=profile.forbidden_commands,
    )


def _load_state(workspace: Path) -> ActivationState:
    path = _state_path(workspace)
    if not path.is_file() or path.is_symlink():
        raise ArisActivationError("ARIS activation state is missing or not a regular file")
    try:
        return ActivationState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ArisActivationError("ARIS activation state is invalid") from exc


def _write_state(workspace: Path, state: ActivationState) -> None:
    path = _state_path(workspace)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(state.model_dump_json() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rollback_install(installer: Path, workspace: Path, a1_snapshot: Path, home: Path) -> None:
    try:
        _run_installer(
            _installer_argv(
                installer,
                workspace,
                a1_snapshot,
                "--uninstall",
                "--quiet",
                "--no-doc",
            ),
            home=home,
            capture_output=True,
        )
    except ArisActivationError:
        pass


def install_profile(lock: ArisLock, profile: CapabilityProfile, workspace: Path) -> ActivationState:
    _assert_managed_paths_safe(workspace)
    _check_rules_compatible(workspace, profile)
    state_path = _state_path(workspace)
    manifest_path = workspace / ".aris" / "installed-skills-codex.txt"
    if state_path.exists() or state_path.is_symlink():
        return preflight(lock, profile, workspace)
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ArisActivationError("an unmanaged ARIS installer manifest already exists")
    snapshot = create_activation_snapshot(lock, profile, workspace)
    activation_manifest = verify_activation_snapshot(lock, profile, workspace, snapshot)
    a1_snapshot = workspace / ".aris" / "vendor" / "aris" / lock.digest()
    installer = a1_snapshot / "tools" / "install_aris_codex.sh"
    home = workspace / ".aris" / "installer-home"
    home.mkdir(parents=True, exist_ok=True)
    installed = False
    rules_path: Path | None = None
    try:
        _run_installer(
            _installer_argv(
                installer,
                workspace,
                snapshot,
                "--all",
                "--quiet",
                "--no-doc",
            ),
            home=home,
            capture_output=True,
        )
        installed = True
        official_sha256, _ = _validate_official_install(workspace, snapshot, profile)
        rules_path = _write_rules(workspace, profile)
        rules_sha256 = _sha256_bytes(rules_path.read_bytes())
        state = _expected_state(
            lock,
            profile,
            snapshot,
            activation_manifest,
            official_sha256,
            rules_sha256,
        )
        _write_state(workspace, state)
        return preflight(lock, profile, workspace)
    except Exception:
        if installed or manifest_path.is_file():
            _rollback_install(installer, workspace, a1_snapshot, home)
        _state_path(workspace).unlink(missing_ok=True)
        if rules_path is not None and rules_path.is_file():
            if rules_path.read_bytes() == _rules_bytes(profile):
                rules_path.unlink()
        raise


def preflight(lock: ArisLock, profile: CapabilityProfile, workspace: Path) -> ActivationState:
    _assert_managed_paths_safe(workspace)
    snapshot = workspace / ".aris" / "vendor" / "aris-activation" / profile.digest()
    activation_manifest = verify_activation_snapshot(lock, profile, workspace, snapshot)
    official_sha256, _ = _validate_official_install(workspace, snapshot, profile)
    rules_path = workspace / ".codex" / "rules" / "aris-deny.rules"
    if not rules_path.is_file() or rules_path.is_symlink():
        raise ArisActivationError("Stage A2 deny rules are missing or not a regular file")
    rules = rules_path.read_bytes()
    if rules != _rules_bytes(profile):
        raise ArisActivationError("Stage A2 deny rules do not match the profile")
    expected = _expected_state(
        lock,
        profile,
        snapshot,
        activation_manifest,
        official_sha256,
        _sha256_bytes(rules),
    )
    actual = _load_state(workspace)
    if actual != expected:
        raise ArisActivationError("ARIS activation state does not match the installed profile")
    return actual


def uninstall_profile(lock: ArisLock, profile: CapabilityProfile, workspace: Path) -> None:
    _assert_managed_paths_safe(workspace)
    snapshot = workspace / ".aris" / "vendor" / "aris-activation" / profile.digest()
    activation_manifest = verify_activation_snapshot(lock, profile, workspace, snapshot)
    official_sha256, _ = _validate_official_install(workspace, snapshot, profile)
    state = _load_state(workspace)
    expected_state = _expected_state(
        lock,
        profile,
        snapshot,
        activation_manifest,
        official_sha256,
        _sha256_bytes(_rules_bytes(profile)),
    )
    if state != expected_state:
        raise ArisActivationError("ARIS activation state does not match the installed profile")
    a1_snapshot = workspace / ".aris" / "vendor" / "aris" / lock.digest()
    installer = a1_snapshot / "tools" / "install_aris_codex.sh"
    home = workspace / ".aris" / "installer-home"
    _run_installer(
        _installer_argv(
            installer,
            workspace,
            a1_snapshot,
            "--uninstall",
            "--quiet",
            "--no-doc",
        ),
        home=home,
        capture_output=True,
    )
    rules_path = workspace / ".codex" / "rules" / "aris-deny.rules"
    if rules_path.is_file() and not rules_path.is_symlink():
        if _sha256_bytes(rules_path.read_bytes()) == state.deny_rules_sha256:
            rules_path.unlink()
    _state_path(workspace).unlink()
    verify_activation_snapshot(lock, profile, workspace, snapshot)
