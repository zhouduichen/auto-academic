from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
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
