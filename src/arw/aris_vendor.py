from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TreeEntry = tuple[bytes, bytes, bytes, bytes]


class ArisVendorError(RuntimeError):
    pass


class ArisLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    repo_path: Path
    origin: str
    commit: Commit
    license_spdx: Literal["MIT"]
    inventory_count: int = Field(ge=1)
    shared_reference_count: int = Field(ge=1)
    inventory_sha256: Sha256
    source_tree_sha256: Sha256
    activation_mode: Literal["disabled"]
    unknown_skill_policy: Literal["block"]

    def digest(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


class VendorReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_names: tuple[str, ...]
    inventory_sha256: Sha256
    source_tree_sha256: Sha256
    profile_sha256: Sha256


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    aris_origin: str
    aris_commit: Commit
    profile_sha256: Sha256
    inventory_count: int = Field(ge=1)
    shared_reference_count: int = Field(ge=1)
    inventory_sha256: Sha256
    source_tree_sha256: Sha256
    activation_mode: Literal["disabled"]
    unknown_skill_policy: Literal["block"]


def load_lock(path: Path) -> ArisLock:
    return ArisLock.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _run(repo: Path, *argv: str) -> str:
    try:
        return subprocess.run(  # noqa: S603
            list(argv), cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "command failed"
        raise ArisVendorError(detail) from exc


def _run_bytes(repo: Path, *argv: str, input_data: bytes | None = None) -> bytes:
    try:
        return subprocess.run(  # noqa: S603
            list(argv), cwd=repo, input=input_data, check=True, capture_output=True
        ).stdout
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip()
        stdout = exc.stdout.decode(errors="replace").strip()
        raise ArisVendorError(stderr or stdout or "command failed") from exc


def _digest_entries(entries: list[TreeEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item[0]):
        for field in entry:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _git_source_tree_digest(repo: Path, commit: str) -> str:
    listing = _run_bytes(repo, "git", "ls-tree", "-r", "-t", "-z", "--full-tree", commit)
    records = [record for record in listing.split(b"\0") if record]
    metadata: list[tuple[bytes, bytes, bytes, bytes]] = []
    blob_ids: list[bytes] = []
    for record in records:
        try:
            header, path = record.split(b"\t", 1)
            mode, object_type, object_id = header.split(b" ", 2)
        except ValueError as exc:
            raise ArisVendorError("invalid Git tree entry") from exc
        if object_type == b"tree" and mode == b"040000":
            metadata.append((path, object_type, mode, b""))
        elif object_type == b"blob" and mode in {b"100644", b"100755"}:
            metadata.append((path, object_type, mode, object_id))
            blob_ids.append(object_id)
        else:
            raise ArisVendorError(f"unsupported Git tree entry: {path!r}")

    batch = _run_bytes(
        repo,
        "git",
        "cat-file",
        "--batch",
        input_data=b"".join(object_id + b"\n" for object_id in blob_ids),
    )
    contents: dict[bytes, bytes] = {}
    offset = 0
    for expected_id in blob_ids:
        try:
            header_end = batch.index(b"\n", offset)
            object_id, object_type, size_bytes = batch[offset:header_end].split(b" ", 2)
            size = int(size_bytes)
        except (ValueError, OverflowError) as exc:
            raise ArisVendorError("invalid git cat-file response") from exc
        offset = header_end + 1
        content = batch[offset : offset + size]
        offset += size
        if (
            object_id != expected_id
            or object_type != b"blob"
            or len(content) != size
            or batch[offset : offset + 1] != b"\n"
        ):
            raise ArisVendorError("Git blob response does not match the tree")
        offset += 1
        contents[object_id] = content
    if offset != len(batch):
        raise ArisVendorError("unexpected trailing Git blob data")

    entries = [
        (path, object_type, mode, contents[content] if object_type == b"blob" else content)
        for path, object_type, mode, content in metadata
    ]
    return _digest_entries(entries)


def verify_repo(lock: ArisLock) -> VendorReport:
    repo = lock.repo_path
    if not repo.is_dir() or repo.is_symlink():
        raise ArisVendorError(f"ARIS repo is not a real directory: {repo}")
    if _run(repo, "git", "remote", "get-url", "origin") != lock.origin:
        raise ArisVendorError("ARIS origin does not match the lock")
    if _run(repo, "git", "rev-parse", "HEAD") != lock.commit:
        raise ArisVendorError("ARIS commit does not match the lock")
    if _run(repo, "git", "status", "--porcelain=v1", "--untracked-files=all"):
        raise ArisVendorError("ARIS clone is not clean")
    if not (repo / "LICENSE").read_text(encoding="utf-8").startswith("MIT License"):
        raise ArisVendorError("ARIS license is not MIT")
    _run(repo, sys.executable, "tools/check_skills_inventory.py")
    mainline = {path.parent.name for path in (repo / "skills").glob("*/SKILL.md")}
    codex = {path.parent.name for path in (repo / "skills" / "skills-codex").glob("*/SKILL.md")}
    if mainline != codex:
        raise ArisVendorError("ARIS mainline and Codex inventories differ")
    names = tuple(sorted(codex))
    encoded = "".join(f"{name}\n" for name in names).encode()
    inventory_digest = hashlib.sha256(encoded).hexdigest()
    references = tuple((repo / "skills" / "shared-references").glob("*.md"))
    if len(names) != lock.inventory_count or len(references) != lock.shared_reference_count:
        raise ArisVendorError("ARIS inventory count does not match the lock")
    if inventory_digest != lock.inventory_sha256:
        raise ArisVendorError("ARIS inventory digest does not match the lock")
    source_digest = _git_source_tree_digest(repo, lock.commit)
    if source_digest != lock.source_tree_sha256:
        raise ArisVendorError("ARIS source tree digest does not match the lock")
    return VendorReport(
        skill_names=names,
        inventory_sha256=inventory_digest,
        source_tree_sha256=source_digest,
        profile_sha256=lock.digest(),
    )


def _snapshot_source_tree_digest(root: Path) -> str:
    entries: list[TreeEntry] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == ".autoacademic" or relative.startswith(".autoacademic/"):
            continue
        info = path.lstat()
        encoded_path = relative.encode()
        if stat.S_ISDIR(info.st_mode):
            entries.append((encoded_path, b"tree", b"040000", b""))
        elif stat.S_ISREG(info.st_mode):
            mode = b"100755" if info.st_mode & 0o111 else b"100644"
            entries.append((encoded_path, b"blob", mode, path.read_bytes()))
        else:
            raise ArisVendorError(f"unsupported snapshot entry: {relative}")
    return _digest_entries(entries)


def _remove_write_bits(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~0o222)


def _assert_read_only(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        if path.lstat().st_mode & 0o222:
            raise ArisVendorError(f"snapshot entry is writable: {path.relative_to(root)}")


def _vendor_root(workspace: Path) -> Path:
    vendor = workspace / ".aris" / "vendor" / "aris"
    for path in (workspace / ".aris", workspace / ".aris" / "vendor", vendor):
        if path.is_symlink():
            raise ArisVendorError(f"vendor path contains a symlink: {path}")
        if path.exists() and not path.is_dir():
            raise ArisVendorError(f"vendor path is not a directory: {path}")
    return vendor


def _expected_manifest(lock: ArisLock) -> SnapshotManifest:
    return SnapshotManifest(
        schema_version=1,
        aris_origin=lock.origin,
        aris_commit=lock.commit,
        profile_sha256=lock.digest(),
        inventory_count=lock.inventory_count,
        shared_reference_count=lock.shared_reference_count,
        inventory_sha256=lock.inventory_sha256,
        source_tree_sha256=lock.source_tree_sha256,
        activation_mode=lock.activation_mode,
        unknown_skill_policy=lock.unknown_skill_policy,
    )


def create_snapshot(lock: ArisLock, workspace: Path) -> Path:
    if not workspace.is_dir() or workspace.is_symlink():
        raise ArisVendorError(f"workspace is not a real directory: {workspace}")
    report = verify_repo(lock)
    vendor = _vendor_root(workspace)
    destination = vendor / report.profile_sha256
    if destination.exists() or destination.is_symlink():
        verify_snapshot(lock, destination)
        return destination
    vendor.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=vendor))
    archive = temporary / "aris.tar"
    root = temporary / "root"
    root.mkdir()
    try:
        _run(
            lock.repo_path,
            "git",
            "archive",
            "--format=tar",
            f"--output={archive}",
            lock.commit,
        )
        with tarfile.open(archive) as handle:
            handle.extractall(root, filter="data")
        if (root / ".autoacademic").exists() or (root / ".autoacademic").is_symlink():
            raise ArisVendorError("ARIS archive reserves the .autoacademic path")
        if _snapshot_source_tree_digest(root) != report.source_tree_sha256:
            raise ArisVendorError("ARIS archive source tree digest does not match the lock")
        metadata = root / ".autoacademic"
        metadata.mkdir()
        manifest = _expected_manifest(lock)
        (metadata / "vendor.json").write_text(manifest.model_dump_json() + "\n", encoding="utf-8")
        _remove_write_bits(root)
        os.replace(root, destination)
        destination.chmod(destination.stat().st_mode & ~0o222)
        verify_snapshot(lock, destination)
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def verify_snapshot(lock: ArisLock, snapshot: Path) -> SnapshotManifest:
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ArisVendorError("snapshot is not a real directory")
    if snapshot.name != lock.digest():
        raise ArisVendorError("snapshot path does not match the profile digest")
    try:
        manifest = SnapshotManifest.model_validate_json(
            (snapshot / ".autoacademic" / "vendor.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ArisVendorError("snapshot manifest is invalid") from exc
    if manifest != _expected_manifest(lock):
        raise ArisVendorError("snapshot manifest does not match the lock")
    if _snapshot_source_tree_digest(snapshot) != lock.source_tree_sha256:
        raise ArisVendorError("snapshot source tree digest does not match the lock")
    _assert_read_only(snapshot)
    return manifest
