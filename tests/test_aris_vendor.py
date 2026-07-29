import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from arw.aris_vendor import (
    ArisVendorError,
    create_snapshot,
    load_lock,
    verify_repo,
    verify_snapshot,
)

pytestmark = pytest.mark.stage_a1
LOCK = Path(__file__).parents[1] / "configs" / "integrations" / "aris.yaml"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture_source_digest(repo: Path, commit: str) -> str:
    listing = subprocess.run(  # noqa: S603
        ["git", "ls-tree", "-r", "-t", "-z", "--full-tree", commit],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    entries: list[tuple[bytes, bytes, bytes, bytes]] = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        header, path = record.split(b"\t", 1)
        mode, object_type, _ = header.split(b" ", 2)
        content = b"" if object_type == b"tree" else (repo / path.decode()).read_bytes()
        entries.append((path, object_type, mode, content))
    digest = hashlib.sha256()
    for entry in sorted(entries):
        for field in entry:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _fixture_repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    repo = tmp_path / "aris"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "ARW Test")
    _git(repo, "config", "user.email", "arw-test@example.invalid")
    _git(repo, "remote", "add", "origin", "https://example.invalid/aris.git")
    (repo / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    for name in ("alpha", "beta"):
        for root in (repo / "skills", repo / "skills" / "skills-codex"):
            skill = root / name
            skill.mkdir(parents=True, exist_ok=True)
            (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
    for index in range(2):
        for root in (
            repo / "skills" / "shared-references",
            repo / "skills" / "skills-codex" / "shared-references",
        ):
            root.mkdir(parents=True, exist_ok=True)
            (root / f"ref-{index}.md").write_text("reference\n", encoding="utf-8")
    tools = repo / "tools"
    tools.mkdir()
    (tools / "check_skills_inventory.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    inventory_digest = hashlib.sha256(b"alpha\nbeta\n").hexdigest()
    return repo, commit, inventory_digest, _fixture_source_digest(repo, commit)


def _fixture_lock(
    tmp_path: Path,
    repo: Path,
    commit: str,
    inventory_digest: str,
    source_digest: str,
) -> Path:
    path = tmp_path / "aris.yaml"
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"repo_path: {repo}",
                "origin: https://example.invalid/aris.git",
                f"commit: {commit}",
                "license_spdx: MIT",
                "inventory_count: 2",
                "shared_reference_count: 2",
                f"inventory_sha256: '{inventory_digest}'",
                f"source_tree_sha256: '{source_digest}'",
                "activation_mode: disabled",
                "unknown_skill_policy: block",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_lock_pins_full_but_inactive_aris() -> None:
    document = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    assert document["repo_path"] == "/Users/huangjiahao/自动化科研/aris"
    assert document["commit"] == "53562a7c64cc1d55946cba1fb8a8416137143d14"
    assert document["inventory_count"] == 81
    assert document["shared_reference_count"] == 30
    assert document["activation_mode"] == "disabled"
    assert document["unknown_skill_policy"] == "block"


def test_verify_repo_delegates_inventory_and_checks_pin(tmp_path: Path) -> None:
    repo, commit, inventory_digest, source_digest = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, inventory_digest, source_digest))
    report = verify_repo(lock)
    assert report.skill_names == ("alpha", "beta")
    assert report.inventory_sha256 == inventory_digest
    assert report.source_tree_sha256 == source_digest


def test_verify_repo_rejects_dirty_content(tmp_path: Path) -> None:
    repo, commit, inventory_digest, source_digest = _fixture_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, inventory_digest, source_digest))
    with pytest.raises(ArisVendorError, match="not clean"):
        verify_repo(lock)


def test_verify_repo_rejects_source_tree_drift(tmp_path: Path) -> None:
    repo, commit, inventory_digest, _ = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, inventory_digest, "0" * 64))
    with pytest.raises(ArisVendorError, match="source tree digest"):
        verify_repo(lock)


def test_snapshot_is_complete_read_only_and_inactive(tmp_path: Path) -> None:
    repo, commit, inventory_digest, source_digest = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, inventory_digest, source_digest))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = create_snapshot(lock, workspace)
    manifest = verify_snapshot(lock, snapshot)
    assert snapshot.name == lock.digest()
    assert manifest.aris_commit == commit
    assert manifest.source_tree_sha256 == source_digest
    assert manifest.activation_mode == "disabled"
    assert (snapshot / "skills" / "skills-codex" / "alpha" / "SKILL.md").is_file()
    assert not (workspace / ".agents").exists()


def test_snapshot_rejects_content_tampering(tmp_path: Path) -> None:
    repo, commit, inventory_digest, source_digest = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, inventory_digest, source_digest))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = create_snapshot(lock, workspace)
    target = snapshot / "skills" / "skills-codex" / "alpha" / "SKILL.md"
    target.chmod(0o644)
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ArisVendorError, match="source tree digest"):
        verify_snapshot(lock, snapshot)


def test_snapshot_rejects_manifest_drift(tmp_path: Path) -> None:
    repo, commit, inventory_digest, source_digest = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, inventory_digest, source_digest))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = create_snapshot(lock, workspace)
    manifest_path = snapshot / ".autoacademic" / "vendor.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["aris_commit"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArisVendorError, match="manifest does not match"):
        verify_snapshot(lock, snapshot)


def test_snapshot_rejects_symlinked_vendor_parent(tmp_path: Path) -> None:
    repo, commit, inventory_digest, source_digest = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, inventory_digest, source_digest))
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / ".aris").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArisVendorError, match="symlink"):
        create_snapshot(lock, workspace)
