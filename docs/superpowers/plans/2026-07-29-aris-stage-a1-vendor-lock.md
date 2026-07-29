# ARIS Stage A1 Vendor Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the complete ARIS repository to its formal sibling path, verify its exact upstream identity with ARIS's own checks, and create one content-addressed but deliberately inactive snapshot.

**Architecture:** Stage A1 does not install project-local skills. AutoAcademic adds one strict profile and one small Python module around Git plus ARIS's existing `check_skills_inventory.py`; `git archive` produces a byte-reused snapshot named by the complete profile digest and verified against an external source-tree lock. Because `.agents/skills` is not created, all 81 skills remain unavailable until the separately reviewed Stage A2 activation broker exists.

**Tech Stack:** Python 3.12 standard library, existing Pydantic/PyYAML/Typer dependencies, Git, ARIS upstream tests, pytest, ruff, mypy.

## Global Constraints

- Formal ARIS path is `/Users/huangjiahao/自动化科研/aris`.
- Origin is `https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git`.
- Commit is `53562a7c64cc1d55946cba1fb8a8416137143d14`.
- License is MIT.
- Codex inventory is 81 skill names; sorted-name SHA-256 is `0d0bd0f7113ca64afbe671664fcc901c472d28b857f310d6b5d10f792c094c93`.
- The committed source-tree SHA-256 (length-framed `path`, `type`, `mode`, and `content`) is `0e05726b74febb1ae926c5baece83b8d98a4b280fdd2c1e7bf5253959cbc176b`.
- Mainline and Codex skill names must match; mainline shared references count is 30.
- Activation mode is `disabled`; unknown skill policy is `block`.
- Do not edit the ARIS or Karpathy clone, create `.agents/skills`, invoke the ARIS installer, install globally, access Windows, start a process in the background, or run training.
- Default AutoAcademic tests use a temporary local Git fixture and do not require the formal clone or network.
- Reuse ARIS's `tools/check_skills_inventory.py` and four pinned upstream test files; do not reproduce their mirror/catalog/install assertions.

---

### Task 1: Formal clone placement and declarative lock

**Files:**
- Move without editing: `/Users/huangjiahao/Documents/Codex/2026-07-29/an/work/vendor-eval/aris` → `/Users/huangjiahao/自动化科研/aris`
- Create: `/Users/huangjiahao/自动化科研/auto-academic/configs/integrations/aris.yaml`
- Create: `/Users/huangjiahao/自动化科研/auto-academic/tests/test_aris_vendor.py`
- Modify: `/Users/huangjiahao/自动化科研/auto-academic/pyproject.toml`

**Interfaces:**
- Consumes: the already evaluated clean ARIS clone.
- Produces: a strict, reviewable lock used by Tasks 2–4.

- [ ] **Step 1: Verify and move the complete clone**

```bash
test ! -e '/Users/huangjiahao/自动化科研/aris'
test "$(git -C '/Users/huangjiahao/Documents/Codex/2026-07-29/an/work/vendor-eval/aris' remote get-url origin)" = 'https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git'
test "$(git -C '/Users/huangjiahao/Documents/Codex/2026-07-29/an/work/vendor-eval/aris' rev-parse HEAD)" = '53562a7c64cc1d55946cba1fb8a8416137143d14'
test -z "$(git -C '/Users/huangjiahao/Documents/Codex/2026-07-29/an/work/vendor-eval/aris' status --porcelain=v1 --untracked-files=all)"
mv '/Users/huangjiahao/Documents/Codex/2026-07-29/an/work/vendor-eval/aris' '/Users/huangjiahao/自动化科研/aris'
```

Expected: the full clean repository is at the formal path with unchanged origin and commit.

- [ ] **Step 2: Write the failing lock test**

Create `tests/test_aris_vendor.py`:

```python
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.stage_a1
LOCK = Path(__file__).parents[1] / "configs" / "integrations" / "aris.yaml"


def test_lock_pins_full_but_inactive_aris() -> None:
    document = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    assert document == {
        "schema_version": 1,
        "repo_path": "/Users/huangjiahao/自动化科研/aris",
        "origin": "https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git",
        "commit": "53562a7c64cc1d55946cba1fb8a8416137143d14",
        "license_spdx": "MIT",
        "inventory_count": 81,
        "shared_reference_count": 30,
        "inventory_sha256": (
            "0d0bd0f7113ca64afbe671664fcc901c472d28b857f310d6b5d10f792c094c93"
        ),
        "source_tree_sha256": (
            "0e05726b74febb1ae926c5baece83b8d98a4b280fdd2c1e7bf5253959cbc176b"
        ),
        "activation_mode": "disabled",
        "unknown_skill_policy": "block",
    }
```

- [ ] **Step 3: Run the red test**

Run: `uv run pytest tests/test_aris_vendor.py -q`

Expected: FAIL with `FileNotFoundError` for the lock file.

- [ ] **Step 4: Add the lock and marker**

Create `configs/integrations/aris.yaml`:

```yaml
schema_version: 1
repo_path: /Users/huangjiahao/自动化科研/aris
origin: https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git
commit: 53562a7c64cc1d55946cba1fb8a8416137143d14
license_spdx: MIT
inventory_count: 81
shared_reference_count: 30
inventory_sha256: 0d0bd0f7113ca64afbe671664fcc901c472d28b857f310d6b5d10f792c094c93
source_tree_sha256: 0e05726b74febb1ae926c5baece83b8d98a4b280fdd2c1e7bf5253959cbc176b
activation_mode: disabled
unknown_skill_policy: block
```

Extend the existing `[tool.pytest.ini_options]` table in `pyproject.toml`:

```toml
markers = [
  "stage_a1: pinned inactive ARIS vendor snapshot gate",
]
```

- [ ] **Step 5: Run the green test and commit**

```bash
uv run pytest tests/test_aris_vendor.py -q
git add configs/integrations/aris.yaml tests/test_aris_vendor.py pyproject.toml
git commit -m 'chore: lock complete ARIS upstream'
```

Expected: the test passes and the commit contains no ARIS source files.

---

### Task 2: Thin verifier that delegates ARIS semantics upstream

**Files:**
- Create: `/Users/huangjiahao/自动化科研/auto-academic/src/arw/aris_vendor.py`
- Modify: `/Users/huangjiahao/自动化科研/auto-academic/tests/test_aris_vendor.py`

**Interfaces:**
- Consumes: `configs/integrations/aris.yaml`.
- Produces: `load_lock(path: Path) -> ArisLock`, `verify_repo(lock: ArisLock) -> VendorReport`, and `ArisVendorError`.

- [ ] **Step 1: Add failing verifier tests**

Merge these imports into the existing import block:

```python
import hashlib
import subprocess

from arw.aris_vendor import ArisVendorError, load_lock, verify_repo
```

Append:

```python
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603,S607 -- isolated local Git fixture
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fixture_tree_digest(repo: Path) -> str:
    digest = hashlib.sha256()
    names = _git(repo, "ls-files", "-z").split("\0")
    for name in sorted(item for item in names if item):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((repo / name).read_bytes())
    return digest.hexdigest()


def _fixture_repo(tmp_path: Path, *, checker_exit: int = 0) -> tuple[Path, str, str, str]:
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
    (tools / "check_skills_inventory.py").write_text(
        f"raise SystemExit({checker_exit})\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    digest = hashlib.sha256(b"alpha\nbeta\n").hexdigest()
    return repo, commit, digest, _fixture_tree_digest(repo)


def _fixture_lock(
    tmp_path: Path, repo: Path, commit: str, digest: str, tree_digest: str
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
                f"inventory_sha256: {digest}",
                f"source_tree_sha256: {tree_digest}",
                "activation_mode: disabled",
                "unknown_skill_policy: block",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_verify_repo_delegates_inventory_and_checks_pin(tmp_path: Path) -> None:
    repo, commit, digest, tree_digest = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, digest, tree_digest))
    report = verify_repo(lock)
    assert report.skill_names == ("alpha", "beta")
    assert report.inventory_sha256 == digest


def test_verify_repo_rejects_dirty_or_drifted_content(tmp_path: Path) -> None:
    repo, commit, digest, tree_digest = _fixture_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, digest, tree_digest))
    with pytest.raises(ArisVendorError, match="not clean"):
        verify_repo(lock)


def test_verify_repo_fails_closed_when_upstream_checker_fails(tmp_path: Path) -> None:
    repo, commit, digest, tree_digest = _fixture_repo(tmp_path, checker_exit=9)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, digest, tree_digest))
    with pytest.raises(ArisVendorError):
        verify_repo(lock)


@pytest.mark.parametrize("field", ["origin", "commit", "inventory_sha256", "source_tree_sha256"])
def test_verify_repo_rejects_each_drifted_pin(tmp_path: Path, field: str) -> None:
    repo, commit, digest, tree_digest = _fixture_repo(tmp_path)
    lock_path = _fixture_lock(tmp_path, repo, commit, digest, tree_digest)
    document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    document[field] = "0" * (40 if field == "commit" else 64) if field != "origin" else "https://wrong.invalid/repo.git"
    lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ArisVendorError):
        verify_repo(load_lock(lock_path))
```

- [ ] **Step 2: Run the red verifier tests**

Run: `uv run pytest tests/test_aris_vendor.py -q`

Expected: collection FAIL because `arw.aris_vendor` does not exist.

- [ ] **Step 3: Implement the verifier**

Implementation note: the source-tree digest is computed from the pinned Git commit, not
from the mutable working tree. It length-frames every tracked entry's path, type, mode,
and content before hashing, and compares the result to `source_tree_sha256` in the lock.

Create `src/arw/aris_vendor.py`:

```python
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


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


def load_lock(path: Path) -> ArisLock:
    return ArisLock.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _run(repo: Path, *argv: str) -> str:
    try:
        return subprocess.run(  # noqa: S603,S607 -- fixed Git/Python executables only
            list(argv), cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "command failed"
        raise ArisVendorError(detail) from exc


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
    tracked = tuple(
        sorted(name for name in _run(repo, "git", "ls-files", "-z").split("\0") if name)
    )
    if any((repo / name).is_symlink() or not (repo / name).is_file() for name in tracked):
        raise ArisVendorError("ARIS tracked tree contains a non-regular file")
    tree_digest = hashlib.sha256()
    for name in tracked:
        tree_digest.update(name.encode())
        tree_digest.update(b"\0")
        tree_digest.update((repo / name).read_bytes())
    source_tree_sha256 = tree_digest.hexdigest()
    mainline = {path.parent.name for path in (repo / "skills").glob("*/SKILL.md")}
    codex = {path.parent.name for path in (repo / "skills" / "skills-codex").glob("*/SKILL.md")}
    if mainline != codex:
        raise ArisVendorError("ARIS mainline and Codex inventories differ")
    names = tuple(sorted(codex))
    encoded = "".join(f"{name}\n" for name in names).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    references = tuple((repo / "skills" / "shared-references").glob("*.md"))
    codex_references = tuple(
        (repo / "skills" / "skills-codex" / "shared-references").glob("*.md")
    )
    if (
        len(names) != lock.inventory_count
        or len(references) != lock.shared_reference_count
        or len(codex_references) != lock.shared_reference_count
    ):
        raise ArisVendorError("ARIS inventory count does not match the lock")
    if digest != lock.inventory_sha256:
        raise ArisVendorError("ARIS inventory digest does not match the lock")
    if source_tree_sha256 != lock.source_tree_sha256:
        raise ArisVendorError("ARIS source tree digest does not match the lock")
    return VendorReport(
        skill_names=names,
        inventory_sha256=digest,
        source_tree_sha256=source_tree_sha256,
        profile_sha256=lock.digest(),
    )
```

- [ ] **Step 4: Run verifier tests and quality**

```bash
uv run pytest tests/test_aris_vendor.py -q
make quality
```

Expected: all tests pass; no dependency is added.

- [ ] **Step 5: Commit**

```bash
git add src/arw/aris_vendor.py tests/test_aris_vendor.py
git commit -m 'feat: verify complete ARIS vendor lock'
```

---

### Task 3: Content-addressed inactive snapshot

**Files:**
- Modify: `/Users/huangjiahao/自动化科研/auto-academic/src/arw/aris_vendor.py`
- Modify: `/Users/huangjiahao/自动化科研/auto-academic/tests/test_aris_vendor.py`

**Interfaces:**
- Consumes: verified `ArisLock` and `VendorReport`.
- Produces: `create_snapshot(lock: ArisLock, workspace: Path) -> Path` and `verify_snapshot(lock: ArisLock, snapshot: Path) -> SnapshotManifest`.

- [ ] **Step 1: Add failing snapshot tests**

Merge `create_snapshot` and `verify_snapshot` into the existing import from `arw.aris_vendor`, then append:

```python
def test_snapshot_is_profile_addressed_complete_and_inactive(tmp_path: Path) -> None:
    repo, commit, digest, tree_digest = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, digest, tree_digest))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = create_snapshot(lock, workspace)
    manifest = verify_snapshot(lock, snapshot)
    assert snapshot.name == lock.digest()
    assert manifest.activation_mode == "disabled"
    assert (snapshot / "skills" / "skills-codex" / "alpha" / "SKILL.md").is_file()
    assert not (workspace / ".agents").exists()


def test_snapshot_rejects_content_tampering(tmp_path: Path) -> None:
    repo, commit, digest, tree_digest = _fixture_repo(tmp_path)
    lock = load_lock(_fixture_lock(tmp_path, repo, commit, digest, tree_digest))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = create_snapshot(lock, workspace)
    target = snapshot / "skills" / "skills-codex" / "alpha" / "SKILL.md"
    target.chmod(0o644)
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ArisVendorError, match="snapshot digest"):
        verify_snapshot(lock, snapshot)
```

- [ ] **Step 2: Run the red snapshot tests**

Run: `uv run pytest tests/test_aris_vendor.py -q`

Expected: FAIL because snapshot functions are undefined.

- [ ] **Step 3: Implement the minimal snapshot publisher**

Implementation note: the shipped `SnapshotManifest` is typed and contains every locked
identity/count/policy field. Publication rejects symlink parents under
`workspace/.aris/vendor/aris`, checks the archive against the external source-tree digest,
removes write bits, atomically publishes, and then calls `verify_snapshot`.

Add imports to `src/arw/aris_vendor.py`:

```python
import os
import shutil
import tarfile
import tempfile
```

Add:

```python
class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    aris_commit: Commit
    profile_sha256: Sha256
    inventory_sha256: Sha256
    activation_mode: Literal["disabled"]
    tree_sha256: Sha256


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == ".autoacademic/vendor.json":
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _remove_write_bits(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def create_snapshot(lock: ArisLock, workspace: Path) -> Path:
    if not workspace.is_dir() or workspace.is_symlink():
        raise ArisVendorError(f"workspace is not a real directory: {workspace}")
    report = verify_repo(lock)
    vendor = workspace / ".aris" / "vendor" / "aris"
    destination = vendor / report.profile_sha256
    if destination.exists():
        verify_snapshot(lock, destination)
        return destination
    vendor.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=vendor))
    archive = temporary / "aris.tar"
    root = temporary / "root"
    root.mkdir()
    try:
        _run(lock.repo_path, "git", "archive", "--format=tar", f"--output={archive}", lock.commit)
        with tarfile.open(archive) as handle:
            handle.extractall(root, filter="data")
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ArisVendorError("ARIS archive contains a symlink")
        metadata = root / ".autoacademic"
        metadata.mkdir()
        manifest = {
            "schema_version": 1,
            "aris_commit": lock.commit,
            "profile_sha256": report.profile_sha256,
            "inventory_sha256": report.inventory_sha256,
            "activation_mode": "disabled",
            "tree_sha256": report.source_tree_sha256,
        }
        (metadata / "vendor.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        _remove_write_bits(root)
        os.replace(root, destination)
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def verify_snapshot(lock: ArisLock, snapshot: Path) -> SnapshotManifest:
    if snapshot.name != lock.digest():
        raise ArisVendorError("snapshot path does not match the profile digest")
    manifest = SnapshotManifest.model_validate_json(
        (snapshot / ".autoacademic" / "vendor.json").read_text(encoding="utf-8")
    )
    expected = SnapshotManifest(
        schema_version=1,
        aris_commit=lock.commit,
        profile_sha256=lock.digest(),
        inventory_sha256=lock.inventory_sha256,
        activation_mode="disabled",
        tree_sha256=lock.source_tree_sha256,
    )
    if manifest != expected:
        raise ArisVendorError("snapshot profile digest does not match the lock")
    if manifest.tree_sha256 != _tree_digest(snapshot):
        raise ArisVendorError("snapshot digest does not match its manifest")
    return manifest
```

- [ ] **Step 4: Run snapshot tests and quality**

```bash
uv run pytest tests/test_aris_vendor.py -q
make quality
```

Expected: tests pass; no `.agents` directory is created.

- [ ] **Step 5: Commit**

```bash
git add src/arw/aris_vendor.py tests/test_aris_vendor.py
git commit -m 'feat: publish inactive ARIS vendor snapshot'
```

---

### Task 4: Minimal CLI and formal A1 gate

**Files:**
- Modify: `/Users/huangjiahao/自动化科研/auto-academic/src/arw/cli.py`
- Create: `/Users/huangjiahao/自动化科研/auto-academic/tests/test_aris_vendor_cli.py`
- Create outside Git: `/Users/huangjiahao/自动化科研/workspaces/karpathy-autoresearch`

**Interfaces:**
- Consumes: Task 2–3 functions.
- Produces: `arw aris verify` and `arw aris snapshot --workspace PATH --confirm`.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_aris_vendor_cli.py`:

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from arw.cli import app

pytestmark = pytest.mark.stage_a1
runner = CliRunner()


def test_snapshot_requires_confirmation(tmp_path: Path) -> None:
    result = runner.invoke(app, ["aris", "snapshot", "--workspace", str(tmp_path)])
    assert result.exit_code == 2
    assert "--confirm" in result.output


def test_aris_cli_exposes_no_install_or_run() -> None:
    result = runner.invoke(app, ["aris", "--help"])
    assert result.exit_code == 0
    assert "verify" in result.output
    assert "snapshot" in result.output
    assert "install" not in result.output
    assert "run" not in result.output
```

- [ ] **Step 2: Run the red CLI tests**

Run: `uv run pytest tests/test_aris_vendor_cli.py -q`

Expected: FAIL because the `aris` command group is absent.

- [ ] **Step 3: Add the two commands**

Replace `src/arw/cli.py` with:

```python
from pathlib import Path

import typer

from arw.aris_vendor import create_snapshot, load_lock, verify_repo

app = typer.Typer(no_args_is_help=True)
aris_app = typer.Typer(no_args_is_help=True)
app.add_typer(aris_app, name="aris")

DEFAULT_LOCK = Path(__file__).resolve().parents[2] / "configs" / "integrations" / "aris.yaml"


@app.callback()
def main() -> None:
    """Control AutoResearch Workbench from the Mac."""


@aris_app.command("verify")
def aris_verify(
    config: Path = typer.Option(DEFAULT_LOCK, exists=True, dir_okay=False),
) -> None:
    lock = load_lock(config)
    report = verify_repo(lock)
    typer.echo(f"ARIS vendor OK: {len(report.skill_names)} skills @ {lock.commit[:12]}")


@aris_app.command("snapshot")
def aris_snapshot(
    workspace: Path = typer.Option(..., exists=True, file_okay=False),
    confirm: bool = typer.Option(False, "--confirm"),
    config: Path = typer.Option(DEFAULT_LOCK, exists=True, dir_okay=False),
) -> None:
    if not confirm:
        raise typer.BadParameter("--confirm is required to create the snapshot")
    lock = load_lock(config)
    snapshot = create_snapshot(lock, workspace)
    typer.echo(f"Inactive ARIS snapshot: {snapshot}")
```

- [ ] **Step 4: Run CLI tests and quality**

```bash
uv run pytest tests/test_aris_vendor_cli.py -q
make quality
```

Expected: tests pass; CLI has no install/run/update command.

- [ ] **Step 5: Run upstream gates before publishing the formal snapshot**

```bash
cd '/Users/huangjiahao/自动化科研/aris'
python3 tools/check_skills_inventory.py
bash -n tools/install_aris_codex.sh
uv run --project '/Users/huangjiahao/自动化科研/auto-academic' python -m pytest -q \
  tests/test_codex_skill_mirror.py \
  tests/test_codex_install_update.py \
  tests/test_skill_groups.py \
  tests/test_argument_hint_lint.py
```

Expected: checker and shell syntax exit 0; the pinned conformance subset reports exactly `43 passed` with no skip/xfailed.

- [ ] **Step 6: Publish and verify the inactive formal snapshot**

```bash
cd '/Users/huangjiahao/自动化科研/auto-academic'
mkdir -p '/Users/huangjiahao/自动化科研/workspaces/karpathy-autoresearch'
uv run arw aris verify
uv run arw aris snapshot \
  --workspace '/Users/huangjiahao/自动化科研/workspaces/karpathy-autoresearch' \
  --confirm
test ! -e '/Users/huangjiahao/自动化科研/workspaces/karpathy-autoresearch/.agents'
uv run pytest -q -m stage_a1
make quality
```

Expected: snapshot succeeds, `.agents` remains absent, the Stage A1 subset is non-empty, and quality passes.

- [ ] **Step 7: Prove both upstream clones remain clean**

```bash
test -z "$(git -C '/Users/huangjiahao/自动化科研/aris' status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C '/Users/huangjiahao/自动化科研/karpathy-autoresearch' status --porcelain=v1 --untracked-files=all)"
```

Expected: both checks exit 0.

- [ ] **Step 8: Commit and push**

```bash
git add src/arw/cli.py tests/test_aris_vendor_cli.py
git commit -m 'feat: expose inactive ARIS vendor snapshot'
git push origin main
```

## Plan Self-Review

- This plan deliberately completes only Stage A1; it does not claim the Stage A safe activation profile is complete.
- Full ARIS content and upstream tests are reused; no skill, installer, dependency graph, queue, server, adapter, or broker is rewritten.
- The formal snapshot is lock-addressed, content-bound by a source-tree digest, and inactive; there is no `.agents/skills` bypass surface.
- Stage A2 must design the deny-by-default capability manifest, blocked/adapted targets, runtime broker, independent OS ownership, mandatory preflight, and manifest-owned uninstall before activation.
- Stage B–F remain unchanged: typed experiment API, Windows Coordinator/Runner, ARIS adapters, full W1→W3, and recovery gates.
- Function signatures remain consistent across Tasks 2–4; no placeholder or floating version is present.
