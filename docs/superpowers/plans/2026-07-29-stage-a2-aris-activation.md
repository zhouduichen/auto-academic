# ARIS Stage A2 Development-Safe Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the pinned A1 ARIS snapshot for local Codex development with 68 native skills, 13 fail-closed skills, official ARIS install/uninstall semantics, mandatory preflight, and a hardened local Codex launcher.

**Architecture:** A typed capability profile partitions the exact A1 inventory. AutoAcademic copies the verified A1 snapshot into a content-addressed activation snapshot, replaces only the 13 execution-capable Codex skill directories with deterministic blocked adapters, and invokes ARIS's unmodified `install_aris_codex.sh --all`. A project-local deny rules file plus Codex's `workspace-write` OS sandbox keeps `.agents` recursively read-only during `arw aris run`; network stays disabled and approval policy is `never`.

**Tech Stack:** Python 3.12, Pydantic, PyYAML, Typer, Git, ARIS Bash 3.2-compatible installer, Codex CLI, pytest, ruff, mypy.

## Global Constraints

- Work only on branch `feat/stage-a2-activation` in the isolated Stage A2 worktree.
- Reuse `/Users/huangjiahao/自动化科研/aris` at commit `53562a7c64cc1d55946cba1fb8a8416137143d14` and the verified A1 snapshot; do not edit either.
- Capability inventory is exactly 81 unique names: 68 `native`, 13 `blocked-adapter`; unknown names fail closed.
- Blocked names are `auto-review-loop`, `auto-review-loop-minimax`, `dse-loop`, `experiment-bridge`, `experiment-queue`, `monitor-experiment`, `qzcli`, `result-to-claim`, `run-experiment`, `serverless-modal`, `system-profile`, `training-check`, and `vast-gpu`.
- Official `--all` installs 82 manifest-owned entries: 81 skills plus `shared-references` support.
- Do not implement an MCP server, HTTP execution API, Windows service, scheduler, training command, SSH path, or cloud/GPU adapter.
- Tests use temporary fixtures and mocked subprocess boundaries; they perform no network calls.
- `arw aris run` launches local Codex with `workspace-write`, approval `never`, `sandbox_workspace_write.network_access=false`, and trusted project rules. Codex documents `.agents` as recursively read-only inside a writable root.
- Project rules explicitly forbid `ssh`, `scp`, `rsync`, `screen`, `tmux`, `modal`, `vast`, `vastai`, `qzcli`, and `tailscale` command prefixes.

---

### Task 1: Exact capability profile

**Files:**
- Create: `configs/integrations/aris-capabilities.yaml`
- Create: `src/arw/aris_activation.py`
- Create: `tests/test_aris_activation.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `ArisLock`, `VendorReport`, and `SnapshotManifest` from `arw.aris_vendor`.
- Produces: `CapabilityProfile`, `load_capability_profile(path: Path)`, `verify_capability_profile(profile, vendor_report)`, and `ArisActivationError`.

- [ ] **Step 1: Add the Stage A2 marker and exact profile**

The YAML contains `schema_version: 1`, A1 `vendor_profile_sha256`, A1 `inventory_sha256`, `unknown_skill_policy: block`, `blocked_adapter_version: 1`, the exact 68-name `native_skills` list, the exact 13-name `blocked_skills` list, and the ten forbidden command names. Add `stage_a2` to the pytest marker list.

- [ ] **Step 2: Write failing profile tests**

Tests assert that both lists are duplicate-free and disjoint, their union equals `verify_repo(lock).skill_names`, counts are 68/13/81, profile inventory hashes match A1, and adding or removing any name raises `ArisActivationError`.

Run: `uv run pytest tests/test_aris_activation.py -q`

Expected: collection fails because `arw.aris_activation` does not exist.

- [ ] **Step 3: Implement typed validation**

Use frozen, `extra="forbid"` Pydantic models. `CapabilityProfile.digest()` hashes canonical JSON. `verify_capability_profile` compares the profile's A1 digests and exact name union to `VendorReport`, and rejects duplicates, overlap, unexpected counts, or unknown policy values.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest tests/test_aris_activation.py -q
make quality
git add configs/integrations/aris-capabilities.yaml src/arw/aris_activation.py tests/test_aris_activation.py pyproject.toml
git commit -m 'feat: classify complete ARIS capability inventory'
```

---

### Task 2: Verified activation snapshot and blocked adapters

**Files:**
- Modify: `src/arw/aris_activation.py`
- Modify: `tests/test_aris_activation.py`

**Interfaces:**
- Produces: `ActivationManifest`, `create_activation_snapshot(lock, profile, workspace) -> Path`, and `verify_activation_snapshot(lock, profile, workspace, snapshot) -> ActivationManifest`.

- [ ] **Step 1: Add failing snapshot tests**

Tests build a temporary regular-file A1 fixture and assert:

- the activation path is addressed by `CapabilityProfile.digest()`;
- all 68 native skill trees are byte/mode identical to A1;
- all 13 blocked directories contain only a deterministic `SKILL.md` with the same name, `allowed-tools: Read`, `skill_not_activated`, and no `Bash`;
- `shared-references`, tools, templates, and docs remain present;
- every activation file and directory is non-writable;
- a changed native byte, adapter byte, manifest field, symlink, or writable bit fails verification.

- [ ] **Step 2: Implement deterministic derivation**

Copy the verified A1 snapshot to a temporary sibling directory, temporarily add owner-write bits, replace the 13 blocked directories, and write `.autoacademic/activation.json`. Build the expected activation entry map independently from the A1 path/type/mode/content entries plus exact adapter bytes; compare it to the staged tree before publication. Remove write bits, atomically rename to `.aris/vendor/aris-activation/<profile-digest>`, and verify after publication.

`ActivationManifest` records every locked A1 identity, capability profile digest, 68/13 counts and lists, `entry_count: 82`, adapter version, forbidden commands, and expected activation tree SHA-256. Verification uses a typed manifest and checks every field exactly.

- [ ] **Step 3: Run tests and commit**

```bash
uv run pytest tests/test_aris_activation.py -q
make quality
git add src/arw/aris_activation.py tests/test_aris_activation.py
git commit -m 'feat: derive fail-closed ARIS activation snapshot'
```

---

### Task 3: Official plan, install, preflight, and uninstall

**Files:**
- Modify: `src/arw/aris_activation.py`
- Modify: `tests/test_aris_activation.py`

**Interfaces:**
- Produces: `plan_install(...) -> str`, `install_profile(...) -> ActivationState`, `preflight(...) -> ActivationState`, and `uninstall_profile(...) -> None`.

- [ ] **Step 1: Add failing lifecycle tests**

Use a temporary fake installer boundary and exact subprocess assertions. Tests prove:

- plan uses the A1 installer with activation snapshot `--aris-repo`, `--all`, `--no-doc`, and `--dry-run`, without `--quiet`, and does not mutate the workspace;
- install uses isolated `HOME=<workspace>/.aris/installer-home`, `--all --quiet --no-doc`, then records the official manifest hash;
- preflight accepts exactly 81 `kind=skill` entries plus one `kind=support` `shared-references`, and checks every canonical symlink target;
- a missing, retargeted, real-path, extra manifest-owned, or original-upstream target fails;
- install rejects symlink parents for `.aris`, `.agents`, `.codex`, and their managed children;
- post-install failure invokes official uninstall rollback;
- uninstall invokes the A1 installer with `--uninstall --quiet --no-doc`, preserves user paths, and removes only an exact owned deny rules file.

- [ ] **Step 2: Implement the thin lifecycle wrapper**

Execute the unmodified A1 `tools/install_aris_codex.sh`; do not reproduce its selection, conflicts, locking, linking, or uninstall mutations. Parse its committed TSV manifest only for independent preflight. Store `.aris/autoacademic-activation.json` as a frozen typed state with activation digest/path, official manifest SHA-256, rules SHA-256, 82-entry count, and exact skill lists.

Create `.codex/rules/aris-deny.rules` only when absent or byte-identical. It contains one `prefix_rule(decision="forbidden")` per forbidden command with `match` examples. Treat a different existing file as a conflict. On uninstall, delete only the byte-identical owned file and retain all other `.codex`, `.agents`, `.aris`, user, and research files.

- [ ] **Step 3: Run tests and commit**

```bash
uv run pytest tests/test_aris_activation.py -q
make quality
git add src/arw/aris_activation.py tests/test_aris_activation.py
git commit -m 'feat: wrap official ARIS activation lifecycle'
```

---

### Task 4: CLI surface and hardened Codex launcher

**Files:**
- Modify: `src/arw/cli.py`
- Create: `tests/test_aris_activation_cli.py`

**Interfaces:**
- Produces: `arw aris plan`, `install`, `preflight`, `uninstall`, and `run` while preserving A1 `verify` and `snapshot`.

- [ ] **Step 1: Add failing CLI tests**

Tests require explicit `--confirm` for install/uninstall, assert all lifecycle commands appear in help, and mock `subprocess.run` for `run`. The exact launch prefix is:

```python
[
    "codex",
    "--strict-config",
    "--sandbox", "workspace-write",
    "--ask-for-approval", "never",
    "-c", "sandbox_workspace_write.network_access=false",
    "-c", f'projects."{workspace}".trust_level="trusted"',
    "--cd", str(workspace),
]
```

The final positional prompt starts with the immutable Stage A2 safety instructions and appends the user's prompt. No `--search`, `--add-dir`, bypass, remote, or network-enabling option is present.

- [ ] **Step 2: Implement CLI and run boundary**

`run_profile` calls `preflight`, locates `codex` with `shutil.which`, invokes it without a shell and without captured stdin/stdout, and calls `preflight` again after it exits. The safety prefix forbids SSH/SCP/rsync, screen/tmux, Modal/Vast/qzcli/Tailscale, background execution, training, and editing `.agents`, `.codex`, activation snapshots, or manifests. Network denial and Codex's protected `.agents` path are the technical boundaries; prompt text is defense in depth.

- [ ] **Step 3: Run tests and commit**

```bash
uv run pytest tests/test_aris_activation_cli.py -q
make quality
git add src/arw/cli.py tests/test_aris_activation_cli.py
git commit -m 'feat: run ARIS through hardened local Codex'
```

---

### Task 5: Formal gate

**Files:** No source additions.

- [ ] **Step 1: Run all local and upstream gates**

```bash
make quality
cd '/Users/huangjiahao/自动化科研/aris'
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_skills_inventory.py
bash -n tools/install_aris_codex.sh
PYTHONDONTWRITEBYTECODE=1 uv run --project '/Users/huangjiahao/自动化科研/auto-academic-stage-a2' \
  python -m pytest -q -p no:cacheprovider \
  tests/test_codex_skill_mirror.py tests/test_codex_install_update.py \
  tests/test_skill_groups.py tests/test_argument_hint_lint.py
```

Expected: AutoAcademic quality passes and upstream reports exactly `43 passed`.

- [ ] **Step 2: Run a temporary lifecycle smoke test**

Use a temporary workspace, run plan/install/preflight/uninstall, verify 82 official entries during install, verify all 13 blocked links resolve to adapters, verify no blocked link resolves into the A1 native directory, and verify uninstall leaves no manifest-owned links while retaining content-addressed snapshots.

- [ ] **Step 3: Final review**

Run `git diff --check`, verify no Windows/server/MCP/API files changed, and confirm the branch contains small reviewable commits only.

## Self-Review

- Coverage: exact 81-way partition, 13 blocked adapters, 82 official entries, activation snapshot, plan/install/preflight/uninstall, Codex run sandbox, explicit deny rules, and zero-network tests each have a task.
- Scope: MCP, execution API, Windows, training, remote shell, and cloud adapters remain deferred.
- Types: public lifecycle signatures and manifest names are consistent across tasks.
- Placeholders: no TBD/TODO or floating upstream version is present.
