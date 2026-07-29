from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from arw.aris_vendor import Sha256, VendorReport

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
