import hashlib
import json
from pathlib import Path

import pytest

from arw.aris_activation import (
    ArisActivationError,
    CapabilityProfile,
    load_capability_profile,
    verify_capability_profile,
)
from arw.aris_vendor import VendorReport, load_lock, verify_repo

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
