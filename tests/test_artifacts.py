from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from arw.artifacts import download_artifact
from arw.errors import ArtifactError, NetworkError
from arw.models import ArtifactEntry


class FakeClient:
    def __init__(self, content: bytes, *, failure: bool = False) -> None:
        self.content = content
        self.failure = failure

    @contextmanager
    def stream_artifact(self, experiment_id: str, artifact_id: str) -> Iterator[httpx.Response]:
        assert experiment_id == "exp-1"
        assert artifact_id == "artifact-1"
        if self.failure:
            raise NetworkError("stream failed")
        yield httpx.Response(200, content=self.content)


def entry(content: bytes, filename: str = "metrics.json") -> ArtifactEntry:
    return ArtifactEntry(
        artifact_id="artifact-1",
        filename=filename,
        byte_size=len(content),
        sha256=sha256(content).hexdigest(),
        media_type="application/json",
    )


def test_download_is_verified_and_private(tmp_path: Path) -> None:
    path = download_artifact(FakeClient(b"result"), "exp-1", entry(b"result"), tmp_path)
    assert path.read_bytes() == b"result"
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".arw-download-*"))


@pytest.mark.parametrize("filename", ["", "/outside", "../x", "a/b", "a\\b", ".", ".."])
def test_unsafe_names_are_rejected(tmp_path: Path, filename: str) -> None:
    with pytest.raises((ArtifactError, ValueError)):
        download_artifact(FakeClient(b"x"), "exp-1", entry(b"x", filename), tmp_path)


def test_existing_destination_is_not_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "metrics.json"
    existing.write_text("keep")
    with pytest.raises(ArtifactError, match="exists"):
        download_artifact(FakeClient(b"new"), "exp-1", entry(b"new"), tmp_path)
    assert existing.read_text() == "keep"


@pytest.mark.parametrize("wrong", ["size", "hash"])
def test_integrity_failures_clean_temporary_file(tmp_path: Path, wrong: str) -> None:
    artifact = entry(b"expected")
    if wrong == "size":
        artifact = artifact.model_copy(update={"byte_size": 1})
    else:
        artifact = artifact.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ArtifactError):
        download_artifact(FakeClient(b"expected"), "exp-1", artifact, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_stream_failure_cleans_temporary_file(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="download failed"):
        download_artifact(FakeClient(b"x", failure=True), "exp-1", entry(b"x"), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_symlink_destination_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        download_artifact(FakeClient(b"x"), "exp-1", entry(b"x"), link)
