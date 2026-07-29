import os
import tempfile
from hashlib import sha256
from pathlib import Path, PurePath

from arw.client import ArwClient
from arw.errors import ArtifactError, ArwError
from arw.models import ArtifactEntry


def _safe_filename(filename: str) -> str:
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or PurePath(filename).is_absolute()
        or Path(filename).name != filename
    ):
        raise ArtifactError("artifact filename is unsafe")
    return filename


def download_artifact(
    client: ArwClient,
    experiment_id: str,
    artifact: ArtifactEntry,
    destination: Path,
) -> Path:
    filename = _safe_filename(artifact.filename)
    if destination.is_symlink():
        raise ArtifactError("artifact destination cannot be a symlink")
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactError("unable to create artifact destination") from exc
    if not destination.is_dir() or destination.is_symlink():
        raise ArtifactError("artifact destination must be a real directory")

    final_path = destination / filename
    if final_path.exists() or final_path.is_symlink():
        raise ArtifactError(f"artifact destination already exists: {filename}")

    descriptor = -1
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".arw-download-", dir=destination)
        os.fchmod(descriptor, 0o600)
        digest = sha256()
        byte_count = 0
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            with client.stream_artifact(experiment_id, artifact.artifact_id) as response:
                for chunk in response.iter_bytes():
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if byte_count != artifact.byte_size:
            raise ArtifactError(
                f"artifact size mismatch: expected {artifact.byte_size}, received {byte_count}"
            )
        if digest.hexdigest() != artifact.sha256:
            raise ArtifactError("artifact SHA-256 mismatch")
        try:
            os.link(temporary_name, final_path)
        except FileExistsError as exc:
            raise ArtifactError(f"artifact destination already exists: {filename}") from exc
        os.unlink(temporary_name)
        temporary_name = None
        return final_path
    except ArtifactError:
        raise
    except (ArwError, OSError) as exc:
        raise ArtifactError("artifact download failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
