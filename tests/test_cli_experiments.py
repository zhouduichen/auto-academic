import json
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import httpx
from typer.testing import CliRunner

import arw.cli as cli
from arw.errors import ConflictError
from arw.models import (
    ArtifactEntry,
    ArtifactManifestResponse,
    ExperimentDetail,
    ExperimentEvent,
    ExperimentEventListResponse,
    ExperimentListResponse,
    ExperimentResponse,
    ExperimentSummary,
)

runner = CliRunner()


def detail(state: str = "submitted") -> ExperimentDetail:
    return ExperimentDetail(
        experiment_id="exp-1",
        project_id="p",
        source_commit="a" * 40,
        title="trial",
        state=state,
        version=1,
        plan_id="plan",
        candidate_patch_sha256=sha256(b"patch").hexdigest(),
        matrix={"seeds": [1], "time_budget_seconds": 30, "max_parallel": 1},
        result=None,
        error=None,
    )


class FakeClient:
    calls: ClassVar[list[tuple[str, object]]] = []
    fail_cancel: ClassVar[bool] = False

    def __init__(self, settings: object) -> None:
        self.settings = settings

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def submit_experiment(self, request: object) -> ExperimentResponse:
        self.calls.append(("submit", request))
        return ExperimentResponse(api_version="1.0", request_id="r", data=detail())

    def list_experiments(self, **kwargs: object) -> ExperimentListResponse:
        self.calls.append(("list", kwargs))
        item = ExperimentSummary(
            **detail().model_dump(
                exclude={"plan_id", "candidate_patch_sha256", "matrix", "result", "error"}
            )
        )
        return ExperimentListResponse(
            api_version="1.0", request_id="r", items=[item], next_cursor=None
        )

    def get_experiment(self, experiment_id: str) -> ExperimentResponse:
        self.calls.append(("show", experiment_id))
        return ExperimentResponse(api_version="1.0", request_id="r", data=detail())

    def list_events(self, experiment_id: str, **kwargs: object) -> ExperimentEventListResponse:
        self.calls.append(("events", (experiment_id, kwargs)))
        event = ExperimentEvent(
            event_id="e1",
            experiment_id=experiment_id,
            version=1,
            event_type="submitted",
            occurred_at="2026-07-29T00:00:00Z",
            state="submitted",
        )
        return ExperimentEventListResponse(
            api_version="1.0", request_id="r", items=[event], next_cursor=None
        )

    def cancel_experiment(self, experiment_id: str, request: object) -> ExperimentResponse:
        self.calls.append(("cancel", (experiment_id, request)))
        if self.fail_cancel:
            raise ConflictError("version conflict", request_id="r", status=409)
        return ExperimentResponse(api_version="1.0", request_id="r", data=detail(state="cancelled"))

    def get_artifact_manifest(self, experiment_id: str) -> ArtifactManifestResponse:
        self.calls.append(("artifacts", experiment_id))
        artifact = ArtifactEntry(
            artifact_id="a1",
            filename="metrics.json",
            byte_size=2,
            sha256=sha256(b"{}").hexdigest(),
            media_type="application/json",
        )
        return ArtifactManifestResponse(api_version="1.0", request_id="r", items=[artifact])

    @contextmanager
    def stream_artifact(self, experiment_id: str, artifact_id: str) -> Iterator[httpx.Response]:
        self.calls.append(("download", (experiment_id, artifact_id)))
        yield httpx.Response(200, content=b"{}")


def install_fake(monkeypatch: object) -> None:
    monkeypatch.setattr(cli, "CLIENT_FACTORY", FakeClient)
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    FakeClient.calls = []
    FakeClient.fail_cancel = False


def test_submit_builds_hash_and_matrix(tmp_path: Path, monkeypatch: object) -> None:
    install_fake(monkeypatch)
    patch_file = tmp_path / "candidate.diff"
    patch_file.write_text("patch")
    result = runner.invoke(
        cli.app,
        [
            "experiments",
            "submit",
            str(patch_file),
            "--project-id",
            "p",
            "--source-commit",
            "a" * 40,
            "--title",
            "trial",
            "--plan-id",
            "plan",
            "--seed",
            "1",
            "--time-budget",
            "30",
            "--max-parallel",
            "1",
            "--config-id",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1"
    request = FakeClient.calls[0][1]
    assert request.candidate.patch_sha256 == sha256(b"patch").hexdigest()
    assert request.matrix.seeds == [1]
    assert request.matrix.config_id == 0
    assert request.matrix.epochs == 1
    assert request.matrix.batch_size == 32


def test_read_and_cancel_commands(monkeypatch: object) -> None:
    install_fake(monkeypatch)
    for arguments, expected in [
        (["experiments", "list"], "exp-1"),
        (["experiments", "show", "exp-1"], "submitted"),
        (["experiments", "events", "exp-1"], "submitted"),
        (["experiments", "artifacts", "exp-1"], "metrics.json"),
        (
            ["experiments", "cancel", "exp-1", "--reason", "stop", "--expected-version", "1"],
            "cancelled",
        ),
    ]:
        result = runner.invoke(cli.app, arguments)
        assert result.exit_code == 0, result.output
        assert expected in result.stdout


def test_local_error_uses_stderr_and_stable_exit(monkeypatch: object) -> None:
    install_fake(monkeypatch)
    FakeClient.fail_cancel = True
    result = runner.invoke(
        cli.app, ["experiments", "cancel", "exp-1", "--reason", "stop", "--expected-version", "1"]
    )
    assert result.exit_code == 5
    assert result.stdout == ""
    assert "version conflict" in result.stderr


def test_artifacts_can_be_downloaded(tmp_path: Path, monkeypatch: object) -> None:
    install_fake(monkeypatch)
    result = runner.invoke(
        cli.app,
        ["experiments", "artifacts", "exp-1", "--download-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "metrics.json").read_bytes() == b"{}"
    assert "metrics.json" in result.stdout


def test_missing_required_option_sends_no_requests(monkeypatch: object) -> None:
    install_fake(monkeypatch)
    result = runner.invoke(cli.app, ["experiments", "cancel", "exp-1"])
    assert result.exit_code != 0
    assert FakeClient.calls == []


def test_invalid_utf8_patch_is_safe_config_error(tmp_path: Path, monkeypatch: object) -> None:
    install_fake(monkeypatch)
    patch_file = tmp_path / "candidate.diff"
    patch_file.write_bytes(b"\xff")
    result = runner.invoke(
        cli.app,
        [
            "experiments",
            "submit",
            str(patch_file),
            "--project-id",
            "p",
            "--source-commit",
            "a" * 40,
            "--title",
            "trial",
            "--plan-id",
            "plan",
            "--seed",
            "1",
            "--time-budget",
            "30",
            "--max-parallel",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "unable to read" in result.stderr
    assert FakeClient.calls == []


def test_validation_error_does_not_render_pydantic_input(
    tmp_path: Path, monkeypatch: object
) -> None:
    install_fake(monkeypatch)
    patch_file = tmp_path / "candidate.diff"
    patch_file.write_text("", encoding="utf-8")
    result = runner.invoke(
        cli.app,
        [
            "experiments",
            "submit",
            str(patch_file),
            "--project-id",
            "p",
            "--source-commit",
            "a" * 40,
            "--title",
            "trial",
            "--plan-id",
            "plan",
            "--seed",
            "1",
            "--time-budget",
            "30",
            "--max-parallel",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "invalid experiment submission" in result.stderr
    assert "input_value" not in result.stderr
    assert "validation error" not in result.stderr.lower()
    assert FakeClient.calls == []
