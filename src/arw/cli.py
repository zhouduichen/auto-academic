from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from arw.aris_activation import (
    install_profile,
    load_capability_profile,
    plan_install,
    preflight,
    run_profile,
    uninstall_profile,
)
from arw.aris_vendor import create_snapshot, load_lock, verify_repo
from arw.artifacts import download_artifact
from arw.client import ArwClient
from arw.config import Settings, load_settings
from arw.errors import ArwError, ConfigError
from arw.models import (
    ArtifactManifestResponse,
    CandidatePatch,
    ExperimentCancelRequest,
    ExperimentMatrix,
    ExperimentSubmitRequest,
)
from arw.output import stable_json

app = typer.Typer(no_args_is_help=True)
aris_app = typer.Typer(no_args_is_help=True)
experiments_app = typer.Typer(no_args_is_help=True)
app.add_typer(aris_app, name="aris")
app.add_typer(experiments_app, name="experiments")

CLIENT_FACTORY: Callable[[Settings], ArwClient] = ArwClient

DEFAULT_LOCK = Path(__file__).resolve().parents[2] / "configs" / "integrations" / "aris.yaml"
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[2] / "configs" / "integrations" / "aris-capabilities.yaml"
)


@app.callback()
def main() -> None:
    """Control AutoResearch Workbench from the Mac."""


@aris_app.command("verify")
def aris_verify(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_LOCK,
) -> None:
    lock = load_lock(config)
    report = verify_repo(lock)
    typer.echo(f"ARIS vendor OK: {len(report.skill_names)} skills @ {lock.commit[:12]}")


@aris_app.command("snapshot")
def aris_snapshot(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_LOCK,
) -> None:
    if not confirm:
        raise typer.BadParameter("--confirm is required to create the snapshot")
    lock = load_lock(config)
    snapshot = create_snapshot(lock, workspace)
    typer.echo(f"Inactive ARIS snapshot: {snapshot}")


def _call[Result](action: Callable[[ArwClient], Result]) -> Result:
    try:
        with CLIENT_FACTORY(load_settings()) as client:
            return action(client)
    except ArwError as exc:
        request = f" (request {exc.request_id})" if exc.request_id else ""
        typer.echo(f"error: {exc}{request}", err=True)
        raise typer.Exit(exc.exit_code) from exc


def _emit(kind: str, response: BaseModel, *, json_output: bool, human: str) -> None:
    typer.echo(stable_json(kind, response) if json_output else human)


@experiments_app.command("submit")
def experiments_submit(
    patch_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    project_id: Annotated[str, typer.Option("--project-id")],
    source_commit: Annotated[str, typer.Option("--source-commit")],
    title: Annotated[str, typer.Option("--title")],
    plan_id: Annotated[str, typer.Option("--plan-id")],
    seed: Annotated[list[int], typer.Option("--seed")],
    time_budget: Annotated[int, typer.Option("--time-budget", min=30, max=3600)],
    max_parallel: Annotated[int, typer.Option("--max-parallel", min=1, max=8)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        patch = patch_file.read_text(encoding="utf-8")
        request = ExperimentSubmitRequest(
            project_id=project_id,
            source_commit=source_commit,
            title=title,
            plan_id=plan_id,
            candidate=CandidatePatch(patch_sha256=sha256(patch.encode()).hexdigest(), patch=patch),
            matrix=ExperimentMatrix(
                seeds=seed,
                time_budget_seconds=time_budget,
                max_parallel=max_parallel,
            ),
        )
    except (OSError, UnicodeError) as exc:
        error = ConfigError("unable to read the UTF-8 patch file")
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(error.exit_code) from exc
    except ValidationError as exc:
        raise typer.BadParameter("invalid experiment submission") from exc
    response = _call(lambda client: client.submit_experiment(request))
    _emit(
        "experiment",
        response,
        json_output=json_output,
        human=f"{response.data.experiment_id}\t{response.data.state}",
    )


@experiments_app.command("list")
def experiments_list(
    limit: Annotated[int, typer.Option(min=1, max=100)] = 100,
    cursor: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    response = _call(lambda client: client.list_experiments(limit=limit, cursor=cursor))
    human = "\n".join(
        f"{item.experiment_id}\t{item.state}\t{item.title}" for item in response.items
    )
    _emit("experiment-list", response, json_output=json_output, human=human)


@experiments_app.command("show")
def experiments_show(
    experiment_id: str,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    response = _call(lambda client: client.get_experiment(experiment_id))
    data = response.data
    _emit(
        "experiment",
        response,
        json_output=json_output,
        human=f"{data.experiment_id}\t{data.state}\tversion={data.version}\t{data.title}",
    )


@experiments_app.command("events")
def experiments_events(
    experiment_id: str,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 100,
    cursor: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    response = _call(lambda client: client.list_events(experiment_id, limit=limit, cursor=cursor))
    human = "\n".join(
        f"{event.version}\t{event.state}\t{event.event_type}\t{event.occurred_at}"
        for event in response.items
    )
    _emit("experiment-events", response, json_output=json_output, human=human)


@experiments_app.command("cancel")
def experiments_cancel(
    experiment_id: str,
    reason: Annotated[str, typer.Option("--reason")],
    expected_version: Annotated[int, typer.Option("--expected-version", min=1)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        request = ExperimentCancelRequest(reason=reason, expected_version=expected_version)
    except ValidationError as exc:
        raise typer.BadParameter("invalid experiment cancellation request") from exc
    response = _call(lambda client: client.cancel_experiment(experiment_id, request))
    _emit(
        "experiment",
        response,
        json_output=json_output,
        human=f"{response.data.experiment_id}\t{response.data.state}",
    )


@experiments_app.command("artifacts")
def experiments_artifacts(
    experiment_id: str,
    download_dir: Annotated[Path | None, typer.Option("--download-dir")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def get_and_download(client: ArwClient) -> tuple[ArtifactManifestResponse, list[Path]]:
        manifest = client.get_artifact_manifest(experiment_id)
        if download_dir is None:
            return manifest, []
        downloaded = [
            download_artifact(client, experiment_id, item, download_dir) for item in manifest.items
        ]
        return manifest, downloaded

    response, paths = _call(get_and_download)
    if not paths:
        human = "\n".join(
            f"{item.filename}\t{item.byte_size}\t{item.sha256}" for item in response.items
        )
    else:
        human = "\n".join(str(path) for path in paths)
    _emit("artifact-manifest", response, json_output=json_output, human=human)


@aris_app.command("plan")
def aris_plan(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_LOCK,
    profile_config: Annotated[
        Path, typer.Option("--profile", exists=True, dir_okay=False)
    ] = DEFAULT_PROFILE,
) -> None:
    lock = load_lock(config)
    profile = load_capability_profile(profile_config)
    typer.echo(plan_install(lock, profile, workspace.resolve()), nl=False)


@aris_app.command("install")
def aris_install(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_LOCK,
    profile_config: Annotated[
        Path, typer.Option("--profile", exists=True, dir_okay=False)
    ] = DEFAULT_PROFILE,
) -> None:
    if not confirm:
        raise typer.BadParameter("--confirm is required to install the ARIS profile")
    lock = load_lock(config)
    profile = load_capability_profile(profile_config)
    state = install_profile(lock, profile, workspace.resolve())
    typer.echo(f"ARIS Stage A2 installed: {state.entry_count} managed entries")


@aris_app.command("preflight")
def aris_preflight(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_LOCK,
    profile_config: Annotated[
        Path, typer.Option("--profile", exists=True, dir_okay=False)
    ] = DEFAULT_PROFILE,
) -> None:
    lock = load_lock(config)
    profile = load_capability_profile(profile_config)
    state = preflight(lock, profile, workspace.resolve())
    typer.echo(f"ARIS Stage A2 preflight OK: {state.entry_count} managed entries")


@aris_app.command("uninstall")
def aris_uninstall(
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_LOCK,
    profile_config: Annotated[
        Path, typer.Option("--profile", exists=True, dir_okay=False)
    ] = DEFAULT_PROFILE,
) -> None:
    if not confirm:
        raise typer.BadParameter("--confirm is required to uninstall the ARIS profile")
    lock = load_lock(config)
    profile = load_capability_profile(profile_config)
    uninstall_profile(lock, profile, workspace.resolve())
    typer.echo("ARIS Stage A2 uninstalled; verified snapshots retained")


@aris_app.command("run")
def aris_run(
    prompt: Annotated[str, typer.Argument(help="Task for the local Codex session")],
    workspace: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_LOCK,
    profile_config: Annotated[
        Path, typer.Option("--profile", exists=True, dir_okay=False)
    ] = DEFAULT_PROFILE,
) -> None:
    lock = load_lock(config)
    profile = load_capability_profile(profile_config)
    return_code = run_profile(lock, profile, workspace.resolve(), prompt)
    if return_code:
        raise typer.Exit(return_code)
