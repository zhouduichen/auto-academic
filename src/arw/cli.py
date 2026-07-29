from pathlib import Path
from typing import Annotated

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
