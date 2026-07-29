import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Control AutoResearch Workbench from the Mac."""
