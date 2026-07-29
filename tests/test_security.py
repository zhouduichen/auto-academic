from typer.testing import CliRunner

from arw.cli import app


def test_experiment_help_exposes_no_unsafe_execution_or_transport_options() -> None:
    runner = CliRunner()
    for command in ("submit", "list", "show", "events", "cancel", "artifacts"):
        result = runner.invoke(app, ["experiments", command, "--help"])
        assert result.exit_code == 0, result.output
        lowered = result.stdout.lower()
        for forbidden in ("--token", "--insecure", "--host", "--command", "--argv", "--env"):
            assert forbidden not in lowered
