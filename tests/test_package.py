from typer.testing import CliRunner

from arw import __version__
from arw.cli import app

runner = CliRunner()


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output
