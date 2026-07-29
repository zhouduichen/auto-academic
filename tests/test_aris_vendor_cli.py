from pathlib import Path

import pytest
from typer.testing import CliRunner

from arw.cli import app

pytestmark = pytest.mark.stage_a1
runner = CliRunner()


def test_snapshot_requires_confirmation(tmp_path: Path) -> None:
    result = runner.invoke(app, ["aris", "snapshot", "--workspace", str(tmp_path)])
    assert result.exit_code == 2
    assert "--confirm" in result.output


def test_aris_cli_preserves_a1_vendor_commands() -> None:
    result = runner.invoke(app, ["aris", "--help"])
    assert result.exit_code == 0
    assert "verify" in result.output
    assert "snapshot" in result.output
