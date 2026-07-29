import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from arw.config import ConfigError, load_settings


def test_settings_precedence_and_secret(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "arw"
    config.mkdir(parents=True)
    (config / "node.yaml").write_text(
        "server: https://yaml.example\napi_token: yaml-token\nread_timeout: 10\n",
        encoding="utf-8",
    )
    env_file = config / "env"
    env_file.write_text("ARW_SERVER=https://env-file.example\nARW_API_TOKEN=file-token\n")
    env_file.chmod(0o600)
    settings = load_settings(
        home=tmp_path,
        environ={"ARW_SERVER": "https://process.example", "ARW_API_TOKEN": "process-token"},
    )
    assert settings.server == "https://process.example"
    assert isinstance(settings.api_token, SecretStr)
    assert settings.api_token.get_secret_value() == "process-token"
    assert settings.read_timeout == 10
    assert "process-token" not in repr(settings)


def test_env_file_permissions_are_enforced(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "arw"
    config.mkdir(parents=True)
    env_file = config / "env"
    env_file.write_text("ARW_SERVER=https://node.example\nARW_API_TOKEN=x\n")
    env_file.chmod(0o644)
    with pytest.raises(ConfigError, match="permissions"):
        load_settings(home=tmp_path, environ={})


def test_https_and_ca_are_required(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="HTTPS"):
        load_settings(
            home=tmp_path,
            environ={"ARW_SERVER": "http://node.example", "ARW_API_TOKEN": "x"},
        )
    with pytest.raises(ConfigError, match="CA bundle"):
        load_settings(
            home=tmp_path,
            environ={
                "ARW_SERVER": "https://node.example",
                "ARW_API_TOKEN": "x",
                "ARW_CA_BUNDLE": str(tmp_path / "missing.pem"),
            },
        )


def test_env_file_must_be_owned_by_current_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / ".config" / "arw"
    config.mkdir(parents=True)
    env_file = config / "env"
    env_file.write_text("ARW_SERVER=https://node.example\nARW_API_TOKEN=x\n")
    env_file.chmod(0o600)
    monkeypatch.setattr(os, "getuid", lambda: env_file.stat().st_uid + 1)
    with pytest.raises(ConfigError, match="owned"):
        load_settings(home=tmp_path, environ={})
