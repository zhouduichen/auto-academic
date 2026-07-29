import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from arw.config import ConfigError, load_settings


def test_settings_precedence_and_secret(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "arw"
    config.mkdir(parents=True)
    (config / "node.yaml").write_text(
        "server:\n"
        "  api_url: https://yaml.example\n"
        "  token_env: CUSTOM_ARW_TOKEN\n"
        "read_timeout: 10\n",
        encoding="utf-8",
    )
    env_file = config / "env"
    env_file.write_text("ARW_SERVER=https://env-file.example\nCUSTOM_ARW_TOKEN=file-token\n")
    env_file.chmod(0o600)
    settings = load_settings(
        home=tmp_path,
        environ={"ARW_SERVER": "https://process.example", "CUSTOM_ARW_TOKEN": "process-token"},
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


def test_nested_node_defaults_and_custom_token_env(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "arw"
    config.mkdir(parents=True)
    (config / "node.yaml").write_text(
        "server:\n  api_url: https://node.example:8443\n  token_env: NODE_TOKEN\n",
        encoding="utf-8",
    )
    env_file = config / "env"
    env_file.write_text("NODE_TOKEN=file-secret\n", encoding="utf-8")
    env_file.chmod(0o600)
    settings = load_settings(home=tmp_path, environ={})
    assert settings.server == "https://node.example:8443"
    assert settings.api_token.get_secret_value() == "file-secret"


@pytest.mark.parametrize(
    "yaml_text",
    [
        "server:\n  api_url: https://node.example\n  api_token: forbidden\n",
        "server:\n  api_url: https://node.example\napi_token: forbidden\n",
    ],
)
def test_yaml_plaintext_token_is_rejected(tmp_path: Path, yaml_text: str) -> None:
    config = tmp_path / ".config" / "arw"
    config.mkdir(parents=True)
    (config / "node.yaml").write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match="must not contain api_token"):
        load_settings(home=tmp_path, environ={"ARW_API_TOKEN": "process-secret"})


def test_token_is_read_only_from_declared_environment_name(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "arw"
    config.mkdir(parents=True)
    (config / "node.yaml").write_text(
        "server:\n  api_url: https://node.example\n  token_env: CUSTOM_TOKEN\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="CUSTOM_TOKEN"):
        load_settings(home=tmp_path, environ={"ARW_API_TOKEN": "wrong-token"})


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
