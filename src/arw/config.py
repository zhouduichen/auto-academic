import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import Field, SecretStr, ValidationError, model_validator

from arw.errors import ConfigError
from arw.models import StrictModel

REPLACEMENT_VALUE = "REPLACE"


class Settings(StrictModel):
    server: str
    api_token: SecretStr
    ca_bundle: Path | None = None
    connect_timeout: float = Field(default=3.0, gt=0)
    read_timeout: float = Field(default=15.0, gt=0)
    artifact_read_timeout: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def validate_security(self) -> "Settings":
        parsed = urlsplit(self.server)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("server must be an HTTPS origin without credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("server must be an HTTPS origin without a path")
        if self.ca_bundle is not None and not self.ca_bundle.is_file():
            raise ValueError("CA bundle must be an existing file")
        return self


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ConfigError("ARW env file must be a regular file")
    if metadata.st_uid != os.getuid():
        raise ConfigError("ARW env file must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ConfigError("ARW env file permissions must be 0600 or stricter")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"invalid ARW env file entry on line {line_number}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError("unable to read ARW node configuration") from exc
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ConfigError("ARW node configuration must be a mapping")
    return loaded


def _contains_plaintext_token(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key == "api_token" or _contains_plaintext_token(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_plaintext_token(item) for item in value)
    return False


def _node_defaults(raw_node: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if _contains_plaintext_token(raw_node):
        raise ConfigError("ARW node configuration must not contain api_token")
    raw_server = raw_node.get("server", {})
    if not isinstance(raw_server, dict):
        raise ConfigError("ARW node server configuration must be a mapping")
    api_url = raw_server.get("api_url", "")
    token_env = raw_server.get("token_env", "ARW_API_TOKEN")
    if not isinstance(api_url, str):
        raise ConfigError("ARW node api_url must be a string")
    if not isinstance(token_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env):
        raise ConfigError("ARW node token_env must name an environment variable")
    values = {
        key: raw_node[key]
        for key in (
            "ca_bundle",
            "connect_timeout",
            "read_timeout",
            "artifact_read_timeout",
        )
        if key in raw_node
    }
    values["server"] = api_url
    return values, token_env


def load_settings(
    *, environ: Mapping[str, str] | None = None, home: Path | None = None
) -> Settings:
    environment = os.environ if environ is None else environ
    config_dir = (Path.home() if home is None else home) / ".config" / "arw"
    values, token_env = _node_defaults(_read_yaml(config_dir / "node.yaml"))
    file_environment = _read_env_file(config_dir / "env")
    non_secret_env_keys = {
        "ARW_SERVER": "server",
        "ARW_CA_BUNDLE": "ca_bundle",
        "ARW_CONNECT_TIMEOUT": "connect_timeout",
        "ARW_READ_TIMEOUT": "read_timeout",
        "ARW_ARTIFACT_READ_TIMEOUT": "artifact_read_timeout",
    }
    for source in (file_environment, environment):
        for env_key, setting_key in non_secret_env_keys.items():
            if env_key in source:
                values[setting_key] = source[env_key]
    token = environment.get(token_env) or file_environment.get(token_env)
    if not token or token == REPLACEMENT_VALUE:
        raise ConfigError(f"{token_env} is missing or still uses the replacement value")
    values["api_token"] = token
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        messages = " ".join(error["msg"] for error in exc.errors())
        raise ConfigError(f"invalid ARW configuration: {messages}") from exc


__all__ = ["ConfigError", "Settings", "load_settings"]
