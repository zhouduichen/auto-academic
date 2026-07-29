import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import Field, SecretStr, ValidationError, model_validator

from arw.errors import ConfigError
from arw.models import StrictModel


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


def load_settings(
    *, environ: Mapping[str, str] | None = None, home: Path | None = None
) -> Settings:
    environment = os.environ if environ is None else environ
    config_dir = (Path.home() if home is None else home) / ".config" / "arw"
    values = _read_yaml(config_dir / "node.yaml")
    file_environment = _read_env_file(config_dir / "env")
    env_keys = {
        "ARW_SERVER": "server",
        "ARW_API_TOKEN": "api_token",
        "ARW_CA_BUNDLE": "ca_bundle",
        "ARW_CONNECT_TIMEOUT": "connect_timeout",
        "ARW_READ_TIMEOUT": "read_timeout",
        "ARW_ARTIFACT_READ_TIMEOUT": "artifact_read_timeout",
    }
    for source in (file_environment, environment):
        for env_key, setting_key in env_keys.items():
            if env_key in source:
                values[setting_key] = source[env_key]
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        messages = " ".join(error["msg"] for error in exc.errors())
        raise ConfigError(f"invalid ARW configuration: {messages}") from exc


__all__ = ["ConfigError", "Settings", "load_settings"]
