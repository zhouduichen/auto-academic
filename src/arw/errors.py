from typing import ClassVar

import httpx
from pydantic import ValidationError

from arw.models import ErrorResponse


class ArwError(Exception):
    exit_code: ClassVar[int] = 1

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        request_id: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.status = status


class ConfigError(ArwError):
    exit_code = 2


class AuthError(ArwError):
    exit_code = 3


class NotFoundError(ArwError):
    exit_code = 4


class ConflictError(ArwError):
    exit_code = 5


class NetworkError(ArwError):
    exit_code = 6


class ServerError(ArwError):
    exit_code = 7


class ArtifactError(ArwError):
    exit_code = 8


def map_http_error(response: httpx.Response) -> ArwError:
    error_type: type[ArwError]
    if response.status_code in {401, 403}:
        error_type = AuthError
    elif response.status_code == 404:
        error_type = NotFoundError
    elif response.status_code in {409, 412}:
        error_type = ConflictError
    else:
        error_type = ServerError

    message = f"server returned HTTP {response.status_code}"
    code: str | None = None
    request_id: str | None = response.headers.get("X-Request-ID")
    try:
        payload = ErrorResponse.model_validate(response.json())
    except (ValueError, ValidationError):
        pass
    else:
        message = payload.error.message
        code = payload.error.code
        request_id = payload.request_id
    return error_type(message, code=code, request_id=request_id, status=response.status_code)
