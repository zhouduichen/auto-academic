import httpx
import pytest

from arw.errors import (
    ArtifactError,
    AuthError,
    ConfigError,
    ConflictError,
    NetworkError,
    NotFoundError,
    ServerError,
    map_http_error,
)


def test_exit_codes_are_stable() -> None:
    assert [
        error.exit_code
        for error in (
            ConfigError("x"),
            AuthError("x"),
            NotFoundError("x"),
            ConflictError("x"),
            NetworkError("x"),
            ServerError("x"),
            ArtifactError("x"),
        )
    ] == [2, 3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, AuthError),
        (403, AuthError),
        (404, NotFoundError),
        (409, ConflictError),
        (412, ConflictError),
        (500, ServerError),
    ],
)
def test_http_error_mapping_is_safe(status: int, error_type: type[Exception]) -> None:
    response = httpx.Response(
        status,
        request=httpx.Request("GET", "https://node/api/v1/x"),
        json={
            "api_version": "1.0",
            "request_id": "req-1",
            "error": {"code": "denied", "message": "safe"},
        },
    )
    error = map_http_error(response)
    assert isinstance(error, error_type)
    assert error.request_id == "req-1"
    assert "safe" in str(error)


def test_invalid_error_body_is_not_reflected() -> None:
    response = httpx.Response(
        500,
        request=httpx.Request("GET", "https://node/api/v1/x"),
        content=b"secret raw body",
    )
    error = map_http_error(response)
    assert "secret" not in str(error)
