"""Bearer token authentication with scope enforcement."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

SECURITY_SCHEME = HTTPBearer(auto_error=False)

TokenVerifier = Callable[[str, str], bool]


class ScopedToken:
    """A pre-shared API token with associated scopes."""

    __slots__ = ("_token", "scopes")

    def __init__(self, token: str, scopes: frozenset[str]) -> None:
        self._token = token
        self.scopes = scopes

    def verify(self, candidate: str) -> bool:
        return secrets.compare_digest(self._token, candidate)


def require_scope(scope: str) -> Callable[[ScopedToken], ScopedToken]:
    """FastAPI dependency that requires a specific scope on the authenticated token."""

    def _check(token: ScopedToken = Depends(_authenticate)) -> ScopedToken:
        if scope not in token.scopes:
            raise HTTPException(
                status_code=403,
                detail={
                    "api_version": "1.0",
                    "request_id": "",
                    "error": {
                        "code": "insufficient_scope",
                        "message": f"token is missing required scope: {scope}",
                    },
                },
            )
        return token

    return _check


async def _authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(SECURITY_SCHEME),
) -> ScopedToken:
    verifier: TokenVerifier | None = getattr(request.app.state, "token_verifier", None)
    if verifier is None:
        raise HTTPException(status_code=500, detail="server not configured")
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail={
                "api_version": "1.0",
                "request_id": "",
                "error": {"code": "missing_token", "message": "Authorization header is required"},
            },
        )
    token = credentials.credentials
    if not verifier(token, ""):
        raise HTTPException(
            status_code=401,
            detail={
                "api_version": "1.0",
                "request_id": "",
                "error": {"code": "invalid_token", "message": "API token is invalid"},
            },
        )
    scopes: frozenset[str] = getattr(request.app.state, "token_scopes", frozenset())
    return ScopedToken(token, scopes)
