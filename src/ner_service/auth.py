from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


async def authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if request.scope["path"] in {"/v1/health", "/v1/ready"}:
        return
    key = request.app.state.settings.ner_api_key
    if key is None:
        return
    actual = credentials.credentials if credentials is not None else ""
    if not secrets.compare_digest(actual.encode(), key.get_secret_value().encode()):
        raise HTTPException(
            401, "valid Bearer token required", headers={"WWW-Authenticate": "Bearer"}
        )
