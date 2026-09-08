from __future__ import annotations

import re
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class RequestBoundaryMiddleware:
    """Bound JSON bodies before parsing, including chunked requests, and propagate safe IDs."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        candidate = headers.get("x-request-id", "")
        request_id = candidate if _REQUEST_ID.fullmatch(candidate) else str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["x-request-id"] = request_id
            await send(message)

        async def reject(status: int, code: str, message: str) -> None:
            response = JSONResponse(
                status_code=status,
                content={
                    "error": {
                        "code": code,
                        "message": message,
                        "details": {},
                        "request_id": request_id,
                    }
                },
            )
            await response(scope, receive, send_with_id)

        length = headers.get("content-length")
        if length is not None:
            if not length.isascii() or not length.isdecimal():
                await reject(400, "invalid_request", "invalid Content-Length")
                return
            if len(length) > 20 or int(length) > self.max_body_bytes:
                await reject(413, "request_too_large", "request body exceeds configured limit")
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await reject(413, "request_too_large", "request body exceeds configured limit")
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send_with_id)
