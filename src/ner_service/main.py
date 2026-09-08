from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from ner_service.cache import MemoryCache, ResultCache
from ner_service.config import Settings, get_settings
from ner_service.config_store import ConfigNotFoundError, PromptTemplateError
from ner_service.errors import provider_details, provider_headers, public_error
from ner_service.metrics import setup_metrics
from ner_service.middleware import RequestBoundaryMiddleware
from ner_service.providers.base import ProviderError
from ner_service.providers.registry import get_provider
from ner_service.routes import router as v1_router
from ner_service.service import NerService
from ner_service.stores import SQLiteStore
from ner_service.telemetry import setup_tracing

logger = logging.getLogger(__name__)


async def lifespan(app: FastAPI) -> Any:
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else get_settings()
    app.state.settings = settings
    injected_service: NerService | None = getattr(app.state, "service", None)
    if injected_service is not None:
        try:
            yield
        finally:
            await injected_service.aclose()
            app.state.tracer_provider.shutdown()
        return
    provider = get_provider(settings)
    cache = None
    if settings.cache_enabled:
        cache = ResultCache(
            MemoryCache(max_size=settings.cache_max_size),
            ttl=settings.cache_ttl_seconds,
        )
    app.state.service = NerService(
        provider,
        default_model=settings.ner_model,
        max_tokens=settings.max_tokens,
        limits=settings.runtime_limits(),
        cache=cache,
        config_store=SQLiteStore(settings.config_db_path),
        token_pricing=settings.token_pricing(),
    )
    try:
        await app.state.service.ready()
        yield
    finally:
        await app.state.service.aclose()
        app.state.tracer_provider.shutdown()


def create_app(settings: Settings | None = None, service: NerService | None = None) -> FastAPI:
    app = FastAPI(
        title="NER Service",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings if settings is not None else get_settings()
    if service is not None:
        app.state.service = service

    app.include_router(v1_router, prefix="/v1")
    setup_metrics(app)
    setup_tracing(app)

    app.add_middleware(
        RequestBoundaryMiddleware, max_body_bytes=app.state.settings.max_request_body_bytes
    )
    _register_exception_handlers(app)
    original_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        schema = original_openapi()
        for path in schema.get("paths", {}).values():
            for operation in path.values():
                if (
                    isinstance(operation, dict)
                    and operation.get("security")
                    and {} not in operation["security"]
                ):
                    operation["security"].append({})
        return schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    async def known_error(request: Request, exc: Exception) -> JSONResponse:
        status, code, message = public_error(exc)
        details = provider_details(exc) if isinstance(exc, ProviderError) else {}
        headers = provider_headers(exc) if isinstance(exc, ProviderError) else None
        return _error_response(
            request,
            status,
            code,
            message,
            details={"provider": details} if details else None,
            headers=headers,
        )

    for kind in (ConfigNotFoundError, PromptTemplateError, ValueError, ProviderError):
        app.add_exception_handler(kind, known_error)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            request,
            422,
            "validation_error",
            "request validation failed",
            details={"errors": _validation_errors(exc)},
        )

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        message = str(exc.detail) if exc.detail else "http error"
        return _error_response(
            request, exc.status_code, "http_error", message, headers=dict(exc.headers or {})
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected request failure", exc_info=exc)
        return _error_response(request, 500, "internal_error", "internal server error")


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    content: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }
    response_headers = dict(headers or {})
    response_headers["x-request-id"] = request_id
    return JSONResponse(status_code=status_code, content=content, headers=response_headers)


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if isinstance(value, str) and value:
        return value
    return str(uuid4())


def _validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for error in exc.errors():
        item: dict[str, Any] = {
            "loc": list(error.get("loc", [])),
            "msg": error.get("msg", "validation error"),
            "type": error.get("type", "value_error"),
        }
        errors.append(item)
    return errors


app = create_app()
