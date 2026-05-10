from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ner_service.cache import MemoryCache, ResultCache
from ner_service.config import Settings, get_settings
from ner_service.config_store import ConfigNotFoundError, PromptTemplateError
from ner_service.providers.base import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUpstreamError,
)
from ner_service.providers.registry import get_provider
from ner_service.routes import router as v1_router
from ner_service.service import NerService
from ner_service.stores import SQLiteStore
from ner_service.telemetry import setup_tracing

logger = __import__("logging").getLogger(__name__)


async def lifespan(app: FastAPI) -> Any:
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else get_settings()
    app.state.settings = settings
    injected_service: NerService | None = getattr(app.state, "service", None)
    if injected_service is not None:
        try:
            yield
        finally:
            await injected_service.aclose()
        return
    provider = get_provider(settings)
    app.state.service = NerService(
        provider,
        default_model=settings.ner_model,
        max_tokens=settings.max_tokens,
        limits=settings.runtime_limits(),
        cache=ResultCache(MemoryCache()),
        config_store=SQLiteStore(),
    )
    try:
        yield
    finally:
        await app.state.service.aclose()


def create_app(settings: Settings | None = None, service: NerService | None = None) -> FastAPI:
    app = FastAPI(
        title="NER Service",
        version="1.0",
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings
    if service is not None:
        app.state.service = service

    app.include_router(v1_router, prefix="/v1")
    setup_tracing(app)

    _register_middleware(app)
    _register_exception_handlers(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id") or str(__import__("uuid").uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ConfigNotFoundError)
    async def _config_not_found(request: Request, exc: ConfigNotFoundError) -> JSONResponse:
        return _error_response(request, 404, "config_not_found", f"config not found: {exc}")

    @app.exception_handler(PromptTemplateError)
    async def _prompt_template(request: Request, exc: PromptTemplateError) -> JSONResponse:
        return _error_response(request, 422, "prompt_template_error", str(exc))

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(request, 422, "invalid_request", str(exc))

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
        return _error_response(request, exc.status_code, "http_error", message)

    @app.exception_handler(ProviderAuthError)
    async def _auth(request: Request, exc: ProviderAuthError) -> JSONResponse:
        return _provider_response(
            request,
            502,
            exc,
            "provider_auth_failed",
            prefix="provider auth failed",
        )

    @app.exception_handler(ProviderRateLimitError)
    async def _rate(request: Request, exc: ProviderRateLimitError) -> JSONResponse:
        return _provider_response(request, 429, exc, "provider_rate_limited")

    @app.exception_handler(ProviderQuotaError)
    async def _quota(request: Request, exc: ProviderQuotaError) -> JSONResponse:
        return _provider_response(request, 402, exc, "provider_quota_exhausted")

    @app.exception_handler(ProviderPermissionError)
    async def _permission(request: Request, exc: ProviderPermissionError) -> JSONResponse:
        return _provider_response(request, 403, exc, "provider_permission_denied")

    @app.exception_handler(ProviderBadRequestError)
    async def _bad(request: Request, exc: ProviderBadRequestError) -> JSONResponse:
        return _provider_response(request, 400, exc, "provider_bad_request")

    @app.exception_handler(ProviderUpstreamError)
    async def _upstream(request: Request, exc: ProviderUpstreamError) -> JSONResponse:
        return _provider_response(request, 502, exc, "provider_upstream_error")

    @app.exception_handler(ProviderError)
    async def _provider(request: Request, exc: ProviderError) -> JSONResponse:
        return _provider_response(request, 502, exc, "provider_error")

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected request failure", exc_info=exc)
        return _error_response(request, 500, "internal_error", "internal server error")


def _provider_response(
    request: Request,
    status_code: int,
    exc: ProviderError,
    code: str,
    *,
    prefix: str | None = None,
) -> JSONResponse:
    message = f"{prefix}: {exc}" if prefix else str(exc)
    details: dict[str, Any] = {}
    if exc.details:
        details["provider"] = _sanitize_error_details(exc.details)
    return _error_response(
        request,
        status_code,
        code,
        message,
        details=details or None,
        headers=exc.headers,
    )


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
    return str(__import__("uuid").uuid4())


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


def _sanitize_error_details(details: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in details.items():
        lower = key.lower()
        if lower in {"body", "authorization", "api_key", "apikey", "token"}:
            continue
        if isinstance(value, dict):
            sanitized[key] = _sanitize_error_details(value)
        elif isinstance(value, list):
            sanitized[key] = [_sanitize_error_value(item) for item in value]
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_error_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _sanitize_error_details(value)
    if isinstance(value, list):
        return [_sanitize_error_value(item) for item in value]
    return value


app = create_app()
