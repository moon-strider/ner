from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ner_service.config import Settings, get_settings
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
from ner_service.schemas import ExtractRequest, ExtractResponse
from ner_service.service import NerService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else get_settings()
    app.state.settings = settings
    provider = get_provider(settings)
    app.state.service = NerService(provider, max_tokens=settings.max_tokens)
    try:
        yield
    finally:
        await app.state.service.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="NER Service",
        version="1.0",
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings

    _register_routes(app)
    _register_exception_handlers(app)
    return app


def _get_service(request: Request) -> NerService:
    service: NerService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return service


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/providers")
    async def providers(request: Request) -> dict[str, Any]:
        service = _get_service(request)
        return {"provider": service.provider.name, "model": service.provider.model}

    @app.post("/extract", response_model=ExtractResponse, response_model_exclude_none=True)
    async def extract(
        payload: ExtractRequest,
        service: NerService = Depends(_get_service),
    ) -> ExtractResponse:
        return await service.extract(payload)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProviderAuthError)
    async def _auth(_: Request, exc: ProviderAuthError) -> JSONResponse:
        return _provider_response(502, exc, prefix="provider auth failed")

    @app.exception_handler(ProviderRateLimitError)
    async def _rate(_: Request, exc: ProviderRateLimitError) -> JSONResponse:
        return _provider_response(429, exc)

    @app.exception_handler(ProviderQuotaError)
    async def _quota(_: Request, exc: ProviderQuotaError) -> JSONResponse:
        return _provider_response(402, exc)

    @app.exception_handler(ProviderPermissionError)
    async def _permission(_: Request, exc: ProviderPermissionError) -> JSONResponse:
        return _provider_response(403, exc)

    @app.exception_handler(ProviderBadRequestError)
    async def _bad(_: Request, exc: ProviderBadRequestError) -> JSONResponse:
        return _provider_response(400, exc)

    @app.exception_handler(ProviderUpstreamError)
    async def _upstream(_: Request, exc: ProviderUpstreamError) -> JSONResponse:
        return _provider_response(502, exc)

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
        return _provider_response(502, exc)


def _provider_response(
    status_code: int,
    exc: ProviderError,
    *,
    prefix: str | None = None,
) -> JSONResponse:
    message = f"{prefix}: {exc}" if prefix else str(exc)
    detail: dict[str, Any] = {"message": message}
    if exc.details:
        detail["provider"] = exc.details
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=exc.headers)


app = create_app()
