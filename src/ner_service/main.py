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
    app.state.service = NerService(provider)
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

    @app.post("/extract", response_model=ExtractResponse)
    async def extract(
        payload: ExtractRequest,
        service: NerService = Depends(_get_service),
    ) -> ExtractResponse:
        return await service.extract(payload)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProviderAuthError)
    async def _auth(_: Request, exc: ProviderAuthError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": f"provider auth failed: {exc}"})

    @app.exception_handler(ProviderRateLimitError)
    async def _rate(_: Request, exc: ProviderRateLimitError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(ProviderBadRequestError)
    async def _bad(_: Request, exc: ProviderBadRequestError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ProviderUpstreamError)
    async def _upstream(_: Request, exc: ProviderUpstreamError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})


app = create_app()
