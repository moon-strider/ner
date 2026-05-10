from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ner_service.config import Settings
from ner_service.metrics import MetricsCollector, extraction_timer
from ner_service.schemas import (
    ExtractEnvelope,
    ExtractRequest,
    ExtractResponse,
    ExtractResponseData,
    NERConfig,
    NERConfigPatch,
    NERConfigRecord,
    ResponseMeta,
)
from ner_service.service import NerService


def _get_service(request: Request) -> NerService:
    svc: NerService | None = getattr(request.app.state, "service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return svc


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if isinstance(value, str) and value:
        return value
    import uuid
    return str(uuid.uuid4())


def _extract_envelope(
    response: ExtractResponse,
    *,
    request_id: str,
    latency_ms: float,
) -> ExtractEnvelope:
    return ExtractEnvelope(
        data=ExtractResponseData(
            entities=response.entities,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
        ),
        meta=ResponseMeta(
            request_id=request_id,
            latency_ms=latency_ms,
            attempts=response.attempts,
            warnings=response.warnings,
        ),
    )


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    svc: NerService | None = getattr(request.app.state, "service", None)
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if svc is None or settings is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return {
        "status": "ready",
        "provider": svc.provider.name,
        "model": svc.provider.model,
    }


@router.get("/providers")
async def providers(request: Request) -> dict[str, Any]:
    svc = _get_service(request)
    return {"provider": svc.provider.name, "model": svc.provider.model}


@router.post("/configs", response_model=NERConfigRecord)
async def create_config(
    payload: NERConfig,
    svc: NerService = Depends(_get_service),
) -> NERConfigRecord:
    return svc.create_config(payload)


@router.get("/configs", response_model=list[NERConfigRecord])
async def list_configs(
    svc: NerService = Depends(_get_service),
) -> list[NERConfigRecord]:
    return svc.list_configs()


@router.get("/configs/{config_id}", response_model=NERConfigRecord)
async def get_config(
    config_id: str,
    svc: NerService = Depends(_get_service),
) -> NERConfigRecord:
    return svc.get_config(config_id)


@router.put("/configs/{config_id}", response_model=NERConfigRecord)
async def put_config(
    config_id: str,
    payload: NERConfig,
    svc: NerService = Depends(_get_service),
) -> NERConfigRecord:
    return svc.put_config(config_id, payload)


@router.patch("/configs/{config_id}", response_model=NERConfigRecord)
async def patch_config(
    config_id: str,
    payload: NERConfigPatch,
    svc: NerService = Depends(_get_service),
) -> NERConfigRecord:
    return svc.patch_config(config_id, payload)


@router.delete("/configs/{config_id}", status_code=204)
async def delete_config(
    config_id: str,
    svc: NerService = Depends(_get_service),
) -> Response:
    svc.delete_config(config_id)
    return Response(status_code=204)


@router.post("/extract", response_model=ExtractEnvelope, response_model_exclude_none=True)
async def extract(
    request: Request,
    payload: ExtractRequest,
    svc: NerService = Depends(_get_service),
) -> ExtractEnvelope:
    metrics = MetricsCollector()
    with extraction_timer() as timer:
        try:
            response = await svc.extract(payload)
            metrics.record_attempt(
                provider=svc.provider.name,
                model=response.model,
                duration_ms=timer[0],
                success=True,
            )
            metrics.record_tokens(
                provider=svc.provider.name,
                model=response.model,
                usage=response.usage,
            )
        except Exception as exc:
            metrics.record_attempt(
                provider=svc.provider.name,
                model=svc.provider.model,
                duration_ms=timer[0],
                success=False,
            )
            error_type = exc.__class__.__name__
            metrics.record_error(provider=svc.provider.name, error_type=error_type)
            raise
    return _extract_envelope(
        response, request_id=_request_id(request), latency_ms=timer[0]
    )
