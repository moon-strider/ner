from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from ner_service.schemas import ExtractEnvelope, ExtractRequest
from ner_service.service import NerService


class BatchExtractRequest(BaseModel):
    items: list[ExtractRequest] = Field(..., min_length=1, max_length=100)


class BatchExtractItemMeta(BaseModel):
    request_id: str
    latency_ms: float = Field(..., ge=0.0)
    attempts: int = Field(..., ge=0)
    warnings: list[str] = Field(default_factory=list)


class BatchExtractItem(BaseModel):
    index: int
    data: ExtractEnvelope | None = None
    error: dict[str, str] | None = None
    meta: BatchExtractItemMeta


class BatchExtractMeta(BaseModel):
    total: int
    succeeded: int
    failed: int
    latency_ms: float = Field(..., ge=0.0)


class BatchExtractResponse(BaseModel):
    items: list[BatchExtractItem]
    meta: BatchExtractMeta


async def bulk_extract(
    service: NerService,
    requests: list[ExtractRequest],
    *,
    request_id_factory: Callable[[], str],
    envelope_factory: Callable[..., ExtractEnvelope],
    concurrency: int = 10,
) -> BatchExtractResponse:
    semaphore = asyncio.Semaphore(concurrency)
    started_all = time.perf_counter()

    async def _one(index: int, item: ExtractRequest) -> BatchExtractItem:
        async with semaphore:
            started = time.perf_counter()
            request_id = request_id_factory()
            try:
                response = await service.extract(item)
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                return BatchExtractItem(
                    index=index,
                    error={
                        "code": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    meta=BatchExtractItemMeta(
                        request_id=request_id,
                        latency_ms=latency_ms,
                        attempts=0,
                        warnings=[],
                    ),
                )

            latency_ms = (time.perf_counter() - started) * 1000
            envelope = envelope_factory(
                response,
                request_id=request_id,
                latency_ms=latency_ms,
            )
            return BatchExtractItem(
                index=index,
                data=envelope,
                meta=BatchExtractItemMeta(
                    request_id=envelope.meta.request_id,
                    latency_ms=envelope.meta.latency_ms,
                    attempts=envelope.meta.attempts,
                    warnings=envelope.meta.warnings,
                ),
            )

    tasks = [asyncio.create_task(_one(index, item)) for index, item in enumerate(requests)]
    items = await asyncio.gather(*tasks)
    total_latency_ms = (time.perf_counter() - started_all) * 1000
    failed = sum(1 for item in items if item.error is not None)
    return BatchExtractResponse(
        items=items,
        meta=BatchExtractMeta(
            total=len(items),
            succeeded=len(items) - failed,
            failed=failed,
            latency_ms=total_latency_ms,
        ),
    )
