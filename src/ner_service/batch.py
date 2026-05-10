from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from ner_service.schemas import Entity, ExtractRequest, ExtractResponse
from ner_service.service import NerService


class BatchExtractRequest(BaseModel):
    items: list[ExtractRequest] = Field(..., min_length=1, max_length=100)


class BatchExtractItem(BaseModel):
    index: int
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    meta: dict[str, Any]


class BatchExtractResponse(BaseModel):
    items: list[BatchExtractItem]


async def bulk_extract(
    service: NerService,
    requests: list[ExtractRequest],
    *,
    concurrency: int = 10,
) -> BatchExtractResponse:
    sem = asyncio.Semaphore(concurrency)
    started_all = time.perf_counter()

    async def _one(idx: int, req: ExtractRequest) -> BatchExtractItem:
        async with sem:
            started = time.perf_counter()
            try:
                response = await service.extract(req)
                latency_ms = (time.perf_counter() - started) * 1000
                return BatchExtractItem(
                    index=idx,
                    data={
                        "entities": [
                            {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                            for e in response.entities
                        ],
                        "model": response.model,
                        "provider": response.provider,
                        "usage": response.usage,
                    },
                    meta={
                        "latency_ms": latency_ms,
                        "attempts": response.attempts,
                        "warnings": response.warnings,
                    },
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                return BatchExtractItem(
                    index=idx,
                    error={
                        "code": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    meta={
                        "latency_ms": latency_ms,
                        "attempts": 0,
                        "warnings": [],
                    },
                )

    tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(requests)]
    results = await asyncio.gather(*tasks)

    total_latency_ms = (time.perf_counter() - started_all) * 1000
    response = BatchExtractResponse(items=results)
    response.total_latency_ms = total_latency_ms  # type: ignore
    return response
