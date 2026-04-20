from __future__ import annotations

from ner_service.offsets import attach_offsets
from ner_service.providers.base import NerProvider
from ner_service.schemas import Entity, ExtractRequest, ExtractResponse


class NerService:
    def __init__(self, provider: NerProvider, *, max_tokens: int = 1024) -> None:
        self._provider = provider
        self._max_tokens = max_tokens

    @property
    def provider(self) -> NerProvider:
        return self._provider

    async def extract(
        self,
        request: ExtractRequest,
        *,
        reasoning_effort: str | None = None,
    ) -> ExtractResponse:
        raw = await self._provider.extract(
            request.text,
            request.labels,
            require_offsets=request.require_offsets,
            retries=request.retries,
            max_tokens=request.max_tokens or self._max_tokens,
            reasoning_effort=reasoning_effort,
        )
        entities = (
            attach_offsets(request.text, raw.entities)
            if request.require_offsets
            else [Entity(text=e.text, label=e.label) for e in raw.entities]
        )
        return ExtractResponse(
            entities=entities,
            model=self._provider.model,
            provider=self._provider.name,
            usage=raw.usage,
        )

    async def aclose(self) -> None:
        await self._provider.aclose()
