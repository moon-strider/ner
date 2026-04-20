from __future__ import annotations

from ner_service.offsets import attach_offsets
from ner_service.providers.base import NerProvider
from ner_service.schemas import ExtractRequest, ExtractResponse


class NerService:
    def __init__(self, provider: NerProvider) -> None:
        self._provider = provider

    @property
    def provider(self) -> NerProvider:
        return self._provider

    async def extract(self, request: ExtractRequest) -> ExtractResponse:
        raw = await self._provider.extract(request.text, request.labels)
        entities = attach_offsets(request.text, raw.entities)
        return ExtractResponse(
            entities=entities,
            model=self._provider.model,
            provider=self._provider.name,
            usage=raw.usage,
        )

    async def aclose(self) -> None:
        await self._provider.aclose()
