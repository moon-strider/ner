from __future__ import annotations

from typing import Protocol, runtime_checkable

from ner_service.schemas import EntityLabel, RawEntities


class ProviderError(Exception):
    pass


class ProviderAuthError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderBadRequestError(ProviderError):
    pass


class ProviderUpstreamError(ProviderError):
    pass


@runtime_checkable
class NerProvider(Protocol):
    name: str
    model: str

    async def extract(self, text: str, labels: list[EntityLabel]) -> RawEntities: ...

    async def aclose(self) -> None: ...
