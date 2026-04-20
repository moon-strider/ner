from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ner_service.config_store import PreparedNERConfig
from ner_service.schemas import RawEntities


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details or {}
        self.headers = headers or {}


class ProviderAuthError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderQuotaError(ProviderError):
    pass


class ProviderPermissionError(ProviderError):
    pass


class ProviderBadRequestError(ProviderError):
    pass


class ProviderUpstreamError(ProviderError):
    pass


@runtime_checkable
class NerProvider(Protocol):
    name: str
    model: str

    async def extract(
        self,
        text: str,
        *,
        prepared: PreparedNERConfig,
        system_prompt: str,
    ) -> RawEntities: ...

    async def aclose(self) -> None: ...
