from __future__ import annotations

from ner_service.config import Settings
from ner_service.providers.base import NerProvider
from ner_service.providers.cerebras import CerebrasProvider


def get_provider(settings: Settings) -> NerProvider:
    match settings.ner_provider.lower():
        case "cerebras":
            if settings.cerebras_api_key is None:
                raise RuntimeError("CEREBRAS_API_KEY is required when NER_PROVIDER=cerebras")
            return CerebrasProvider(
                api_key=settings.cerebras_api_key.get_secret_value(),
                model=settings.ner_model,
                timeout=settings.request_timeout_s,
                max_retries=settings.transport_retries,
            )
        case other:
            raise ValueError(f"unknown provider: {other!r}")
