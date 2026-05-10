from __future__ import annotations

from ner_service.config import Settings
from ner_service.providers.base import NerProvider
from ner_service.providers.openai_compatible import OpenAICompatibleProvider


def _openai_provider(settings: Settings) -> NerProvider:
    from pydantic import SecretStr
    api_key = settings.openai_api_key
    if api_key is None or (isinstance(api_key, SecretStr) and not api_key.get_secret_value()):
        raise RuntimeError("OPENAI_API_KEY is required when NER_PROVIDER=openai")
    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    return OpenAICompatibleProvider(
        api_key=key,
        base_url=settings.openai_base_url or "https://api.openai.com/v1",
        model=settings.ner_model,
        timeout=settings.request_timeout_s,
        max_retries=settings.transport_retries,
        provider_name="openai",
    )


def _anthropic_provider(settings: Settings) -> NerProvider:
    from pydantic import SecretStr
    api_key = settings.anthropic_api_key
    if api_key is None or (isinstance(api_key, SecretStr) and not api_key.get_secret_value()):
        raise RuntimeError("ANTHROPIC_API_KEY is required when NER_PROVIDER=anthropic")
    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    return OpenAICompatibleProvider(
        api_key=key,
        base_url=settings.anthropic_base_url or "https://api.anthropic.com/v1",
        model=settings.ner_model,
        timeout=settings.request_timeout_s,
        max_retries=settings.transport_retries,
        provider_name="anthropic",
    )


def _cerebras_provider(settings: Settings) -> NerProvider:
    from pydantic import SecretStr
    api_key = settings.cerebras_api_key
    if api_key is None or (isinstance(api_key, SecretStr) and not api_key.get_secret_value()):
        raise RuntimeError("CEREBRAS_API_KEY is required when NER_PROVIDER=cerebras")
    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    return OpenAICompatibleProvider(
        api_key=key,
        base_url=settings.cerebras_base_url or "https://api.cerebras.ai/v1",
        model=settings.ner_model,
        timeout=settings.request_timeout_s,
        max_retries=settings.transport_retries,
        provider_name="cerebras",
    )


def _openrouter_provider(settings: Settings) -> NerProvider:
    from pydantic import SecretStr
    api_key = settings.openrouter_api_key
    if api_key is None or (isinstance(api_key, SecretStr) and not api_key.get_secret_value()):
        raise RuntimeError("OPENROUTER_API_KEY is required when NER_PROVIDER=openrouter")
    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    return OpenAICompatibleProvider(
        api_key=key,
        base_url=settings.openrouter_base_url or "https://openrouter.ai/api/v1",
        model=settings.ner_model,
        timeout=settings.request_timeout_s,
        max_retries=settings.transport_retries,
        provider_name="openrouter",
    )


def _vllm_provider(settings: Settings) -> NerProvider:
    base_url = settings.vllm_base_url
    if not base_url:
        raise RuntimeError("VLLM_BASE_URL is required when NER_PROVIDER=vllm")
    api_key = settings.vllm_api_key or "not-needed"
    return OpenAICompatibleProvider(
        api_key=api_key,
        base_url=base_url,
        model=settings.ner_model,
        timeout=settings.request_timeout_s,
        max_retries=settings.transport_retries,
        provider_name="vllm",
    )


_REGISTRY: dict[str, callable] = {
    "openai": _openai_provider,
    "anthropic": _anthropic_provider,
    "cerebras": _cerebras_provider,
    "openrouter": _openrouter_provider,
    "vllm": _vllm_provider,
}


def get_provider(settings: Settings) -> NerProvider:
    provider_id = settings.ner_provider.lower()
    factory = _REGISTRY.get(provider_id)
    if factory is None:
        raise ValueError(f"unknown provider: {provider_id!r}; supported: {sorted(_REGISTRY)}")
    return factory(settings)


def register_provider(name: str, factory: callable) -> None:
    _REGISTRY[name.lower()] = factory
