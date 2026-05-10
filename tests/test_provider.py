from __future__ import annotations

from typing import Any

import httpx
import pytest

from ner_service.config_store import prepare_config
from ner_service.providers.base import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUpstreamError,
)
from ner_service.providers.openai_compatible import OpenAICompatibleProvider
from ner_service.schemas import EntityLabel, NERConfig


def _labels() -> list[EntityLabel]:
    return [EntityLabel(name="PERSON", description="People")]


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="test-model",
        timeout=5.0,
        max_retries=1,
        provider_name="test",
    )


def _completion_json(content: str, total_tokens: int = 10) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": total_tokens - 1,
            "completion_tokens": 1,
            "total_tokens": total_tokens,
        },
    }


@pytest.mark.asyncio
async def test_extract_returns_entities_on_valid_json(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=2, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json=_completion_json('{"entities":[{"text":"Tim Cook","label":"PERSON"}]}', 10),
    )

    result = await provider.extract(
        "Tim Cook visited Berlin.",
        prepared=prepared,
        system_prompt="Extract entities.",
    )

    assert [(e.text, e.label) for e in result.entities] == [("Tim Cook", "PERSON")]
    assert result.usage == {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}
    assert result.attempts == 1


@pytest.mark.asyncio
async def test_extract_retries_on_invalid_json(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=3, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json=_completion_json("{", 5),
    )
    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json=_completion_json('{"entities":[{"text":"Tim Cook","label":"PERSON"}]}', 10),
    )

    result = await provider.extract(
        "Tim Cook visited Berlin.",
        prepared=prepared,
        system_prompt="Extract entities.",
    )

    assert [(e.text, e.label) for e in result.entities] == [("Tim Cook", "PERSON")]
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_extract_fails_after_all_retries(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=2, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json=_completion_json("garbage", 5),
    )
    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json=_completion_json("still garbage", 5),
    )

    from ner_service.providers.base import ProviderError

    with pytest.raises(ProviderError, match="invalid structured output"):
        await provider.extract(
            "Tim Cook visited Berlin.",
            prepared=prepared,
            system_prompt="Extract entities.",
        )


@pytest.mark.asyncio
async def test_401_raises_provider_auth_error(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        status_code=401,
        json={"error": {"message": "Invalid API key"}},
    )

    with pytest.raises(ProviderAuthError) as exc_info:
        await provider.extract("Tim Cook", prepared=prepared, system_prompt="Extract.")

    assert exc_info.value.details["status_code"] == 401


@pytest.mark.asyncio
async def test_429_raises_provider_rate_limit_error_with_headers(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        status_code=429,
        json={"error": {"message": "Too many requests"}},
        headers={"retry-after": "15", "x-ratelimit-remaining": "0"},
    )

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await provider.extract("Tim Cook", prepared=prepared, system_prompt="Extract.")

    assert exc_info.value.headers == {
        "retry-after": "15",
        "x-ratelimit-remaining": "0",
    }


@pytest.mark.asyncio
async def test_402_raises_provider_quota_error(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        status_code=402,
        json={"error": {"message": "No credits"}},
    )

    with pytest.raises(ProviderQuotaError):
        await provider.extract("Tim Cook", prepared=prepared, system_prompt="Extract.")


@pytest.mark.asyncio
async def test_403_raises_provider_permission_error(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        status_code=403,
        json={"error": {"message": "Permission denied"}},
    )

    with pytest.raises(ProviderPermissionError):
        await provider.extract("Tim Cook", prepared=prepared, system_prompt="Extract.")


@pytest.mark.asyncio
async def test_500_raises_provider_upstream_error(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        status_code=500,
        json={"error": {"message": "Internal server error"}},
    )

    with pytest.raises(ProviderUpstreamError):
        await provider.extract("Tim Cook", prepared=prepared, system_prompt="Extract.")


@pytest.mark.asyncio
async def test_timeout_raises_provider_upstream_error(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    httpx_mock.add_exception(
        url="https://api.example.com/v1/chat/completions",
        exception=httpx.TimeoutException("Request timed out"),
    )

    with pytest.raises(ProviderUpstreamError, match="upstream timeout"):
        await provider.extract("Tim Cook", prepared=prepared, system_prompt="Extract.")


@pytest.mark.asyncio
async def test_400_raises_provider_bad_request_error(httpx_mock: Any) -> None:
    provider = _provider()
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        status_code=400,
        json={"error": {"message": "Bad request"}},
    )

    with pytest.raises(ProviderBadRequestError):
        await provider.extract("Tim Cook", prepared=prepared, system_prompt="Extract.")
