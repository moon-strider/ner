from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from cerebras.cloud.sdk import APIStatusError, RateLimitError

from ner_service.config_store import prepare_config
from ner_service.providers.base import ProviderQuotaError, ProviderRateLimitError
from ner_service.providers.cerebras import CerebrasProvider
from ner_service.schemas import EntityLabel, NERConfig


def _labels() -> list[EntityLabel]:
    return [EntityLabel(name="PERSON", description="People")]


def _completion(content: str, total_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=total_tokens - 1,
            completion_tokens=1,
            total_tokens=total_tokens,
        ),
    )


async def test_cerebras_provider_retries_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = CerebrasProvider(api_key="test")
    calls: list[dict[str, Any]] = []
    prepared = prepare_config(NERConfig(labels=_labels(), retries=3, max_tokens=1024))

    async def fake_create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 1:
            return _completion("{", 3)
        return _completion('{"entities":[{"text":"Tim Cook","label":"PERSON"}]}', 5)

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    result = await provider.extract(
        "Tim Cook visited Berlin.",
        prepared=prepared,
        system_prompt="Extract entities.",
    )

    assert [(e.text, e.label) for e in result.entities] == [("Tim Cook", "PERSON")]
    assert result.usage == {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8}
    assert len(calls) == 2
    assert calls[1]["last_output"] == "{"
    assert "invalid JSON" in calls[1]["last_error"]


async def test_cerebras_provider_maps_402_to_quota_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = CerebrasProvider(api_key="test")
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    async def fake_create(**_: Any) -> None:
        request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
        response = httpx.Response(402, request=request)
        raise APIStatusError(
            "Payment required",
            response=response,
            body={"error": {"message": "No credits"}, "status_code": 402},
        )

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    with pytest.raises(ProviderQuotaError) as exc_info:
        await provider.extract(
            "Tim Cook",
            prepared=prepared,
            system_prompt="Extract entities.",
        )

    assert exc_info.value.details["status_code"] == 402


async def test_cerebras_provider_preserves_rate_limit_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CerebrasProvider(api_key="test")
    prepared = prepare_config(NERConfig(labels=_labels(), retries=1, max_tokens=1024))

    async def fake_create(**_: Any) -> None:
        request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
        response = httpx.Response(
            429,
            request=request,
            headers={
                "retry-after": "12",
                "x-ratelimit-remaining-tokens-minute": "0",
            },
        )
        raise RateLimitError(
            "Too many requests",
            response=response,
            body={"error": {"message": "Too many requests"}, "status_code": 429},
        )

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await provider.extract(
            "Tim Cook",
            prepared=prepared,
            system_prompt="Extract entities.",
        )

    assert exc_info.value.headers == {
        "retry-after": "12",
        "x-ratelimit-remaining-tokens-minute": "0",
    }
    assert exc_info.value.details["headers"] == exc_info.value.headers
