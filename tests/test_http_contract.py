from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from ner_service.config import RuntimeLimits, Settings
from ner_service.main import create_app
from ner_service.providers.base import ProviderRateLimitError
from ner_service.schemas import RawEntities, RawEntity
from ner_service.service import NerService


class HttpFakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def extract(self, text: str, *, prepared: Any, system_prompt: str) -> RawEntities:
        self.calls.append({"text": text, "prepared": prepared, "system_prompt": system_prompt})
        if self.error is not None:
            raise self.error
        return RawEntities(
            entities=[RawEntity(text="Tim Cook", label="PERSON")],
            usage={"total_tokens": 11},
            attempts=2,
        )

    async def aclose(self) -> None:
        return None


def _client(
    provider: HttpFakeProvider | None = None,
    *,
    limits: RuntimeLimits | None = None,
) -> TestClient:
    settings = Settings(cerebras_api_key="test")
    service = NerService(provider or HttpFakeProvider(), limits=limits)
    return TestClient(create_app(settings=settings, service=service))


def test_ready_returns_initialized_provider() -> None:
    with _client() as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "provider": "fake", "model": "fake-model"}


def test_extract_returns_success_envelope_with_request_id() -> None:
    with _client() as client:
        config_response = client.post(
            "/configs",
            json={"labels": [{"name": "PERSON", "description": "People"}]},
        )
        config_id = config_response.json()["id"]
        response = client.post(
            "/extract",
            headers={"x-request-id": "req-1"},
            json={"text": "Tim Cook visited Berlin.", "config_id": config_id},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-1"
    payload = response.json()
    assert payload["data"]["entities"] == [{"text": "Tim Cook", "label": "PERSON"}]
    assert payload["data"]["model"] == "llama3.1-8b"
    assert payload["data"]["provider"] == "fake"
    assert payload["data"]["usage"] == {"total_tokens": 11}
    assert payload["meta"]["request_id"] == "req-1"
    assert payload["meta"]["attempts"] == 2
    assert payload["meta"]["warnings"] == []
    assert payload["meta"]["latency_ms"] >= 0


def test_validation_errors_use_error_envelope() -> None:
    with _client(limits=RuntimeLimits(max_text_length=3)) as client:
        response = client.post(
            "/extract",
            headers={"x-request-id": "req-2"},
            json={"text": "too long", "config_id": "missing"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "invalid_request",
            "message": "text length must be <= 3",
            "details": {},
            "request_id": "req-2",
        }
    }


def test_provider_errors_use_redacted_error_envelope() -> None:
    provider = HttpFakeProvider(
        error=ProviderRateLimitError(
            "provider rate limit exceeded",
            details={
                "status_code": 429,
                "body": {"error": {"message": "raw provider body"}},
                "headers": {"retry-after": "3"},
            },
            headers={"retry-after": "3"},
        )
    )
    with _client(provider) as client:
        config_response = client.post(
            "/configs",
            json={"labels": [{"name": "PERSON", "description": "People"}]},
        )
        response = client.post(
            "/extract",
            headers={"x-request-id": "req-3"},
            json={"text": "Tim Cook", "config_id": config_response.json()["id"]},
        )

    payload = response.json()
    assert response.status_code == 429
    assert response.headers["retry-after"] == "3"
    assert payload["error"]["code"] == "provider_rate_limited"
    assert payload["error"]["request_id"] == "req-3"
    assert payload["error"]["details"]["provider"] == {
        "status_code": 429,
        "headers": {"retry-after": "3"},
    }
