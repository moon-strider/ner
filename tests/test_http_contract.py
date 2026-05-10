from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from ner_service.config import RuntimeLimits, Settings
from ner_service.main import create_app
from ner_service.providers.base import ProviderRateLimitError
from ner_service.schemas import RawEntities, RawEntity
from ner_service.service import NerService
from ner_service.telemetry import setup_tracing


class HttpFakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(
        self,
        *,
        error: Exception | None = None,
        error_by_text: dict[str, Exception] | None = None,
    ) -> None:
        self.error = error
        self.error_by_text = error_by_text or {}
        self.calls: list[dict[str, Any]] = []

    async def extract(self, text: str, *, prepared: Any, system_prompt: str) -> RawEntities:
        self.calls.append({"text": text, "prepared": prepared, "system_prompt": system_prompt})
        if text in self.error_by_text:
            raise self.error_by_text[text]
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
        response = client.get("/v1/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "provider": "fake", "model": "fake-model"}


def test_extract_returns_success_envelope_with_request_id() -> None:
    with _client() as client:
        config_response = client.post(
            "/v1/configs",
            json={"labels": [{"name": "PERSON", "description": "People"}]},
        )
        config_id = config_response.json()["id"]
        response = client.post(
            "/v1/extract",
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
            "/v1/extract",
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
            "/v1/configs",
            json={"labels": [{"name": "PERSON", "description": "People"}]},
        )
        response = client.post(
            "/v1/extract",
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


def test_batch_extract_returns_mixed_results() -> None:
    provider = HttpFakeProvider(
        error_by_text={
            "Grace Hopper": ProviderRateLimitError(
                "provider rate limit exceeded",
                headers={"retry-after": "5"},
            )
        }
    )
    with _client(provider) as client:
        config_response = client.post(
            "/v1/configs",
            json={"labels": [{"name": "PERSON", "description": "People"}]},
        )
        config_id = config_response.json()["id"]
        response = client.post(
            "/v1/batch/extract",
            json={
                "items": [
                    {"text": "Tim Cook", "config_id": config_id},
                    {"text": "Grace Hopper", "config_id": config_id},
                ]
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert len(payload["items"]) == 2
    assert payload["meta"]["total"] == 2
    assert payload["meta"]["succeeded"] == 1
    assert payload["meta"]["failed"] == 1
    assert payload["meta"]["latency_ms"] >= 0
    assert payload["items"][0]["index"] == 0
    assert payload["items"][0]["data"]["data"]["provider"] == "fake"
    assert payload["items"][0]["meta"]["attempts"] == 2
    assert payload["items"][1]["index"] == 1
    assert payload["items"][1]["error"] == {
        "code": "ProviderRateLimitError",
        "message": "provider rate limit exceeded",
    }
    assert payload["items"][1]["meta"]["attempts"] == 0


def test_setup_tracing_marks_app_initialized() -> None:
    app = create_app(
        settings=Settings(cerebras_api_key="test"),
        service=NerService(HttpFakeProvider()),
    )

    assert app.state.tracing_initialized is True
    assert app.state.tracer_provider is not None


def test_setup_tracing_with_endpoint_sets_provider() -> None:
    app = create_app(
        settings=Settings(cerebras_api_key="test", otel_endpoint="http://otel.example/v1/traces"),
        service=NerService(HttpFakeProvider()),
    )

    setup_tracing(app)

    assert app.state.tracing_initialized is True
    assert app.state.tracer_provider is not None


def test_metrics_endpoint_is_exposed() -> None:
    with _client() as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "ner_extraction_total" in response.text
