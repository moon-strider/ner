import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ner_service.config import Settings
from ner_service.main import create_app
from ner_service.providers.base import ProviderAuthError
from ner_service.schemas import NERConfigPatch
from ner_service.service import NerService
from ner_service.stores import MemoryStore, SQLiteStore
from tests.test_http_contract import HttpFakeProvider
from tests.test_service import FakeProvider, _config


@pytest.mark.parametrize(
    "headers, status",
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Authorization": "Bearer secret"}, 200),
    ],
)
def test_optional_auth(headers, status):
    app = create_app(Settings(ner_api_key="secret"), NerService(HttpFakeProvider()))
    with TestClient(app) as client:
        response = client.get("/v1/configs", headers=headers)
        assert response.status_code == status
        if status == 401:
            assert response.headers["www-authenticate"] == "Bearer"
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/ready").status_code == 200


@pytest.mark.parametrize("path", ["/v1/extract", "/v1/batch/extract"])
def test_provider_message_cannot_leak_through_any_endpoint(path):
    app = create_app(
        Settings(),
        NerService(
            HttpFakeProvider(
                error=ProviderAuthError(
                    "secret upstream API key",
                    details={"last_error": "private text", "body": "private body"},
                )
            )
        ),
    )
    item = {"text": "Hello", "config": {"labels": [{"name": "PERSON", "description": "People"}]}}
    with TestClient(app) as client:
        response = client.post(path, json={"items": [item]} if "batch" in path else item)
    assert "secret" not in response.text and "private" not in response.text
    assert "provider_auth_failed" in response.text


def test_internal_batch_error_is_generic():
    app = create_app(Settings(), NerService(HttpFakeProvider(error=RuntimeError("private path"))))
    with TestClient(app) as client:
        response = client.post(
            "/v1/batch/extract",
            json={
                "items": [
                    {
                        "text": "Hello",
                        "config": {"labels": [{"name": "PERSON", "description": "People"}]},
                    }
                ]
            },
        )
    assert response.json()["items"][0]["error"] == {
        "code": "internal_error",
        "message": "internal server error",
    }


def test_unknown_route_and_method_use_error_envelopes():
    with TestClient(create_app(Settings(), NerService(HttpFakeProvider()))) as client:
        for response in [client.get("/missing"), client.delete("/v1/health")]:
            assert response.status_code in (404, 405)
            assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_body_limit_rejects_declared_and_chunked_bodies():
    with TestClient(
        create_app(Settings(max_request_body_bytes=10), NerService(HttpFakeProvider()))
    ) as client:
        for content in [b"x" * 11, iter([b"x" * 6, b"x" * 5])]:
            response = client.post("/v1/extract", content=content)
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "request_too_large"


async def test_body_boundary_counts_individual_asgi_chunks():
    from ner_service.middleware import RequestBoundaryMiddleware

    called = []

    async def downstream(scope, receive, send):
        called.append(True)

    messages = iter(
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"78901", "more_body": False},
        ]
    )

    async def receive():
        return next(messages)

    sent = []

    async def send(message):
        sent.append(message)

    await RequestBoundaryMiddleware(downstream, 10)(
        {"type": "http", "headers": [], "state": {}}, receive, send
    )
    assert not called
    assert sent[0]["status"] == 413


@pytest.mark.parametrize("request_id", ["a" * 129, "bad id", "line\nbreak"])
def test_unsafe_request_ids_are_replaced(request_id):
    with TestClient(create_app(Settings(), NerService(HttpFakeProvider()))) as client:
        response = client.get("/v1/health", headers={"x-request-id": request_id})
    assert response.status_code == 200
    assert response.headers["x-request-id"] != request_id
    assert len(response.headers["x-request-id"]) == 36


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "true"])
def test_invalid_token_pricing_is_rejected(value):
    settings = Settings(
        token_pricing_json=('{"m":{"input_per_million":' + value + ',"output_per_million":1}}')
    )
    with pytest.raises(ValueError):
        settings.token_pricing()


def test_empty_auth_key_is_configuration_error():
    with pytest.raises(ValidationError):
        Settings(ner_api_key="")


async def test_sqlite_memory_database_survives_operations():
    store = SQLiteStore(":memory:")
    try:
        await store.set("a", {"labels": []})
        assert await store.get("a") == {"labels": []}
        assert len(await store.list()) == 1
        await store.delete("a")
        assert await store.get("a") is None
    finally:
        await store.aclose()


async def test_sqlite_persists_after_reopen_and_concurrent_writes(tmp_path):
    path = str(tmp_path / "configs.db")
    store = SQLiteStore(path)
    await asyncio.gather(*(store.set(str(i), {"value": i}) for i in range(20)))
    await store.aclose()
    reopened = SQLiteStore(path)
    try:
        assert len(await reopened.list()) == 20
        assert await reopened.get("7") == {"value": 7}
    finally:
        await reopened.aclose()


async def test_store_does_not_share_mutable_state():
    store = MemoryStore()
    data = {"nested": {"x": 1}}
    await store.set("a", data)
    data["nested"]["x"] = 2
    result = await store.get("a")
    result["nested"]["x"] = 3
    assert await store.get("a") == {"nested": {"x": 1}}


async def test_concurrent_patches_do_not_lose_unrelated_fields():
    service = NerService(FakeProvider())
    record = await service.create_config(_config())
    await asyncio.gather(
        service.patch_config(record.id, NERConfigPatch(max_tokens=42)),
        service.patch_config(record.id, NERConfigPatch(case_sensitive=False)),
    )
    result = (await service.get_config(record.id)).config
    assert result.max_tokens == 42
    assert result.case_sensitive is False


def test_tracing_reads_environment_for_default_factory(monkeypatch):
    monkeypatch.setenv("OTEL_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
    app = create_app()
    assert app.state.settings.otel_endpoint == "http://127.0.0.1:4318/v1/traces"
    app.state.tracer_provider.shutdown()


async def test_stored_config_is_revalidated_against_current_limits():
    from ner_service.config import RuntimeLimits

    store = MemoryStore()
    first = NerService(FakeProvider(), config_store=store)
    record = await first.create_config(_config(max_tokens=1024))
    second = NerService(
        FakeProvider(), config_store=store, limits=RuntimeLimits(max_output_tokens=512)
    )
    from ner_service.schemas import ExtractRequest

    with pytest.raises(ValueError, match="max_tokens"):
        await second.extract(ExtractRequest(text="Hi", config_id=record.id))
