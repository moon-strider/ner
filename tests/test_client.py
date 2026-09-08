import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients/python/ner-client"))
from ner_client import Client
from ner_client.api.default.extract_v1_extract_post import asyncio as extract_async
from ner_client.api.default.extract_v1_extract_post import sync as extract_sync
from ner_client.models import (
    EntityLabel,
    ErrorEnvelope,
    ExtractEnvelope,
    ExtractRequest,
    NERConfig,
)

from ner_service.config import Settings
from ner_service.main import create_app
from ner_service.service import NerService
from tests.test_http_contract import HttpFakeProvider


def test_client_does_not_override_runtime_defaults():
    app = create_app(
        Settings(), NerService(HttpFakeProvider(), default_model="configured-model", max_tokens=99)
    )
    with TestClient(app) as transport:
        client = Client(base_url="http://testserver").set_httpx_client(transport)
        config = NERConfig(labels=[EntityLabel(name="PERSON", description="People")])
        assert "model" not in config.to_dict() and "max_tokens" not in config.to_dict()
        response = extract_sync(client=client, body=ExtractRequest(text="Tim Cook", config=config))
        assert isinstance(response, ExtractEnvelope)
        assert response.data.model == "configured-model"
        error = extract_sync(client=client, body=ExtractRequest(text="Hi", config_id="absent"))
        assert isinstance(error, ErrorEnvelope) and error.error.code == "config_not_found"
        invalid = extract_sync(client=client, body=ExtractRequest(text="Hi"))
        assert isinstance(invalid, ErrorEnvelope) and invalid.error.code == "validation_error"


@pytest.mark.asyncio
async def test_async_client_parses_success_and_auth_error():
    app = create_app(Settings(ner_api_key="secret"), NerService(HttpFakeProvider()))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as transport,
    ):
        client = Client(base_url="http://testserver").set_async_httpx_client(transport)
        request = ExtractRequest(
            text="Tim Cook",
            config=NERConfig(labels=[EntityLabel(name="PERSON", description="People")]),
        )
        response = await extract_async(client=client, body=request)
        assert isinstance(response, ErrorEnvelope) and response.error.code == "http_error"
        transport.headers["Authorization"] = "Bearer secret"
        response = await extract_async(client=client, body=request)
        assert isinstance(response, ExtractEnvelope)
