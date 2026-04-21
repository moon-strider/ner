from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from ner_service.config import Settings
from ner_service.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not os.environ.get("CEREBRAS_API_KEY"):
        pytest.skip("CEREBRAS_API_KEY is not set")
    settings = Settings()
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_providers_endpoint(client: TestClient) -> None:
    r = client.get("/providers")
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "cerebras"
    assert data["model"]


def test_extract_recovers_gold_entities(client: TestClient, samples: list[dict]) -> None:
    for sample in samples[:3]:
        cfg = client.post(
            "/configs",
            json={"labels": sample["labels"], "require_offsets": True},
        )
        assert cfg.status_code == 200, cfg.text
        config_id = cfg.json()["id"]

        r = client.post(
            "/extract",
            json={"text": sample["text"], "config_id": config_id},
        )
        assert r.status_code == 200, r.text
        returned = {(e["text"], e["label"]) for e in r.json()["data"]["entities"]}
        gold = {(e["text"], e["label"]) for e in sample["gold"]}
        missing = gold - returned
        assert not missing, f"missing gold entities: {missing} (got {returned})"

        for e in r.json()["data"]["entities"]:
            assert sample["text"][e["start"] : e["end"]] == e["text"]
