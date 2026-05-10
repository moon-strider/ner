from __future__ import annotations

from pathlib import Path

import pytest

from ner_service.cache import MemoryCache, ResultCache
from ner_service.config_store import ConfigNotFoundError, PromptTemplateError
from ner_service.schemas import EntityLabel, ExtractRequest, NERConfig, RawEntities, RawEntity
from ner_service.service import NerService
from ner_service.stores import SQLiteStore


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, entities: list[RawEntity] | None = None) -> None:
        self.calls: list[dict] = []
        self.entities = entities or [
            RawEntity(text="Tim Cook", label="PERSON"),
            RawEntity(text="Berlin", label="LOCATION"),
        ]

    async def extract(
        self,
        text: str,
        *,
        prepared,
        system_prompt: str,
    ) -> RawEntities:
        self.calls.append(
            {
                "text": text,
                "prepared": prepared,
                "system_prompt": system_prompt,
            }
        )
        return RawEntities(entities=self.entities, usage={"total_tokens": 7})

    async def aclose(self) -> None:
        return None


def _labels() -> list[EntityLabel]:
    return [
        EntityLabel(name="PERSON", description="People"),
        EntityLabel(name="LOCATION", description="Places"),
    ]


def _config(**kwargs) -> NERConfig:
    return NERConfig(labels=_labels(), **kwargs)


def _has_aiosqlite() -> bool:
    import importlib.util

    return importlib.util.find_spec("aiosqlite") is not None


async def test_extract_uses_cache_for_duplicate_inline_requests() -> None:
    provider = FakeProvider()
    service = NerService(provider, cache=ResultCache(MemoryCache(), ttl=600))
    request = ExtractRequest(text="Tim Cook visited Berlin.", config=_config())

    first = await service.extract(request)
    second = await service.extract(request)

    assert [(e.text, e.label) for e in first.entities] == [
        (e.text, e.label) for e in second.entities
    ]
    assert first.usage == second.usage
    assert len(provider.calls) == 1


async def test_extract_with_stored_config_returns_dictionary_entities() -> None:
    provider = FakeProvider()
    service = NerService(provider, default_model="default-model", max_tokens=777)
    record = await service.create_config(_config(reasoning_effort="low"))

    response = await service.extract(
        ExtractRequest(text="Tim Cook visited Berlin.", config_id=record.id)
    )

    assert [(e.text, e.label, e.start, e.end) for e in response.entities] == [
        ("Tim Cook", "PERSON", None, None),
        ("Berlin", "LOCATION", None, None),
    ]
    prepared = provider.calls[0]["prepared"]
    assert prepared.config.model == "default-model"
    assert prepared.config.max_tokens == 777
    assert prepared.config.reasoning_effort == "low"
    assert response.model == "default-model"
    assert response.usage == {"total_tokens": 7}


async def test_extract_with_inline_config_can_require_offsets() -> None:
    provider = FakeProvider()
    service = NerService(provider, max_tokens=777)

    response = await service.extract(
        ExtractRequest(
            text="Tim Cook visited Berlin.",
            config=_config(require_offsets=True, retries=2, max_tokens=42),
        )
    )

    assert [(e.text, e.label, e.start, e.end) for e in response.entities] == [
        ("Tim Cook", "PERSON", 0, 8),
        ("Berlin", "LOCATION", 17, 23),
    ]
    prepared = provider.calls[0]["prepared"]
    assert prepared.config.require_offsets is True
    assert prepared.config.retries == 2
    assert prepared.config.max_tokens == 42


async def test_case_insensitive_offsets_return_input_casing() -> None:
    provider = FakeProvider([RawEntity(text="tim cook", label="PERSON")])
    service = NerService(provider)

    response = await service.extract(
        ExtractRequest(
            text="Tim Cook visited Berlin.",
            config=_config(require_offsets=True, case_sensitive=False),
        )
    )

    assert [(e.text, e.start, e.end) for e in response.entities] == [("Tim Cook", 0, 8)]


async def test_case_insensitive_dictionary_canonicalizes_input_casing() -> None:
    provider = FakeProvider([RawEntity(text="tim cook", label="PERSON")])
    service = NerService(provider)

    response = await service.extract(
        ExtractRequest(
            text="Tim Cook visited Berlin.",
            config=_config(case_sensitive=False),
        )
    )

    assert [(e.text, e.label, e.start, e.end) for e in response.entities] == [
        ("Tim Cook", "PERSON", None, None)
    ]


async def test_prompt_template_renders_cfg_schema_and_payload() -> None:
    provider = FakeProvider()
    service = NerService(provider)
    record = await service.create_config(
        _config(system_prompt="schema={cfg.schema}; model={cfg.model}; n={payload.number}; {{ok}}")
    )

    await service.extract(
        ExtractRequest(
            text="Tim Cook visited Berlin.",
            config_id=record.id,
            prompt_payload={"number": 7},
        )
    )

    prompt = provider.calls[0]["system_prompt"]
    assert '"entities"' in prompt
    assert "model=llama3.1-8b" in prompt
    assert "n=7" in prompt
    assert "{ok}" in prompt


async def test_prompt_template_missing_payload_value_raises() -> None:
    service = NerService(FakeProvider())
    record = await service.create_config(_config(system_prompt="n={payload.number}"))

    with pytest.raises(PromptTemplateError, match="missing"):
        await service.extract(ExtractRequest(text="Tim Cook", config_id=record.id))


async def test_config_store_crud() -> None:
    service = NerService(FakeProvider())
    created = await service.create_config(_config(model="m1"))

    assert (await service.get_config(created.id)).config.model == "m1"
    assert (await service.list_configs())[0].id == created.id

    replaced = await service.put_config(created.id, _config(model="m2"))
    assert replaced.config.model == "m2"

    patched = await service.patch_config(created.id, _config_patch({"max_tokens": 123}))
    assert patched.config.model == "m2"
    assert patched.config.max_tokens == 123

    await service.delete_config(created.id)
    with pytest.raises(ConfigNotFoundError):
        await service.get_config(created.id)


@pytest.mark.skipif(not _has_aiosqlite(), reason="aiosqlite not installed")
async def test_sqlite_config_store_crud(tmp_path: Path) -> None:
    service = NerService(FakeProvider(), config_store=SQLiteStore(str(tmp_path / "configs.db")))
    assert (await service.ready())["config_store"] == {
        "backend": "sqlite",
        "status": "ok",
        "path": str(tmp_path / "configs.db"),
    }
    created = await service.create_config(_config(model="m1"))

    fetched = await service.get_config(created.id)
    listed = await service.list_configs()
    assert fetched.config.model == "m1"
    assert listed[0].id == created.id

    replaced = await service.put_config(created.id, _config(model="m2"))
    assert replaced.config.model == "m2"

    patched = await service.patch_config(created.id, _config_patch({"max_tokens": 123}))
    assert patched.config.model == "m2"
    assert patched.config.max_tokens == 123

    await service.delete_config(created.id)
    with pytest.raises(ConfigNotFoundError):
        await service.get_config(created.id)


def _config_patch(data: dict):
    from ner_service.schemas import NERConfigPatch

    return NERConfigPatch.model_validate(data)


async def test_estimated_cost_metric_is_exposed() -> None:
    from ner_service.config import TokenPricing

    provider = FakeProvider()
    provider.entities = [RawEntity(text="Tim Cook", label="PERSON")]
    service = NerService(
        provider,
        token_pricing={"llama3.1-8b": TokenPricing(input_per_million=1.0, output_per_million=2.0)},
    )

    await service.extract(ExtractRequest(text="Tim Cook", config=_config()))

    from prometheus_client import generate_latest

    metrics = generate_latest().decode()
    assert "ner_estimated_cost_usd_total" in metrics
    assert "ner_cache_total" in metrics
