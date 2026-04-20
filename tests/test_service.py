from __future__ import annotations

from ner_service.schemas import EntityLabel, ExtractRequest, RawEntities, RawEntity
from ner_service.service import NerService


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def extract(
        self,
        text: str,
        labels: list[EntityLabel],
        *,
        require_offsets: bool,
        retries: int,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> RawEntities:
        self.calls.append(
            {
                "text": text,
                "labels": labels,
                "require_offsets": require_offsets,
                "retries": retries,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            }
        )
        return RawEntities(
            entities=[
                RawEntity(text="Tim Cook", label="PERSON"),
                RawEntity(text="Berlin", label="LOCATION"),
            ],
            usage={"total_tokens": 7},
        )

    async def aclose(self) -> None:
        return None


def _labels() -> list[EntityLabel]:
    return [
        EntityLabel(name="PERSON", description="People"),
        EntityLabel(name="LOCATION", description="Places"),
    ]


async def test_extract_defaults_to_dictionary_entities_without_offsets() -> None:
    provider = FakeProvider()
    service = NerService(provider, max_tokens=777)

    response = await service.extract(
        ExtractRequest(text="Tim Cook visited Berlin.", labels=_labels()),
        reasoning_effort="low",
    )

    assert [(e.text, e.label, e.start, e.end) for e in response.entities] == [
        ("Tim Cook", "PERSON", None, None),
        ("Berlin", "LOCATION", None, None),
    ]
    assert provider.calls[0]["require_offsets"] is False
    assert provider.calls[0]["retries"] == 3
    assert provider.calls[0]["max_tokens"] == 777
    assert provider.calls[0]["reasoning_effort"] == "low"
    assert response.usage == {"total_tokens": 7}


async def test_extract_can_require_offsets_and_override_max_tokens() -> None:
    provider = FakeProvider()
    service = NerService(provider, max_tokens=777)

    response = await service.extract(
        ExtractRequest(
            text="Tim Cook visited Berlin.",
            labels=_labels(),
            require_offsets=True,
            retries=2,
            max_tokens=42,
        )
    )

    assert [(e.text, e.label, e.start, e.end) for e in response.entities] == [
        ("Tim Cook", "PERSON", 0, 8),
        ("Berlin", "LOCATION", 17, 23),
    ]
    assert provider.calls[0]["require_offsets"] is True
    assert provider.calls[0]["retries"] == 2
    assert provider.calls[0]["max_tokens"] == 42
