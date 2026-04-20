from __future__ import annotations

import pytest

from ner_service.schema_builder import (
    build_ner_json_schema,
    build_response_format,
    build_system_prompt,
)
from ner_service.schemas import EntityLabel


def _labels(*names: str) -> list[EntityLabel]:
    return [EntityLabel(name=n, description=f"desc {n}") for n in names]


def test_schema_has_strict_shape() -> None:
    schema = build_ner_json_schema(_labels("PERSON", "LOCATION"))

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["entities"]

    items = schema["properties"]["entities"]["items"]
    assert items["type"] == "object"
    assert items["additionalProperties"] is False
    assert set(items["required"]) == {"text", "label"}
    assert items["properties"]["text"] == {"type": "string"}
    assert items["properties"]["label"]["enum"] == ["PERSON", "LOCATION"]


def test_schema_enum_preserves_order() -> None:
    schema = build_ner_json_schema(_labels("ORG", "DATE", "MONEY"))
    assert schema["properties"]["entities"]["items"]["properties"]["label"]["enum"] == [
        "ORG",
        "DATE",
        "MONEY",
    ]


def test_schema_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="at least one label"):
        build_ner_json_schema([])


def test_schema_rejects_duplicate_labels() -> None:
    dup = [
        EntityLabel(name="PERSON", description="a"),
        EntityLabel(name="PERSON", description="b"),
    ]
    with pytest.raises(ValueError, match="unique"):
        build_ner_json_schema(dup)


def test_response_format_wraps_schema() -> None:
    rf = build_response_format(_labels("PERSON"))
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "ner_extraction"
    assert rf["json_schema"]["schema"]["required"] == ["entities"]


def test_system_prompt_includes_label_names_and_descriptions() -> None:
    prompt = build_system_prompt(
        [
            EntityLabel(name="PERSON", description="People, real or fictional"),
            EntityLabel(name="LOCATION", description="Cities, countries"),
        ]
    )
    assert "PERSON: People, real or fictional" in prompt
    assert "LOCATION: Cities, countries" in prompt
    assert "named entity recognition" in prompt.lower()
