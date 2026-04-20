from __future__ import annotations

import pytest
from pydantic import ValidationError

from ner_service.schemas import EntityLabel, ExtractRequest


def test_accepts_minimal_valid_request() -> None:
    req = ExtractRequest(
        text="Hello",
        labels=[EntityLabel(name="PERSON", description="people")],
    )
    assert req.text == "Hello"
    assert req.labels[0].name == "PERSON"


def test_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        ExtractRequest(text="", labels=[EntityLabel(name="X", description="d")])


def test_rejects_empty_labels() -> None:
    with pytest.raises(ValidationError):
        ExtractRequest(text="Hi", labels=[])


def test_rejects_duplicate_label_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ExtractRequest(
            text="Hi",
            labels=[
                EntityLabel(name="PERSON", description="a"),
                EntityLabel(name="PERSON", description="b"),
            ],
        )


@pytest.mark.parametrize("bad", ["person", "Person", "PER-SON", "1PER", "", "PER SON"])
def test_rejects_invalid_label_name(bad: str) -> None:
    with pytest.raises(ValidationError):
        EntityLabel(name=bad, description="d")


def test_rejects_empty_description() -> None:
    with pytest.raises(ValidationError):
        EntityLabel(name="PERSON", description="")


def test_rejects_text_over_limit() -> None:
    with pytest.raises(ValidationError):
        ExtractRequest(
            text="x" * 32_001,
            labels=[EntityLabel(name="X", description="d")],
        )


def test_rejects_too_many_labels() -> None:
    many = [EntityLabel(name=f"L{i}", description="d") for i in range(51)]
    with pytest.raises(ValidationError):
        ExtractRequest(text="hi", labels=many)
