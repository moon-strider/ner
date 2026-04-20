from __future__ import annotations

import pytest
from pydantic import ValidationError

from ner_service.schemas import EntityLabel, ExtractRequest, NERConfig


def _config() -> NERConfig:
    return NERConfig(labels=[EntityLabel(name="PERSON", description="people")])


def test_accepts_valid_config_defaults() -> None:
    cfg = _config()
    assert cfg.model == "llama3.1-8b"
    assert cfg.require_offsets is False
    assert cfg.case_sensitive is True
    assert cfg.retries == 3
    assert cfg.max_tokens == 1024
    assert cfg.reasoning_effort is None


def test_accepts_config_id_extract_request() -> None:
    req = ExtractRequest(text="Hello", config_id="cfg-1")
    assert req.text == "Hello"
    assert req.config_id == "cfg-1"
    assert req.prompt_payload == {}


def test_accepts_inline_config_extract_request() -> None:
    req = ExtractRequest(text="Hello", config=_config(), prompt_payload={"number": 7})
    assert req.config is not None
    assert req.prompt_payload == {"number": 7}


def test_rejects_extract_without_config_source() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ExtractRequest(text="Hi")


def test_rejects_extract_with_two_config_sources() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ExtractRequest(text="Hi", config_id="cfg-1", config=_config())


def test_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        ExtractRequest(text="", config_id="cfg-1")


def test_rejects_empty_labels() -> None:
    with pytest.raises(ValidationError):
        NERConfig(labels=[])


def test_rejects_duplicate_label_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        NERConfig(
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
        ExtractRequest(text="x" * 32_001, config_id="cfg-1")


def test_rejects_too_many_labels() -> None:
    many = [EntityLabel(name=f"L{i}", description="d") for i in range(51)]
    with pytest.raises(ValidationError):
        NERConfig(labels=many)


def test_rejects_invalid_retries() -> None:
    with pytest.raises(ValidationError):
        NERConfig(labels=[EntityLabel(name="PERSON", description="people")], retries=0)


def test_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValidationError):
        NERConfig(labels=[EntityLabel(name="PERSON", description="people")], max_tokens=0)
