from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ner_service.candidates import Candidate
from ner_service.config_store import PreparedNERConfig, prepare_config
from ner_service.providers.base import ProviderUpstreamError
from ner_service.schemas import Entity, EntityLabel, NERConfig, SpanPipelinePolicy
from ner_service.span_pipeline import JevSpanPipeline
from ner_service.typesafe import (
    NONE_OF_THESE,
    CandidateJudgment,
    JevVerdict,
    parse_response,
)

Handler = Callable[..., JevVerdict]
TWO_TEXT = "Alpha qqxxa Beta"
OVERLAP_TEXT = "Alpha Beta"


class RecordingJevClient:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    async def judge(
        self,
        text: str,
        candidates: list[Candidate],
        labels: list[EntityLabel],
        *,
        window_words: int | None = None,
        negative_focus: bool = False,
    ) -> JevVerdict:
        self.calls.append(
            {
                "text": text,
                "candidates": list(candidates),
                "labels": list(labels),
                "window_words": window_words,
                "negative_focus": negative_focus,
            }
        )
        return self.handler(
            text,
            list(candidates),
            list(labels),
            window_words=window_words,
            negative_focus=negative_focus,
        )

    async def aclose(self) -> None:
        return None


def _labels() -> list[EntityLabel]:
    return [
        EntityLabel(name="PERSON", description="people and characters"),
        EntityLabel(name="ORG", description="companies and institutions"),
    ]


def _prepared(**policy: Any) -> PreparedNERConfig:
    return prepare_config(NERConfig(labels=_labels(), span_pipeline=SpanPipelinePolicy(**policy)))


def _pipeline(handler: Handler) -> tuple[JevSpanPipeline, RecordingJevClient]:
    client = RecordingJevClient(handler)
    return JevSpanPipeline(client), client


def _verdict(
    rows: list[tuple[Candidate, str | None, float]],
    *,
    usage: dict[str, int] | None = None,
    no_judgment: int = 0,
) -> JevVerdict:
    return JevVerdict(
        judgments=tuple(
            CandidateJudgment(
                candidate=candidate,
                label=label,
                probability=probability,
                probabilities={} if label is None else {label: probability},
                confidence=None,
            )
            for candidate, label, probability in rows
        ),
        model="jev-test",
        usage={"input_tokens": 10, "output_tokens": 1} if usage is None else usage,
        no_judgment=no_judgment,
    )


def _accept_all(
    *,
    probability: float = 0.9,
    label: str = "PERSON",
    usage: dict[str, int] | None = None,
    no_judgment: int = 0,
) -> Handler:
    def handler(
        text: str,
        candidates: list[Candidate],
        labels: list[EntityLabel],
        **kwargs: Any,
    ) -> JevVerdict:
        return _verdict(
            [(candidate, label, probability) for candidate in candidates],
            usage=usage,
            no_judgment=no_judgment,
        )

    return handler


def _by_text(mapping: dict[str, float], *, label: str = "PERSON") -> Handler:
    def handler(
        text: str,
        candidates: list[Candidate],
        labels: list[EntityLabel],
        **kwargs: Any,
    ) -> JevVerdict:
        return _verdict([(candidate, label, mapping[candidate.text]) for candidate in candidates])

    return handler


def _unavailable() -> Handler:
    def handler(
        text: str,
        candidates: list[Candidate],
        labels: list[EntityLabel],
        **kwargs: Any,
    ) -> JevVerdict:
        raise ProviderUpstreamError("typesafe unavailable")

    return handler


async def test_threshold_drops_low_probability_and_keeps_boundary() -> None:
    def handler(
        text: str,
        candidates: list[Candidate],
        labels: list[EntityLabel],
        **kwargs: Any,
    ) -> JevVerdict:
        return _verdict([(candidates[0], "PERSON", 0.49), (candidates[1], "PERSON", 0.5)])

    pipeline, client = _pipeline(handler)
    result = await pipeline.extract(
        TWO_TEXT, prepared=_prepared(prefilter="capitalized", min_label_probability=0.5)
    )
    assert [[candidate.text for candidate in call["candidates"]] for call in client.calls] == [
        ["Alpha", "Beta"]
    ]
    assert result.entities == [Entity(text="Beta", label="PERSON", start=12, end=16)]
    assert result.degraded is False


async def test_none_of_these_is_dropped() -> None:
    def handler(
        text: str,
        candidates: list[Candidate],
        labels: list[EntityLabel],
        **kwargs: Any,
    ) -> JevVerdict:
        return _verdict([(candidates[0], NONE_OF_THESE, 0.99), (candidates[1], "PERSON", 0.9)])

    pipeline, _ = _pipeline(handler)
    result = await pipeline.extract(
        TWO_TEXT, prepared=_prepared(prefilter="capitalized", min_label_probability=0.5)
    )
    assert result.entities == [Entity(text="Beta", label="PERSON", start=12, end=16)]


async def test_unknown_label_from_provider_payload_is_dropped() -> None:
    def handler(
        text: str,
        candidates: list[Candidate],
        labels: list[EntityLabel],
        **kwargs: Any,
    ) -> JevVerdict:
        return parse_response(
            {
                "answers": {
                    "c0": {"choice": "LOCATION", "probabilities": {"LOCATION": 0.99}},
                    "c1": {"choice": "PERSON", "probabilities": {"PERSON": 0.9}},
                },
                "model": "jev-test",
            },
            candidates,
            labels,
        )

    pipeline, _ = _pipeline(handler)
    result = await pipeline.extract(
        TWO_TEXT, prepared=_prepared(prefilter="capitalized", min_label_probability=0.5)
    )
    assert result.entities == [Entity(text="Beta", label="PERSON", start=12, end=16)]


async def test_overlapping_candidates_keep_highest_probability_and_sort_by_position() -> None:
    pipeline, client = _pipeline(_by_text({"Alpha": 0.2, "Alpha Beta": 0.9, "Beta": 0.1}))
    result = await pipeline.extract(
        OVERLAP_TEXT, prepared=_prepared(prefilter="capitalized", min_label_probability=0.1)
    )
    assert result.entities == [Entity(text="Alpha Beta", label="PERSON", start=0, end=10)]
    assert [candidate.text for candidate in client.calls[0]["candidates"]] == [
        "Alpha",
        "Alpha Beta",
        "Beta",
    ]

    pipeline, _ = _pipeline(_by_text({"Alpha": 0.9, "Alpha Beta": 0.2, "Beta": 0.9}))
    result = await pipeline.extract(
        OVERLAP_TEXT, prepared=_prepared(prefilter="capitalized", min_label_probability=0.1)
    )
    assert result.entities == [
        Entity(text="Alpha", label="PERSON", start=0, end=5),
        Entity(text="Beta", label="PERSON", start=6, end=10),
    ]


async def test_chunking_sends_single_document_in_three_requests() -> None:
    text = " ".join(f"Zeta{index:02d}Xq fillerq" for index in range(65))
    pipeline, client = _pipeline(_accept_all(probability=0.9))
    result = await pipeline.extract(
        text,
        prepared=_prepared(
            prefilter="capitalized",
            min_label_probability=0.5,
            max_candidates_per_request=30,
        ),
    )
    assert len(client.calls) == 3
    assert [len(call["candidates"]) for call in client.calls] == [30, 30, 5]
    assert {call["text"] for call in client.calls} == {text}
    assert result.usage == {"input_tokens": 30, "output_tokens": 3}
    assert len(result.entities) == 65
    assert result.attempts == 3
    assert result.degraded is False


async def test_warnings_contain_only_counters() -> None:
    text = "Zeta00Xq fillerq Zeta01Xq"

    def handler(
        text: str,
        candidates: list[Candidate],
        labels: list[EntityLabel],
        **kwargs: Any,
    ) -> JevVerdict:
        return _verdict([(candidates[0], "PERSON", 0.9)], no_judgment=1)

    pipeline, _ = _pipeline(handler)
    result = await pipeline.extract(
        text, prepared=_prepared(prefilter="capitalized", min_label_probability=0.5)
    )
    assert result.warnings
    for warning in result.warnings:
        assert "Zeta" not in warning
        assert "fillerq" not in warning
        assert text not in warning
        assert "PERSON" not in warning
        assert "ORG" not in warning
        assert any(character.isdigit() for character in warning)


async def test_degrade_returns_empty_result_instead_of_raising() -> None:
    pipeline, _ = _pipeline(_unavailable())
    result = await pipeline.extract(
        TWO_TEXT,
        prepared=_prepared(prefilter="capitalized", on_unavailable="degrade"),
    )
    assert result.entities == []
    assert result.degraded is True
    assert result.reason == "ProviderUpstreamError"
    assert result.warnings
    assert any("ProviderUpstreamError" in warning for warning in result.warnings)


async def test_fail_propagates_provider_error() -> None:
    pipeline, _ = _pipeline(_unavailable())
    with pytest.raises(ProviderUpstreamError):
        await pipeline.extract(
            TWO_TEXT,
            prepared=_prepared(prefilter="capitalized", on_unavailable="fail"),
        )


async def test_extract_without_policy_raises_value_error() -> None:
    pipeline, client = _pipeline(_accept_all())
    prepared = prepare_config(NERConfig(labels=_labels()))
    with pytest.raises(ValueError, match="span pipeline policy is required"):
        await pipeline.extract(TWO_TEXT, prepared=prepared)
    assert client.calls == []


async def test_cascade_reasks_only_band_candidates_with_negative_focus() -> None:
    text = "Alpha fillerq Beta fillerq Gamma"
    pipeline, client = _pipeline(_by_text({"Alpha": 0.4, "Beta": 0.6, "Gamma": 0.9}))
    result = await pipeline.extract(
        text,
        prepared=_prepared(
            prefilter="capitalized",
            min_label_probability=0.3,
            cascade=True,
            cascade_low=0.5,
            cascade_high=0.7,
            window_words=2,
            cascade_window_words=7,
        ),
    )
    assert len(client.calls) == 2
    assert [candidate.text for candidate in client.calls[0]["candidates"]] == [
        "Alpha",
        "Beta",
        "Gamma",
    ]
    assert client.calls[0]["negative_focus"] is False
    assert client.calls[0]["window_words"] == 2
    assert [candidate.text for candidate in client.calls[1]["candidates"]] == ["Beta"]
    assert client.calls[1]["negative_focus"] is True
    assert client.calls[1]["window_words"] == 7
    assert result.usage == {"input_tokens": 20, "output_tokens": 2}
    assert any("re-asked 1 uncertain candidates" in warning for warning in result.warnings)
    assert len(result.entities) == 3
