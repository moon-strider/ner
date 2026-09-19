from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from ner_service.candidates import Candidate
from ner_service.providers.base import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUpstreamError,
)
from ner_service.schemas import EntityLabel
from ner_service.typesafe import (
    DEFAULT_MODEL,
    NONE_OF_THESE,
    JevClient,
    build_request,
    parse_response,
)

BASE_URL = "https://api.example.test"
SYSTEMONE_URL = f"{BASE_URL}/v1/systemone"
SAMPLE_TEXT = "one two Alice three four five Bob six seven"
THREE_TEXT = "one two Alice three four Bob five six Carol"


def _labels() -> list[EntityLabel]:
    return [
        EntityLabel(name="PERSON", description="people and characters"),
        EntityLabel(name="ORG", description="companies and institutions"),
    ]


def _candidate_at(text: str, surface: str) -> Candidate:
    start = text.index(surface)
    return Candidate(
        text=surface,
        start=start,
        end=start + len(surface),
        source="capitalized",
        priority=1,
    )


def _candidates() -> list[Candidate]:
    return [_candidate_at(SAMPLE_TEXT, "Alice"), _candidate_at(SAMPLE_TEXT, "Bob")]


def _client(**overrides: Any) -> JevClient:
    settings: dict[str, Any] = {"api_key": "secret-key", "base_url": BASE_URL}
    settings.update(overrides)
    return JevClient(**settings)


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "answers": {},
        "model": "jev-response-model",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    body.update(overrides)
    return body


async def test_judge_posts_systemone_request(httpx_mock: HTTPXMock) -> None:
    answers = {
        "c0": {"choice": "PERSON", "probabilities": {"PERSON": 0.9}, "confidence": 0.8},
        "c1": {"choice": "ORG", "probabilities": {"ORG": 0.7}},
    }
    httpx_mock.add_response(url=SYSTEMONE_URL, json=_body(answers=answers))
    client = _client()
    try:
        await client.judge(SAMPLE_TEXT, _candidates(), _labels())
    finally:
        await client.aclose()

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    request = requests[0]
    assert request.headers["authorization"] == "Bearer secret-key"

    payload = json.loads(request.content)
    state = payload["state"]
    assert [key for key in state if key == "labels"] == ["labels"]
    assert state["labels"] == [
        {"name": "PERSON", "description": "people and characters"},
        {"name": "ORG", "description": "companies and institutions"},
    ]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert serialized.count("people and characters") == 1
    assert serialized.count("companies and institutions") == 1

    assert state["text"] == SAMPLE_TEXT
    assert [item["id"] for item in state["candidates"]] == ["c0", "c1"]
    assert state["candidates"][0]["surface"] == "Alice"
    assert state["candidates"][0]["context"] == "one two Alice three four"
    assert state["candidates"][1]["surface"] == "Bob"
    assert state["candidates"][1]["context"] == "four five Bob six seven"

    assert payload["model"] == DEFAULT_MODEL
    assert list(payload["questions"]) == ["c0", "c1"]
    for question in payload["questions"].values():
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"PERSON", "ORG", NONE_OF_THESE}
    assert "Alice" in payload["questions"]["c0"]["instructions"]["question"]
    assert "Bob" in payload["questions"]["c1"]["instructions"]["question"]


def test_build_request_uses_requested_window() -> None:
    payload = build_request(SAMPLE_TEXT, _candidates(), _labels(), window_words=1)
    candidates = payload["state"]["candidates"]
    assert candidates[0]["context"] == "two Alice three"
    assert candidates[1]["context"] == "five Bob six"


def test_build_request_rejects_reserved_label_name() -> None:
    labels = [EntityLabel(name=NONE_OF_THESE, description="reserved")]
    with pytest.raises(ValueError, match="reserved"):
        build_request(SAMPLE_TEXT, _candidates(), labels)


@pytest.mark.parametrize("choice", ["LOCATION", NONE_OF_THESE])
def test_parse_response_rejects_unknown_and_none_choice(choice: str) -> None:
    body = _body(answers={"c0": {"choice": choice, "probabilities": {choice: 0.99}}})
    verdict = parse_response(body, _candidates()[:1], _labels())
    judgment = verdict.judgments[0]
    assert judgment.label is None
    assert judgment.probability == 0.99


def test_parse_response_counts_missing_answers_and_keeps_the_rest() -> None:
    answers = {
        "c0": {"choice": "PERSON", "probabilities": {"PERSON": 0.8}},
        "c2": {"choice": "ORG", "probabilities": {"ORG": 0.7}},
    }
    candidates = [
        _candidate_at(THREE_TEXT, "Alice"),
        _candidate_at(THREE_TEXT, "Bob"),
        _candidate_at(THREE_TEXT, "Carol"),
    ]
    verdict = parse_response(_body(answers=answers), candidates, _labels())
    assert verdict.no_judgment == 1
    assert [judgment.candidate.text for judgment in verdict.judgments] == ["Alice", "Carol"]
    assert [judgment.label for judgment in verdict.judgments] == ["PERSON", "ORG"]
    assert [judgment.probability for judgment in verdict.judgments] == [0.8, 0.7]


def test_parse_response_ignores_non_numeric_probabilities() -> None:
    answers = {
        "c0": {
            "choice": "PERSON",
            "probabilities": {"PERSON": 0.5, "ORG": "high", "OTHER": None, "BAD": [0.1]},
            "confidence": "high",
        }
    }
    verdict = parse_response(_body(answers=answers), _candidates()[:1], _labels())
    judgment = verdict.judgments[0]
    assert judgment.probabilities == {"PERSON": 0.5}
    assert judgment.probability == 0.5
    assert judgment.confidence is None


def test_parse_response_reads_model_usage_and_confidence() -> None:
    answers = {"c0": {"choice": "PERSON", "probabilities": {"PERSON": 0.6}, "confidence": 3}}
    body = _body(
        answers=answers,
        model="jev-9.9.9",
        usage={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16, "bad": "many"},
    )
    verdict = parse_response(body, _candidates()[:1], _labels())
    assert verdict.model == "jev-9.9.9"
    assert verdict.usage == {"input_tokens": 11, "output_tokens": 5}
    assert verdict.judgments[0].confidence == 3.0


def test_parse_response_defaults_model_when_missing() -> None:
    verdict = parse_response({"answers": {}, "model": ""}, _candidates()[:1], _labels())
    assert verdict.model == DEFAULT_MODEL
    assert verdict.no_judgment == 1
    assert verdict.usage == {}


def test_parse_response_requires_answers_object() -> None:
    with pytest.raises(ProviderUpstreamError, match="answers"):
        parse_response({"model": "jev-1.13.0"}, _candidates(), _labels())


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ProviderAuthError),
        (403, ProviderPermissionError),
        (402, ProviderQuotaError),
        (400, ProviderBadRequestError),
        (422, ProviderBadRequestError),
        (429, ProviderRateLimitError),
        (404, ProviderUpstreamError),
        (500, ProviderUpstreamError),
        (503, ProviderUpstreamError),
    ],
)
async def test_judge_maps_error_status(
    httpx_mock: HTTPXMock, status: int, expected: type[Exception]
) -> None:
    httpx_mock.add_response(url=SYSTEMONE_URL, status_code=status, text="provider failure")
    client = _client()
    try:
        with pytest.raises(expected):
            await client.judge(SAMPLE_TEXT, _candidates(), _labels())
    finally:
        await client.aclose()


async def test_judge_maps_transport_timeout_to_upstream_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("timed out"), url=SYSTEMONE_URL)
    client = _client()
    try:
        with pytest.raises(ProviderUpstreamError):
            await client.judge(SAMPLE_TEXT, _candidates(), _labels())
    finally:
        await client.aclose()


async def test_judge_maps_non_json_body_to_upstream_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=SYSTEMONE_URL, text="<html>not json</html>")
    client = _client()
    try:
        with pytest.raises(ProviderUpstreamError):
            await client.judge(SAMPLE_TEXT, _candidates(), _labels())
    finally:
        await client.aclose()


async def test_judge_maps_unexpected_body_type_to_upstream_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=SYSTEMONE_URL, json=[1, 2, 3])
    client = _client()
    try:
        with pytest.raises(ProviderUpstreamError):
            await client.judge(SAMPLE_TEXT, _candidates(), _labels())
    finally:
        await client.aclose()


async def test_judge_rejects_oversized_request_without_network(httpx_mock: HTTPXMock) -> None:
    client = _client()
    text = "alpha " * 60_000
    try:
        with pytest.raises(ValueError, match="token budget"):
            await client.judge(text, [_candidate_at(text, "alpha")], _labels())
    finally:
        await client.aclose()
    assert httpx_mock.get_requests() == []


async def test_judge_negative_focus_adds_negative_phrasing(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=SYSTEMONE_URL, json=_body())
    httpx_mock.add_response(url=SYSTEMONE_URL, json=_body())
    client = _client()
    try:
        await client.judge(SAMPLE_TEXT, _candidates(), _labels(), negative_focus=False)
        await client.judge(SAMPLE_TEXT, _candidates(), _labels(), negative_focus=True)
    finally:
        await client.aclose()

    requests = httpx_mock.get_requests()
    plain = json.loads(requests[0].content)
    negative = json.loads(requests[1].content)
    plain_instructions = plain["questions"]["c0"]["instructions"]
    negative_instructions = negative["questions"]["c0"]["instructions"]

    assert plain["questions"]["c0"]["criteria"][NONE_OF_THESE] is None
    negative_none = negative["questions"]["c0"]["criteria"][NONE_OF_THESE]
    assert isinstance(negative_none, str)
    assert "named entity" in negative_none
    assert negative_instructions["focus"] != plain_instructions["focus"]
    assert negative_instructions["focus"].startswith(plain_instructions["focus"])
    assert "never named entities" in negative_instructions["focus"]
    assert "never named entities" not in plain_instructions["focus"]
    assert negative_none not in json.dumps(plain, ensure_ascii=False)


async def test_judge_sends_configured_model(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=SYSTEMONE_URL, json=_body())
    client = _client(model="jev-custom")
    try:
        await client.judge(SAMPLE_TEXT, _candidates(), _labels())
    finally:
        await client.aclose()
    payload = json.loads(httpx_mock.get_request().content)
    assert payload["model"] == "jev-custom"
