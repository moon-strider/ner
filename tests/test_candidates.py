from __future__ import annotations

import pytest

from ner_service.candidates import (
    CANDIDATE_GENERATOR_VERSION,
    Candidate,
    cap_frac,
    choose_prefilter,
    generate_candidates,
)

RU_PERSON = "\u0418\u0432\u0430\u043d\u043e\u0432"
RU_MOVED = "\u043f\u0435\u0440\u0435\u0435\u0445\u0430\u043b"
RU_CITY = "\u0421\u0430\u043d\u043a\u0442-\u041f\u0435\u0442\u0435\u0440\u0431\u0443\u0440\u0433"


def _by_text(candidates: list[Candidate], text: str) -> list[Candidate]:
    return [candidate for candidate in candidates if candidate.text == text]


def test_version_constant() -> None:
    assert CANDIDATE_GENERATOR_VERSION == 1


def test_offsets_match_slices_and_are_exact() -> None:
    text = "Tim Cook met Figma and mcp in Australia."
    candidates = generate_candidates(text, mode="name_like")

    assert candidates
    for candidate in candidates:
        assert text[candidate.start : candidate.end] == candidate.text

    assert [(candidate.start, candidate.end) for candidate in _by_text(candidates, "Figma")] == [
        (13, 18)
    ]
    assert [(candidate.start, candidate.end) for candidate in _by_text(candidates, "mcp")] == [
        (23, 26)
    ]
    assert [
        (candidate.start, candidate.end) for candidate in _by_text(candidates, "Australia")
    ] == [(30, 39)]


def test_new_york_keeps_inner_stopword() -> None:
    text = "She moved to New York last year."
    candidates = generate_candidates(text, mode="name_like")
    found = _by_text(candidates, "New York")

    assert len(found) == 1
    assert (found[0].start, found[0].end) == (13, 21)
    assert text[found[0].start : found[0].end] == "New York"


def test_ministry_for_span_keeps_connector() -> None:
    text = "The Ministry for Digital Development announced a plan."
    candidates = generate_candidates(text, mode="name_like")
    found = _by_text(candidates, "Ministry for Digital")

    assert len(found) == 1
    assert text[found[0].start : found[0].end] == "Ministry for Digital"
    assert (found[0].start, found[0].end) == (4, 24)


def test_single_token_entities_survive() -> None:
    text = "Figma and mcp run on linux in Australia."
    candidates = generate_candidates(text, mode="name_like")

    for surface in ("Figma", "mcp", "linux", "Australia"):
        found = _by_text(candidates, surface)
        assert len(found) == 1, surface
        assert text[found[0].start : found[0].end] == surface


def test_lowercase_text_routing() -> None:
    text = "figma deployed on eu-west-1."

    assert generate_candidates(text, mode="name_like")
    assert generate_candidates(text, mode="capitalized") == []


def test_repeated_ngrams_are_candidates() -> None:
    candidates = generate_candidates("release release notes", mode="name_like")

    assert [candidate.start for candidate in _by_text(candidates, "release")] == [0, 8]
    assert generate_candidates("release once", mode="name_like") == []


def test_repeated_bigram_marks_all_tokens() -> None:
    text = "release package release package"
    candidates = generate_candidates(text, mode="name_like")

    assert _by_text(candidates, "release package")
    assert generate_candidates("release package", mode="name_like") == []


def test_form_tokens_are_candidates() -> None:
    text = "Deployed v0.4.2 to eu-west-1 at api.example.io."
    candidates = generate_candidates(text, mode="name_like")

    for surface in ("v0.4.2", "eu-west-1", "api.example.io"):
        found = _by_text(candidates, surface)
        assert len(found) == 1, surface
        assert text[found[0].start : found[0].end] == surface
        assert found[0].source == "form"
        assert found[0].priority == 0


def test_generator_is_deterministic() -> None:
    text = "The Ministry for Digital Development uses Figma, mcp and v0.4.2."

    assert generate_candidates(text, mode="name_like") == generate_candidates(
        text, mode="name_like"
    )
    assert generate_candidates(text, mode="capitalized") == generate_candidates(
        text, mode="capitalized"
    )


def test_results_are_deduplicated_and_ordered() -> None:
    text = "Figma Figma mcp v1.2.3 New York"
    candidates = generate_candidates(text, mode="name_like")
    keys = [(candidate.start, candidate.end) for candidate in candidates]

    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_form_priority_beats_capitalized() -> None:
    candidates = generate_candidates("V0.4.2", mode="name_like")

    assert len(candidates) == 1
    assert candidates[0].text == "V0.4.2"
    assert candidates[0].source == "form"
    assert candidates[0].priority == 0


def test_known_sources_and_priorities() -> None:
    text = "Figma New York release release v0.4.2"
    candidates = generate_candidates(text, mode="name_like")
    sources = {candidate.source for candidate in candidates}

    assert sources <= {"capitalized", "rare", "form", "repeat", "joined"}
    for candidate in candidates:
        assert candidate.priority >= 0


def test_cap_frac_and_choose_prefilter() -> None:
    assert cap_frac("Figma released v0.4.2") > 0
    assert cap_frac("figma released") == 0.0
    assert cap_frac("") == 0.0
    assert choose_prefilter("Figma released v0.4.2") == "capitalized"
    assert choose_prefilter("figma released v0.4.2") == "name_like"
    assert choose_prefilter("") == "name_like"


def test_unicode_offsets_are_codepoint_based() -> None:
    text = f"{RU_PERSON} {RU_MOVED} \u0432 {RU_CITY}."
    candidates = generate_candidates(text, mode="name_like")

    person = _by_text(candidates, RU_PERSON)
    city = _by_text(candidates, RU_CITY)
    assert len(person) == 1
    assert len(city) == 1
    assert (person[0].start, person[0].end) == (0, 6)
    assert (city[0].start, city[0].end) == (
        text.index(RU_CITY),
        text.index(RU_CITY) + len(RU_CITY),
    )
    for candidate in candidates:
        assert text[candidate.start : candidate.end] == candidate.text


def test_max_candidates_bounds_output_and_keeps_document_start() -> None:
    text = " ".join(f"Entity{index}" for index in range(50))
    candidates = generate_candidates(text, mode="name_like", max_candidates=5)

    assert len(candidates) == 5
    assert candidates[0].start == 0
    assert generate_candidates(text, mode="name_like", max_candidates=0) == []


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_candidates("Figma", mode="bogus")
