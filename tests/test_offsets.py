from __future__ import annotations

from ner_service.offsets import attach_offsets
from ner_service.schemas import RawEntity


def _raw(text: str, label: str = "X") -> RawEntity:
    return RawEntity(text=text, label=label)


def test_single_occurrence() -> None:
    text = "Tim Cook visited Berlin."
    out = attach_offsets(text, [_raw("Tim Cook", "PERSON"), _raw("Berlin", "LOCATION")])

    assert [(e.text, e.label, e.start, e.end) for e in out] == [
        ("Tim Cook", "PERSON", 0, 8),
        ("Berlin", "LOCATION", 17, 23),
    ]
    for e in out:
        assert text[e.start : e.end] == e.text


def test_duplicate_surface_maps_to_distinct_offsets() -> None:
    text = "Paris is Paris."
    out = attach_offsets(text, [_raw("Paris"), _raw("Paris")])
    assert [(e.start, e.end) for e in out] == [(0, 5), (9, 14)]


def test_more_duplicates_than_occurrences_drops_extras() -> None:
    text = "Berlin only once."
    out = attach_offsets(text, [_raw("Berlin"), _raw("Berlin")])
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (0, 6)


def test_hallucinated_span_is_dropped() -> None:
    text = "Apple released something."
    out = attach_offsets(text, [_raw("Microsoft")])
    assert out == []


def test_unicode_offsets_are_codepoint_based() -> None:
    text = "Café opened in São Paulo."
    out = attach_offsets(text, [_raw("Café"), _raw("São Paulo")])

    assert [(e.start, e.end) for e in out] == [(0, 4), (15, 24)]
    for e in out:
        assert text[e.start : e.end] == e.text


def test_empty_input_returns_empty() -> None:
    assert attach_offsets("any text", []) == []


def test_empty_surface_is_skipped() -> None:
    out = attach_offsets("Hello world", [_raw("")])
    assert out == []


def test_overlapping_spans_pick_next_non_overlapping() -> None:
    text = "ab ab ab"
    out = attach_offsets(text, [_raw("ab"), _raw("ab"), _raw("ab")])
    assert [(e.start, e.end) for e in out] == [(0, 2), (3, 5), (6, 8)]


def test_label_is_preserved_even_when_surface_matches_other_label() -> None:
    text = "Foo Bar."
    out = attach_offsets(text, [_raw("Foo", "PERSON"), _raw("Bar", "LOCATION")])
    assert [(e.text, e.label) for e in out] == [("Foo", "PERSON"), ("Bar", "LOCATION")]


def test_case_insensitive_offsets_return_input_casing() -> None:
    text = "Tim Cook visited Berlin."
    out = attach_offsets(text, [_raw("tim cook", "PERSON")], case_sensitive=False)
    assert [(e.text, e.label, e.start, e.end) for e in out] == [("Tim Cook", "PERSON", 0, 8)]
