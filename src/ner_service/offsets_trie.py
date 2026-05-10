from __future__ import annotations

from typing import Any

try:
    import ahocorasick
    _HAS_AHOCORASICK = True
except ImportError:
    _HAS_AHOCORASICK = False

from ner_service.schemas import Entity, RawEntity


def attach_offsets_trie(
    text: str,
    raw_entities: list[RawEntity],
    *,
    case_sensitive: bool = True,
) -> list[Entity]:
    if not _HAS_AHOCORASICK:
        raise ImportError("pyahocorasick required")

    result: list[Entity] = []
    consumed: list[tuple[int, int]] = []

    if case_sensitive:
        automaton = ahocorasick.Automaton()
        for raw in raw_entities:
            if raw.text:
                automaton.add_word(raw.text, (raw.text, raw.label))
        automaton.make_automaton()

        for end_index, (surface, label) in automaton.iter(text):
            start = end_index - len(surface) + 1
            end = end_index + 1
            span = (start, end)
            if not any(_overlaps(span, c) for c in consumed):
                consumed.append(span)
                result.append(Entity(text=text[start:end], label=label, start=start, end=end))
    else:
        automaton = ahocorasick.Automaton()
        lower_text = text.lower()
        for raw in raw_entities:
            if raw.text:
                automaton.add_word(raw.text.lower(), (raw.text, raw.label))
        automaton.make_automaton()

        for end_index, (surface_lower, label) in automaton.iter(lower_text):
            start = end_index - len(surface_lower) + 1
            end = end_index + 1
            span = (start, end)
            if not any(_overlaps(span, c) for c in consumed):
                consumed.append(span)
                original = text[start:end]
                result.append(Entity(text=original, label=label, start=start, end=end))

    return result


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
