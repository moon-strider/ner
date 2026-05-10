from __future__ import annotations

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
    if not raw_entities:
        return []

    result: list[Entity] = []
    consumed: list[tuple[int, int]] = []
    items = [raw for raw in raw_entities if raw.text]
    if not items:
        return []

    automaton = ahocorasick.Automaton()
    if case_sensitive:
        for raw in items:
            automaton.add_word(raw.text, (raw.text, raw.label))
        automaton.make_automaton()
        for end_index, (surface, label) in automaton.iter(text):
            start = end_index - len(surface) + 1
            end = end_index + 1
            span = (start, end)
            if any(_overlaps(span, existing) for existing in consumed):
                continue
            consumed.append(span)
            result.append(Entity(text=text[start:end], label=label, start=start, end=end))
        return result

    lowered_text = text.lower()
    for raw in items:
        automaton.add_word(raw.text.lower(), (raw.text.lower(), raw.label))
    automaton.make_automaton()
    for end_index, (surface, label) in automaton.iter(lowered_text):
        start = end_index - len(surface) + 1
        end = end_index + 1
        span = (start, end)
        if any(_overlaps(span, existing) for existing in consumed):
            continue
        consumed.append(span)
        result.append(Entity(text=text[start:end], label=label, start=start, end=end))
    return result


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
