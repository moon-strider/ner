from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from ner_service.schemas import Entity, RawEntity

logger = logging.getLogger(__name__)


def attach_offsets(
    text: str,
    raw_entities: Iterable[RawEntity],
    *,
    case_sensitive: bool = True,
) -> list[Entity]:
    result: list[Entity] = []
    consumed: list[tuple[int, int]] = []

    for raw in raw_entities:
        surface = raw.text
        if not surface:
            continue

        span = _find_next_span(text, surface, consumed, case_sensitive=case_sensitive)
        if span is None:
            logger.warning(
                "entity surface %r (label=%s) not found in input text; dropping",
                surface,
                raw.label,
            )
            continue

        start, end = span
        consumed.append(span)
        result.append(Entity(text=text[start:end], label=raw.label, start=start, end=end))

    return result


def canonicalize_entities(
    text: str,
    raw_entities: Iterable[RawEntity],
    *,
    case_sensitive: bool = True,
) -> list[Entity]:
    result: list[Entity] = []
    for raw in raw_entities:
        surface = raw.text
        if not surface:
            continue
        if case_sensitive:
            result.append(Entity(text=surface, label=raw.label))
            continue
        span = _find_next_span(text, surface, [], case_sensitive=False)
        if span is None:
            result.append(Entity(text=surface, label=raw.label))
            continue
        start, end = span
        result.append(Entity(text=text[start:end], label=raw.label))
    return result


def _find_next_span(
    text: str,
    surface: str,
    consumed: list[tuple[int, int]],
    *,
    case_sensitive: bool,
) -> tuple[int, int] | None:
    if not case_sensitive:
        return _find_next_span_case_insensitive(text, surface, consumed)

    search_from = 0
    while True:
        idx = text.find(surface, search_from)
        if idx == -1:
            return None
        end = idx + len(surface)
        if not any(_overlaps((idx, end), c) for c in consumed):
            return (idx, end)
        search_from = idx + 1


def _find_next_span_case_insensitive(
    text: str,
    surface: str,
    consumed: list[tuple[int, int]],
) -> tuple[int, int] | None:
    pattern = re.compile(re.escape(surface), re.IGNORECASE)
    for match in pattern.finditer(text):
        span = match.span()
        if not any(_overlaps(span, c) for c in consumed):
            return span
    return None


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
