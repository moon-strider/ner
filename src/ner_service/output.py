from __future__ import annotations

from typing import Any

from ner_service.schemas import Entity


def to_bio(text: str, entities: list[Entity]) -> list[dict[str, Any]]:
    tokens = text.split()
    bio = [{"token": tok, "label": "O"} for tok in tokens]
    char_offset = 0
    token_offsets: list[tuple[int, int]] = []
    for tok in tokens:
        token_offsets.append((char_offset, char_offset + len(tok)))
        char_offset += len(tok) + 1
    for ent in sorted(entities, key=lambda e: e.start or 0):
        if ent.start is None or ent.end is None:
            continue
        for idx, (t_start, t_end) in enumerate(token_offsets):
            if t_start >= ent.start and t_end <= ent.end:
                prefix = "B-" if idx == 0 or bio[idx - 1]["label"] == "O" else "I-"
                bio[idx]["label"] = f"{prefix}{ent.label}"
    return bio


def to_spans(entities: list[Entity]) -> list[dict[str, Any]]:
    return [
        {"label": e.label, "start": e.start, "end": e.end}
        for e in entities
        if e.start is not None and e.end is not None
    ]


def to_dict(entities: list[Entity]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for e in entities:
        result.setdefault(e.label, []).append(e.text)
    return result
