from __future__ import annotations

from typing import Any

from ner_service.schemas import EntityLabel


def build_ner_json_schema(labels: list[EntityLabel]) -> dict[str, Any]:
    if not labels:
        raise ValueError("at least one label is required")
    names = [label.name for label in labels]
    if len(set(names)) != len(names):
        raise ValueError("label names must be unique")

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "label"],
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string", "enum": names},
                    },
                },
            }
        },
    }


def build_response_format(labels: list[EntityLabel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ner_extraction",
            "strict": True,
            "schema": build_ner_json_schema(labels),
        },
    }


def build_system_prompt(labels: list[EntityLabel], *, require_offsets: bool = False) -> str:
    lines = [
        "You are a named entity recognition system.",
        "Extract entities from the user text that belong to the listed entity types.",
        "Return JSON only, matching the provided schema.",
        "Return the exact surface form from the text. Preserve casing, punctuation, and spacing.",
        "Do not infer entities that are not explicitly present in the text.",
        "If no entities are present, return an empty list.",
    ]
    if require_offsets:
        lines.extend(
            [
                "Return one entity per occurrence, in reading order.",
                "Each returned text must be a contiguous substring of the user text.",
            ]
        )
    else:
        lines.extend(
            [
                "Return each unique (text, label) pair at most once.",
                "Do not return character positions or offsets.",
            ]
        )
    lines.extend(["", "Entity types:"])
    for label in labels:
        lines.append(f"- {label.name}: {label.description}")
    return "\n".join(lines)
