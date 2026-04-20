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


def build_system_prompt(labels: list[EntityLabel]) -> str:
    lines = [
        "You are a named entity recognition system.",
        "Extract all spans from the user text that belong to the listed entity types.",
        "Return the exact surface form from the text (preserve casing and punctuation).",
        "If the same entity appears multiple times, return it once per occurrence in order.",
        "If no entities are present, return an empty list.",
        "",
        "Entity types:",
    ]
    for label in labels:
        lines.append(f"- {label.name}: {label.description}")
    return "\n".join(lines)
