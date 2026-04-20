from __future__ import annotations

import json
import string
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ner_service.schema_builder import (
    build_ner_json_schema,
    build_response_format,
    build_system_prompt,
)
from ner_service.schemas import NERConfig, NERConfigPatch, NERConfigRecord


class ConfigNotFoundError(Exception):
    pass


class PromptTemplateError(Exception):
    pass


@dataclass(frozen=True)
class PreparedNERConfig:
    id: str | None
    config: NERConfig
    response_format: dict[str, Any]
    schema_json: str
    allowed_labels: set[str]
    default_system_prompt: str


class ConfigStore:
    def __init__(self) -> None:
        self._items: dict[str, PreparedNERConfig] = {}

    def create(self, config: NERConfig) -> NERConfigRecord:
        config_id = str(uuid.uuid4())
        prepared = prepare_config(config, config_id=config_id)
        self._items[config_id] = prepared
        return NERConfigRecord(id=config_id, config=prepared.config)

    def list(self) -> list[NERConfigRecord]:
        return [
            NERConfigRecord(id=config_id, config=prepared.config)
            for config_id, prepared in self._items.items()
        ]

    def get(self, config_id: str) -> PreparedNERConfig:
        try:
            return self._items[config_id]
        except KeyError as e:
            raise ConfigNotFoundError(config_id) from e

    def put(self, config_id: str, config: NERConfig) -> NERConfigRecord:
        if config_id not in self._items:
            raise ConfigNotFoundError(config_id)
        prepared = prepare_config(config, config_id=config_id)
        self._items[config_id] = prepared
        return NERConfigRecord(id=config_id, config=prepared.config)

    def patch(self, config_id: str, patch: NERConfigPatch) -> NERConfigRecord:
        current = self.get(config_id).config
        data = current.model_dump()
        data.update(patch.model_dump(exclude_unset=True))
        try:
            config = NERConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(str(e)) from e
        prepared = prepare_config(config, config_id=config_id)
        self._items[config_id] = prepared
        return NERConfigRecord(id=config_id, config=prepared.config)

    def delete(self, config_id: str) -> None:
        if config_id not in self._items:
            raise ConfigNotFoundError(config_id)
        del self._items[config_id]


def prepare_config(config: NERConfig, *, config_id: str | None = None) -> PreparedNERConfig:
    schema = build_ner_json_schema(config.labels)
    schema_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return PreparedNERConfig(
        id=config_id,
        config=config,
        response_format=build_response_format(config.labels),
        schema_json=schema_json,
        allowed_labels={label.name for label in config.labels},
        default_system_prompt=build_system_prompt(
            config.labels,
            require_offsets=config.require_offsets,
            case_sensitive=config.case_sensitive,
        ),
    )


def render_system_prompt(
    prepared: PreparedNERConfig,
    prompt_payload: dict[str, Any],
) -> str:
    template = prepared.config.system_prompt or prepared.default_system_prompt
    if "{" not in template and "}" not in template:
        return template

    cfg = prepared.config.model_dump(mode="json")
    cfg["schema"] = prepared.schema_json
    roots = {"cfg": cfg, "payload": prompt_payload}

    pieces: list[str] = []
    formatter = string.Formatter()
    try:
        parsed = formatter.parse(template)
    except ValueError as e:
        raise PromptTemplateError(str(e)) from e

    for literal, field_name, format_spec, conversion in parsed:
        pieces.append(literal)
        if field_name is None:
            continue
        if not field_name:
            raise PromptTemplateError("empty template placeholder is not supported")
        if format_spec:
            raise PromptTemplateError("template format specs are not supported")
        if conversion:
            raise PromptTemplateError("template conversions are not supported")
        pieces.append(_stringify_template_value(_resolve_placeholder(field_name, roots)))
    return "".join(pieces)


def _resolve_placeholder(field_name: str, roots: dict[str, Any]) -> Any:
    parts = field_name.split(".")
    root = parts[0]
    if root not in roots:
        raise PromptTemplateError(f"unknown template root: {root}")
    value: Any = roots[root]
    for part in parts[1:]:
        if not isinstance(value, dict) or part not in value:
            raise PromptTemplateError(f"missing template value: {field_name}")
        value = value[part]
    return value


def _stringify_template_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError as e:
        raise PromptTemplateError(f"template value is not JSON serializable: {value!r}") from e
