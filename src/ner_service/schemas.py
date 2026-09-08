from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LABEL_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class EntityLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1)

    @field_validator("name")
    @classmethod
    def _name_format(cls, v: str) -> str:
        if not LABEL_NAME_RE.fullmatch(v):
            raise ValueError(
                "name must match ^[A-Z][A-Z0-9_]*$ (uppercase letters, digits, underscores)"
            )
        return v


class RawEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    label: str


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    entities: list[RawEntity] = Field(default_factory=list, max_length=2048)


def _runtime_defaults_schema(schema: dict[str, Any]) -> None:
    # Runtime settings resolve omitted values. SDKs must not send hard-coded defaults.
    for name in ("model", "max_tokens"):
        schema["properties"][name].pop("default", None)


class NERConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_runtime_defaults_schema)

    labels: list[EntityLabel] = Field(..., min_length=1)
    model: str = Field(default="llama3.1-8b", min_length=1, max_length=128)
    require_offsets: bool = False
    case_sensitive: bool = True
    retries: int = Field(default=3, ge=1)
    max_tokens: int = Field(default=1024, gt=0)
    reasoning_effort: str | None = None
    system_prompt: str | None = Field(default=None, min_length=1)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_label_names(self) -> NERConfig:
        names = [label.name for label in self.labels]
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        for example in self.few_shot_examples:
            for entity in example.entities:
                if entity.label not in names or not entity.text or entity.text not in example.text:
                    raise ValueError("few-shot entities must use configured labels and source text")
        return self


class NERConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[EntityLabel] | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1, max_length=128)
    require_offsets: bool | None = None
    case_sensitive: bool | None = None
    retries: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: str | None = None
    system_prompt: str | None = Field(default=None, min_length=1)
    few_shot_examples: list[FewShotExample] | None = None

    @model_validator(mode="after")
    def _unique_label_names(self) -> NERConfigPatch:
        if self.labels is None:
            return self
        names = [label.name for label in self.labels]
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        return self


class NERConfigRecord(BaseModel):
    id: str
    config: NERConfig


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    config_id: str | None = Field(default=None, min_length=1)
    config: NERConfig | None = None
    prompt_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_config_source(self) -> ExtractRequest:
        if (self.config_id is None) == (self.config is None):
            raise ValueError("exactly one of config_id or config is required")
        return self


class RawEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[RawEntity] = Field(max_length=2048)
    usage: dict[str, Any] | None = None
    attempts: int = Field(default=1, ge=1)


class Entity(BaseModel):
    text: str
    label: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class ExtractResponse(BaseModel):
    cache_hit: bool = False
    entities: list[Entity]
    model: str
    provider: str
    usage: dict[str, Any] | None = None
    attempts: int = Field(default=1, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ExtractResponseData(BaseModel):
    entities: list[Entity]
    model: str
    provider: str
    usage: dict[str, Any] | None = None


class ResponseMeta(BaseModel):
    cache_hit: bool = False
    request_id: str
    latency_ms: float = Field(..., ge=0.0)
    attempts: int = Field(..., ge=0)
    warnings: list[str] = Field(default_factory=list)


class ExtractEnvelope(BaseModel):
    data: ExtractResponseData
    meta: ResponseMeta


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
