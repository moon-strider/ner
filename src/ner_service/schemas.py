from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LABEL_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class EntityLabel(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=500)

    @field_validator("name")
    @classmethod
    def _name_format(cls, v: str) -> str:
        if not LABEL_NAME_RE.match(v):
            raise ValueError(
                "name must match ^[A-Z][A-Z0-9_]*$ (uppercase letters, digits, underscores)"
            )
        return v


class NERConfig(BaseModel):
    labels: list[EntityLabel] = Field(..., min_length=1, max_length=50)
    model: str = Field(default="llama3.1-8b", min_length=1, max_length=128)
    require_offsets: bool = False
    case_sensitive: bool = True
    retries: int = Field(default=3, ge=1)
    max_tokens: int = Field(default=1024, gt=0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    system_prompt: str | None = Field(default=None, min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def _unique_label_names(self) -> NERConfig:
        names = [label.name for label in self.labels]
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        return self


class NERConfigPatch(BaseModel):
    labels: list[EntityLabel] | None = Field(default=None, min_length=1, max_length=50)
    model: str | None = Field(default=None, min_length=1, max_length=128)
    require_offsets: bool | None = None
    case_sensitive: bool | None = None
    retries: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    system_prompt: str | None = Field(default=None, min_length=1, max_length=20_000)

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
    text: str = Field(..., min_length=1, max_length=32_000)
    config_id: str | None = Field(default=None, min_length=1, max_length=128)
    config: NERConfig | None = None
    prompt_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_config_source(self) -> ExtractRequest:
        if (self.config_id is None) == (self.config is None):
            raise ValueError("exactly one of config_id or config is required")
        return self


class RawEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    label: str


class RawEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[RawEntity]
    usage: dict[str, Any] | None = None


class Entity(BaseModel):
    text: str
    label: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class ExtractResponse(BaseModel):
    entities: list[Entity]
    model: str
    provider: str
    usage: dict[str, Any] | None = None
