from __future__ import annotations

import re
from typing import Any

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


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=32_000)
    labels: list[EntityLabel] = Field(..., min_length=1, max_length=50)

    @model_validator(mode="after")
    def _unique_label_names(self) -> ExtractRequest:
        names = [label.name for label in self.labels]
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        return self


class RawEntity(BaseModel):
    text: str
    label: str


class RawEntities(BaseModel):
    entities: list[RawEntity]
    usage: dict[str, Any] | None = None


class Entity(BaseModel):
    text: str
    label: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class ExtractResponse(BaseModel):
    entities: list[Entity]
    model: str
    provider: str
    usage: dict[str, Any] | None = None
