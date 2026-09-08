from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="RawEntity")


@_attrs_define
class RawEntity:
    """
    Attributes:
        text (str):
        label (str):
    """

    text: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        label = self.label

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "text": text,
                "label": label,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        text = d.pop("text")

        label = d.pop("label")

        raw_entity = cls(
            text=text,
            label=label,
        )

        return raw_entity
