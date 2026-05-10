from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ResponseMeta")


@_attrs_define
class ResponseMeta:
    request_id: str
    latency_ms: float
    attempts: int
    warnings: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_id = self.request_id
        latency_ms = self.latency_ms
        attempts = self.attempts
        warnings: list[str] | Unset = UNSET
        if not isinstance(self.warnings, Unset):
            warnings = self.warnings
        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {"request_id": request_id, "latency_ms": latency_ms, "attempts": attempts}
        )
        if warnings is not UNSET:
            field_dict["warnings"] = warnings
        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        request_id = d.pop("request_id")
        latency_ms = d.pop("latency_ms")
        attempts = d.pop("attempts")
        warnings = cast(list[str], d.pop("warnings", UNSET))
        response_meta = cls(
            request_id=request_id, latency_ms=latency_ms, attempts=attempts, warnings=warnings
        )
        response_meta.additional_properties = d
        return response_meta

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
