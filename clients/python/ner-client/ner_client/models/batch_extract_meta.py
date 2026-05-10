from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="BatchExtractMeta")


@_attrs_define
class BatchExtractMeta:
    total: int
    succeeded: int
    failed: int
    latency_ms: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total = self.total
        succeeded = self.succeeded
        failed = self.failed
        latency_ms = self.latency_ms
        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {"total": total, "succeeded": succeeded, "failed": failed, "latency_ms": latency_ms}
        )
        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        total = d.pop("total")
        succeeded = d.pop("succeeded")
        failed = d.pop("failed")
        latency_ms = d.pop("latency_ms")
        batch_extract_meta = cls(
            total=total, succeeded=succeeded, failed=failed, latency_ms=latency_ms
        )
        batch_extract_meta.additional_properties = d
        return batch_extract_meta

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
