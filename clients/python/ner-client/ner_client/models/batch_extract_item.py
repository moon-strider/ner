from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.batch_extract_item_error_type_0 import BatchExtractItemErrorType0
    from ..models.batch_extract_item_meta import BatchExtractItemMeta
    from ..models.extract_envelope import ExtractEnvelope
T = TypeVar("T", bound="BatchExtractItem")


@_attrs_define
class BatchExtractItem:
    index: int
    meta: BatchExtractItemMeta
    data: ExtractEnvelope | None | Unset = UNSET
    error: BatchExtractItemErrorType0 | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.batch_extract_item_error_type_0 import BatchExtractItemErrorType0
        from ..models.extract_envelope import ExtractEnvelope

        index = self.index
        meta = self.meta.to_dict()
        data: dict[str, Any] | None | Unset
        if isinstance(self.data, Unset):
            data = UNSET
        elif isinstance(self.data, ExtractEnvelope):
            data = self.data.to_dict()
        else:
            data = self.data
        error: dict[str, Any] | None | Unset
        if isinstance(self.error, Unset):
            error = UNSET
        elif isinstance(self.error, BatchExtractItemErrorType0):
            error = self.error.to_dict()
        else:
            error = self.error
        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({"index": index, "meta": meta})
        if data is not UNSET:
            field_dict["data"] = data
        if error is not UNSET:
            field_dict["error"] = error
        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.batch_extract_item_error_type_0 import BatchExtractItemErrorType0
        from ..models.batch_extract_item_meta import BatchExtractItemMeta
        from ..models.extract_envelope import ExtractEnvelope

        d = dict(src_dict)
        index = d.pop("index")
        meta = BatchExtractItemMeta.from_dict(d.pop("meta"))

        def _parse_data(data: object) -> ExtractEnvelope | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                data_type_0 = ExtractEnvelope.from_dict(data)
                return data_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ExtractEnvelope | None | Unset, data)

        data = _parse_data(d.pop("data", UNSET))

        def _parse_error(data: object) -> BatchExtractItemErrorType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                error_type_0 = BatchExtractItemErrorType0.from_dict(data)
                return error_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(BatchExtractItemErrorType0 | None | Unset, data)

        error = _parse_error(d.pop("error", UNSET))
        batch_extract_item = cls(index=index, meta=meta, data=data, error=error)
        batch_extract_item.additional_properties = d
        return batch_extract_item

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
