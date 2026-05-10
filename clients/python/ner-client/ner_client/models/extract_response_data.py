from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.entity import Entity
    from ..models.extract_response_data_usage_type_0 import ExtractResponseDataUsageType0
T = TypeVar("T", bound="ExtractResponseData")


@_attrs_define
class ExtractResponseData:
    entities: list[Entity]
    model: str
    provider: str
    usage: ExtractResponseDataUsageType0 | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.extract_response_data_usage_type_0 import ExtractResponseDataUsageType0

        entities = []
        for entities_item_data in self.entities:
            entities_item = entities_item_data.to_dict()
            entities.append(entities_item)
        model = self.model
        provider = self.provider
        usage: dict[str, Any] | None | Unset
        if isinstance(self.usage, Unset):
            usage = UNSET
        elif isinstance(self.usage, ExtractResponseDataUsageType0):
            usage = self.usage.to_dict()
        else:
            usage = self.usage
        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({"entities": entities, "model": model, "provider": provider})
        if usage is not UNSET:
            field_dict["usage"] = usage
        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.entity import Entity
        from ..models.extract_response_data_usage_type_0 import ExtractResponseDataUsageType0

        d = dict(src_dict)
        entities = []
        _entities = d.pop("entities")
        for entities_item_data in _entities:
            entities_item = Entity.from_dict(entities_item_data)
            entities.append(entities_item)
        model = d.pop("model")
        provider = d.pop("provider")

        def _parse_usage(data: object) -> ExtractResponseDataUsageType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                usage_type_0 = ExtractResponseDataUsageType0.from_dict(data)
                return usage_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ExtractResponseDataUsageType0 | None | Unset, data)

        usage = _parse_usage(d.pop("usage", UNSET))
        extract_response_data = cls(entities=entities, model=model, provider=provider, usage=usage)
        extract_response_data.additional_properties = d
        return extract_response_data

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
