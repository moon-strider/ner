from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.raw_entity import RawEntity


T = TypeVar("T", bound="FewShotExample")


@_attrs_define
class FewShotExample:
    """
    Attributes:
        text (str):
        entities (list[RawEntity] | Unset):
    """

    text: str
    entities: list[RawEntity] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        entities: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.entities, Unset):
            entities = []
            for entities_item_data in self.entities:
                entities_item = entities_item_data.to_dict()
                entities.append(entities_item)

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "text": text,
            }
        )
        if entities is not UNSET:
            field_dict["entities"] = entities

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.raw_entity import RawEntity

        d = dict(src_dict)
        text = d.pop("text")

        _entities = d.pop("entities", UNSET)
        entities: list[RawEntity] | Unset = UNSET
        if _entities is not UNSET:
            entities = []
            for entities_item_data in _entities:
                entities_item = RawEntity.from_dict(entities_item_data)

                entities.append(entities_item)

        few_shot_example = cls(
            text=text,
            entities=entities,
        )

        return few_shot_example
