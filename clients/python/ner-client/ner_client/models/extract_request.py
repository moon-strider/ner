from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.ner_config import NERConfig
    from ..models.prompt_payload import PromptPayload
T = TypeVar("T", bound="ExtractRequest")


@_attrs_define
class ExtractRequest:
    text: str
    config_id: None | str | Unset = UNSET
    config: NERConfig | None | Unset = UNSET
    prompt_payload: PromptPayload | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.ner_config import NERConfig

        text = self.text
        config_id: None | str | Unset
        if isinstance(self.config_id, Unset):
            config_id = UNSET
        else:
            config_id = self.config_id
        config: dict[str, Any] | None | Unset
        if isinstance(self.config, Unset):
            config = UNSET
        elif isinstance(self.config, NERConfig):
            config = self.config.to_dict()
        else:
            config = self.config
        prompt_payload: dict[str, Any] | Unset = UNSET
        if not isinstance(self.prompt_payload, Unset):
            prompt_payload = self.prompt_payload.to_dict()
        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({"text": text})
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if config is not UNSET:
            field_dict["config"] = config
        if prompt_payload is not UNSET:
            field_dict["prompt_payload"] = prompt_payload
        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.ner_config import NERConfig
        from ..models.prompt_payload import PromptPayload

        d = dict(src_dict)
        text = d.pop("text")

        def _parse_config_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        config_id = _parse_config_id(d.pop("config_id", UNSET))

        def _parse_config(data: object) -> NERConfig | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                config_type_0 = NERConfig.from_dict(data)
                return config_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(NERConfig | None | Unset, data)

        config = _parse_config(d.pop("config", UNSET))
        _prompt_payload = d.pop("prompt_payload", UNSET)
        prompt_payload: PromptPayload | Unset
        if isinstance(_prompt_payload, Unset):
            prompt_payload = UNSET
        else:
            prompt_payload = PromptPayload.from_dict(_prompt_payload)
        extract_request = cls(
            text=text, config_id=config_id, config=config, prompt_payload=prompt_payload
        )
        extract_request.additional_properties = d
        return extract_request

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
