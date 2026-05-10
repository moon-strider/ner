from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.entity_label import EntityLabel
    from ..models.few_shot_example import FewShotExample
T = TypeVar("T", bound="NERConfig")


@_attrs_define
class NERConfig:
    labels: list[EntityLabel]
    model: str | Unset = "llama3.1-8b"
    require_offsets: bool | Unset = False
    case_sensitive: bool | Unset = True
    retries: int | Unset = 3
    max_tokens: int | Unset = 1024
    reasoning_effort: None | str | Unset = UNSET
    system_prompt: None | str | Unset = UNSET
    few_shot_examples: list[FewShotExample] | Unset = UNSET
    confidence: bool | Unset = False
    output_format: str | Unset = "json"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        labels = []
        for labels_item_data in self.labels:
            labels_item = labels_item_data.to_dict()
            labels.append(labels_item)
        model = self.model
        require_offsets = self.require_offsets
        case_sensitive = self.case_sensitive
        retries = self.retries
        max_tokens = self.max_tokens
        reasoning_effort: None | str | Unset
        if isinstance(self.reasoning_effort, Unset):
            reasoning_effort = UNSET
        else:
            reasoning_effort = self.reasoning_effort
        system_prompt: None | str | Unset
        if isinstance(self.system_prompt, Unset):
            system_prompt = UNSET
        else:
            system_prompt = self.system_prompt
        few_shot_examples: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.few_shot_examples, Unset):
            few_shot_examples = []
            for few_shot_examples_item_data in self.few_shot_examples:
                few_shot_examples_item = few_shot_examples_item_data.to_dict()
                few_shot_examples.append(few_shot_examples_item)
        confidence = self.confidence
        output_format = self.output_format
        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({"labels": labels})
        if model is not UNSET:
            field_dict["model"] = model
        if require_offsets is not UNSET:
            field_dict["require_offsets"] = require_offsets
        if case_sensitive is not UNSET:
            field_dict["case_sensitive"] = case_sensitive
        if retries is not UNSET:
            field_dict["retries"] = retries
        if max_tokens is not UNSET:
            field_dict["max_tokens"] = max_tokens
        if reasoning_effort is not UNSET:
            field_dict["reasoning_effort"] = reasoning_effort
        if system_prompt is not UNSET:
            field_dict["system_prompt"] = system_prompt
        if few_shot_examples is not UNSET:
            field_dict["few_shot_examples"] = few_shot_examples
        if confidence is not UNSET:
            field_dict["confidence"] = confidence
        if output_format is not UNSET:
            field_dict["output_format"] = output_format
        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.entity_label import EntityLabel
        from ..models.few_shot_example import FewShotExample

        d = dict(src_dict)
        labels = []
        _labels = d.pop("labels")
        for labels_item_data in _labels:
            labels_item = EntityLabel.from_dict(labels_item_data)
            labels.append(labels_item)
        model = d.pop("model", UNSET)
        require_offsets = d.pop("require_offsets", UNSET)
        case_sensitive = d.pop("case_sensitive", UNSET)
        retries = d.pop("retries", UNSET)
        max_tokens = d.pop("max_tokens", UNSET)

        def _parse_reasoning_effort(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        reasoning_effort = _parse_reasoning_effort(d.pop("reasoning_effort", UNSET))

        def _parse_system_prompt(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        system_prompt = _parse_system_prompt(d.pop("system_prompt", UNSET))
        _few_shot_examples = d.pop("few_shot_examples", UNSET)
        few_shot_examples: list[FewShotExample] | Unset = UNSET
        if _few_shot_examples is not UNSET:
            few_shot_examples = []
            for few_shot_examples_item_data in _few_shot_examples:
                few_shot_examples_item = FewShotExample.from_dict(few_shot_examples_item_data)
                few_shot_examples.append(few_shot_examples_item)
        confidence = d.pop("confidence", UNSET)
        output_format = d.pop("output_format", UNSET)
        ner_config = cls(
            labels=labels,
            model=model,
            require_offsets=require_offsets,
            case_sensitive=case_sensitive,
            retries=retries,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
            few_shot_examples=few_shot_examples,
            confidence=confidence,
            output_format=output_format,
        )
        ner_config.additional_properties = d
        return ner_config

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
