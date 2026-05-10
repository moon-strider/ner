from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.entity_label import EntityLabel
    from ..models.few_shot_example import FewShotExample
T = TypeVar("T", bound="NERConfigPatch")


@_attrs_define
class NERConfigPatch:
    labels: list[EntityLabel] | None | Unset = UNSET
    model: None | str | Unset = UNSET
    require_offsets: bool | None | Unset = UNSET
    case_sensitive: bool | None | Unset = UNSET
    retries: int | None | Unset = UNSET
    max_tokens: int | None | Unset = UNSET
    reasoning_effort: None | str | Unset = UNSET
    system_prompt: None | str | Unset = UNSET
    few_shot_examples: list[FewShotExample] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        labels: list[dict[str, Any]] | None | Unset
        if isinstance(self.labels, Unset):
            labels = UNSET
        elif isinstance(self.labels, list):
            labels = []
            for labels_type_0_item_data in self.labels:
                labels_type_0_item = labels_type_0_item_data.to_dict()
                labels.append(labels_type_0_item)
        else:
            labels = self.labels
        model: None | str | Unset
        if isinstance(self.model, Unset):
            model = UNSET
        else:
            model = self.model
        require_offsets: bool | None | Unset
        if isinstance(self.require_offsets, Unset):
            require_offsets = UNSET
        else:
            require_offsets = self.require_offsets
        case_sensitive: bool | None | Unset
        if isinstance(self.case_sensitive, Unset):
            case_sensitive = UNSET
        else:
            case_sensitive = self.case_sensitive
        retries: int | None | Unset
        if isinstance(self.retries, Unset):
            retries = UNSET
        else:
            retries = self.retries
        max_tokens: int | None | Unset
        if isinstance(self.max_tokens, Unset):
            max_tokens = UNSET
        else:
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
        few_shot_examples: list[dict[str, Any]] | None | Unset
        if isinstance(self.few_shot_examples, Unset):
            few_shot_examples = UNSET
        elif isinstance(self.few_shot_examples, list):
            few_shot_examples = []
            for few_shot_examples_type_0_item_data in self.few_shot_examples:
                few_shot_examples_type_0_item = few_shot_examples_type_0_item_data.to_dict()
                few_shot_examples.append(few_shot_examples_type_0_item)
        else:
            few_shot_examples = self.few_shot_examples
        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if labels is not UNSET:
            field_dict["labels"] = labels
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
        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.entity_label import EntityLabel
        from ..models.few_shot_example import FewShotExample

        d = dict(src_dict)

        def _parse_labels(data: object) -> list[EntityLabel] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                labels_type_0 = []
                _labels_type_0 = data
                for labels_type_0_item_data in _labels_type_0:
                    labels_type_0_item = EntityLabel.from_dict(labels_type_0_item_data)
                    labels_type_0.append(labels_type_0_item)
                return labels_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[EntityLabel] | None | Unset, data)

        labels = _parse_labels(d.pop("labels", UNSET))

        def _parse_model(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        model = _parse_model(d.pop("model", UNSET))

        def _parse_require_offsets(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        require_offsets = _parse_require_offsets(d.pop("require_offsets", UNSET))

        def _parse_case_sensitive(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        case_sensitive = _parse_case_sensitive(d.pop("case_sensitive", UNSET))

        def _parse_retries(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        retries = _parse_retries(d.pop("retries", UNSET))

        def _parse_max_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        max_tokens = _parse_max_tokens(d.pop("max_tokens", UNSET))

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

        def _parse_few_shot_examples(data: object) -> list[FewShotExample] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                few_shot_examples_type_0 = []
                _few_shot_examples_type_0 = data
                for few_shot_examples_type_0_item_data in _few_shot_examples_type_0:
                    few_shot_examples_type_0_item = FewShotExample.from_dict(
                        few_shot_examples_type_0_item_data
                    )
                    few_shot_examples_type_0.append(few_shot_examples_type_0_item)
                return few_shot_examples_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[FewShotExample] | None | Unset, data)

        few_shot_examples = _parse_few_shot_examples(d.pop("few_shot_examples", UNSET))
        ner_config_patch = cls(
            labels=labels,
            model=model,
            require_offsets=require_offsets,
            case_sensitive=case_sensitive,
            retries=retries,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
            few_shot_examples=few_shot_examples,
        )
        ner_config_patch.additional_properties = d
        return ner_config_patch

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
