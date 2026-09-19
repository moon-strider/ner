from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.entity_label import EntityLabel
    from ..models.few_shot_example import FewShotExample
    from ..models.span_pipeline_policy import SpanPipelinePolicy


T = TypeVar("T", bound="NERConfig")


@_attrs_define
class NERConfig:
    """
    Attributes:
        labels (list[EntityLabel]):
        model (str | Unset):
        require_offsets (bool | Unset):  Default: False.
        case_sensitive (bool | Unset):  Default: True.
        retries (int | Unset):  Default: 3.
        max_tokens (int | Unset):
        reasoning_effort (None | str | Unset):
        system_prompt (None | str | Unset):
        few_shot_examples (list[FewShotExample] | Unset):
        span_pipeline (None | SpanPipelinePolicy | Unset):
    """

    labels: list[EntityLabel]
    model: str | Unset = UNSET
    require_offsets: bool | Unset = False
    case_sensitive: bool | Unset = True
    retries: int | Unset = 3
    max_tokens: int | Unset = UNSET
    reasoning_effort: None | str | Unset = UNSET
    system_prompt: None | str | Unset = UNSET
    few_shot_examples: list[FewShotExample] | Unset = UNSET
    span_pipeline: None | SpanPipelinePolicy | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.span_pipeline_policy import SpanPipelinePolicy

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

        span_pipeline: dict[str, Any] | None | Unset
        if isinstance(self.span_pipeline, Unset):
            span_pipeline = UNSET
        elif isinstance(self.span_pipeline, SpanPipelinePolicy):
            span_pipeline = self.span_pipeline.to_dict()
        else:
            span_pipeline = self.span_pipeline

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "labels": labels,
            }
        )
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
        if span_pipeline is not UNSET:
            field_dict["span_pipeline"] = span_pipeline

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.entity_label import EntityLabel
        from ..models.few_shot_example import FewShotExample
        from ..models.span_pipeline_policy import SpanPipelinePolicy

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

        def _parse_span_pipeline(data: object) -> None | SpanPipelinePolicy | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                span_pipeline_type_0 = SpanPipelinePolicy.from_dict(data)

                return span_pipeline_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SpanPipelinePolicy | Unset, data)

        span_pipeline = _parse_span_pipeline(d.pop("span_pipeline", UNSET))

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
            span_pipeline=span_pipeline,
        )

        return ner_config
