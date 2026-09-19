from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..models.span_pipeline_policy_on_unavailable import SpanPipelinePolicyOnUnavailable
from ..models.span_pipeline_policy_prefilter import SpanPipelinePolicyPrefilter
from ..types import UNSET, Unset

T = TypeVar("T", bound="SpanPipelinePolicy")


@_attrs_define
class SpanPipelinePolicy:
    """
    Attributes:
        model (str | Unset):  Default: 'jev-1.13.0'.
        prefilter (SpanPipelinePolicyPrefilter | Unset):  Default: SpanPipelinePolicyPrefilter.AUTO.
        min_label_probability (float | Unset):  Default: 0.6.
        max_candidates (int | Unset):  Default: 256.
        max_candidates_per_request (int | Unset):  Default: 30.
        window_words (int | Unset):  Default: 2.
        cascade (bool | Unset):  Default: False.
        cascade_low (float | Unset):  Default: 0.4.
        cascade_high (float | Unset):  Default: 0.6.
        cascade_window_words (int | Unset):  Default: 5.
        on_unavailable (SpanPipelinePolicyOnUnavailable | Unset):  Default: SpanPipelinePolicyOnUnavailable.FAIL.
    """

    model: str | Unset = "jev-1.13.0"
    prefilter: SpanPipelinePolicyPrefilter | Unset = SpanPipelinePolicyPrefilter.AUTO
    min_label_probability: float | Unset = 0.6
    max_candidates: int | Unset = 256
    max_candidates_per_request: int | Unset = 30
    window_words: int | Unset = 2
    cascade: bool | Unset = False
    cascade_low: float | Unset = 0.4
    cascade_high: float | Unset = 0.6
    cascade_window_words: int | Unset = 5
    on_unavailable: SpanPipelinePolicyOnUnavailable | Unset = SpanPipelinePolicyOnUnavailable.FAIL

    def to_dict(self) -> dict[str, Any]:
        model = self.model

        prefilter: str | Unset = UNSET
        if not isinstance(self.prefilter, Unset):
            prefilter = self.prefilter.value

        min_label_probability = self.min_label_probability

        max_candidates = self.max_candidates

        max_candidates_per_request = self.max_candidates_per_request

        window_words = self.window_words

        cascade = self.cascade

        cascade_low = self.cascade_low

        cascade_high = self.cascade_high

        cascade_window_words = self.cascade_window_words

        on_unavailable: str | Unset = UNSET
        if not isinstance(self.on_unavailable, Unset):
            on_unavailable = self.on_unavailable.value

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if model is not UNSET:
            field_dict["model"] = model
        if prefilter is not UNSET:
            field_dict["prefilter"] = prefilter
        if min_label_probability is not UNSET:
            field_dict["min_label_probability"] = min_label_probability
        if max_candidates is not UNSET:
            field_dict["max_candidates"] = max_candidates
        if max_candidates_per_request is not UNSET:
            field_dict["max_candidates_per_request"] = max_candidates_per_request
        if window_words is not UNSET:
            field_dict["window_words"] = window_words
        if cascade is not UNSET:
            field_dict["cascade"] = cascade
        if cascade_low is not UNSET:
            field_dict["cascade_low"] = cascade_low
        if cascade_high is not UNSET:
            field_dict["cascade_high"] = cascade_high
        if cascade_window_words is not UNSET:
            field_dict["cascade_window_words"] = cascade_window_words
        if on_unavailable is not UNSET:
            field_dict["on_unavailable"] = on_unavailable

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        model = d.pop("model", UNSET)

        _prefilter = d.pop("prefilter", UNSET)
        prefilter: SpanPipelinePolicyPrefilter | Unset
        if isinstance(_prefilter, Unset):
            prefilter = UNSET
        else:
            prefilter = SpanPipelinePolicyPrefilter(_prefilter)

        min_label_probability = d.pop("min_label_probability", UNSET)

        max_candidates = d.pop("max_candidates", UNSET)

        max_candidates_per_request = d.pop("max_candidates_per_request", UNSET)

        window_words = d.pop("window_words", UNSET)

        cascade = d.pop("cascade", UNSET)

        cascade_low = d.pop("cascade_low", UNSET)

        cascade_high = d.pop("cascade_high", UNSET)

        cascade_window_words = d.pop("cascade_window_words", UNSET)

        _on_unavailable = d.pop("on_unavailable", UNSET)
        on_unavailable: SpanPipelinePolicyOnUnavailable | Unset
        if isinstance(_on_unavailable, Unset):
            on_unavailable = UNSET
        else:
            on_unavailable = SpanPipelinePolicyOnUnavailable(_on_unavailable)

        span_pipeline_policy = cls(
            model=model,
            prefilter=prefilter,
            min_label_probability=min_label_probability,
            max_candidates=max_candidates,
            max_candidates_per_request=max_candidates_per_request,
            window_words=window_words,
            cascade=cascade,
            cascade_low=cascade_low,
            cascade_high=cascade_high,
            cascade_window_words=cascade_window_words,
            on_unavailable=on_unavailable,
        )

        return span_pipeline_policy
