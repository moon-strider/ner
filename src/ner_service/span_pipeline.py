from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ner_service.candidates import (
    CANDIDATE_GENERATOR_VERSION,
    Candidate,
    choose_prefilter,
    generate_candidates,
)
from ner_service.config_store import PreparedNERConfig
from ner_service.providers.base import ProviderError
from ner_service.schemas import Entity, SpanPipelinePolicy
from ner_service.typesafe import NONE_OF_THESE, JevClient


@dataclass(frozen=True)
class SpanResult:
    entities: list[Entity]
    usage: dict[str, Any] | None
    warnings: list[str]
    attempts: int = 1
    degraded: bool = False
    reason: str | None = None
    model: str = ""


class SpanPipeline(Protocol):
    name: str

    async def extract(self, text: str, *, prepared: PreparedNERConfig) -> SpanResult: ...

    async def aclose(self) -> None: ...


def _resolve_mode(text: str, policy: SpanPipelinePolicy) -> str:
    if policy.prefilter == "auto":
        return choose_prefilter(text)
    return policy.prefilter


def _chunks(items: list[Candidate], size: int) -> list[list[Candidate]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _select(accepted: list[tuple[Candidate, str, float]]) -> list[Entity]:
    ordered = sorted(
        accepted, key=lambda row: (row[0].start, -row[2], -(row[0].end - row[0].start))
    )
    chosen: list[Entity] = []
    taken: list[tuple[int, int]] = []
    for candidate, label, _probability in ordered:
        if any(candidate.start < end and start < candidate.end for start, end in taken):
            continue
        taken.append((candidate.start, candidate.end))
        chosen.append(
            Entity(text=candidate.text, label=label, start=candidate.start, end=candidate.end)
        )
    return sorted(chosen, key=lambda entity: (entity.start or 0, entity.end or 0))


class JevSpanPipeline:
    name = "typesafe"

    def __init__(self, client: JevClient) -> None:
        self._client = client

    async def extract(self, text: str, *, prepared: PreparedNERConfig) -> SpanResult:
        policy = prepared.config.span_pipeline
        if policy is None:
            raise ValueError("span pipeline policy is required")
        labels = list(prepared.config.labels)
        allowed_labels = {label.name for label in labels}
        mode = _resolve_mode(text, policy)
        candidates = generate_candidates(text, mode=mode, max_candidates=policy.max_candidates)
        warnings: list[str] = []
        if len(candidates) == policy.max_candidates:
            warnings.append(
                f"Candidate generation reached the {policy.max_candidates}-candidate limit; "
                "lower-priority candidates may have been dropped."
            )
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        accepted: list[tuple[Candidate, str, float]] = []
        no_judgment = 0
        requests = 0
        model = ""
        try:
            for chunk in _chunks(candidates, policy.max_candidates_per_request):
                verdict = await self._client.judge(
                    text,
                    chunk,
                    labels,
                    window_words=policy.window_words,
                )
                requests += 1
                model = verdict.model or model
                no_judgment += verdict.no_judgment
                for key, value in verdict.usage.items():
                    usage[key] = usage.get(key, 0) + value
                for judgment in verdict.judgments:
                    label = judgment.label
                    if label is None or label == NONE_OF_THESE:
                        continue
                    if label not in allowed_labels:
                        continue
                    if judgment.probability < policy.min_label_probability:
                        continue
                    accepted.append((judgment.candidate, label, judgment.probability))
            if policy.cascade and accepted:
                accepted = await self._cascade(
                    text, accepted, policy, labels, allowed_labels, usage, warnings
                )
        except ProviderError as exc:
            if policy.on_unavailable == "fail":
                raise
            return SpanResult(
                entities=[],
                usage=usage or None,
                warnings=[
                    *warnings,
                    f"Span pipeline unavailable; no entities returned ({type(exc).__name__}).",
                ],
                attempts=requests,
                degraded=True,
                reason=type(exc).__name__,
                model=model,
            )
        entities = _select(accepted)
        if candidates:
            warnings.append(
                f"Span pipeline judged {len(candidates)} candidates, accepted "
                f"{len(accepted)}, returned {len(entities)}."
            )
        if no_judgment:
            warnings.append(f"{no_judgment} candidates received no judgment.")
        return SpanResult(
            entities=entities,
            usage=usage or None,
            warnings=warnings,
            attempts=max(requests, 1),
            model=model,
        )

    async def _cascade(
        self,
        text: str,
        accepted: list[tuple[Candidate, str, float]],
        policy: SpanPipelinePolicy,
        labels: list[Any],
        allowed_labels: set[str],
        usage: dict[str, int],
        warnings: list[str],
    ) -> list[tuple[Candidate, str, float]]:
        band = [row for row in accepted if policy.cascade_low <= row[2] <= policy.cascade_high]
        if not band:
            return accepted
        by_span = {(row[0].start, row[0].end): row for row in accepted}
        candidates = [row[0] for row in band]
        for chunk in _chunks(candidates, policy.max_candidates_per_request):
            verdict = await self._client.judge(
                text,
                chunk,
                labels,
                window_words=policy.cascade_window_words,
                negative_focus=True,
            )
            for key, value in verdict.usage.items():
                usage[key] = usage.get(key, 0) + value
            for judgment in verdict.judgments:
                span = (judgment.candidate.start, judgment.candidate.end)
                if span not in by_span:
                    continue
                label = judgment.label
                if (
                    label is None
                    or label == NONE_OF_THESE
                    or label not in allowed_labels
                    or judgment.probability < policy.min_label_probability
                ):
                    by_span.pop(span, None)
                    continue
                by_span[span] = (judgment.candidate, label, judgment.probability)
        warnings.append(f"Span pipeline re-asked {len(candidates)} uncertain candidates.")
        return list(by_span.values())

    async def aclose(self) -> None:
        await self._client.aclose()


def generator_version() -> int:
    return CANDIDATE_GENERATOR_VERSION
