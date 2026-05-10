from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ner_service.config import Settings
from ner_service.providers.registry import get_provider
from ner_service.schemas import EntityLabel, ExtractRequest, NERConfig
from ner_service.service import NerService

LABELS = [
    EntityLabel(name="PERSON", description="People"),
    EntityLabel(name="ORG", description="Organizations"),
    EntityLabel(name="LOCATION", description="Locations"),
]
SEED_SENTENCE = (
    "Tim Cook visited Berlin with OpenAI researchers before Microsoft joined the meeting. "
)


@dataclass(frozen=True)
class RunResult:
    latency_ms: float
    usage: dict[str, Any] | None
    error: str | None


def _build_text(length: int, index: int) -> str:
    prefix = f"sample {index}: "
    repeated = (SEED_SENTENCE * ((length // len(SEED_SENTENCE)) + 2)).strip()
    body = repeated[: max(length - len(prefix), 1)]
    return prefix + body


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[position]


def _sum_usage(results: Sequence[RunResult]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for result in results:
        if result.usage is None:
            continue
        for key, value in result.usage.items():
            if isinstance(value, int | float):
                total[key] = total.get(key, 0) + value
            elif isinstance(value, dict):
                existing = total.setdefault(key, {})
                if isinstance(existing, dict):
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, int | float):
                            existing[nested_key] = existing.get(nested_key, 0) + nested_value
            elif key not in total:
                total[key] = value
    return total


async def _run_once(
    service: NerService,
    config_id: str,
    text: str,
    semaphore: asyncio.Semaphore,
) -> RunResult:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await service.extract(ExtractRequest(text=text, config_id=config_id))
        except Exception as exc:
            return RunResult(
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=None,
                error=str(exc),
            )
        return RunResult(
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=response.usage,
            error=None,
        )


async def _profile_length(
    service: NerService,
    config_id: str,
    length: int,
    texts_count: int,
    concurrency: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    texts = [_build_text(length, index) for index in range(1, texts_count + 1)]
    started = time.perf_counter()
    results = await asyncio.gather(
        *[_run_once(service, config_id, text, semaphore) for text in texts]
    )
    wall_s = time.perf_counter() - started
    latencies = [result.latency_ms for result in results]
    successes = [result for result in results if result.error is None]
    errors = [result.error for result in results if result.error is not None]
    return {
        "text_length": length,
        "requests": len(results),
        "successes": len(successes),
        "errors": len(errors),
        "error_messages": errors[:5],
        "throughput_rps": (len(results) / wall_s) if wall_s else 0.0,
        "wall_time_s": wall_s,
        "latency_ms": {
            "min": min(latencies) if latencies else 0.0,
            "mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0.0,
        },
        "_latency_samples_ms": latencies,
        "usage": _sum_usage(successes),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    settings_kwargs: dict[str, Any] = {}
    if args.provider is not None:
        settings_kwargs["ner_provider"] = args.provider
    if args.model is not None:
        settings_kwargs["ner_model"] = args.model
    if args.max_tokens is not None:
        settings_kwargs["max_tokens"] = args.max_tokens
    settings = Settings(**settings_kwargs)
    provider = get_provider(settings)
    service = NerService(provider, default_model=settings.ner_model, max_tokens=settings.max_tokens)
    config = NERConfig(
        labels=LABELS,
        model=settings.ner_model,
        retries=args.retries,
        max_tokens=settings.max_tokens,
    )
    config_id = (await service.create_config(config)).id
    started_at = datetime.now(UTC).isoformat()
    try:
        results = []
        for length in args.text_lengths:
            results.append(
                await _profile_length(
                    service=service,
                    config_id=config_id,
                    length=length,
                    texts_count=args.texts_count,
                    concurrency=args.concurrency,
                )
            )
    finally:
        await service.aclose()
    all_latencies = [latency for result in results for latency in result["_latency_samples_ms"]]
    completed_requests = sum(result["requests"] for result in results)
    total_wall_s = sum(result["wall_time_s"] for result in results)
    success_count = sum(result["successes"] for result in results)
    error_count = sum(result["errors"] for result in results)
    usage_total: dict[str, Any] = {}
    for result in results:
        for key, value in result["usage"].items():
            if isinstance(value, int | float):
                usage_total[key] = usage_total.get(key, 0) + value
            elif isinstance(value, dict):
                existing = usage_total.setdefault(key, {})
                if isinstance(existing, dict):
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, int | float):
                            existing[nested_key] = existing.get(nested_key, 0) + nested_value
        result.pop("_latency_samples_ms", None)
    return {
        "started_at": started_at,
        "provider": settings.ner_provider,
        "model": settings.ner_model,
        "concurrency": args.concurrency,
        "texts_count": args.texts_count,
        "text_lengths": args.text_lengths,
        "max_tokens": settings.max_tokens,
        "retries": args.retries,
        "results": results,
        "overall": {
            "requests": completed_requests,
            "successes": success_count,
            "errors": error_count,
            "wall_time_s": total_wall_s,
            "throughput_rps": (completed_requests / total_wall_s) if total_wall_s else 0.0,
            "latency_ms": {
                "min": min(all_latencies) if all_latencies else 0.0,
                "mean": (sum(all_latencies) / len(all_latencies)) if all_latencies else 0.0,
                "p50": _percentile(all_latencies, 0.50),
                "p95": _percentile(all_latencies, 0.95),
                "p99": _percentile(all_latencies, 0.99),
                "max": max(all_latencies) if all_latencies else 0.0,
            },
            "usage": usage_total,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--texts-count", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--text-lengths", default="64,256,1024,4096")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.text_lengths = [int(item) for item in str(args.text_lengths).split(",") if item.strip()]
    report = asyncio.run(_run(args))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
