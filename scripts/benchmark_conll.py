from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ner_service.config import Settings
from ner_service.providers.registry import get_provider
from ner_service.schemas import EntityLabel, ExtractRequest, NERConfig
from ner_service.service import NerService

CONLL_LABEL_MAP = {
    "PER": EntityLabel(name="PERSON", description="People, real or fictional"),
    "ORG": EntityLabel(name="ORG", description="Companies, agencies, institutions"),
    "LOC": EntityLabel(name="LOCATION", description="Cities, countries, geographic locations"),
    "MISC": EntityLabel(
        name="MISC",
        description="Miscellaneous named entities (events, nationalities, products)",
    ),
}

CONLL_TO_SCHEMA = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "MISC": "MISC",
}

DEFAULT_DATASET_CACHE = Path("data/benchmarks/conll2003-test.jsonl")


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class EntityKey:
    label: str
    text: str


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int
    duration_s: float
    error: bool
    usage: dict[str, Any] | None


def _load_conll(limit: int | None, cache_path: Path) -> list[tuple[str, list[Span]]]:
    if cache_path.exists():
        return _read_conll_cache(limit, cache_path)

    try:
        from datasets import ClassLabel, load_dataset
    except ImportError:
        sys.exit("datasets package is required. Install with: uv sync --extra dev")

    ds = load_dataset("eriktks/conll2003", split="test", revision="refs/convert/parquet")
    ner_feature = ds.features["ner_tags"]
    tag_feature = getattr(ner_feature, "feature", None)
    assert isinstance(tag_feature, ClassLabel)
    names: list[str] = tag_feature.names

    rows: list[tuple[str, list[Span]]] = []
    for row in ds:
        tokens = row["tokens"]
        tags = [names[t] for t in row["ner_tags"]]
        text, spans = _bio_to_spans(tokens, tags)
        rows.append((text, spans))
    _write_conll_cache(cache_path, rows)
    return rows[:limit] if limit is not None else rows


def _read_conll_cache(limit: int | None, cache_path: Path) -> list[tuple[str, list[Span]]]:
    rows: list[tuple[str, list[Span]]] = []
    with cache_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            item = json.loads(line)
            rows.append(
                (
                    item["text"],
                    [Span(label=s["label"], start=s["start"], end=s["end"]) for s in item["spans"]],
                )
            )
    return rows


def _write_conll_cache(cache_path: Path, rows: list[tuple[str, list[Span]]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for text, spans in rows:
            item = {
                "text": text,
                "spans": [
                    {"label": span.label, "start": span.start, "end": span.end} for span in spans
                ],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _bio_to_spans(tokens: list[str], tags: list[str]) -> tuple[str, list[Span]]:
    pieces: list[str] = []
    cursor = 0
    token_starts: list[int] = []
    for tok in tokens:
        if pieces:
            pieces.append(" ")
            cursor += 1
        token_starts.append(cursor)
        pieces.append(tok)
        cursor += len(tok)
    text = "".join(pieces)

    spans: list[Span] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O":
            i += 1
            continue
        prefix, _, raw_label = tag.partition("-")
        label = CONLL_TO_SCHEMA.get(raw_label)
        if label is None or prefix != "B":
            i += 1
            continue
        start = token_starts[i]
        end = token_starts[i] + len(tokens[i])
        j = i + 1
        while j < len(tags):
            nxt_prefix, _, nxt_raw = tags[j].partition("-")
            if nxt_prefix == "I" and nxt_raw == raw_label:
                end = token_starts[j] + len(tokens[j])
                j += 1
            else:
                break
        spans.append(Span(label=label, start=start, end=end))
        i = j
    return text, spans


async def _score_one(
    service: NerService,
    config_id: str,
    idx: int,
    text: str,
    gold_spans: list[Span],
    sem: asyncio.Semaphore,
    require_offsets: bool,
) -> Score:
    if not text.strip():
        return Score(0, 0, 0, 0.0, False, None)
    async with sem:
        started = time.perf_counter()
        try:
            response = await service.extract(
                ExtractRequest(
                    text=text,
                    config_id=config_id,
                )
            )
        except Exception as e:
            print(f"[{idx}] error: {e}", file=sys.stderr)
            duration = time.perf_counter() - started
            fn = len(gold_spans) if require_offsets else len(_dictionary_gold(text, gold_spans))
            return Score(0, 0, fn, duration, True, None)

    duration = time.perf_counter() - started
    if require_offsets:
        pred = {
            Span(label=e.label, start=e.start, end=e.end)
            for e in response.entities
            if e.start is not None and e.end is not None
        }
        gold = set(gold_spans)
    else:
        pred = {EntityKey(label=e.label, text=e.text) for e in response.entities}
        gold = _dictionary_gold(text, gold_spans)
    return Score(
        len(pred & gold),
        len(pred - gold),
        len(gold - pred),
        duration,
        False,
        response.usage,
    )


def _dictionary_gold(text: str, gold_spans: list[Span]) -> set[EntityKey]:
    return {EntityKey(label=s.label, text=text[s.start : s.end]) for s in gold_spans}


async def _run(
    limit: int | None,
    concurrency: int,
    model: str | None,
    reasoning_effort: str | None,
    require_offsets: bool,
    case_sensitive: bool,
    retries: int,
    max_tokens: int | None,
    dataset_cache: Path,
) -> None:
    settings_kwargs: dict[str, Any] = {}
    if model is not None:
        settings_kwargs["ner_model"] = model
    if max_tokens is not None:
        settings_kwargs["max_tokens"] = max_tokens
    settings = Settings(**settings_kwargs)
    provider = get_provider(settings)
    service = NerService(provider, default_model=settings.ner_model, max_tokens=settings.max_tokens)

    labels = list(CONLL_LABEL_MAP.values())
    config = NERConfig(
        labels=labels,
        model=settings.ner_model,
        require_offsets=require_offsets,
        case_sensitive=case_sensitive,
        retries=retries,
        max_tokens=settings.max_tokens,
        reasoning_effort=reasoning_effort,
    )
    config_id = service.create_config(config).id
    rows = _load_conll(limit, dataset_cache)
    print(
        f"Loaded {len(rows)} examples; mode={'offsets' if require_offsets else 'dictionary'}; "
        f"model={settings.ner_model}; reasoning_effort={reasoning_effort or 'default'}; "
        f"max_tokens={settings.max_tokens}; retries={retries}; concurrency={concurrency}; "
        f"case_sensitive={case_sensitive}; dataset_cache={dataset_cache}"
    )

    sem = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _score_one(
                service,
                config_id,
                i,
                text,
                gold,
                sem,
                require_offsets,
            )
        )
        for i, (text, gold) in enumerate(rows, 1)
    ]

    tp = fp = fn = 0
    errors = 0
    durations: list[float] = []
    usage_total: dict[str, Any] = {}
    total = len(tasks)
    for done, coro in enumerate(asyncio.as_completed(tasks), 1):
        score = await coro
        tp += score.tp
        fp += score.fp
        fn += score.fn
        errors += int(score.error)
        durations.append(score.duration_s)
        _merge_usage(usage_total, score.usage)
        if done % 25 == 0 or done == total:
            avg = sum(durations) / len(durations) if durations else 0.0
            print(f"[{done}/{total}] tp={tp} fp={fp} fn={fn} errors={errors} avg_s={avg:.3f}")

    await service.aclose()

    elapsed = time.perf_counter() - started
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_s = sum(durations) / len(durations) if durations else 0.0
    min_s = min(durations) if durations else 0.0
    max_s = max(durations) if durations else 0.0
    throughput = total / elapsed if elapsed else 0.0

    print(f"\nmode={'offsets' if require_offsets else 'dictionary'}")
    print(f"model={settings.ner_model}")
    print(f"reasoning_effort={reasoning_effort or 'default'}")
    print(f"case_sensitive={case_sensitive}")
    print(
        f"examples={total} concurrency={concurrency} "
        f"max_tokens={settings.max_tokens} retries={retries}"
    )
    print(f"\nmicro-P={precision:.3f} micro-R={recall:.3f} micro-F1={f1:.3f}")
    print(f"tp={tp} fp={fp} fn={fn} errors={errors}")
    print(
        f"total_s={elapsed:.3f} avg_example_s={avg_s:.3f} "
        f"min_example_s={min_s:.3f} max_example_s={max_s:.3f} throughput_eps={throughput:.3f}"
    )
    if usage_total:
        print("usage=" + " ".join(f"{k}={v}" for k, v in sorted(usage_total.items())))


def _merge_usage(total: dict[str, Any], usage: dict[str, Any] | None) -> None:
    if usage is None:
        return
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            existing = total.setdefault(key, {})
            if isinstance(existing, dict):
                _merge_usage(existing, value)
        elif key not in total:
            total[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="number of examples (default: all)")
    parser.add_argument("--model", default=None, help="model id (default: NER_MODEL)")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="reasoning effort for supported models",
    )
    parser.add_argument("--require-offsets", action="store_true", help="score exact offsets")
    case_group = parser.add_mutually_exclusive_group()
    case_group.add_argument(
        "--case-sensitive",
        dest="case_sensitive",
        action="store_true",
        default=True,
        help="match entity text case-sensitively (default)",
    )
    case_group.add_argument(
        "--case-insensitive",
        dest="case_sensitive",
        action="store_false",
        help="match and canonicalize entity text case-insensitively",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="total attempts per example (default: 3)",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="max completion tokens")
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=DEFAULT_DATASET_CACHE,
        help=f"local CoNLL-2003 test cache path (default: {DEFAULT_DATASET_CACHE})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=40,
        help="max concurrent in-flight requests (default: 40)",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            args.limit,
            args.concurrency,
            args.model,
            args.reasoning_effort,
            args.require_offsets,
            args.case_sensitive,
            args.retries,
            args.max_tokens,
            args.dataset_cache,
        )
    )


if __name__ == "__main__":
    main()
