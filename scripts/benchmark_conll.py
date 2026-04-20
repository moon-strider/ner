from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

from ner_service.config import Settings
from ner_service.providers.registry import get_provider
from ner_service.schemas import EntityLabel, ExtractRequest
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


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int


def _load_conll(limit: int | None) -> list[tuple[str, list[Span]]]:
    try:
        from datasets import ClassLabel, load_dataset
    except ImportError:
        sys.exit("datasets package is required. Install with: uv sync --extra dev")

    ds = load_dataset("conll2003", split="test", trust_remote_code=True)
    tag_feature = ds.features["ner_tags"].feature
    assert isinstance(tag_feature, ClassLabel)
    names: list[str] = tag_feature.names

    rows: list[tuple[str, list[Span]]] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        tokens = row["tokens"]
        tags = [names[t] for t in row["ner_tags"]]
        text, spans = _bio_to_spans(tokens, tags)
        rows.append((text, spans))
    return rows


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


async def _run(limit: int | None) -> None:
    settings = Settings()
    provider = get_provider(settings)
    service = NerService(provider)

    labels = list(CONLL_LABEL_MAP.values())
    rows = _load_conll(limit)
    print(f"Loaded {len(rows)} examples")

    tp = fp = fn = 0
    for idx, (text, gold_spans) in enumerate(rows, 1):
        if not text.strip():
            continue
        try:
            response = await service.extract(ExtractRequest(text=text, labels=labels))
        except Exception as e:
            print(f"[{idx}] error: {e}", file=sys.stderr)
            fn += len(gold_spans)
            continue
        pred = {Span(label=e.label, start=e.start, end=e.end) for e in response.entities}
        gold = set(gold_spans)
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        if idx % 25 == 0:
            print(f"[{idx}] tp={tp} fp={fp} fn={fn}")

    await service.aclose()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\nmicro-P={precision:.3f} micro-R={recall:.3f} micro-F1={f1:.3f}")
    print(f"tp={tp} fp={fp} fn={fn}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="number of examples (default: 200)")
    args = parser.parse_args()
    asyncio.run(_run(args.limit))


if __name__ == "__main__":
    main()
