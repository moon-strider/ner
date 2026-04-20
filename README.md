# ner-service

HTTP NER (Named Entity Recognition) service powered by LLMs with structured output.

- FastAPI + Python 3.12 + [uv](https://docs.astral.sh/uv/)
- Client-defined entity labels per request (dynamic JSON Schema)
- Optional character offsets (`start`, `end`) recovered via substring matching

## Quick start

```bash
cp .env.example .env
uv sync --extra dev
uv run uvicorn ner_service.main:app --reload
```

Request:

```bash
curl -s -X POST http://localhost:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Tim Cook visited Berlin last week.",
    "labels": [
      {"name": "PERSON", "description": "People, real or fictional"},
      {"name": "LOCATION", "description": "Cities, countries, places"}
    ]
  }' | jq
```

Response:

```json
{
  "entities": [
    {"text": "Tim Cook", "label": "PERSON"},
    {"text": "Berlin", "label": "LOCATION"}
  ],
  "model": "llama3.1-8b",
  "provider": "cerebras"
}
```

## API

- `GET /health` — liveness probe.
- `GET /providers` — current provider and model.
- `POST /extract` — body: `{text, labels: [{name, description}, ...], require_offsets=false, retries=3, max_tokens=null}`. Label `name` must match `^[A-Z][A-Z0-9_]*$`.
- Set `require_offsets=true` to recover `start` / `end` through exact substring matching.
- `max_tokens` overrides the configured completion-token limit for that request.

## Configuration

Environment variables (also read from `.env`):

| Variable | Default | Description |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | — | Required when `NER_PROVIDER=cerebras`. |
| `NER_PROVIDER` | `cerebras` | Provider id. |
| `NER_MODEL` | `llama3.1-8b` | Model identifier passed to the provider. |
| `REQUEST_TIMEOUT_S` | `30` | Per-request upstream timeout. |
| `MAX_TOKENS` | `1024` | Default `max_completion_tokens` passed to the provider. |

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -m "not integration"
CEREBRAS_API_KEY=... uv run pytest -m integration
```

### CoNLL-2003 benchmark

```bash
CEREBRAS_API_KEY=... uv run --extra dev python scripts/benchmark_conll.py --model llama3.1-8b --concurrency 40
CEREBRAS_API_KEY=... uv run --extra dev python scripts/benchmark_conll.py --model llama3.1-8b --require-offsets --concurrency 40
CEREBRAS_API_KEY=... uv run --extra dev python scripts/benchmark_conll.py --model gpt-oss-120b --reasoning-effort low --concurrency 40
CEREBRAS_API_KEY=... uv run --extra dev python scripts/benchmark_conll.py --model gpt-oss-120b --reasoning-effort low --require-offsets --concurrency 40
```

Default scoring is exact unique `(label, text)` pairs. `--require-offsets` scores exact `(label, start, end)` triples. The script prints micro-P / micro-R / micro-F1, errors, token usage, total time, throughput, and avg/min/max per-example latency.

The CoNLL-2003 `test` split is cached locally at `data/benchmarks/conll2003-test.jsonl` after the first load. Later runs reuse that file instead of downloading the dataset again.

Latest full CoNLL-2003 `test` results (3453 examples, concurrency 40, max_tokens 1024, retries 3):

| Model | Mode | Reasoning | micro-P | micro-R | micro-F1 | TP | FP | FN | Errors | Total s | Avg s | Min s | Max s | Examples/s | Total tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.1-8b` | dictionary | default | 0.531 | 0.737 | 0.617 | 4115 | 3637 | 1469 | 2 | 68.163 | 0.772 | 0.350 | 5.083 | 50.658 | 1436014 |
| `llama3.1-8b` | offsets | default | 0.509 | 0.726 | 0.598 | 4098 | 3950 | 1550 | 1 | 114.363 | 1.314 | 0.342 | 63.124 | 30.193 | 1445445 |
| `gpt-oss-120b` | dictionary | low | 0.778 | 0.776 | 0.777 | 4335 | 1240 | 1249 | 0 | 204.446 | 2.343 | 0.298 | 66.743 | 16.890 | 1758947 |
| `gpt-oss-120b` | offsets | low | 0.771 | 0.774 | 0.773 | 4374 | 1297 | 1274 | 0 | 207.912 | 2.214 | 0.304 | 62.849 | 16.608 | 1815023 |

Cost to run:
`llama3.1-8b`: ±0.3$
`gpt-oss-120b`: ±0.69$

## Docker

```bash
docker build -t ner-service .
docker run --rm -p 8000:8000 --env-file .env ner-service
```
