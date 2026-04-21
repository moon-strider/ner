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
CONFIG_ID="$(curl -s -X POST http://localhost:8000/configs \
  -H 'Content-Type: application/json' \
  -d '{
    "labels": [
      {"name": "PERSON", "description": "People, real or fictional"},
      {"name": "LOCATION", "description": "Cities, countries, places"}
    ]
  }' | jq -r .id)"

curl -s -X POST http://localhost:8000/extract \
  -H 'Content-Type: application/json' \
  -d "{
    \"text\": \"Tim Cook visited Berlin last week.\",
    \"config_id\": \"${CONFIG_ID}\"
  }" | jq
```

Response:

```json
{
  "data": {
    "entities": [
      {"text": "Tim Cook", "label": "PERSON"},
      {"text": "Berlin", "label": "LOCATION"}
    ],
    "model": "llama3.1-8b",
    "provider": "cerebras"
  },
  "meta": {
    "request_id": "0b4c6a80-4c4b-4cd0-9b1f-30b66b8e3e22",
    "latency_ms": 824.6,
    "attempts": 1,
    "warnings": []
  }
}
```

## API

- `GET /health` — liveness probe.
- `GET /ready` — readiness probe; verifies service initialization without making an LLM call.
- `GET /providers` — current provider and model.
- `POST /configs` — create an in-memory NER config; returns `{id, config}`.
- `GET /configs` — list configs.
- `GET /configs/{id}` — get one config.
- `PUT /configs/{id}` — replace one config.
- `PATCH /configs/{id}` — partially update one config.
- `DELETE /configs/{id}` — delete one config.
- `POST /extract` — body: `{text, config_id?, config?, prompt_payload?}`. Exactly one of `config_id` or inline `config` is required. Returns `{data, meta}`.

`NERConfig` fields:

```json
{
  "labels": [{"name": "PERSON", "description": "People"}],
  "model": "llama3.1-8b",
  "require_offsets": false,
  "case_sensitive": true,
  "retries": 3,
  "max_tokens": 1024,
  "reasoning_effort": null,
  "system_prompt": null
}
```

Configs are stored only in process memory and are lost on service restart. Create a config once and reuse `config_id` for high-volume extraction to avoid resending labels, prompt, and schema on every request. Inline configs are still supported for one-off calls.

Set `require_offsets=true` to recover `start` / `end` through substring matching. Set `case_sensitive=false` to match model surfaces case-insensitively; returned entity text uses the source text casing when a match is found.

`system_prompt` is a full prompt override template. Placeholders support `{cfg.field}` and `{payload.field}` with dotted access. `cfg.schema` is the compact JSON Schema generated from labels; `payload` is supplied per `/extract` call:

```json
{
  "text": "Tim Cook visited Berlin last week.",
  "config_id": "CONFIG_ID",
  "prompt_payload": {"number": 42}
}
```

Example template:

```text
follow schema strictly: {cfg.schema}; remember this number: {payload.number}
```

Use `{{` and `}}` for literal braces.

## Configuration

Environment variables (also read from `.env`):

| Variable | Default | Description |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | — | Required when `NER_PROVIDER=cerebras`. |
| `NER_PROVIDER` | `cerebras` | Provider id. |
| `NER_MODEL` | `llama3.1-8b` | Model identifier passed to the provider. |
| `REQUEST_TIMEOUT_S` | `30` | Per-request upstream timeout. |
| `TRANSPORT_RETRIES` | `2` | SDK/network retries for upstream transport failures. |
| `MAX_TOKENS` | `1024` | Default `max_completion_tokens` passed to the provider. |
| `MAX_TEXT_LENGTH` | `32000` | Maximum input text length accepted by the service. |
| `MAX_LABELS` | `50` | Maximum labels per NER config. |
| `MAX_SYSTEM_PROMPT_LENGTH` | `20000` | Maximum custom `system_prompt` length. |
| `MAX_LABEL_DESCRIPTION_LENGTH` | `500` | Maximum label description length. |
| `MAX_CONFIG_ID_LENGTH` | `128` | Maximum config id length accepted by API paths and payloads. |

`NERConfig.retries` controls model repair attempts when the provider returns invalid structured output. `TRANSPORT_RETRIES` controls SDK/network retries before the service receives a provider response.

Errors use a single envelope:

```json
{
  "error": {
    "code": "validation_error",
    "message": "request validation failed",
    "details": {},
    "request_id": "0b4c6a80-4c4b-4cd0-9b1f-30b66b8e3e22"
  }
}
```

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

Default scoring is exact unique `(label, text)` pairs. `--require-offsets` scores exact `(label, start, end)` triples. The script creates one in-memory NER config and then extracts by `config_id`. It prints micro-P / micro-R / micro-F1, errors, token usage, total time, throughput, and avg/min/max per-example latency.

The CoNLL-2003 `test` split is cached locally at `data/benchmarks/conll2003-test.jsonl` after the first load. Later runs reuse that file instead of downloading the dataset again.

Full CoNLL-2003 `test` baseline results (3453 examples, concurrency 40, max_tokens 1024, retries 3):

| Model | Mode | Reasoning | micro-P | micro-R | micro-F1 | TP | FP | FN | Errors | Total s | Avg s | Min s | Max s | Examples/s | Total tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.1-8b` | dictionary | N/A | 0.531 | 0.737 | 0.617 | 4115 | 3637 | 1469 | 2 | 68.163 | 0.772 | 0.350 | 5.083 | 50.658 | 1436014 |
| `llama3.1-8b` | offsets | N/A | 0.509 | 0.726 | 0.598 | 4098 | 3950 | 1550 | 1 | 114.363 | 1.314 | 0.342 | 63.124 | 30.193 | 1445445 |
| `gpt-oss-120b` | dictionary | low | 0.778 | 0.776 | 0.777 | 4335 | 1240 | 1249 | 0 | 204.446 | 2.343 | 0.298 | 66.743 | 16.890 | 1758947 |
| `gpt-oss-120b` | offsets | low | 0.771 | 0.774 | 0.773 | 4374 | 1297 | 1274 | 0 | 207.912 | 2.214 | 0.304 | 62.849 | 16.608 | 1815023 |

Cost to run:

- `llama3.1-8b`: ±0.3$ per run
- `gpt-oss-120b`: ±0.69$ per run

## Docker

```bash
docker build -t ner-service .
docker run --rm -p 8000:8000 --env-file .env ner-service
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
```
