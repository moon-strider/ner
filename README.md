# ner-service

HTTP NER (Named Entity Recognition) service powered by LLMs with structured output.

- FastAPI + Python 3.12 + [uv](https://docs.astral.sh/uv/)
- Client-defined entity labels per request (dynamic JSON Schema)
- Character offsets (`start`, `end`) recovered via substring matching

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
    {"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8},
    {"text": "Berlin", "label": "LOCATION", "start": 17, "end": 23}
  ],
  "model": "llama3.1-8b",
  "provider": "cerebras"
}
```

## API

- `GET /health` — liveness probe.
- `GET /providers` — current provider and model.
- `POST /extract` — body: `{text, labels: [{name, description}, ...]}`. Label `name` must match `^[A-Z][A-Z0-9_]*$`.

## Configuration

Environment variables (also read from `.env`):

| Variable | Default | Description |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | — | Required when `NER_PROVIDER=cerebras`. |
| `NER_PROVIDER` | `cerebras` | Provider id. |
| `NER_MODEL` | `llama3.1-8b` | Model identifier passed to the provider. |
| `REQUEST_TIMEOUT_S` | `30` | Per-request upstream timeout. |

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
CEREBRAS_API_KEY=... uv run --extra dev python scripts/benchmark_conll.py --limit 200
```

Prints micro-P / micro-R / micro-F1 on (label, start, end) triples.

## Docker

```bash
docker build -t ner-service .
docker run --rm -p 8000:8000 --env-file .env ner-service
```

## License

MIT.
