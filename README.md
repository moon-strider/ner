# NER Service

[![CI](https://github.com/moon-strider/ner/actions/workflows/ci.yml/badge.svg)](https://github.com/moon-strider/ner/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB)](pyproject.toml)
[![MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Extract the entity types you define, through one HTTP API.**

Send text and a set of labels. Get structured entities, optional source offsets,
and request metadata. Use a local CPU model or an OpenAI-compatible cloud endpoint.

```json
{
  "text": "Tim Cook visited Berlin.",
  "config": {
    "labels": [
      {"name": "PERSON", "description": "Names of people"},
      {"name": "LOCATION", "description": "Cities and countries"}
    ],
    "require_offsets": true
  }
}
```

Example entity output (illustrative; recognition depends on the model):

```json
[
  {"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8},
  {"text": "Berlin", "label": "LOCATION", "start": 17, "end": 23}
]
```

- **Custom labels:** a JSON Schema is built for each configuration; invalid model output is checked and retried.
- **Grounded text:** unmatched entity surfaces are dropped. Offsets refer to the original Python string.
- **Reusable configurations:** inline requests or SQLite-backed config IDs, with CRUD and batch extraction.
- **Operational controls:** prompt-aware caching, provider concurrency limits, circuit breaker, optional Bearer authentication, Prometheus, and OpenTelemetry.
- **Typed Python client:** sync and async calls generated from the API, including error responses.

This is an LLM extraction service. JSON Schema constrains the response shape;
it does not guarantee that a label or entity is semantically correct.

## Jev

Jev is a model from TypeSafe that answers a fixed question instead of writing text. In this
mode the service stops asking a chat model to produce entities: simple code picks candidate
mentions out of the text, and Jev decides which of your labels each one gets.

Set `TYPESAFE_API_KEY` and add `span_pipeline` to a request or a stored config. Leave it unset
and nothing changes — the default provider path, including the local keyless setup, stays as it
is.

The mode adds a second service to the data flow. While it is on, the input text, the candidate
windows, your label names and descriptions, and the question wording go to TypeSafe
(`https://api.typesafe.ai` by default) under `TYPESAFE_API_KEY`. The default path and the local
quick start never contact it.

What I measured, where it breaks and when it is worth turning on is in [Jev](docs/jev.md).
Config fields are in the [API contract](docs/api.md); environment variables are in
[configuration](docs/configuration.md).

## Try it locally

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md).
No cloud API key is needed. The example uses SmolLM2 1.7B in Q4_K_M quantization
(about 1 GB of model weights). Allow additional RAM for the runtime and context.

Start the model server in one terminal:

```bash
llama-server -hf HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M \
  --alias smollm2-1.7b --host 127.0.0.1 --port 8080 \
  -c 4096 -t 4 -ngl 0 -np 1
```

In another terminal:

```bash
git clone https://github.com/moon-strider/ner.git
cd ner
cp .env.example .env
uv sync --frozen --no-dev
uv run --frozen --no-dev uvicorn ner_service.main:app --host 127.0.0.1 --port 8000
```

Make a request:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"I visited Berlin.","config":{"labels":[{"name":"LOCATION","description":"Cities and countries"}],"require_offsets":true}}'
```

Open [interactive API docs](http://127.0.0.1:8000/docs), or run the end-to-end smoke check:

```bash
uv run --frozen python scripts/smoke.py
```

The small CPU model is useful for trying the service, not an accuracy baseline.
See [local inference](docs/local-inference.md) for the tested runtime, limitations,
and cloud-provider setup in [configuration](docs/configuration.md).

## API at a glance

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/extract` | Extract with exactly one of `config` or `config_id` |
| `POST /v1/batch/extract` | Up to 100 items; ordered, per-item results |
| `POST /v1/configs` | Save a configuration |
| `GET /v1/configs` | List saved configurations |
| `GET /v1/configs/{id}` | Read a configuration |
| `PUT /v1/configs/{id}` | Replace an existing configuration |
| `PATCH /v1/configs/{id}` | Update selected fields |
| `DELETE /v1/configs/{id}` | Delete a configuration |
| `GET /v1/health`, `GET /v1/ready` | Liveness and local-storage readiness |
| `GET /v1/providers` | Configured provider and default model |
| `GET /metrics` | Prometheus metrics |

See the [API contract](docs/api.md) for response envelopes, offsets, caching,
few-shot examples, templates, and error codes.

## Run and develop

```bash
# Containerized service; reads .env and persists configs in a named volume.
docker compose up -d --build

# Add Prometheus and Grafana.
docker compose --profile observability up -d --build

# Development checks.
uv sync --frozen --extra dev
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src
uv run --frozen pytest -m 'not integration'
uv run --frozen python scripts/generate_client.py --check
```

For a model server running on the Docker host, set
`LLAMA_CPP_BASE_URL=http://host.docker.internal:8080/v1` in `.env` and make that
server reachable from the Docker bridge. See [operations](docs/operations.md).

| Guide | Contents |
| --- | --- |
| [API](docs/api.md) | Requests, responses, errors, and matching semantics |
| [Configuration](docs/configuration.md) | Providers and environment variables |
| [Jev](docs/jev.md) | What the judgment model measures, where it breaks, and when to use it |
| [Operations](docs/operations.md) | Docker, persistence, access control, metrics, tracing |
| [Local inference](docs/local-inference.md) | CPU setup and reproducible smoke testing |
| [Development](CONTRIBUTING.md) | Tests, client generation, packaging |
| [Benchmarks](docs/benchmarks.md) | Scoring methodology and historical results |
| [Audit](docs/audit.md) | Findings, fixes, validation evidence, remaining limits |

Licensed under [MIT](LICENSE). Model weights and evaluation datasets retain their own licenses.
