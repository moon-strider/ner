# Contributing

Use Python 3.12 or 3.13 and uv. Install the locked development environment:

```bash
uv sync --frozen --extra dev
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src
uv run --frozen pytest -m 'not integration' --cov=ner_service --cov-report=term-missing
```

Tests use synthetic fixtures and mocked HTTP providers. The live smoke check is
opt-in; see [local inference](docs/local-inference.md). Provider edge cases, Unicode
offsets, cancellation, persistence, cache isolation, HTTP errors, and SDK behavior
are part of the regression suite. Fixes should include a test that exposes the
behavioral bug. Do not add real API keys or private text to fixtures.

## Layout

```text
src/ner_service/        API, extraction pipeline, providers, storage, metrics
tests/                  Unit, contract, SDK, and opt-in integration checks
clients/python/         Generated Python SDK and generator configuration
scripts/                SDK generation, HTTP smoke check, profiling, benchmark
docs/                   API and operational documentation, audit evidence
observability/          Optional Prometheus and Grafana configuration
```

## Python client

The generated package has its own README and can be installed from
`clients/python/ner-client`. Do not hand-edit generated code.

```bash
uv run --frozen python scripts/generate_client.py
uv run --frozen python scripts/generate_client.py --check
```

Generation happens in a temporary directory before replacing the checked-in SDK.
`--check` is read-only and fails on drift. Omitted model/token fields must stay
omitted in SDK requests so server runtime defaults apply. Error response models
must match the actual API envelope. Sync and async SDK calls are tested against
the FastAPI app, with additional live validation recorded in the audit.

The service declares auth as optional in OpenAPI because `NER_API_KEY` is a
runtime deployment choice. Use `AuthenticatedClient` when the server requires it.

## Packaging and dependency changes

```bash
uv build
uv build clients/python/ner-client --out-dir dist/client
```

Runtime dependencies are pinned in `pyproject.toml` and `uv.lock`; benchmarks are
an optional extra. After changing dependencies, run `uv lock`, sync, test, and
run `uvx pip-audit --path .venv/lib/python3.12/site-packages`. The CI workflow is configured to test Python 3.12
and 3.13, checks generated-client drift, builds both packages, scans installed
dependencies, and boots a non-root container to verify SQLite readiness.

The container check does not call a paid model. The observability Compose profile
is configuration-validated in CI; its complete dashboard stack is not an automated
browser test. Keep user-facing docs honest about which scenarios were executed.
