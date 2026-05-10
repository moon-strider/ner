from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram


_latency = Histogram(
    "ner_extraction_latency_ms",
    "NER extraction latency in milliseconds",
    ["provider", "model"],
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)

_total = Counter(
    "ner_extraction_total",
    "Total NER extractions",
    ["provider", "model", "status"],
)

_errors = Counter(
    "provider_errors_total",
    "Provider errors by type",
    ["provider", "error_type"],
)

_tokens = Counter(
    "ner_tokens_total",
    "Total tokens consumed",
    ["provider", "model", "token_type"],
)


class MetricsCollector:
    def record_attempt(self, provider: str, model: str, duration_ms: float, success: bool) -> None:
        status = "success" if success else "error"
        _total.labels(provider=provider, model=model, status=status).inc()
        _latency.labels(provider=provider, model=model).observe(duration_ms)

    def record_error(self, provider: str, error_type: str) -> None:
        _errors.labels(provider=provider, error_type=error_type).inc()

    def record_tokens(self, provider: str, model: str, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        if prompt := usage.get("prompt_tokens"):
            _tokens.labels(provider=provider, model=model, token_type="prompt").inc(prompt)
        if completion := usage.get("completion_tokens"):
            _tokens.labels(provider=provider, model=model, token_type="completion").inc(completion)


@contextmanager
def extraction_timer() -> Generator[list[float], None, None]:
    import time
    started = time.perf_counter()
    duration_ms = [0.0]
    try:
        yield duration_ms
    finally:
        duration_ms[0] = (time.perf_counter() - started) * 1000


def setup_metrics(app: Any) -> None:
    Instrumentator().instrument(app).expose(app)
