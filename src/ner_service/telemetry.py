from __future__ import annotations

from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(app: Any, service_name: str = "ner-service") -> None:
    if getattr(app.state, "tracing_initialized", False):
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = app.state.settings.otel_endpoint if hasattr(app.state, "settings") else None
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    app.state.tracer_provider = provider
    app.state.tracing_initialized = True
