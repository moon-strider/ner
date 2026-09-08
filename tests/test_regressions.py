"""Regressions found during the public-readiness audit."""

import asyncio

import pytest

from ner_service.cache import MemoryCache, ResultCache
from ner_service.circuit_breaker import CircuitBreaker, State
from ner_service.config_store import PromptTemplateError, prepare_config, render_system_prompt
from ner_service.offsets import attach_offsets, canonicalize_entities
from ner_service.schemas import EntityLabel, ExtractRequest, NERConfig, RawEntity
from ner_service.service import NerService
from tests.test_service import FakeProvider, _config


async def test_cache_varies_with_rendered_prompt():
    provider = FakeProvider()
    service = NerService(provider, cache=ResultCache(MemoryCache()))
    config = _config(system_prompt="context={payload.context}")
    for context in ("one", "two", "two"):
        await service.extract(
            ExtractRequest(text="Tim Cook", config=config, prompt_payload={"context": context})
        )
    assert len(provider.calls) == 2


async def test_cache_does_not_bypass_prompt_validation():
    service = NerService(FakeProvider(), cache=ResultCache(MemoryCache()))
    config = _config(system_prompt="context={payload.context}")
    await service.extract(
        ExtractRequest(text="Tim Cook", config=config, prompt_payload={"context": "one"})
    )
    with pytest.raises(PromptTemplateError):
        await service.extract(ExtractRequest(text="Tim Cook", config=config))


@pytest.mark.parametrize(
    ("text", "entities", "spans"),
    [
        ("Paris is Paris.", [RawEntity(text="Paris", label="X")], [(0, 5)]),
        ("İ Paris", [RawEntity(text="paris", label="X")], [(2, 7)]),
        (
            "New York",
            [RawEntity(text="New York", label="X"), RawEntity(text="New", label="X")],
            [(0, 8)],
        ),
    ],
)
def test_offsets_preserve_source_and_model_occurrence_count(text, entities, spans):
    actual = attach_offsets(text, entities, case_sensitive=False)
    assert [(e.start, e.end) for e in actual] == spans
    assert all(text[e.start : e.end] == e.text for e in actual)


def test_dictionary_drops_hallucinations_and_duplicates():
    raw = [RawEntity(text=value, label="X") for value in ["Paris", "Paris", "London"]]
    assert [e.text for e in canonicalize_entities("Paris", raw)] == ["Paris"]


def test_label_description_braces_are_literal():
    config = NERConfig(labels=[EntityLabel(name="X", description="Objects like {x}")])
    assert "Objects like {x}" in render_system_prompt(prepare_config(config), {})


async def test_half_open_failure_reopens_even_after_old_failures_expire(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("ner_service.circuit_breaker.time.perf_counter", lambda: now[0])
    circuit = CircuitBreaker(failure_threshold=2, recovery_timeout=100)

    async def fail():
        raise RuntimeError("failure")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await circuit.call(fail)
    now[0] = 101
    with pytest.raises(RuntimeError):
        await circuit.call(fail)
    assert circuit.state == State.OPEN


async def test_cancelled_half_open_probe_releases_slot():
    circuit = CircuitBreaker(failure_threshold=1, recovery_timeout=0)

    async def fail():
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError):
        await circuit.call(fail)

    async def cancel():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await circuit.call(cancel)

    async def success():
        return "ok"

    assert await circuit.call(success) == "ok"


async def test_allowed_models_applies_after_runtime_defaults():
    from ner_service.config import RuntimeLimits

    provider = FakeProvider()
    service = NerService(
        provider, default_model="allowed", limits=RuntimeLimits(allowed_models=("allowed",))
    )
    await service.extract(ExtractRequest(text="Tim Cook", config=_config()))
    with pytest.raises(ValueError, match="ALLOWED_MODELS"):
        await service.extract(ExtractRequest(text="Tim Cook", config=_config(model="expensive")))
    assert len(provider.calls) == 1


async def test_rate_limiter_cancellation_during_refill_releases_concurrency_slot():
    from ner_service.rate_limiter import RateLimiter

    now = [0.0]
    entered = asyncio.Event()

    async def sleeper(delay):
        entered.set()
        await asyncio.Event().wait()

    limiter = RateLimiter(
        rate_per_second=1, burst=1, provider_concurrency=1, clock=lambda: now[0], sleeper=sleeper
    )
    await limiter.acquire("p")
    limiter.release("p")
    waiting = asyncio.create_task(limiter.acquire("p"))
    await entered.wait()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    now[0] = 2.0
    await asyncio.wait_for(limiter.acquire("p"), timeout=1)
    limiter.release("p")


def test_cache_expiry_eviction_and_mutation_isolation(monkeypatch):
    from ner_service.cache import MemoryCache

    now = [0.0]
    monkeypatch.setattr("ner_service.cache.time.monotonic", lambda: now[0])
    cache = MemoryCache(max_size=2)
    original = {"entities": [{"text": "Paris"}]}
    cache.set("a", original, ttl=5)
    original["entities"][0]["text"] = "mutated"
    cache.set("b", {}, ttl=10)
    fetched = cache.get("a")
    assert fetched["entities"][0]["text"] == "Paris"
    fetched["entities"][0]["text"] = "also mutated"
    cache.set("c", {}, ttl=10)
    assert cache.get("b") is None
    assert cache.get("a")["entities"][0]["text"] == "Paris"
    now[0] = 5
    assert cache.get("a") is None
