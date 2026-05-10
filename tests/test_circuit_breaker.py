from __future__ import annotations

import asyncio

import pytest

from ner_service.circuit_breaker import CircuitBreaker, CircuitBreakerOpen


@pytest.mark.asyncio
async def test_circuit_opens_after_failures() -> None:
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=9999.0)

    async def fail() -> str:
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(fail)

    with pytest.raises(CircuitBreakerOpen):
        await cb.call(fail)


@pytest.mark.asyncio
async def test_circuit_recovers_after_timeout() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

    async def fail() -> str:
        raise RuntimeError("boom")

    async def ok() -> str:
        return "success"

    with pytest.raises(RuntimeError):
        await cb.call(fail)

    await asyncio.sleep(0.1)

    assert await cb.call(ok) == "success"
    assert await cb.call(ok) == "success"


@pytest.mark.asyncio
async def test_half_open_quota() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, half_open_max_calls=1)

    async def fail() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await cb.call(fail)

    await asyncio.sleep(0.1)

    with pytest.raises(RuntimeError):
        await cb.call(fail)

    with pytest.raises(CircuitBreakerOpen):
        await cb.call(fail)


@pytest.mark.asyncio
async def test_success_resets_consecutive_failures() -> None:
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=9999.0)

    async def fail() -> str:
        raise RuntimeError("boom")

    async def ok() -> str:
        return "success"

    with pytest.raises(RuntimeError):
        await cb.call(fail)

    assert await cb.call(ok) == "success"

    with pytest.raises(RuntimeError):
        await cb.call(fail)

    assert cb.state.value == "closed"
