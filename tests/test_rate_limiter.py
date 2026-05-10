from __future__ import annotations

import asyncio

import pytest

from ner_service.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.asyncio
async def test_rate_limiter_waits_for_next_token() -> None:
    clock = FakeClock()
    limiter = RateLimiter(rate_per_second=1.0, burst=1, clock=clock, sleeper=clock.sleep)

    await limiter.acquire("test")
    limiter.release("test")
    await limiter.acquire("test")
    limiter.release("test")

    assert clock.now == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rate_limiter_blocks_on_provider_concurrency() -> None:
    limiter = RateLimiter(rate_per_second=10.0, burst=10, provider_concurrency=1)

    await limiter.acquire("test")
    waiter = asyncio.create_task(limiter.acquire("test"))
    await asyncio.sleep(0)

    assert not waiter.done()

    limiter.release("test")
    await asyncio.wait_for(waiter, timeout=1)
    limiter.release("test")
