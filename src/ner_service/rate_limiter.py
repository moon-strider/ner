from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class TokenBucket:
    def __init__(
        self,
        rate_per_second: float,
        burst: int,
        *,
        clock: Clock = time.perf_counter,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")
        self._rate_per_second = rate_per_second
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._clock = clock
        self._sleeper = sleeper
        self._last_refill = clock()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            raise ValueError("tokens must be > 0")
        if tokens > self._capacity:
            raise ValueError("tokens must be <= burst")
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_time = deficit / self._rate_per_second
            await self._sleeper(wait_time)

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
        self._last_refill = now


class RateLimiter:
    def __init__(
        self,
        *,
        rate_per_second: float = 100.0,
        burst: int = 200,
        provider_concurrency: int = 50,
        clock: Clock = time.perf_counter,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        if provider_concurrency <= 0:
            raise ValueError("provider_concurrency must be > 0")
        self._global = TokenBucket(
            rate_per_second=rate_per_second,
            burst=burst,
            clock=clock,
            sleeper=sleeper,
        )
        self._provider_concurrency = provider_concurrency
        self._provider_sem: dict[str, asyncio.Semaphore] = {}

    async def acquire(self, provider: str) -> None:
        await self._global.acquire()
        semaphore = self._provider_sem.setdefault(
            provider,
            asyncio.Semaphore(self._provider_concurrency),
        )
        await semaphore.acquire()

    def release(self, provider: str) -> None:
        semaphore = self._provider_sem.get(provider)
        if semaphore is not None:
            semaphore.release()
