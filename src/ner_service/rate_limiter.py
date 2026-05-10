from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any


class TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> bool:
        async with self._lock:
            now = time.perf_counter()
            elapsed = now - self._last
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait(self, tokens: float = 1.0) -> None:
        while not await self.acquire(tokens):
            deficit = tokens - self._tokens
            wait_time = deficit / self._rate
            await asyncio.sleep(wait_time)


class RateLimiter:
    def __init__(
        self,
        *,
        global_rps: float = 100.0,
        provider_concurrency: int = 50,
    ) -> None:
        self._global = TokenBucket(rate=global_rps, capacity=global_rps * 2)
        self._provider_sem: dict[str, asyncio.Semaphore] = {}
        self._provider_concurrency = provider_concurrency

    async def acquire(self, provider: str) -> None:
        if not await self._global.acquire():
            raise RateLimitExceeded("global rate limit exceeded")
        sem = self._provider_sem.setdefault(
            provider, asyncio.Semaphore(self._provider_concurrency)
        )
        acquired = await sem.acquire()
        if not acquired:
            raise RateLimitExceeded("provider concurrency limit exceeded")

    def release(self, provider: str) -> None:
        sem = self._provider_sem.get(provider)
        if sem is not None:
            sem.release()


class RateLimitExceeded(Exception):
    pass
