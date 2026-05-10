from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = State.CLOSED
        self._failures: deque[float] = deque()
        self._half_open_calls = 0
        self._last_failure: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> State:
        return self._state

    async def call(self, coro: Any) -> Any:
        async with self._lock:
            await self._update_state()
            if self._state == State.OPEN:
                raise CircuitBreakerOpen(f"circuit breaker is open; last failure at {self._last_failure}")
            if self._state == State.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    raise CircuitBreakerOpen("circuit breaker half-open quota exhausted")
                self._half_open_calls += 1

        try:
            result = await coro
        except Exception as e:
            await self._record_failure()
            raise
        else:
            await self._record_success()
            return result

    async def _update_state(self) -> None:
        if self._state == State.OPEN:
            if self._last_failure is not None and (time.perf_counter() - self._last_failure >= self._recovery_timeout):
                self._state = State.HALF_OPEN
                self._half_open_calls = 0

    async def _record_failure(self) -> None:
        async with self._lock:
            now = time.perf_counter()
            self._failures.append(now)
            self._last_failure = now
            while self._failures and now - self._failures[0] > 60:
                self._failures.popleft()
            if len(self._failures) >= self._failure_threshold:
                self._state = State.OPEN

    async def _record_success(self) -> None:
        async with self._lock:
            if self._state == State.HALF_OPEN:
                self._state = State.CLOSED
                self._failures.clear()
                self._half_open_calls = 0


class CircuitBreakerOpen(Exception):
    pass
