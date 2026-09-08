from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Consecutive-failure breaker; stale in-flight calls cannot change a newer state."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        if failure_threshold < 1 or recovery_timeout < 0 or half_open_max_calls < 1:
            raise ValueError("invalid circuit breaker limits")
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._state = State.CLOSED
        self._failures = 0
        self._half_open_calls = 0
        self._last_failure = 0.0
        self._generation = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> State:
        return self._state

    async def call(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        is_failure: Callable[[Exception], bool] | None = None,
    ) -> T:
        async with self._lock:
            if (
                self._state == State.OPEN
                and time.perf_counter() - self._last_failure >= self._recovery_timeout
            ):
                self._transition(State.HALF_OPEN)
            if self._state == State.OPEN:
                raise CircuitBreakerOpen("circuit breaker is open")
            if self._state == State.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    raise CircuitBreakerOpen("circuit breaker half-open quota exhausted")
                self._half_open_calls += 1
            generation = self._generation

        try:
            result = await operation()
        except asyncio.CancelledError:
            async with self._lock:
                if generation == self._generation and self._state == State.HALF_OPEN:
                    self._half_open_calls -= 1
            raise
        except Exception as exc:
            await self._record(generation, is_failure is None or is_failure(exc))
            raise
        else:
            await self._record(generation, False)
            return result

    def _transition(self, state: State) -> None:
        self._state = state
        self._generation += 1
        self._half_open_calls = 0

    async def _record(self, generation: int, failed: bool) -> None:
        async with self._lock:
            if generation != self._generation:
                return
            if failed:
                self._failures += 1
                self._last_failure = time.perf_counter()
                if self._state == State.HALF_OPEN or self._failures >= self._failure_threshold:
                    self._transition(State.OPEN)
            elif self._state == State.HALF_OPEN:
                self._half_open_calls -= 1
                if self._half_open_calls == 0:
                    self._failures = 0
                    self._transition(State.CLOSED)
            else:
                self._failures = 0


class CircuitBreakerOpen(Exception):
    pass
