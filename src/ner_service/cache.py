from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def set(self, key: str, value: dict[str, Any], ttl: int) -> None: ...


class MemoryCache(CacheBackend):
    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._data: dict[str, dict[str, Any]] = {}
        self._expires: dict[str, float] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        if key in self._expires and self._expires[key] < now:
            self._delete(key)
            return None
        return self._data.get(key)

    def set(self, key: str, value: dict[str, Any], ttl: int) -> None:
        if len(self._data) >= self._max_size and key not in self._data:
            oldest = min(self._expires, key=lambda k: self._expires[k])
            self._delete(oldest)
        self._data[key] = value
        self._expires[key] = time.time() + ttl

    def _delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._expires.pop(key, None)


def _make_key(text: str, config_key: str) -> str:
    raw = json.dumps([text, config_key], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class ResultCache:
    def __init__(self, backend: CacheBackend | None = None, ttl: int = 600) -> None:
        self._backend = backend
        self._ttl = ttl

    def get(self, text: str, config_key: str) -> dict[str, Any] | None:
        if self._backend is None:
            return None
        return self._backend.get(_make_key(text, config_key))

    def set(self, text: str, config_key: str, result: dict[str, Any]) -> None:
        if self._backend is None:
            return
        self._backend.set(_make_key(text, config_key), result, self._ttl)
