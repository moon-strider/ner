from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import aiosqlite


@dataclass(frozen=True)
class StoredConfig:
    id: str
    data: dict[str, Any]


class ConfigStoreBackend(ABC):
    @abstractmethod
    async def get(self, config_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def set(self, config_id: str, config: dict[str, Any]) -> None: ...

    @abstractmethod
    async def delete(self, config_id: str) -> None: ...

    @abstractmethod
    async def list(self) -> list[StoredConfig]: ...

    @abstractmethod
    async def healthcheck(self) -> dict[str, Any]: ...

    async def aclose(self) -> None:
        return None


class MemoryStore(ConfigStoreBackend):
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    async def get(self, config_id: str) -> dict[str, Any] | None:
        return deepcopy(self._items.get(config_id))

    async def set(self, config_id: str, config: dict[str, Any]) -> None:
        self._items[config_id] = deepcopy(config)

    async def delete(self, config_id: str) -> None:
        self._items.pop(config_id, None)

    async def list(self) -> list[StoredConfig]:
        return [StoredConfig(id=key, data=deepcopy(value)) for key, value in self._items.items()]

    async def healthcheck(self) -> dict[str, Any]:
        return {"backend": "memory", "status": "ok"}


class SQLiteStore(ConfigStoreBackend):
    """One serialized connection per store, including for SQLite's :memory: databases."""

    def __init__(self, path: str = "configs.db") -> None:
        self._path = path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._lock:
            if self._db is None:
                db = await aiosqlite.connect(self._path, timeout=5)
                try:
                    await db.execute("PRAGMA journal_mode=WAL")
                    await db.execute(
                        "CREATE TABLE IF NOT EXISTS configs "
                        "(id TEXT PRIMARY KEY, data TEXT NOT NULL)"
                    )
                    await db.commit()
                except BaseException:
                    await db.close()
                    raise
                self._db = db
            try:
                yield self._db
            except BaseException:
                await self._db.rollback()
                raise

    async def get(self, config_id: str) -> dict[str, Any] | None:
        async with (
            self._connection() as db,
            db.execute("SELECT data FROM configs WHERE id = ?", (config_id,)) as cursor,
        ):
            row = await cursor.fetchone()
        return cast(dict[str, Any], json.loads(row[0])) if row is not None else None

    async def set(self, config_id: str, config: dict[str, Any]) -> None:
        payload = json.dumps(config, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        async with self._connection() as db:
            await db.execute(
                "INSERT INTO configs (id, data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                (config_id, payload),
            )
            await db.commit()

    async def delete(self, config_id: str) -> None:
        async with self._connection() as db:
            await db.execute("DELETE FROM configs WHERE id = ?", (config_id,))
            await db.commit()

    async def list(self) -> list[StoredConfig]:
        async with (
            self._connection() as db,
            db.execute("SELECT id, data FROM configs ORDER BY rowid ASC") as cursor,
        ):
            rows = await cursor.fetchall()
        return [StoredConfig(id=row[0], data=json.loads(row[1])) for row in rows]

    async def healthcheck(self) -> dict[str, Any]:
        async with self._connection() as db, db.execute("SELECT id FROM configs LIMIT 1") as cursor:
            await cursor.fetchone()
        return {"backend": "sqlite", "status": "ok"}

    async def aclose(self) -> None:
        async with self._lock:
            if self._db is not None:
                await self._db.close()
                self._db = None
