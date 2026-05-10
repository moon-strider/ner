from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast


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


class MemoryStore(ConfigStoreBackend):
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    async def get(self, config_id: str) -> dict[str, Any] | None:
        item = self._items.get(config_id)
        if item is None:
            return None
        return dict(item)

    async def set(self, config_id: str, config: dict[str, Any]) -> None:
        self._items[config_id] = dict(config)

    async def delete(self, config_id: str) -> None:
        self._items.pop(config_id, None)

    async def list(self) -> list[StoredConfig]:
        return [
            StoredConfig(id=config_id, data=dict(data)) for config_id, data in self._items.items()
        ]

    async def healthcheck(self) -> dict[str, Any]:
        return {"backend": "memory", "status": "ok"}


class SQLiteStore(ConfigStoreBackend):
    def __init__(self, path: str = "configs.db") -> None:
        self._path = path
        self._ready = False

    async def _init(self) -> None:
        if self._ready:
            return
        import aiosqlite

        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS configs (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            await db.commit()
        self._ready = True

    async def get(self, config_id: str) -> dict[str, Any] | None:
        await self._init()
        import aiosqlite

        async with (
            aiosqlite.connect(self._path) as db,
            db.execute("SELECT data FROM configs WHERE id = ?", (config_id,)) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return cast(dict[str, Any], json.loads(row[0]))

    async def set(self, config_id: str, config: dict[str, Any]) -> None:
        await self._init()
        import aiosqlite

        payload = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO configs (id, data) VALUES (?, ?)",
                (config_id, payload),
            )
            await db.commit()

    async def delete(self, config_id: str) -> None:
        await self._init()
        import aiosqlite

        async with aiosqlite.connect(self._path) as db:
            await db.execute("DELETE FROM configs WHERE id = ?", (config_id,))
            await db.commit()

    async def list(self) -> list[StoredConfig]:
        await self._init()
        import aiosqlite

        async with (
            aiosqlite.connect(self._path) as db,
            db.execute("SELECT id, data FROM configs ORDER BY rowid ASC") as cursor,
        ):
            rows = await cursor.fetchall()
        return [
            StoredConfig(id=cast(str, row[0]), data=cast(dict[str, Any], json.loads(row[1])))
            for row in rows
        ]

    async def healthcheck(self) -> dict[str, Any]:
        await self._init()
        import aiosqlite

        async with (
            aiosqlite.connect(self._path) as db,
            db.execute("SELECT 1") as cursor,
        ):
            row = await cursor.fetchone()
        if row is None or row[0] != 1:
            raise RuntimeError("sqlite healthcheck failed")
        return {"backend": "sqlite", "status": "ok", "path": self._path}
