from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConfigStoreBackend(ABC):
    @abstractmethod
    async def get(self, config_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def set(self, config_id: str, config: dict[str, Any]) -> None: ...

    @abstractmethod
    async def delete(self, config_id: str) -> None: ...

    @abstractmethod
    async def list(self) -> list[dict[str, Any]]: ...


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
                "CREATE TABLE IF NOT EXISTS configs (id TEXT PRIMARY KEY, data TEXT)"
            )
            await db.commit()
        self._ready = True

    async def get(self, config_id: str) -> dict[str, Any] | None:
        await self._init()
        import aiosqlite
        async with aiosqlite.connect(self._path) as db, db.execute(
            "SELECT data FROM configs WHERE id=?", (config_id,)
        ) as cursor:
                row = await cursor.fetchone()
                if row:
                    import json
                    return json.loads(row[0])
                return None

    async def set(self, config_id: str, config: dict[str, Any]) -> None:
        await self._init()
        import json

        import aiosqlite
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO configs (id, data) VALUES (?, ?)",
                (config_id, json.dumps(config)),
            )
            await db.commit()

    async def delete(self, config_id: str) -> None:
        await self._init()
        import aiosqlite
        async with aiosqlite.connect(self._path) as db:
            await db.execute("DELETE FROM configs WHERE id=?", (config_id,))
            await db.commit()

    async def list(self) -> list[dict[str, Any]]:
        await self._init()
        import json

        import aiosqlite
        async with aiosqlite.connect(self._path) as db, db.execute(
            "SELECT data FROM configs"
        ) as cursor:
                rows = await cursor.fetchall()
                return [json.loads(r[0]) for r in rows]
