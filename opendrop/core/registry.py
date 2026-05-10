"""SQLite model registry for OpenDrop.

Stores metadata for all models and LoRA adapters managed by the system.
Uses aiosqlite for async access from the server and sync wrappers for CLI.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS models (
    id              TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    architecture    TEXT NOT NULL DEFAULT '',
    params_b        REAL NOT NULL DEFAULT 0,
    quant           TEXT NOT NULL DEFAULT '',
    format          TEXT NOT NULL DEFAULT 'gguf',
    path            TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    license_id      TEXT NOT NULL DEFAULT '',
    license_warning TEXT NOT NULL DEFAULT '',
    tags            TEXT NOT NULL DEFAULT '[]',
    pipeline_tag    TEXT NOT NULL DEFAULT '',
    added_at        TEXT NOT NULL,
    last_used       TEXT,
    server_port     INTEGER,
    extra           TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS adapters (
    id          TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    method      TEXT NOT NULL DEFAULT 'lora',
    path        TEXT NOT NULL,
    dataset_url TEXT NOT NULL DEFAULT '',
    added_at    TEXT NOT NULL,
    extra       TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_adapters_model ON adapters(model_id);
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelRecord:
    id: str
    model_id: str
    source_url: str
    display_name: str
    architecture: str
    params_b: float
    quant: str
    format: str
    path: str
    size_bytes: int
    license_id: str
    license_warning: str
    tags: list
    pipeline_tag: str
    added_at: str
    last_used: Optional[str]
    server_port: Optional[int]
    extra: dict

    def path_obj(self) -> Path:
        return Path(self.path)

    def size_human(self) -> str:
        gb = self.size_bytes / (1024 ** 3)
        return f"{gb:.2f} GB" if gb >= 1 else f"{self.size_bytes / (1024 ** 2):.0f} MB"


@dataclass
class AdapterRecord:
    id: str
    model_id: str
    name: str
    method: str
    path: str
    dataset_url: str
    added_at: str
    extra: dict


# ---------------------------------------------------------------------------
# Sync helpers
# ---------------------------------------------------------------------------

def _row_to_model(row: sqlite3.Row) -> ModelRecord:
    d = dict(row)
    d["tags"] = json.loads(d["tags"])
    d["extra"] = json.loads(d["extra"])
    return ModelRecord(**d)


def _row_to_adapter(row: sqlite3.Row) -> AdapterRecord:
    d = dict(row)
    d["extra"] = json.loads(d["extra"])
    return AdapterRecord(**d)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Sync Registry (used from CLI)
# ---------------------------------------------------------------------------

class Registry:
    """Synchronous SQLite model registry."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- Models ---

    def add_model(self, record: ModelRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO models
                (id, model_id, source_url, display_name, architecture, params_b,
                 quant, format, path, size_bytes, license_id, license_warning,
                 tags, pipeline_tag, added_at, last_used, server_port, extra)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.id, record.model_id, record.source_url,
                    record.display_name, record.architecture, record.params_b,
                    record.quant, record.format, record.path, record.size_bytes,
                    record.license_id, record.license_warning,
                    json.dumps(record.tags), record.pipeline_tag,
                    record.added_at, record.last_used, record.server_port,
                    json.dumps(record.extra),
                ),
            )

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """Lookup by short ID or display name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE id=? OR display_name=?",
                (model_id, model_id),
            ).fetchone()
            return _row_to_model(row) if row else None

    def list_models(self) -> list[ModelRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM models ORDER BY added_at DESC"
            ).fetchall()
            return [_row_to_model(r) for r in rows]

    def remove_model(self, model_id: str) -> bool:
        rec = self.get_model(model_id)
        if not rec:
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM models WHERE id=?", (rec.id,))
        return True

    def touch_model(self, model_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE models SET last_used=? WHERE id=?", (_now(), model_id)
            )

    def set_port(self, model_id: str, port: Optional[int]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE models SET server_port=? WHERE id=?", (port, model_id)
            )

    # --- Adapters ---

    def add_adapter(self, record: AdapterRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO adapters
                (id, model_id, name, method, path, dataset_url, added_at, extra)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    record.id, record.model_id, record.name, record.method,
                    record.path, record.dataset_url, record.added_at,
                    json.dumps(record.extra),
                ),
            )

    def list_adapters(self, model_id: Optional[str] = None) -> list[AdapterRecord]:
        with self._connect() as conn:
            if model_id:
                rows = conn.execute(
                    "SELECT * FROM adapters WHERE model_id=? ORDER BY added_at DESC",
                    (model_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM adapters ORDER BY added_at DESC"
                ).fetchall()
            return [_row_to_adapter(r) for r in rows]

    def remove_adapter(self, adapter_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM adapters WHERE id=?", (adapter_id,))
            return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Async Registry (used from FastAPI server)
# ---------------------------------------------------------------------------

class AsyncRegistry:
    """Asynchronous SQLite model registry using aiosqlite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def init(self) -> None:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_DDL)
            await db.commit()

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator:
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def get_model(self, model_id: str) -> Optional[ModelRecord]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM models WHERE id=? OR display_name=?",
                (model_id, model_id),
            ) as cur:
                row = await cur.fetchone()
                return _row_to_model(dict(row)) if row else None  # type: ignore[arg-type]

    async def list_models(self) -> list[ModelRecord]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM models ORDER BY added_at DESC"
            ) as cur:
                rows = await cur.fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    d["tags"] = json.loads(d["tags"])
                    d["extra"] = json.loads(d["extra"])
                    result.append(ModelRecord(**d))
                return result

    async def touch_model(self, model_id: str) -> None:
        async with self._conn() as db:
            await db.execute(
                "UPDATE models SET last_used=? WHERE id=?", (_now(), model_id)
            )
            await db.commit()
