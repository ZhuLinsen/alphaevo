"""SQLite-backed persistence for validated factors.

Stores factor metadata, code, validation metrics, and usage statistics.
Follows the same pattern as ExperienceStore / StrategyStore.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FactorRecord(BaseModel):
    """A persisted factor with its validation results."""

    name: str
    description: str
    rationale: str
    code: str
    expected_direction: Literal["positive", "negative"] = "positive"
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    monthly_win_rate: float = 0.0
    turnover: float = 0.0
    status: Literal["active", "retired", "failed"] = "active"
    usage_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


_FACTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    code TEXT NOT NULL,
    expected_direction TEXT NOT NULL DEFAULT 'positive',
    ic_mean REAL DEFAULT 0.0,
    ic_std REAL DEFAULT 0.0,
    ir REAL DEFAULT 0.0,
    monthly_win_rate REAL DEFAULT 0.0,
    turnover REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'active',
    usage_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_factor_status ON factors(status);
CREATE INDEX IF NOT EXISTS idx_factor_ir ON factors(ir);
"""


class FactorStore:
    """SQLite store for factor persistence and retrieval.

    Example::

        store = FactorStore("~/.alphaevo/alphaevo.db")
        store.save(factor_record)
        top = store.top_factors(limit=5)
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = _normalize_db_path(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(self._db_path, timeout=30.0, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA busy_timeout=5000")
                if self._db_path != ":memory:":
                    self._conn.execute("PRAGMA journal_mode=WAL")
            return self._conn

    def _ensure_table(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_FACTOR_SCHEMA)
            conn.commit()

    def save(self, record: FactorRecord) -> None:
        """Insert or update a factor record."""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO factors (
                    name, description, rationale, code, expected_direction,
                    ic_mean, ic_std, ir, monthly_win_rate, turnover,
                    status, usage_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    rationale = excluded.rationale,
                    code = excluded.code,
                    expected_direction = excluded.expected_direction,
                    ic_mean = excluded.ic_mean,
                    ic_std = excluded.ic_std,
                    ir = excluded.ir,
                    monthly_win_rate = excluded.monthly_win_rate,
                    turnover = excluded.turnover,
                    status = excluded.status,
                    usage_count = excluded.usage_count,
                    updated_at = ?
                """,
                (
                    record.name,
                    record.description,
                    record.rationale,
                    record.code,
                    record.expected_direction,
                    record.ic_mean,
                    record.ic_std,
                    record.ir,
                    record.monthly_win_rate,
                    record.turnover,
                    record.status,
                    record.usage_count,
                    record.created_at.isoformat(),
                    now,
                    now,  # updated_at in ON CONFLICT
                ),
            )
            conn.commit()

    def save_if_absent(self, record: FactorRecord) -> tuple[FactorRecord, bool]:
        """Atomically insert ``record`` only if ``name`` does not already exist.

        Returns ``(stored_record, inserted)`` so concurrent writers cannot
        silently overwrite or reactivate an existing/retired factor.
        """
        with self._lock:
            conn = self._get_conn()
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                INSERT INTO factors (
                    name, description, rationale, code, expected_direction,
                    ic_mean, ic_std, ir, monthly_win_rate, turnover,
                    status, usage_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (
                    record.name,
                    record.description,
                    record.rationale,
                    record.code,
                    record.expected_direction,
                    record.ic_mean,
                    record.ic_std,
                    record.ir,
                    record.monthly_win_rate,
                    record.turnover,
                    record.status,
                    record.usage_count,
                    record.created_at.isoformat(),
                    now,
                ),
            )
            conn.commit()
            inserted = cursor.rowcount == 1
            row = conn.execute("SELECT * FROM factors WHERE name = ?", (record.name,)).fetchone()
            if row is None:
                return record, inserted
            return self._row_to_record(row), inserted

    def get(self, name: str) -> FactorRecord | None:
        """Retrieve a factor by name."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM factors WHERE name = ?", (name,)).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list_all(
        self,
        *,
        status: str | None = None,
    ) -> list[FactorRecord]:
        """List all factors, optionally filtered by status."""
        with self._lock:
            conn = self._get_conn()
            if status:
                rows = conn.execute(
                    "SELECT * FROM factors WHERE status = ? ORDER BY ir DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM factors ORDER BY ir DESC").fetchall()
            return [self._row_to_record(r) for r in rows]

    def top_factors(
        self,
        *,
        limit: int = 10,
        status: str = "active",
    ) -> list[FactorRecord]:
        """Get top factors ranked by IR."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM factors WHERE status = ? ORDER BY ir DESC LIMIT ?",
                (status, limit),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def increment_usage(self, name: str) -> None:
        """Increment usage count for a factor."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE factors SET usage_count = usage_count + 1, updated_at = ? WHERE name = ?",
                (datetime.now(timezone.utc).isoformat(), name),
            )
            conn.commit()

    def retire(self, name: str) -> None:
        """Mark a factor as retired."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE factors SET status = 'retired', updated_at = ? WHERE name = ?",
                (datetime.now(timezone.utc).isoformat(), name),
            )
            conn.commit()

    def delete(self, name: str) -> bool:
        """Delete a factor. Returns True if it existed."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM factors WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0

    def count(self, *, status: str | None = None) -> int:
        """Count factors, optionally by status."""
        with self._lock:
            conn = self._get_conn()
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) FROM factors WHERE status = ?", (status,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM factors").fetchone()
            return cast("int", row[0])

    def close(self) -> None:
        """Close the backing SQLite connection if open."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> FactorStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FactorRecord:
        return FactorRecord(
            name=row["name"],
            description=row["description"],
            rationale=row["rationale"],
            code=row["code"],
            expected_direction=row["expected_direction"],
            ic_mean=row["ic_mean"],
            ic_std=row["ic_std"],
            ir=row["ir"],
            monthly_win_rate=row["monthly_win_rate"],
            turnover=row["turnover"],
            status=row["status"],
            usage_count=row["usage_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


def _normalize_db_path(db_path: str | Path) -> str:
    """Return a writable sqlite path or fall back to in-memory storage."""
    if str(db_path) == ":memory:":
        return ":memory:"

    path = Path(db_path).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    probe = resolved.parent / f".{resolved.stem}.write_probe"
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return str(resolved)
    except OSError:
        logger.warning(
            "Factor DB path %s is not writable; falling back to in-memory store",
            resolved,
        )
        return ":memory:"
