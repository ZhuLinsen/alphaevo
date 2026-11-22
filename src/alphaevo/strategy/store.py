"""SQLite-backed strategy store — CRUD and version tracking for strategies.

Manages persistence of Strategy objects and their EvaluationReport results.
Uses the StrategySerializer/StrategyParser for YAML ↔ Strategy conversion.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from alphaevo.models.execution import EvaluationReport
from alphaevo.models.strategy import Strategy
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.dsl.serializer import StrategySerializer

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    parent_id TEXT,
    family_id TEXT,
    category TEXT,
    market TEXT,
    yaml_content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    batch_id TEXT,
    confidence_score REAL DEFAULT 0.0,
    win_rate REAL DEFAULT 0.0,
    avg_return REAL DEFAULT 0.0,
    profit_loss_ratio REAL DEFAULT 0.0,
    max_drawdown REAL DEFAULT 0.0,
    sharpe_ratio REAL DEFAULT 0.0,
    signal_count INTEGER DEFAULT 0,
    json_content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE INDEX IF NOT EXISTS idx_strategies_family ON strategies(family_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_strategy ON evaluations(strategy_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_score ON evaluations(confidence_score DESC);
"""


class StrategyStore:
    """SQLite-backed store for strategies and evaluation results."""

    def __init__(self, db_path: str | Path = "~/.alphaevo/alphaevo.db") -> None:
        """Initialize store, create tables if needed.

        Args:
            db_path: Path to the SQLite database file.
                     Use `:memory:` for an in-memory database (testing).
        """
        normalized_path, use_memory = _normalize_db_path(db_path)
        self._is_memory = use_memory
        self._lock = threading.RLock()
        if self._is_memory:
            self._db_path = ":memory:"
            # Keep a persistent connection for in-memory databases,
            # since each sqlite3.connect(":memory:") creates a *new* DB.
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._shared_conn.execute("PRAGMA foreign_keys=ON")
        else:
            self._db_path = normalized_path
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._shared_conn = None

        self._parser = StrategyParser()
        self._serializer = StrategySerializer()
        self.init_db()

    def init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ── Strategy CRUD ─────────────────────────────────────────────────

    def save(self, strategy: Strategy) -> None:
        """Save or update a strategy (upsert by id)."""
        yaml_content = self._serializer.to_yaml(strategy)
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO strategies
                    (id, name, version, parent_id, family_id, category, market,
                     yaml_content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    version = excluded.version,
                    parent_id = excluded.parent_id,
                    family_id = excluded.family_id,
                    category = excluded.category,
                    market = excluded.market,
                    yaml_content = excluded.yaml_content,
                    updated_at = excluded.updated_at
                """,
                (
                    strategy.meta.id,
                    strategy.meta.name,
                    strategy.meta.version,
                    strategy.meta.parent_id,
                    strategy.meta.family_id,
                    strategy.meta.category.value
                    if hasattr(strategy.meta.category, "value")
                    else str(strategy.meta.category),
                    strategy.meta.market.value
                    if hasattr(strategy.meta.market, "value")
                    else str(strategy.meta.market),
                    yaml_content,
                    strategy.meta.created_at.isoformat()
                    if isinstance(strategy.meta.created_at, datetime)
                    else str(strategy.meta.created_at),
                    now,
                ),
            )

    def get(self, strategy_id: str) -> Strategy | None:
        """Fetch a strategy by id. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT yaml_content FROM strategies WHERE id = ?",
                (strategy_id,),
            ).fetchone()

        if row is None:
            return None
        return self._parser.parse_yaml(row[0])

    def list_all(self) -> list[Strategy]:
        """List all strategies, ordered by created_at descending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT yaml_content FROM strategies ORDER BY created_at DESC"
            ).fetchall()

        return [self._parser.parse_yaml(row[0]) for row in rows]

    def list_by_family(self, family_id: str) -> list[Strategy]:
        """List all versions of a strategy family, ordered by version."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT yaml_content FROM strategies WHERE family_id = ? ORDER BY version ASC",
                (family_id,),
            ).fetchall()

        return [self._parser.parse_yaml(row[0]) for row in rows]

    def delete(self, strategy_id: str) -> bool:
        """Delete a strategy by id. Returns True if a row was deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM strategies WHERE id = ?",
                (strategy_id,),
            )
        return cursor.rowcount > 0

    # ── Evaluation persistence ────────────────────────────────────────

    def save_evaluation(self, report: EvaluationReport) -> None:
        """Save an evaluation report."""
        json_content = report.model_dump_json()
        eval_id = report.evaluation_id or f"{report.strategy_id}_{report.created_at.isoformat()}"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluations
                    (id, strategy_id, batch_id, confidence_score,
                     win_rate, avg_return, profit_loss_ratio,
                     max_drawdown, sharpe_ratio, signal_count,
                     json_content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    confidence_score = excluded.confidence_score,
                    json_content = excluded.json_content
                """,
                (
                    eval_id,
                    report.strategy_id,
                    report.batch_id,
                    report.confidence_score,
                    report.overall.win_rate,
                    report.overall.avg_return,
                    report.overall.profit_loss_ratio,
                    report.overall.max_drawdown,
                    report.overall.sharpe_ratio,
                    report.overall.signal_count,
                    json_content,
                    report.created_at.isoformat(),
                ),
            )

    def get_evaluations(self, strategy_id: str) -> list[EvaluationReport]:
        """Get all evaluations for a strategy, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT json_content FROM evaluations "
                "WHERE strategy_id = ? ORDER BY created_at DESC",
                (strategy_id,),
            ).fetchall()

        return [EvaluationReport.model_validate_json(row[0]) for row in rows]

    def get_leaderboard(
        self,
        limit: int = 20,
        min_signal_count: int = 30,
    ) -> list[tuple[Strategy, EvaluationReport]]:
        """Get top strategies ranked by best confidence_score.

        For each strategy, picks the evaluation with the highest score.
        Returns (Strategy, EvaluationReport) pairs sorted by score descending.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.yaml_content, e.json_content
                FROM evaluations e
                JOIN strategies s ON e.strategy_id = s.id
                WHERE e.signal_count >= ?
                  AND e.id IN (
                    SELECT e2.id
                    FROM evaluations e2
                    WHERE e2.strategy_id = e.strategy_id
                      AND e2.signal_count >= ?
                    ORDER BY e2.confidence_score DESC
                    LIMIT 1
                )
                ORDER BY e.confidence_score DESC
                LIMIT ?
                """,
                (min_signal_count, min_signal_count, limit),
            ).fetchall()

        results: list[tuple[Strategy, EvaluationReport]] = []
        for yaml_content, json_content in rows:
            strategy = self._parser.parse_yaml(yaml_content)
            report = EvaluationReport.model_validate_json(json_content)
            results.append((strategy, report))
        return results

    # ── Import helpers ────────────────────────────────────────────────

    def import_from_file(self, path: Path) -> Strategy:
        """Import a strategy from a YAML file and save to store.

        Returns the parsed Strategy.
        """
        strategy = self._parser.parse_file(Path(path))
        self.save(strategy)
        return strategy

    def import_builtin_strategies(self, strategies_dir: Path) -> int:
        """Import all .yaml files from a directory.

        Skips files that fail to parse. Returns the count of
        successfully imported strategies.
        """
        imported = 0
        strategies_dir = Path(strategies_dir)
        if not strategies_dir.is_dir():
            return 0

        for yaml_path in sorted(strategies_dir.glob("*.yaml")):
            try:
                self.import_from_file(yaml_path)
                imported += 1
            except Exception:
                continue
        return imported

    # ── Internal helpers ──────────────────────────────────────────────

    def close(self) -> None:
        """Close persistent in-memory SQLite connection if open."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._lock = lock
        with lock:
            conn = getattr(self, "_shared_conn", None)
            if conn is not None:
                conn.close()
                self._shared_conn = None

    def __enter__(self) -> StrategyStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a thread-safe SQLite connection and close file-backed handles."""
        shared_conn = getattr(self, "_shared_conn", None)
        if shared_conn is not None:
            lock = getattr(self, "_lock", None)
            if lock is None:
                lock = threading.RLock()
                self._lock = lock
            with lock:
                if getattr(self, "_shared_conn", None) is None:
                    raise RuntimeError("StrategyStore connection is closed")
                with shared_conn:
                    yield shared_conn
            return
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:
                yield conn
        finally:
            conn.close()


def _normalize_db_path(db_path: str | Path) -> tuple[str, bool]:
    """Return a writable sqlite path or fall back to in-memory storage."""
    if str(db_path) == ":memory:":
        return ":memory:", True

    try:
        resolved = Path(db_path).expanduser().resolve()
    except OSError:
        resolved = Path(db_path).expanduser()

    probe = resolved.parent / f".{resolved.stem}.write_probe"
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return str(resolved), False
    except OSError:
        logger.warning(
            "SQLite path %s is not writable; falling back to in-memory store",
            resolved,
        )
        return ":memory:", True
