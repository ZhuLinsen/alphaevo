"""Evolution experience store — persists lessons learned from each evolution round.

Records which changes worked vs. failed, indexed by strategy family, change type,
and metric impacted. This lets the LLM reflection see what was already tried and
learn from cross-strategy evolution history.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from alphaevo.models.enums import ChangeType, EvidenceLevel

logger = logging.getLogger(__name__)


class ExperienceRecord(BaseModel):
    """A single lesson learned from an evolution round."""

    strategy_family: str
    strategy_id: str
    round_num: int
    change_type: ChangeType
    target: str
    from_value: Any = None
    to_value: Any = None
    reason: str = ""
    score_before: float = 0.0
    score_after: float = 0.0
    score_delta: float = 0.0
    worked: bool = False
    failure_patterns: list[str] = Field(default_factory=list)
    lesson: str = ""
    # New fields for research agent upgrade
    hypothesis: str = ""  # what hypothesis this change was testing
    action_type: str = ""  # llm | heuristic | param_search | pattern | factor_discovery
    regime: str = ""  # market regime when this change was tested
    source: str = ""  # which component produced this change
    evidence_level: EvidenceLevel = EvidenceLevel.BACKTEST
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperienceQuery(BaseModel):
    """Query parameters for retrieving relevant experience."""

    strategy_family: str | None = None
    change_types: list[ChangeType] | None = None
    only_worked: bool | None = None
    exclude_test_sources: bool = False
    limit: int = 20


_EXPERIENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS evolution_experience (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_family TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    change_type TEXT NOT NULL,
    target TEXT NOT NULL,
    from_value TEXT,
    to_value TEXT,
    reason TEXT DEFAULT '',
    score_before REAL DEFAULT 0.0,
    score_after REAL DEFAULT 0.0,
    score_delta REAL DEFAULT 0.0,
    worked INTEGER DEFAULT 0,
    failure_patterns TEXT DEFAULT '[]',
    lesson TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    hypothesis TEXT DEFAULT '',
    action_type TEXT DEFAULT '',
    regime TEXT DEFAULT '',
    source TEXT DEFAULT '',
    evidence_level TEXT DEFAULT 'backtest'
);

CREATE INDEX IF NOT EXISTS idx_exp_family ON evolution_experience(strategy_family);
CREATE INDEX IF NOT EXISTS idx_exp_worked ON evolution_experience(worked);
CREATE INDEX IF NOT EXISTS idx_exp_change_type ON evolution_experience(change_type);
"""

_EXPERIENCE_MIGRATION = """
ALTER TABLE evolution_experience ADD COLUMN hypothesis TEXT DEFAULT '';
ALTER TABLE evolution_experience ADD COLUMN action_type TEXT DEFAULT '';
ALTER TABLE evolution_experience ADD COLUMN regime TEXT DEFAULT '';
ALTER TABLE evolution_experience ADD COLUMN source TEXT DEFAULT '';
"""


class ExperienceStore:
    """SQLite-backed store for evolution lessons learned.

    Can share the same database as StrategyStore (just adds its own table),
    or use a separate DB.
    """

    def __init__(self, db_path: str | Path = "~/.alphaevo/alphaevo.db") -> None:
        normalized_path, use_memory = _normalize_db_path(db_path)
        self._is_memory = use_memory
        if self._is_memory:
            self._db_path = ":memory:"
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(":memory:")
            self._shared_conn.execute("PRAGMA foreign_keys=ON")
        else:
            self._db_path = normalized_path
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._shared_conn = None

        self.init_db()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_EXPERIENCE_SCHEMA)
            # Migrate existing tables that lack the new columns
            try:
                cols = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(evolution_experience)").fetchall()
                }
                for col in ("hypothesis", "action_type", "regime", "source"):
                    if col not in cols:
                        conn.execute(
                            f"ALTER TABLE evolution_experience ADD COLUMN {col} TEXT DEFAULT ''"
                        )
                if "evidence_level" not in cols:
                    conn.execute(
                        "ALTER TABLE evolution_experience ADD COLUMN evidence_level TEXT DEFAULT 'backtest'"
                    )
            except Exception:
                pass

    def record(self, exp: ExperienceRecord) -> None:
        """Persist a single experience record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_experience
                    (strategy_family, strategy_id, round_num, change_type,
                     target, from_value, to_value, reason,
                     score_before, score_after, score_delta, worked,
                     failure_patterns, lesson, created_at,
                     hypothesis, action_type, regime, source, evidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp.strategy_family,
                    exp.strategy_id,
                    exp.round_num,
                    exp.change_type.value,
                    exp.target,
                    json.dumps(exp.from_value),
                    json.dumps(exp.to_value),
                    exp.reason,
                    exp.score_before,
                    exp.score_after,
                    exp.score_delta,
                    1 if exp.worked else 0,
                    json.dumps(exp.failure_patterns),
                    exp.lesson,
                    exp.created_at.isoformat(),
                    exp.hypothesis,
                    exp.action_type,
                    exp.regime,
                    exp.source,
                    exp.evidence_level.value,
                ),
            )

    def record_batch(self, records: list[ExperienceRecord]) -> None:
        """Persist multiple records in a single transaction."""
        if not records:
            return

        rows = [
            (
                rec.strategy_family,
                rec.strategy_id,
                rec.round_num,
                rec.change_type.value,
                rec.target,
                json.dumps(rec.from_value),
                json.dumps(rec.to_value),
                rec.reason,
                rec.score_before,
                rec.score_after,
                rec.score_delta,
                1 if rec.worked else 0,
                json.dumps(rec.failure_patterns),
                rec.lesson,
                rec.created_at.isoformat(),
                rec.hypothesis,
                rec.action_type,
                rec.regime,
                rec.source,
                rec.evidence_level.value,
            )
            for rec in records
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO evolution_experience
                    (strategy_family, strategy_id, round_num, change_type,
                     target, from_value, to_value, reason,
                     score_before, score_after, score_delta, worked,
                     failure_patterns, lesson, created_at,
                     hypothesis, action_type, regime, source, evidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def query(self, q: ExperienceQuery) -> list[ExperienceRecord]:
        """Retrieve experience records matching the query."""
        clauses: list[str] = []
        params: list[Any] = []

        if q.strategy_family is not None:
            clauses.append("strategy_family = ?")
            params.append(q.strategy_family)

        if q.change_types:
            placeholders = ",".join("?" for _ in q.change_types)
            clauses.append(f"change_type IN ({placeholders})")
            params.extend(ct.value for ct in q.change_types)

        if q.only_worked is not None:
            clauses.append("worked = ?")
            params.append(1 if q.only_worked else 0)

        if q.exclude_test_sources:
            clauses.append(self._exclude_test_clause())

        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(q.limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM evolution_experience WHERE {where} "
                f"ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_family_lessons(
        self,
        family_id: str,
        limit: int = 10,
        *,
        exclude_test_sources: bool = False,
    ) -> list[ExperienceRecord]:
        """Get lessons for a specific strategy family (most recent first)."""
        return self.query(
            ExperienceQuery(
                strategy_family=family_id,
                limit=limit,
                exclude_test_sources=exclude_test_sources,
            )
        )

    def get_failed_signatures(
        self,
        family_id: str,
        *,
        min_failures: int = 2,
        limit: int = 500,
        exclude_test_sources: bool = False,
    ) -> set[tuple[str, str, str]]:
        """Return change signatures that failed repeatedly for one family.

        A single failed attempt may be noisy. This API only returns signatures
        that have failed at least ``min_failures`` times and never succeeded in
        the inspected window.
        """
        records = self.get_family_lessons(
            family_id,
            limit=limit,
            exclude_test_sources=exclude_test_sources,
        )
        stats: dict[tuple[str, str, str], dict[str, int]] = {}

        for rec in records:
            sig = _fuzzy_signature(
                rec.change_type.value,
                rec.target,
                rec.to_value,
            )
            bucket = stats.setdefault(sig, {"ok": 0, "fail": 0})
            if rec.worked:
                bucket["ok"] += 1
            else:
                bucket["fail"] += 1

        return {
            sig
            for sig, count in stats.items()
            if count["fail"] >= min_failures and count["ok"] == 0
        }

    def get_success_rate_by_change_type(
        self,
        family_id: str | None = None,
        *,
        exclude_test_sources: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Return per-change-type success rates.

        Returns ``{change_type_value: {"rate": float, "total": int, "succeeded": int}}``.
        """
        clause = ""
        params: list[Any] = []
        if family_id is not None:
            clause = "WHERE strategy_family = ?"
            params.append(family_id)
        if exclude_test_sources:
            test_clause = self._exclude_test_clause()
            clause = f"{clause} AND {test_clause}" if clause else f"WHERE {test_clause}"

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT change_type, worked, COUNT(*) "
                f"FROM evolution_experience {clause} "
                f"GROUP BY change_type, worked",
                params,
            ).fetchall()

        stats: dict[str, dict[str, int]] = {}
        for change_type, worked, count in rows:
            bucket = stats.setdefault(change_type, {"succeeded": 0, "total": 0})
            bucket["total"] += count
            if worked:
                bucket["succeeded"] += count

        return {
            ct: {
                "rate": s["succeeded"] / s["total"] if s["total"] else 0.0,
                "total": s["total"],
                "succeeded": s["succeeded"],
            }
            for ct, s in stats.items()
        }

    def get_global_lessons(
        self,
        only_worked: bool = True,
        limit: int = 20,
        *,
        exclude_test_sources: bool = False,
    ) -> list[ExperienceRecord]:
        """Get cross-strategy lessons (successful changes by default)."""
        return self.query(
            ExperienceQuery(
                only_worked=only_worked,
                limit=limit,
                exclude_test_sources=exclude_test_sources,
            )
        )

    def format_for_prompt(
        self,
        family_id: str | None = None,
        limit: int = 10,
        *,
        exclude_test_sources: bool = False,
    ) -> str:
        """Format experience records as text for LLM prompt injection.

        Returns a human-readable summary of past lessons, with recency
        weighting — newer lessons are listed first and labelled [RECENT],
        older ones are labelled [OLDER] so the LLM can weight accordingly.
        """
        records: list[ExperienceRecord] = []

        # Family-specific lessons first
        if family_id:
            records.extend(
                self.get_family_lessons(
                    family_id,
                    limit=limit,
                    exclude_test_sources=exclude_test_sources,
                )
            )

        # Pad with cross-strategy successful lessons
        remaining = limit - len(records)
        if remaining > 0:
            global_recs = self.get_global_lessons(
                only_worked=True,
                limit=remaining,
                exclude_test_sources=exclude_test_sources,
            )
            seen_ids = {(r.strategy_id, r.round_num) for r in records}
            for r in global_recs:
                if (r.strategy_id, r.round_num) not in seen_ids:
                    records.append(r)

        if not records:
            return ""

        # Records are already ordered by created_at DESC from the query.
        # Tag top half as RECENT, rest as OLDER.
        recent_cutoff = max(1, len(records) // 2)

        lines: list[str] = []
        for i, r in enumerate(records[:limit]):
            outcome = "IMPROVED" if r.worked else "NO IMPROVEMENT"
            recency = "RECENT" if i < recent_cutoff else "OLDER"
            detail_parts = [
                f"- [{recency}][{outcome}] {r.strategy_id} round {r.round_num}: "
                f"{r.change_type.value} on {r.target} "
                f"({r.from_value} → {r.to_value}) — "
                f"score {r.score_before:.1%} → {r.score_after:.1%} "
                f"({r.score_delta:+.1%})",
            ]
            # Add enriched context when available
            if r.hypothesis:
                detail_parts.append(f"  Hypothesis: {r.hypothesis}")
            if r.regime:
                detail_parts.append(f"  Regime: {r.regime}")
            if r.action_type:
                detail_parts.append(f"  Method: {r.action_type}")
            if r.lesson:
                detail_parts.append(f"  Lesson: {r.lesson}")
            lines.extend(detail_parts)

        return "\n".join(lines)

    def _exclude_test_clause(self) -> str:
        """Build a SQL predicate that hides test/fixture experience records.

        Real persistent databases may contain historical records from older test
        runs that were only identifiable by a ``test_*`` strategy id. We still
        filter those out for on-disk stores to protect live evolution.

        In-memory stores are primarily used by unit tests and isolated ad-hoc
        experiments, so filtering by ``strategy_id LIKE 'test_%'`` there would
        hide the very records the caller just inserted. In that mode we only
        exclude rows that were explicitly tagged via ``source``.
        """
        clauses = ["COALESCE(source, '') NOT IN ('test', 'fixture')"]
        if not self._is_memory:
            clauses.append("strategy_id NOT LIKE 'test_%'")
        return f"({' AND '.join(clauses)})"

    def _row_to_record(self, row: tuple) -> ExperienceRecord:
        created_at = datetime.now(timezone.utc)
        if len(row) > 15 and row[15]:
            with contextlib.suppress(ValueError, TypeError):
                created_at = datetime.fromisoformat(row[15])
        return ExperienceRecord(
            strategy_family=row[1],
            strategy_id=row[2],
            round_num=row[3],
            change_type=ChangeType(row[4]),
            target=row[5],
            from_value=json.loads(row[6]) if row[6] else None,
            to_value=json.loads(row[7]) if row[7] else None,
            reason=row[8] or "",
            score_before=row[9],
            score_after=row[10],
            score_delta=row[11],
            worked=bool(row[12]),
            failure_patterns=json.loads(row[13]) if row[13] else [],
            lesson=row[14] or "",
            created_at=created_at,
            hypothesis=row[16] if len(row) > 16 else "",
            action_type=row[17] if len(row) > 17 else "",
            regime=row[18] if len(row) > 18 else "",
            source=row[19] if len(row) > 19 else "",
            evidence_level=(
                EvidenceLevel(row[20])
                if len(row) > 20 and row[20] in {e.value for e in EvidenceLevel}
                else EvidenceLevel.BACKTEST
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


def _fuzzy_signature(
    change_type: str,
    target: str,
    to_value: object,
) -> tuple[str, str, str]:
    """Create a fuzzy change signature for deduplication.

    Numeric values are rounded to 2 decimal places so that similar changes
    (e.g., stop_loss 4%→3.0% vs 4%→2.9%) share the same signature.
    """
    normalized_value = to_value
    if isinstance(to_value, float):
        normalized_value = round(to_value, 2)
    elif isinstance(to_value, (int, bool)):
        normalized_value = to_value
    elif isinstance(to_value, str):
        with contextlib.suppress(ValueError, TypeError):
            normalized_value = round(float(to_value), 2)
    return (
        change_type,
        target.strip(),
        json.dumps(normalized_value, sort_keys=True, default=str),
    )


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
            "Experience DB path %s is not writable; falling back to in-memory store",
            resolved,
        )
        return ":memory:", True
