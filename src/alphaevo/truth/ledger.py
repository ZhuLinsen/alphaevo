"""L0 Truth Ledger — append-only trial accounting & Deflated Sharpe Ratio."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from alphaevo.models.enums import RSILevel

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.models.strategy import Strategy

_EULER_GAMMA = 0.5772156649015329
_SRC_ROOT = Path(__file__).resolve().parents[1]
_KERNEL_FILES = (
    _SRC_ROOT / "evaluator" / "metrics.py",
    _SRC_ROOT / "truth" / "ledger.py",
    _SRC_ROOT / "truth" / "null_control.py",
    _SRC_ROOT / "truth" / "pit_probe.py",
)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function without SciPy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via bisection (~1e-15 accuracy)."""
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    lo, hi = -10.0, 10.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class DeflatedSharpeResult(BaseModel):
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio diagnostics."""

    observed_sharpe: float
    per_trade_sharpe: float
    expected_max_null_sharpe: float
    dsr_probability: float
    n_trials: int
    n_observations: int
    passes_dsr: bool


def compute_deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    *,
    per_trade_sharpe: float | None = None,
    trial_sharpe_std: float | None = None,
    min_dsr_prob: float = 0.70,
) -> DeflatedSharpeResult:
    """Compute Deflated Sharpe Ratio (DSR) probability adjusted for multiple testing.

    All internal hypothesis math operates in non-annualized per-observation Sharpe
    units so sample size ``n_observations`` and trial count ``n_trials`` scale
    consistently with finite-sample Sharpe variance.
    """
    trials = max(1, int(n_trials))
    obs = max(2, int(n_observations))

    sr_hat = (
        float(per_trade_sharpe)
        if per_trade_sharpe is not None
        else float(sharpe) / math.sqrt(252.0)
    )

    sigma_null = 1.0 / math.sqrt(max(1, obs - 1))
    sigma_trials = max(sigma_null, float(trial_sharpe_std or 0.0))

    if trials <= 1:
        sr_zero = 0.0
    else:
        p1 = min(1.0 - 1e-12, 1.0 - 1.0 / trials)
        p2 = min(1.0 - 1e-12, 1.0 - 1.0 / (trials * math.e))
        z_max = (1.0 - _EULER_GAMMA) * _norm_ppf(p1) + _EULER_GAMMA * _norm_ppf(p2)
        sr_zero = sigma_trials * z_max

    denom = max(1, obs - 1)
    kurt = max(1.0, float(kurtosis))
    variance_term = max(
        1e-8,
        (1.0 - float(skewness) * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat**2)) / denom,
    )
    sr_std = math.sqrt(variance_term)
    z_score = (sr_hat - sr_zero) / sr_std if sr_std > 0 else 0.0
    dsr_prob = _norm_cdf(z_score)

    return DeflatedSharpeResult(
        observed_sharpe=round(float(sharpe), 4),
        per_trade_sharpe=round(sr_hat, 4),
        expected_max_null_sharpe=round(sr_zero, 4),
        dsr_probability=round(dsr_prob, 4),
        n_trials=trials,
        n_observations=obs,
        passes_dsr=dsr_prob >= min_dsr_prob and sr_hat > 0.0,
    )


def compute_strategy_hash(strategy: Strategy) -> str:
    """Compute deterministic SHA-256 hex digest for a Strategy DSL definition."""
    payload = strategy.model_dump(mode="json", exclude={"complexity_score"})
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_data_fingerprint(data: dict[str, pd.DataFrame]) -> str:
    """Compute a compact deterministic SHA-256 fingerprint for market OHLCV data."""
    summary: list[dict[str, Any]] = []
    for symbol in sorted(data):
        df = data[symbol]
        if df.empty:
            summary.append({"symbol": symbol, "rows": 0})
            continue
        start = str(df["date"].iloc[0]) if "date" in df else "0"
        end = str(df["date"].iloc[-1]) if "date" in df else str(len(df) - 1)
        last_close = round(float(df["close"].iloc[-1]), 6) if "close" in df else 0.0
        summary.append(
            {
                "symbol": symbol,
                "rows": len(df),
                "start": start,
                "end": end,
                "last_close": last_close,
            }
        )
    raw = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_kernel_fingerprint() -> str:
    """Compute SHA-256 over immutable L0 truth & evaluation source files."""
    hasher = hashlib.sha256()
    for path in _KERNEL_FILES:
        if path.is_file():
            hasher.update(path.name.encode("utf-8"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


class TrialRecord(BaseModel):
    """One append-only evaluation trial recorded in the L0 Truth Ledger."""

    trial_id: str
    trial_family_id: str
    strategy_id: str
    strategy_hash: str
    data_fingerprint: str = ""
    kernel_hash: str = ""
    rsi_level: RSILevel = RSILevel.L1_STRATEGY
    round_num: int = 0
    signal_count: int = 0
    sharpe_ratio: float = 0.0
    per_trade_sharpe: float = 0.0
    confidence_score: float = 0.0
    deflated_sharpe_prob: float = 0.0
    passed_null_control: bool | None = None
    accepted: bool = False
    notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


_TRIAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS rsi_trial_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT UNIQUE NOT NULL,
    trial_family_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_hash TEXT NOT NULL,
    data_fingerprint TEXT DEFAULT '',
    kernel_hash TEXT DEFAULT '',
    rsi_level TEXT NOT NULL,
    round_num INTEGER DEFAULT 0,
    signal_count INTEGER DEFAULT 0,
    sharpe_ratio REAL DEFAULT 0.0,
    per_trade_sharpe REAL DEFAULT 0.0,
    confidence_score REAL DEFAULT 0.0,
    deflated_sharpe_prob REAL DEFAULT 0.0,
    passed_null_control INTEGER,
    accepted INTEGER DEFAULT 0,
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trial_family ON rsi_trial_ledger(trial_family_id);
CREATE INDEX IF NOT EXISTS idx_trial_strategy ON rsi_trial_ledger(strategy_id);
"""


class TrialLedger:
    """Append-only SQLite trial ledger preventing silent p-hacking & rename resets."""

    def __init__(self, db_path: str | Path = "~/.alphaevo/alphaevo.db") -> None:
        normalized, use_memory = _normalize_db_path(db_path)
        self._is_memory = use_memory
        if use_memory:
            self._db_path = ":memory:"
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(":memory:")
            self._shared_conn.execute("PRAGMA foreign_keys=ON")
        else:
            self._db_path = normalized
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._shared_conn = None
        self.init_db()

    def init_db(self) -> None:
        """Create schema if absent."""
        with self._connect() as conn:
            conn.executescript(_TRIAL_SCHEMA)

    def resolve_trial_family_id(
        self,
        strategy: Strategy,
        explicit_family_id: str | None = None,
    ) -> str:
        """Resolve root lineage family ID so child renames never reset trial counts."""
        if explicit_family_id:
            return explicit_family_id
        parent_id = strategy.meta.parent_id
        if parent_id:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT trial_family_id FROM rsi_trial_ledger "
                    "WHERE strategy_id = ? ORDER BY id DESC LIMIT 1",
                    (parent_id,),
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
        return strategy.meta.family_id

    def record_trial(
        self,
        *,
        strategy: Strategy,
        trial_family_id: str | None = None,
        data_fingerprint: str = "",
        rsi_level: RSILevel = RSILevel.L1_STRATEGY,
        round_num: int = 0,
        signal_count: int = 0,
        sharpe_ratio: float = 0.0,
        returns: list[float] | None = None,
        confidence_score: float = 0.0,
        passed_null_control: bool | None = None,
        accepted: bool = False,
        notes: str = "",
    ) -> TrialRecord:
        """Record one evaluated trial (including failed/rejected trials)."""
        family_id = self.resolve_trial_family_id(strategy, trial_family_id)
        prior_trials = self.count_family_trials(family_id)
        total_trials = prior_trials + 1

        dsr_result = self.compute_dsr_for_trials(
            family_id=family_id,
            n_trials=total_trials,
            returns=returns or [],
            observed_sharpe=sharpe_ratio,
            signal_count=signal_count,
        )

        record = TrialRecord(
            trial_id=f"tr_{uuid.uuid4().hex[:12]}",
            trial_family_id=family_id,
            strategy_id=strategy.meta.id,
            strategy_hash=compute_strategy_hash(strategy),
            data_fingerprint=data_fingerprint,
            kernel_hash=compute_kernel_fingerprint(),
            rsi_level=rsi_level,
            round_num=round_num,
            signal_count=signal_count,
            sharpe_ratio=sharpe_ratio,
            per_trade_sharpe=dsr_result.per_trade_sharpe,
            confidence_score=confidence_score,
            deflated_sharpe_prob=dsr_result.dsr_probability,
            passed_null_control=passed_null_control,
            accepted=accepted,
            notes=notes,
        )

        null_int: int | None = None
        if passed_null_control is not None:
            null_int = 1 if passed_null_control else 0

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rsi_trial_ledger (
                    trial_id, trial_family_id, strategy_id, strategy_hash,
                    data_fingerprint, kernel_hash, rsi_level, round_num,
                    signal_count, sharpe_ratio, per_trade_sharpe,
                    confidence_score, deflated_sharpe_prob,
                    passed_null_control, accepted, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.trial_id,
                    record.trial_family_id,
                    record.strategy_id,
                    record.strategy_hash,
                    record.data_fingerprint,
                    record.kernel_hash,
                    record.rsi_level.value,
                    record.round_num,
                    record.signal_count,
                    record.sharpe_ratio,
                    record.per_trade_sharpe,
                    record.confidence_score,
                    record.deflated_sharpe_prob,
                    null_int,
                    1 if record.accepted else 0,
                    record.notes,
                    record.created_at.isoformat(),
                ),
            )
            conn.commit()
        return record

    def count_family_trials(self, trial_family_id: str) -> int:
        """Count all recorded trials in one lineage family (pass + fail)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM rsi_trial_ledger WHERE trial_family_id = ?",
                (trial_family_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def mark_trial_accepted(self, trial_id: str, accepted: bool = True) -> None:
        """Update the acceptance outcome flag of a recorded trial."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE rsi_trial_ledger SET accepted = ? WHERE trial_id = ?",
                (1 if accepted else 0, trial_id),
            )
            conn.commit()

    def count_total_trials(self) -> int:
        """Count total recorded trials across all families."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM rsi_trial_ledger").fetchone()
        return int(row[0]) if row else 0

    def list_trials(
        self,
        *,
        trial_family_id: str | None = None,
        limit: int = 100,
    ) -> list[TrialRecord]:
        """List recorded trials ordered newest-first."""
        with self._connect() as conn:
            if trial_family_id is not None:
                rows = conn.execute(
                    "SELECT * FROM rsi_trial_ledger WHERE trial_family_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (trial_family_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM rsi_trial_ledger ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def compute_dsr_for_trials(
        self,
        *,
        family_id: str,
        n_trials: int,
        returns: list[float],
        observed_sharpe: float,
        signal_count: int,
    ) -> DeflatedSharpeResult:
        """Compute DSR from trade returns and historical family Sharpe dispersion."""
        n_obs = len(returns) if returns else max(2, signal_count)
        per_trade_sr: float | None = None
        skew = 0.0
        kurt = 3.0
        if len(returns) >= 3:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 0.0
            if std_r > 1e-12:
                per_trade_sr = mean_r / std_r
                m3 = sum(((r - mean_r) / std_r) ** 3 for r in returns) / len(returns)
                m4 = sum(((r - mean_r) / std_r) ** 4 for r in returns) / len(returns)
                skew = m3
                kurt = max(1.0, m4)
            else:
                per_trade_sr = 0.0

        trial_std: float | None = None
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT per_trade_sharpe FROM rsi_trial_ledger WHERE trial_family_id = ?",
                (family_id,),
            ).fetchall()
        past_srs = [float(r[0]) for r in rows if r[0] is not None]
        if len(past_srs) >= 2:
            mean_sr = sum(past_srs) / len(past_srs)
            var_sr = sum((x - mean_sr) ** 2 for x in past_srs) / (len(past_srs) - 1)
            trial_std = math.sqrt(var_sr)

        return compute_deflated_sharpe(
            observed_sharpe,
            n_trials=n_trials,
            n_observations=n_obs,
            skewness=skew,
            kurtosis=kurt,
            per_trade_sharpe=per_trade_sr,
            trial_sharpe_std=trial_std,
        )

    def _row_to_record(self, row: tuple[Any, ...]) -> TrialRecord:
        created_at = datetime.now(timezone.utc)
        if row[17]:
            with contextlib.suppress(ValueError, TypeError):
                created_at = datetime.fromisoformat(str(row[17]))
        null_val: bool | None = None
        if row[14] is not None:
            null_val = bool(row[14])
        return TrialRecord(
            trial_id=str(row[1]),
            trial_family_id=str(row[2]),
            strategy_id=str(row[3]),
            strategy_hash=str(row[4]),
            data_fingerprint=str(row[5] or ""),
            kernel_hash=str(row[6] or ""),
            rsi_level=RSILevel(str(row[7])),
            round_num=int(row[8] or 0),
            signal_count=int(row[9] or 0),
            sharpe_ratio=float(row[10] or 0.0),
            per_trade_sharpe=float(row[11] or 0.0),
            confidence_score=float(row[12] or 0.0),
            deflated_sharpe_prob=float(row[13] or 0.0),
            passed_null_control=null_val,
            accepted=bool(row[15]),
            notes=str(row[16] or ""),
            created_at=created_at,
        )

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


def _normalize_db_path(db_path: str | Path) -> tuple[str, bool]:
    """Return writable SQLite path, or explicit in-memory mode for ``':memory:'``."""
    if str(db_path) == ":memory:":
        return ":memory:", True
    resolved = Path(db_path).expanduser().resolve()
    probe = resolved.parent / f".{resolved.stem}.trial_probe"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    probe.touch(exist_ok=True)
    probe.unlink(missing_ok=True)
    return str(resolved), False


__all__ = [
    "DeflatedSharpeResult",
    "TrialLedger",
    "TrialRecord",
    "compute_data_fingerprint",
    "compute_deflated_sharpe",
    "compute_kernel_fingerprint",
    "compute_strategy_hash",
]
