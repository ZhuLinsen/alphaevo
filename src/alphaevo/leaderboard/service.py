"""Scoped leaderboard ranking separating local research runs from official benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from alphaevo.provenance.contracts import DataReproducibility, ProtocolName
from alphaevo.provenance.store import RunArtifactStore
from alphaevo.strategy.store import StrategyStore

LeaderboardScope = Literal["local", "official"]


class LeaderboardEntry(BaseModel):
    """Single row returned by :class:`LeaderboardService`."""

    strategy_id: str
    run_id: str | None = None
    protocol: ProtocolName | str = "full_research_v1"
    scope: LeaderboardScope = "local"
    win_rate: float = 0.0
    avg_return: float = 0.0
    profit_loss_ratio: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    signal_count: int = 0
    confidence_score: float = 0.0
    code_dirty: bool = True
    data_reproducibility: DataReproducibility | str = "refetch_dependent"
    source: Literal["artifact", "store"] = "store"


class LeaderboardService:
    """Rank strategies cleanly within ``local`` or strict ``official`` scope."""

    def __init__(
        self,
        *,
        db_path: str | Path = "~/.alphaevo/alphaevo.db",
        artifact_root: str | Path = "reports",
    ) -> None:
        self._db_path = db_path
        self._artifact_root = artifact_root
        self._strategy_store = StrategyStore(db_path)
        self._artifact_store = RunArtifactStore(artifact_root=artifact_root, db_path=db_path)

    def get_entries(
        self,
        *,
        scope: LeaderboardScope = "local",
        limit: int = 20,
        min_signal_count: int = 30,
    ) -> list[LeaderboardEntry]:
        """Return deduplicated top entries for *scope* (never promotes local to official)."""
        if scope == "official":
            runs = self._artifact_store.list_runs(
                official_only=True,
                limit=max(50, limit * 5),
            )
            best_by_strategy: dict[str, LeaderboardEntry] = {}
            for row in runs:
                if row.signal_count < min_signal_count:
                    continue
                entry = LeaderboardEntry(
                    strategy_id=row.strategy_id,
                    run_id=row.run_id,
                    protocol=row.effective_protocol,
                    scope="official",
                    win_rate=row.win_rate,
                    avg_return=row.avg_return,
                    profit_loss_ratio=row.profit_loss_ratio,
                    max_drawdown=row.max_drawdown,
                    sharpe_ratio=row.sharpe_ratio,
                    signal_count=row.signal_count,
                    confidence_score=row.score,
                    code_dirty=row.code_dirty,
                    data_reproducibility=row.data_reproducibility,
                    source="artifact",
                )
                prev = best_by_strategy.get(entry.strategy_id)
                if prev is None or entry.confidence_score > prev.confidence_score:
                    best_by_strategy[entry.strategy_id] = entry
            ranked = sorted(
                best_by_strategy.values(),
                key=lambda e: e.confidence_score,
                reverse=True,
            )
            return ranked[:limit]

        # Local scope: combine indexed provenance runs + StrategyStore evaluations
        local_best: dict[str, LeaderboardEntry] = {}
        for row in self._artifact_store.list_runs(limit=max(100, limit * 5)):
            if row.signal_count < min_signal_count:
                continue
            entry = LeaderboardEntry(
                strategy_id=row.strategy_id,
                run_id=row.run_id,
                protocol=row.effective_protocol,
                scope="local",
                win_rate=row.win_rate,
                avg_return=row.avg_return,
                profit_loss_ratio=row.profit_loss_ratio,
                max_drawdown=row.max_drawdown,
                sharpe_ratio=row.sharpe_ratio,
                signal_count=row.signal_count,
                confidence_score=row.score,
                code_dirty=row.code_dirty,
                data_reproducibility=row.data_reproducibility,
                source="artifact",
            )
            prev = local_best.get(entry.strategy_id)
            if prev is None or entry.confidence_score > prev.confidence_score:
                local_best[entry.strategy_id] = entry

        for strat, ev in self._strategy_store.get_leaderboard(
            limit=max(50, limit * 3),
            min_signal_count=min_signal_count,
        ):
            sid = strat.meta.id
            store_entry = LeaderboardEntry(
                strategy_id=sid,
                run_id=None,
                protocol="full_research_v1",
                scope="local",
                win_rate=float(ev.overall.win_rate),
                avg_return=float(ev.overall.avg_return),
                profit_loss_ratio=float(ev.overall.profit_loss_ratio),
                max_drawdown=float(ev.overall.max_drawdown),
                sharpe_ratio=float(ev.overall.sharpe_ratio),
                signal_count=int(ev.overall.signal_count),
                confidence_score=float(ev.confidence_score),
                code_dirty=True,
                data_reproducibility="refetch_dependent",
                source="store",
            )
            prev = local_best.get(sid)
            if prev is None or store_entry.confidence_score > prev.confidence_score:
                local_best[sid] = store_entry

        ranked_local = sorted(
            local_best.values(),
            key=lambda e: e.confidence_score,
            reverse=True,
        )
        return ranked_local[:limit]


__all__ = ["LeaderboardEntry", "LeaderboardScope", "LeaderboardService"]
