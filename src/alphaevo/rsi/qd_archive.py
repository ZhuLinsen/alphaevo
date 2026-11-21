"""MAP-Elites Quality-Diversity archive for regime/style/horizon stock niches."""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field

from alphaevo.models.enums import MarketRegime
from alphaevo.models.execution import EvaluationReport
from alphaevo.models.strategy import Strategy

RegimeBucket = Literal["bull_trend", "range_reversal", "defensive_volatile"]
HorizonBucket = Literal["short_1_5d", "swing_6_15d", "position_16d_plus"]


class NicheCoord(BaseModel):
    """3D behavioral descriptor for a stock strategy cell."""

    regime_bucket: RegimeBucket
    style_bucket: str
    horizon_bucket: HorizonBucket

    @property
    def key(self) -> str:
        return f"{self.regime_bucket}|{self.style_bucket}|{self.horizon_bucket}"


class ArchiveCell(BaseModel):
    """Winning strategy specimen stored inside one behavioral niche."""

    niche: NicheCoord
    strategy_id: str
    strategy: Strategy
    evaluation: EvaluationReport
    fitness_score: float


class ArchiveSummary(BaseModel):
    """Coverage & best-per-niche snapshot of the MAP-Elites grid."""

    total_cells_occupied: int
    total_insertions: int
    total_replacements: int
    best_strategy_id: str | None = None
    best_fitness: float = 0.0
    cells: list[ArchiveCell] = Field(default_factory=list)


def classify_niche(strategy: Strategy, evaluation: EvaluationReport) -> NicheCoord:
    """Map a strategy and its evaluation behavior into a 3D QD cell."""
    preferred = {r.lower() for r in strategy.meta.preferred_regime}
    regime_bucket: RegimeBucket = "range_reversal"
    if preferred & {MarketRegime.TRENDING_UP.value, MarketRegime.EUPHORIA.value}:
        regime_bucket = "bull_trend"
    elif preferred & {
        MarketRegime.PANIC.value,
        MarketRegime.VOLATILE.value,
        MarketRegime.TRENDING_DOWN.value,
    }:
        regime_bucket = "defensive_volatile"
    elif evaluation.by_regime:
        best_rm = max(evaluation.by_regime, key=lambda rm: rm.avg_return)
        if best_rm.regime in {MarketRegime.TRENDING_UP, MarketRegime.EUPHORIA}:
            regime_bucket = "bull_trend"
        elif best_rm.regime in {
            MarketRegime.PANIC,
            MarketRegime.VOLATILE,
            MarketRegime.TRENDING_DOWN,
        }:
            regime_bucket = "defensive_volatile"

    days = strategy.exit.max_holding_days
    if days <= 5:
        horizon: HorizonBucket = "short_1_5d"
    elif days <= 15:
        horizon = "swing_6_15d"
    else:
        horizon = "position_16d_plus"

    return NicheCoord(
        regime_bucket=regime_bucket,
        style_bucket=strategy.meta.category.value,
        horizon_bucket=horizon,
    )


class MAPElitesArchive:
    """Preserves niche champions across regimes, styles, and holding horizons."""

    def __init__(self) -> None:
        self._cells: dict[str, ArchiveCell] = {}
        self._insertions = 0
        self._replacements = 0

    def insert(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        fitness_score: float,
    ) -> bool:
        """Insert candidate if niche is empty or candidate strictly beats cell incumbent."""
        self._insertions += 1
        niche = classify_niche(strategy, evaluation)
        key = niche.key
        incumbent = self._cells.get(key)
        if incumbent is None or fitness_score > incumbent.fitness_score:
            if incumbent is not None:
                self._replacements += 1
            self._cells[key] = ArchiveCell(
                niche=niche,
                strategy_id=strategy.meta.id,
                strategy=strategy,
                evaluation=evaluation,
                fitness_score=round(fitness_score, 6),
            )
            return True
        return False

    def get(self, key: str) -> ArchiveCell | None:
        return self._cells.get(key)

    def select_elites(self, k: int = 3) -> list[ArchiveCell]:
        """Return top-k niche champions sorted by fitness descending."""
        ordered = sorted(self._cells.values(), key=lambda c: c.fitness_score, reverse=True)
        return ordered[: max(0, k)]

    def sample_elite(
        self,
        *,
        exclude_id: str | None = None,
        rng: random.Random | None = None,
    ) -> ArchiveCell | None:
        """Select a niche champion different from *exclude_id* when available."""
        candidates = [c for c in self._cells.values() if c.strategy_id != exclude_id]
        if not candidates:
            return None
        picker = rng or random.Random(42)
        candidates.sort(key=lambda c: c.fitness_score, reverse=True)
        top_pool = candidates[: max(1, min(3, len(candidates)))]
        return picker.choice(top_pool)

    def coverage_summary(self) -> ArchiveSummary:
        """Summarize archive diversity and highest-fitness cell."""
        ordered = self.select_elites(len(self._cells))
        best = ordered[0] if ordered else None
        return ArchiveSummary(
            total_cells_occupied=len(self._cells),
            total_insertions=self._insertions,
            total_replacements=self._replacements,
            best_strategy_id=best.strategy_id if best is not None else None,
            best_fitness=best.fitness_score if best is not None else 0.0,
            cells=ordered,
        )


__all__ = [
    "ArchiveCell",
    "ArchiveSummary",
    "MAPElitesArchive",
    "NicheCoord",
    "classify_niche",
]
