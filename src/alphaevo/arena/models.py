"""Pydantic v2 models for the Live Self-Evolving Arena & HTML Dashboards."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EquityPoint(BaseModel):
    """One chronological point on strategy vs benchmark equity and drawdown curves."""

    date: str
    strategy_equity: float
    benchmark_equity: float
    drawdown: float = 0.0


class EvolutionNodeView(BaseModel):
    """Single mutation / ratchet evaluation node in an evolution history."""

    round_num: int
    strategy_id: str
    parent_id: str | None = None
    status: str = "baseline"
    level: str = "L1"
    score: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    max_drawdown: float = 0.0
    signals: int = 0
    dsr_prob: float = 0.0
    null_passed: bool = False
    description: str = ""
    yaml_diff: list[str] = Field(default_factory=list)


class ArenaStrategyEntry(BaseModel):
    """Full verifiable evaluation, evolution, and shadow record for one strategy."""

    strategy_id: str
    champion_id: str
    name: str
    category: str
    market: str
    baseline_score: float = 0.0
    champion_score: float = 0.0
    score_delta: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    signal_count: int = 0
    dsr_prob: float = 0.0
    null_passed: bool = False
    promotions: int = 0
    vetoes: int = 0
    trials_recorded: int = 0
    shadow_stage: str = "shadow"
    shadow_mature_signals: int = 0
    shadow_avg_return: float = 0.0
    shadow_chain_verified: bool = True
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    evolution_nodes: list[EvolutionNodeView] = Field(default_factory=list)
    committee_verdicts: list[dict[str, Any]] = Field(default_factory=list)
    champion_yaml: str = ""
    baseline_yaml: str = ""
    yaml_diff: list[str] = Field(default_factory=list)


class ArenaSnapshot(BaseModel):
    """Full snapshot rendered to JSON, Markdown digest, and static Arena HTML."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market_label: str = "US Tech Snapshot"
    data_source: str = "bundled_snapshot"
    date_range: tuple[str, str] = ("", "")
    symbols: list[str] = Field(default_factory=list)
    total_strategies: int = 0
    total_promotions: int = 0
    total_vetoes: int = 0
    chain_verified: bool = True
    entries: list[ArenaStrategyEntry] = Field(default_factory=list)
    allocation_weights: dict[str, float] = Field(default_factory=dict)
    allocation_metrics: dict[str, float] = Field(default_factory=dict)
    factor_governance: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "ArenaSnapshot",
    "ArenaStrategyEntry",
    "EquityPoint",
    "EvolutionNodeView",
]
