"""Pydantic contracts for Bitemporal Forward Shadow Ledger & Promotion."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from alphaevo.models.enums import PromotionStage


class BitemporalEnvelope(BaseModel):
    """Separates point-in-time market date (as_of) from UTC record time."""

    as_of: date
    available_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StageTransition(BaseModel):
    """Audit trail entry for a strategy promotion/demotion event."""

    from_stage: PromotionStage
    to_stage: PromotionStage
    reason: str
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CandidateSpec(BaseModel):
    """Preregistered candidate strategy tracked in the Forward Shadow Ledger."""

    strategy_id: str
    family_id: str
    strategy_hash: str
    stage: PromotionStage = PromotionStage.SHADOW
    backtest_avg_return: float = 0.0
    backtest_win_rate: float = 0.0
    backtest_sharpe: float = 0.0
    deflated_sharpe_prob: float = 0.0
    null_control_passed: bool = False
    stage_entered_as_of: date | None = None
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage_history: list[StageTransition] = Field(default_factory=list)


class SetupInstance(BaseModel):
    """Concrete forward stock trade setup committed before outcome realization."""

    signal_id: str
    strategy_id: str
    symbol: str
    sector: str = "unknown"
    regime: str = "unknown"
    envelope: BitemporalEnvelope
    entry_reference_price: float
    planned_stop_price: float | None = None
    target_price: float | None = None
    planned_rr: float | None = None
    structural_invalidation: str = ""
    quality_grade: Literal["A", "B", "C"] = "B"
    conviction_score: float = 0.5
    ready_now: bool = True


class ForecastCommit(BaseModel):
    """Hash-chained append-only commitment of forward signals for one day."""

    commit_id: str
    strategy_id: str
    as_of: date
    committed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market_temperature: Literal["normal", "cautious", "defensive"] = "normal"
    total_raw_candidates: int = 0
    sector_capped_count: int = 0
    is_replay: bool = False
    signals: list[SetupInstance] = Field(default_factory=list)
    prev_hash: str = "0" * 64
    record_hash: str = ""


class OutcomeHorizon(BaseModel):
    """Realized post-commitment evaluation at one T+H horizon."""

    horizon_days: Literal[5, 10, 20]
    mature: bool = False
    exit_date: date | None = None
    raw_signal_return: float = 0.0
    tradable_post_cost_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    stopped_out: bool = False


class OutcomeEvent(BaseModel):
    """Out-of-sample outcome event linked by signal_id to a committed SetupInstance."""

    signal_id: str
    strategy_id: str
    symbol: str
    sector: str = "unknown"
    as_of: date
    evaluated_as_of: date
    horizons: dict[int, OutcomeHorizon] = Field(default_factory=dict)


class ForwardDecayReport(BaseModel):
    """Forward vs Backtest performance decay and cluster-adjusted sample report."""

    strategy_id: str
    stage: PromotionStage
    horizon_days: int = 10
    mature_trading_days: int = 0
    total_committed_signals: int = 0
    mature_signals: int = 0
    immature_signals: int = 0
    effective_signals: float = 0.0
    forward_win_rate: float = 0.0
    forward_avg_post_cost_return: float = 0.0
    forward_avg_excess_return: float = 0.0
    forward_lower_bound_90: float = 0.0
    backtest_avg_return: float = 0.0
    decay_ratio: float | None = None
    max_forward_drawdown: float = 0.0
    worst_signals: list[str] = Field(default_factory=list)


class PromotionDecision(BaseModel):
    """Asymmetric Slow-Promote / Fast-Demote verdict on a tracked candidate."""

    strategy_id: str
    previous_stage: PromotionStage
    new_stage: PromotionStage
    action: Literal["promote", "hold", "halt", "retire"]
    passed_gates: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = [
    "BitemporalEnvelope",
    "CandidateSpec",
    "ForecastCommit",
    "ForwardDecayReport",
    "OutcomeEvent",
    "OutcomeHorizon",
    "PromotionDecision",
    "SetupInstance",
    "StageTransition",
]
