"""Execution, evaluation, and reflection models."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from alphaevo.models.enums import (
    ChangeType,
    ExitReason,
    MarketRegime,
    SamplingMethod,
    SignalDirection,
)
from alphaevo.models.strategy import Strategy

# ── Sampling ──────────────────────────────────────────────────────────


class SampleBatch(BaseModel):
    """A batch of sampled stocks for backtesting."""

    batch_id: str
    strategy_id: str
    symbols: list[str]
    date_range: tuple[date, date]
    market_regimes: list[MarketRegime] = Field(default_factory=list)
    sampling_method: SamplingMethod = SamplingMethod.REPRESENTATIVE
    sampling_reason: str = ""
    requested_max_symbols: int = 0
    sampling_attempt: int = 1
    signal_count_target: int = 0
    signal_count_reached: int = 0
    insufficient_signals: bool = False
    sampling_history: list["SamplingAttempt"] = Field(default_factory=list)


class SamplingAttempt(BaseModel):
    """One sampling attempt, including auto-expansion retries."""

    attempt_num: int
    max_symbols: int
    date_range: tuple[date, date]
    sampling_method: SamplingMethod = SamplingMethod.REPRESENTATIVE
    sampling_reason: str = ""
    selected_symbols: int = 0
    signal_count: int = 0
    accepted: bool = False
    note: str = ""


# ── Backtest ──────────────────────────────────────────────────────────


class TradeSignal(BaseModel):
    """A single trade signal and its outcome."""

    symbol: str
    signal_date: date
    direction: SignalDirection
    entry_price: float
    exit_price: float | None = None
    exit_date: date | None = None
    exit_reason: ExitReason | None = None
    return_pct: float = 0.0
    holding_days: int = 0
    regime: MarketRegime | None = None
    sector: str | None = None
    # Indicator snapshot at entry — enables data-driven LLM reflection
    indicator_snapshot: dict[str, float | bool] = Field(default_factory=dict)


class BacktestResult(BaseModel):
    """Results from backtesting a strategy on a sample batch."""

    strategy_id: str
    batch_id: str
    signals: list[TradeSignal]
    total_signals: int = 0
    executed_signals: int = 0
    skipped_signals: int = 0
    date_range: tuple[date, date] = Field(default=(date.today(), date.today()))


# ── Evaluation ────────────────────────────────────────────────────────


class OverallMetrics(BaseModel):
    """Aggregate performance metrics."""

    win_rate: float = 0.0
    avg_return: float = 0.0
    avg_win_return: float = 0.0
    avg_loss_return: float = 0.0
    profit_loss_ratio: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    signal_count: int = 0
    avg_holding_days: float = 0.0
    max_consecutive_loss: int = 0
    median_return: float = 0.0
    total_return: float = 0.0


class RegimeMetrics(BaseModel):
    """Performance metrics segmented by market regime."""

    regime: MarketRegime
    win_rate: float = 0.0
    avg_return: float = 0.0
    signal_count: int = 0


class AntiFitMetrics(BaseModel):
    """Anti-overfitting quality metrics."""

    train_win_rate: float = 0.0
    val_win_rate: float = 0.0
    test_win_rate: float = 0.0
    train_val_gap: float = 0.0
    val_test_gap: float = 0.0
    yearly_consistency: float = 0.0  # 1 - std/mean
    walk_forward_gap: float = 0.0  # mean train-test gap across rolling folds
    walk_forward_pass_rate: float = 0.0  # share of folds within acceptable gap
    param_sensitivity: float = 0.0  # performance drop on param perturbation
    complexity_penalty: float = 0.0

    @property
    def is_overfit(self) -> bool:
        """Heuristic overfit detection."""
        return self.train_val_gap > 0.15 or self.val_test_gap > 0.10 or self.param_sensitivity > 0.3


class BenchmarkComparison(BaseModel):
    """Comparison of strategy performance against buy-and-hold baseline."""

    benchmark_return: float = 0.0  # buy-and-hold return over same period
    strategy_return: float = 0.0  # strategy total return
    excess_return: float = 0.0  # strategy - benchmark (alpha)
    benchmark_max_drawdown: float = 0.0
    benchmark_sharpe: float = 0.0
    symbols_used: int = 0
    random_baseline_mean: float | None = None
    random_baseline_std: float | None = None
    random_baseline_ci_lower: float | None = None
    random_baseline_ci_upper: float | None = None
    random_baseline_beat_fraction: float | None = None

    @property
    def beats_benchmark(self) -> bool:
        return self.excess_return > 0


class EventContextMetrics(BaseModel):
    """Coverage and source-quality summary for event/news context in one run."""

    total_symbols: int = 0
    provider_symbols: int = 0
    mixed_symbols: int = 0
    proxy_symbols: int = 0
    provider_coverage: float = 0.0
    proxy_only_coverage: float = 0.0
    relevant_indicators: list[str] = Field(default_factory=list)
    source_breakdown: dict[str, int] = Field(default_factory=dict)

    @property
    def uses_event_indicators(self) -> bool:
        """Whether the strategy evaluation actually depends on event/news fields."""
        return bool(self.relevant_indicators)

    @property
    def is_proxy_dominant(self) -> bool:
        """True when event/news signals are mostly price/volume proxy-derived."""
        return self.uses_event_indicators and (
            self.provider_coverage < 0.30 or self.proxy_only_coverage > 0.50
        )

    @property
    def has_proxy_caveat(self) -> bool:
        """True when any active event/news indicator used proxy-derived context."""
        return self.uses_event_indicators and self.proxy_only_coverage > 0.0


class WalkForwardFoldMetrics(BaseModel):
    """One rolling walk-forward fold."""

    fold_num: int
    train_signal_count: int = 0
    test_signal_count: int = 0
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    gap: float = 0.0


class WalkForwardProtocol(BaseModel):
    """Configured walk-forward evaluation protocol for one report."""

    requested_folds: int = 3
    effective_folds: int = 0
    train_pct: float = 0.7
    test_pct: float = 0.3
    pass_gap: float = 0.10
    min_signals_per_split: int = 5


class RegimeHoldoutCase(BaseModel):
    """One leave-one-regime-out robustness check."""

    regime: MarketRegime
    preferred: bool = False
    in_sample_signal_count: int = 0
    holdout_signal_count: int = 0
    in_sample_win_rate: float = 0.0
    holdout_win_rate: float = 0.0
    gap: float = 0.0


class RegimeHoldoutMetrics(BaseModel):
    """Aggregate cross-regime holdout diagnostics."""

    preferred_regimes: list[str] = Field(default_factory=list)
    pass_gap: float = 0.10
    total_cases: int = 0
    pass_rate: float = 0.0
    worst_gap: float = 0.0
    worst_regime: MarketRegime | None = None
    holdouts: list[RegimeHoldoutCase] = Field(default_factory=list)


class StressWindowCase(BaseModel):
    """Performance during one benchmark-defined stress window."""

    window_num: int
    start_date: date
    end_date: date
    benchmark_return: float = 0.0
    benchmark_drawdown: float = 0.0
    signal_count: int = 0
    strategy_win_rate: float = 0.0
    strategy_avg_return: float = 0.0
    strategy_total_return: float = 0.0
    alpha: float = 0.0


class StressWindowMetrics(BaseModel):
    """Aggregate resilience metrics over the benchmark's worst windows."""

    window_days: int = 20
    top_k: int = 3
    alpha_pass_threshold: float = 0.0
    total_windows: int = 0
    pass_rate: float = 0.0
    average_alpha: float = 0.0
    worst_alpha: float = 0.0
    worst_window_start: date | None = None
    worst_window_end: date | None = None
    worst_benchmark_return: float = 0.0
    windows: list[StressWindowCase] = Field(default_factory=list)


class CPCVMetrics(BaseModel):
    """Combinatorial Purged Cross-Validation diagnostics."""

    n_groups: int = 6
    n_test_groups: int = 2
    purge_days: int = 5
    n_paths: int = 0
    mean_gap: float = 0.0
    max_gap: float = 0.0
    mean_test_win_rate: float = 0.0
    std_test_win_rate: float = 0.0


class EvaluationReport(BaseModel):
    """Comprehensive strategy evaluation report."""

    evaluation_id: str = ""
    strategy_id: str
    batch_id: str = ""
    overall: OverallMetrics = Field(default_factory=OverallMetrics)
    by_regime: list[RegimeMetrics] = Field(default_factory=list)
    by_sector: dict[str, OverallMetrics] = Field(default_factory=dict)
    failure_cases: list[TradeSignal] = Field(default_factory=list)
    top_patterns: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    anti_overfit: AntiFitMetrics = Field(default_factory=AntiFitMetrics)
    benchmark: BenchmarkComparison | None = None
    event_context: EventContextMetrics | None = None
    walk_forward: list[WalkForwardFoldMetrics] = Field(default_factory=list)
    walk_forward_protocol: WalkForwardProtocol | None = None
    regime_holdout: RegimeHoldoutMetrics | None = None
    stress_windows: StressWindowMetrics | None = None
    cpcv: CPCVMetrics | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Reflection ────────────────────────────────────────────────────────


class StrategyChange(BaseModel):
    """A single proposed change to a strategy."""

    change_type: ChangeType
    target: str  # field path being modified
    from_value: Any = None
    to_value: Any = None
    reason: str = ""


class ResearchHypothesis(BaseModel):
    """A hypothesis about why a strategy underperforms and what might fix it."""

    problem: str  # identified root cause
    hypothesis: str  # proposed explanation
    expected_outcome: str = ""  # what we expect if the hypothesis is correct
    confidence: float = 0.5  # LLM's self-assessed confidence


class CandidateExperiment(BaseModel):
    """One candidate experiment proposed by the LLM research agent."""

    hypothesis: ResearchHypothesis
    proposed_changes: list[StrategyChange] = Field(default_factory=list)
    priority_score: float = 0.0  # critic-assigned priority (higher = try first)
    rationale: str = ""  # why this experiment is worth running


class LLMCallTelemetry(BaseModel):
    """Structured telemetry for one LLM call inside reflection."""

    stage: Literal["diagnosis", "experiment_design", "single_step"]
    model: str = ""
    timeout_seconds: int = 0
    max_retries: int = 0
    duration_ms: int = 0
    success: bool = True
    error: str = ""


class LLMReflectionTelemetry(BaseModel):
    """Observability metadata for one reflection attempt."""

    path: Literal[
        "two_step",
        "single_step_fallback",
        "heuristic_fallback",
        "heuristic_only",
    ] = "heuristic_only"
    fallback_trigger: str = ""
    total_duration_ms: int = 0
    calls: list[LLMCallTelemetry] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    """Output of reflection/attribution analysis."""

    strategy_id: str
    evaluation_id: str
    failure_patterns: list[str] = Field(default_factory=list)
    proposed_changes: list[StrategyChange] = Field(default_factory=list)
    candidates: list[CandidateExperiment] = Field(default_factory=list)
    diagnosis: str = ""  # root-cause analysis from LLM (step 1)
    next_strategy_id: str = ""
    next_strategy: Strategy | None = None
    reflection_summary: str = ""
    llm_telemetry: LLMReflectionTelemetry | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Evolution Tree ────────────────────────────────────────────────────


class EvolutionNode(BaseModel):
    """A node in the strategy evolution tree."""

    strategy_id: str
    parent_id: str | None = None
    version: int = 1
    changes_from_parent: list[StrategyChange] = Field(default_factory=list)
    evaluation: EvaluationReport | None = None
    status: Literal["active", "pruned", "champion"] = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
