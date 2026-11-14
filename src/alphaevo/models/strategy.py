"""Strategy definition models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, computed_field

from alphaevo.models.enums import MarketType, StrategyCategory, StrategyStatus

if TYPE_CHECKING:
    from alphaevo.models.execution import EvaluationReport


class StrategyMeta(BaseModel):
    """Strategy metadata."""

    id: str
    name: str
    version: int = 1
    parent_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market: MarketType = MarketType.A_SHARE
    category: StrategyCategory = StrategyCategory.TREND
    tags: list[str] = Field(default_factory=list)
    status: StrategyStatus = StrategyStatus.ACTIVE
    preferred_regime: list[str] = Field(default_factory=list)
    experimental: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def family_id(self) -> str:
        """Root strategy family identifier (strips version suffix)."""
        parts = self.id.rsplit("_v", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else self.id


class StrategyCondition(BaseModel):
    """A single condition in strategy entry/filter."""

    indicator: str
    op: Literal["==", "!=", ">", ">=", "<", "<="]
    value: float | bool | str


class ExecutionConfig(BaseModel):
    """Entry execution configuration (v0.3)."""

    timing: Literal["next_open", "close", "breakout_high"] = "next_open"
    slippage: float | None = Field(default=None, ge=0, le=0.1)


class StrategyEntry(BaseModel):
    """Entry conditions for a strategy."""

    logic: Literal["and", "or"] = "and"
    triggers: list[StrategyCondition] = Field(default_factory=list)
    guards: list[StrategyCondition] = Field(default_factory=list)
    conditions: list[StrategyCondition] = Field(default_factory=list)
    filters: list[StrategyCondition] = Field(default_factory=list)
    execution: ExecutionConfig | None = None


class StopLossConfig(BaseModel):
    """Stop loss configuration."""

    type: str = "pct"
    value: float | None = None
    multiplier: float | None = None
    atr_period: int | None = Field(default=None, ge=1)
    reference: str | None = None
    conditions: list[StrategyCondition] | None = None


class TakeProfitConfig(BaseModel):
    """Take profit configuration."""

    type: str = "rr"  # rr, pct, target_ma, trailing
    value: float | None = None
    target: str | None = None
    trigger_pct: float | None = None
    trail_pct: float | None = None


class StrategyExit(BaseModel):
    """Exit rules for a strategy."""

    triggers: list[StrategyCondition] = Field(default_factory=list)
    stop_loss: StopLossConfig
    take_profit: TakeProfitConfig
    max_holding_days: int = 10


class UniverseFilter(BaseModel):
    """A single universe filter condition."""

    field: str
    op: Literal["==", "!=", ">", ">=", "<", "<="]
    value: float | bool | str


class UniverseConfig(BaseModel):
    """Stock universe selection configuration."""

    market: list[str] = Field(default_factory=lambda: ["a_share_main"])
    filters: list[UniverseFilter] = Field(default_factory=list)


class TunableParam(BaseModel):
    """Definition of a tunable parameter for evolution."""

    target: (
        str  # e.g. "entry.triggers[indicator=rsi_14].value"
        # or "entry.guards[indicator=relative_strength_20d].value"
        # or "entry.conditions[indicator=rsi_14].value" (legacy)
        # or "...close_above_ma60].indicator"
        # or "...volume_ratio_1d_5d].indicator"
        # or "...relative_strength_20d].indicator"
        # or "...rsi_14].indicator"
        # or "...ma5_ge_ma10_or_crossing].indicator.fast/.slow"
        # or "...macd_histogram].indicator.fast/.slow/.signal"
        # or "...bollinger_band_width].indicator/.std"
    )
    range: tuple[float, float]
    step: float
    label: str | None = None


class StrategyParams(BaseModel):
    """Tunable parameters configuration."""

    tunable: list[TunableParam] = Field(default_factory=list)


class MarketRuleConfig(BaseModel):
    """Market-specific trading rules."""

    t_plus_1: bool = False  # buy today, sell tomorrow
    limit_up_down: bool = False  # limit-up can't buy, limit-down can't sell
    suspension: bool = False  # skip suspended days


class SetupSpec(BaseModel):
    """Explicit trade-setup contract linking signal triggers to execution and invalidation."""

    trigger_rule: str = ""
    planned_stop_pct: float | None = Field(default=None, gt=0.0, le=0.5)
    structural_invalidation: str = ""
    target_rr: float | None = Field(default=None, gt=0.0)
    min_rr: float = Field(default=1.5, gt=0.0)


class PortfolioProfile(BaseModel):
    """Portfolio concentration and regime-capacity guardrails."""

    mode: Literal["none", "hard_cap"] = "none"
    max_per_sector: int = Field(default=2, ge=1)
    max_positions: int = Field(default=10, ge=1)
    market_temp_cap: bool = False


class MarketHypothesisSummary(BaseModel):
    """Human-readable summary of the market belief behind a strategy."""

    thesis: str
    expected_regimes: list[str] = Field(default_factory=list)
    key_indicators: list[str] = Field(default_factory=list)
    signal_style: str = ""
    execution_assumption: str = ""
    risk_assumption: str = ""


class MarketHypothesisAssessment(BaseModel):
    """Heuristic assessment of whether the strategy thesis is holding up."""

    summary: MarketHypothesisSummary
    status: Literal[
        "unproven_small_sample",
        "parameter_misaligned",
        "execution_misaligned",
        "thesis_misaligned",
        "partially_validated",
    ]
    rationale: str
    next_step: str


class Strategy(BaseModel):
    """Complete strategy definition with dual representation."""

    meta: StrategyMeta
    description: str  # Human-readable description
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    entry: StrategyEntry
    exit: StrategyExit
    params: StrategyParams = Field(default_factory=StrategyParams)
    market_rules: dict[str, MarketRuleConfig] = Field(default_factory=dict)
    setup: SetupSpec | None = None
    portfolio_profile: PortfolioProfile | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def complexity_score(self) -> float:
        """Auto-computed complexity penalty.
        More conditions = higher penalty. Used in anti-overfitting scoring.
        """
        n_conditions = len(self.entry.conditions) + len(self.entry.triggers)
        n_filters = len(self.entry.filters) + len(self.entry.guards)
        n_exit_rules = 1  # base
        n_exit_rules += len(self.exit.triggers)
        if self.exit.stop_loss.conditions:
            n_exit_rules += len(self.exit.stop_loss.conditions)
        total = n_conditions + n_filters + n_exit_rules
        # Sigmoid-like penalty: low for <=5, ramps up for >5
        return min(1.0, max(0.0, (total - 3) / 10))

    def build_market_hypothesis(self) -> MarketHypothesisSummary:
        """Summarize the market thesis implied by the strategy."""
        default_thesis = {
            StrategyCategory.TREND: (
                "Relative-strength names keep working after orderly pullbacks "
                "or continuation setups."
            ),
            StrategyCategory.REVERSAL: (
                "Short-term dislocations mean-revert once selling pressure is exhausted."
            ),
            StrategyCategory.EVENT: (
                "Event-driven price dislocations create temporary mispricings that can be traded."
            ),
            StrategyCategory.ROTATION: (
                "Leadership rotates across sectors and strong groups keep absorbing flows."
            ),
            StrategyCategory.FRAMEWORK: (
                "Different market regimes need different entry and risk templates."
            ),
        }
        default_regimes = {
            StrategyCategory.TREND: ["trending_up"],
            StrategyCategory.REVERSAL: ["range_bound", "oversold_rebound"],
            StrategyCategory.EVENT: ["event_driven"],
            StrategyCategory.ROTATION: ["sector_rotation"],
            StrategyCategory.FRAMEWORK: ["mixed"],
        }

        description = " ".join(self.description.strip().split())
        first_sentence = description.split("。", 1)[0].split(".", 1)[0].strip()
        thesis = first_sentence or default_thesis.get(
            self.meta.category,
            "This strategy encodes a repeatable market edge hypothesis.",
        )

        key_indicators: list[str] = []
        for condition in [
            *self.entry.triggers,
            *self.entry.conditions,
            *self.entry.guards,
            *self.entry.filters,
        ]:
            if condition.indicator not in key_indicators:
                key_indicators.append(condition.indicator)

        if self.entry.logic == "and":
            signal_style = (
                "High-conviction gating: every entry condition must align before the strategy acts."
            )
        else:
            signal_style = (
                "Multi-path triggering: alternative entry patterns can fire the strategy."
            )

        execution = self.entry.execution
        timing = execution.timing if execution is not None else "next_open"
        execution_assumption = {
            "next_open": "Assumes the edge survives overnight and is still tradable at the next open.",
            "close": "Assumes same-day closes still capture the intended signal edge.",
            "breakout_high": "Assumes confirmation above prior highs matters more than early entry.",
        }.get(
            timing,
            "Assumes the chosen execution timing preserves the edge encoded by the setup.",
        )

        stop_type = self.exit.stop_loss.type or "pct"
        take_profit_type = self.exit.take_profit.type or "rr"
        risk_assumption = (
            f"Risk is managed with `{stop_type}` stops and `{take_profit_type}` profit taking, "
            "so the thesis must survive that exit profile."
        )

        return MarketHypothesisSummary(
            thesis=thesis,
            expected_regimes=list(self.meta.preferred_regime)
            or default_regimes.get(self.meta.category, []),
            key_indicators=key_indicators,
            signal_style=signal_style,
            execution_assumption=execution_assumption,
            risk_assumption=risk_assumption,
        )

    def assess_market_hypothesis(self, evaluation: EvaluationReport) -> MarketHypothesisAssessment:
        """Classify whether performance issues look like thesis, parameter, or execution problems."""
        summary = self.build_market_hypothesis()
        overall = evaluation.overall
        anti = evaluation.anti_overfit
        benchmark = evaluation.benchmark

        preferred_regimes = set(summary.expected_regimes)
        preferred_metrics = [
            regime for regime in evaluation.by_regime if regime.regime.value in preferred_regimes
        ]
        preferred_regime_weak = any(
            regime.win_rate < 0.45 or regime.avg_return < 0 for regime in preferred_metrics
        )
        overfit_symptoms = (
            anti.is_overfit or anti.train_val_gap > 0.10 or anti.param_sensitivity > 0.30
        )
        benchmark_trails_without_edge = (
            benchmark is not None and benchmark.excess_return < -0.03 and overall.win_rate < 0.48
        )
        thesis_break_signals = preferred_regime_weak or benchmark_trails_without_edge

        if overall.signal_count < 30:
            return MarketHypothesisAssessment(
                summary=summary,
                status="unproven_small_sample",
                rationale=(
                    f"Only {overall.signal_count} signals were observed, which is below the "
                    "minimum evidence threshold for judging the thesis."
                ),
                next_step=(
                    "Expand the sampling window or universe before deciding whether the thesis or "
                    "parameters are wrong."
                ),
            )

        if overall.win_rate >= 0.52 and overall.avg_return <= 0:
            return MarketHypothesisAssessment(
                summary=summary,
                status="execution_misaligned",
                rationale=(
                    "Signals are winning often enough, but the payoff profile stays weak or "
                    "negative. That usually points to entry timing, stop placement, or exit design."
                ),
                next_step=(
                    "Rework execution timing or exit logic before abandoning the underlying thesis."
                ),
            )

        if thesis_break_signals and (
            overall.win_rate < 0.46 or overall.avg_return <= 0 or anti.train_val_gap <= 0.20
        ):
            rationale = (
                "The strategy underperforms in the regimes it claims to target, or it trails the "
                "benchmark without showing convincing edge. That usually means the market belief "
                "itself is not holding."
            )
            if overfit_symptoms:
                rationale += (
                    " The overfit symptoms look more like a downstream effect of a weak thesis "
                    "than a pure parameter-tuning problem."
                )
            return MarketHypothesisAssessment(
                summary=summary,
                status="thesis_misaligned",
                rationale=rationale,
                next_step=(
                    "Shift to a different regime assumption or perform a structural rewrite instead "
                    "of more parameter tuning."
                ),
            )

        if overfit_symptoms:
            return MarketHypothesisAssessment(
                summary=summary,
                status="parameter_misaligned",
                rationale=(
                    "The strategy shows weak generalization or high sensitivity, which suggests "
                    "the current filter levels are too brittle even if the high-level thesis may "
                    "still be directionally correct."
                ),
                next_step=(
                    "Simplify conditions, reduce tuning sensitivity, and prefer broader structural "
                    "changes over stacking more thresholds."
                ),
            )

        return MarketHypothesisAssessment(
            summary=summary,
            status="partially_validated",
            rationale=(
                "The strategy has enough evidence to keep iterating on the same thesis. Remaining "
                "gaps look more like optimization and robustness work than a broken market belief."
            ),
            next_step=(
                "Keep the thesis, but refine structure, exits, or filters with out-of-sample checks."
            ),
        )
