"""Stable request/response contracts reserved for future web entrypoints.

This module deliberately does not depend on FastAPI or any frontend stack.
It provides Pydantic models and thin view-builders that future web services
can reuse while the core orchestrator remains CLI-first.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from alphaevo.evaluator.maturity import (
    MaturityAction,
    MaturityStatus,
    ResearchMaturityReport,
    build_research_maturity_report,
)
from alphaevo.models.enums import EvolutionMethod, SamplingMethod, StrategyCategory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from alphaevo.models.execution import EvaluationReport
    from alphaevo.models.strategy import Strategy
    from alphaevo.orchestrator.evolution import EvolutionResult
    from alphaevo.orchestrator.pipeline import RunResult
    from alphaevo.research_log.logger import ResearchEvent


class WebCapabilityFlags(BaseModel):
    """Feature flags for a future web workspace."""

    strategy_playground: bool = True
    evolution_lab: bool = True
    research_feed: bool = True
    factor_factory: bool = True
    leaderboard: bool = True
    async_jobs: bool = True
    websocket_progress: bool = False


class WebRouteSpec(BaseModel):
    """A future API route exposed to the web frontend."""

    name: str
    path: str
    method: Literal["GET", "POST", "DELETE"]
    purpose: str


class WebManifest(BaseModel):
    """Reserved web manifest for API discovery and feature gating."""

    version: str = "v0"
    capabilities: WebCapabilityFlags = Field(default_factory=WebCapabilityFlags)
    routes: list[WebRouteSpec] = Field(default_factory=list)


class RunJobRequest(BaseModel):
    """Request payload for launching a strategy research run from the web."""

    strategy_id: str
    max_symbols: int = 60
    sampling_method: SamplingMethod | None = None
    walk_forward_folds: int | None = None
    walk_forward_train_pct: float | None = None
    walk_forward_pass_gap: float | None = None
    stress_window_days: int | None = None
    stress_window_top_k: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    save_report: bool = True


class EvolutionJobRequest(BaseModel):
    """Request payload for launching a self-evolution session from the web."""

    strategy_id: str
    rounds: int = 3
    method: EvolutionMethod = EvolutionMethod.HYBRID
    max_symbols: int = 60
    sampling_method: SamplingMethod | None = None
    walk_forward_folds: int | None = None
    walk_forward_train_pct: float | None = None
    walk_forward_pass_gap: float | None = None
    stress_window_days: int | None = None
    stress_window_top_k: int | None = None
    hint: str | None = None
    focus: str | None = None
    avoid: str | None = None


class MaturityCheckView(BaseModel):
    """One web-facing maturity gate result."""

    check_id: str
    title: str
    status: MaturityStatus
    evidence: list[str] = Field(default_factory=list)
    recommendation: str


class MaturityNextActionView(BaseModel):
    """Web-facing recommended action for continuing the research loop."""

    action: MaturityAction
    priority: Literal["high", "medium", "low"]
    title: str
    rationale: str
    commands: list[str] = Field(default_factory=list)


class ResearchMaturitySummaryView(BaseModel):
    """Compact maturity summary for dashboards and strategy detail pages."""

    status: MaturityStatus
    score: float
    checks: list[MaturityCheckView] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    watch_checks: list[str] = Field(default_factory=list)
    next_action: MaturityNextActionView


class EvaluationSummaryView(BaseModel):
    """Compact evaluation summary for cards, tables, and dashboard tiles."""

    confidence_score: float
    win_rate: float
    avg_return: float
    total_return: float
    signal_count: int
    max_drawdown: float
    benchmark_excess_return: float | None = None
    benchmark_beats_random: float | None = None
    event_context_provider_coverage: float | None = None
    event_context_proxy_only_coverage: float | None = None
    event_context_mixed_symbols: int | None = None
    event_context_is_proxy_dominant: bool | None = None
    event_context_has_proxy_caveat: bool | None = None
    event_context_relevant_indicators: list[str] = Field(default_factory=list)
    event_context_source_breakdown: dict[str, int] = Field(default_factory=dict)
    walk_forward_gap: float | None = None
    walk_forward_pass_rate: float | None = None
    walk_forward_requested_folds: int | None = None
    walk_forward_effective_folds: int | None = None
    walk_forward_train_pct: float | None = None
    walk_forward_pass_gap: float | None = None
    regime_holdout_worst_gap: float | None = None
    regime_holdout_pass_rate: float | None = None
    regime_holdout_worst_regime: str | None = None
    stress_window_pass_rate: float | None = None
    stress_window_average_alpha: float | None = None
    stress_window_worst_alpha: float | None = None
    stress_window_window_days: int | None = None
    research_maturity: ResearchMaturitySummaryView | None = None


class StrategyCardView(BaseModel):
    """Web-facing strategy summary card."""

    strategy_id: str
    family_id: str
    name: str
    version: int
    category: StrategyCategory
    market: str
    status: str
    tags: list[str] = Field(default_factory=list)
    preferred_regime: list[str] = Field(default_factory=list)
    complexity_score: float = 0.0
    description: str = ""
    latest_evaluation: EvaluationSummaryView | None = None


class RunSummaryView(BaseModel):
    """Web-facing summary of a completed research run."""

    strategy_id: str
    batch_id: str
    sampling_method: SamplingMethod
    sampling_reason: str
    market_regimes: list[str] = Field(default_factory=list)
    evaluation: EvaluationSummaryView
    report_path: str | None = None


class EvolutionRoundView(BaseModel):
    """A single round in the web evolution timeline."""

    round_num: int
    strategy_id: str
    confidence_score: float
    improved: bool
    change_count: int = 0
    reflection_summary: str | None = None
    meta_insights: list[str] = Field(default_factory=list)
    experience_lessons: list[str] = Field(default_factory=list)
    pattern_context: list[str] = Field(default_factory=list)
    recommended_method: str | None = None
    recommended_intensity: float | None = None
    recommended_max_changes: int | None = None


class EvolutionSessionView(BaseModel):
    """Web-facing summary for an evolution session."""

    original_strategy_id: str
    champion_id: str | None = None
    champion_score: float = 0.0
    improvement: float = 0.0
    total_rounds: int = 0
    early_stopped: bool = False
    stop_reason: str = ""
    rounds: list[EvolutionRoundView] = Field(default_factory=list)


class ResearchEventView(BaseModel):
    """Slim research-event DTO for activity feeds and round timelines."""

    timestamp: datetime
    event_type: str
    content: str
    round_num: int
    strategy_id: str


def default_web_manifest() -> WebManifest:
    """Return the default API surface reserved for the future web UI."""
    return WebManifest(
        routes=[
            WebRouteSpec(
                name="health",
                path="/api/health",
                method="GET",
                purpose="Service health and capability flags.",
            ),
            WebRouteSpec(
                name="strategies_list",
                path="/api/strategies",
                method="GET",
                purpose="List strategy cards for the hub and leaderboard.",
            ),
            WebRouteSpec(
                name="run_strategy",
                path="/api/runs",
                method="POST",
                purpose="Launch a research run and stream progress.",
            ),
            WebRouteSpec(
                name="evolve_strategy",
                path="/api/evolutions",
                method="POST",
                purpose="Launch a multi-round self-evolution session.",
            ),
            WebRouteSpec(
                name="research_feed",
                path="/api/research-feed",
                method="GET",
                purpose="Read structured research events for timeline views.",
            ),
            WebRouteSpec(
                name="factors_list",
                path="/api/factors",
                method="GET",
                purpose="List active Alpha Factory factors for discovery pages.",
            ),
        ]
    )


def build_evaluation_summary(
    report: EvaluationReport,
    *,
    strategy: Strategy | None = None,
) -> EvaluationSummaryView:
    """Convert a full evaluation report into a web-friendly summary."""
    benchmark = report.benchmark
    event_context = report.event_context
    anti_fit = report.anti_overfit
    protocol = report.walk_forward_protocol
    regime_holdout = report.regime_holdout
    stress_windows = report.stress_windows
    maturity = build_research_maturity_report(report, strategy)
    return EvaluationSummaryView(
        confidence_score=report.confidence_score,
        win_rate=report.overall.win_rate,
        avg_return=report.overall.avg_return,
        total_return=report.overall.total_return,
        signal_count=report.overall.signal_count,
        max_drawdown=report.overall.max_drawdown,
        benchmark_excess_return=benchmark.excess_return if benchmark is not None else None,
        benchmark_beats_random=(
            benchmark.random_baseline_beat_fraction if benchmark is not None else None
        ),
        event_context_provider_coverage=(
            event_context.provider_coverage if event_context is not None else None
        ),
        event_context_proxy_only_coverage=(
            event_context.proxy_only_coverage if event_context is not None else None
        ),
        event_context_mixed_symbols=(
            event_context.mixed_symbols if event_context is not None else None
        ),
        event_context_is_proxy_dominant=(
            event_context.is_proxy_dominant if event_context is not None else None
        ),
        event_context_has_proxy_caveat=(
            event_context.has_proxy_caveat if event_context is not None else None
        ),
        event_context_relevant_indicators=(
            list(event_context.relevant_indicators) if event_context is not None else []
        ),
        event_context_source_breakdown=(
            dict(event_context.source_breakdown) if event_context is not None else {}
        ),
        walk_forward_gap=anti_fit.walk_forward_gap,
        walk_forward_pass_rate=anti_fit.walk_forward_pass_rate,
        walk_forward_requested_folds=(protocol.requested_folds if protocol is not None else None),
        walk_forward_effective_folds=(protocol.effective_folds if protocol is not None else None),
        walk_forward_train_pct=protocol.train_pct if protocol is not None else None,
        walk_forward_pass_gap=protocol.pass_gap if protocol is not None else None,
        regime_holdout_worst_gap=(regime_holdout.worst_gap if regime_holdout is not None else None),
        regime_holdout_pass_rate=(regime_holdout.pass_rate if regime_holdout is not None else None),
        regime_holdout_worst_regime=(
            regime_holdout.worst_regime.value
            if regime_holdout is not None and regime_holdout.worst_regime is not None
            else None
        ),
        stress_window_pass_rate=(stress_windows.pass_rate if stress_windows is not None else None),
        stress_window_average_alpha=(
            stress_windows.average_alpha if stress_windows is not None else None
        ),
        stress_window_worst_alpha=(
            stress_windows.worst_alpha if stress_windows is not None else None
        ),
        stress_window_window_days=(
            stress_windows.window_days if stress_windows is not None else None
        ),
        research_maturity=_build_research_maturity_summary(maturity),
    )


def build_strategy_card(
    strategy: Strategy,
    *,
    latest_evaluation: EvaluationReport | None = None,
) -> StrategyCardView:
    """Convert a strategy into a stable web card view."""
    return StrategyCardView(
        strategy_id=strategy.meta.id,
        family_id=strategy.meta.family_id,
        name=strategy.meta.name,
        version=strategy.meta.version,
        category=strategy.meta.category,
        market=strategy.meta.market.value,
        status=strategy.meta.status.value,
        tags=list(strategy.meta.tags),
        preferred_regime=list(strategy.meta.preferred_regime),
        complexity_score=strategy.complexity_score,
        description=strategy.description,
        latest_evaluation=(
            build_evaluation_summary(latest_evaluation, strategy=strategy)
            if latest_evaluation is not None
            else None
        ),
    )


def build_run_summary(run_result: RunResult) -> RunSummaryView:
    """Convert a pipeline result into a dashboard-friendly run summary."""
    return RunSummaryView(
        strategy_id=run_result.strategy.meta.id,
        batch_id=run_result.batch.batch_id,
        sampling_method=run_result.batch.sampling_method,
        sampling_reason=run_result.batch.sampling_reason,
        market_regimes=[regime.value for regime in run_result.batch.market_regimes],
        evaluation=build_evaluation_summary(run_result.evaluation, strategy=run_result.strategy),
        report_path=_path_to_str(run_result.report_path),
    )


def build_evolution_session(result: EvolutionResult) -> EvolutionSessionView:
    """Convert an evolution result into a web timeline session."""
    rounds = [
        EvolutionRoundView(
            round_num=round_result.round_num,
            strategy_id=round_result.strategy.meta.id,
            confidence_score=round_result.evaluation.confidence_score,
            improved=round_result.improved,
            change_count=(
                len(round_result.reflection.proposed_changes)
                if round_result.reflection is not None
                else 0
            ),
            reflection_summary=(
                round_result.reflection.reflection_summary
                if round_result.reflection is not None
                else None
            ),
            meta_insights=list(round_result.meta_insights),
            experience_lessons=list(round_result.experience_lessons),
            pattern_context=list(round_result.pattern_context),
            recommended_method=round_result.recommended_method,
            recommended_intensity=round_result.recommended_intensity,
            recommended_max_changes=round_result.recommended_max_changes,
        )
        for round_result in result.rounds
    ]
    return EvolutionSessionView(
        original_strategy_id=result.original_strategy_id,
        champion_id=result.champion_id,
        champion_score=result.champion_score,
        improvement=result.improvement,
        total_rounds=result.total_rounds,
        early_stopped=result.early_stopped,
        stop_reason=result.stop_reason,
        rounds=rounds,
    )


def build_research_feed(
    events: Sequence[ResearchEvent],
    *,
    limit: int | None = None,
) -> list[ResearchEventView]:
    """Convert research events into a feed ordered from newest to oldest."""
    ordered = sorted(events, key=lambda event: event.timestamp, reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    return [
        ResearchEventView(
            timestamp=event.timestamp,
            event_type=event.event_type,
            content=event.content,
            round_num=event.round_num,
            strategy_id=event.strategy_id,
        )
        for event in ordered
    ]


def _path_to_str(path: Path | None) -> str | None:
    """Serialize optional paths for web responses."""
    if path is None:
        return None
    return str(path)


def _build_research_maturity_summary(
    maturity: ResearchMaturityReport,
) -> ResearchMaturitySummaryView:
    """Convert maturity diagnostics into a stable web-facing DTO."""
    return ResearchMaturitySummaryView(
        status=maturity.status,
        score=maturity.score,
        checks=[
            MaturityCheckView(
                check_id=check.check_id,
                title=check.title,
                status=check.status,
                evidence=list(check.evidence),
                recommendation=check.recommendation,
            )
            for check in maturity.checks
        ],
        failed_checks=[check.check_id for check in maturity.failed_checks],
        watch_checks=[check.check_id for check in maturity.watch_checks],
        next_action=MaturityNextActionView(
            action=maturity.next_action.action,
            priority=maturity.next_action.priority,
            title=maturity.next_action.title,
            rationale=maturity.next_action.rationale,
            commands=list(maturity.next_action.commands),
        ),
    )
