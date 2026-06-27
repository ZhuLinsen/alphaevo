"""Tests for deterministic research committee verdicts."""

from __future__ import annotations

from pathlib import Path

from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import (
    EvaluationReport,
    EventContextMetrics,
    OverallMetrics,
    StrategyChange,
)
from alphaevo.research_committee import ResearchCommittee
from alphaevo.strategy.dsl.parser import StrategyParser


def test_committee_flags_zero_signal_strategy() -> None:
    strategy = StrategyParser().parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    report = EvaluationReport(
        strategy_id=strategy.meta.id,
        overall=OverallMetrics(signal_count=0),
        confidence_score=0.081,
    )

    verdict = ResearchCommittee().review(
        strategy,
        report,
        data_source="snapshot",
        symbols=["AAPL", "MSFT", "NVDA", "AMD", "TSLA"],
        mutation_plan=[
            StrategyChange(
                change_type=ChangeType.CHANGE_LOGIC,
                target="entry.logic",
                from_value="and",
                to_value="or",
                reason="Unlock more signals",
            )
        ],
    )

    assert verdict.overall_status == "fail"
    assert verdict.failed_count >= 1
    assert verdict.mutation_plan[0].target == "entry.logic"
    assert any(v.analyst == "Technical Analyst" and v.status == "fail" for v in verdict.verdicts)


def test_committee_passes_usable_positive_strategy_shape() -> None:
    strategy = StrategyParser().parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    report = EvaluationReport(
        strategy_id=strategy.meta.id,
        overall=OverallMetrics(
            signal_count=80,
            win_rate=0.55,
            avg_return=0.012,
            profit_loss_ratio=1.5,
            max_drawdown=0.18,
            max_consecutive_loss=3,
        ),
        confidence_score=0.45,
    )

    verdict = ResearchCommittee().review(
        strategy,
        report,
        data_source="snapshot",
        symbols=["AAPL", "MSFT", "NVDA", "AMD", "TSLA"],
    )

    assert verdict.overall_status in {"pass", "watch"}
    assert any(v.analyst == "Technical Analyst" and v.status == "pass" for v in verdict.verdicts)


def test_committee_gates_proxy_dominant_event_filters() -> None:
    strategy = StrategyParser().parse_file(Path("strategies/builtin/trend_pullback_rebound.yaml"))
    report = EvaluationReport(
        strategy_id=strategy.meta.id,
        overall=OverallMetrics(
            signal_count=80,
            win_rate=0.55,
            avg_return=0.012,
            profit_loss_ratio=1.5,
            max_drawdown=0.18,
        ),
        confidence_score=0.45,
        event_context=EventContextMetrics(
            total_symbols=63,
            provider_symbols=2,
            proxy_symbols=61,
            provider_coverage=0.032,
            proxy_only_coverage=0.968,
            relevant_indicators=["negative_news_score"],
        ),
    )

    verdict = ResearchCommittee().review(
        strategy,
        report,
        data_source="yfinance-fallback",
        symbols=[f"SYM{i}" for i in range(63)],
    )

    assert verdict.overall_status == "fail"
    data_quality = next(v for v in verdict.verdicts if v.analyst == "Data Quality Auditor")
    assert data_quality.status == "fail"
    assert any("event_provider_coverage=3.2%" in item for item in data_quality.evidence)
    assert any("OHLCV-only" in item for item in data_quality.recommendations)
