"""Tests for research maturity diagnostics."""

from alphaevo.evaluator.maturity import build_research_maturity_report
from alphaevo.models.enums import MarketType, StrategyCategory
from alphaevo.models.execution import (
    AntiFitMetrics,
    BenchmarkComparison,
    EvaluationReport,
    EventContextMetrics,
    OverallMetrics,
    WalkForwardFoldMetrics,
)
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)


def _strategy(extra_conditions: int = 0) -> Strategy:
    conditions = [StrategyCondition(indicator="rsi_14", op="<", value=30)]
    conditions.extend(
        StrategyCondition(indicator=f"momentum_{idx + 5}d", op=">", value=0)
        for idx in range(extra_conditions)
    )
    return Strategy(
        meta=StrategyMeta(
            id="maturity_v1",
            name="Maturity Test",
            version=1,
            market=MarketType.A_SHARE,
            category=StrategyCategory.TREND,
        ),
        description="Test",
        entry=StrategyEntry(conditions=conditions),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
        ),
    )


def _report() -> EvaluationReport:
    return EvaluationReport(
        evaluation_id="eval-maturity",
        strategy_id="maturity_v1",
        overall=OverallMetrics(
            signal_count=80,
            win_rate=0.56,
            avg_return=0.018,
            profit_loss_ratio=1.8,
            max_drawdown=0.12,
            total_return=0.22,
        ),
        confidence_score=0.61,
        anti_overfit=AntiFitMetrics(
            train_val_gap=0.04,
            val_test_gap=0.03,
            walk_forward_gap=0.05,
            walk_forward_pass_rate=0.67,
        ),
        benchmark=BenchmarkComparison(
            benchmark_return=0.10,
            strategy_return=0.22,
            excess_return=0.12,
            symbols_used=20,
            random_baseline_mean=0.03,
            random_baseline_beat_fraction=0.72,
        ),
        walk_forward=[
            WalkForwardFoldMetrics(
                fold_num=1,
                train_signal_count=40,
                test_signal_count=20,
                train_win_rate=0.58,
                test_win_rate=0.54,
                gap=0.04,
            )
        ],
    )


def test_maturity_report_passes_for_complete_research_protocol() -> None:
    maturity = build_research_maturity_report(_report(), _strategy())

    assert maturity.status == "pass"
    assert maturity.score == 1.0
    assert maturity.next_action.action == "optimize_strategy"
    assert maturity.next_action.priority == "medium"
    assert {check.check_id for check in maturity.checks} == {
        "sample_evidence",
        "baseline_protocol",
        "robustness_protocol",
        "data_quality",
        "complexity",
        "optimization_readiness",
    }


def test_proxy_dominant_event_context_blocks_maturity() -> None:
    report = _report()
    report.event_context = EventContextMetrics(
        total_symbols=10,
        provider_symbols=1,
        proxy_symbols=9,
        provider_coverage=0.10,
        proxy_only_coverage=0.90,
        relevant_indicators=["negative_news_score"],
    )

    maturity = build_research_maturity_report(report, _strategy())

    assert maturity.status == "fail"
    assert maturity.next_action.action == "repair_data"
    failed = {check.check_id for check in maturity.failed_checks}
    assert "data_quality" in failed
    assert "optimization_readiness" in failed


def test_missing_baseline_is_watch_not_blocker() -> None:
    report = _report()
    report.benchmark = None

    maturity = build_research_maturity_report(report, _strategy())

    assert maturity.status == "watch"
    assert maturity.next_action.action == "add_baseline"
    baseline = next(check for check in maturity.checks if check.check_id == "baseline_protocol")
    assert baseline.status == "watch"
    assert "Add buy-and-hold" in baseline.recommendation


def test_high_complexity_blocks_promotion() -> None:
    maturity = build_research_maturity_report(_report(), _strategy(extra_conditions=12))

    complexity = next(check for check in maturity.checks if check.check_id == "complexity")
    assert maturity.status == "fail"
    assert maturity.next_action.action == "simplify_strategy"
    assert complexity.status == "fail"


def test_sparse_sample_next_action_expands_evidence() -> None:
    report = _report()
    report.overall.signal_count = 8

    maturity = build_research_maturity_report(report, _strategy())

    assert maturity.status == "fail"
    assert maturity.next_action.action == "expand_sample"
    assert "alphaevo run maturity_v1" in maturity.next_action.commands[0]


def test_robustness_gap_next_action_runs_walk_forward() -> None:
    report = _report()
    report.walk_forward = []
    report.anti_overfit.walk_forward_pass_rate = 0.0

    maturity = build_research_maturity_report(report, _strategy())

    assert maturity.status == "watch"
    assert maturity.next_action.action == "run_robustness"
    assert "--wf-folds 5" in maturity.next_action.commands[0]
