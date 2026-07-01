"""Tests for future web contracts and view builders."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from alphaevo.models.enums import (
    ChangeType,
    MarketRegime,
    MarketType,
    SamplingMethod,
    StrategyCategory,
    StrategyStatus,
)
from alphaevo.models.execution import (
    BacktestResult,
    BenchmarkComparison,
    EvaluationReport,
    EventContextMetrics,
    OverallMetrics,
    ReflectionResult,
    RegimeHoldoutCase,
    RegimeHoldoutMetrics,
    SampleBatch,
    StrategyChange,
    StressWindowCase,
    StressWindowMetrics,
    WalkForwardProtocol,
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
from alphaevo.orchestrator.evolution import EvolutionResult, EvolutionRound
from alphaevo.orchestrator.pipeline import RunResult
from alphaevo.research_log.logger import ResearchEvent
from alphaevo.web import (
    EvolutionJobRequest,
    RunJobRequest,
    build_evaluation_summary,
    build_evolution_session,
    build_research_feed,
    build_run_summary,
    build_strategy_card,
    default_web_manifest,
)


def _make_strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id="trend_alpha_v3",
            name="Trend Alpha",
            version=3,
            market=MarketType.A_SHARE,
            category=StrategyCategory.TREND,
            tags=["trend", "pullback"],
            status=StrategyStatus.CHAMPION,
            preferred_regime=["trending_up"],
        ),
        description="Trend-following pullback strategy.",
        entry=StrategyEntry(
            conditions=[StrategyCondition(indicator="rsi_14", op="<", value=35)],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.05),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
    )


def _make_evaluation() -> EvaluationReport:
    report = EvaluationReport(
        evaluation_id="eval-1",
        strategy_id="trend_alpha_v3",
        batch_id="batch-1",
        overall=OverallMetrics(
            win_rate=0.58,
            avg_return=0.018,
            total_return=0.21,
            signal_count=42,
            max_drawdown=0.09,
        ),
        confidence_score=0.64,
    )
    report.benchmark = BenchmarkComparison(
        excess_return=0.07,
        random_baseline_beat_fraction=0.81,
    )
    report.event_context = EventContextMetrics(
        total_symbols=4,
        provider_symbols=1,
        mixed_symbols=1,
        proxy_symbols=2,
        provider_coverage=0.5,
        proxy_only_coverage=0.5,
        relevant_indicators=["negative_news_score"],
        source_breakdown={"provider:mock_feed": 1, "provider:mock_feed+proxy": 1, "proxy": 2},
    )
    report.anti_overfit.walk_forward_gap = 0.08
    report.anti_overfit.walk_forward_pass_rate = 0.67
    report.walk_forward_protocol = WalkForwardProtocol(
        requested_folds=3,
        effective_folds=2,
        train_pct=0.7,
        test_pct=0.3,
        pass_gap=0.1,
        min_signals_per_split=5,
    )
    report.regime_holdout = RegimeHoldoutMetrics(
        preferred_regimes=["trending_up"],
        pass_gap=0.1,
        total_cases=2,
        pass_rate=0.5,
        worst_gap=0.18,
        worst_regime=MarketRegime.VOLATILE,
        holdouts=[
            RegimeHoldoutCase(
                regime=MarketRegime.VOLATILE,
                preferred=False,
                in_sample_signal_count=21,
                holdout_signal_count=21,
                in_sample_win_rate=0.62,
                holdout_win_rate=0.44,
                gap=0.18,
            )
        ],
    )
    report.stress_windows = StressWindowMetrics(
        window_days=20,
        top_k=2,
        alpha_pass_threshold=0.0,
        total_windows=2,
        pass_rate=0.5,
        average_alpha=0.03,
        worst_alpha=-0.04,
        windows=[
            StressWindowCase(
                window_num=1,
                start_date=date(2025, 2, 1),
                end_date=date(2025, 2, 20),
                benchmark_return=-0.1,
                benchmark_drawdown=0.12,
                signal_count=3,
                strategy_win_rate=0.67,
                strategy_total_return=-0.14,
                alpha=-0.04,
            )
        ],
    )
    return report


def test_default_web_manifest_has_core_routes() -> None:
    manifest = default_web_manifest()
    paths = {route.path for route in manifest.routes}
    assert "/api/health" in paths
    assert "/api/runs" in paths
    assert "/api/evolutions" in paths
    assert manifest.capabilities.strategy_playground is True


def test_job_requests_accept_stress_window_overrides() -> None:
    run_req = RunJobRequest(
        strategy_id="trend_alpha_v3",
        stress_window_days=15,
        stress_window_top_k=2,
    )
    evolve_req = EvolutionJobRequest(
        strategy_id="trend_alpha_v3",
        stress_window_days=18,
        stress_window_top_k=4,
    )
    assert run_req.stress_window_days == 15
    assert run_req.stress_window_top_k == 2
    assert evolve_req.stress_window_days == 18
    assert evolve_req.stress_window_top_k == 4


def test_build_strategy_card_and_run_summary() -> None:
    strategy = _make_strategy()
    evaluation = _make_evaluation()
    card = build_strategy_card(strategy, latest_evaluation=evaluation)

    assert card.strategy_id == "trend_alpha_v3"
    assert card.family_id == "trend_alpha"
    assert card.latest_evaluation is not None
    assert card.latest_evaluation.confidence_score == 0.64
    assert card.latest_evaluation.research_maturity is not None
    assert card.latest_evaluation.research_maturity.status == "watch"
    assert card.latest_evaluation.research_maturity.next_action.action == "add_baseline"
    assert (
        card.latest_evaluation.research_maturity.next_action.commands[0]
        == "alphaevo run trend_alpha_v3 --samples 120 --sampling strategy_scoped"
    )

    run_result = RunResult(
        strategy=strategy,
        batch=SampleBatch(
            batch_id="batch-1",
            strategy_id="trend_alpha_v3",
            symbols=["000001.SZ", "000002.SZ"],
            date_range=(date(2025, 1, 1), date(2025, 12, 31)),
            sampling_method=SamplingMethod.REGIME_BASED,
            sampling_reason="regime-aware sampling for trending_up market",
        ),
        backtest_result=BacktestResult(
            strategy_id="trend_alpha_v3",
            batch_id="batch-1",
            signals=[],
        ),
        evaluation=evaluation,
        report_path=Path("reports/trend_alpha_v3.md"),
    )
    run_result.batch.market_regimes = []

    summary = build_run_summary(run_result)
    assert summary.strategy_id == "trend_alpha_v3"
    assert summary.sampling_method == SamplingMethod.REGIME_BASED
    assert summary.report_path == "reports/trend_alpha_v3.md"
    assert summary.evaluation.benchmark_excess_return == 0.07
    assert summary.evaluation.event_context_provider_coverage == 0.5
    assert summary.evaluation.event_context_proxy_only_coverage == 0.5
    assert summary.evaluation.event_context_mixed_symbols == 1
    assert summary.evaluation.event_context_is_proxy_dominant is False
    assert summary.evaluation.event_context_has_proxy_caveat is True
    assert summary.evaluation.event_context_relevant_indicators == ["negative_news_score"]
    assert summary.evaluation.event_context_source_breakdown == {
        "provider:mock_feed": 1,
        "provider:mock_feed+proxy": 1,
        "proxy": 2,
    }
    assert summary.evaluation.walk_forward_gap == 0.08
    assert summary.evaluation.walk_forward_pass_rate == 0.67
    assert summary.evaluation.walk_forward_requested_folds == 3
    assert summary.evaluation.walk_forward_effective_folds == 2
    assert summary.evaluation.walk_forward_train_pct == 0.7
    assert summary.evaluation.walk_forward_pass_gap == 0.1
    assert summary.evaluation.regime_holdout_worst_gap == 0.18
    assert summary.evaluation.regime_holdout_pass_rate == 0.5
    assert summary.evaluation.regime_holdout_worst_regime == "volatile"
    assert summary.evaluation.stress_window_pass_rate == 0.5
    assert summary.evaluation.stress_window_average_alpha == 0.03
    assert summary.evaluation.stress_window_worst_alpha == -0.04
    assert summary.evaluation.stress_window_window_days == 20
    assert summary.evaluation.research_maturity is not None
    assert "baseline_protocol" in summary.evaluation.research_maturity.watch_checks
    assert "complexity" not in summary.evaluation.research_maturity.watch_checks


def test_evaluation_summary_exposes_data_quality_next_action() -> None:
    evaluation = _make_evaluation()
    evaluation.event_context = EventContextMetrics(
        total_symbols=4,
        provider_symbols=0,
        proxy_symbols=4,
        provider_coverage=0.0,
        proxy_only_coverage=1.0,
        relevant_indicators=["negative_news_score", "news_sentiment_score"],
        source_breakdown={"proxy": 4},
    )

    summary = build_evaluation_summary(evaluation, strategy=_make_strategy())

    assert summary.research_maturity is not None
    assert summary.research_maturity.status == "fail"
    assert "data_quality" in summary.research_maturity.failed_checks
    assert summary.research_maturity.next_action.action == "repair_data"
    assert summary.research_maturity.next_action.priority == "high"
    assert summary.research_maturity.next_action.commands == [
        "alphaevo run trend_alpha_v3 --sampling strategy_scoped",
        (
            "alphaevo strategy revise trend_alpha_v3 "
            '"switch to OHLCV-only event logic or remove proxy-dominant conditions"'
        ),
    ]


def test_build_evolution_session_and_research_feed() -> None:
    strategy = _make_strategy()
    evaluation = _make_evaluation()
    reflection = ReflectionResult(
        strategy_id="trend_alpha_v3",
        evaluation_id="eval-1",
        reflection_summary="Tighten entry filters and preserve profit targets.",
    )
    reflection.proposed_changes.append(
        StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=35,
            to_value=30,
            reason="Reduce late entries.",
        )
    )

    evo = EvolutionResult(
        original_strategy_id="trend_alpha_v1",
        champion_id="trend_alpha_v3",
        champion_score=0.64,
        total_rounds=3,
    )
    evo.rounds.append(
        EvolutionRound(
            round_num=1,
            strategy=strategy,
            evaluation=evaluation,
            reflection=reflection,
            improved=True,
            meta_insights=["[change_effectiveness, 50% conf] Tighten filters worked best."],
            experience_lessons=["[worked] Tighter RSI reduced late entries."],
            pattern_context=["[family/entry_combo] Family Entry: RSI + volume confirmation."],
            recommended_method="hybrid",
            recommended_intensity=0.9,
            recommended_max_changes=2,
        )
    )

    session = build_evolution_session(evo)
    assert session.champion_id == "trend_alpha_v3"
    assert session.rounds[0].change_count == 1
    assert session.rounds[0].reflection_summary is not None
    assert session.rounds[0].recommended_method == "hybrid"
    assert session.rounds[0].experience_lessons[0].startswith("[worked]")
    assert session.rounds[0].pattern_context[0].startswith("[family/")

    events = [
        ResearchEvent(
            timestamp=datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc),
            event_type="result",
            content="Confidence improved to 64%.",
            round_num=2,
            strategy_id="trend_alpha_v3",
        ),
        ResearchEvent(
            timestamp=datetime(2026, 4, 6, 9, 0, tzinfo=timezone.utc),
            event_type="hypothesis",
            content="Trend strategy may need tighter RSI.",
            round_num=1,
            strategy_id="trend_alpha_v2",
        ),
    ]
    feed = build_research_feed(events, limit=1)
    assert len(feed) == 1
    assert feed[0].event_type == "result"
