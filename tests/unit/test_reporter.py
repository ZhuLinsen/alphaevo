"""Tests for the Reporter module."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from alphaevo.evaluator.reporter import Reporter
from alphaevo.models.enums import (
    ChangeType,
    ExitReason,
    MarketRegime,
    MarketType,
    SamplingMethod,
    SignalDirection,
    StrategyCategory,
)
from alphaevo.models.execution import (
    AntiFitMetrics,
    BenchmarkComparison,
    EvaluationReport,
    EventContextMetrics,
    OverallMetrics,
    ReflectionResult,
    RegimeHoldoutCase,
    RegimeHoldoutMetrics,
    RegimeMetrics,
    SampleBatch,
    SamplingAttempt,
    StrategyChange,
    StressWindowCase,
    StressWindowMetrics,
    TradeSignal,
    WalkForwardFoldMetrics,
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

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_signal(
    symbol: str = "AAPL",
    return_pct: float = -0.05,
    exit_reason: ExitReason = ExitReason.STOP_LOSS,
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        signal_date=date(2025, 3, 10),
        direction=SignalDirection.LONG,
        entry_price=100.0,
        exit_price=100.0 * (1 + return_pct),
        exit_date=date(2025, 3, 15),
        exit_reason=exit_reason,
        return_pct=return_pct,
        holding_days=5,
    )


def _make_strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id="trend_v1",
            name="Trend Pullback",
            version=1,
            market=MarketType.A_SHARE,
            category=StrategyCategory.TREND,
        ),
        description="Buy on pullback to MA10 in a strong uptrend.",
        entry=StrategyEntry(
            conditions=[StrategyCondition(indicator="rsi_14", op="<", value=30)],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
        ),
    )


def _make_report(
    strategy_id: str = "trend_v1",
    win_rate: float = 0.58,
    confidence: float = 0.62,
    with_regime: bool = False,
) -> EvaluationReport:
    overall = OverallMetrics(
        win_rate=win_rate,
        avg_return=0.025,
        median_return=0.018,
        profit_loss_ratio=1.8,
        max_drawdown=0.12,
        sharpe_ratio=1.35,
        signal_count=45,
        avg_holding_days=4.2,
        max_consecutive_loss=3,
        total_return=0.35,
    )
    failure_cases = [
        _make_signal("FAIL1", -0.08, ExitReason.STOP_LOSS),
        _make_signal("FAIL2", -0.06, ExitReason.MAX_HOLD),
    ]
    by_regime: list[RegimeMetrics] = []
    if with_regime:
        by_regime = [
            RegimeMetrics(
                regime=MarketRegime.TRENDING_UP,
                win_rate=0.70,
                avg_return=0.04,
                signal_count=20,
            ),
            RegimeMetrics(
                regime=MarketRegime.VOLATILE,
                win_rate=0.40,
                avg_return=0.005,
                signal_count=15,
            ),
        ]
    return EvaluationReport(
        evaluation_id="eval_001",
        strategy_id=strategy_id,
        batch_id="batch_001",
        overall=overall,
        by_regime=by_regime,
        failure_cases=failure_cases,
        confidence_score=confidence,
        anti_overfit=AntiFitMetrics(
            train_val_gap=0.05,
            val_test_gap=0.03,
            yearly_consistency=0.82,
            param_sensitivity=0.15,
        ),
        created_at=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


# ── JSON tests ────────────────────────────────────────────────────────


class TestToJson:
    def test_produces_valid_json(self) -> None:
        report = _make_report()
        output = Reporter.to_json(report)
        parsed = json.loads(output)
        assert parsed["strategy_id"] == "trend_v1"
        assert parsed["overall"]["win_rate"] == 0.58
        assert parsed["confidence_score"] == 0.62

    def test_includes_strategy_info(self) -> None:
        report = _make_report()
        strategy = _make_strategy()
        output = Reporter.to_json(report, strategy)
        parsed = json.loads(output)
        assert "strategy" in parsed
        assert parsed["strategy"]["name"] == "Trend Pullback"
        assert parsed["strategy"]["version"] == 1
        assert parsed["strategy"]["category"] == "trend"
        assert "pullback" in parsed["strategy"]["description"].lower()

    def test_without_strategy(self) -> None:
        report = _make_report()
        output = Reporter.to_json(report)
        parsed = json.loads(output)
        assert "strategy" not in parsed

    def test_failure_cases_serialized(self) -> None:
        report = _make_report()
        parsed = json.loads(Reporter.to_json(report))
        assert len(parsed["failure_cases"]) == 2
        assert parsed["failure_cases"][0]["symbol"] == "FAIL1"


# ── Markdown tests ────────────────────────────────────────────────────


class TestToMarkdown:
    def test_contains_key_sections(self) -> None:
        report = _make_report()
        strategy = _make_strategy()
        md = Reporter.to_markdown(report, strategy)

        assert "# Strategy Evaluation Report: Trend Pullback" in md
        assert "## Description" in md
        assert "## Strategy Hypothesis" in md
        assert "## Performance Summary" in md
        assert "## Top Failure Cases" in md
        assert "## Anti-Overfitting Check" in md
        assert "Generated by AlphaEvo" in md
        assert "not investment advice" in md

    def test_metrics_formatted(self) -> None:
        report = _make_report()
        md = Reporter.to_markdown(report)

        assert "58.0%" in md  # win rate
        assert "2.50%" in md  # avg return
        assert "1.80" in md  # P/L ratio
        assert "12.0%" in md  # max drawdown
        assert "1.35" in md  # sharpe
        assert "45" in md  # signal count
        assert "35.00%" in md  # total return

    def test_confidence_score_displayed(self) -> None:
        report = _make_report(confidence=0.62)
        md = Reporter.to_markdown(report)
        assert "62.0%" in md

    def test_failure_cases_listed(self) -> None:
        report = _make_report()
        md = Reporter.to_markdown(report)
        assert "FAIL1" in md
        assert "FAIL2" in md
        assert "stop_loss" in md
        assert "max_hold" in md

    def test_top_patterns_rendered(self) -> None:
        report = _make_report()
        report.top_patterns = [
            "[family/entry_combo] Family Entry: Tighten momentum filters (score 66.0%)",
            "[library/exit_config] ATR Exit: ATR exits reduce drawdown (score 61.0%)",
        ]
        md = Reporter.to_markdown(report)
        assert "## Reusable Research Patterns" in md
        assert "Family Entry" in md
        assert "ATR Exit" in md

    def test_research_maturity_checklist_rendered(self) -> None:
        report = _make_report()
        strategy = _make_strategy()

        md = Reporter.to_markdown(report, strategy)

        assert "## Research Maturity Checklist" in md
        assert "Sample Evidence" in md
        assert "Baseline Protocol" in md
        assert "Optimization Readiness" in md
        assert "Recommended Action" in md
        assert "Suggested Commands" in md

    def test_event_context_rendered(self) -> None:
        report = _make_report()
        report.event_context = EventContextMetrics(
            total_symbols=4,
            provider_symbols=1,
            mixed_symbols=1,
            proxy_symbols=2,
            provider_coverage=0.5,
            proxy_only_coverage=0.5,
            relevant_indicators=["negative_news_score", "days_since_event"],
            source_breakdown={
                "provider:mock_feed": 1,
                "provider:mock_feed+proxy": 1,
                "proxy": 2,
            },
        )
        md = Reporter.to_markdown(report)
        assert "## Event/News Context" in md
        assert "Provider Coverage: 50.0% (2/4 symbols)" in md
        assert "negative_news_score, days_since_event" in md
        assert "provider:mock_feed x 1" in md
        assert "Caveat: some event/news values are price/volume proxy-derived" in md

    def test_proxy_dominant_event_context_warns_before_mutation(self) -> None:
        report = _make_report()
        report.event_context = EventContextMetrics(
            total_symbols=10,
            provider_symbols=1,
            proxy_symbols=9,
            provider_coverage=0.1,
            proxy_only_coverage=0.9,
            relevant_indicators=["negative_news_score"],
            source_breakdown={"provider:mock_feed": 1, "proxy": 9},
        )

        md = Reporter.to_markdown(report)

        assert "Provider Coverage: 10.0% (1/10 symbols)" in md
        assert "Proxy-Only Coverage: 90.0% (9/10 symbols)" in md
        assert "Data Quality Caveat" in md
        assert "before mutating or optimizing these filters" in md

    def test_regime_holdout_rendered(self) -> None:
        report = _make_report(with_regime=True)
        report.regime_holdout = RegimeHoldoutMetrics(
            preferred_regimes=["trending_up"],
            pass_gap=0.1,
            total_cases=2,
            pass_rate=0.5,
            worst_gap=0.3,
            worst_regime=MarketRegime.VOLATILE,
            holdouts=[
                RegimeHoldoutCase(
                    regime=MarketRegime.TRENDING_UP,
                    preferred=True,
                    in_sample_signal_count=15,
                    holdout_signal_count=20,
                    in_sample_win_rate=0.4,
                    holdout_win_rate=0.7,
                    gap=0.3,
                ),
                RegimeHoldoutCase(
                    regime=MarketRegime.VOLATILE,
                    preferred=False,
                    in_sample_signal_count=20,
                    holdout_signal_count=15,
                    in_sample_win_rate=0.7,
                    holdout_win_rate=0.4,
                    gap=0.3,
                ),
            ],
        )
        md = Reporter.to_markdown(report)
        assert "## Regime Holdout" in md
        assert "Preferred Regimes: trending_up" in md
        assert "Worst Holdout Gap: 30.0% (volatile)" in md
        assert "| volatile | no | 70.0% | 40.0% | 30.0% | 20 | 15 |" in md

    def test_stress_windows_rendered(self) -> None:
        report = _make_report()
        report.stress_windows = StressWindowMetrics(
            window_days=20,
            top_k=2,
            alpha_pass_threshold=0.0,
            total_windows=2,
            pass_rate=0.5,
            average_alpha=0.04,
            worst_alpha=-0.03,
            windows=[
                StressWindowCase(
                    window_num=1,
                    start_date=date(2025, 2, 1),
                    end_date=date(2025, 2, 20),
                    benchmark_return=-0.12,
                    benchmark_drawdown=0.15,
                    signal_count=4,
                    strategy_win_rate=0.5,
                    strategy_total_return=-0.15,
                    alpha=-0.03,
                )
            ],
        )
        md = Reporter.to_markdown(report)
        assert "## Stress-Window Benchmark" in md
        assert "Window Size: 20 trading days" in md
        assert "Worst Alpha: -3.00%" in md
        assert (
            "| 1 | 2025-02-01 → 2025-02-20 | -12.00% | 15.00% | 4 | 50.0% | -15.00% | -3.00% |"
            in md
        )

    def test_anti_overfit_pass(self) -> None:
        report = _make_report()
        report.walk_forward = [
            WalkForwardFoldMetrics(
                fold_num=1,
                train_signal_count=10,
                test_signal_count=5,
                train_win_rate=0.6,
                test_win_rate=0.6,
                gap=0.0,
            )
        ]
        report.walk_forward_protocol = WalkForwardProtocol(
            requested_folds=3,
            effective_folds=1,
            train_pct=0.7,
            test_pct=0.3,
            pass_gap=0.1,
            min_signals_per_split=5,
        )
        report.anti_overfit.walk_forward_gap = 0.0
        report.anti_overfit.walk_forward_pass_rate = 1.0
        md = Reporter.to_markdown(report)
        assert "Train-Val Gap: 5.0% ✅" in md
        assert "Yearly Consistency: 82.0%" in md
        assert "Param Sensitivity: 15.0%" in md
        assert "## Walk-Forward Validation" in md
        assert "Protocol: requested 3 folds, effective 1 folds" in md
        assert "Train/Test Split: 70%/30%" in md
        assert "Pass Rate (gap <= 10.0%): 100.0%" in md

    def test_anti_overfit_warn(self) -> None:
        report = _make_report()
        report.anti_overfit.train_val_gap = 0.20
        report.anti_overfit.param_sensitivity = 0.35
        md = Reporter.to_markdown(report)
        assert "⚠️" in md
        assert "Potential overfitting detected" in md

    def test_without_strategy_uses_id(self) -> None:
        report = _make_report()
        md = Reporter.to_markdown(report)
        assert "# Strategy Evaluation Report: trend_v1" in md

    def test_regime_breakdown_included(self) -> None:
        report = _make_report(with_regime=True)
        md = Reporter.to_markdown(report)
        assert "## Performance by Market Regime" in md
        assert "trending_up" in md
        assert "volatile" in md

    def test_no_regime_section_when_empty(self) -> None:
        report = _make_report(with_regime=False)
        md = Reporter.to_markdown(report)
        assert "## Performance by Market Regime" not in md


# ── File output tests ─────────────────────────────────────────────────


class TestToFile:
    def test_write_markdown(self, tmp_path: Path) -> None:
        report = _make_report()
        strategy = _make_strategy()
        dest = tmp_path / "reports" / "eval.md"

        Reporter.to_file(report, dest, format="markdown", strategy=strategy)

        assert dest.exists()
        content = dest.read_text(encoding="utf-8")
        assert "# Strategy Evaluation Report" in content
        assert "Trend Pullback" in content

    def test_write_json(self, tmp_path: Path) -> None:
        report = _make_report()
        dest = tmp_path / "eval.json"

        Reporter.to_file(report, dest, format="json")

        assert dest.exists()
        parsed = json.loads(dest.read_text(encoding="utf-8"))
        assert parsed["strategy_id"] == "trend_v1"
        assert parsed["overall"]["win_rate"] == 0.58

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        report = _make_report()
        dest = tmp_path / "a" / "b" / "c" / "report.md"

        Reporter.to_file(report, dest)

        assert dest.exists()


# ── Summary table tests ───────────────────────────────────────────────


class TestSummaryTable:
    def test_multiple_strategies(self) -> None:
        reports = [
            ("Trend Pullback", _make_report("t1", win_rate=0.58, confidence=0.62)),
            ("Mean Reversion", _make_report("t2", win_rate=0.52, confidence=0.55)),
            ("Breakout", _make_report("t3", win_rate=0.61, confidence=0.70)),
        ]
        table = Reporter.summary_table(reports)

        assert "Strategy" in table
        assert "WinRate" in table
        assert "Score" in table
        # Sorted by confidence desc: Breakout first
        lines = table.strip().split("\n")
        data_lines = [ln for ln in lines if "---" not in ln and "Strategy" not in ln]
        assert "Breakout" in data_lines[0]
        assert "Mean Reversion" in data_lines[-1]

    def test_single_strategy(self) -> None:
        reports = [("Solo", _make_report())]
        table = Reporter.summary_table(reports)
        assert "Solo" in table
        assert table.count("\n") >= 2  # header + sep + 1 row

    def test_empty_reports(self) -> None:
        table = Reporter.summary_table([])
        assert "no strategies" in table.lower()

    def test_long_name_truncated(self) -> None:
        long_name = "A" * 50
        reports = [(long_name, _make_report())]
        table = Reporter.summary_table(reports)
        # Should be truncated with ellipsis
        assert "…" in table
        # Name column should not exceed 24 chars
        for line in table.split("\n")[2:]:  # skip header + sep
            if line.strip():
                first_col = line.split("|")[1].strip()
                assert len(first_col) <= 24


# ── Benchmark section tests ───────────────────────────────────────────


class TestBenchmarkInMarkdown:
    def test_benchmark_section_present(self) -> None:
        report = _make_report()
        report.benchmark = BenchmarkComparison(
            benchmark_return=0.10,
            strategy_return=0.35,
            excess_return=0.25,
            benchmark_max_drawdown=0.15,
            benchmark_sharpe=0.80,
            symbols_used=10,
        )
        md = Reporter.to_markdown(report)
        assert "## Benchmark Comparison (Buy & Hold)" in md
        assert "25.00%" in md  # excess return
        assert "🟢" in md  # beats benchmark
        assert "10 symbols" in md

    def test_benchmark_negative_alpha(self) -> None:
        report = _make_report()
        report.benchmark = BenchmarkComparison(
            benchmark_return=0.20,
            strategy_return=0.10,
            excess_return=-0.10,
            symbols_used=5,
        )
        md = Reporter.to_markdown(report)
        assert "🔴" in md

    def test_no_benchmark_no_section(self) -> None:
        report = _make_report()
        md = Reporter.to_markdown(report)
        assert "Benchmark Comparison" not in md


# ── Evolution report tests ────────────────────────────────────────────


def _make_evolution_result():
    """Build a minimal EvolutionResult for testing."""
    from alphaevo.orchestrator.evolution import EvolutionResult, EvolutionRound

    rounds = [
        EvolutionRound(
            round_num=0,
            strategy=_make_strategy(),
            evaluation=_make_report("trend_v1", win_rate=0.42, confidence=0.30),
        ),
        EvolutionRound(
            round_num=1,
            strategy=_make_strategy(),
            evaluation=_make_report("trend_v2", win_rate=0.55, confidence=0.50),
            batch=SampleBatch(
                batch_id="batch_001",
                strategy_id="trend_v2",
                symbols=["AAPL", "MSFT", "NVDA"],
                date_range=(date(2024, 1, 1), date(2025, 1, 1)),
                sampling_method=SamplingMethod.STRATEGY_SCOPED,
                sampling_reason="filtered by strategy universe constraints",
                requested_max_symbols=60,
                sampling_attempt=2,
                signal_count_target=30,
                signal_count_reached=35,
                sampling_history=[
                    SamplingAttempt(
                        attempt_num=1,
                        max_symbols=60,
                        date_range=(date(2024, 6, 1), date(2025, 1, 1)),
                        sampling_method=SamplingMethod.REPRESENTATIVE,
                        sampling_reason="stratified sampling across sectors",
                        selected_symbols=20,
                        signal_count=12,
                        accepted=False,
                        note="initial sampling plan",
                    ),
                    SamplingAttempt(
                        attempt_num=2,
                        max_symbols=80,
                        date_range=(date(2024, 1, 1), date(2025, 1, 1)),
                        sampling_method=SamplingMethod.STRATEGY_SCOPED,
                        sampling_reason="filtered by strategy universe constraints",
                        selected_symbols=24,
                        signal_count=35,
                        accepted=True,
                        note="expanded date range and symbol budget from representative",
                    ),
                ],
            ),
            reflection=ReflectionResult(
                strategy_id="trend_v1",
                evaluation_id="eval_001",
                failure_patterns=["Late entries due to RSI threshold too high"],
                proposed_changes=[
                    StrategyChange(
                        change_type=ChangeType.TIGHTEN_FILTER,
                        target="entry.conditions[rsi_14].value",
                        from_value=70,
                        to_value=65,
                        reason="Tighter RSI to reduce late entries",
                    )
                ],
                reflection_summary="Tighten RSI and preserve existing exit structure.",
                llm_telemetry={
                    "path": "single_step_fallback",
                    "fallback_trigger": "Request timed out after 45s",
                    "total_duration_ms": 1820,
                    "calls": [
                        {
                            "stage": "diagnosis",
                            "model": "test-reflect-model",
                            "timeout_seconds": 45,
                            "max_retries": 0,
                            "duration_ms": 900,
                            "success": False,
                            "error": "Request timed out after 45s",
                        },
                        {
                            "stage": "single_step",
                            "model": "test-reflect-model",
                            "timeout_seconds": 60,
                            "max_retries": 0,
                            "duration_ms": 920,
                            "success": True,
                            "error": "",
                        },
                    ],
                },
            ),
            improved=True,
            meta_insights=[
                "[change_effectiveness, 60% conf] Best: tighten_filter (67% success, n=6)"
            ],
            experience_lessons=[
                "[worked] Tightening RSI reduced late entries and improved win rate."
            ],
            pattern_context=[
                "[family/entry_combo] Family Entry: Volume-backed pullback entry (score 64.0%, used 3x, success 67%)"
            ],
            cross_strategy_memory=[
                "Across 7 prior cross-strategy experiments from 2 other family/families, `loosen_filter` on `entry.conditions[indicator=volume_ratio_1d_5d].value` succeeded 71% (5/7); recently seen in `ma_crossover_v3, trend_pullback_v2`.",
                "Borrowed [entry_combo] from `ma_crossover_v3`: Entry from MA crossover champion; used 4x with 75% success.",
            ],
            recommended_method="hybrid",
            recommended_intensity=0.8,
            recommended_max_changes=2,
        ),
    ]

    return EvolutionResult(
        original_strategy_id="trend_v1",
        rounds=rounds,
        champion_id="trend_v2",
        champion_score=0.50,
        total_rounds=2,
    )


class TestEvolutionReport:
    def test_contains_key_sections(self) -> None:
        result = _make_evolution_result()
        md = Reporter.evolution_report(result)
        assert "# 🧬 Evolution Report" in md
        assert "## Hypothesis Lens" in md
        assert "## Score Progression" in md
        assert "## Sampling Adequacy" in md
        assert "## Self-Evolution Signals" in md
        assert "## Collective Memory" in md
        assert "## Round Details" in md
        assert "## Summary" in md
        assert "Generated by AlphaEvo" in md

    def test_score_progression_table(self) -> None:
        result = _make_evolution_result()
        md = Reporter.evolution_report(result)
        assert "trend_v1" in md
        assert "trend_v2" in md
        assert "30.0%" in md
        assert "50.0%" in md

    def test_changes_displayed(self) -> None:
        result = _make_evolution_result()
        md = Reporter.evolution_report(result)
        assert "entry.conditions[rsi_14].value" in md
        assert "70" in md
        assert "65" in md
        assert "Tighter RSI" in md

    def test_failure_patterns_displayed(self) -> None:
        result = _make_evolution_result()
        md = Reporter.evolution_report(result)
        assert "Late entries" in md

    def test_meta_and_experience_context_displayed(self) -> None:
        result = _make_evolution_result()
        md = Reporter.evolution_report(result)
        assert "Sampling Context" in md
        assert "Attempt 2" in md
        assert "Meta Recommendation" in md
        assert "Meta-Learning Context" in md
        assert "Experience Lessons Reused" in md
        assert "Reusable Patterns Consulted" in md
        assert "Cross-Strategy Memory Applied" in md
        assert "Tightening RSI reduced late entries" in md
        assert "Family Entry" in md
        assert "Across 7 prior cross-strategy experiments" in md

    def test_research_report_combines_story_and_llm_evidence(self) -> None:
        result = _make_evolution_result()
        md = Reporter.research_report(result)
        assert "# 🧬 Evolution Report" in md
        assert "## LLM Evidence Appendix" in md
        assert "# LLM Evidence Report" in md

    def test_improvement_summary(self) -> None:
        result = _make_evolution_result()
        md = Reporter.evolution_report(result)
        assert "Starting score" in md
        assert "+20.0%" in md  # improvement

    def test_empty_rounds(self) -> None:
        from alphaevo.orchestrator.evolution import EvolutionResult

        result = EvolutionResult(original_strategy_id="test")
        md = Reporter.evolution_report(result)
        assert "No evolution rounds recorded" in md

    def test_early_stop_noted(self) -> None:
        result = _make_evolution_result()
        result.early_stopped = True
        result.stop_reason = "Overfit detected"
        md = Reporter.evolution_report(result)
        assert "Early stopped" in md
        assert "Overfit detected" in md


class TestLlmEvidenceReport:
    def test_contains_llm_evidence_sections(self) -> None:
        result = _make_evolution_result()
        md = Reporter.llm_evidence_report(result)
        assert "# LLM Evidence Report" in md
        assert "## Summary" in md
        assert "## Round-by-Round LLM Evidence" in md
        assert "Tighten RSI" in md
        assert "Starting Score" in md
        assert "Timeout-like failures" in md
        assert "single_step_fallback" in md

    def test_empty_rounds(self) -> None:
        from alphaevo.orchestrator.evolution import EvolutionResult

        md = Reporter.llm_evidence_report(EvolutionResult(original_strategy_id="test"))
        assert "No evolution rounds recorded" in md
