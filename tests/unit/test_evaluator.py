"""Tests for the Evaluator module."""

import math
from datetime import date, timedelta

import pandas as pd

from alphaevo.evaluator.benchmark import BenchmarkComparator
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.models.enums import (
    ExitReason,
    MarketRegime,
    MarketType,
    SignalDirection,
    StrategyCategory,
)
from alphaevo.models.execution import (
    AntiFitMetrics,
    BacktestResult,
    OverallMetrics,
    TradeSignal,
)
from alphaevo.models.market import IndicatorContext
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    TunableParam,
)


def _signal(return_pct: float, holding_days: int = 3) -> TradeSignal:
    return TradeSignal(
        symbol="TEST",
        signal_date=date(2025, 1, 1),
        direction=SignalDirection.LONG,
        entry_price=100.0,
        exit_price=100.0 * (1 + return_pct),
        exit_date=date(2025, 1, 4),
        exit_reason=ExitReason.TAKE_PROFIT if return_pct > 0 else ExitReason.STOP_LOSS,
        return_pct=return_pct,
        holding_days=holding_days,
    )


def _dated_signal(
    return_pct: float,
    signal_date: date,
    holding_days: int = 3,
) -> TradeSignal:
    """Signal with explicit date for chronological split tests."""
    return TradeSignal(
        symbol="TEST",
        signal_date=signal_date,
        direction=SignalDirection.LONG,
        entry_price=100.0,
        exit_price=100.0 * (1 + return_pct),
        exit_date=signal_date,
        exit_reason=ExitReason.TAKE_PROFIT if return_pct > 0 else ExitReason.STOP_LOSS,
        return_pct=return_pct,
        holding_days=holding_days,
    )


def _strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id="eval_test_v1",
            name="Eval Test",
            version=1,
            market=MarketType.A_SHARE,
            category=StrategyCategory.TREND,
        ),
        description="Test",
        entry=StrategyEntry(
            conditions=[StrategyCondition(indicator="rsi_14", op="<", value=30)],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
        ),
    )


def _make_strategy(conditions: int = 1) -> Strategy:
    """Build strategy with a variable number of entry conditions."""
    conds = [
        StrategyCondition(indicator="rsi_14", op="<", value=30 + i)
        for i in range(conditions)
    ]
    return Strategy(
        meta=StrategyMeta(
            id="gate_test_v1",
            name="Gate Test",
            version=1,
            market=MarketType.A_SHARE,
            category=StrategyCategory.TREND,
        ),
        description="Test",
        entry=StrategyEntry(conditions=conds),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
        ),
    )


def _market_data() -> dict[str, pd.DataFrame]:
    base = date(2025, 1, 1)
    base_dates = [base + timedelta(days=i) for i in range(40)]
    return {
        "AAPL": pd.DataFrame(
            {
                "date": base_dates,
                "open": [100 + i for i in range(40)],
                "high": [101 + i for i in range(40)],
                "low": [99 + i for i in range(40)],
                "close": [100.5 + i for i in range(40)],
                "volume": [1_000_000] * 40,
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "date": base_dates,
                "open": [200 + i for i in range(40)],
                "high": [201 + i for i in range(40)],
                "low": [199 + i for i in range(40)],
                "close": [200.5 + i for i in range(40)],
                "volume": [1_200_000] * 40,
            }
        ),
    }


def test_evaluate_fast_keeps_gate_metrics_without_heavy_diagnostics() -> None:
    result = BacktestResult(
        strategy_id="eval_test_v1",
        batch_id="batch",
        signals=[
            _signal(0.03),
            _signal(-0.01),
            _signal(0.02),
            _signal(-0.02),
        ],
    )

    report = Evaluator().evaluate_fast(result, _strategy())

    assert report.overall.signal_count == 4
    assert report.overall.win_rate == 0.5
    assert report.confidence_score > 0
    assert report.benchmark is None
    assert report.walk_forward == []
    assert report.cpcv is None


class TestEvaluatorMetrics:
    def setup_method(self) -> None:
        self.evaluator = Evaluator()

    def test_empty_signals(self) -> None:
        m = self.evaluator.compute_metrics([])
        assert m.win_rate == 0.0
        assert m.signal_count == 0

    def test_all_winners(self) -> None:
        signals = [_signal(0.05), _signal(0.03), _signal(0.02)]
        m = self.evaluator.compute_metrics(signals)
        assert m.win_rate == 1.0
        assert m.avg_return > 0
        assert m.max_consecutive_loss == 0

    def test_all_losers(self) -> None:
        signals = [_signal(-0.03), _signal(-0.02), _signal(-0.04)]
        m = self.evaluator.compute_metrics(signals)
        assert m.win_rate == 0.0
        assert m.avg_return < 0
        assert m.max_consecutive_loss == 3

    def test_mixed_signals(self) -> None:
        signals = [
            _signal(0.05),
            _signal(-0.02),
            _signal(0.03),
            _signal(-0.01),
            _signal(0.04),
        ]
        m = self.evaluator.compute_metrics(signals)
        assert 0 < m.win_rate < 1
        assert m.signal_count == 5
        assert m.avg_win_return == 0.04
        assert m.avg_loss_return == -0.015
        assert m.profit_loss_ratio > 0

    def test_max_drawdown(self) -> None:
        # 3 consecutive losses should create drawdown
        signals = [_signal(-0.05), _signal(-0.05), _signal(-0.05)]
        m = self.evaluator.compute_metrics(signals)
        assert m.max_drawdown > 0.10

    def test_sharpe_ratio(self) -> None:
        signals = [_signal(0.05), _signal(0.03), _signal(0.04), _signal(0.02)]
        m = self.evaluator.compute_metrics(signals)
        assert m.sharpe_ratio > 0

    def test_holding_days(self) -> None:
        signals = [_signal(0.02, 3), _signal(0.01, 5), _signal(-0.01, 7)]
        m = self.evaluator.compute_metrics(signals)
        assert m.avg_holding_days == 5.0


class TestConfidenceScore:
    def setup_method(self) -> None:
        self.evaluator = Evaluator()

    def test_perfect_strategy_high_score(self) -> None:
        m = OverallMetrics(
            win_rate=0.70,
            avg_return=0.05,
            profit_loss_ratio=2.5,
            max_drawdown=0.05,
            sharpe_ratio=2.0,
            signal_count=100,
        )
        score = self.evaluator.compute_confidence_score(m, complexity_score=0.0)
        assert score >= 0.7

    def test_terrible_strategy_low_score(self) -> None:
        m = OverallMetrics(
            win_rate=0.20,
            avg_return=-0.02,
            profit_loss_ratio=0.3,
            max_drawdown=0.30,
            sharpe_ratio=-0.5,
            signal_count=50,
        )
        score = self.evaluator.compute_confidence_score(m, complexity_score=0.5)
        assert score < 0.3

    def test_score_bounded_0_1(self) -> None:
        # Extreme positive
        m1 = OverallMetrics(
            win_rate=1.0,
            avg_return=0.50,
            profit_loss_ratio=10.0,
            sharpe_ratio=5.0,
        )
        s1 = self.evaluator.compute_confidence_score(m1, 0.0)
        assert 0.0 <= s1 <= 1.0

        # Extreme negative
        m2 = OverallMetrics(
            win_rate=0.0,
            avg_return=-0.50,
            max_drawdown=0.80,
        )
        s2 = self.evaluator.compute_confidence_score(m2, 1.0)
        assert 0.0 <= s2 <= 1.0

    def test_complexity_penalty(self) -> None:
        m = OverallMetrics(win_rate=0.60, avg_return=0.03)
        s_simple = self.evaluator.compute_confidence_score(m, complexity_score=0.0)
        s_complex = self.evaluator.compute_confidence_score(m, complexity_score=0.8)
        assert s_simple > s_complex

    def test_overfit_penalty(self) -> None:
        m = OverallMetrics(win_rate=0.60, avg_return=0.03)
        anti_fit_ok = AntiFitMetrics(train_val_gap=0.02, val_test_gap=0.02)
        anti_fit_bad = AntiFitMetrics(train_val_gap=0.20, val_test_gap=0.15)

        s_ok = self.evaluator.compute_confidence_score(m, 0.0, anti_fit_ok)
        s_bad = self.evaluator.compute_confidence_score(m, 0.0, anti_fit_bad)
        assert s_ok > s_bad


class TestEvaluateReport:
    def test_full_evaluate(self) -> None:
        signals = [
            _signal(0.05),
            _signal(-0.02),
            _signal(0.03),
            _signal(-0.01),
            _signal(0.04),
            _signal(-0.03),
        ]
        result = BacktestResult(
            strategy_id="test_v1",
            batch_id="b1",
            signals=signals,
            total_signals=6,
            executed_signals=6,
        )
        evaluator = Evaluator()
        report = evaluator.evaluate(result, _strategy())

        assert report.strategy_id == "test_v1"
        assert report.overall.signal_count == 6
        assert report.overall.win_rate > 0
        assert 0.0 <= report.confidence_score <= 1.0
        assert len(report.failure_cases) <= 10

    def test_benchmark_comparator_ignores_nan_returns(self) -> None:
        trades = [_signal(0.03, holding_days=2), _signal(-0.01, holding_days=2)]
        dates = pd.date_range("2025-01-01", periods=20, freq="B")
        market_data = {
            "AAPL": pd.DataFrame(
                {
                    "date": dates,
                    "open": [100 + i for i in range(20)],
                    "high": [101 + i for i in range(20)],
                    "low": [99 + i for i in range(20)],
                    "close": [
                        100.0,
                        101.0,
                        102.0,
                        float("nan"),
                        104.0,
                        105.0,
                        106.0,
                        107.0,
                        108.0,
                        109.0,
                        110.0,
                        111.0,
                        112.0,
                        113.0,
                        114.0,
                        115.0,
                        116.0,
                        117.0,
                        118.0,
                        119.0,
                    ],
                    "volume": [1_000_000] * 20,
                }
            )
        }

        result = BenchmarkComparator(n_random_simulations=10).compare(trades, market_data)

        assert result.random_baseline is not None
        assert math.isfinite(result.buy_hold.benchmark_return)
        assert math.isfinite(result.random_baseline.mean_return)
        assert math.isfinite(result.random_baseline.std_return)

    def test_evaluate_populates_anti_overfit(self) -> None:
        """Anti-overfit metrics should be populated when enough signals exist."""
        # Create 30 signals spread across dates for a meaningful split
        signals = []
        for i in range(30):
            ret = 0.03 if i % 2 == 0 else -0.02
            d = date(2025, 1, 1 + i)
            signals.append(_dated_signal(ret, d))

        result = BacktestResult(
            strategy_id="test_v1",
            batch_id="b1",
            signals=signals,
            total_signals=30,
            executed_signals=30,
        )
        evaluator = Evaluator()
        report = evaluator.evaluate(result, _strategy())

        af = report.anti_overfit
        # Should have real values, not all zeros
        assert af.train_win_rate > 0
        assert af.val_win_rate > 0
        assert af.test_win_rate > 0

    def test_evaluate_populates_contextual_breakdowns(self) -> None:
        base = date(2025, 1, 1)
        signals = []
        for i in range(30):
            signals.append(
                TradeSignal(
                    symbol="AAPL" if i % 2 == 0 else "MSFT",
                    signal_date=base + timedelta(days=i),
                    direction=SignalDirection.LONG,
                    entry_price=100.0,
                    exit_price=103.0 if i % 3 != 0 else 97.0,
                    exit_date=base + timedelta(days=i + 3),
                    exit_reason=ExitReason.TAKE_PROFIT if i % 3 != 0 else ExitReason.STOP_LOSS,
                    return_pct=0.03 if i % 3 != 0 else -0.03,
                    holding_days=3,
                    regime=(MarketRegime.TRENDING_UP if i < 15 else MarketRegime.VOLATILE),
                    sector="Technology" if i % 2 == 0 else "Financial",
                )
            )

        result = BacktestResult(
            strategy_id="test_v1",
            batch_id="b1",
            signals=signals,
            total_signals=30,
            executed_signals=30,
        )
        strategy = _strategy()
        strategy.entry.filters.append(
            StrategyCondition(indicator="negative_news_score", op="<", value=0.4)
        )
        benchmark_df = _market_data()["AAPL"]
        contexts = {
            "AAPL": IndicatorContext(
                benchmark_df=benchmark_df,
                event_context_source="provider",
            ),
            "MSFT": IndicatorContext(
                benchmark_df=benchmark_df,
                event_context_source="proxy",
            ),
        }

        evaluator = Evaluator()
        report = evaluator.evaluate(
            result,
            strategy,
            market_data=_market_data(),
            contexts=contexts,
        )

        assert len(report.by_regime) == 2
        assert "Technology" in report.by_sector
        assert report.benchmark is not None
        assert report.event_context is not None
        assert report.event_context.provider_symbols == 1
        assert report.event_context.proxy_symbols == 1
        assert report.event_context.source_breakdown == {"provider": 1, "proxy": 1}
        assert "negative_news_score" in report.event_context.relevant_indicators
        assert report.walk_forward_protocol is not None
        assert report.walk_forward_protocol.effective_folds >= 1
        assert report.regime_holdout is not None
        assert report.regime_holdout.total_cases >= 2
        assert len(report.top_patterns) > 0

    def test_event_context_proxy_dominant_pattern_prefers_data_quality(self) -> None:
        strategy = _strategy()
        strategy.entry.filters.append(
            StrategyCondition(indicator="negative_news_score", op="<", value=0.4)
        )
        contexts: dict[str, object] = {
            f"SYM{i}": IndicatorContext(event_context_source="proxy") for i in range(8)
        }
        contexts["PROVIDER"] = IndicatorContext(event_context_source="provider:mock_feed")

        event_context = Evaluator().compute_event_context_metrics(strategy, contexts)

        assert event_context is not None
        assert event_context.provider_coverage < 0.30
        assert event_context.proxy_only_coverage > 0.50
        assert event_context.is_proxy_dominant
        assert event_context.source_breakdown == {"proxy": 8, "provider:mock_feed": 1}
        patterns = Evaluator.compute_top_patterns([], {}, event_context=event_context)
        assert any("Data-quality remediation" in pattern for pattern in patterns)


class TestAntiOverfit:
    def setup_method(self) -> None:
        self.evaluator = Evaluator()

    def test_too_few_signals_returns_neutral(self) -> None:
        """With < 15 signals, should return neutral defaults."""
        signals = [_dated_signal(0.05, date(2025, 1, i + 1)) for i in range(10)]
        af = self.evaluator.compute_anti_overfit(signals)
        assert af.yearly_consistency == 0.5
        assert af.train_win_rate == 0.0  # not computed; defaults

    def test_chronological_split_works(self) -> None:
        """Signals from different periods should be split chronologically."""
        # First 60% = all winners, next 20% = all losers, last 20% = all losers
        # This should create a large train-val gap
        signals = []
        for i in range(30):
            if i < 18:
                ret = 0.05  # first 60%: all wins
            else:
                ret = -0.05  # last 40%: all losses
            signals.append(_dated_signal(ret, date(2025, 1, i + 1)))

        af = self.evaluator.compute_anti_overfit(signals)
        assert af.train_win_rate == 1.0  # all wins in train
        assert af.val_win_rate == 0.0  # all losses in val
        assert af.train_val_gap == 1.0
        assert af.is_overfit  # should detect this as overfit

    def test_consistent_strategy_low_gap(self) -> None:
        """Uniformly mixed signals should have small train-val gap."""
        signals = []
        for i in range(30):
            ret = 0.03 if i % 2 == 0 else -0.02  # alternating win/loss
            signals.append(_dated_signal(ret, date(2025, 1, i + 1)))

        af = self.evaluator.compute_anti_overfit(signals)
        # All splits should have ~50% win rate
        assert abs(af.train_win_rate - af.val_win_rate) < 0.2
        assert not af.is_overfit

    def test_yearly_consistency_multiple_years(self) -> None:
        """With multi-year data, yearly consistency should be computed."""
        signals = []
        # Year 1: 60% win rate — 30 signals
        for i in range(30):
            ret = 0.03 if i < 18 else -0.02
            signals.append(_dated_signal(ret, date(2024, 1, i + 1)))
        # Year 2: similar 60% — 30 signals
        for i in range(30):
            ret = 0.03 if i < 18 else -0.02
            signals.append(_dated_signal(ret, date(2025, 1, i + 1)))

        af = self.evaluator.compute_anti_overfit(signals)
        # Consistent across years → high consistency
        assert af.yearly_consistency > 0.7

    def test_yearly_consistency_single_year_neutral(self) -> None:
        """With only one year of data, consistency should be neutral (0.5)."""
        signals = [_dated_signal(0.03, date(2025, 1, i + 1)) for i in range(20)]
        af = self.evaluator.compute_anti_overfit(signals)
        assert af.yearly_consistency == 0.5


class TestParamSensitivity:
    """Tests for param perturbation sensitivity analysis."""

    def setup_method(self) -> None:
        self.evaluator = Evaluator()

    def test_no_tunable_params_returns_zero(self) -> None:
        """Strategy with no tunable params should have 0.0 sensitivity."""
        s = _strategy()
        # No params.tunable → 0.0
        sensitivity = self.evaluator.compute_param_sensitivity(
            s,
            [_signal(0.03)],
            0.5,
            object(),
            {},
            object(),
        )
        assert sensitivity == 0.0

    def test_zero_base_score_returns_zero(self) -> None:
        """Base score 0 → sensitivity 0 (avoid division by zero)."""
        s = _strategy()
        sensitivity = self.evaluator.compute_param_sensitivity(
            s,
            [_signal(0.03)],
            0.0,
            object(),
            {},
            object(),
        )
        assert sensitivity == 0.0

    def test_non_engine_returns_zero(self) -> None:
        """Non-BacktestEngine object → sensitivity 0."""
        s = _strategy()
        s.params.tunable = [
            TunableParam(
                target="entry.conditions[indicator=rsi_14].value",
                range=[20, 40],
                step=2,
            )
        ]
        sensitivity = self.evaluator.compute_param_sensitivity(
            s,
            [_signal(0.03)],
            0.5,
            "not_an_engine",
            {},
            object(),
        )
        assert sensitivity == 0.0

    def test_sparse_signals_skip_param_sensitivity(self) -> None:
        """Too few executed signals should skip expensive perturbation reruns."""
        s = _strategy()
        s.params.tunable = [
            TunableParam(
                target="entry.conditions[indicator=rsi_14].value",
                range=[20, 40],
                step=2,
            )
        ]

        sensitivity = self.evaluator.compute_param_sensitivity(
            s,
            [_signal(0.03) for _ in range(10)],
            0.5,
            object(),
            {},
            object(),
        )

        assert sensitivity == 0.0

    def test_resolve_and_set_tunable_ma_period(self) -> None:
        s = _strategy()
        s.entry.conditions = [StrategyCondition(indicator="close_above_ma60", op="==", value=True)]
        s.exit.take_profit = TakeProfitConfig(type="target_ma", target="ma60")
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=close_above_ma60].indicator",
                    range=[20, 120],
                    step=5,
                ),
                TunableParam(
                    target="exit.take_profit.target",
                    range=[20, 120],
                    step=5,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 60
        assert self.evaluator._resolve_tunable(s, s.params.tunable[1]) == 60

        self.evaluator._set_tunable(s, s.params.tunable[0], 55)
        self.evaluator._set_tunable(s, s.params.tunable[1], 50)

        assert s.entry.conditions[0].indicator == "close_above_ma55"
        assert s.exit.take_profit.target == "ma50"

    def test_resolve_and_set_tunable_entry_triggers_and_guards(self) -> None:
        s = _strategy()
        s.entry.triggers = [StrategyCondition(indicator="rsi_14", op="<", value=30)]
        s.entry.guards = [
            StrategyCondition(indicator="relative_strength_20d", op=">", value=0.08)
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.triggers[indicator=rsi_14].value",
                    range=[20, 40],
                    step=1,
                ),
                TunableParam(
                    target="entry.guards[indicator=relative_strength_20d].indicator",
                    range=[10, 60],
                    step=5,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 30
        assert self.evaluator._resolve_tunable(s, s.params.tunable[1]) == 20

        self.evaluator._set_tunable(s, s.params.tunable[0], 32)
        self.evaluator._set_tunable(s, s.params.tunable[1], 30)

        assert s.entry.triggers[0].value == 32
        assert s.entry.guards[0].indicator == "relative_strength_30d"

    def test_resolve_and_set_tunable_dual_ma_periods(self) -> None:
        s = _strategy()
        s.entry.conditions = [
            StrategyCondition(indicator="ma5_ge_ma10_or_crossing", op="==", value=True)
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast",
                    range=[3, 8],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow",
                    range=[9, 20],
                    step=1,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 5
        assert self.evaluator._resolve_tunable(s, s.params.tunable[1]) == 10

        self.evaluator._set_tunable(s, s.params.tunable[0], 6)
        self.evaluator._set_tunable(s, s.params.tunable[1], 12)

        assert s.entry.conditions[0].indicator == "ma6_ge_ma12_or_crossing"

    def test_resolve_and_set_tunable_window_indicator_periods(self) -> None:
        s = _strategy()
        s.entry.conditions = [
            StrategyCondition(indicator="rsi_14", op="<", value=30),
            StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.2),
            StrategyCondition(indicator="relative_strength_20d", op=">", value=0.05),
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=rsi_14].indicator",
                    range=[7, 21],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=volume_ratio_1d_5d].indicator",
                    range=[3, 20],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=relative_strength_20d].indicator",
                    range=[10, 60],
                    step=5,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 14
        assert self.evaluator._resolve_tunable(s, s.params.tunable[1]) == 5
        assert self.evaluator._resolve_tunable(s, s.params.tunable[2]) == 20

        self.evaluator._set_tunable(s, s.params.tunable[0], 10)
        self.evaluator._set_tunable(s, s.params.tunable[1], 10)
        self.evaluator._set_tunable(s, s.params.tunable[2], 30)

        assert s.entry.conditions[0].indicator == "rsi_10"
        assert s.entry.conditions[1].indicator == "volume_ratio_1d_10d"
        assert s.entry.conditions[2].indicator == "relative_strength_30d"

    def test_resolve_and_set_tunable_atr_and_bollinger_alias_periods(self) -> None:
        s = _strategy()
        s.entry.conditions = [
            StrategyCondition(indicator="atr", op=">", value=0.5),
            StrategyCondition(indicator="bollinger_band_width", op="<", value=0.2),
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=atr].indicator",
                    range=[7, 21],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=bollinger_band_width].indicator",
                    range=[10, 40],
                    step=5,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 14
        assert self.evaluator._resolve_tunable(s, s.params.tunable[1]) == 20

        self.evaluator._set_tunable(s, s.params.tunable[0], 21)
        self.evaluator._set_tunable(s, s.params.tunable[1], 30)

        assert s.entry.conditions[0].indicator == "atr_21"
        assert s.entry.conditions[1].indicator == "bollinger_band_width_30d"

    def test_resolve_and_set_tunable_bollinger_std(self) -> None:
        s = _strategy()
        s.entry.conditions = [
            StrategyCondition(indicator="bollinger_band_width", op="<", value=0.2),
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=bollinger_band_width].indicator.std",
                    range=[1.0, 3.0],
                    step=0.5,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 2.0

        self.evaluator._set_tunable(s, s.params.tunable[0], 1.5)

        assert s.entry.conditions[0].indicator == "bollinger_band_width_20d_std1p5"

    def test_resolve_and_set_tunable_atr_stop_loss_period(self) -> None:
        s = _strategy()
        s.exit.stop_loss = StopLossConfig(type="atr", multiplier=2.0)
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="exit.stop_loss.atr_period",
                    range=[7, 30],
                    step=1,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 14

        self.evaluator._set_tunable(s, s.params.tunable[0], 21)

        assert s.exit.stop_loss.atr_period == 21

    def test_resolve_and_set_tunable_macd_components(self) -> None:
        s = _strategy()
        s.entry.conditions = [
            StrategyCondition(indicator="macd_histogram", op=">", value=0),
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=macd_histogram].indicator.fast",
                    range=[6, 18],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=macd_histogram].indicator.slow",
                    range=[20, 40],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=macd_histogram].indicator.signal",
                    range=[5, 15],
                    step=1,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 12
        assert self.evaluator._resolve_tunable(s, s.params.tunable[1]) == 26
        assert self.evaluator._resolve_tunable(s, s.params.tunable[2]) == 9

        self.evaluator._set_tunable(s, s.params.tunable[0], 10)
        self.evaluator._set_tunable(s, s.params.tunable[1], 30)
        self.evaluator._set_tunable(s, s.params.tunable[2], 7)

        assert s.entry.conditions[0].indicator == "macd_histogram_fast10_slow30_signal7"

    def test_set_tunable_rejects_invalid_dual_ma_period(self) -> None:
        s = _strategy()
        s.entry.conditions = [
            StrategyCondition(indicator="ma5_ge_ma10_or_crossing", op="==", value=True)
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast",
                    range=[3, 8],
                    step=1,
                ),
            ]
        )

        applied = self.evaluator._set_tunable(s, s.params.tunable[0], 10)

        assert applied is False
        assert s.entry.conditions[0].indicator == "ma5_ge_ma10_or_crossing"

    def test_set_tunable_rejects_invalid_macd_fast_period(self) -> None:
        s = _strategy()
        s.entry.conditions = [
            StrategyCondition(indicator="macd_histogram_fast12_slow26_signal9", op=">", value=0)
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=macd_histogram_fast12_slow26_signal9].indicator.fast",
                    range=[6, 18],
                    step=1,
                ),
            ]
        )

        applied = self.evaluator._set_tunable(s, s.params.tunable[0], 30)

        assert applied is False
        assert s.entry.conditions[0].indicator == "macd_histogram_fast12_slow26_signal9"

    def test_set_tunable_rejects_invalid_bollinger_std(self) -> None:
        s = _strategy()
        s.entry.conditions = [
            StrategyCondition(indicator="bollinger_band_width_30d", op="<", value=0.2)
        ]
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=bollinger_band_width_30d].indicator.std",
                    range=[1.0, 3.0],
                    step=0.5,
                ),
            ]
        )

        applied = self.evaluator._set_tunable(s, s.params.tunable[0], -1.0)

        assert applied is False
        assert s.entry.conditions[0].indicator == "bollinger_band_width_30d"

    def test_resolve_and_set_tunable_trailing_tp_params(self) -> None:
        """trigger_pct and trail_pct should be resolvable and settable."""
        s = _strategy()
        s.exit.take_profit = TakeProfitConfig(
            type="trailing", trigger_pct=0.08, trail_pct=0.04,
        )
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="exit.take_profit.trigger_pct",
                    range=[0.05, 0.15],
                    step=0.01,
                ),
                TunableParam(
                    target="exit.take_profit.trail_pct",
                    range=[0.02, 0.08],
                    step=0.01,
                ),
            ]
        )

        assert self.evaluator._resolve_tunable(s, s.params.tunable[0]) == 0.08
        assert self.evaluator._resolve_tunable(s, s.params.tunable[1]) == 0.04

        self.evaluator._set_tunable(s, s.params.tunable[0], 0.10)
        self.evaluator._set_tunable(s, s.params.tunable[1], 0.06)

        assert s.exit.take_profit.trigger_pct == 0.10
        assert s.exit.take_profit.trail_pct == 0.06


class TestConfidenceScoreSignalGate:
    """Verify that confidence_score enforces hard caps for small samples."""

    def setup_method(self) -> None:
        self.evaluator = Evaluator()

    def _make_metrics(self, signal_count: int, win_rate: float = 0.65) -> OverallMetrics:
        return OverallMetrics(
            signal_count=signal_count,
            win_rate=win_rate,
            avg_return=0.03,
            median_return=0.02,
            profit_loss_ratio=2.0,
            max_drawdown=0.10,
            sharpe_ratio=1.5,
            avg_holding_days=5.0,
            max_consecutive_loss=2,
            total_return=0.30,
        )

    def test_very_few_signals_capped_at_015(self) -> None:
        """Strategies with <10 signals must never score above 0.15."""
        metrics = self._make_metrics(signal_count=5, win_rate=0.80)
        score = self.evaluator.compute_confidence_score(metrics)
        assert score <= 0.15

    def test_moderate_signals_not_capped(self) -> None:
        """Strategies with 30+ signals should not be artificially capped."""
        metrics = self._make_metrics(signal_count=40, win_rate=0.65)
        score = self.evaluator.compute_confidence_score(metrics)
        assert score > 0.15
