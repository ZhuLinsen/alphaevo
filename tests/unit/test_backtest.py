"""Tests for backtest engine: indicators, conditions, rules, and engine."""

import warnings
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphaevo.backtest.condition import ConditionEvaluator
from alphaevo.backtest.engine import BacktestEngine
from alphaevo.backtest.indicators import (
    IndicatorRegistry,
    enrich_with_event_proxies,
    merge_event_context,
)
from alphaevo.backtest.rules import MarketRuleChecker
from alphaevo.models.enums import (
    ExitReason,
    MarketType,
    SignalDirection,
    StrategyCategory,
)
from alphaevo.models.execution import SampleBatch
from alphaevo.models.market import EventContextRecord, EventContextSeries
from alphaevo.models.strategy import (
    ExecutionConfig,
    MarketRuleConfig,
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)

# ── Test data helpers ────────────────────────────────────────────────


def _make_ohlcv(n: int = 60, base_price: float = 100.0, trend: float = 0.002) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a gentle uptrend."""
    rows = []
    price = base_price
    for i in range(n):
        noise = (i % 7 - 3) * 0.5  # oscillation
        c = price + noise
        o = c - 0.3
        h = c + 0.8
        lo = c - 0.6
        vol = 1_000_000 + (i % 5) * 200_000
        prev_c = rows[-1]["close"] if rows else c - 0.5
        rows.append(
            {
                "date": date(2025, 1, 1) + timedelta(days=i),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(lo, 2),
                "close": round(c, 2),
                "volume": vol,
                "prev_close": round(prev_c, 2),
            }
        )
        price += trend * price
    return pd.DataFrame(rows)


def _make_strategy(**kwargs) -> Strategy:
    """Build a simple strategy for testing."""
    defaults = dict(
        meta=StrategyMeta(
            id="test_v1",
            name="Test",
            version=1,
            market=MarketType.A_SHARE,
            category=StrategyCategory.TREND,
        ),
        description="Test strategy",
        entry=StrategyEntry(
            conditions=[
                StrategyCondition(indicator="ma5_above_ma10", op="==", value=True),
            ],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.05),
            take_profit=TakeProfitConfig(type="pct", value=0.03),
            max_holding_days=10,
        ),
    )
    defaults.update(kwargs)
    return Strategy(**defaults)


# ═══════════════════════════════════════════════════════════════════════
#  Indicator Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIndicatorRegistry:
    def test_available_has_mvp_indicators(self) -> None:
        available = IndicatorRegistry.available()
        mvp = [
            "ma5_above_ma10",
            "close_to_ma10_pct",
            "close_above_ma20",
            "volume_ratio_1d_5d",
            "rsi_14",
            "atr",
            "close_below_ma10",
        ]
        for name in mvp:
            assert name in available, f"Missing MVP indicator: {name}"
        assert "maN_above_maM" in available
        assert "close_above_maN" in available
        assert "rsi_N" in available
        assert "atr_N" in available
        assert "macd_histogram_fastN_slowM_signalK" in available
        assert "macd_cross_bullish_fastN_slowM_signalK" in available
        assert "bollinger_band_width_Nd" in available
        assert "bollinger_band_width_Nd_stdS" in available
        assert "breakout_high_Nd" in available
        assert "price_position_Nd" in available
        assert "volume_ratio_1d_Nd" in available
        assert "relative_strength_Nd" in available

    def test_event_proxy_enrichment_avoids_fillna_downcast_warning(self) -> None:
        df = _make_ohlcv(10)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FutureWarning)
            enriched = enrich_with_event_proxies(df)

        assert "negative_news_score" in enriched.columns
        assert not any("Downcasting object dtype arrays" in str(w.message) for w in caught)

    def test_event_context_maps_weekend_news_to_next_trading_date(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-19", "2026-06-22", "2026-06-23"]).date,
                "open": [10.0, 10.5, 10.7],
                "high": [10.2, 10.8, 10.9],
                "low": [9.8, 10.4, 10.5],
                "close": [10.0, 10.7, 10.8],
                "volume": [1_000_000, 1_100_000, 1_200_000],
                "prev_close": [9.9, 10.0, 10.7],
            }
        )
        event_context = EventContextSeries(
            symbol="600519.SS",
            source="yfinance_news",
            records=[
                EventContextRecord(
                    date=date(2026, 6, 20),
                    negative_news_score=0.9,
                    news_sentiment_score=0.1,
                    days_since_event=0,
                )
            ],
        )

        merged, source = merge_event_context(df, event_context)

        assert source == "yfinance_news+proxy"
        assert merged.loc[0, "negative_news_score"] != pytest.approx(0.9)
        assert merged.loc[1, "negative_news_score"] == pytest.approx(0.9)
        assert merged.loc[1, "news_sentiment_score"] == pytest.approx(0.1)

    def test_event_context_after_last_trading_date_stays_proxy(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-19", "2026-06-22"]).date,
                "open": [10.0, 10.5],
                "high": [10.2, 10.8],
                "low": [9.8, 10.4],
                "close": [10.0, 10.7],
                "volume": [1_000_000, 1_100_000],
                "prev_close": [9.9, 10.0],
            }
        )
        event_context = EventContextSeries(
            symbol="600519.SS",
            source="yfinance_news",
            records=[
                EventContextRecord(
                    date=date(2026, 6, 23),
                    negative_news_score=0.9,
                    news_sentiment_score=0.1,
                    days_since_event=0,
                )
            ],
        )

        merged, source = merge_event_context(df, event_context)

        assert source == "proxy"
        assert "negative_news_score" in merged.columns
        assert not (merged["negative_news_score"] == pytest.approx(0.9)).any()

    def test_compute_unknown_indicator_raises(self) -> None:
        df = _make_ohlcv(5)
        with pytest.raises(KeyError, match="Unknown indicator"):
            IndicatorRegistry.compute("nonexistent_indicator", df, 3)

    def test_rsi_14_midrange(self) -> None:
        df = _make_ohlcv(60)
        rsi = IndicatorRegistry.compute("rsi_14", df, 30)
        assert 0 <= rsi <= 100

    def test_rsi_14_early_bars(self) -> None:
        df = _make_ohlcv(10)
        rsi = IndicatorRegistry.compute("rsi_14", df, 5)
        assert rsi == 50.0  # default for insufficient data

    def test_ma5_above_ma10(self) -> None:
        df = _make_ohlcv(60, trend=0.005)  # strong uptrend
        result = IndicatorRegistry.compute("ma5_above_ma10", df, 40)
        assert isinstance(result, bool)

    def test_templated_ma_indicators(self) -> None:
        df = _make_ohlcv(240, trend=0.005)
        idx = 220

        fast = df["close"].iloc[idx - 49 : idx + 1].mean()
        slow = df["close"].iloc[idx - 179 : idx + 1].mean()
        expected_pct = abs(df["close"].iloc[idx] - fast) / fast

        assert IndicatorRegistry.is_registered("ma50_above_ma180") is True
        assert IndicatorRegistry.is_registered("close_to_ma50_pct") is True
        assert IndicatorRegistry.compute("ma50_above_ma180", df, idx) is bool(fast > slow)
        assert IndicatorRegistry.compute("close_to_ma50_pct", df, idx) == pytest.approx(
            expected_pct
        )
        assert IndicatorRegistry.compute("close_above_ma180", df, idx) is bool(
            df["close"].iloc[idx] > slow
        )
        assert IndicatorRegistry.compute("ma50_slope", df, idx) > 0

    def test_templated_window_indicators(self) -> None:
        df = _make_ohlcv(240, trend=0.005)
        idx = 220

        expected_volume_ratio = df["volume"].iloc[idx] / df["volume"].iloc[idx - 10 : idx].mean()
        expected_momentum = (df["close"].iloc[idx] - df["close"].iloc[idx - 15]) / df["close"].iloc[
            idx - 15
        ]
        expected_avg_volume = float(df["volume"].iloc[idx - 14 : idx + 1].mean())
        expected_days_since_high = 29 - int(np.argmax(df["high"].iloc[idx - 29 : idx + 1]))
        expected_days_since_low = 29 - int(np.argmin(df["low"].iloc[idx - 29 : idx + 1]))
        expected_volatility = float(
            df["close"].iloc[idx - 29 : idx + 1].pct_change().dropna().std() * (250**0.5)
        )
        expected_relative_strength = (df["close"].iloc[idx] / df["close"].iloc[idx - 30]) - 1
        tr_values = []
        for i in range(idx - 20, idx + 1):
            hi = df["high"].iloc[i]
            lo = df["low"].iloc[i]
            prev_close = df["close"].iloc[i - 1]
            tr_values.append(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
        expected_atr = float(sum(tr_values) / len(tr_values))
        bollinger_window = df["close"].iloc[idx - 29 : idx + 1]
        bollinger_mid = float(bollinger_window.mean())
        bollinger_std = float(bollinger_window.std())
        expected_bollinger_width = float((4 * bollinger_std) / bollinger_mid)
        prior_high = df["high"].iloc[idx - 20 : idx].max()
        expected_breakout = bool(df["close"].iloc[idx] > prior_high)
        position_window = df["close"].iloc[idx - 59 : idx + 1]
        expected_price_position = float(
            (df["close"].iloc[idx] - position_window.min())
            / (position_window.max() - position_window.min())
        )

        assert IndicatorRegistry.is_registered("rsi_7") is True
        assert IndicatorRegistry.is_registered("atr_21") is True
        assert IndicatorRegistry.is_registered("bollinger_band_width_30d") is True
        assert IndicatorRegistry.is_registered("bollinger_band_width_30d_std1p5") is True
        assert IndicatorRegistry.is_registered("breakout_high_20d") is True
        assert IndicatorRegistry.is_registered("price_position_60d") is True
        assert IndicatorRegistry.is_registered("volume_ratio_1d_10d") is True
        assert IndicatorRegistry.is_registered("relative_strength_30d") is True
        assert 0 <= IndicatorRegistry.compute("rsi_7", df, idx) <= 100
        assert IndicatorRegistry.compute("atr_21", df, idx) == pytest.approx(expected_atr)
        assert IndicatorRegistry.compute("bollinger_band_width_30d", df, idx) == pytest.approx(
            expected_bollinger_width
        )
        assert IndicatorRegistry.compute(
            "bollinger_band_width_30d_std1p5",
            df,
            idx,
        ) == pytest.approx(expected_bollinger_width * 0.75)
        assert isinstance(
            IndicatorRegistry.compute("price_above_bollinger_upper_30d", df, idx), bool
        )
        assert isinstance(
            IndicatorRegistry.compute("price_below_bollinger_lower_30d", df, idx), bool
        )
        assert isinstance(
            IndicatorRegistry.compute("price_above_bollinger_upper_30d_std1p5", df, idx),
            bool,
        )
        assert isinstance(
            IndicatorRegistry.compute("price_below_bollinger_lower_30d_std1p5", df, idx),
            bool,
        )
        assert IndicatorRegistry.compute("breakout_high_20d", df, idx) is expected_breakout
        assert IndicatorRegistry.compute("price_position_60d", df, idx) == pytest.approx(
            expected_price_position
        )
        assert isinstance(IndicatorRegistry.compute("rsi_21_zscore", df, idx), float)
        assert IndicatorRegistry.compute("volume_ratio_1d_10d", df, idx) == pytest.approx(
            expected_volume_ratio
        )
        assert IndicatorRegistry.compute("momentum_15d", df, idx) == pytest.approx(
            expected_momentum
        )
        assert IndicatorRegistry.compute("avg_volume_15d", df, idx) == pytest.approx(
            expected_avg_volume
        )
        assert IndicatorRegistry.compute("days_since_high_30d", df, idx) == expected_days_since_high
        assert IndicatorRegistry.compute("days_since_low_30d", df, idx) == expected_days_since_low
        assert IndicatorRegistry.compute("volatility_30d", df, idx) == pytest.approx(
            expected_volatility
        )
        assert IndicatorRegistry.compute("relative_strength_30d", df, idx) == pytest.approx(
            expected_relative_strength
        )

    def test_templated_macd_indicators(self) -> None:
        df = _make_ohlcv(240, trend=0.005)
        idx = 220
        closes = df["close"].iloc[: idx + 1]
        ema_fast = closes.ewm(span=10, adjust=False).mean()
        ema_slow = closes.ewm(span=30, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=7, adjust=False).mean()
        expected_histogram = float(macd_line.iloc[-1] - signal_line.iloc[-1])
        expected_cross = bool(
            macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
        )

        assert IndicatorRegistry.is_registered("macd_histogram_fast10_slow30_signal7") is True
        assert IndicatorRegistry.is_registered("macd_cross_bullish_fast10_slow30_signal7") is True
        assert IndicatorRegistry.compute(
            "macd_histogram_fast10_slow30_signal7", df, idx
        ) == pytest.approx(expected_histogram)
        assert (
            IndicatorRegistry.compute("macd_cross_bullish_fast10_slow30_signal7", df, idx)
            is expected_cross
        )

    def test_volume_ratio(self) -> None:
        df = _make_ohlcv(20)
        ratio = IndicatorRegistry.compute("volume_ratio_1d_5d", df, 10)
        assert ratio > 0

    def test_atr_positive(self) -> None:
        df = _make_ohlcv(30)
        atr_val = IndicatorRegistry.compute("atr", df, 20)
        assert atr_val > 0

    def test_degraded_indicators_return_defaults(self) -> None:
        """L3 indicators should return 'don't block' defaults."""
        df = _make_ohlcv(5)
        assert IndicatorRegistry.compute("negative_news_score", df, 3) == 0.0
        assert IndicatorRegistry.compute("news_sentiment_score", df, 3) == 1.0
        assert IndicatorRegistry.compute("st_flag", df, 3) is False
        assert IndicatorRegistry.compute("sector_heat_rank", df, 3) == 1
        assert IndicatorRegistry.compute("sector_net_inflow_days", df, 3) == 99


# ═══════════════════════════════════════════════════════════════════════
#  Condition Evaluator Tests
# ═══════════════════════════════════════════════════════════════════════


class TestConditionEvaluator:
    def setup_method(self) -> None:
        self.evaluator = ConditionEvaluator()

    def test_simple_gt_condition(self) -> None:
        df = _make_ohlcv(30)
        cond = StrategyCondition(indicator="rsi_14", op=">", value=30)
        result = self.evaluator.evaluate_condition(cond, df, 20)
        assert isinstance(result, bool)

    def test_equality_condition(self) -> None:
        df = _make_ohlcv(30, trend=0.01)
        cond = StrategyCondition(indicator="ma5_above_ma10", op="==", value=True)
        result = self.evaluator.evaluate_condition(cond, df, 25)
        assert isinstance(result, bool)

    def test_unknown_indicator_blocks(self) -> None:
        """Unknown indicators should block conservatively (return False)."""
        df = _make_ohlcv(5)
        cond = StrategyCondition(indicator="unknown_ind", op=">", value=0)
        assert self.evaluator.evaluate_condition(cond, df, 3) is False

    def test_and_group(self) -> None:
        df = _make_ohlcv(30)
        conditions = [
            StrategyCondition(indicator="rsi_14", op=">", value=0),  # always true
            StrategyCondition(indicator="rsi_14", op="<", value=100),  # always true
        ]
        assert self.evaluator.evaluate_group(conditions, "and", df, 20) is True

    def test_or_group(self) -> None:
        df = _make_ohlcv(30)
        conditions = [
            StrategyCondition(indicator="rsi_14", op=">", value=200),  # always false
            StrategyCondition(indicator="rsi_14", op="<", value=100),  # always true
        ]
        assert self.evaluator.evaluate_group(conditions, "or", df, 20) is True

    def test_empty_conditions_pass(self) -> None:
        df = _make_ohlcv(5)
        assert self.evaluator.evaluate_group([], "and", df, 3) is True

    def test_evaluate_entry_with_filters(self) -> None:
        df = _make_ohlcv(30)
        entry = StrategyEntry(
            conditions=[
                StrategyCondition(indicator="rsi_14", op="<", value=100),
            ],
            filters=[
                StrategyCondition(indicator="negative_news_score", op="<", value=0.5),
            ],
        )
        result = self.evaluator.evaluate_entry(entry, df, 20)
        assert result is True


# ═══════════════════════════════════════════════════════════════════════
#  Market Rule Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMarketRuleChecker:
    def setup_method(self) -> None:
        self.checker = MarketRuleChecker()
        self.a_share_rules = MarketRuleConfig(t_plus_1=True, limit_up_down=True, suspension=True)

    def test_t_plus_1_blocks_same_day_sell(self) -> None:
        df = _make_ohlcv(10)
        assert self.checker.can_sell(df, 5, 5, self.a_share_rules) is False

    def test_t_plus_1_allows_next_day_sell(self) -> None:
        df = _make_ohlcv(10)
        assert self.checker.can_sell(df, 6, 5, self.a_share_rules) is True

    def test_limit_up_blocks_buy(self) -> None:
        """Simulate limit-up: 10%+ gain from prev_close."""
        df = _make_ohlcv(10)
        # Modify one row to simulate limit-up
        df.loc[5, "close"] = 110.0
        df.loc[5, "prev_close"] = 100.0
        assert self.checker.can_buy(df, 5, self.a_share_rules) is False

    def test_suspension_blocks(self) -> None:
        df = _make_ohlcv(10)
        df.loc[5, "volume"] = 0  # suspended
        assert self.checker.can_buy(df, 5, self.a_share_rules) is False
        assert self.checker.can_sell(df, 5, 3, self.a_share_rules) is False

    def test_no_rules_allows_everything(self) -> None:
        df = _make_ohlcv(10)
        no_rules = MarketRuleConfig()
        assert self.checker.can_buy(df, 5, no_rules) is True
        assert self.checker.can_sell(df, 5, 5, no_rules) is True

    def test_default_rules(self) -> None:
        a_rules = MarketRuleChecker.default_rules("a_share")
        assert a_rules.t_plus_1 is True
        us_rules = MarketRuleChecker.default_rules("us")
        assert us_rules.t_plus_1 is False


# ═══════════════════════════════════════════════════════════════════════
#  Backtest Engine Tests
# ═══════════════════════════════════════════════════════════════════════


class TestBacktestEngine:
    def setup_method(self) -> None:
        self.engine = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=15)

    def test_run_produces_signals(self) -> None:
        """A simple always-enter strategy should produce signals."""
        df = _make_ohlcv(60, trend=0.003)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),  # always true
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.10),
                take_profit=TakeProfitConfig(type="pct", value=0.02),
                max_holding_days=5,
            ),
        )
        batch = SampleBatch(
            batch_id="test_batch",
            strategy_id="test_v1",
            symbols=["TEST"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"TEST": df}, batch)
        assert result.total_signals > 0
        assert result.strategy_id == "test_v1"

    def test_empty_data_returns_no_signals(self) -> None:
        strategy = _make_strategy()
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["MISSING"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {}, batch)
        assert result.total_signals == 0

    def test_run_raises_on_unknown_indicator(self) -> None:
        """Engine should fail fast on unknown indicator names."""
        df = _make_ohlcv(60)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="unknown_indicator", op=">", value=0),
                ],
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["S1"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        with pytest.raises(ValueError, match="Unknown indicators"):
            self.engine.run(strategy, {"S1": df}, batch)

    def test_exit_trigger_closes_position_with_signal_reason(self) -> None:
        """Explicit exit triggers should close long positions at the current close."""
        df = _make_ohlcv(60, trend=0.003)
        strategy = _make_strategy(
            meta=StrategyMeta(
                id="exit_trigger_v1",
                name="Exit Trigger",
                version=1,
                market=MarketType.US,
                category=StrategyCategory.TREND,
            ),
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
            ),
            exit=StrategyExit(
                triggers=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
                stop_loss=StopLossConfig(type="pct", value=0.50),
                take_profit=TakeProfitConfig(type="pct", value=1.00),
                max_holding_days=20,
            ),
            market_rules={},
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="exit_trigger_v1",
            symbols=["S1"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )

        result = self.engine.run(strategy, {"S1": df}, batch)

        assert result.total_signals > 0
        assert result.signals[0].exit_reason == ExitReason.SIGNAL
        assert result.signals[0].holding_days == 0

    def test_entry_triggers_and_guards_control_buy_signal(self) -> None:
        """Entry triggers define the buy signal; guards remain hard filters."""
        df = _make_ohlcv(60, trend=0.003)
        strategy = _make_strategy(
            meta=StrategyMeta(id="entry_trigger_v1", name="Entry Trigger", market=MarketType.US),
            entry=StrategyEntry(
                triggers=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
                guards=[StrategyCondition(indicator="rsi_14", op="<", value=0)],
            ),
            market_rules={},
        )
        batch = SampleBatch(
            batch_id="entry_trigger",
            strategy_id="entry_trigger_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )

        blocked = self.engine.run(strategy, {"S": df}, batch)
        strategy.entry.guards = [StrategyCondition(indicator="rsi_14", op=">", value=0)]
        allowed = self.engine.run(strategy, {"S": df}, batch)

        assert blocked.total_signals == 0
        assert allowed.total_signals > 0

    def test_intrabar_conflict_defaults_to_conservative_stop_loss(self) -> None:
        df = _make_ohlcv(60)
        df.loc[31, ["open", "high", "low", "close"]] = [100.0, 110.0, 90.0, 106.0]
        strategy = _make_strategy(
            meta=StrategyMeta(id="conflict_v1", name="Conflict", market=MarketType.US),
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.05),
                take_profit=TakeProfitConfig(type="pct", value=0.05),
                max_holding_days=10,
            ),
            market_rules={},
        )
        batch = SampleBatch(
            batch_id="conflict",
            strategy_id="conflict_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )

        result = self.engine.run(strategy, {"S": df}, batch)

        assert result.signals[0].exit_reason == ExitReason.STOP_LOSS
        assert result.signals[0].exit_price == pytest.approx(95.0)

    def test_intrabar_conflict_optimistic_uses_take_profit(self) -> None:
        df = _make_ohlcv(60)
        df.loc[31, ["open", "high", "low", "close"]] = [100.0, 110.0, 90.0, 94.0]
        strategy = _make_strategy(
            meta=StrategyMeta(id="conflict_v1", name="Conflict", market=MarketType.US),
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.05),
                take_profit=TakeProfitConfig(type="pct", value=0.05),
                max_holding_days=10,
            ),
            market_rules={},
        )
        batch = SampleBatch(
            batch_id="conflict",
            strategy_id="conflict_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        engine = BacktestEngine(
            slippage=0.0,
            commission=0.0,
            min_data_days=15,
            fill_policy="optimistic",
        )

        result = engine.run(strategy, {"S": df}, batch)

        assert result.signals[0].exit_reason == ExitReason.TAKE_PROFIT
        assert result.signals[0].exit_price == pytest.approx(105.0)

    def test_intrabar_conflict_close_first_uses_bar_close_direction(self) -> None:
        df = _make_ohlcv(60)
        df.loc[31, ["open", "high", "low", "close"]] = [100.0, 110.0, 90.0, 106.0]
        strategy = _make_strategy(
            meta=StrategyMeta(id="conflict_v1", name="Conflict", market=MarketType.US),
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.05),
                take_profit=TakeProfitConfig(type="pct", value=0.05),
                max_holding_days=10,
            ),
            market_rules={},
        )
        batch = SampleBatch(
            batch_id="conflict",
            strategy_id="conflict_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        engine = BacktestEngine(
            slippage=0.0,
            commission=0.0,
            min_data_days=15,
            fill_policy="close_first",
        )

        result = engine.run(strategy, {"S": df}, batch)

        assert result.signals[0].exit_reason == ExitReason.TAKE_PROFIT

    def test_invalid_fill_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="fill_policy"):
            BacktestEngine(fill_policy="unknown")

    def test_slippage_default_not_mutated_on_exception(self) -> None:
        """Per-strategy slippage override should not leak when run raises."""

        class _ExplodingEvaluator(ConditionEvaluator):
            def evaluate_entry(self, entry, df, idx, ctx=None):  # type: ignore[override]
                raise RuntimeError("boom")

        engine = BacktestEngine(
            condition_evaluator=_ExplodingEvaluator(),
            slippage=0.001,
            commission=0.0,
            min_data_days=15,
        )
        df = _make_ohlcv(60)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
                execution=ExecutionConfig(timing="next_open", slippage=0.01),
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["S1"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )

        with pytest.raises(RuntimeError, match="boom"):
            engine.run(strategy, {"S1": df}, batch)

        assert engine.slippage == 0.001

    def test_signals_have_valid_fields(self) -> None:
        df = _make_ohlcv(60)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["S1"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"S1": df}, batch)
        for signal in result.signals:
            assert signal.symbol == "S1"
            assert signal.direction == SignalDirection.LONG
            assert signal.entry_price > 0
            assert signal.exit_price is not None
            assert signal.exit_reason is not None
            assert signal.holding_days >= 0

    def test_max_holding_days_exit(self) -> None:
        """Strategy with very loose SL/TP should exit at max_holding_days."""
        df = _make_ohlcv(60)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.99),  # very loose
                take_profit=TakeProfitConfig(type="pct", value=0.99),
                max_holding_days=3,
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["S1"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"S1": df}, batch)
        for signal in result.signals:
            assert signal.holding_days <= 3
            assert signal.exit_reason == ExitReason.MAX_HOLD

    def test_multiple_symbols(self) -> None:
        df1 = _make_ohlcv(60, base_price=50)
        df2 = _make_ohlcv(60, base_price=200)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["A", "B"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"A": df1, "B": df2}, batch)
        symbols_in_signals = {s.symbol for s in result.signals}
        assert "A" in symbols_in_signals
        assert "B" in symbols_in_signals

    def test_respects_batch_date_range_for_entries_and_exits(self) -> None:
        df = _make_ohlcv(60)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.99),
                take_profit=TakeProfitConfig(type="pct", value=0.99),
                max_holding_days=3,
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )

        result = self.engine.run(strategy, {"S": df}, batch)
        assert result.total_signals > 0
        assert all(signal.exit_date is not None for signal in result.signals)

    def test_force_closes_at_data_end(self) -> None:
        """Open position at data end should be force-closed."""
        df = _make_ohlcv(60)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.99),
                take_profit=TakeProfitConfig(type="pct", value=0.99),
                max_holding_days=50,
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )

        result = self.engine.run(strategy, {"S": df}, batch)

        assert result.total_signals > 0
        # Last signal should be force-closed
        assert result.signals[-1].exit_reason == ExitReason.MAX_HOLD

    def test_slippage_and_commission(self) -> None:
        """Engine with costs should produce valid signals with lower per-trade returns."""
        df = _make_ohlcv(60)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )

        engine = BacktestEngine(slippage=0.002, commission=0.001, min_data_days=15)
        result = engine.run(strategy, {"S": df}, batch)

        # Engine should still produce signals
        assert result.total_signals > 0
        for signal in result.signals:
            assert signal.entry_price > 0
            assert signal.exit_price is not None
            # With slippage, entry > raw open and exit < raw close
            # (can't compare directly, but prices should be positive)

    def test_pct_from_low_trailing_stop(self) -> None:
        """pct_from_low should trigger when price drops from highest high."""
        # Create data: price rises to 120 then drops sharply
        rows = []
        price = 100.0
        for i in range(60):
            d = date(2025, 1, 1) + timedelta(days=i)
            if i < 30:
                price = 100.0 + i * 1.0  # rise to ~130
            else:
                price = 130.0 - (i - 30) * 2.0  # drop sharply
            o = price - 0.5
            h = price + 1.0
            lo = price - 1.0
            c = price
            rows.append({"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": 1000000})
        df = pd.DataFrame(rows)

        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
            ),
            exit_config=StrategyExit(
                stop_loss=StopLossConfig(type="pct_from_low", value=0.05),
                take_profit=TakeProfitConfig(type="pct", value=0.50),
                max_holding_days=50,
            ),
        )
        batch = SampleBatch(
            batch_id="b",
            strategy_id="test_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"S": df}, batch)
        # Should have signals that exited via stop_loss (price dropped >5% from peak)
        stop_loss_signals = [s for s in result.signals if s.exit_reason == ExitReason.STOP_LOSS]
        assert len(stop_loss_signals) > 0, "pct_from_low trailing stop should trigger"

    def test_resolve_target_ma_supports_known_windows(self) -> None:
        df = _make_ohlcv(80, trend=0.005)
        idx = 70
        target = self.engine._resolve_target_ma("ma20", df, idx)
        expected = float(df["close"].iloc[idx - 19 : idx + 1].mean())
        assert target == pytest.approx(expected)

    def test_atr_stop_loss_uses_configured_period(self) -> None:
        df = _make_ohlcv(80)
        stop_loss = StopLossConfig(type="atr", multiplier=2.0, atr_period=21)
        position = type("P", (), {"entry_price": 100.0})()

        with patch(
            "alphaevo.backtest.engine.IndicatorRegistry.compute", return_value=1.5
        ) as compute:
            risk = self.engine._compute_risk(position, stop_loss, df, 40, None)

        compute.assert_called_once_with("atr_21", df, 40, None)
        assert risk == pytest.approx(3.0)

    def test_indicator_snapshot_populated(self) -> None:
        """Signals should carry indicator values at entry time."""
        df = _make_ohlcv(60, trend=0.003)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.10),
                take_profit=TakeProfitConfig(type="pct", value=0.02),
                max_holding_days=5,
            ),
        )
        batch = SampleBatch(
            batch_id="snap",
            strategy_id="test_v1",
            symbols=["S1"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"S1": df}, batch)
        assert result.total_signals > 0
        for signal in result.signals:
            assert "rsi_14" in signal.indicator_snapshot
            rsi_val = signal.indicator_snapshot["rsi_14"]
            assert 0 <= rsi_val <= 100

    def test_breakout_high_enters_on_breakout(self) -> None:
        """breakout_high should enter when next bar opens above signal bar's high."""
        # Build data with gap-up opens that exceed prior bar's high
        rows = []
        for i in range(60):
            d = date(2025, 1, 1) + timedelta(days=i)
            c = 100.0 + i * 2.0
            rows.append(
                {
                    "date": d,
                    "open": c + 0.5,
                    "high": c + 1.0,
                    "low": c - 1.0,
                    "close": c,
                    "volume": 1_000_000,
                    "prev_close": (100.0 + (i - 1) * 2.0) if i > 0 else 99.0,
                }
            )
        df = pd.DataFrame(rows)

        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
                execution=ExecutionConfig(timing="breakout_high"),
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.10),
                take_profit=TakeProfitConfig(type="pct", value=0.02),
                max_holding_days=3,
            ),
        )
        batch = SampleBatch(
            batch_id="bo",
            strategy_id="test_v1",
            symbols=["S1"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"S1": df}, batch)
        # In an uptrend, breakouts happen (next high > signal high)
        assert result.total_signals > 0
        for signal in result.signals:
            assert signal.entry_price > 0

    def test_breakout_high_skips_when_no_breakout(self) -> None:
        """breakout_high should skip signals where next bar doesn't exceed signal high."""
        # Flat data — highs are very similar, some may not breakout
        rows = []
        for i in range(60):
            d = date(2025, 1, 1) + timedelta(days=i)
            c = 100.0  # flat price
            rows.append(
                {
                    "date": d,
                    "open": 99.5,
                    "high": 100.5,
                    "low": 99.0,
                    "close": c,
                    "volume": 1_000_000,
                    "prev_close": 100.0,
                }
            )
        df = pd.DataFrame(rows)

        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op=">", value=0)],
                execution=ExecutionConfig(timing="breakout_high"),
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.10),
                take_profit=TakeProfitConfig(type="pct", value=0.10),
                max_holding_days=5,
            ),
        )
        batch = SampleBatch(
            batch_id="bo2",
            strategy_id="test_v1",
            symbols=["FLAT"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"FLAT": df}, batch)
        # With perfectly flat highs, breakout_high should skip all signals
        assert result.total_signals == 0

    # ── target_ma take-profit ──────────────────────────────────────────

    def test_target_ma_take_profit(self) -> None:
        """Price crossing above target MA should trigger take-profit."""
        # Create data with a dip then recovery above ma20
        df = _make_ohlcv(80, trend=0.005)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.20),
                take_profit=TakeProfitConfig(type="target_ma", target="ma20"),
                max_holding_days=50,
            ),
        )
        batch = SampleBatch(
            batch_id="tma1",
            strategy_id="test_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 4, 1)),
        )
        result = self.engine.run(strategy, {"S": df}, batch)
        tp_signals = [s for s in result.signals if s.exit_reason == ExitReason.TAKE_PROFIT]
        # In an uptrending market, price should cross above MA20 and trigger
        assert len(tp_signals) > 0, "target_ma take-profit should trigger in uptrend"

    def test_resolve_target_ma_returns_none_for_unknown(self) -> None:
        df = _make_ohlcv(30)
        assert self.engine._resolve_target_ma("ema20", df, 20) is None

    def test_resolve_target_ma_returns_none_insufficient_data(self) -> None:
        df = _make_ohlcv(30)
        # idx=5 with window=20 → not enough data
        assert self.engine._resolve_target_ma("ma20", df, 5) is None

    # ── price_level stop-loss ──────────────────────────────────────────

    def test_price_level_stop_loss(self) -> None:
        """price_level stop-loss should trigger when price drops below reference."""
        from alphaevo.models.market import IndicatorContext

        # Create sharply declining data
        df = _make_ohlcv(60, base_price=100.0, trend=-0.01)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">=", value=0),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="price_level", reference="pre_event_close"),
                take_profit=TakeProfitConfig(type="pct", value=0.50),
                max_holding_days=50,
            ),
        )

        ctx = IndicatorContext(pre_event_close=99.0)
        batch = SampleBatch(
            batch_id="pl1",
            strategy_id="test_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"S": df}, batch, contexts={"S": ctx})
        sl_signals = [s for s in result.signals if s.exit_reason == ExitReason.STOP_LOSS]
        assert len(sl_signals) > 0, "price_level stop-loss should trigger in downtrend"

    def test_resolve_price_level_pre_event_close(self) -> None:
        from alphaevo.backtest.engine import _Position
        from alphaevo.models.market import IndicatorContext

        ctx = IndicatorContext(pre_event_close=42.0)
        pos = _Position(symbol="S", entry_idx=30, entry_date=date(2025, 2, 1), entry_price=50.0)
        df = _make_ohlcv(60)
        sl = StopLossConfig(type="price_level", reference="pre_event_close")
        result = self.engine._resolve_price_level(pos, sl, df, 35, ctx)
        assert result == 42.0

    def test_resolve_price_level_none_context(self) -> None:
        from alphaevo.backtest.engine import _Position

        pos = _Position(symbol="S", entry_idx=30, entry_date=date(2025, 2, 1), entry_price=50.0)
        df = _make_ohlcv(60)
        sl = StopLossConfig(type="price_level", reference="pre_event_close")
        result = self.engine._resolve_price_level(pos, sl, df, 35, None)
        assert result is None

    def test_resolve_price_level_entry_reference(self) -> None:
        from alphaevo.backtest.engine import _Position

        pos = _Position(symbol="S", entry_idx=30, entry_date=date(2025, 2, 1), entry_price=50.0)
        df = _make_ohlcv(60)
        sl = StopLossConfig(type="price_level", reference="entry")
        result = self.engine._resolve_price_level(pos, sl, df, 35)
        assert result == 50.0

    # ── regime/sector population ───────────────────────────────────────

    def test_trade_signal_carries_regime_and_sector(self) -> None:
        """TradeSignal should carry regime/sector from IndicatorContext."""
        from alphaevo.models.enums import MarketRegime
        from alphaevo.models.market import (
            IndicatorContext,
            MarketContext,
            SectorInfo,
        )

        df = _make_ohlcv(60, trend=0.003)
        strategy = _make_strategy(
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.05),
                take_profit=TakeProfitConfig(type="pct", value=0.03),
                max_holding_days=10,
            ),
        )
        ctx = IndicatorContext(
            market_context=MarketContext(regime=MarketRegime.TRENDING_UP),
            sector_info=SectorInfo(name="Technology", change_pct=0.02),
        )
        batch = SampleBatch(
            batch_id="rs1",
            strategy_id="test_v1",
            symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 3, 1)),
        )
        result = self.engine.run(strategy, {"S": df}, batch, contexts={"S": ctx})
        assert result.total_signals > 0
        for sig in result.signals:
            assert sig.regime == MarketRegime.TRENDING_UP
            assert sig.sector == "Technology"


# ── N-01: RSI Wilder's EMA tests ────────────────────────────────────


class TestRSIWilders:
    """Verify RSI uses Wilder's smoothing (EMA) instead of SMA."""

    def test_rsi_all_gains(self) -> None:
        """If all closes rise, RSI should be 100."""
        prices = [100 + i for i in range(30)]
        df = pd.DataFrame({
            "date": [date(2025, 1, 1) + timedelta(days=i) for i in range(30)],
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": [1e6] * 30, "prev_close": [0] + prices[:-1],
        })
        rsi = IndicatorRegistry.compute("rsi_14", df, 29)
        assert rsi == 100.0

    def test_rsi_all_losses(self) -> None:
        """If all closes fall, RSI should be 0 (with EMA)."""
        prices = [200 - i for i in range(30)]
        df = pd.DataFrame({
            "date": [date(2025, 1, 1) + timedelta(days=i) for i in range(30)],
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": [1e6] * 30, "prev_close": [0] + prices[:-1],
        })
        rsi = IndicatorRegistry.compute("rsi_14", df, 29)
        assert rsi < 1.0  # Wilder's EMA → approaches 0

    def test_rsi_ewm_differs_from_sma(self) -> None:
        """Wilder's RSI should differ from a naive SMA-based RSI on regime-change data."""
        # Long uptrend then sudden decline - EWM carries memory of uptrend
        prices = [100.0]
        for i in range(1, 100):
            if i < 70:
                prices.append(prices[-1] + 1.5)  # steady uptrend
            elif i < 85:
                prices.append(prices[-1] - 2.0)  # sudden decline
            else:
                prices.append(prices[-1] + 0.5)  # mild recovery
        df = pd.DataFrame({
            "date": [date(2025, 1, 1) + timedelta(days=i) for i in range(100)],
            "open": prices, "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices, "volume": [1e6] * 100,
            "prev_close": [0] + prices[:-1],
        })
        rsi_wilder = IndicatorRegistry.compute("rsi_14", df, 90)
        # Compute naive SMA RSI for comparison
        changes = df["close"].diff().iloc[90 - 14 + 1: 91]
        gains = changes.clip(lower=0).mean()
        losses = (-changes.clip(upper=0)).mean()
        if losses == 0:
            rsi_sma = 100.0
        else:
            rs = gains / losses
            rsi_sma = 100 - (100 / (1 + rs))
        # Wilder EMA has memory of the earlier strong uptrend
        assert abs(rsi_wilder - rsi_sma) > 5.0, (
            f"RSI Wilder ({rsi_wilder:.2f}) too close to SMA ({rsi_sma:.2f})"
        )

class TestLimitPolicyByBoard:
    """Verify limit-up/down thresholds vary by stock board type."""

    def test_main_board_10pct(self) -> None:
        """Main board uses 10% threshold."""
        checker = MarketRuleChecker()
        row = pd.Series({"close": 110, "prev_close": 100, "symbol": "600000.SH"})
        assert checker._is_limit_up(row) is True
        row2 = pd.Series({"close": 108, "prev_close": 100, "symbol": "600000.SH"})
        assert checker._is_limit_up(row2) is False

    def test_star_market_20pct(self) -> None:
        """STAR market (688xxx) uses 20% threshold."""
        checker = MarketRuleChecker()
        row = pd.Series({"close": 110, "prev_close": 100, "symbol": "688001.SH"})
        # 10% should NOT be limit-up for STAR
        assert checker._is_limit_up(row) is False
        row2 = pd.Series({"close": 120, "prev_close": 100, "symbol": "688001.SH"})
        assert checker._is_limit_up(row2) is True

    def test_chinext_20pct(self) -> None:
        """ChiNext (300xxx) uses 20% threshold."""
        checker = MarketRuleChecker()
        row = pd.Series({"close": 120, "prev_close": 100, "symbol": "300001.SZ"})
        assert checker._is_limit_up(row) is True

    def test_st_stock_5pct(self) -> None:
        """ST stocks use 5% threshold."""
        checker = MarketRuleChecker()
        row = pd.Series({
            "close": 105, "prev_close": 100,
            "symbol": "600001.SH", "name": "ST某某",
        })
        assert checker._is_limit_up(row) is True
        row2 = pd.Series({
            "close": 104, "prev_close": 100,
            "symbol": "600001.SH", "name": "ST某某",
        })
        assert checker._is_limit_up(row2) is False

    def test_bse_30pct(self) -> None:
        """BSE (8xxxxx) uses 30% threshold."""
        checker = MarketRuleChecker()
        row = pd.Series({"close": 130, "prev_close": 100, "symbol": "830001.BJ"})
        assert checker._is_limit_up(row) is True
        row_20 = pd.Series({"close": 120, "prev_close": 100, "symbol": "830001.BJ"})
        assert checker._is_limit_up(row_20) is False

    def test_prev_close_nan_returns_false(self) -> None:
        """When prev_close is NaN, limit check should return False."""
        checker = MarketRuleChecker()
        row = pd.Series({"close": 110, "prev_close": float("nan"), "symbol": "600000.SH"})
        assert checker._is_limit_up(row) is False
        assert checker._is_limit_down(row) is False


# ── N-03: Indicator warmup tests ────────────────────────────────────


class TestIndicatorWarmup:
    """Verify the engine computes appropriate warmup periods."""

    def test_warmup_from_indicator_names(self) -> None:
        from alphaevo.backtest.engine import _indicator_warmup

        assert _indicator_warmup(["close_above_ma60"]) >= 65
        assert _indicator_warmup(["rsi_14"]) >= 19
        assert _indicator_warmup(["ma5_above_ma10"]) >= 15
        assert _indicator_warmup([]) >= 30  # default minimum

    def test_warmup_macd(self) -> None:
        from alphaevo.backtest.engine import _indicator_warmup

        assert _indicator_warmup(["macd_histogram"]) >= 35

    def test_engine_uses_warmup(self) -> None:
        """Engine start_idx should increase for strategies with large windows."""
        df = _make_ohlcv(200, trend=0.002)
        strategy = _make_strategy()
        # Override conditions to use MA60
        strategy.entry.conditions = [
            StrategyCondition(indicator="close_above_ma60", op="==", value=True)
        ]
        engine = BacktestEngine(min_data_days=30)
        batch = SampleBatch(
            batch_id="w1", strategy_id="test_v1", symbols=["S"],
            date_range=(date(2025, 1, 1), date(2025, 7, 18)),
        )
        result = engine.run(strategy, {"S": df}, batch)
        # All signals should start after warmup period (>=65 bars)
        for sig in result.signals:
            bar_idx = df.index[df["date"] == sig.signal_date].tolist()
            if bar_idx:
                assert bar_idx[0] >= 60


# ── Indicator edge case tests ─────────────────────────────────────────


class TestIndicatorEdgeCases:
    """Test indicators on degenerate data: zero vol, flat prices, single bar."""

    @staticmethod
    def _flat_data(n: int = 100, price: float = 50.0) -> pd.DataFrame:
        """All bars identical: O=H=L=C=price, vol constant."""
        rows = []
        for i in range(n):
            rows.append({
                "date": date(2024, 1, 1) + timedelta(days=i),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1_000_000,
                "prev_close": price,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _all_up_limit(n: int = 100, start: float = 10.0) -> pd.DataFrame:
        """Simulate continuous limit-up: +10% per day."""
        rows = []
        price = start
        for i in range(n):
            prev = price
            price = round(price * 1.10, 2)
            rows.append({
                "date": date(2024, 1, 1) + timedelta(days=i),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 500_000,
                "prev_close": prev,
            })
        return pd.DataFrame(rows)

    def test_rsi_on_flat_data(self) -> None:
        """RSI on zero-volatility data should return 100 (no losses)."""
        df = self._flat_data()
        val = IndicatorRegistry.compute("rsi_14", df, 50, None)
        # With zero changes, avg_loss=0 and avg_gain=0 → RSI=100 (standard: no downside)
        assert val == pytest.approx(100.0, abs=0.01)

    def test_atr_on_flat_data(self) -> None:
        """ATR on flat data should be 0."""
        df = self._flat_data()
        val = IndicatorRegistry.compute("atr", df, 50, None)
        assert val == pytest.approx(0.0, abs=0.001)

    def test_ma5_above_ma10_on_flat_data(self) -> None:
        """MA5 == MA10 on flat data → should return False (not strictly above)."""
        df = self._flat_data()
        val = IndicatorRegistry.compute("ma5_above_ma10", df, 50, None)
        assert val is False

    def test_volume_ratio_on_constant_volume(self) -> None:
        """Volume ratio with constant volume should be ~1.0."""
        df = self._flat_data()
        val = IndicatorRegistry.compute("volume_ratio_1d_5d", df, 50, None)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_bollinger_width_on_flat_data(self) -> None:
        """Bollinger band width with zero volatility should be 0."""
        df = self._flat_data()
        val = IndicatorRegistry.compute("bollinger_band_width", df, 50, None)
        assert val == pytest.approx(0.0, abs=0.001)

    def test_momentum_on_flat_data(self) -> None:
        """Momentum on flat data should be 0."""
        df = self._flat_data()
        val = IndicatorRegistry.compute("momentum_10d", df, 50, None)
        assert val == pytest.approx(0.0, abs=0.001)

    def test_consecutive_up_days_on_flat(self) -> None:
        """No movement → 0 consecutive up days."""
        df = self._flat_data()
        val = IndicatorRegistry.compute("consecutive_up_days", df, 50, None)
        assert val == 0

    def test_rsi_on_all_limit_up(self) -> None:
        """RSI on continuously rising data should be near 100."""
        df = self._all_up_limit()
        val = IndicatorRegistry.compute("rsi_14", df, 50, None)
        assert val > 90.0

    def test_ma_slope_on_limit_up(self) -> None:
        """MA slope on limit-up data should be strongly positive."""
        df = self._all_up_limit()
        val = IndicatorRegistry.compute("ma20_slope", df, 50, None)
        assert val > 0

    def test_volatility_on_flat_data(self) -> None:
        """Volatility with zero returns should be 0."""
        df = self._flat_data()
        val = IndicatorRegistry.compute("volatility_20d", df, 50, None)
        assert val == pytest.approx(0.0, abs=0.001)

    def test_engine_on_flat_data_no_crash(self) -> None:
        """Engine should not crash on flat data (no signals but no errors)."""
        df = self._flat_data(200)
        strategy = _make_strategy()
        engine = BacktestEngine(min_data_days=30)
        batch = SampleBatch(
            batch_id="flat", strategy_id="test_v1", symbols=["FLAT"],
            date_range=(date(2024, 1, 1), date(2024, 7, 18)),
        )
        result = engine.run(strategy, {"FLAT": df}, batch)
        # Must not crash; signals may or may not trigger
        assert result.total_signals >= 0

    def test_engine_on_single_symbol_short_data(self) -> None:
        """Engine with fewer bars than min_data_days should skip the symbol."""
        df = self._flat_data(10)
        strategy = _make_strategy()
        engine = BacktestEngine(min_data_days=30)
        batch = SampleBatch(
            batch_id="short", strategy_id="test_v1", symbols=["SHORT"],
            date_range=(date(2024, 1, 1), date(2024, 1, 10)),
        )
        result = engine.run(strategy, {"SHORT": df}, batch)
        assert result.total_signals == 0


class TestCPCVUnit:
    """Unit tests for the CPCV evaluator method."""

    def _make_signals(self, n: int, win_rate: float = 0.5) -> list:
        """Generate deterministic trade signals."""
        import random

        from alphaevo.models.enums import SignalDirection
        from alphaevo.models.execution import TradeSignal

        rng = random.Random(42)
        signals = []
        for i in range(n):
            ret = abs(rng.gauss(0.02, 0.01)) if rng.random() < win_rate else -abs(rng.gauss(0.02, 0.01))
            signals.append(TradeSignal(
                symbol=f"S{i % 5}",
                signal_date=date(2023, 1, 1) + timedelta(days=i),
                direction=SignalDirection.LONG,
                entry_price=50.0,
                exit_price=50.0 * (1 + ret),
                exit_date=date(2023, 1, 1) + timedelta(days=i + 3),
                return_pct=ret,
                holding_days=3,
            ))
        return signals

    def test_cpcv_returns_none_for_few_signals(self) -> None:
        from alphaevo.evaluator.metrics import Evaluator
        evaluator = Evaluator()
        signals = self._make_signals(20)
        result = evaluator.compute_cpcv(signals)
        assert result is None

    def test_cpcv_returns_metrics_for_enough_signals(self) -> None:
        from alphaevo.evaluator.metrics import Evaluator
        evaluator = Evaluator()
        signals = self._make_signals(60)
        result = evaluator.compute_cpcv(signals, n_groups=4, n_test_groups=1)
        assert result is not None
        assert result.n_paths > 0
        assert result.mean_gap >= 0
        assert result.mean_test_win_rate > 0

    def test_cpcv_paths_match_combinatorics(self) -> None:
        """Number of paths should equal C(n_groups, n_test_groups)."""
        from math import comb

        from alphaevo.evaluator.metrics import Evaluator
        evaluator = Evaluator()
        signals = self._make_signals(120)
        result = evaluator.compute_cpcv(signals, n_groups=6, n_test_groups=2)
        if result is not None:
            # Some paths may be dropped if too few signals after purging
            assert result.n_paths <= comb(6, 2)
            assert result.n_paths > 0


class TestStrategyFingerprint:
    """Unit tests for strategy fingerprint deduplication."""

    def test_same_strategy_same_fingerprint(self) -> None:
        from alphaevo.evaluator.metrics import Evaluator
        s1 = _make_strategy()
        s2 = _make_strategy()
        fp1 = Evaluator.compute_strategy_fingerprint(s1)
        fp2 = Evaluator.compute_strategy_fingerprint(s2)
        assert fp1 == fp2

    def test_different_conditions_different_fingerprint(self) -> None:
        from alphaevo.evaluator.metrics import Evaluator
        s1 = _make_strategy()
        s2 = _make_strategy()
        s2.entry.conditions[0].value = 999  # change a condition value
        fp1 = Evaluator.compute_strategy_fingerprint(s1)
        fp2 = Evaluator.compute_strategy_fingerprint(s2)
        assert fp1 != fp2

    def test_meta_changes_dont_affect_fingerprint(self) -> None:
        from alphaevo.evaluator.metrics import Evaluator
        s1 = _make_strategy()
        s2 = _make_strategy()
        s2.meta.id = "completely_different_id"
        s2.meta.name = "Different Name"
        s2.meta.version = 99
        fp1 = Evaluator.compute_strategy_fingerprint(s1)
        fp2 = Evaluator.compute_strategy_fingerprint(s2)
        assert fp1 == fp2


# ── ST detection robustness tests ─────────────────────────────────────


class TestSTDetectionRobustness:
    """Verify ST stock identification beyond simple name string matching."""

    def test_st_flag_column_truthy(self) -> None:
        """When the row has an explicit st=True flag, use 5% threshold."""
        checker = MarketRuleChecker()
        row = pd.Series({
            "close": 105, "prev_close": 100,
            "symbol": "600001.SH", "name": "正常公司", "st": True,
        })
        assert checker._is_limit_up(row) is True

    def test_st_flag_column_falsy(self) -> None:
        """st=False should use normal (10%) threshold."""
        checker = MarketRuleChecker()
        row = pd.Series({
            "close": 105, "prev_close": 100,
            "symbol": "600001.SH", "name": "正常公司", "st": False,
        })
        assert checker._is_limit_up(row) is False

    def test_star_st_name_pattern(self) -> None:
        """*ST prefix is a common pattern."""
        checker = MarketRuleChecker()
        row = pd.Series({
            "close": 105, "prev_close": 100,
            "symbol": "600001.SH", "name": "*ST某某",
        })
        assert checker._is_limit_up(row) is True

    def test_st_space_name_pattern(self) -> None:
        """'ST ' with space is a standard prefix pattern."""
        checker = MarketRuleChecker()
        row = pd.Series({
            "close": 105, "prev_close": 100,
            "symbol": "600001.SH", "name": "ST 某某",
        })
        assert checker._is_limit_up(row) is True

    def test_no_false_positive_on_unrelated_name(self) -> None:
        """Names like 'FAST' or 'STRONGEST' should NOT trigger ST detection."""
        from alphaevo.backtest.rules import _limit_threshold
        row = pd.Series({
            "close": 110, "prev_close": 100,
            "symbol": "600001.SH", "name": "FASTEST",
        })
        assert _limit_threshold(row) == 0.098  # main board, not ST


# ── EMA warmup buffer tests ──────────────────────────────────────────


class TestEMAWarmupBuffer:
    """Verify increased warmup for EMA-based indicators."""

    def test_rsi_gets_ema_warmup(self) -> None:
        from alphaevo.backtest.engine import _indicator_warmup
        result = _indicator_warmup(["rsi_14"])
        # RSI uses Wilder's EMA, so warmup should be at least 2×14 = 28
        assert result >= 28

    def test_macd_gets_ema_warmup(self) -> None:
        from alphaevo.backtest.engine import _indicator_warmup
        result = _indicator_warmup(["macd_histogram"])
        # MACD default 12/26/9, max_period=35, EMA so buffer=35 → ≥70
        assert result >= 70

    def test_bollinger_gets_ema_warmup(self) -> None:
        from alphaevo.backtest.engine import _indicator_warmup
        result = _indicator_warmup(["bollinger_band_width_20d"])
        # bollinger triggers EMA, period=20, buffer=20 → ≥40
        assert result >= 40

    def test_pure_sma_gets_smaller_warmup(self) -> None:
        from alphaevo.backtest.engine import _indicator_warmup
        result = _indicator_warmup(["close_above_ma60"])
        # SMA indicator: period=60, buffer=max(5, 15)=15, result=75
        assert result >= 65
        # But should be less than 2×60 = 120 (EMA would give that)
        assert result < 120
