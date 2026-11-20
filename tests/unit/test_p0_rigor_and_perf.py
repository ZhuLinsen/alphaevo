"""Comprehensive P0 quantitative correctness, security, speedup, and L0/shadow regressions."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from alphaevo.alpha_factory.factory import AlphaFactory, _build_indicator_fn
from alphaevo.backtest.engine import BacktestEngine
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.backtest.portfolio import (
    PortfolioBacktester,
    PortfolioConfig,
    PortfolioSnapshot,
    _compute_portfolio_sharpe,
)
from alphaevo.backtest.rules import MarketRuleChecker
from alphaevo.cli.demo import _find_builtin_strategy_dir, _showcase_data_dir
from alphaevo.core.config import LLMConfig
from alphaevo.core.llm import LLMClient
from alphaevo.data.adapter import DataAdapter, DataManager
from alphaevo.data.cache import DataCache
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.forward.contracts import OutcomeEvent, OutcomeHorizon
from alphaevo.forward.shadow_ledger import ShadowLedger
from alphaevo.models.enums import (
    ExitReason,
    MarketRegime,
    MarketType,
    PromotionStage,
    RSILevel,
    SignalDirection,
    StrategyCategory,
)
from alphaevo.models.execution import (
    AntiFitMetrics,
    BacktestResult,
    OverallMetrics,
    SampleBatch,
    TradeSignal,
)
from alphaevo.models.market import IndicatorContext, MarketContext, SectorInfo, StockInfo
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
from alphaevo.rsi.engine import StockRSIEngine
from alphaevo.rsi.program import RSICharter
from alphaevo.sampler.regime import RegimeDetector
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.truth.ledger import TrialLedger
from alphaevo.truth.null_control import NullSignalController, generate_block_shuffled_market


def _make_ohlcv(
    n: int = 120,
    *,
    start: date = date(2024, 1, 1),
    trend: float = 0.002,
    vol: float = 0.015,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, vol, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.005, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.005, 0.004, n)))
    df = pd.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(n)],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(500_000, 3_000_000, n).astype(float),
        }
    )
    df["prev_close"] = df["close"].shift(1)
    return df


def _minimal_strategy(
    *,
    sid: str = "p0_test_v1",
    market: MarketType = MarketType.A_SHARE,
    stop_loss: StopLossConfig | None = None,
    take_profit: TakeProfitConfig | None = None,
    timing: str = "next_open",
    preferred_regime: list[str] | None = None,
    conditions: list[StrategyCondition] | None = None,
) -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id=sid,
            name="P0 Regression Strategy",
            version=1,
            market=market,
            category=StrategyCategory.TREND,
            preferred_regime=preferred_regime or [],
        ),
        description="P0 test strategy",
        entry=StrategyEntry(
            conditions=conditions or [StrategyCondition(indicator="rsi_14", op=">=", value=0.0)],
            execution=ExecutionConfig(timing=timing, slippage=0.0),
        ),
        exit=StrategyExit(
            stop_loss=stop_loss or StopLossConfig(type="pct", value=0.05),
            take_profit=take_profit or TakeProfitConfig(type="pct", value=0.50),
            max_holding_days=10,
        ),
    )


# ── 1. Look-Ahead & Date Alignment ──────────────────────────────────────


def test_benchmark_date_alignment_blocks_offset_lookahead() -> None:
    sym_df = _make_ohlcv(80, start=date(2024, 3, 1), trend=0.001, seed=11)
    # Benchmark starts 30 days earlier with completely different price action
    bm_df = _make_ohlcv(140, start=date(2024, 1, 15), trend=0.008, seed=22)
    ctx = IndicatorContext(benchmark_df=bm_df)

    # Locate the matching date row inside benchmark_df
    target_idx = 50
    target_date = sym_df["date"].iloc[target_idx]
    bm_dates = list(bm_df["date"])
    bm_match = bm_dates.index(target_date)
    expected_bm_ret = (bm_df["close"].iloc[bm_match] / bm_df["close"].iloc[bm_match - 20]) - 1.0
    expected_sym_ret = (
        sym_df["close"].iloc[target_idx] / sym_df["close"].iloc[target_idx - 20]
    ) - 1.0

    rs_val = IndicatorRegistry.compute("relative_strength_20d", sym_df, target_idx, ctx)
    assert rs_val == pytest.approx(expected_sym_ret - expected_bm_ret, abs=1e-9)

    engine = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=20)
    engine._regime_series_cache = {}
    engine._regime_cache = {}
    bm_regimes = RegimeDetector().detect_series(bm_df)
    assert engine._get_current_regime(sym_df, target_idx, ctx) == bm_regimes[bm_match]


def test_static_market_context_ignored_on_long_series_and_warns() -> None:
    # Strongly trending up 100-bar series without benchmark_df
    df = _make_ohlcv(100, trend=0.015, vol=0.003, seed=3)
    strat = _minimal_strategy(
        conditions=[
            StrategyCondition(indicator="sector_heat_rank", op="<=", value=5),
            StrategyCondition(indicator="st_flag", op="==", value=False),
        ],
        preferred_regime=["trending_up", "euphoria"],
    )
    ctx = IndicatorContext(
        market_context=MarketContext(regime=MarketRegime.PANIC),
        sector_info=SectorInfo(name="Tech", change_pct=0.01, heat_rank=1),
        stock_info=StockInfo(symbol="000001", name="Test", market=MarketType.A_SHARE, is_st=False),
    )
    engine = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=30)
    batch = SampleBatch(
        batch_id="b",
        strategy_id=strat.meta.id,
        symbols=["000001"],
        date_range=(df["date"].min(), df["date"].max()),
    )
    bt = engine.run(strat, {"000001": df}, batch, contexts={"000001": ctx})
    # Static MarketContext(PANIC) must NOT override causal TRENDING_UP/EUPHORIA detection
    assert bt.total_signals > 0

    rep = Evaluator().evaluate(bt, strat, market_data={"000001": df}, contexts={"000001": ctx})
    assert len(rep.static_context_warnings) >= 2


# ── 2. Historical Data Range & Cache Isolation / Locking ────────────────


class _RangeAdapter(DataAdapter):
    def __init__(self, name_val: str, df: pd.DataFrame) -> None:
        self._name = name_val
        self._df = df
        self.range_calls = 0
        self.daily_calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        self.daily_calls += 1
        return self._df.tail(days).copy()

    async def get_history_range(self, symbol: str, start: date, end: date) -> pd.DataFrame | None:
        self.range_calls += 1
        return self._df.copy()

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        return []


@pytest.mark.asyncio
async def test_data_manager_range_and_cache_namespace(tmp_path: Path) -> None:
    df1 = _make_ohlcv(40, start=date(2020, 1, 1), seed=1)
    df2 = _make_ohlcv(40, start=date(2020, 1, 1), seed=99)
    cache = DataCache(tmp_path / "cache")

    mgr1 = DataManager([_RangeAdapter("src_a", df1)], cache=cache)
    mgr2 = DataManager([_RangeAdapter("src_b", df2)], cache=cache)

    got1 = await mgr1.get_history("AAPL", date(2020, 1, 5), date(2020, 2, 5))
    got2 = await mgr2.get_history("AAPL", date(2020, 1, 5), date(2020, 2, 5))
    assert not got1.empty and not got2.empty
    assert float(got1["close"].iloc[-1]) != float(got2["close"].iloc[-1])


# ── 3. Gap Fills, Frozen Entry ATR, next_open & A-share Rules ──────────


def test_gap_fills_frozen_atr_and_next_open_can_buy() -> None:
    # Bar 30: signal, Bar 31: entry open=100, Bar 32: gap down open=80 (stop at 95 -> fills at 80)
    rows = []
    for i in range(40):
        d = date(2025, 1, 1) + timedelta(days=i)
        o = h = lo = c = 100.0
        vol = 1_000_000.0
        if i == 32:
            o = h = 82.0
            lo = 79.0
            c = 80.0
        rows.append(
            {
                "date": d,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": vol,
                "prev_close": 100.0,
            }
        )
    df = pd.DataFrame(rows)
    strat = _minimal_strategy(
        market=MarketType.US,
        stop_loss=StopLossConfig(type="pct", value=0.05),
        take_profit=TakeProfitConfig(type="pct", value=0.50),
    )
    bt = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=25).run(
        strat,
        {"AAPL": df},
        SampleBatch(
            batch_id="g",
            strategy_id=strat.meta.id,
            symbols=["AAPL"],
            date_range=(date(2025, 1, 1), date(2025, 2, 15)),
        ),
    )
    first_sl = next(s for s in bt.signals if s.exit_reason == ExitReason.STOP_LOSS)
    assert first_sl.exit_price == pytest.approx(82.0)

    # Gap-up TP fill test
    rows_tp = []
    for i in range(36):
        d = date(2025, 1, 1) + timedelta(days=i)
        o = h = lo = c = 100.0
        if i == 32:
            o = 125.0
            h = 128.0
            lo = 124.0
            c = 126.0
        rows_tp.append(
            {
                "date": d,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": 1_000_000.0,
                "prev_close": 100.0,
            }
        )
    df_tp = pd.DataFrame(rows_tp)
    strat_tp = _minimal_strategy(
        market=MarketType.US,
        stop_loss=StopLossConfig(type="pct", value=0.20),
        take_profit=TakeProfitConfig(type="pct", value=0.10),
    )
    bt_tp = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=25).run(
        strat_tp,
        {"AAPL": df_tp},
        SampleBatch(
            batch_id="tp",
            strategy_id=strat_tp.meta.id,
            symbols=["AAPL"],
            date_range=(date(2025, 1, 1), date(2025, 2, 15)),
        ),
    )
    first_tp = next(s for s in bt_tp.signals if s.exit_reason == ExitReason.TAKE_PROFIT)
    assert first_tp.exit_price == pytest.approx(125.0)

    # Frozen ATR stop loss must not widen after later volatility spike
    rows_atr = []
    for i in range(40):
        d = date(2025, 1, 1) + timedelta(days=i)
        spread = 0.5 if i <= 31 else 10.0  # tiny ATR=1.0 through bar 31, huge spread afterwards
        rows_atr.append(
            {
                "date": d,
                "open": 100.0,
                "high": 100.0 + spread,
                "low": 100.0 - spread,
                "close": 100.0,
                "volume": 1_000_000.0,
                "prev_close": 100.0,
            }
        )
    df_atr = pd.DataFrame(rows_atr)
    strat_atr = _minimal_strategy(
        market=MarketType.US,
        stop_loss=StopLossConfig(type="atr", multiplier=1.0),
        take_profit=TakeProfitConfig(type="pct", value=0.50),
    )
    bt_atr = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=25).run(
        strat_atr,
        {"AAPL": df_atr},
        SampleBatch(
            batch_id="atr",
            strategy_id=strat_atr.meta.id,
            symbols=["AAPL"],
            date_range=(date(2025, 1, 1), date(2025, 2, 15)),
        ),
    )
    sl_atr = next(s for s in bt_atr.signals if s.exit_reason == ExitReason.STOP_LOSS)
    # Stop level frozen near 99.0; bar 32 opens at 100 and drops to 90 -> fills near stop ~99.0 (not 80!)
    assert sl_atr.exit_price > 95.0

    # A-share board prefix limits + ST flag
    checker = MarketRuleChecker()
    rules = MarketRuleConfig(t_plus_1=True, limit_up_down=True, suspension=True)
    df_lim = pd.DataFrame(
        [
            {
                "open": 10.0,
                "high": 11.5,
                "low": 10.0,
                "close": 11.5,
                "prev_close": 10.0,
                "volume": 1000.0,
            }
        ]
    )
    assert (
        checker.can_buy(df_lim, 0, rules, symbol="600000") is False
    )  # +15% exceeds 10% main board
    assert checker.can_buy(df_lim, 0, rules, symbol="sh688001") is True  # +15% allowed on 20% STAR
    assert (
        checker.can_buy(df_lim, 0, rules, symbol="sz300750") is True
    )  # +15% allowed on 20% ChiNext
    df_st = pd.DataFrame(
        [
            {
                "open": 10.0,
                "high": 10.6,
                "low": 10.0,
                "close": 10.6,
                "prev_close": 10.0,
                "volume": 1000.0,
            }
        ]
    )
    assert (
        checker.can_buy(df_st, 0, rules, symbol="000001", is_st=True) is False
    )  # +6% blocked on 5% ST


def test_next_open_no_same_bar_entry_on_last_bar_and_blocks_limit_up_next_day() -> None:
    df = _make_ohlcv(36, trend=0.0, vol=0.001, seed=4)
    # Bar 31 hits limit-up so next_open entry for bar 30 signal cannot buy at bar 31
    df.loc[31, "prev_close"] = 100.0
    df.loc[31, "close"] = 112.0
    df.loc[31, "open"] = 110.0
    engine = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=30)
    pos_last = engine._open_position("000001", df, len(df) - 1, "next_open", 0.0, None)
    assert pos_last is None
    rules = MarketRuleConfig(t_plus_1=True, limit_up_down=True, suspension=True)
    pos_limit = engine._open_position("000001", df, 30, "next_open", 0.0, None, rules=rules)
    assert pos_limit is None


# ── 4. Evaluator Chronology, Daily MTM, Neutral Sensitivity & Portfolio ─


def test_evaluator_chronology_daily_mtm_and_portfolio_fixes() -> None:
    s1 = TradeSignal(
        symbol="A",
        signal_date=date(2025, 1, 2),
        exit_date=date(2025, 1, 6),
        direction=SignalDirection.LONG,
        entry_price=100.0,
        exit_price=110.0,
        return_pct=0.098,
        holding_days=2,
    )
    s2 = TradeSignal(
        symbol="B",
        signal_date=date(2025, 1, 3),
        exit_date=date(2025, 1, 7),
        direction=SignalDirection.LONG,
        entry_price=100.0,
        exit_price=90.0,
        return_pct=-0.102,
        holding_days=2,
    )
    ev = Evaluator()
    m_ord = ev.compute_metrics([s1, s2])
    m_rev = ev.compute_metrics([s2, s1])
    assert m_ord.max_drawdown == m_rev.max_drawdown
    assert m_ord.total_return == m_rev.total_return

    # Neutral sensitivity (0.5) when param_sensitivity_measured=False
    base_m = OverallMetrics(
        win_rate=0.6,
        avg_return=0.03,
        profit_loss_ratio=2.0,
        max_drawdown=0.10,
        sharpe_ratio=1.5,
        signal_count=40,
    )
    sc_none = ev.compute_confidence_score(base_m, 0.0, anti_fit=None)
    sc_unmeas = ev.compute_confidence_score(
        base_m,
        0.0,
        anti_fit=AntiFitMetrics(
            yearly_consistency=0.5, param_sensitivity=0.0, param_sensitivity_measured=False
        ),
    )
    sc_meas = ev.compute_confidence_score(
        base_m,
        0.0,
        anti_fit=AntiFitMetrics(
            yearly_consistency=0.5, param_sensitivity=0.0, param_sensitivity_measured=True
        ),
    )
    assert sc_none == pytest.approx(sc_unmeas, abs=1e-6)
    assert sc_meas > sc_unmeas

    # Portfolio commission-consistent PnL & zero-return preservation in Sharpe
    bt = BacktestResult(
        strategy_id="x",
        batch_id="b",
        signals=[s1, s2],
        total_signals=2,
        executed_signals=2,
        skipped_signals=0,
        date_range=(date(2025, 1, 2), date(2025, 1, 7)),
    )
    port = PortfolioBacktester(
        PortfolioConfig(initial_capital=100_000.0, max_positions=2, position_size_pct=0.5)
    ).simulate(bt)
    assert port.trades[0].pnl == pytest.approx(port.trades[0].capital_allocated * s1.return_pct)

    flat_and_up = [
        PortfolioSnapshot(
            date=date(2025, 1, i), equity=100.0, cash=100.0, positions_open=0, daily_return=0.0
        )
        for i in range(1, 10)
    ] + [
        PortfolioSnapshot(
            date=date(2025, 1, 10), equity=102.0, cash=102.0, positions_open=0, daily_return=0.02
        ),
        PortfolioSnapshot(
            date=date(2025, 1, 11), equity=104.04, cash=104.04, positions_open=0, daily_return=0.02
        ),
    ]
    # Retaining zero-return days lowers Sharpe relative to filtering them out
    assert _compute_portfolio_sharpe(flat_and_up) > 0.0


# ── 5. Vectorized O(N) Parity & Signature Caching ──────────────────────


def test_vectorized_precompute_and_regime_parity() -> None:
    df = _make_ohlcv(140, seed=19)
    names = [
        "ma5_above_ma10",
        "ma10_above_ma30",
        "ma5_ge_ma10_or_crossing",
        "close_to_ma10_pct",
        "close_above_ma20",
        "close_below_ma10",
        "deviation_from_ma20_pct",
        "ma20_slope",
        "rsi_14",
        "rsi_7",
        "rsi_14_zscore",
        "atr",
        "atr_21",
        "macd_histogram",
        "macd_cross_bullish",
        "bollinger_band_width",
        "bollinger_band_width_15d_std1p5",
        "price_above_bollinger_upper",
        "price_below_bollinger_lower",
        "breakout_high_20d",
        "price_position_20d",
        "price_position_52w",
        "volume_ratio_1d_5d",
        "volume_ratio_1d_20d",
        "momentum_10d",
        "avg_volume_20d",
        "consecutive_up_days",
        "consecutive_down_days",
        "days_since_high_20d",
        "days_since_low_20d",
        "volatility_20d",
        "gap_up_pct",
        "body_to_range_ratio",
        "has_stop_signal",
        "volume_shrink_then_rise",
        "negative_news_score",
        "news_sentiment_score",
        "days_since_event",
    ]
    pre = IndicatorRegistry.precompute(names, df)
    for name in names:
        for i in range(len(df)):
            v_pre = pre[name][i]
            v_sca = IndicatorRegistry.compute(name, df, i)
            if isinstance(v_sca, bool):
                assert bool(v_pre) == v_sca, f"{name} mismatch at {i}"
            else:
                assert float(v_pre) == pytest.approx(float(v_sca), abs=1e-9), (
                    f"{name} mismatch at {i}"
                )

    det = RegimeDetector()
    ser = det.detect_series(df)
    for i in range(len(df)):
        assert ser[i] == det.detect(df.iloc[: i + 1].reset_index(drop=True))


# ── 6. AlphaFactory Sandbox & PIT Runtime Enforcement ──────────────────


def test_alpha_factory_sandbox_and_runtime_pit() -> None:
    from alphaevo.alpha_factory.sandbox import FactorSandbox

    sb = FactorSandbox()
    # Registered dynamic factor must not see future bars df.iloc[idx+1]
    lookfuture_code = """
def compute(df, idx):
    return float(len(df) - 1 - idx)
"""
    fn = _build_indicator_fn(lookfuture_code, sb)
    df = _make_ohlcv(50)
    # Because df passed to compute is PIT-truncated to [:idx+1], len(pit_df) - 1 - idx == 0.0
    assert fn(df, 10) == pytest.approx(0.0)

    fac = AlphaFactory(LLMClient(LLMConfig()), db_path=":memory:")
    assert fac._verify_pit is True


# ── 7. TrialLedger, Joint Null Worlds & ShadowLedger Hardening ─────────


def test_trial_ledger_unwritable_raises_and_joint_null_worlds(tmp_path: Path) -> None:
    blocker_file = tmp_path / "not_a_dir"
    blocker_file.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        TrialLedger(blocker_file / "nested.db")

    mem = TrialLedger(":memory:")
    assert mem.count_total_trials() == 0

    m = {"A": _make_ohlcv(60, seed=1), "B": _make_ohlcv(60, seed=2)}
    shuf = generate_block_shuffled_market(m, block_size=5, seed=7, joint=True)
    # Synchronous dates preserved
    assert list(shuf["A"]["date"]) == list(shuf["B"]["date"])
    strat = _minimal_strategy()
    from alphaevo.models.execution import EvaluationReport

    rep = EvaluationReport(
        strategy_id=strat.meta.id, overall=OverallMetrics(avg_return=0.02, signal_count=10)
    )
    null_rep = NullSignalController().evaluate_strategy_against_null_worlds(
        strat, m, rep, n_worlds=2
    )
    assert null_rep.block_shuffle.method == "strategy_rerun"

    # Tier-2 zero-promotion audit on noisy null market
    def _runner(st: Strategy, world: dict[str, pd.DataFrame]) -> int:
        eng = StockRSIEngine(db_path=":memory:")
        ch = RSICharter(
            name="tier2_test",
            seed_strategy_id=st.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=1,
            require_null_control=True,
            null_permutations=20,
        )
        return eng.run(st, world, charter=ch).promotions

    audit = NullSignalController().verify_zero_promotion_under_shuffled_returns(
        strat, m, _runner, n_worlds=1
    )
    assert audit.passed is True


def test_shadow_ledger_hardening_guards(tmp_path: Path) -> None:
    ledger = ShadowLedger(tmp_path / "shadow")
    strat = StrategyParser().parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    from alphaevo.models.execution import EvaluationReport

    ev = EvaluationReport(
        strategy_id=strat.meta.id,
        overall=OverallMetrics(avg_return=0.02, win_rate=0.55, signal_count=30),
    )
    ledger.register_candidate(strat, ev)

    data = {"AAPL": _make_ohlcv(50, start=date(2025, 1, 1), seed=31)}
    # Backdated commit rejection
    ledger.commit_signals(strat, data, as_of=date(2025, 2, 10), is_replay=True)
    with pytest.raises(ValueError, match="Non-monotonic"):
        ledger.commit_signals(strat, data, as_of=date(2025, 2, 5), is_replay=True)

    # Strategy hash tamper rejection
    tampered = strat.model_copy(deep=True)
    tampered.exit.max_holding_days = 99
    with pytest.raises(ValueError, match="hash mismatch"):
        ledger.commit_signals(tampered, data, as_of=date(2025, 2, 12), is_replay=True)

    # HALTED overwrite protection & CANARY dwell check
    cands = ledger._load_candidates()
    cands[strat.meta.id].stage = PromotionStage.HALTED
    ledger._save_candidates(cands)
    with pytest.raises(ValueError, match="HALTED"):
        ledger.register_candidate(strat, ev)
    dec_halted = ledger.evaluate_promotion(strat.meta.id)
    assert dec_halted.new_stage == PromotionStage.HALTED

    # Replay-only promotion block + CANARY dwell enforcement + non-downgrading backfill
    ledger2 = ShadowLedger(tmp_path / "shadow2")
    ledger2.register_candidate(strat, ev)
    ledger2.commit_signals(strat, data, as_of=date(2025, 2, 10), is_replay=True)
    dec_rep = ledger2.evaluate_promotion(
        strat.meta.id, min_trading_days=1, min_mature_signals=1, min_effective_signals=1.0
    )
    assert any("Replay-only" in r for r in dec_rep.blocking_reasons)

    # Atomic backfill preserves already-mature horizons on later calls with truncated future
    ledger2.backfill_outcomes(data, evaluated_as_of=date(2025, 2, 19))
    prior_ev = OutcomeEvent(
        signal_id="sig_mature",
        strategy_id=strat.meta.id,
        symbol="AAPL",
        sector="tech",
        as_of=date(2025, 1, 10),
        evaluated_as_of=date(2025, 2, 1),
        horizons={
            10: OutcomeHorizon(
                horizon_days=10, mature=True, tradable_post_cost_return=0.03, excess_return=0.02
            )
        },
    )
    with ledger2.outcomes_file.open("a", encoding="utf-8") as fh:
        fh.write(prior_ev.model_dump_json() + "\n")
    ledger2.backfill_outcomes(
        {"AAPL": data["AAPL"].iloc[:5].copy()}, evaluated_as_of=date(2025, 2, 19)
    )
    reloaded = {e.signal_id: e for e in ledger2.load_outcomes(strat.meta.id)}
    assert reloaded["sig_mature"].horizons[10].mature is True

    # CANARY -> ACTIVE dwell blocks if mature days strictly after stage_entered_as_of < min_trading_days
    ledger3 = ShadowLedger(tmp_path / "shadow3")
    ledger3.register_candidate(strat, ev, stage=PromotionStage.CANARY)
    c3 = ledger3._load_candidates()
    c3[strat.meta.id].stage_entered_as_of = date(2025, 2, 1)
    ledger3._save_candidates(c3)
    events3 = [
        OutcomeEvent(
            signal_id=f"c_{i}",
            strategy_id=strat.meta.id,
            symbol="AAPL",
            sector=f"sec_{i % 3}",
            as_of=date(2025, 1, 15 + i),
            evaluated_as_of=date(2025, 2, 5),
            horizons={
                10: OutcomeHorizon(
                    horizon_days=10,
                    mature=True,
                    tradable_post_cost_return=0.03,
                    excess_return=0.02,
                )
            },
        )
        for i in range(10)
    ]
    with ledger3.outcomes_file.open("w", encoding="utf-8") as fh:
        for item in events3:
            fh.write(item.model_dump_json() + "\n")
    dec_dwell = ledger3.evaluate_promotion(
        strat.meta.id, min_trading_days=5, min_mature_signals=5, min_effective_signals=1.0
    )
    assert any("Canary dwell" in r for r in dec_dwell.blocking_reasons)


def test_bundled_asset_helpers_resolve_cleanly() -> None:
    bdir = _find_builtin_strategy_dir()
    sdir = _showcase_data_dir()
    assert bdir is not None and (bdir / "rsi_reversion.yaml").is_file()
    assert sdir is not None and (sdir / "manifest.json").is_file()
