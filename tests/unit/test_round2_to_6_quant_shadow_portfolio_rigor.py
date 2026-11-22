"""Regression tests for Rounds 2-6: ShadowLedger, MarketRuleChecker, BacktestEngine, PortfolioBacktester, Allocator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from alphaevo.backtest.allocator import (
    RegimeAdaptiveAllocator,
    RiskBudgetConfig,
    _build_causal_market_proxy,
)
from alphaevo.backtest.engine import BacktestEngine
from alphaevo.backtest.portfolio import PortfolioBacktester, PortfolioConfig
from alphaevo.backtest.rules import MarketRuleChecker
from alphaevo.forward.contracts import OutcomeHorizon
from alphaevo.forward.shadow_ledger import ShadowLedger
from alphaevo.models.enums import (
    ExitReason,
    MarketType,
    PromotionStage,
    SignalDirection,
    StrategyCategory,
)
from alphaevo.models.execution import (
    AntiFitMetrics,
    BacktestResult,
    EvaluationReport,
    OverallMetrics,
    SampleBatch,
    TradeSignal,
)
from alphaevo.models.strategy import (
    ExecutionConfig,
    MarketRuleConfig,
    SetupSpec,
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)


def _eval_report() -> EvaluationReport:
    return EvaluationReport(
        strategy_id="test_strat",
        batch_id="b1",
        overall=OverallMetrics(
            win_rate=0.60,
            avg_return=0.02,
            sharpe_ratio=1.5,
            signal_count=40,
        ),
        anti_overfit=AntiFitMetrics(
            train_val_gap=0.01,
            val_test_gap=0.01,
            param_sensitivity=0.05,
        ),
        deflated_sharpe_prob=0.90,
        null_control_passed=True,
    )


def _make_strategy(
    *,
    sid: str = "r2_strat",
    market: MarketType = MarketType.US,
    rsi_max: float = 100.0,
    stop_pct: float = 0.05,
    timing: str = "next_open",
) -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id=sid,
            name=sid,
            market=market,
            category=StrategyCategory.REVERSAL,
        ),
        description="Deterministic test strategy",
        setup=SetupSpec(
            trigger_rule="rsi_14 <= cap",
            planned_stop_pct=stop_pct,
            target_rr=2.0,
            min_rr=1.0,
        ),
        entry=StrategyEntry(
            logic="and",
            triggers=[StrategyCondition(indicator="rsi_14", op="<=", value=rsi_max)],
            execution=ExecutionConfig(timing=timing, slippage=0.0),  # type: ignore[arg-type]
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=stop_pct),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
    )


def _make_bars(
    start: date,
    n_bars: int,
    *,
    step: float = 0.008,
    base_price: float = 100.0,
) -> pd.DataFrame:
    rows = []
    price = base_price
    cur = start
    while len(rows) < n_bars:
        if cur.weekday() < 5:
            prev_c = price
            price = prev_c * (1.0 + step)
            rows.append(
                {
                    "date": cur,
                    "open": round(prev_c, 4),
                    "high": round(max(prev_c, price) * 1.005, 4),
                    "low": round(min(prev_c, price) * 0.998, 4),
                    "close": round(price, 4),
                    "prev_close": round(prev_c, 4),
                    "volume": 1_000_000.0,
                }
            )
        cur += timedelta(days=1)
    return pd.DataFrame(rows)


def test_round2_mixed_replay_and_live_cannot_promote_and_hash_re_reg_isolates(
    tmp_path: Path,
) -> None:
    ledger = ShadowLedger(tmp_path / "ledger")
    strat_v1 = _make_strategy(sid="mixed_strat", market=MarketType.US, rsi_max=100.0)
    ledger.register_candidate(strat_v1, _eval_report())

    bars = _make_bars(date(2025, 1, 2), 90, step=0.008)
    trading_days = pd.to_datetime(bars["date"]).dt.date.tolist()
    market = {f"US{i}": bars.copy() for i in range(4)}

    # 20 replay commits yielding plenty of positive mature outcomes
    for d in trading_days[20:40]:
        ledger.commit_signals(strat_v1, market, as_of=d, is_replay=True)

    # 1 live commit so all(c.is_replay for c in commits) is False, yet live samples < gate!
    ledger.commit_signals(strat_v1, market, as_of=trading_days[41], is_replay=False)
    ledger.backfill_outcomes(market, evaluated_as_of=trading_days[-1])

    dec = ledger.evaluate_promotion(
        strat_v1.meta.id,
        horizon_days=5,
        min_trading_days=5,
        min_mature_signals=5,
        min_effective_signals=2.0,
    )
    assert dec.action == "hold"
    assert dec.new_stage == PromotionStage.SHADOW
    assert any("Mature trading days 1 < 5" in r for r in dec.blocking_reasons)

    # Re-register SAME strategy_id with changed rule -> decay report must NOT pool old hash outcomes
    strat_v2 = _make_strategy(sid="mixed_strat", market=MarketType.US, rsi_max=95.0)
    spec_v2 = ledger.register_candidate(strat_v2, _eval_report(), force_reset=True)
    assert spec_v2.strategy_hash != ledger.load_commits("mixed_strat")[0].strategy_hash
    rep_v2 = ledger.build_decay_report("mixed_strat", horizon_days=5, live_only=False)
    assert rep_v2.mature_signals == 0
    assert rep_v2.total_committed_signals == 0


def test_round3_concurrent_commits_and_cutoff_safe_non_overwriting_backfill(
    tmp_path: Path,
) -> None:
    ledger = ShadowLedger(tmp_path / "conc_ledger")
    bars = _make_bars(date(2025, 1, 2), 65, step=0.006)
    trading_days = pd.to_datetime(bars["date"]).dt.date.tolist()
    market = {"AAPL": bars.copy(), "MSFT": bars.copy()}

    strats = [_make_strategy(sid=f"conc_{i}", market=MarketType.US) for i in range(6)]
    as_of_d = trading_days[22]

    def _emit(s: Strategy) -> str:
        return ledger.commit_signals(s, market, as_of=as_of_d, is_replay=False).commit_id

    with ThreadPoolExecutor(max_workers=6) as pool:
        commit_ids = list(pool.map(_emit, strats))
    assert len(commit_ids) == 6

    ok, errs = ledger.verify_chain()
    assert ok is True, errs
    assert len(ledger.load_commits()) == 6

    # Earlier-cutoff safety & immutability of already matured outcomes
    ledger_cut = ShadowLedger(tmp_path / "cut_ledger")
    s_cut = _make_strategy(sid="cut_strat", market=MarketType.US)
    ledger_cut.commit_signals(s_cut, {"AAPL": bars}, as_of=trading_days[20], is_replay=False)

    # Mature up to day 35 (T+5 and T+10 both mature)
    ev_late = ledger_cut.backfill_outcomes({"AAPL": bars}, evaluated_as_of=trading_days[35])
    assert ev_late[0].horizons[5].mature is True
    assert ev_late[0].horizons[10].mature is True
    orig_ret_5 = ev_late[0].horizons[5].tradable_post_cost_return

    # Re-evaluate at EARLIER cutoff day 27: T+5 matured at day 25 (<=27), T+10 at day 30 (>27 must NOT leak!)
    ev_early = ledger_cut.backfill_outcomes({"AAPL": bars}, evaluated_as_of=trading_days[27])
    assert ev_early[0].horizons[5].mature is True
    assert ev_early[0].horizons[10].mature is False

    # Mutate BOTH T+5 (bar 25) and T+10 (bar 30) prices after earlier cutoff -> both stay immutable!
    orig_ret_10 = ev_late[0].horizons[10].tradable_post_cost_return
    mutated = bars.copy()
    mutated.loc[25, "close"] = 1.0
    mutated.loc[30, "close"] = 1.0
    ev_mutated = ledger_cut.backfill_outcomes({"AAPL": mutated}, evaluated_as_of=trading_days[45])
    assert ev_mutated[0].horizons[5].tradable_post_cost_return == pytest.approx(orig_ret_5)
    assert ev_mutated[0].horizons[10].tradable_post_cost_return == pytest.approx(orig_ret_10)

    # Pre-upgrade legacy commit JSON (without strategy_hash/market fields) still passes verify_chain()
    legacy_ledger = ShadowLedger(tmp_path / "legacy_ledger")
    import hashlib
    import json

    legacy_payload = {
        "commit_id": "cmt_20240101_00000001",
        "strategy_id": "legacy_v1",
        "as_of": "2024-01-01",
        "committed_at": "2024-01-01T00:00:00Z",
        "market_temperature": "normal",
        "total_raw_candidates": 0,
        "sector_capped_count": 0,
        "is_replay": False,
        "signals": [],
        "prev_hash": "0" * 64,
        "record_hash": "",
    }
    legacy_raw = json.dumps(
        legacy_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    legacy_payload["record_hash"] = hashlib.sha256(legacy_raw.encode("utf-8")).hexdigest()
    legacy_ledger.commits_file.write_text(json.dumps(legacy_payload) + "\n", encoding="utf-8")
    legacy_ledger.head_hash_file.write_text(legacy_payload["record_hash"] + "\n", encoding="utf-8")
    ok_leg, errs_leg = legacy_ledger.verify_chain()
    assert ok_leg is True, errs_leg

    # Replay losses must NOT halt a candidate that already has live commits
    live_safe = ShadowLedger(tmp_path / "live_safe")
    s_live = _make_strategy(sid="live_safe_v1", market=MarketType.US)
    live_safe.register_candidate(s_live, _eval_report())
    live_bars_up = _make_bars(date(2025, 1, 2), 65, step=0.008)
    up_days = pd.to_datetime(live_bars_up["date"]).dt.date.tolist()
    live_bars_down = _make_bars(date(2025, 1, 2), 65, step=-0.018)
    # 16 losing replay commits earlier, then 2 live commits on winning bars
    for d in up_days[20:36]:
        live_safe.commit_signals(s_live, {"AAPL": live_bars_down}, as_of=d, is_replay=True)
    live_safe.commit_signals(s_live, {"AAPL": live_bars_up}, as_of=up_days[37], is_replay=False)
    live_safe.backfill_outcomes({"AAPL": live_bars_up}, evaluated_as_of=up_days[-1])
    dec_safe = live_safe.evaluate_promotion(
        s_live.meta.id,
        horizon_days=5,
        min_trading_days=5,
        min_mature_signals=5,
        min_effective_signals=2.0,
    )
    assert dec_safe.action == "hold"
    assert dec_safe.new_stage == PromotionStage.SHADOW


def test_round3_t_plus_1_gap_down_stop_and_benchmark_coverage_gate(
    tmp_path: Path,
) -> None:
    ledger = ShadowLedger(tmp_path / "t1_ledger")
    s_cn = _make_strategy(sid="cn_t1", market=MarketType.A_SHARE, stop_pct=0.05)
    ledger.register_candidate(s_cn, _eval_report())

    bars = _make_bars(date(2025, 1, 2), 55, step=0.003)
    commit_idx = 20
    commit_date = bars["date"].iloc[commit_idx]
    market_cn = {"600519": bars.copy()}
    ledger.commit_signals(s_cut := s_cn, market_cn, as_of=commit_date, is_replay=False)

    entry_ref = float(bars["close"].iloc[commit_idx])
    stop_lvl = round(entry_ref * 0.95, 4)

    # Bar 21 (pos=0, purchase day under next_open): intraday low dips -> MUST NOT trigger under T+1!
    bars.loc[21, "open"] = entry_ref
    bars.loc[21, "low"] = stop_lvl * 0.92
    bars.loc[21, "close"] = entry_ref

    # Bar 22 (pos=1, T+1): tradable gap-down open (-6.9% inside +-10% main board!) -> fills at gap_open!
    gap_open = round(stop_lvl * 0.98, 4)
    bars.loc[22, "open"] = gap_open
    bars.loc[22, "high"] = gap_open * 1.01
    bars.loc[22, "low"] = gap_open * 0.99
    bars.loc[22, "close"] = gap_open

    bench = _make_bars(date(2025, 1, 2), 55, step=0.001)
    bench.loc[21, "open"] = 100.0
    bench.loc[22, "close"] = 102.0

    events = ledger.backfill_outcomes(
        {"600519": bars},
        evaluated_as_of=bars["date"].iloc[-1],
        benchmark_df=bench,
        round_trip_cost=0.0,
    )
    hor5 = events[0].horizons[5]
    assert hor5.mature is True
    assert hor5.stopped_out is True
    assert hor5.exit_date == bars["date"].iloc[22]
    assert hor5.horizon_end_date == bars["date"].iloc[25]
    expected_tradable = (gap_open / entry_ref) - 1.0
    assert hor5.tradable_post_cost_return == pytest.approx(expected_tradable, abs=1e-4)
    assert hor5.benchmark_available is True
    assert hor5.benchmark_return == pytest.approx(0.02, abs=1e-4)

    # Early-stopped T+20 horizon MUST NOT leak as mature into an earlier T+3 cutoff!
    early_view = ledger.backfill_outcomes(
        {"600519": bars},
        evaluated_as_of=bars["date"].iloc[24],
        benchmark_df=bench,
        round_trip_cost=0.0,
    )
    assert early_view[0].horizons[20].mature is False

    # Benchmark missing on early stop before first bench bar -> benchmark_available=False (fresh ledger)
    ledger_late_bm = ShadowLedger(tmp_path / "late_bm")
    ledger_late_bm.register_candidate(s_cn, _eval_report())
    ledger_late_bm.commit_signals(s_cn, market_cn, as_of=commit_date, is_replay=False)
    bench_late = bench.iloc[25:].copy()
    events_late_bm = ledger_late_bm.backfill_outcomes(
        {"600519": bars},
        evaluated_as_of=bars["date"].iloc[-1],
        benchmark_df=bench_late,
        round_trip_cost=0.0,
    )
    assert events_late_bm[0].horizons[5].benchmark_available is False

    # require_benchmark=True blocks promotion when benchmark_df is missing despite positive committed returns
    ledger_nobm = ShadowLedger(tmp_path / "nobm")
    s_us = _make_strategy(sid="nobm_us", market=MarketType.US, stop_pct=0.05)
    ledger_nobm.register_candidate(s_us, _eval_report())
    us_bars = _make_bars(date(2025, 1, 2), 65, step=0.006)
    us_days = pd.to_datetime(us_bars["date"]).dt.date.tolist()
    us_mkt = {f"U{i}": us_bars.copy() for i in range(4)}
    for d in us_days[20:28]:
        ledger_nobm.commit_signals(s_us, us_mkt, as_of=d, is_replay=False)
    ledger_nobm.backfill_outcomes(us_mkt, evaluated_as_of=us_days[-1], benchmark_df=None)
    dec_req = ledger_nobm.evaluate_promotion(
        s_us.meta.id,
        horizon_days=5,
        min_trading_days=5,
        min_mature_signals=5,
        min_effective_signals=2.0,
        require_benchmark=True,
    )
    assert dec_req.action == "hold"
    assert any("Benchmark coverage" in r for r in dec_req.blocking_reasons)

    # Uncommitted orphan outcomes in outcomes.jsonl are ignored once commits exist
    from alphaevo.forward.contracts import OutcomeEvent

    with ledger_nobm.outcomes_file.open("a", encoding="utf-8") as fh:
        ev_orphan = OutcomeEvent(
            signal_id="orphan_fake_1",
            strategy_id=s_us.meta.id,
            symbol="FAKE",
            sector="tech",
            as_of=date(2025, 6, 1),
            evaluated_as_of=date(2025, 6, 15),
            horizons={
                5: OutcomeHorizon(horizon_days=5, mature=True, tradable_post_cost_return=0.5)
            },
        )
        fh.write(ev_orphan.model_dump_json() + "\n")
    dec_after_orphan = ledger_nobm.build_decay_report(s_us.meta.id, horizon_days=5, live_only=True)
    assert dec_after_orphan.mature_signals <= 32
    _ = s_cut


def test_round4_rules_prev_close_fallback_boards_open_buy_and_eod_t1() -> None:
    checker = MarketRuleChecker()
    rules = MarketRuleConfig(t_plus_1=True, limit_up_down=True, suspension=True)

    # Missing prev_close column -> falls back to df['close'].iloc[idx - 1]
    df_no_prev = pd.DataFrame(
        [
            {
                "date": date(2025, 1, 2),
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1000.0,
            },
            {
                "date": date(2025, 1, 3),
                "open": 10.5,
                "high": 11.5,
                "low": 10.5,
                "close": 11.5,
                "volume": 1000.0,
            },
            {
                "date": date(2025, 1, 4),
                "open": 11.5,
                "high": 15.0,
                "low": 11.5,
                "close": 15.0,
                "volume": 1000.0,
            },
        ]
    )
    assert (
        checker.can_buy(df_no_prev, 1, rules, symbol="600519") is False
    )  # +15% main board blocked
    assert (
        checker.can_buy(df_no_prev, 1, rules, symbol="301001") is True
    )  # +15% ChiNext 301 allowed (20%)
    assert (
        checker.can_buy(df_no_prev, 1, rules, symbol="689009") is True
    )  # +15% STAR 689 allowed (20%)
    assert checker.can_buy(df_no_prev, 2, rules, symbol="920001") is False  # +30.4% BSE 920 blocked

    # BacktestEngine: next_open uses open price check (not EOD close lookahead!)
    rows_buy = []
    for i in range(42):
        d = date(2025, 1, 1) + timedelta(days=i)
        o = h = lo = c = 100.0
        if i == 31:
            # Opens flat at 100 (tradable!), closes limit-up at 110
            o = 100.0
            h = 110.0
            lo = 99.5
            c = 110.0
        rows_buy.append(
            {"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": 1_000_000.0}
        )
    df_buy = pd.DataFrame(rows_buy)
    strat_buy = _make_strategy(
        sid="cn_open", market=MarketType.A_SHARE, stop_pct=0.05, timing="next_open"
    )
    bt_buy = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=25).run(
        strat_buy,
        {"600519": df_buy},
        SampleBatch(
            batch_id="cn_open",
            strategy_id=strat_buy.meta.id,
            symbols=["600519"],
            date_range=(date(2025, 1, 1), date(2025, 2, 20)),
        ),
    )
    assert any(s.signal_date == date(2025, 2, 1) for s in bt_buy.signals)

    # Intraday stop fills at -5% (above -10% limit-down) even when EOD closes limit-down (-11%)
    rows_stop = []
    for i in range(42):
        d = date(2025, 1, 1) + timedelta(days=i)
        o = h = lo = c = 100.0
        if i == 32:
            o = 99.0
            h = 99.5
            lo = 94.0
            c = 89.0
        rows_stop.append(
            {"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": 1_000_000.0}
        )
    df_stop = pd.DataFrame(rows_stop)
    bt_stop = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=25).run(
        strat_buy,
        {"600519": df_stop},
        SampleBatch(
            batch_id="cn_stop",
            strategy_id=strat_buy.meta.id,
            symbols=["600519"],
            date_range=(date(2025, 1, 1), date(2025, 2, 20)),
        ),
    )
    assert any(
        s.signal_date == date(2025, 2, 1) and s.exit_reason == ExitReason.STOP_LOSS
        for s in bt_stop.signals
    )

    # End-of-data last-bar entry must NOT close with holding_days=0 under A-share T+1
    df_last = _make_bars(date(2025, 1, 2), 36, step=0.0)
    # Only allow entry on bar 34 -> opens at bar 35 (very last bar!)
    df_last["close"] = 100.0
    df_last["open"] = 100.0
    df_last.loc[34, "close"] = (
        70.0  # artificial drop won't trigger rsi_14<=100; force signal via volume ratio
    )
    strat_last = Strategy(
        meta=StrategyMeta(
            id="last_bar_t1",
            name="last",
            market=MarketType.A_SHARE,
            category=StrategyCategory.TREND,
        ),
        description="Trigger only on high volume spike",
        entry=StrategyEntry(
            logic="and",
            triggers=[StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=3.0)],
            execution=ExecutionConfig(timing="next_open", slippage=0.0),
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.05),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=5,
        ),
    )
    df_last.loc[34, "volume"] = 10_000_000.0
    bt_last = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=25).run(
        strat_last,
        {"600519": df_last},
        SampleBatch(
            batch_id="last_t1",
            strategy_id=strat_last.meta.id,
            symbols=["600519"],
            date_range=(date(2025, 1, 2), date(2025, 3, 15)),
        ),
    )
    assert len(bt_last.signals) == 1
    assert bt_last.signals[0].exit_price is None
    assert bt_last.executed_signals == 0

    # Same-day limit-down stop + sellable take-profit falls back to take-profit
    rows_fb = []
    for i in range(36):
        d = date(2025, 1, 1) + timedelta(days=i)
        o = h = lo = c = 100.0
        if i == 32:
            # Opens limit-down at 89 (below 5% stop -> unfillable!), then rallies to 110 TP target!
            o = 89.0
            lo = 89.0
            h = 110.0
            c = 109.0
        rows_fb.append(
            {"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": 1_000_000.0}
        )
    bt_fb = BacktestEngine(slippage=0.0, commission=0.0, min_data_days=25).run(
        strat_buy,
        {"600519": pd.DataFrame(rows_fb)},
        SampleBatch(
            batch_id="cn_fb",
            strategy_id=strat_buy.meta.id,
            symbols=["600519"],
            date_range=(date(2025, 1, 1), date(2025, 2, 20)),
        ),
    )
    assert any(
        s.signal_date == date(2025, 2, 1) and s.exit_reason == ExitReason.TAKE_PROFIT
        for s in bt_fb.signals
    )


def test_round5_portfolio_causal_sizing_and_utilization() -> None:
    sigs = [
        TradeSignal(
            symbol=f"S{i}",
            signal_date=date(2024, 6, 3),
            direction=SignalDirection.LONG,
            entry_price=50.0,
            exit_price=55.0,
            exit_date=date(2024, 6, 10),
            return_pct=0.10,
            holding_days=5,
            exit_reason=ExitReason.TAKE_PROFIT,
        )
        for i in range(5)
    ]
    market_bars = {
        f"S{i}": pd.DataFrame([{"date": date(2024, 6, d), "close": 50.0} for d in range(3, 11)])
        for i in range(5)
    }
    res = PortfolioBacktester(
        PortfolioConfig(initial_capital=100_000.0, max_positions=5, position_size_pct=0.20)
    ).simulate(
        BacktestResult(
            strategy_id="p5", batch_id="b5", signals=sigs, total_signals=5, executed_signals=5
        ),
        market_data=market_bars,
    )
    assert res.total_trades == 5
    # Equal 20% slices from start-of-day equity (no geometric decay!)
    assert all(t.capital_allocated == pytest.approx(20_000.0) for t in res.trades)
    assert res.capital_utilization > 0.80

    # Same-day causality: max_positions=1, pos A exits on June 10, signal B enters June 10 -> skipped!
    sig_a = TradeSignal(
        symbol="A",
        signal_date=date(2024, 6, 3),
        direction=SignalDirection.LONG,
        entry_price=50.0,
        exit_price=75.0,
        exit_date=date(2024, 6, 10),
        return_pct=0.50,
        holding_days=5,
        exit_reason=ExitReason.TAKE_PROFIT,
    )
    sig_b = TradeSignal(
        symbol="B",
        signal_date=date(2024, 6, 10),
        direction=SignalDirection.LONG,
        entry_price=50.0,
        exit_price=55.0,
        exit_date=date(2024, 6, 14),
        return_pct=0.10,
        holding_days=4,
        exit_reason=ExitReason.TAKE_PROFIT,
    )
    res_causal = PortfolioBacktester(
        PortfolioConfig(initial_capital=100_000.0, max_positions=1, position_size_pct=1.0)
    ).simulate(
        BacktestResult(
            strategy_id="p_causal",
            batch_id="b_causal",
            signals=[sig_a, sig_b],
            total_signals=2,
            executed_signals=2,
        )
    )
    assert res_causal.total_trades == 1
    assert res_causal.trades[0].signal.symbol == "A"


def test_round6_allocator_zero_weight_dead_strategy_and_proxy_gap_nat() -> None:
    bars = _make_bars(date(2025, 1, 2), 90, step=0.004)
    market = {"AAPL": bars, "MSFT": _make_bars(date(2025, 1, 2), 90, step=0.003)}

    active_s = _make_strategy(sid="active_s", market=MarketType.US, rsi_max=100.0)
    dead_s = _make_strategy(sid="dead_s", market=MarketType.US, rsi_max=-50.0)

    rep = RegimeAdaptiveAllocator(
        RiskBudgetConfig(lookback_days=20, max_strategy_weight=0.60)
    ).allocate([active_s, dead_s], market)

    # Once active_s accumulates non-zero returns, dead_s gets 0.0 weight (not 60% or cap excess!)
    tail_weights = [p.weights["dead_s"] for p in rep.daily_curve[-15:]]
    assert all(w == 0.0 for w in tail_weights)
    assert rep.strategies[1].signal_count == 0

    # Proxy builder tolerates NaT rows and does NOT spike +100% across a multi-day suspension gap
    all_dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(6)]
    gap_df = pd.DataFrame(
        [
            {"date": all_dates[0], "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
            {"date": pd.NaT, "open": 99.0, "high": 99.0, "low": 99.0, "close": 99.0},
            {"date": all_dates[5], "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
        ]
    )
    proxy = _build_causal_market_proxy(all_dates, {"GAP": gap_df})
    assert len(proxy) == 6
    assert proxy["close"].iloc[-1] == pytest.approx(100.0)

    # _apply_weight_cap conserves budget across multiple active under-cap strategies without waking inactive 0s
    from alphaevo.backtest.allocator import _apply_weight_cap

    capped = _apply_weight_cap([0.70, 0.25, 0.05, 0.0], 0.40)
    assert capped == pytest.approx([0.40, 0.40, 0.20, 0.0], abs=1e-9)


def test_round2_to_6_codex_followup_invariants(tmp_path: Path) -> None:
    # 1. ACTIVE candidate with zero live commits must NOT be halted by replay losses
    act_ledger = ShadowLedger(tmp_path / "act_ledger")
    s_act = _make_strategy(sid="act_strat", market=MarketType.US)
    act_ledger.register_candidate(s_act, _eval_report(), stage=PromotionStage.ACTIVE)
    down_bars = _make_bars(date(2025, 1, 2), 65, step=-0.018)
    down_days = pd.to_datetime(down_bars["date"]).dt.date.tolist()
    for d in down_days[20:36]:
        act_ledger.commit_signals(s_act, {"AAPL": down_bars}, as_of=d, is_replay=True)
    act_ledger.backfill_outcomes({"AAPL": down_bars}, evaluated_as_of=down_days[-1])
    dec_act = act_ledger.evaluate_promotion(s_act.meta.id, horizon_days=5)
    assert dec_act.action != "halt"
    assert dec_act.new_stage == PromotionStage.ACTIVE

    # 2. Late benchmark arrival upgrades benchmark_available=False -> True on previously matured horizon
    late_bm_ledger = ShadowLedger(tmp_path / "late_bm_upgrade")
    s_bm = _make_strategy(sid="bm_up", market=MarketType.US)
    late_bm_ledger.register_candidate(s_bm, _eval_report())
    up_bars = _make_bars(date(2025, 1, 2), 45, step=0.005)
    bm_bars = _make_bars(date(2025, 1, 2), 45, step=0.002)
    commit_d = up_bars["date"].iloc[20]
    late_bm_ledger.commit_signals(s_bm, {"AAPL": up_bars}, as_of=commit_d, is_replay=False)
    ev1 = late_bm_ledger.backfill_outcomes(
        {"AAPL": up_bars}, evaluated_as_of=up_bars["date"].iloc[-1], benchmark_df=None
    )
    assert ev1[0].horizons[5].mature is True
    assert ev1[0].horizons[5].benchmark_available is False
    ev2 = late_bm_ledger.backfill_outcomes(
        {"AAPL": up_bars}, evaluated_as_of=up_bars["date"].iloc[-1], benchmark_df=bm_bars
    )
    assert ev2[0].horizons[5].benchmark_available is True

    # 3. Synthetic zero-observation proxy window returns benchmark_available=False
    zero_obs_dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(6)]
    gap_df = pd.DataFrame(
        [
            {"date": zero_obs_dates[0], "close": 100.0},
            {"date": zero_obs_dates[5], "close": 100.0},
        ]
    )
    proxy_bm = _build_causal_market_proxy(zero_obs_dates, {"GAP": gap_df})
    _, ok_zero_obs = ShadowLedger._compute_benchmark_return(
        proxy_bm, entry_date=zero_obs_dates[1], exit_date=zero_obs_dates[4], horizon_days=3
    )
    assert ok_zero_obs is False

    # 4. PortfolioBacktester marks unclosed terminal holdings when market_data is provided
    open_sig = TradeSignal(
        symbol="600519",
        signal_date=date(2025, 1, 2),
        direction=SignalDirection.LONG,
        entry_price=100.0,
        exit_price=None,
        exit_date=None,
        exit_reason=None,
        return_pct=0.0,
        holding_days=0,
    )
    open_mkt = {
        "600519": pd.DataFrame(
            [
                {"date": date(2025, 1, 2), "close": 100.0},
                {"date": date(2025, 1, 3), "close": 80.0},
            ]
        )
    }
    res_open = PortfolioBacktester(
        PortfolioConfig(initial_capital=100_000.0, max_positions=1, position_size_pct=1.0)
    ).simulate(
        BacktestResult(
            strategy_id="open_mtm",
            batch_id="b_open",
            signals=[open_sig],
            total_signals=1,
            executed_signals=0,
        ),
        market_data=open_mkt,
    )
    assert res_open.final_equity == pytest.approx(80_000.0)
    assert res_open.max_drawdown == pytest.approx(0.20, abs=1e-4)

    # 5. CLI shadow emit rejects --no-replay on offline bundled data
    from typer.testing import CliRunner

    from alphaevo.cli.main import app

    r_bad = CliRunner().invoke(
        app, ["shadow", "emit", "ma_crossover_v1", "--no-replay"], env={"NO_COLOR": "1"}
    )
    assert r_bad.exit_code != 0

    # 6. Day-1 drawdown from initial_capital is captured (100k -> 70k reports 0.30, not 0.0)
    day1_loss = TradeSignal(
        symbol="AAPL",
        signal_date=date(2025, 1, 2),
        direction=SignalDirection.LONG,
        entry_price=100.0,
        exit_price=70.0,
        exit_date=date(2025, 1, 2),
        return_pct=-0.30,
        holding_days=0,
        exit_reason=ExitReason.STOP_LOSS,
    )
    res_day1 = PortfolioBacktester(
        PortfolioConfig(initial_capital=100_000.0, max_positions=1, position_size_pct=1.0)
    ).simulate(
        BacktestResult(
            strategy_id="d1_dd",
            batch_id="d1_dd",
            signals=[day1_loss],
            total_signals=1,
            executed_signals=1,
        )
    )
    assert res_day1.max_drawdown == pytest.approx(0.30, abs=1e-4)

    # 7. Equal-weight benchmark proxy preserves entry-day intraday open->close return
    proxy_dates = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)]
    intraday_mkt = {
        sym: pd.DataFrame(
            [
                {
                    "date": proxy_dates[0],
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                },
                {
                    "date": proxy_dates[1],
                    "open": 100.0,
                    "high": 110.0,
                    "low": 100.0,
                    "close": 110.0,
                },
                {
                    "date": proxy_dates[2],
                    "open": 110.0,
                    "high": 110.0,
                    "low": 110.0,
                    "close": 110.0,
                },
            ]
        )
        for sym in ("S1", "S2")
    }
    ew_bench = ShadowLedger.build_equal_weight_benchmark(intraday_mkt)
    ew_ret, ew_ok = ShadowLedger._compute_benchmark_return(
        ew_bench, entry_date=proxy_dates[1], exit_date=proxy_dates[2], horizon_days=2
    )
    assert ew_ok is True
    assert ew_ret == pytest.approx(0.10, abs=1e-4)

    # 8. Live commit fill date is strictly > committed_at.date() while Monday still accepts Friday session
    from datetime import datetime, timezone
    from unittest.mock import patch

    from alphaevo.forward import shadow_ledger as sl_mod
    from alphaevo.forward.contracts import BitemporalEnvelope, ForecastCommit, SetupInstance

    mon_dt = datetime(2025, 1, 6, 8, 0, tzinfo=timezone.utc)
    with patch.object(sl_mod, "datetime") as mock_dt:
        mock_dt.now.return_value = mon_dt
        mock_dt.strptime = datetime.strptime
        friday_cmt = late_bm_ledger.commit_signals(
            _make_strategy(sid="mon_fri_ok", market=MarketType.US),
            {"AAPL": _make_bars(date(2024, 12, 1), 30, step=0.002)},
            as_of=date(2025, 1, 3),
            enforce_wall_clock=True,
        )
        assert friday_cmt.as_of == date(2025, 1, 3)

    wc_ledger = ShadowLedger(tmp_path / "wc_post_close")
    s_wc = _make_strategy(sid="wc_post", market=MarketType.US)
    wc_bars = _make_bars(date(2025, 1, 6), 15, step=0.002)
    wc_sig = SetupInstance(
        signal_id="wc_1",
        strategy_id=s_wc.meta.id,
        symbol="AAPL",
        envelope=BitemporalEnvelope(as_of=date(2025, 1, 5)),
        entry_reference_price=100.0,
        planned_stop_price=95.0,
    )
    wc_cmt = ForecastCommit(
        commit_id="cmt_wc_1",
        strategy_id=s_wc.meta.id,
        market="us",
        as_of=date(2025, 1, 5),
        committed_at=datetime(2025, 1, 6, 22, 0, tzinfo=timezone.utc),
        is_replay=False,
        wall_clock_enforced=True,
        signals=[wc_sig],
        prev_hash="0" * 64,
        record_hash="",
    )
    wc_cmt.record_hash = sl_mod._canonical_commit_hash(wc_cmt)
    wc_ledger.commits_file.write_text(wc_cmt.model_dump_json() + "\n", encoding="utf-8")
    wc_ledger.head_hash_file.write_text(wc_cmt.record_hash + "\n", encoding="utf-8")
    wc_evs = wc_ledger.backfill_outcomes(
        {"AAPL": wc_bars}, evaluated_as_of=wc_bars["date"].iloc[-1]
    )
    assert wc_evs[0].horizons[5].entry_date == date(2025, 1, 7)

    # 9. Allocator MTM does NOT book stop loss on bar -2 when final stop bar has NaN close
    nan_bars = _make_bars(date(2025, 1, 2), 69, step=0.001)
    nan_bars.loc[68, ["open", "low", "close"]] = [100.0, 90.0, float("nan")]
    s_nan = _make_strategy(sid="nan_stop", stop_pct=0.05, market=MarketType.US)
    s_nan.entry.triggers = [StrategyCondition(indicator="close_above_ma20", op="==", value=True)]
    s_nan.exit.max_holding_days = 100
    r_nan = RegimeAdaptiveAllocator().allocate([s_nan], {"AAPL": nan_bars})
    assert r_nan.daily_curve[-2].portfolio_return >= 0.0
    assert r_nan.daily_curve[-1].portfolio_return < 0.0
