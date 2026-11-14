"""Unit tests for Bitemporal Forward Shadow Ledger & Stock Setup DSL."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from alphaevo.forward.contracts import OutcomeEvent, OutcomeHorizon
from alphaevo.forward.shadow_ledger import ShadowLedger
from alphaevo.models.enums import MarketRegime, MarketType, PromotionStage, StrategyCategory
from alphaevo.models.execution import AntiFitMetrics, EvaluationReport, OverallMetrics
from alphaevo.models.strategy import (
    PortfolioProfile,
    SetupSpec,
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.dsl.serializer import StrategySerializer


def _make_setup_strategy(
    *,
    sid: str = "setup_rsi_v1",
    with_setup: bool = True,
    max_per_sector: int = 2,
    max_positions: int = 6,
    market_temp_cap: bool = True,
) -> Strategy:
    setup = (
        SetupSpec(
            trigger_rule="close > 0 (always-ready test trigger)",
            planned_stop_pct=0.05,
            structural_invalidation="close < ma60",
            target_rr=2.5,
            min_rr=1.5,
        )
        if with_setup
        else None
    )
    profile = (
        PortfolioProfile(
            mode="hard_cap",
            max_per_sector=max_per_sector,
            max_positions=max_positions,
            market_temp_cap=market_temp_cap,
        )
        if with_setup
        else None
    )
    return Strategy(
        meta=StrategyMeta(
            id=sid,
            name="Setup Test Strategy",
            market=MarketType.US,
            category=StrategyCategory.REVERSAL,
        ),
        description="Test setup & portfolio hard_cap",
        entry=StrategyEntry(
            logic="and",
            triggers=[StrategyCondition(indicator="rsi_14", op="<=", value=100.0)],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.05),
            take_profit=TakeProfitConfig(type="rr", value=2.5),
            max_holding_days=10,
        ),
        setup=setup,
        portfolio_profile=profile,
    )


def _make_eval_report(sid: str = "setup_rsi_v1", *, is_overfit: bool = False) -> EvaluationReport:
    return EvaluationReport(
        strategy_id=sid,
        overall=OverallMetrics(
            win_rate=0.58,
            avg_return=0.025,
            sharpe_ratio=1.6,
            signal_count=60,
        ),
        anti_overfit=AntiFitMetrics(
            train_val_gap=0.25 if is_overfit else 0.04,
            val_test_gap=0.03,
        ),
        deflated_sharpe_prob=0.82,
        null_control_passed=True,
    )


def _make_daily_bars(
    start: date,
    days: int,
    *,
    start_price: float = 100.0,
    daily_step: float = 0.005,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    price = start_price
    for i in range(days):
        d = start + timedelta(days=i)
        o = price
        c = price * (1.0 + daily_step)
        h = max(o, c) * 1.01
        low = min(o, c) * 0.995
        rows.append(
            {
                "date": d,
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(low, 4),
                "close": round(c, 4),
                "volume": 1_000_000,
                "prev_close": round(price, 4),
            }
        )
        price = c
    return pd.DataFrame(rows)


def test_setup_dsl_roundtrip_and_legacy_yaml_compatibility() -> None:
    strat = _make_setup_strategy(with_setup=True)
    yaml_text = StrategySerializer().to_yaml(strat)
    parsed = StrategyParser().parse_yaml(yaml_text)
    assert parsed.setup is not None
    assert parsed.setup.planned_stop_pct == 0.05
    assert parsed.portfolio_profile is not None
    assert parsed.portfolio_profile.mode == "hard_cap"
    assert parsed.portfolio_profile.max_per_sector == 2

    # Legacy YAML without setup/portfolio_profile must remain clean on serialize
    legacy_path = (
        Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "rsi_reversion.yaml"
    )
    legacy = StrategyParser().parse_file(legacy_path)
    legacy_yaml = StrategySerializer().to_yaml(legacy)
    assert "setup:" not in legacy_yaml
    assert "portfolio_profile:" not in legacy_yaml


def test_shadow_ledger_strict_as_of_pit_truncation_and_hard_cap(tmp_path: Path) -> None:
    strat = _make_setup_strategy(max_per_sector=2, max_positions=4, market_temp_cap=True)
    ledger = ShadowLedger(tmp_path / "ledger")
    ledger.register_candidate(strat, _make_eval_report(strat.meta.id))

    start = date(2025, 1, 1)
    as_of = start + timedelta(days=24)
    market_clean = {f"SYM{i}": _make_daily_bars(start, 45, daily_step=0.004) for i in range(5)}
    sectors = {f"SYM{i}": "semiconductors" for i in range(5)}

    commit_clean = ledger.commit_signals(
        strat,
        market_clean,
        as_of=as_of,
        sectors=sectors,
        market_regime=MarketRegime.TRENDING_UP,
    )
    assert commit_clean.total_raw_candidates == 5
    # Sector hard_cap=2 limits 5 semiconductor names to 2 committed setups
    assert commit_clean.sector_capped_count == 2
    assert len(commit_clean.signals) == 2

    # Mutate future prices AFTER as_of by +500% — strict PIT truncation must yield identical entry prices
    market_corrupted = {sym: df.copy(deep=True) for sym, df in market_clean.items()}
    for df in market_corrupted.values():
        future_mask = pd.to_datetime(df["date"]).dt.date > as_of
        df.loc[future_mask, "close"] = df.loc[future_mask, "close"] * 6.0

    ledger2 = ShadowLedger(tmp_path / "ledger2")
    ledger2.register_candidate(strat, _make_eval_report(strat.meta.id))
    commit_fut = ledger2.commit_signals(
        strat,
        market_corrupted,
        as_of=as_of,
        sectors=sectors,
        market_regime=MarketRegime.TRENDING_UP,
    )
    assert [s.entry_reference_price for s in commit_clean.signals] == [
        s.entry_reference_price for s in commit_fut.signals
    ]

    # Market temperature cap during PANIC halves max_positions (4 -> 2 across different sectors)
    diverse_sectors = {f"SYM{i}": f"sector_{i}" for i in range(5)}
    commit_panic = ledger2.commit_signals(
        strat,
        market_clean,
        as_of=as_of,
        sectors=diverse_sectors,
        market_regime=MarketRegime.PANIC,
    )
    assert commit_panic.market_temperature == "defensive"
    assert commit_panic.sector_capped_count == 2


def test_hash_chain_detects_tampering(tmp_path: Path) -> None:
    strat = _make_setup_strategy()
    ledger = ShadowLedger(tmp_path / "ledger")
    ledger.register_candidate(strat, _make_eval_report())
    start = date(2025, 1, 1)
    market = {"AAPL": _make_daily_bars(start, 45), "MSFT": _make_daily_bars(start, 45)}

    ledger.commit_signals(strat, market, as_of=start + timedelta(days=20))
    ledger.commit_signals(strat, market, as_of=start + timedelta(days=21))
    ok, errs = ledger.verify_chain()
    assert ok is True
    assert not errs

    # Tamper first commit's total_raw_candidates in JSONL without updating hash
    lines = ledger.commits_file.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["total_raw_candidates"] = 999
    lines[0] = json.dumps(first)
    ledger.commits_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok_after, errs_after = ledger.verify_chain()
    assert ok_after is False
    assert errs_after


def test_immature_labels_neff_and_asymmetric_promotion_halt(tmp_path: Path) -> None:
    strat = _make_setup_strategy(max_per_sector=5, max_positions=5)
    ledger = ShadowLedger(tmp_path / "ledger")

    # Overfit candidate cannot enter SHADOW
    with pytest.raises(ValueError, match="anti-overfit"):
        ledger.register_candidate(strat, _make_eval_report(is_overfit=True))

    ledger.register_candidate(strat, _make_eval_report())
    start = date(2025, 1, 1)
    as_of = start + timedelta(days=20)
    market = {f"S{i}": _make_daily_bars(start, 50, daily_step=0.008) for i in range(4)}

    ledger.commit_signals(strat, market, as_of=as_of, sectors={f"S{i}": "tech" for i in range(4)})
    # Evaluate 7 days after commitment -> T+5 is mature, T+10 and T+20 must stay immature!
    events_day7 = ledger.backfill_outcomes(market, evaluated_as_of=as_of + timedelta(days=7))
    assert len(events_day7) == 4
    assert events_day7[0].horizons[5].mature is True
    assert events_day7[0].horizons[10].mature is False
    assert events_day7[0].horizons[20].mature is False

    # Promotion must block when < 30 days / < 100 mature signals
    dec_block = ledger.evaluate_promotion(strat.meta.id, horizon_days=10)
    assert dec_block.action == "hold"
    assert dec_block.new_stage == PromotionStage.SHADOW
    assert dec_block.blocking_reasons

    # Cross-sectional clustering N_eff test: 10 same-day same-sector events << 10
    same_day_events = [
        OutcomeEvent(
            signal_id=f"sig_{i}",
            strategy_id=strat.meta.id,
            symbol=f"C{i}",
            sector="semiconductors",
            as_of=as_of,
            evaluated_as_of=as_of + timedelta(days=12),
            horizons={
                10: OutcomeHorizon(
                    horizon_days=10,
                    mature=True,
                    tradable_post_cost_return=0.04,
                    excess_return=0.03,
                )
            },
        )
        for i in range(10)
    ]
    n_eff = ShadowLedger.compute_effective_sample_size(same_day_events, horizon_days=10)
    assert n_eff < 3.0

    # Inject 35 separate-day positive mature outcomes + fake empty chain -> promotes SHADOW -> CANARY
    ledger_promo = ShadowLedger(tmp_path / "promo")
    ledger_promo.register_candidate(strat, _make_eval_report())
    multi_day_events = [
        OutcomeEvent(
            signal_id=f"m_{d}_{j}",
            strategy_id=strat.meta.id,
            symbol=f"SYM{j}",
            sector=f"sec_{j}",
            as_of=start + timedelta(days=d),
            evaluated_as_of=start + timedelta(days=d + 12),
            horizons={
                10: OutcomeHorizon(
                    horizon_days=10,
                    mature=True,
                    tradable_post_cost_return=0.025,
                    excess_return=0.015,
                )
            },
        )
        for d in range(32)
        for j in range(2)
    ]
    with ledger_promo.outcomes_file.open("w", encoding="utf-8") as fh:
        for ev in multi_day_events:
            fh.write(ev.model_dump_json() + "\n")

    dec_promo = ledger_promo.evaluate_promotion(
        strat.meta.id,
        horizon_days=10,
        min_trading_days=30,
        min_mature_signals=50,
        min_effective_signals=30.0,
    )
    assert dec_promo.action == "promote"
    assert dec_promo.new_stage == PromotionStage.CANARY

    # Asymmetric fast halt on severe losing outcomes -> HALTED immediately
    losing_events = [
        OutcomeEvent(
            signal_id=f"loss_{i}",
            strategy_id=strat.meta.id,
            symbol=f"L{i}",
            sector="retail",
            as_of=start + timedelta(days=i),
            evaluated_as_of=start + timedelta(days=i + 12),
            horizons={
                10: OutcomeHorizon(
                    horizon_days=10,
                    mature=True,
                    tradable_post_cost_return=-0.06,
                    excess_return=-0.07,
                )
            },
        )
        for i in range(16)
    ]
    with ledger_promo.outcomes_file.open("w", encoding="utf-8") as fh:
        for ev in losing_events:
            fh.write(ev.model_dump_json() + "\n")

    dec_halt = ledger_promo.evaluate_promotion(strat.meta.id, horizon_days=10)
    assert dec_halt.action == "halt"
    assert dec_halt.new_stage == PromotionStage.HALTED
