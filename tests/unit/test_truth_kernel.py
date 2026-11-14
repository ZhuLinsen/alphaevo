"""Unit tests for the L0 Immutable Anti-Delusion Truth Kernel."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

from alphaevo.alpha_factory.sandbox import FactorSandbox
from alphaevo.cli.demo import _generate_synthetic_ohlcv, load_showcase_snapshot
from alphaevo.models.enums import (
    ExitReason,
    MarketType,
    RSILevel,
    SignalDirection,
    StrategyCategory,
)
from alphaevo.models.execution import TradeSignal
from alphaevo.models.strategy import (
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
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.truth.ledger import (
    TrialLedger,
    compute_deflated_sharpe,
    compute_kernel_fingerprint,
)
from alphaevo.truth.null_control import NullSignalController
from alphaevo.truth.pit_probe import probe_point_in_time_invariance


def _make_dummy_strategy(sid: str, parent_id: str | None = None) -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id=sid,
            name=sid,
            parent_id=parent_id,
            market=MarketType.US,
            category=StrategyCategory.REVERSAL,
        ),
        description="Test strategy",
        entry=StrategyEntry(
            logic="and",
            triggers=[StrategyCondition(indicator="rsi_14", op="<", value=35.0)],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.05),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=8,
        ),
    )


def test_deflated_sharpe_penalizes_multiple_testing() -> None:
    dsr_1 = compute_deflated_sharpe(1.8, n_trials=1, n_observations=120, per_trade_sharpe=0.22)
    dsr_10 = compute_deflated_sharpe(1.8, n_trials=10, n_observations=120, per_trade_sharpe=0.22)
    dsr_500 = compute_deflated_sharpe(1.8, n_trials=500, n_observations=120, per_trade_sharpe=0.22)

    assert dsr_1.dsr_probability > dsr_10.dsr_probability > dsr_500.dsr_probability
    assert dsr_500.expected_max_null_sharpe > dsr_10.expected_max_null_sharpe > 0.0


def test_trial_ledger_prevents_rename_reset_and_counts_failures() -> None:
    ledger = TrialLedger(":memory:")
    root = _make_dummy_strategy("alpha_seed_v1")
    renamed_child = _make_dummy_strategy("totally_new_name_v1", parent_id="alpha_seed_v1")

    r1 = ledger.record_trial(
        strategy=root,
        signal_count=0,
        sharpe_ratio=0.0,
        accepted=False,
        notes="zero signals",
    )
    r2 = ledger.record_trial(
        strategy=renamed_child,
        signal_count=25,
        sharpe_ratio=1.1,
        returns=[0.01, -0.005, 0.02, 0.015],
        accepted=True,
        notes="renamed child still inherits family",
    )

    assert r1.trial_family_id == "alpha_seed"
    assert r2.trial_family_id == "alpha_seed"
    assert ledger.count_family_trials("alpha_seed") == 2
    assert compute_kernel_fingerprint() != ""


def test_pit_probe_allows_causal_and_catches_disguised_lookahead() -> None:
    df = _generate_synthetic_ohlcv("TEST", days=60, seed=11)
    sandbox = FactorSandbox(timeout_seconds=10)

    causal_code = """
def compute(df, idx):
    if idx < 10:
        return 0.0
    win = df["close"].iloc[idx - 9 : idx + 1]
    return float(df["close"].iloc[idx] / win.mean() - 1.0)
"""
    assert sandbox.validate_code(causal_code)[0]
    causal_pit = probe_point_in_time_invariance(causal_code, df)
    assert causal_pit.passed is True
    assert not causal_pit.violations

    # Passes AST validator (no iloc[idx+1] or shift(-1)), yet leaks full-series global mean!
    global_mean_leak = """
def compute(df, idx):
    full_mean = float(df["close"].mean())
    return float(df["close"].iloc[idx] - full_mean)
"""
    assert sandbox.validate_code(global_mean_leak)[0]
    leak_pit = probe_point_in_time_invariance(global_mean_leak, df)
    assert leak_pit.passed is False
    assert len(leak_pit.violations) > 0

    # Also test indirect tail access via iloc[-1]
    tail_leak = """
def compute(df, idx):
    return float(df["close"].iloc[-1] - df["close"].iloc[idx])
"""
    assert sandbox.validate_code(tail_leak)[0]
    tail_pit = sandbox.probe_pit(tail_leak, df)
    assert tail_pit.passed is False
    assert sandbox.execute(tail_leak, df, verify_pit=True).success is False


def test_null_control_passes_positive_edge_and_rejects_noise() -> None:
    df = _generate_synthetic_ohlcv("EDGE", days=120, phases=[(120, 0.0, 0.012)], seed=7)
    market_data = {"EDGE": df}
    base_d = date(2024, 7, 1)

    strong_signals = [
        TradeSignal(
            symbol="EDGE",
            signal_date=base_d + timedelta(days=i),
            direction=SignalDirection.LONG,
            entry_price=50.0,
            exit_price=52.5,
            exit_date=base_d + timedelta(days=i + 5),
            exit_reason=ExitReason.TAKE_PROFIT,
            return_pct=0.05,
            holding_days=5,
        )
        for i in range(25)
    ]
    noise_signals = [
        TradeSignal(
            symbol="EDGE",
            signal_date=base_d + timedelta(days=i),
            direction=SignalDirection.LONG,
            entry_price=50.0,
            exit_price=49.8,
            exit_date=base_d + timedelta(days=i + 5),
            exit_reason=ExitReason.MAX_HOLD,
            return_pct=-0.004 if i % 2 == 0 else 0.002,
            holding_days=5,
        )
        for i in range(25)
    ]

    ctrl = NullSignalController()
    rep_strong = ctrl.evaluate_candidate_against_nulls(
        strong_signals, market_data, n_permutations=40, seed=42
    )
    rep_noise = ctrl.evaluate_candidate_against_nulls(
        noise_signals, market_data, n_permutations=40, seed=42
    )
    assert rep_strong.passed is True
    assert rep_noise.passed is False


@pytest.mark.parametrize("n_worlds", [2])
def test_zero_promotion_under_shuffled_returns(n_worlds: int) -> None:
    # Restrict evolution on purely noisy/shuffled stock returns: 0 promotions expected!
    df1 = _generate_synthetic_ohlcv("NULLA", days=140, phases=[(140, -0.0005, 0.015)], seed=101)
    df2 = _generate_synthetic_ohlcv("NULLB", days=140, phases=[(140, -0.0004, 0.015)], seed=202)
    market_data: dict[str, pd.DataFrame] = {"NULLA": df1, "NULLB": df2}
    seed_strat = StrategyParser().parse_file(
        Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "rsi_reversion.yaml"
    )

    def _run(strat: Strategy, shuffled: dict[str, pd.DataFrame]) -> int:
        engine = StockRSIEngine(db_path=":memory:")
        charter = RSICharter(
            name="null_world",
            seed_strategy_id=strat.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=2,
            require_null_control=True,
            null_permutations=30,
        )
        res = engine.run(strat, shuffled, charter=charter)
        return res.promotions

    audit = NullSignalController().verify_zero_promotion_under_shuffled_returns(
        seed_strat,
        market_data,
        _run,
        n_worlds=n_worlds,
    )
    assert audit.passed is True
    assert audit.false_promotions == 0


def test_real_snapshot_fixture_loads() -> None:
    data, _ = load_showcase_snapshot()
    assert len(data) >= 3
