"""Scientific Integrity CI Gates & Architecture Convergence Tests."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaevo.alpha_factory.factor_store import FactorStore
from alphaevo.alpha_factory.factory import load_registered_factors
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.cli.demo import load_showcase_snapshot
from alphaevo.core.config import BacktestConfig
from alphaevo.forward.shadow_ledger import ShadowLedger
from alphaevo.models.enums import RSILevel, SamplingMethod
from alphaevo.models.execution import SampleBatch
from alphaevo.models.market import IndicatorContext
from alphaevo.optimizer import ExitOptimizer, ParamOptimizer
from alphaevo.reflection.experience import ExperienceQuery, ExperienceStore
from alphaevo.rsi.co_steer import BlindSpotMiner
from alphaevo.rsi.engine import StockRSIEngine
from alphaevo.rsi.program import RSICharter, parse_rsi_program
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.store import StrategyStore
from alphaevo.truth import (
    EvaluationKernel,
    NullSignalController,
    PortfolioGateConfig,
    TrialLedger,
    split_market_data_with_holdout,
)


def _make_ohlcv(n_bars: int = 280, seed: int = 17, *, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n_bars)]
    closes = [100.0]
    for _ in range(1, n_bars):
        step = drift + float(rng.normal(0.0, 0.018))
        closes.append(max(5.0, closes[-1] * (1.0 + step)))
    close_arr = np.array(closes, dtype=float)
    open_arr = close_arr * (1.0 + rng.normal(0.0, 0.005, size=n_bars))
    high_arr = np.maximum(open_arr, close_arr) * (
        1.0 + np.abs(rng.normal(0.008, 0.004, size=n_bars))
    )
    low_arr = np.minimum(open_arr, close_arr) * (
        1.0 - np.abs(rng.normal(0.008, 0.004, size=n_bars))
    )
    vol_arr = rng.integers(800_000, 3_500_000, size=n_bars).astype(float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.round(open_arr, 4),
            "high": np.round(high_arr, 4),
            "low": np.round(low_arr, 4),
            "close": np.round(close_arr, 4),
            "volume": vol_arr,
        }
    )


def _corrupt_future(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    corrupted = df.copy()
    mask = corrupted.index > idx
    for col in ("open", "high", "low", "close"):
        corrupted.loc[mask, col] = corrupted.loc[mask, col].astype(float) * 5.0
    corrupted.loc[mask, "volume"] = corrupted.loc[mask, "volume"].astype(float) * 20.0
    return corrupted


def _values_match(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b)
    return float(a) == pytest.approx(float(b), rel=1e-9, abs=1e-9)


def test_all_builtin_indicators_satisfy_pit_and_precompute_invariance() -> None:
    """CI Gate: every static and templated indicator must be strictly PIT-invariant."""
    df = _make_ohlcv(280, seed=23)
    bench_df = _make_ohlcv(280, seed=99)

    names = sorted(IndicatorRegistry._registry.keys()) + [
        "breakout_high_20d",
        "price_position_120d",
        "close_above_ma60",
        "atr_10",
        "rsi_6",
        "macd_histogram_fast8_slow21_signal5",
        "bollinger_band_width_20d_std1p5",
    ]

    for probe_idx in (130, 260):
        truncated_df = df.iloc[: probe_idx + 1].copy()
        truncated_bench = bench_df.iloc[: probe_idx + 1].copy()
        future_df = _corrupt_future(df, probe_idx)
        future_bench = _corrupt_future(bench_df, probe_idx)

        ctx_full = IndicatorContext(benchmark_df=bench_df)
        ctx_trunc = IndicatorContext(benchmark_df=truncated_bench)
        ctx_future = IndicatorContext(benchmark_df=future_bench)

        pre_full = IndicatorRegistry.precompute(names, df, ctx_full)
        pre_future = IndicatorRegistry.precompute(names, future_df, ctx_future)

        for name in names:
            scalar_full = IndicatorRegistry.compute(name, df, probe_idx, ctx_full)
            scalar_trunc = IndicatorRegistry.compute(name, truncated_df, probe_idx, ctx_trunc)
            scalar_future = IndicatorRegistry.compute(name, future_df, probe_idx, ctx_future)

            assert _values_match(scalar_full, scalar_trunc), (
                f"PIT prefix-truncation leak in indicator '{name}' at idx={probe_idx}: "
                f"{scalar_full} vs {scalar_trunc}"
            )
            assert _values_match(scalar_full, scalar_future), (
                f"PIT future-noise leak in indicator '{name}' at idx={probe_idx}: "
                f"{scalar_full} vs {scalar_future}"
            )
            assert _values_match(pre_full[name][probe_idx], scalar_full), (
                f"Precompute vs scalar mismatch in indicator '{name}' at idx={probe_idx}"
            )
            assert _values_match(pre_future[name][probe_idx], scalar_full), (
                f"Precompute PIT leak in indicator '{name}' at idx={probe_idx}"
            )


def test_tier2_zero_false_promotions_on_shuffled_null_worlds() -> None:
    """CI Gate: full StockRSIEngine loop must achieve 0 false promotions on shuffled returns."""
    data, _ = load_showcase_snapshot()
    subset = {sym: data[sym] for sym in sorted(data)[:4]}
    strat_path = (
        Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "rsi_reversion.yaml"
    )
    seed_strategy = StrategyParser().parse_file(strat_path)

    def _run_world(strat, null_market):  # type: ignore[no-untyped-def]
        engine = StockRSIEngine(db_path=":memory:")
        charter = RSICharter(
            name="ci_null_audit",
            seed_strategy_id=strat.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=2,
            require_null_control=True,
            null_permutations=30,
        )
        return engine.run(strat, null_market, charter=charter).promotions

    report = NullSignalController().verify_zero_promotion_under_shuffled_returns(
        seed_strategy,
        subset,
        _run_world,
        n_worlds=3,
    )
    assert report.passed is True
    assert report.false_promotions == 0


def test_evaluation_kernel_locked_holdout_and_portfolio_veto(tmp_path: Path) -> None:
    data, _ = load_showcase_snapshot()
    strat_path = (
        Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "ma_crossover.yaml"
    )
    strat = StrategyParser().parse_file(strat_path)

    vault, split = split_market_data_with_holdout(data, holdout_pct=0.25, warmup_bars=60)
    search_view = vault.search_data
    assert search_view
    for df in search_view.values():
        max_d = pd.to_datetime(df["date"]).dt.date.max()
        assert max_d <= split.cutoff_date

    kernel = EvaluationKernel(
        backtest_config=BacktestConfig(min_data_days=15),
        db_path=tmp_path / "kernel.db",
        min_data_days=15,
    )
    holdout_outcome = vault.evaluate_locked_holdout(strat, kernel, mode="full", record_trial=False)
    for sig in holdout_outcome.backtest_result.signals:
        assert sig.signal_date >= split.holdout_start_date

    strict_gate = PortfolioGateConfig(
        enabled=True,
        max_portfolio_drawdown=0.001,
        min_portfolio_total_return=0.50,
    )
    vetoed_outcome = kernel.evaluate(
        strat,
        data,
        mode="fast",
        portfolio_gate=strict_gate,
        record_trial=True,
    )
    assert vetoed_outcome.portfolio_verdict is not None
    assert vetoed_outcome.portfolio_verdict.passed is False
    assert vetoed_outcome.passed_all_gates is False
    assert any(reason.startswith("portfolio:") for reason in vetoed_outcome.veto_reasons)
    assert TrialLedger(tmp_path / "kernel.db").count_family_trials(strat.meta.family_id) == 1


def test_optimizers_record_trials_into_trial_ledger(tmp_path: Path) -> None:
    data, _ = load_showcase_snapshot()
    subset = {sym: data[sym] for sym in sorted(data)[:3]}
    strat_path = (
        Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "ma_crossover.yaml"
    )
    strat = StrategyParser().parse_file(strat_path)

    dates = pd.to_datetime(next(iter(subset.values()))["date"]).dt.date
    batch = SampleBatch(
        batch_id="opt_trial_test",
        strategy_id=strat.meta.id,
        symbols=sorted(subset.keys()),
        date_range=(dates.min(), dates.max()),
        sampling_method=SamplingMethod.REPRESENTATIVE,
    )
    ledger = TrialLedger(tmp_path / "opt_trials.db")

    exit_res = ExitOptimizer(min_data_days=15).optimize(
        strat,
        subset,
        batch,
        max_candidates=6,
        trial_ledger=ledger,
    )
    after_exit = ledger.count_family_trials(strat.meta.family_id)
    assert len(exit_res.candidates) > 0
    assert after_exit >= 1

    param_res = ParamOptimizer(min_data_days=15).optimize(
        strat,
        subset,
        batch,
        max_candidates=5,
        trial_ledger=ledger,
    )
    after_param = ledger.count_family_trials(strat.meta.family_id)
    assert len(param_res.candidates) > 0
    assert after_param > after_exit


def test_co_steer_persists_factor_and_engine_converges_store_and_shadow(tmp_path: Path) -> None:
    data, _ = load_showcase_snapshot()
    sample_df = next(iter(data.values()))
    db_path = tmp_path / "converge.db"
    factor_store = FactorStore(db_path)

    strat_path = (
        Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "rsi_reversion.yaml"
    )
    strat = StrategyParser().parse_file(strat_path)
    ev = EvaluationRegistry_dummy = None
    del ev, EvaluationRegistry_dummy

    from alphaevo.models.execution import EvaluationReport, OverallMetrics

    weak_eval = EvaluationReport(
        strategy_id=strat.meta.id,
        overall=OverallMetrics(win_rate=0.40, avg_return=-0.01, signal_count=20),
    )
    decision = BlindSpotMiner(factor_store=factor_store).steer(
        strat,
        weak_eval,
        sample_df,
        allowed_levels=[RSILevel.L2_FACTOR],
        stagnant_l1_rounds=3,
    )
    assert decision.synthesized_factor_name is not None
    assert decision.persisted_to_store is True
    factor_name = decision.synthesized_factor_name
    assert factor_store.get(factor_name) is not None

    # Simulate a fresh process clearing dynamic indicators and re-hydrating from FactorStore
    assert IndicatorRegistry.unregister_dynamic(factor_name) is True
    assert not IndicatorRegistry.is_registered(factor_name)
    loaded = load_registered_factors(db_path, status="active")
    assert factor_name in loaded
    assert IndicatorRegistry.is_registered(factor_name)

    # Now verify StockRSIEngine unifies ExperienceStore score_metric, saves champion, and auto-enrolls ShadowLedger
    charter = parse_rsi_program(
        Path(__file__).resolve().parents[2] / "examples" / "rsi_stock_program.md"
    )
    charter.max_rounds = 3
    charter.max_candidates_per_round = 3
    charter.null_permutations = 30
    charter.locked_holdout_pct = 0.20

    shadow = ShadowLedger(tmp_path / "shadow_dir")
    rsi_engine = StockRSIEngine(db_path=db_path)
    res = rsi_engine.run(
        strat,
        data,
        charter=charter,
        output_dir=tmp_path / "rsi_artifacts",
        persist_champion=True,
        shadow_ledger=shadow,
        auto_enroll_shadow=True,
    )

    assert res.holdout_evaluation is not None
    assert res.run_id is not None

    exp_rows = ExperienceStore(db_path).query(ExperienceQuery(limit=20))
    assert exp_rows
    assert all(row.score_metric == "confidence_score" for row in exp_rows)

    if res.champion_promoted:
        assert StrategyStore(db_path).get(res.champion_id) is not None
        assert res.enrolled_in_shadow is True
        assert any(c.strategy_id == res.champion_id for c in shadow.list_candidates())
