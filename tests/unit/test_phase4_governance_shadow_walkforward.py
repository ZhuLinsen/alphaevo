"""Phase 4 unit tests: purged calendar walk-forward, L3 mechanism kernel convergence, rolling shadow simulation, and factor governance."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
from alphaevo.alpha_factory.governance import FactorGovernanceConfig, FactorGovernanceEngine
from alphaevo.cli.main import app
from alphaevo.core.config import AppConfig
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.forward.shadow_ledger import ShadowLedger
from alphaevo.models.enums import (
    ExitReason,
    MarketRegime,
    MarketType,
    PromotionStage,
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
from alphaevo.provenance.bundle import DataBundleStore
from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator
from alphaevo.strategy.dsl.parser import StrategyParser


def _make_oscillating_market(
    n_bars: int = 160, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC")
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    start = date(2025, 1, 2)
    out: dict[str, pd.DataFrame] = {}
    dates = [start + timedelta(days=i) for i in range(n_bars)]
    for s_idx, sym in enumerate(symbols):
        base = 100.0 + s_idx * 10.0
        closes: list[float] = []
        price = base
        for i in range(n_bars):
            cyc = 0.015 * np.sin(i / 4.0 + s_idx) + float(rng.normal(0.0005, 0.008))
            price = max(10.0, price * (1.0 + cyc))
            closes.append(price)
        arr = np.array(closes, dtype=float)
        out[sym] = pd.DataFrame(
            {
                "date": dates,
                "open": arr * 0.998,
                "high": arr * 1.012,
                "low": arr * 0.988,
                "close": arr,
                "volume": np.full(n_bars, 1_500_000.0),
            }
        )
    return out


def _make_trending_market(
    *,
    drift: float,
    n_bars: int = 90,
    symbols: tuple[str, ...] = ("SYM1", "SYM2"),
) -> dict[str, pd.DataFrame]:
    start = date(2025, 3, 3)
    dates = [start + timedelta(days=i) for i in range(n_bars)]
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        closes = [100.0 * ((1.0 + drift) ** i) for i in range(n_bars)]
        arr = np.array(closes, dtype=float)
        out[sym] = pd.DataFrame(
            {
                "date": dates,
                "open": arr * 0.999,
                "high": arr * 1.005,
                "low": arr * 0.995,
                "close": arr,
                "volume": np.full(n_bars, 2_000_000.0),
            }
        )
    return out


def _make_always_enter_strategy(
    strategy_id: str, indicator_name: str = "close_above_ma20"
) -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id=strategy_id,
            name=strategy_id,
            market=MarketType.US,
            category=StrategyCategory.TREND,
        ),
        description="Test strategy",
        entry=StrategyEntry(
            logic="or",
            triggers=[StrategyCondition(indicator=indicator_name, op="==", value=True)],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.05),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=5,
        ),
    )


def test_calendar_purged_walk_forward_aligns_dates_and_purges_overlap() -> None:
    signals: list[TradeSignal] = []
    start = date(2025, 1, 1)
    # 20 unique trading dates x 2 symbols = 40 signals
    for day_idx in range(20):
        sig_d = start + timedelta(days=day_idx)
        # Hold 6 days so late-training signals cross the cutoff boundary
        exit_d = sig_d + timedelta(days=6)
        for sym_idx in range(2):
            win = (day_idx + sym_idx) % 2 == 0
            signals.append(
                TradeSignal(
                    symbol=f"S{sym_idx}",
                    signal_date=sig_d,
                    direction=SignalDirection.LONG,
                    entry_price=100.0,
                    exit_price=103.0 if win else 98.0,
                    exit_date=exit_d,
                    exit_reason=ExitReason.MAX_HOLD,
                    return_pct=0.03 if win else -0.02,
                    holding_days=6,
                    regime=MarketRegime.RANGE_BOUND,
                )
            )

    evaluator = Evaluator()
    folds_purged, proto_purged = evaluator.compute_calendar_walk_forward(
        signals,
        requested_folds=2,
        train_pct=0.60,
        pass_gap=0.15,
        purge_overlap=True,
    )
    folds_unpurged, proto_unpurged = evaluator.compute_calendar_walk_forward(
        signals,
        requested_folds=2,
        train_pct=0.60,
        pass_gap=0.15,
        purge_overlap=False,
    )
    delegated_folds, delegated_proto = evaluator.compute_walk_forward(
        signals,
        requested_folds=2,
        train_pct=0.60,
        pass_gap=0.15,
        split_mode="calendar_purged",
        purge_overlap=True,
    )

    assert proto_purged.split_mode == "calendar_purged"
    assert proto_purged.purge_overlap is True
    assert proto_purged.effective_folds >= 1
    assert proto_purged.total_purged_signals > 0
    assert folds_purged[0].purged_signal_count > 0
    assert folds_purged[0].train_end_date is not None
    assert folds_purged[0].test_start_date is not None
    assert folds_purged[0].train_end_date < folds_purged[0].test_start_date
    assert folds_purged[0].train_signal_count < folds_unpurged[0].train_signal_count
    assert proto_unpurged.total_purged_signals == 0
    assert delegated_proto == proto_purged
    assert delegated_folds == folds_purged


def test_mechanism_evaluator_multi_cutoff_kernel_and_markdown() -> None:
    parser = StrategyParser()
    strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    market = _make_oscillating_market(n_bars=180)

    naive = HarnessVariant(
        variant_id="naive",
        name="Naive Harness",
        require_null_control=False,
        enable_pinned_scars=False,
    )
    guarded = HarnessVariant(
        variant_id="guarded",
        name="Guarded Kernel Harness",
        require_null_control=True,
        enable_pinned_scars=True,
        require_portfolio_gate=False,
    )
    evaluator = MechanismEvaluator()
    result = evaluator.evaluate_ab(
        strat,
        market,
        baseline_variant=naive,
        candidate_variant=guarded,
        max_rounds=1,
        cutoff_ratios=[0.55, 0.70],
        seed=42,
    )

    assert len(result.fold_comparisons) == 2
    assert result.fold_comparisons[0].cutoff_ratio == 0.55
    assert result.fold_comparisons[1].cutoff_ratio == 0.70
    assert result.winner_variant_id in {"naive", "guarded"}
    assert result.baseline_run.trials_recorded >= 1
    assert result.candidate_run.trials_recorded >= 1
    assert isinstance(result.baseline_run.forward_sharpe, float)

    md = evaluator.render_markdown(result)
    assert "# L3 Meta-RSI Mechanism Calibration Report" in md
    assert "## Cutoff Fold Breakdown" in md
    assert "Naive Harness" in md
    assert "Guarded Kernel Harness" in md


def test_shadow_ledger_rolling_simulation_promote_halt_and_cli(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # 1. Winning trend -> promotes SHADOW -> CANARY
    bull_market = _make_trending_market(drift=0.008, n_bars=75)
    bull_strat = _make_always_enter_strategy("bull_runner_v1", "close_above_ma20")
    bull_ledger = ShadowLedger(tmp_path / "bull_shadow")
    bull_dates = sorted(pd.to_datetime(bull_market["SYM1"]["date"]).dt.date.tolist())
    bull_report = bull_ledger.simulate_rolling(
        bull_strat,
        bull_market,
        start_date=bull_dates[25],
        end_date=bull_dates[-1],
        step_days=2,
        horizon_days=5,
        min_trading_days=5,
        min_mature_signals=6,
        min_effective_signals=3.0,
        max_drawdown_halt=0.35,
        is_replay=False,
    )
    assert bull_report.chain_verified is True
    assert bull_report.total_commits >= 10
    assert bull_report.total_committed_signals >= 10
    assert bull_report.final_stage in {PromotionStage.CANARY, PromotionStage.ACTIVE}
    assert bull_report.promoted is True
    assert not bull_report.halted
    assert "SHA-256 Chain Verified: True" in ShadowLedger.render_simulation_markdown(bull_report)

    # 2. Bleeding strategy -> HALTED immediately on forward bleed/drawdown gate
    bear_market = _make_trending_market(drift=-0.015, n_bars=75)
    bear_strat = Strategy(
        meta=StrategyMeta(
            id="bear_bleeder_v1",
            name="bear_bleeder_v1",
            market=MarketType.US,
            category=StrategyCategory.REVERSAL,
        ),
        description="Enters on losing bars",
        entry=StrategyEntry(
            logic="or",
            triggers=[StrategyCondition(indicator="rsi_14", op="<", value=99.0)],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.25),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
    )
    bear_ledger = ShadowLedger(tmp_path / "bear_shadow")
    bear_dates = sorted(pd.to_datetime(bear_market["SYM1"]["date"]).dt.date.tolist())
    bear_report = bear_ledger.simulate_rolling(
        bear_strat,
        bear_market,
        start_date=bear_dates[25],
        end_date=bear_dates[-1],
        step_days=1,
        horizon_days=5,
        min_trading_days=5,
        min_mature_signals=5,
        min_effective_signals=2.0,
        max_drawdown_halt=0.05,
        is_replay=False,
    )
    assert bear_report.chain_verified is True
    assert bear_report.halted is True
    assert bear_report.final_stage == PromotionStage.HALTED
    # Once halted, no additional commits are issued
    halted_indices = [
        i for i, st in enumerate(bear_report.steps) if st.stage_after == PromotionStage.HALTED
    ]
    assert halted_indices
    first_halt_idx = halted_indices[0]
    for later_step in bear_report.steps[first_halt_idx + 1 :]:
        assert later_step.committed_signals == 0

    # 3. CLI `alphaevo shadow simulate`
    import alphaevo.cli.commands.rsi as rsi_cli

    strat_yaml = tmp_path / "bull_strat.yaml"
    from alphaevo.strategy.dsl.serializer import StrategySerializer

    strat_yaml.write_text(StrategySerializer().to_yaml(bull_strat), encoding="utf-8")
    md_out = tmp_path / "shadow_sim.md"
    monkeypatch.setattr(rsi_cli, "_get_config", lambda: AppConfig())
    monkeypatch.setattr(rsi_cli, "_load_market_data", lambda live=False: bull_market)
    cli_res = CliRunner().invoke(
        app,
        [
            "shadow",
            "simulate",
            str(strat_yaml),
            "--days",
            "35",
            "--horizon",
            "5",
            "--min-days",
            "5",
            "--min-signals",
            "6",
            "--min-eff",
            "3.0",
            "--output",
            str(md_out),
        ],
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert md_out.is_file()
    assert "Rolling Forward Shadow Simulation" in md_out.read_text(encoding="utf-8")


def test_factor_governance_audit_prune_and_cli(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Construct mean-reverting dataset where current close deviation strongly predicts 5d forward rebound
    n_bars = 140
    start = date(2025, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_bars)]
    market_data: dict[str, pd.DataFrame] = {}
    for sym_idx, sym in enumerate(("F1", "F2", "F3", "F4")):
        prices = [
            100.0 * (1.0 + 0.04 * np.sin((i + sym_idx * 1.7) * np.pi / 5.0)) for i in range(n_bars)
        ]
        arr = np.array(prices, dtype=float)
        market_data[sym] = pd.DataFrame(
            {
                "date": dates,
                "open": arr * 0.999,
                "high": arr * 1.005,
                "low": arr * 0.995,
                "close": arr,
                "volume": np.full(n_bars, 1_000_000.0 + sym_idx * 10_000.0),
            }
        )

    db_path = tmp_path / "alphaevo.db"
    store = FactorStore(db_path)
    try:
        store.save(
            FactorRecord(
                name="causal_mean_reversion",
                description="5d SMA minus current close ratio (mean-reversion predictor)",
                rationale="Captures short-term reversion",
                code=(
                    "def compute(df, idx):\n"
                    "    if idx < 4:\n"
                    "        return 0.0\n"
                    "    w = df['close'].iloc[idx - 4 : idx + 1]\n"
                    "    return float(w.mean() / df['close'].iloc[idx] - 1.0)\n"
                ),
                ir=1.2,
                ic_mean=0.25,
            )
        )
        store.save(
            FactorRecord(
                name="redundant_clone_reversion",
                description="Redundant scaled clone of causal_mean_reversion",
                rationale="Near identical signal",
                code=(
                    "def compute(df, idx):\n"
                    "    if idx < 4:\n"
                    "        return 0.0\n"
                    "    w = df['close'].iloc[idx - 4 : idx + 1]\n"
                    "    return float((w.mean() / df['close'].iloc[idx] - 1.0) * 1.05 + 0.001)\n"
                ),
                ir=1.0,
                ic_mean=0.20,
            )
        )
        store.save(
            FactorRecord(
                name="future_leak_peeker",
                description="Lookahead future close peeker",
                rationale="Should fail runtime PIT probe",
                code=(
                    "def compute(df, idx):\n"
                    "    return float(df['close'].iloc[-1] / max(1e-6, df['close'].iloc[idx]) - 1.0)\n"
                ),
                ir=0.9,
                ic_mean=0.20,
            )
        )
        store.save(
            FactorRecord(
                name="decayed_zero_factor",
                description="Deterministic constant/noise with zero predictive power",
                rationale="Should classify decayed",
                code="def compute(df, idx):\n    return 0.0\n",
                ir=0.1,
                ic_mean=0.01,
            )
        )
    finally:
        store.close()

    ref_strat = _make_always_enter_strategy("gov_user_strat_v1", "causal_mean_reversion")
    gov_cfg = FactorGovernanceConfig(
        train_ratio=0.55,
        forward_horizon=5,
        min_oos_ic=0.05,
        min_oos_ir=0.20,
        min_decay_ratio=0.25,
        max_redundancy_corr=0.75,
        require_pit_invariance=True,
    )
    engine = FactorGovernanceEngine(db_path=db_path, config=gov_cfg)
    preview = engine.audit_factors(market_data, strategies=[ref_strat], auto_retire=False)

    assert preview.total_audited == 4
    by_name = {entry.name: entry for entry in preview.entries}
    assert by_name["causal_mean_reversion"].health == "healthy"
    assert by_name["causal_mean_reversion"].referenced_strategies == ["gov_user_strat_v1"]
    assert by_name["future_leak_peeker"].health == "pit_unsafe"
    assert by_name["future_leak_peeker"].pit_passed is False
    assert by_name["decayed_zero_factor"].health == "decayed"
    assert by_name["redundant_clone_reversion"].health == "redundant"
    assert by_name["redundant_clone_reversion"].max_peer_corr >= 0.75
    assert preview.retired_count == 0
    assert "Alpha Factory Governance Audit Report" in engine.render_markdown(preview)

    # CLI audit & prune backed by an immutable DataBundleStore bundle
    bundle_root = tmp_path / "bundles"
    bundle_manifest = DataBundleStore(bundle_root).create_bundle(
        market_data, name="gov_test", market="us"
    )

    import alphaevo.cli.commands.factor as factor_cli

    monkeypatch.setattr(
        factor_cli,
        "_get_config",
        lambda: SimpleNamespace(db_path=str(db_path)),
    )
    report_md = tmp_path / "factor_gov.md"
    audit_res = CliRunner().invoke(
        app,
        [
            "factor",
            "audit",
            "--bundle",
            bundle_manifest.bundle_id,
            "--bundle-root",
            str(bundle_root),
            "--min-oos-ic",
            "0.05",
            "--min-oos-ir",
            "0.20",
            "--output",
            str(report_md),
        ],
    )
    assert audit_res.exit_code == 0, audit_res.output
    assert report_md.is_file()

    prune_dry = CliRunner().invoke(
        app,
        [
            "factor",
            "prune",
            "--dry-run",
            "--bundle",
            bundle_manifest.bundle_id,
            "--bundle-root",
            str(bundle_root),
            "--min-oos-ic",
            "0.05",
            "--min-oos-ir",
            "0.20",
        ],
    )
    assert prune_dry.exit_code == 0, prune_dry.output
    assert "Would retire 3 factor(s)" in prune_dry.output

    prune_real = CliRunner().invoke(
        app,
        [
            "factor",
            "prune",
            "--bundle",
            bundle_manifest.bundle_id,
            "--bundle-root",
            str(bundle_root),
            "--min-oos-ic",
            "0.05",
            "--min-oos-ir",
            "0.20",
        ],
    )
    assert prune_real.exit_code == 0, prune_real.output
    assert "Retired 3 factor(s)" in prune_real.output

    verify_store = FactorStore(db_path)
    try:
        active_names = [f.name for f in verify_store.list_all(status="active")]
        retired_names = {f.name for f in verify_store.list_all(status="retired")}
        assert active_names == ["causal_mean_reversion"]
        assert {
            "redundant_clone_reversion",
            "future_leak_peeker",
            "decayed_zero_factor",
        } <= retired_names
    finally:
        verify_store.close()
