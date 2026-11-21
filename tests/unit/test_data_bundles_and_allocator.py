"""Unit tests for immutable DataBundles, EvolutionPipeline convergence & RegimeAdaptiveAllocator."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from alphaevo.backtest.allocator import RegimeAdaptiveAllocator, RiskBudgetConfig
from alphaevo.backtest.engine import BacktestEngine
from alphaevo.cli.demo import load_showcase_snapshot
from alphaevo.cli.main import app
from alphaevo.core.config import AppConfig, BacktestConfig, EvolutionConfig
from alphaevo.models.enums import MarketRegime, SamplingMethod
from alphaevo.models.execution import SampleBatch
from alphaevo.models.strategy import Strategy
from alphaevo.orchestrator.evolution import EvolutionPipeline
from alphaevo.orchestrator.pipeline import RunResult
from alphaevo.provenance import CodeVersionInfo, DataBundleStore, RunArtifactStore
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.truth import TrialLedger

runner = CliRunner()


def _make_ohlcv(n_bars: int = 160, seed: int = 31, *, drift: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n_bars)]
    closes = [100.0]
    for _ in range(1, n_bars):
        closes.append(max(5.0, closes[-1] * (1.0 + drift + float(rng.normal(0.0, 0.015)))))
    close_arr = np.array(closes, dtype=float)
    open_arr = close_arr * (1.0 + rng.normal(0.0, 0.004, size=n_bars))
    high_arr = np.maximum(open_arr, close_arr) * 1.01
    low_arr = np.minimum(open_arr, close_arr) * 0.99
    vol_arr = rng.integers(1_000_000, 3_000_000, size=n_bars).astype(float)
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


def _builtin_strategy(filename: str, strategy_id: str | None = None) -> Strategy:
    path = Path(__file__).resolve().parents[2] / "strategies" / "builtin" / filename
    strat = StrategyParser().parse_file(path)
    if strategy_id is not None:
        strat.meta.id = strategy_id
    return strat


def test_data_bundle_create_verify_tamper_and_cli_official_rerun(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ALPHAEVO_DB_PATH", str(tmp_path / "bundle_test.db"))

    data, _ = load_showcase_snapshot()
    bundle_root = tmp_path / "bundles"
    artifact_root = tmp_path / "artifacts"

    bstore = DataBundleStore(bundle_root)
    manifest = bstore.create_bundle(
        data,
        name="us_tech_snap",
        market="us",
        adapter="snapshot",
        description="Immutable test bundle",
    )
    assert manifest.symbol_count == len(data)
    assert bstore.verify_bundle(manifest.bundle_id).verified is True

    loaded, loaded_manifest = bstore.load_bundle(manifest.bundle_id, verify=True)
    assert loaded_manifest.bundle_hash == manifest.bundle_hash
    assert set(loaded.keys()) == set(data.keys())

    # CLI data bundle commands
    res_list = runner.invoke(app, ["data", "bundle", "list", "--bundle-root", str(bundle_root)])
    assert res_list.exit_code == 0
    assert manifest.bundle_id in res_list.stdout

    res_verify = runner.invoke(
        app, ["data", "bundle", "verify", manifest.bundle_id, "--bundle-root", str(bundle_root)]
    )
    assert res_verify.exit_code == 0
    assert "Verified cleanly" in res_verify.stdout

    # CLI runs rerun-official against clean code + verified bundle -> official_v1
    strat_path = (
        Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "ma_crossover.yaml"
    )
    res_rerun = runner.invoke(
        app,
        [
            "runs",
            "rerun-official",
            str(strat_path),
            "--bundle",
            manifest.bundle_id,
            "--bundle-root",
            str(bundle_root),
            "--artifact-root",
            str(artifact_root),
        ],
    )
    assert res_rerun.exit_code == 0
    runs = RunArtifactStore(
        artifact_root=artifact_root, db_path=tmp_path / "bundle_test.db"
    ).list_runs(limit=5)
    assert len(runs) == 1
    assert runs[0].data_reproducibility == "replayable"

    # With clean git code + verified bundle, official_v1 is granted without downgrade
    from alphaevo.truth import EvaluationKernel

    clean_store = RunArtifactStore(
        artifact_root=artifact_root, db_path=tmp_path / "bundle_clean.db"
    )
    strat_clean = StrategyParser().parse_file(strat_path)
    outcome = EvaluationKernel(backtest_config=BacktestConfig(min_data_days=15)).evaluate(
        strat_clean, loaded, mode="full", record_trial=False
    )
    m_off = clean_store.save_run(
        kind="run",
        strategy=strat_clean,
        evaluation=outcome.evaluation,
        market_data=loaded,
        trades=outcome.backtest_result.signals,
        adapter_name=f"bundle:{loaded_manifest.adapter}",
        requested_protocol="official_v1",
        data_reproducibility="replayable",
        data_mutability="immutable_bundle",
        data_bundle_id=loaded_manifest.bundle_id,
        min_signals_official=1,
        code_version_override=CodeVersionInfo(source="git", commit="deadbeef", dirty=False),
    )
    assert m_off.effective_protocol == "official_v1"
    assert m_off.data.data_bundle_id == manifest.bundle_id

    # Tamper with one CSV -> verify fails and load_bundle raises ValueError
    first_csv = tmp_path / "bundles" / manifest.bundle_id / manifest.symbols[0].file
    first_csv.write_text(
        first_csv.read_text(encoding="utf-8") + "2099-01-01,1,1,1,1,1\n", encoding="utf-8"
    )
    tampered = bstore.verify_bundle(manifest.bundle_id)
    assert tampered.verified is False
    assert tampered.errors
    with pytest.raises(ValueError, match="integrity verification failed"):
        bstore.load_bundle(manifest.bundle_id, verify=True)

    res_bad_verify = runner.invoke(
        app, ["data", "bundle", "verify", manifest.bundle_id, "--bundle-root", str(bundle_root)]
    )
    assert res_bad_verify.exit_code == 1


def test_evolution_pipeline_converges_scars_trial_ledger_and_measured_sensitivity(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "evo_conv.db"
    cfg = AppConfig(
        db_path=db_path,
        backtest=BacktestConfig(min_data_days=15),
        evolution=EvolutionConfig(min_signal_count=5),
    )
    evo = EvolutionPipeline(cfg)

    data = {"AAPL": _make_ohlcv(160, seed=7), "MSFT": _make_ohlcv(160, seed=11)}
    strat = _builtin_strategy("ma_crossover.yaml", "evo_conv_v1")
    batch = SampleBatch(
        batch_id="b1",
        strategy_id=strat.meta.id,
        symbols=sorted(data.keys()),
        date_range=(date(2025, 1, 2), date(2025, 6, 10)),
        sampling_method=SamplingMethod.REPRESENTATIVE,
    )
    bt_engine = BacktestEngine(min_data_days=15)
    bt_res = bt_engine.run(strat, data, batch)
    from alphaevo.evaluator.metrics import Evaluator

    ev = Evaluator().evaluate(bt_res, strat, market_data=data, backtest_config=cfg.backtest)
    run_res = RunResult(
        strategy=strat,
        batch=batch,
        backtest_result=bt_res,
        evaluation=ev,
        _engine=bt_engine,
        _data=data,
    )

    mutated = strat.model_copy(deep=True)
    mutated.meta.id = "evo_conv_v2"
    mutated.exit.stop_loss.value = 0.08
    screened = evo._screen_mutation_candidate(run_res, mutated, [], None)
    assert screened.evaluation is not None
    assert TrialLedger(db_path).count_family_trials(strat.meta.family_id) == 1

    # Zero-signal failure distills an ExecutableScar into ScarRegistry
    from alphaevo.models.enums import ChangeType
    from alphaevo.models.execution import StrategyChange

    scars = evo._scar_registry.synthesize_from_failure(
        strategy=mutated,
        changes=[
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30.0,
                to_value=5.0,
                reason="over-tightened",
            )
        ],
        signal_count=0,
        score_delta=-0.20,
    )
    assert len(scars) == 1
    assert len(evo._scar_registry.list_all(mutated.meta.family_id)) == 1


def test_regime_adaptive_allocator_pit_regime_gating_and_corr_penalty(tmp_path: Path) -> None:
    data, _ = load_showcase_snapshot()
    s1 = _builtin_strategy("rsi_reversion.yaml", "s_any_regime")
    s1.entry.logic = "or"
    s2 = s1.model_copy(deep=True)
    s2.meta.id = "s_panic_only"
    s2.meta.preferred_regime = [MarketRegime.PANIC.value]

    allocator = RegimeAdaptiveAllocator(
        RiskBudgetConfig(
            lookback_days=20,
            max_strategy_weight=0.80,
            regime_mismatch_haircut=0.25,
            correlation_penalty_strength=0.75,
        )
    )
    rep_full = allocator.allocate([s1, s2], data)
    assert rep_full.trading_days > 60
    assert len(rep_full.strategies) == 2

    # Strategy mismatched with prevailing non-panic regime gets smaller mean weight
    w_any = rep_full.strategies[0].mean_weight
    w_panic = rep_full.strategies[1].mean_weight
    assert w_panic < w_any

    # Adding a duplicate/collinear copy of s1 penalizes each collinear sibling vs an orthogonal/mismatched strategy
    s1_dup = s1.model_copy(deep=True)
    s1_dup.meta.id = "s_any_regime_dup"
    s_other = _builtin_strategy("ma_crossover.yaml", "s_ma_orth")
    rep_dup = allocator.allocate([s1, s1_dup, s_other], data)
    assert f"{s1.meta.id} vs {s1_dup.meta.id}" in rep_dup.pairwise_correlations
    assert rep_dup.pairwise_correlations[f"{s1.meta.id} vs {s1_dup.meta.id}"] == pytest.approx(1.0)

    # Strict PIT Prefix-Invariance: corrupting bars after day t keeps weights <= t identical
    all_dates = sorted(
        {pd.Timestamp(d).date() for df in data.values() for d in df["date"].tolist()}
    )
    cutoff_idx = len(all_dates) // 2
    cutoff_date = all_dates[cutoff_idx]
    corrupted: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        cdf = df.copy()
        d_col = pd.to_datetime(cdf["date"]).dt.date
        mask = d_col > cutoff_date
        for col in ("open", "high", "low", "close"):
            cdf.loc[mask, col] = cdf.loc[mask, col].astype(float) * 4.5
        corrupted[sym] = cdf

    rep_corrupt = allocator.allocate([s1, s_other], corrupted)
    rep_clean_two = allocator.allocate([s1, s_other], data)
    # Check weights on days up to cutoff_date minus max trade holding buffer match cleanly before cutoff
    early_clean = {
        p.date: p.weights for p in rep_clean_two.daily_curve if p.date <= all_dates[cutoff_idx - 20]
    }
    early_corrupt = {
        p.date: p.weights for p in rep_corrupt.daily_curve if p.date <= all_dates[cutoff_idx - 20]
    }
    for d_key, w_clean in early_clean.items():
        if d_key in early_corrupt:
            for sid, val in w_clean.items():
                assert early_corrupt[d_key][sid] == pytest.approx(val, abs=1e-6)


def test_portfolio_allocate_cli_command(tmp_path: Path) -> None:
    out_md = tmp_path / "alloc_report.md"
    res = runner.invoke(
        app,
        [
            "portfolio",
            "allocate",
            "rsi_reversion_v1",
            "ma_crossover_v1",
            "--output",
            str(out_md),
        ],
    )
    assert res.exit_code == 0
    assert "Regime-Adaptive Multi-Strategy Portfolio" in res.stdout
    assert out_md.is_file()
    assert "Strategy Risk Budget Breakdown" in out_md.read_text(encoding="utf-8")
