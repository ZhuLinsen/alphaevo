"""Unit tests for immutable run provenance, scoped leaderboards & lazy CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from alphaevo.cli.main import app
from alphaevo.core.config import AppConfig, DataConfig
from alphaevo.leaderboard import LeaderboardService
from alphaevo.models.enums import ExitReason, MarketRegime, SignalDirection
from alphaevo.models.execution import (
    AntiFitMetrics,
    EvaluationReport,
    OverallMetrics,
    TradeSignal,
)
from alphaevo.provenance import CodeVersionInfo, RunArtifactStore
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.store import StrategyStore

runner = CliRunner()


def _sample_market_data() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2025-01-02", periods=30, freq="B").date
    prices = np.linspace(100.0, 115.0, len(dates))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": 1_000_000.0,
        }
    )
    return {"AAPL": df}


def _sample_strategy(strategy_id: str = "ma_crossover_v1"):
    path = Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "ma_crossover.yaml"
    strat = StrategyParser().parse_file(path)
    strat.meta.id = strategy_id
    return strat


def _sample_eval(
    strategy_id: str,
    *,
    score: float = 0.62,
    signal_count: int = 40,
    overfit: bool = False,
) -> EvaluationReport:
    return EvaluationReport(
        strategy_id=strategy_id,
        overall=OverallMetrics(
            win_rate=0.55,
            avg_return=0.015,
            profit_loss_ratio=1.8,
            max_drawdown=0.12,
            sharpe_ratio=1.4,
            signal_count=signal_count,
            metrics_basis="daily_mtm",
        ),
        anti_overfit=AntiFitMetrics(train_val_gap=0.25 if overfit else 0.03),
        confidence_score=score,
    )


def _sample_trades() -> list[TradeSignal]:
    return [
        TradeSignal(
            symbol="AAPL",
            signal_date=date(2025, 1, 10),
            direction=SignalDirection.LONG,
            entry_price=102.0,
            exit_price=106.0,
            exit_date=date(2025, 1, 15),
            exit_reason=ExitReason.TAKE_PROFIT,
            return_pct=0.0392,
            holding_days=3,
            regime=MarketRegime.TRENDING_UP,
        )
    ]


def test_run_artifact_store_persists_immutable_bundle_and_downgrades_official(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "prov.db"
    artifact_root = tmp_path / "reports"
    store = RunArtifactStore(artifact_root=artifact_root, db_path=db_path)

    cfg = AppConfig(
        data=DataConfig(adapter="yfinance", adanos_api_key="super-secret-token"),
        db_path=db_path,
    )
    strat = _sample_strategy("ma_crossover_v1")
    market = _sample_market_data()

    # Dirty code or mutable provider requesting official_v1 must downgrade cleanly
    m_local = store.save_run(
        kind="run",
        strategy=strat,
        evaluation=_sample_eval("ma_crossover_v1", score=0.58),
        market_data=market,
        trades=_sample_trades(),
        config=cfg,
        adapter_name="yfinance",
        requested_protocol="official_v1",
        data_reproducibility="refetch_dependent",
        data_mutability="mutable_provider",
        code_version_override=CodeVersionInfo(source="git", commit="abc1234", dirty=True),
    )
    assert m_local.requested_protocol == "official_v1"
    assert m_local.effective_protocol == "full_research_v1"
    assert m_local.protocol_downgrade_reason is not None

    run_dir = Path(m_local.artifacts.artifact_dir)
    for fname in (
        "manifest.json",
        "strategy_snapshot.yaml",
        "config_snapshot.json",
        "data_fingerprint.json",
        "evaluation.json",
        "trades.jsonl",
        "report.md",
    ):
        assert (run_dir / fname).is_file()

    cfg_text = (run_dir / "config_snapshot.json").read_text(encoding="utf-8")
    assert "super-secret-token" not in cfg_text
    assert json.loads(cfg_text)["data"]["adapter"] == "yfinance"
    assert "## Run Provenance" in (run_dir / "report.md").read_text(encoding="utf-8")

    # Clean code + replayable data + passing anti-overfit qualifies for official_v1
    strat_v2 = _sample_strategy("ma_crossover_v2")
    strat_v2.exit.stop_loss.value = 0.08
    m_official = store.save_run(
        kind="run",
        strategy=strat_v2,
        evaluation=_sample_eval("ma_crossover_v2", score=0.71),
        market_data=market,
        trades=_sample_trades(),
        config=cfg,
        adapter_name="snapshot",
        requested_protocol="official_v1",
        data_reproducibility="replayable",
        data_mutability="replayable_snapshot",
        code_version_override=CodeVersionInfo(source="git", commit="abc1234", dirty=False),
    )
    assert m_official.effective_protocol == "official_v1"
    assert m_official.protocol_downgrade_reason is None

    comp = store.compare_runs(m_local.run_id, m_official.run_id)
    assert comp.comparable is False
    assert comp.score_delta == pytest.approx(0.13)
    assert "exit.stop_loss.value" in comp.changed_strategy_paths


def test_leaderboard_service_separates_local_and_official_scopes(tmp_path: Path) -> None:
    db_path = tmp_path / "lb.db"
    artifact_root = tmp_path / "reports"
    strat = _sample_strategy("local_only_v1")
    StratStore = StrategyStore(db_path)
    StratStore.save(strat)
    StratStore.save_evaluation(_sample_eval("local_only_v1", score=0.80))

    art_store = RunArtifactStore(artifact_root=artifact_root, db_path=db_path)
    art_store.save_run(
        kind="run",
        strategy=strat,
        evaluation=_sample_eval("local_only_v1", score=0.80),
        market_data=_sample_market_data(),
        requested_protocol="full_research_v1",
        data_reproducibility="refetch_dependent",
        code_version_override=CodeVersionInfo(commit="deadbeef", dirty=False),
    )

    service = LeaderboardService(db_path=db_path, artifact_root=artifact_root)
    local_rows = service.get_entries(scope="local", min_signal_count=30)
    official_rows = service.get_entries(scope="official", min_signal_count=30)
    assert len(local_rows) == 1
    assert local_rows[0].strategy_id == "local_only_v1"
    assert official_rows == []


def test_runs_and_scoped_leaderboard_cli_commands(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ALPHAEVO_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("COLUMNS", "200")
    artifact_root = tmp_path / "artifacts"

    store = RunArtifactStore(artifact_root=artifact_root, db_path=tmp_path / "cli.db")
    market = _sample_market_data()
    s1 = _sample_strategy("cli_strat_v1")
    s2 = _sample_strategy("cli_strat_v2")
    s2.exit.max_holding_days = 15

    m1 = store.save_run(
        kind="run",
        strategy=s1,
        evaluation=_sample_eval("cli_strat_v1", score=0.50),
        market_data=market,
        requested_protocol="fast_research_v1",
        code_version_override=CodeVersionInfo(commit="c1", dirty=False),
    )
    m2 = store.save_run(
        kind="run",
        strategy=s2,
        evaluation=_sample_eval("cli_strat_v2", score=0.72),
        market_data=market,
        requested_protocol="official_v1",
        data_reproducibility="replayable",
        data_mutability="replayable_snapshot",
        code_version_override=CodeVersionInfo(commit="c2", dirty=False),
    )

    res_list = runner.invoke(app, ["runs", "list", "--artifact-root", str(artifact_root)])
    assert res_list.exit_code == 0
    assert m1.run_id in res_list.stdout
    assert m2.run_id in res_list.stdout

    res_show = runner.invoke(
        app, ["runs", "show", m2.run_id, "--artifact-root", str(artifact_root)]
    )
    assert res_show.exit_code == 0
    assert "official_v1" in res_show.stdout

    res_cmp = runner.invoke(
        app,
        ["runs", "compare", m1.run_id, m2.run_id, "--artifact-root", str(artifact_root)],
    )
    assert res_cmp.exit_code == 0
    assert "False" in res_cmp.stdout

    res_lb_off = runner.invoke(
        app, ["leaderboard", "--scope", "official", "--artifact-root", str(artifact_root)]
    )
    assert res_lb_off.exit_code == 0
    assert "cli_strat_v2" in res_lb_off.stdout
    assert "cli_strat_v1" not in res_lb_off.stdout


def test_cli_main_lazy_import_does_not_eagerly_import_pandas() -> None:
    src_root = str(Path(__file__).resolve().parents[2] / "src")
    py_paths = [src_root, *(p for p in sys.path if p)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(py_paths)}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys, alphaevo.cli.main; assert "pandas" not in sys.modules',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
