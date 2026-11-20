"""Tests for demo helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from alphaevo.cli.demo import (
    _build_synthetic_data,
    _data_fingerprint,
    _DummyLLM,
    _fetch_real_data,
    _persist_demo_history,
    _record_demo_experience,
    _run_backtest,
    _select_best_demo_mutation,
    _showcase_change_plan,
    load_showcase_snapshot,
)
from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import StrategyChange
from alphaevo.reflection.analyzer import ReflectionAnalyzer
from alphaevo.reflection.critic import SelfCritic
from alphaevo.reflection.experience import ExperienceQuery, ExperienceStore
from alphaevo.reflection.mutator import StrategyMutator
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.store import StrategyStore


def _make_ohlcv(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="B"),
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1_000_000] * rows,
            "prev_close": [None] + [100.5 + i for i in range(rows - 1)],
        }
    )


class TestFetchRealData:
    @pytest.mark.asyncio
    async def test_uses_history_fetch_with_cache_for_yfinance(self, tmp_path):
        config = SimpleNamespace(data=SimpleNamespace(cache_dir=tmp_path / "cache"))
        with (
            patch("alphaevo.data.adapters.yfinance.YFinanceAdapter") as MockAdapter,
            patch("alphaevo.cli.demo.DataManager") as MockDataManager,
            patch("alphaevo.cli.demo.ConfigManager.load", return_value=config),
        ):
            manager = MockDataManager.return_value
            manager.get_history = AsyncMock(return_value=_make_ohlcv())

            data = await _fetch_real_data(["AAPL", "MSFT"], "yfinance")

        assert set(data) == {"AAPL", "MSFT"}
        MockAdapter.assert_called_once_with()
        cache = MockDataManager.call_args.kwargs["cache"]
        assert cache.cache_dir == config.data.cache_dir
        assert manager.get_history.await_count == 2
        for call in manager.get_history.await_args_list:
            assert call.args[1] < call.args[2]
            assert (call.args[2] - call.args[1]).days == 180

    @pytest.mark.asyncio
    async def test_selects_akshare_adapter(self, tmp_path):
        config = SimpleNamespace(data=SimpleNamespace(cache_dir=tmp_path / "cache"))
        with (
            patch("alphaevo.data.adapters.akshare.AkShareAdapter") as MockAdapter,
            patch("alphaevo.cli.demo.DataManager") as MockDataManager,
            patch("alphaevo.cli.demo.ConfigManager.load", return_value=config),
        ):
            manager = MockDataManager.return_value
            manager.get_history = AsyncMock(return_value=_make_ohlcv())

            data = await _fetch_real_data(["000001"], "akshare")

        MockAdapter.assert_called_once_with()
        assert list(data) == ["000001"]

    @pytest.mark.asyncio
    async def test_skips_short_or_failing_symbols(self, tmp_path):
        short_df = _make_ohlcv(rows=10)
        config = SimpleNamespace(data=SimpleNamespace(cache_dir=tmp_path / "cache"))
        with (
            patch("alphaevo.data.adapters.yfinance.YFinanceAdapter"),
            patch("alphaevo.cli.demo.DataManager") as MockDataManager,
            patch("alphaevo.cli.demo.ConfigManager.load", return_value=config),
        ):
            manager = MockDataManager.return_value
            manager.get_history = AsyncMock(
                side_effect=[
                    _make_ohlcv(),
                    short_df,
                    RuntimeError("network error"),
                ]
            )

            data = await _fetch_real_data(["AAPL", "SHORT", "FAIL"], "yfinance")

        assert list(data) == ["AAPL"]


def test_select_best_demo_mutation_finds_a_real_improvement():
    strategy = StrategyParser().parse_file(Path("strategies/builtin/ma_crossover.yaml"))
    data = _build_synthetic_data()
    evaluation, _signals, _trades = _run_backtest(strategy, data)

    candidate = _select_best_demo_mutation(
        strategy,
        evaluation,
        data,
        analyzer=ReflectionAnalyzer(llm=_DummyLLM(), max_changes=3),  # type: ignore[arg-type]
        critic=SelfCritic(experience_store=ExperienceStore(db_path=":memory:"), complexity_limit=8),
        mutator=StrategyMutator(max_changes=3, complexity_limit=8),
    )

    assert candidate is not None
    assert candidate.evaluation.confidence_score > evaluation.confidence_score
    assert candidate.strategy.meta.id == "ma_crossover_v2"


def test_record_demo_experience_uses_validated_score_delta():
    store = ExperienceStore(db_path=":memory:")
    _record_demo_experience(
        store,
        strategy_family="demo",
        strategy_id="ma_crossover_v1",
        round_num=1,
        changes=[
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=volume_ratio_1d_5d].value",
                from_value=1.2,
                to_value=1.1,
                reason="Validated volume ratio threshold from 1.2 to 1.1 on the demo batch",
            )
        ],
        score_before=0.3432,
        score_after=0.3756,
    )

    records = store.query(ExperienceQuery(strategy_family="demo", limit=5))

    assert len(records) == 1
    assert records[0].worked is True
    assert records[0].score_before == pytest.approx(0.3432)
    assert records[0].score_after == pytest.approx(0.3756)
    assert records[0].score_delta == pytest.approx(0.0324)


def test_persist_demo_history_saves_strategies_and_evaluations(tmp_path):
    strategy = StrategyParser().parse_file(Path("strategies/builtin/ma_crossover.yaml"))
    data = _build_synthetic_data()
    evaluation, signals, trades = _run_backtest(strategy, data)
    config = SimpleNamespace(db_path=tmp_path / "demo.db")

    with patch("alphaevo.cli.demo.ConfigManager.load", return_value=config):
        saved = _persist_demo_history([(strategy, evaluation, signals, trades)])

    store = StrategyStore(config.db_path)
    assert saved == 1
    assert store.get(strategy.meta.id) is not None
    assert len(store.get_evaluations(strategy.meta.id)) == 1


def test_load_showcase_snapshot_uses_bundled_real_data() -> None:
    data, manifest = load_showcase_snapshot()

    assert {"AAPL", "MSFT", "NVDA", "AMD", "TSLA"}.issubset(data)
    assert manifest["source_adapter"] == "yfinance"
    assert _data_fingerprint(data)
    assert all(len(df) >= 250 for df in data.values())


def test_showcase_plan_produces_stronger_snapshot_champion() -> None:
    strategy = StrategyParser().parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    data, _manifest = load_showcase_snapshot()
    data = {symbol: data[symbol] for symbol in ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]}
    plan = _showcase_change_plan(strategy)
    mutator = StrategyMutator(max_changes=3, complexity_limit=8)

    current = strategy
    for change in plan:
        current = mutator.mutate(current, [change], atomic=True)

    evaluation, signals, _trades = _run_backtest(current, data)
    metrics = evaluation.overall

    assert [change.target for change in plan] == [
        "entry.logic",
        "exit.stop_loss.value",
        "exit.max_holding_days",
        "exit.take_profit.value",
        "entry.conditions[indicator=close_to_ma20_pct].indicator",
        "entry.conditions[indicator=volume_ratio_1d_5d].value",
    ]
    assert signals >= 30
    assert metrics.avg_return >= 0.025
    # Daily MTM + gap-aware open stop fills yield honest basket drawdown (<= 0.30)
    assert metrics.max_drawdown <= 0.30
    assert evaluation.confidence_score >= 0.65
