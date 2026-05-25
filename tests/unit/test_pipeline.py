"""Unit tests for the RunPipeline orchestrator."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from alphaevo.core.config import AppConfig, BacktestConfig, DataConfig
from alphaevo.models import (
    BacktestResult,
    EvaluationReport,
    MarketContext,
    MarketType,
    OverallMetrics,
    SampleBatch,
    StockInfo,
    StopLossConfig,
    Strategy,
    StrategyCategory,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
    TradeSignal,
)
from alphaevo.models.enums import SamplingMethod, StrategyStatus
from alphaevo.orchestrator.pipeline import RunPipeline, RunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(sid: str = "test_strat_v1") -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id=sid,
            name="Test Strategy",
            version=1,
            market=MarketType.US,
            category=StrategyCategory.TREND,
            status=StrategyStatus.ACTIVE,
        ),
        description="A test strategy.",
        entry=StrategyEntry(
            conditions=[
                StrategyCondition(indicator="rsi_14", op="<", value=30),
            ],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
    )


def _make_ohlcv(rows: int = 60) -> pd.DataFrame:
    base = date.today() - timedelta(days=rows)
    return pd.DataFrame(
        {
            "date": [base + timedelta(days=i) for i in range(rows)],
            "open": [100.0 + i * 0.1 for i in range(rows)],
            "high": [101.0 + i * 0.1 for i in range(rows)],
            "low": [99.0 + i * 0.1 for i in range(rows)],
            "close": [100.5 + i * 0.1 for i in range(rows)],
            "volume": [1_000_000] * rows,
        }
    )


def _make_benchmark_df(rows: int = 80) -> pd.DataFrame:
    base = date.today() - timedelta(days=rows)
    return pd.DataFrame(
        {
            "date": [base + timedelta(days=i) for i in range(rows)],
            "open": [4000.0 + i * 2 for i in range(rows)],
            "high": [4010.0 + i * 2 for i in range(rows)],
            "low": [3990.0 + i * 2 for i in range(rows)],
            "close": [4005.0 + i * 2 for i in range(rows)],
            "volume": [100_000_000] * rows,
        }
    )


def _make_sample_batch(
    strategy_id: str = "test_strat_v1",
    symbols: list[str] | None = None,
) -> SampleBatch:
    symbols = symbols or ["AAPL", "MSFT"]
    end = date.today()
    start = end - timedelta(days=365)
    return SampleBatch(
        batch_id="batch-001",
        strategy_id=strategy_id,
        symbols=symbols,
        date_range=(start, end),
        sampling_method=SamplingMethod.REPRESENTATIVE,
        sampling_reason="test",
    )


def _make_backtest_result(
    strategy_id: str = "test_strat_v1",
    n_signals: int = 5,
) -> BacktestResult:
    end = date.today()
    start = end - timedelta(days=365)
    signals = [
        TradeSignal(
            symbol="AAPL",
            signal_date=start + timedelta(days=i * 30),
            direction="long",
            entry_price=100.0 + i,
            exit_price=102.0 + i,
            exit_date=start + timedelta(days=i * 30 + 5),
            exit_reason="take_profit",
            return_pct=0.02,
            holding_days=5,
        )
        for i in range(n_signals)
    ]
    return BacktestResult(
        strategy_id=strategy_id,
        batch_id="batch-001",
        signals=signals,
        total_signals=n_signals,
        executed_signals=n_signals,
        date_range=(start, end),
    )


def _make_eval_report(
    strategy_id: str = "test_strat_v1",
    score: float = 0.55,
) -> EvaluationReport:
    return EvaluationReport(
        evaluation_id="eval-001",
        strategy_id=strategy_id,
        batch_id="batch-001",
        overall=OverallMetrics(
            win_rate=0.6,
            avg_return=0.015,
            signal_count=40,
        ),
        confidence_score=score,
    )


def _make_config(data: DataConfig | None = None) -> AppConfig:
    """Minimal AppConfig for pipeline tests."""
    return AppConfig(
        data=data or DataConfig(adapter="yfinance"),
        backtest=BacktestConfig(slippage=0.001, commission=0.0003, min_data_days=30),
    )


def _stock_list() -> list[StockInfo]:
    return [
        StockInfo(symbol="AAPL", name="Apple", market=MarketType.US),
        StockInfo(symbol="MSFT", name="Microsoft", market=MarketType.US),
        StockInfo(symbol="GOOG", name="Google", market=MarketType.US),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunResult:
    """Verify the RunResult dataclass."""

    def test_fields(self):
        strat = _make_strategy()
        batch = _make_sample_batch()
        bt = _make_backtest_result()
        ev = _make_eval_report()

        rr = RunResult(
            strategy=strat,
            batch=batch,
            backtest_result=bt,
            evaluation=ev,
        )
        assert rr.report_path is None
        assert rr.strategy is strat
        assert rr.evaluation.confidence_score == 0.55

    def test_with_report_path(self):
        rr = RunResult(
            strategy=_make_strategy(),
            batch=_make_sample_batch(),
            backtest_result=_make_backtest_result(),
            evaluation=_make_eval_report(),
            report_path=Path("reports/test.md"),
        )
        assert rr.report_path == Path("reports/test.md")


class TestRunPipeline:
    """Integration-style tests with full mocking of external dependencies."""

    def _make_pipeline(self) -> RunPipeline:
        """Build a RunPipeline with a mock-friendly config."""
        return RunPipeline(_make_config())

    def test_indicator_lookback_handles_templated_windows(self):
        assert RunPipeline._indicator_lookback("atr_21") == 22
        assert RunPipeline._indicator_lookback("macd_histogram_fast10_slow30_signal7") == 36
        assert RunPipeline._indicator_lookback("macd_cross_bullish_fast10_slow30_signal7") == 37
        assert RunPipeline._indicator_lookback("rsi_21") == 22
        assert RunPipeline._indicator_lookback("rsi_21_zscore") == 50
        assert RunPipeline._indicator_lookback("bollinger_band_width_30d") == 30
        assert RunPipeline._indicator_lookback("bollinger_band_width_30d_std1p5") == 30
        assert RunPipeline._indicator_lookback("price_above_bollinger_upper_30d") == 30
        assert RunPipeline._indicator_lookback("price_above_bollinger_upper_30d_std2p5") == 30
        assert RunPipeline._indicator_lookback("breakout_high_20d") == 21
        assert RunPipeline._indicator_lookback("price_position_120d") == 120
        assert RunPipeline._indicator_lookback("volume_ratio_1d_10d") == 11
        assert RunPipeline._indicator_lookback("volatility_30d") == 31
        assert RunPipeline._indicator_lookback("relative_strength_30d") == 31

    def test_estimate_indicator_warmup_uses_atr_stop_loss_period(self):
        config = _make_config()
        config.backtest.min_data_days = 5
        pipeline = RunPipeline(config)
        strategy = _make_strategy()
        strategy.entry.conditions = []
        strategy.exit.stop_loss = StopLossConfig(type="atr", multiplier=2.0, atr_period=21)

        assert pipeline._estimate_indicator_warmup_days(strategy) == 22

    # -- test full happy path ------------------------------------------

    async def test_run_happy_path(self):
        pipeline = self._make_pipeline()
        strategy = _make_strategy()
        batch = _make_sample_batch()
        bt_result = _make_backtest_result()
        eval_report = _make_eval_report()

        # Mock store
        mock_store = MagicMock()
        mock_store.get.return_value = strategy
        mock_store.save_evaluation = MagicMock()
        pipeline._store = mock_store

        # Mock data_manager
        mock_dm = MagicMock()
        mock_dm.get_stock_list = AsyncMock(return_value=_stock_list())
        mock_dm.get_history = AsyncMock(return_value=_make_ohlcv())
        pipeline._data_manager = mock_dm

        with (
            patch("alphaevo.orchestrator.pipeline.AdaptiveSampler") as MockSampler,
            patch("alphaevo.orchestrator.pipeline.BacktestEngine") as MockEngine,
            patch("alphaevo.orchestrator.pipeline.Evaluator") as MockEvaluator,
        ):
            MockSampler.return_value.sample = AsyncMock(return_value=batch)
            MockEngine.return_value.run.return_value = bt_result
            MockEvaluator.return_value.evaluate.return_value = eval_report

            result = await pipeline.run("test_strat_v1")

        # Assertions
        assert isinstance(result, RunResult)
        assert result.strategy is strategy
        assert result.batch is batch
        assert result.backtest_result is bt_result
        assert result.evaluation is eval_report
        assert result.report_path is None

        mock_store.get.assert_called_once_with("test_strat_v1")
        mock_store.save_evaluation.assert_called_once_with(eval_report)
        mock_dm.get_stock_list.assert_awaited_once()

    # -- test progress callback ----------------------------------------

    async def test_run_calls_on_progress(self):
        pipeline = self._make_pipeline()
        strategy = _make_strategy()
        batch = _make_sample_batch()

        mock_store = MagicMock()
        mock_store.get.return_value = strategy
        pipeline._store = mock_store

        mock_dm = MagicMock()
        mock_dm.get_stock_list = AsyncMock(return_value=_stock_list())
        mock_dm.get_history = AsyncMock(return_value=_make_ohlcv())
        pipeline._data_manager = mock_dm

        progress_msgs: list[str] = []

        with (
            patch("alphaevo.orchestrator.pipeline.AdaptiveSampler") as MockSampler,
            patch("alphaevo.orchestrator.pipeline.BacktestEngine") as MockEngine,
            patch("alphaevo.orchestrator.pipeline.Evaluator") as MockEvaluator,
        ):
            MockSampler.return_value.sample = AsyncMock(return_value=batch)
            MockEngine.return_value.run.return_value = _make_backtest_result()
            MockEvaluator.return_value.evaluate.return_value = _make_eval_report()

            await pipeline.run("test_strat_v1", on_progress=progress_msgs.append)

        assert len(progress_msgs) >= 6
        assert any("Loading" in m for m in progress_msgs)
        assert any("backtest" in m.lower() for m in progress_msgs)
        assert any("Confidence" in m for m in progress_msgs)

    async def test_run_passes_contexts_to_backtest_and_evaluator(self):
        pipeline = self._make_pipeline()
        strategy = _make_strategy()
        batch = _make_sample_batch()
        bt_result = _make_backtest_result()
        eval_report = _make_eval_report()

        mock_store = MagicMock()
        mock_store.get.return_value = strategy
        mock_store.save_evaluation = MagicMock()
        pipeline._store = mock_store

        mock_dm = MagicMock()
        mock_dm.get_stock_list = AsyncMock(return_value=_stock_list())
        mock_dm.get_history = AsyncMock(return_value=_make_ohlcv())
        mock_dm.get_index_data = AsyncMock(return_value=_make_benchmark_df())
        mock_dm.get_market_context = AsyncMock(
            return_value=MarketContext(
                breadth=0.7,
                sentiment_index=0.63,
                sector_leaders=["Technology"],
                sector_laggards=["Energy"],
            )
        )
        mock_dm.get_sector_data = AsyncMock(return_value=None)
        mock_dm.get_event_context = AsyncMock(return_value=None)
        pipeline._data_manager = mock_dm

        with (
            patch("alphaevo.orchestrator.pipeline.AdaptiveSampler") as MockSampler,
            patch("alphaevo.orchestrator.pipeline.BacktestEngine") as MockEngine,
            patch("alphaevo.orchestrator.pipeline.Evaluator") as MockEvaluator,
        ):
            MockSampler.return_value.sample = AsyncMock(return_value=batch)
            MockEngine.return_value.run.return_value = bt_result
            MockEvaluator.return_value.evaluate.return_value = eval_report

            await pipeline.run("test_strat_v1")

        assert MockEngine.call_args.kwargs["fill_policy"] == "conservative"
        engine_kwargs = MockEngine.return_value.run.call_args.kwargs
        assert "contexts" in engine_kwargs
        assert set(engine_kwargs["contexts"].keys()) == {"AAPL", "MSFT"}
        context = engine_kwargs["contexts"]["AAPL"].market_context
        assert context is not None
        assert context.breadth == 0.7
        assert context.sentiment_index == 0.63
        assert context.sector_leaders == ["Technology"]
        evaluator_kwargs = MockEvaluator.return_value.evaluate.call_args.kwargs
        assert evaluator_kwargs["market_data"].keys() == {"AAPL", "MSFT"}
        assert evaluator_kwargs["contexts"].keys() == {"AAPL", "MSFT"}
        assert evaluator_kwargs["backtest_config"] == pipeline.config.backtest

    # -- test strategy not found ---------------------------------------

    async def test_run_strategy_not_found(self):
        pipeline = self._make_pipeline()
        mock_store = MagicMock()
        mock_store.get.return_value = None
        pipeline._store = mock_store

        with pytest.raises(ValueError, match="Strategy not found"):
            await pipeline.run("nonexistent_strategy")

    # -- test no data raises -------------------------------------------

    async def test_run_no_data_raises(self):
        pipeline = self._make_pipeline()
        strategy = _make_strategy()

        mock_store = MagicMock()
        mock_store.get.return_value = strategy
        pipeline._store = mock_store

        mock_dm = MagicMock()
        mock_dm.get_stock_list = AsyncMock(return_value=_stock_list())
        # All fetches return empty DataFrames → no data
        mock_dm.get_history = AsyncMock(return_value=pd.DataFrame())
        pipeline._data_manager = mock_dm

        with (
            patch("alphaevo.orchestrator.pipeline.AdaptiveSampler") as MockSampler,
        ):
            MockSampler.return_value.sample = AsyncMock(return_value=_make_sample_batch())
            with pytest.raises(RuntimeError, match="No valid market data"):
                await pipeline.run("test_strat_v1")

    # -- test report generation ----------------------------------------

    async def test_run_generates_report(self, tmp_path: Path):
        pipeline = self._make_pipeline()
        strategy = _make_strategy()
        batch = _make_sample_batch()

        mock_store = MagicMock()
        mock_store.get.return_value = strategy
        pipeline._store = mock_store

        mock_dm = MagicMock()
        mock_dm.get_stock_list = AsyncMock(return_value=_stock_list())
        mock_dm.get_history = AsyncMock(return_value=_make_ohlcv())
        pipeline._data_manager = mock_dm

        with (
            patch("alphaevo.orchestrator.pipeline.AdaptiveSampler") as MockSampler,
            patch("alphaevo.orchestrator.pipeline.BacktestEngine") as MockEngine,
            patch("alphaevo.orchestrator.pipeline.Evaluator") as MockEvaluator,
            patch("alphaevo.orchestrator.pipeline.Reporter") as MockReporter,
        ):
            MockSampler.return_value.sample = AsyncMock(return_value=batch)
            MockEngine.return_value.run.return_value = _make_backtest_result()
            eval_report = _make_eval_report()
            MockEvaluator.return_value.evaluate.return_value = eval_report

            result = await pipeline.run("test_strat_v1", report_dir=tmp_path / "reports")

        assert result.report_path is not None
        assert result.report_path.name == "test_strat_v1_report.md"
        MockReporter.to_file.assert_called_once()

    # -- test custom date range ----------------------------------------

    async def test_run_custom_date_range(self):
        pipeline = self._make_pipeline()
        strategy = _make_strategy()
        batch = _make_sample_batch()

        mock_store = MagicMock()
        mock_store.get.return_value = strategy
        pipeline._store = mock_store

        mock_dm = MagicMock()
        mock_dm.get_stock_list = AsyncMock(return_value=_stock_list())
        mock_dm.get_history = AsyncMock(return_value=_make_ohlcv())
        pipeline._data_manager = mock_dm

        custom_range = (date(2023, 1, 1), date(2023, 12, 31))

        with (
            patch("alphaevo.orchestrator.pipeline.AdaptiveSampler") as MockSampler,
            patch("alphaevo.orchestrator.pipeline.BacktestEngine") as MockEngine,
            patch("alphaevo.orchestrator.pipeline.Evaluator") as MockEvaluator,
        ):
            MockSampler.return_value.sample = AsyncMock(return_value=batch)
            MockEngine.return_value.run.return_value = _make_backtest_result()
            MockEvaluator.return_value.evaluate.return_value = _make_eval_report()

            await pipeline.run("test_strat_v1", date_range=custom_range)

        # Sampler should receive our custom date_range
        MockSampler.return_value.sample.assert_awaited_once()
        call_kwargs = MockSampler.return_value.sample.call_args
        assert call_kwargs.kwargs.get("date_range") == custom_range
        fetch_call = mock_dm.get_history.await_args_list[0]
        assert fetch_call.args[1] == custom_range[0] - timedelta(days=60)
        assert fetch_call.args[2] == custom_range[1]

    async def test_run_auto_expands_sampling_until_signal_target_reached(self):
        config = _make_config()
        config.evolution.min_signal_count = 30
        config.evolution.max_sample_expansions = 2
        config.evolution.sample_expansion_window_days = 90
        config.evolution.sample_expansion_symbol_step = 15
        pipeline = RunPipeline(config)

        strategy = _make_strategy()
        mock_store = MagicMock()
        mock_store.get.return_value = strategy
        mock_store.save_evaluation = MagicMock()
        pipeline._store = mock_store

        mock_dm = MagicMock()
        mock_dm.get_stock_list = AsyncMock(return_value=_stock_list())
        mock_dm.get_history = AsyncMock(return_value=_make_ohlcv())
        pipeline._data_manager = mock_dm

        batch1 = _make_sample_batch(symbols=["AAPL", "MSFT"])
        batch2 = _make_sample_batch(symbols=["AAPL", "MSFT", "GOOG"])
        eval1 = _make_eval_report(score=0.22)
        eval1.overall.signal_count = 12
        eval2 = _make_eval_report(score=0.46)
        eval2.overall.signal_count = 34

        with (
            patch("alphaevo.orchestrator.pipeline.AdaptiveSampler") as MockSampler,
            patch("alphaevo.orchestrator.pipeline.BacktestEngine") as MockEngine,
            patch("alphaevo.orchestrator.pipeline.Evaluator") as MockEvaluator,
        ):
            MockSampler.return_value.sample = AsyncMock(side_effect=[batch1, batch2])
            MockEngine.return_value.run.side_effect = [
                _make_backtest_result(n_signals=12),
                _make_backtest_result(n_signals=34),
            ]
            MockEvaluator.return_value.evaluate.side_effect = [eval1, eval2]

            result = await pipeline.run("test_strat_v1")

        assert result.evaluation is eval2
        assert result.batch.sampling_attempt == 2
        assert result.batch.signal_count_target == 30
        assert result.batch.signal_count_reached == 34
        assert not result.batch.insufficient_signals
        assert len(result.batch.sampling_history) == 2
        assert result.batch.sampling_history[0].signal_count == 12
        assert result.batch.sampling_history[1].accepted is True
        assert MockSampler.return_value.sample.await_count == 2
        second_call = MockSampler.return_value.sample.await_args_list[1]
        assert second_call.kwargs["date_range"][0] < second_call.kwargs["date_range"][1]

    async def test_run_marks_batch_when_sampling_expansion_exhausted(self):
        config = _make_config()
        config.evolution.min_signal_count = 30
        config.evolution.max_sample_expansions = 1
        config.evolution.sample_expansion_window_days = 60
        pipeline = RunPipeline(config)

        strategy = _make_strategy()
        mock_store = MagicMock()
        mock_store.get.return_value = strategy
        mock_store.save_evaluation = MagicMock()
        pipeline._store = mock_store

        mock_dm = MagicMock()
        mock_dm.get_stock_list = AsyncMock(return_value=_stock_list())
        mock_dm.get_history = AsyncMock(return_value=_make_ohlcv())
        pipeline._data_manager = mock_dm

        eval1 = _make_eval_report(score=0.21)
        eval1.overall.signal_count = 8
        eval2 = _make_eval_report(score=0.24)
        eval2.overall.signal_count = 14

        with (
            patch("alphaevo.orchestrator.pipeline.AdaptiveSampler") as MockSampler,
            patch("alphaevo.orchestrator.pipeline.BacktestEngine") as MockEngine,
            patch("alphaevo.orchestrator.pipeline.Evaluator") as MockEvaluator,
        ):
            MockSampler.return_value.sample = AsyncMock(
                side_effect=[
                    _make_sample_batch(symbols=["AAPL"]),
                    _make_sample_batch(symbols=["AAPL", "MSFT"]),
                ]
            )
            MockEngine.return_value.run.side_effect = [
                _make_backtest_result(n_signals=8),
                _make_backtest_result(n_signals=14),
            ]
            MockEvaluator.return_value.evaluate.side_effect = [eval1, eval2]

            result = await pipeline.run("test_strat_v1")

        assert result.batch.sampling_attempt == 2
        assert result.batch.insufficient_signals is True
        assert result.batch.signal_count_reached == 14
        assert len(result.batch.sampling_history) == 2
        assert all(not attempt.accepted for attempt in result.batch.sampling_history)


class TestFetchData:
    """Tests for the _fetch_data helper."""

    async def test_concurrent_fetch(self):
        pipeline = RunPipeline(_make_config())

        mock_dm = MagicMock()
        mock_dm.get_history = AsyncMock(return_value=_make_ohlcv())
        pipeline._data_manager = mock_dm

        end = date.today()
        start = end - timedelta(days=365)
        data = await pipeline._fetch_data(_make_strategy(), ["AAPL", "MSFT", "GOOG"], (start, end))

        assert len(data) == 3
        assert all(sym in data for sym in ["AAPL", "MSFT", "GOOG"])
        assert mock_dm.get_history.await_count == 3

    async def test_fetch_skips_failures(self):
        pipeline = RunPipeline(_make_config())

        call_count = 0

        async def _side_effect(symbol, start, end):
            nonlocal call_count
            call_count += 1
            if symbol == "BAD":
                raise RuntimeError("network error")
            return _make_ohlcv()

        mock_dm = MagicMock()
        mock_dm.get_history = AsyncMock(side_effect=_side_effect)
        pipeline._data_manager = mock_dm

        end = date.today()
        start = end - timedelta(days=90)
        data = await pipeline._fetch_data(_make_strategy(), ["AAPL", "BAD", "MSFT"], (start, end))

        assert len(data) == 2
        assert "BAD" not in data

    async def test_fetch_skips_empty_dataframes(self):
        pipeline = RunPipeline(_make_config())

        async def _side_effect(symbol, start, end):
            if symbol == "EMPTY":
                return pd.DataFrame()
            return _make_ohlcv()

        mock_dm = MagicMock()
        mock_dm.get_history = AsyncMock(side_effect=_side_effect)
        pipeline._data_manager = mock_dm

        end = date.today()
        start = end - timedelta(days=90)
        data = await pipeline._fetch_data(_make_strategy(), ["AAPL", "EMPTY"], (start, end))

        assert len(data) == 1
        assert "EMPTY" not in data


class TestCreateAdapter:
    """Tests for the adapter factory."""

    def test_unknown_adapter_raises(self):
        with pytest.raises(ValueError, match="Unknown adapter"):
            RunPipeline._create_adapter("unknown_source")

    def test_akshare_adapter(self):
        adapter = RunPipeline._create_adapter("akshare")
        assert adapter.name == "akshare"

    def test_dsa_adapter_bridge_passes_path(self):
        with patch("alphaevo.data.adapters.dsa.DSAAdapter") as MockDSA:
            adapter = MagicMock()
            adapter.name = "dsa"
            MockDSA.return_value = adapter

            result = RunPipeline._create_adapter(
                "dsa",
                dsa_path="/tmp/daily_stock_analysis",
            )

        MockDSA.assert_called_once_with(dsa_path="/tmp/daily_stock_analysis")
        assert result.name == "dsa"

    def test_yfinance_adapter(self):
        adapter = RunPipeline._create_adapter("yfinance")
        assert adapter.name == "yfinance"

    def test_adanos_context_adapter_requires_key_and_yfinance(self):
        no_key = RunPipeline(_make_config(data=DataConfig(adapter="yfinance")))
        assert no_key._create_adanos_context_adapter() is None

        non_yfinance = RunPipeline(
            _make_config(
                data=DataConfig(
                    adapter="akshare",
                    adanos_api_key="test-key",
                )
            )
        )
        assert non_yfinance._create_adanos_context_adapter() is None

    def test_adanos_context_adapter_is_secondary_for_yfinance(self):
        pipeline = RunPipeline(
            _make_config(
                data=DataConfig(
                    adapter="yfinance",
                    adanos_api_key="test-key",
                    adanos_base_url="https://api.example.test",
                )
            )
        )

        manager = pipeline.data_manager

        assert [adapter.name for adapter in manager._adapters] == ["adanos", "yfinance"]


class TestLazyProperties:
    """Verify store and data_manager are lazily initialised."""

    def test_store_lazy_init(self):
        pipeline = RunPipeline(_make_config())
        assert pipeline._store is None
        store = pipeline.store
        assert store is not None
        # Second access returns the same instance
        assert pipeline.store is store

    def test_data_manager_lazy_init(self):
        pipeline = RunPipeline(_make_config())
        assert pipeline._data_manager is None
        dm = pipeline.data_manager
        assert dm is not None
        assert pipeline.data_manager is dm


class TestRunSync:
    """Tests for the synchronous wrapper."""

    def test_run_sync_delegates(self):
        pipeline = RunPipeline(_make_config())
        strategy = _make_strategy()
        expected = RunResult(
            strategy=strategy,
            batch=_make_sample_batch(),
            backtest_result=_make_backtest_result(),
            evaluation=_make_eval_report(),
        )

        with patch.object(pipeline, "run", new=AsyncMock(return_value=expected)):
            result = pipeline.run_sync("test_strat_v1", max_symbols=10)

        assert result is expected


class TestEnsureBuiltinStrategies:
    """Tests for ensure_builtin_strategies."""

    def test_imports_when_dir_exists(self):
        pipeline = RunPipeline(_make_config())
        mock_store = MagicMock()
        mock_store.import_builtin_strategies.return_value = 4
        pipeline._store = mock_store

        with patch("alphaevo.orchestrator.pipeline.Path") as MockPath:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            MockPath.return_value.resolve.return_value.parent.parent.parent.parent.__truediv__ = (
                MagicMock(return_value=mock_dir)
            )
            # Patch Path(__file__) resolution
            MockPath.return_value = mock_dir

            count = pipeline.ensure_builtin_strategies()

        mock_store.import_builtin_strategies.assert_called_once()
        assert count == 4

    def test_returns_zero_when_no_dir(self):
        pipeline = RunPipeline(_make_config())
        mock_store = MagicMock()
        pipeline._store = mock_store

        # Patch Path.exists to always return False so neither builtin dir is found
        with patch.object(Path, "exists", return_value=False):
            count = pipeline.ensure_builtin_strategies()

        assert count == 0
        mock_store.import_builtin_strategies.assert_not_called()


class TestModuleExports:
    """Verify the orchestrator package re-exports correctly."""

    def test_imports_from_package(self):
        from alphaevo.orchestrator import RunPipeline as RP
        from alphaevo.orchestrator import RunResult as RR

        assert RP is RunPipeline
        assert RR is RunResult


class TestAutoloadCustomFactors:
    """Verify pipeline auto-loads active factors from FactorStore."""

    def test_loads_active_factors(self):
        from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
        from alphaevo.backtest.indicators import IndicatorRegistry

        factor_name = "_test_autoload_factor_xyz"
        factor_code = (
            "import pandas as pd\n"
            "def compute(df: pd.DataFrame, idx: int, ctx=None) -> float:\n"
            "    return 42.0\n"
        )
        # Clean up in case left from a prior run
        if IndicatorRegistry.is_registered(factor_name):
            IndicatorRegistry._dynamic_registry.pop(factor_name, None)

        store = FactorStore(":memory:")
        store.save(
            FactorRecord(
                name=factor_name,
                description="test factor",
                rationale="testing",
                code=factor_code,
                status="active",
            )
        )

        pipeline = RunPipeline(_make_config())
        progress_msgs: list[str] = []

        with patch(
            "alphaevo.alpha_factory.factor_store.FactorStore",
            return_value=store,
        ):
            pipeline._autoload_custom_factors(progress_msgs.append)

        assert IndicatorRegistry.is_registered(factor_name)
        assert any("custom factor" in m for m in progress_msgs)

        # Verify the registered factor is actually computable, not just a string
        import pandas as pd

        test_df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000.0],
            }
        )
        value = IndicatorRegistry.compute(factor_name, test_df, 0)
        assert isinstance(value, (int, float)), (
            f"Expected numeric result but got {type(value).__name__}: {value!r}"
        )
        assert value == 42.0

        # Cleanup
        IndicatorRegistry._dynamic_registry.pop(factor_name, None)
        IndicatorRegistry._registry.pop(factor_name, None)
