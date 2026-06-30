"""RunPipeline — end-to-end orchestration for the strategy research loop.

Flow: strategy → sample → fetch data → backtest → evaluate → report.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.backtest.indicators import merge_event_context
from alphaevo.core.config import AppConfig, DataConfig
from alphaevo.data.adapter import DataAdapter, DataManager, DataSourceHealth, get_adapter_chain
from alphaevo.data.cache import DataCache
from alphaevo.data.quality import DataQualityReport, build_data_quality_report
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.evaluator.reporter import Reporter
from alphaevo.models import (
    BacktestResult,
    EvaluationReport,
    EventContextSeries,
    IndicatorContext,
    MarketContext,
    SampleBatch,
    SamplingAttempt,
    SectorInfo,
    StockInfo,
    Strategy,
)
from alphaevo.models.enums import MarketType, SamplingMethod
from alphaevo.sampler.adaptive import AdaptiveSampler
from alphaevo.sampler.regime import RegimeDetector
from alphaevo.strategy.store import StrategyStore

logger = logging.getLogger(__name__)

_INDICATOR_LOOKBACKS: dict[str, int] = {
    "ma5_above_ma10": 10,
    "close_to_ma10_pct": 10,
    "close_above_ma20": 20,
    "close_below_ma10": 10,
    "volume_ratio_1d_5d": 6,
    "rsi_14": 15,
    "deviation_from_ma20_pct": 20,
    "has_stop_signal": 2,
    "volume_shrink_then_rise": 4,
    "ma5_ge_ma10_or_crossing": 11,
    "atr": 15,
    "macd_histogram": 34,
    "macd_cross_bullish": 35,
    "bollinger_band_width": 20,
    "price_above_bollinger_upper": 20,
    "price_below_bollinger_lower": 20,
    "ma20_slope": 25,
    "momentum_10d": 11,
    "avg_volume_20d": 20,
    "consecutive_up_days": 2,
    "consecutive_down_days": 2,
    "days_since_high_20d": 20,
    "days_since_low_20d": 20,
    "rsi_14_zscore": 50,
    "volume_ratio_1d_20d": 21,
    "breakout_high_20d": 21,
    "price_position_120d": 120,
    "price_position_52w": 250,
    "volatility_20d": 21,
    "gap_up_pct": 2,
    "body_to_range_ratio": 1,
    "relative_strength_20d": 21,
    "st_flag": 1,
    "sector_heat_rank": 1,
    "sector_heat_rising_days": 1,
    "intra_sector_strength_rank_pct": 1,
    "negative_news_score": 1,
    "news_sentiment_score": 1,
    "days_since_event": 1,
    "price_above_pre_event": 1,
    "sector_fund_flow_positive": 1,
    "already_overreacted": 1,
    "sector_risk_flag": 1,
    "sector_net_inflow_days": 1,
}


@dataclass
class RunResult:
    """Result of a pipeline run."""

    strategy: Strategy
    batch: SampleBatch
    backtest_result: BacktestResult
    evaluation: EvaluationReport
    report_path: Path | None = field(default=None)
    data_source_health: list[DataSourceHealth] = field(default_factory=list)
    data_quality: DataQualityReport = field(default_factory=DataQualityReport)
    # Retained for post-hoc analysis (param sensitivity, etc.)
    _engine: object | None = field(default=None, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _contexts: dict[str, IndicatorContext] | None = field(default=None, repr=False)


class RunPipeline:
    """Orchestrates the strategy research loop:

    strategy → sample → fetch data → backtest → evaluate → report.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._store: StrategyStore | None = None
        self._data_manager: DataManager | None = None

    # -- lazy singletons ------------------------------------------------

    @property
    def store(self) -> StrategyStore:
        if self._store is None:
            self._store = StrategyStore(self.config.db_path)
        return self._store

    @property
    def data_manager(self) -> DataManager:
        if self._data_manager is None:
            adapters = get_adapter_chain(self.config)
            adanos = self._create_adanos_context_adapter()
            if adanos is not None:
                adapters.insert(0, adanos)
            self._data_manager = DataManager(
                adapters,
                cache=DataCache(self.config.data.cache_dir),
                failure_threshold=self.config.data.source_failure_threshold,
                cooldown_seconds=self.config.data.source_cooldown_seconds,
            )
        return self._data_manager

    # -- adapter factory ------------------------------------------------

    @staticmethod
    def _create_adapters(
        adapter_name: str,
        *,
        dsa_path: str | None = None,
    ) -> list[DataAdapter]:
        """Factory for ordered data adapter chains."""
        config = AppConfig(data=DataConfig(adapter=adapter_name, dsa_path=dsa_path))
        return get_adapter_chain(config)

    @staticmethod
    def _create_adapter(
        adapter_name: str,
        *,
        dsa_path: str | None = None,
    ) -> DataAdapter:
        """Backward-compatible factory returning the first adapter in a chain."""
        return RunPipeline._create_adapters(adapter_name, dsa_path=dsa_path)[0]

    def _create_adanos_context_adapter(self) -> DataAdapter | None:
        """Create the optional Adanos context adapter for yfinance US workflows."""
        api_key = self.config.data.adanos_api_key
        if not api_key or self.config.data.adapter != "yfinance":
            return None

        from alphaevo.data.adapters.adanos import AdanosSentimentAdapter

        return AdanosSentimentAdapter(
            api_key,
            base_url=self.config.data.adanos_base_url,
        )

    # -- main entry point -----------------------------------------------

    async def run(
        self,
        strategy_id: str,
        *,
        max_symbols: int = 60,
        sampling_method: SamplingMethod = SamplingMethod.REPRESENTATIVE,
        date_range: tuple[date, date] | None = None,
        report_dir: Path | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> RunResult:
        """Execute the full research pipeline for *strategy_id*."""

        def _progress(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        # Step 1 — load strategy
        _progress("Loading strategy...")
        strategy = self.store.get(strategy_id)
        if strategy is None:
            raise ValueError(f"Strategy not found: {strategy_id}")

        # Step 2 — stock universe
        _progress("Fetching stock universe...")
        stock_list = await self.data_manager.get_stock_list(strategy.meta.market)

        if date_range is None:
            end = date.today()
            start = end - timedelta(days=365)
            date_range = (start, end)
        target_signals = max(1, self.config.evolution.min_signal_count)
        auto_expand = self.config.evolution.auto_expand_samples
        max_expansions = max(0, self.config.evolution.max_sample_expansions)

        final_result: RunResult | None = None
        sampling_history: list[SamplingAttempt] = []

        for expansion_idx in range(max_expansions + 1 if auto_expand else 1):
            attempt_max_symbols = self._expanded_max_symbols(
                max_symbols,
                expansion_idx,
                len(stock_list),
            )
            attempt_method = self._expanded_sampling_method(
                sampling_method,
                expansion_idx,
            )
            attempt_date_range = self._expanded_date_range(
                date_range,
                expansion_idx,
            )
            attempt_note = self._sampling_attempt_note(expansion_idx, sampling_method)

            if expansion_idx == 0:
                _progress(f"Sampling up to {attempt_max_symbols} stocks...")
            else:
                _progress(
                    "Expanding sample due to sparse signals: "
                    f"attempt {expansion_idx + 1}/{max_expansions + 1}, "
                    f"method={attempt_method.value}, "
                    f"symbols<={attempt_max_symbols}, "
                    f"window={attempt_date_range[0]}→{attempt_date_range[1]}"
                )

            attempt_result = await self._run_attempt(
                strategy,
                stock_list,
                max_symbols=attempt_max_symbols,
                sampling_method=attempt_method,
                date_range=attempt_date_range,
                on_progress=_progress,
            )
            final_result = attempt_result

            signal_count = attempt_result.evaluation.overall.signal_count
            accepted = signal_count >= target_signals
            sampling_history.append(
                SamplingAttempt(
                    attempt_num=expansion_idx + 1,
                    max_symbols=attempt_max_symbols,
                    date_range=attempt_date_range,
                    sampling_method=attempt_method,
                    sampling_reason=attempt_result.batch.sampling_reason,
                    selected_symbols=len(attempt_result.batch.symbols),
                    signal_count=signal_count,
                    accepted=accepted,
                    note=attempt_note,
                )
            )

            if accepted:
                if expansion_idx > 0:
                    _progress(
                        f"Signal target reached after expansion: {signal_count}/{target_signals}"
                    )
                break

            if not auto_expand or expansion_idx >= max_expansions:
                _progress(
                    f"Signal target not reached after sampling attempts: "
                    f"{signal_count}/{target_signals}"
                )
                break

            _progress(f"Only {signal_count}/{target_signals} signals — scheduling broader sampling")

        if final_result is None:
            raise RuntimeError("Pipeline failed before any sampling attempt completed.")

        self._attach_sampling_history(
            final_result.batch,
            sampling_history,
            requested_max_symbols=max_symbols,
            target_signals=target_signals,
            reached_signals=final_result.evaluation.overall.signal_count,
        )

        # Step 7 — persist evaluation
        self.store.save_evaluation(final_result.evaluation)

        # Step 8 — optional report file
        report_path: Path | None = None
        if report_dir is not None:
            report_dir = Path(report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{strategy_id}_report.md"
            Reporter.to_file(
                final_result.evaluation,
                report_path,
                format="markdown",
                strategy=strategy,
            )
            _progress(f"Report saved to {report_path}")

        final_result.report_path = report_path
        return final_result

    # -- helpers ---------------------------------------------------------

    async def _run_attempt(
        self,
        strategy: Strategy,
        stock_list: list[StockInfo],
        *,
        max_symbols: int,
        sampling_method: SamplingMethod,
        date_range: tuple[date, date],
        on_progress: Callable[[str], None],
    ) -> RunResult:
        """Run one sampling/backtest/evaluation attempt."""
        sampler = AdaptiveSampler(max_symbols=max_symbols)
        batch = await sampler.sample(
            strategy,
            stock_list,
            method=sampling_method,
            date_range=date_range,
        )
        on_progress(f"Selected {len(batch.symbols)} symbols")

        on_progress("Fetching historical data...")
        data = await self._fetch_data(strategy, batch.symbols, date_range)
        on_progress(f"Data fetched for {len(data)} symbols")

        if not data:
            raise RuntimeError("No valid market data could be fetched for any sampled symbol.")

        on_progress("Building market context...")
        contexts = await self._build_indicator_contexts(
            strategy,
            stock_list,
            data,
            date_range,
            batch,
        )
        if contexts:
            on_progress(f"Context ready for {len(contexts)} symbols")

        self._autoload_custom_factors(on_progress)

        on_progress("Running backtest...")
        engine = BacktestEngine(
            slippage=self.config.backtest.slippage,
            commission=self.config.backtest.commission,
            min_data_days=self.config.backtest.min_data_days,
            fill_policy=self.config.backtest.fill_policy,
        )
        result = engine.run(strategy, data, batch, contexts=contexts or None)
        on_progress(f"Backtest complete: {result.total_signals} signals")

        on_progress("Evaluating results...")
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(
            result,
            strategy,
            market_data=data,
            contexts=contexts or None,
            backtest_config=self.config.backtest,
        )
        on_progress(f"Confidence score: {evaluation.confidence_score:.2%}")
        data_source_health = self.data_manager.health_status()
        data_quality = build_data_quality_report(
            event_context=evaluation.event_context,
            data_source_health=data_source_health,
        )
        if data_quality.risk_level != "low":
            on_progress(f"Data quality gate: {data_quality.summary}")

        return RunResult(
            strategy=strategy,
            batch=batch,
            backtest_result=result,
            evaluation=evaluation,
            data_source_health=data_source_health,
            data_quality=data_quality,
            _engine=engine,
            _data=data,
            _contexts=contexts,
        )

    def _attach_sampling_history(
        self,
        batch: SampleBatch,
        history: list[SamplingAttempt],
        *,
        requested_max_symbols: int,
        target_signals: int,
        reached_signals: int,
    ) -> None:
        """Attach auto-expansion metadata to the final sample batch."""
        batch.requested_max_symbols = requested_max_symbols
        batch.sampling_attempt = len(history) if history else 1
        batch.signal_count_target = target_signals
        batch.signal_count_reached = reached_signals
        batch.insufficient_signals = reached_signals < target_signals
        batch.sampling_history = history

    def _expanded_max_symbols(
        self,
        requested_max_symbols: int,
        expansion_idx: int,
        available_symbols: int,
    ) -> int:
        """Broaden the symbol budget on each sparse-signal retry."""
        step = max(1, self.config.evolution.sample_expansion_symbol_step)
        requested = max(1, requested_max_symbols)
        expanded = requested + expansion_idx * step
        return min(max(1, available_symbols), expanded)

    def _expanded_date_range(
        self,
        requested_range: tuple[date, date],
        expansion_idx: int,
    ) -> tuple[date, date]:
        """Extend the requested date range backward to gather more signals."""
        if expansion_idx <= 0:
            return requested_range
        extra_days = expansion_idx * max(1, self.config.evolution.sample_expansion_window_days)
        return (requested_range[0] - timedelta(days=extra_days), requested_range[1])

    @staticmethod
    def _expanded_sampling_method(
        requested_method: SamplingMethod,
        expansion_idx: int,
    ) -> SamplingMethod:
        """Escalate to broader/more targeted samplers on later retries."""
        if expansion_idx <= 1:
            return requested_method

        fallback_order = [requested_method]
        for candidate in (
            SamplingMethod.STRATEGY_SCOPED,
            SamplingMethod.REGIME_BASED,
        ):
            if candidate not in fallback_order:
                fallback_order.append(candidate)

        method_idx = min(expansion_idx - 1, len(fallback_order) - 1)
        return fallback_order[method_idx]

    @staticmethod
    def _sampling_attempt_note(
        expansion_idx: int,
        requested_method: SamplingMethod,
    ) -> str:
        """Summarize why this sampling attempt was selected."""
        if expansion_idx == 0:
            return "initial sampling plan"
        if expansion_idx == 1:
            return f"expanded date range and symbol budget from {requested_method.value}"
        return "broadened sampling method after repeated sparse-signal attempts"

    def _autoload_custom_factors(
        self,
        _progress: Callable[[str], None],
    ) -> None:
        """Register active custom factors from FactorStore into IndicatorRegistry."""
        try:
            from alphaevo.alpha_factory.factor_store import FactorStore
            from alphaevo.alpha_factory.factory import register_factor_record
            from alphaevo.alpha_factory.sandbox import FactorSandbox
        except Exception:
            return

        store = FactorStore(self.config.db_path)
        active = store.list_all(status="active")
        if not active:
            return

        from alphaevo.backtest.indicators import IndicatorRegistry

        sandbox = FactorSandbox()
        loaded = 0
        for record in active:
            if IndicatorRegistry.is_registered(record.name):
                continue
            try:
                register_factor_record(record, sandbox=sandbox)
                loaded += 1
            except Exception:
                logger.debug("Failed to load custom factor: %s", record.name, exc_info=True)
        if loaded:
            _progress(f"Loaded {loaded} custom factor(s) from Alpha Factory")

    async def _fetch_data(
        self,
        strategy: Strategy,
        symbols: list[str],
        date_range: tuple[date, date],
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all *symbols* concurrently."""
        fetch_start, fetch_end = self._history_window_for_strategy(strategy, date_range)

        async def _fetch_one(symbol: str) -> tuple[str, pd.DataFrame | None]:
            try:
                df = await self.data_manager.get_history(symbol, fetch_start, fetch_end)
                return (symbol, df)
            except Exception:
                logger.warning("Failed to fetch %s", symbol, exc_info=True)
                return (symbol, None)

        results = await asyncio.gather(*[_fetch_one(s) for s in symbols])
        return {sym: df for sym, df in results if df is not None and not df.empty}

    def _history_window_for_strategy(
        self,
        strategy: Strategy,
        date_range: tuple[date, date],
    ) -> tuple[date, date]:
        """Expand the requested date range with enough warmup history for indicators."""
        warmup_trading_days = self._estimate_indicator_warmup_days(strategy)
        warmup_calendar_days = max(60, math.ceil(warmup_trading_days * 1.6))
        return (date_range[0] - timedelta(days=warmup_calendar_days), date_range[1])

    def _estimate_indicator_warmup_days(self, strategy: Strategy) -> int:
        """Estimate the longest lookback needed by the strategy's indicators."""
        lookbacks = [self.config.backtest.min_data_days]
        conditions = (
            list(strategy.entry.conditions)
            + list(strategy.entry.triggers)
            + list(strategy.entry.filters)
            + list(strategy.entry.guards)
        )
        conditions.extend(strategy.exit.triggers)
        if strategy.exit.stop_loss.conditions:
            conditions.extend(strategy.exit.stop_loss.conditions)

        for condition in conditions:
            lookbacks.append(self._indicator_lookback(condition.indicator))

        if strategy.exit.stop_loss.type == "atr":
            atr_period = strategy.exit.stop_loss.atr_period
            lookbacks.append(
                _INDICATOR_LOOKBACKS["atr"]
                if atr_period is None
                else self._indicator_lookback(f"atr_{atr_period}")
            )

        tp_target = strategy.exit.take_profit.target
        if tp_target:
            lookbacks.append(self._target_ma_lookback(tp_target))

        return max(lookbacks)

    @staticmethod
    def _indicator_lookback(indicator: str) -> int:
        """Infer the lookback window for a strategy indicator token."""
        if indicator in _INDICATOR_LOOKBACKS:
            return _INDICATOR_LOOKBACKS[indicator]

        if match := re.fullmatch(r"ma(\d+)_above_ma(\d+)", indicator):
            return max(int(match.group(1)), int(match.group(2)))
        if match := re.fullmatch(r"ma(\d+)_ge_ma(\d+)_or_crossing", indicator):
            return max(int(match.group(1)), int(match.group(2))) + 1
        if match := re.fullmatch(
            r"(?:close_to_ma|close_above_ma|close_below_ma)(\d+)(?:_pct)?", indicator
        ):
            return int(match.group(1))
        if match := re.fullmatch(r"deviation_from_ma(\d+)_pct", indicator):
            return int(match.group(1))
        if match := re.fullmatch(r"ma(\d+)_slope", indicator):
            return int(match.group(1)) + 5
        if match := re.fullmatch(r"atr_(\d+)", indicator):
            return int(match.group(1)) + 1
        if match := re.fullmatch(
            r"macd_histogram_fast(\d+)_slow(\d+)_signal(\d+)",
            indicator,
        ):
            return int(match.group(2)) + int(match.group(3)) - 1
        if match := re.fullmatch(
            r"macd_cross_bullish_fast(\d+)_slow(\d+)_signal(\d+)",
            indicator,
        ):
            return int(match.group(2)) + int(match.group(3))
        if match := re.fullmatch(r"rsi_(\d+)", indicator):
            return int(match.group(1)) + 1
        if match := re.fullmatch(r"rsi_(\d+)_zscore", indicator):
            return max(50, int(match.group(1)) + 1)
        if match := re.fullmatch(
            r"(?:bollinger_band_width|price_above_bollinger_upper|price_below_bollinger_lower)"
            r"_(\d+)d(?:_std[0-9]+(?:p[0-9]+)?)?",
            indicator,
        ):
            return int(match.group(1))
        if match := re.fullmatch(r"volume_ratio_1d_(\d+)d", indicator):
            return int(match.group(1)) + 1
        if match := re.fullmatch(r"breakout_high_(\d+)d", indicator):
            return int(match.group(1)) + 1
        if match := re.fullmatch(r"price_position_(\d+)d", indicator):
            return int(match.group(1))
        if match := re.fullmatch(r"momentum_(\d+)d", indicator):
            return int(match.group(1)) + 1
        if match := re.fullmatch(r"avg_volume_(\d+)d", indicator):
            return int(match.group(1))
        if match := re.fullmatch(r"days_since_(?:high|low)_(\d+)d", indicator):
            return int(match.group(1))
        if match := re.fullmatch(r"volatility_(\d+)d", indicator):
            return int(match.group(1)) + 1
        if match := re.fullmatch(r"relative_strength_(\d+)d", indicator):
            return int(match.group(1)) + 1

        return 1

    @staticmethod
    def _target_ma_lookback(target: str) -> int:
        """Infer warmup for a take-profit moving-average target."""
        match = re.fullmatch(r"ma(\d+)", target.strip().lower())
        if match is None:
            return 1
        return int(match.group(1))

    async def _build_indicator_contexts(
        self,
        strategy: Strategy,
        stock_list: list[StockInfo],
        data: dict[str, pd.DataFrame],
        date_range: tuple[date, date],
        batch: SampleBatch,
    ) -> dict[str, IndicatorContext]:
        """Build per-symbol indicator context for L2/L3 indicators."""
        stock_lookup = {stock.symbol: stock for stock in stock_list}
        benchmark_df = await self._fetch_benchmark_data(strategy.meta.market, date_range)
        benchmark_context = self._build_market_context(benchmark_df)
        provider_context = await self._call_optional_data_manager_method(
            "get_market_context",
            strategy.meta.market,
        )
        market_context = self._merge_market_context(
            benchmark_context,
            provider_context if isinstance(provider_context, MarketContext) else None,
        )
        if (
            market_context is not None
            and market_context.regime is not None
            and market_context.regime not in batch.market_regimes
        ):
            batch.market_regimes.append(market_context.regime)

        async def _symbol_context(
            symbol: str,
        ) -> tuple[str, SectorInfo | None, EventContextSeries | None]:
            sector = await self._call_optional_data_manager_method(
                "get_sector_data",
                symbol,
            )
            events = await self._call_optional_data_manager_method(
                "get_event_context",
                symbol,
                date_range[0],
                date_range[1],
            )
            return (
                symbol,
                sector if isinstance(sector, SectorInfo) else None,
                events if isinstance(events, EventContextSeries) else None,
            )

        symbol_results = await asyncio.gather(*[_symbol_context(symbol) for symbol in data])
        sector_map = {symbol: sector for symbol, sector, _ in symbol_results}
        event_map = {symbol: events for symbol, _, events in symbol_results}

        contexts: dict[str, IndicatorContext] = {}
        for symbol, df in list(data.items()):
            merged_df, event_source = merge_event_context(df, event_map.get(symbol))
            data[symbol] = merged_df
            contexts[symbol] = IndicatorContext(
                benchmark_df=benchmark_df,
                sector_info=sector_map.get(symbol),
                stock_info=stock_lookup.get(symbol),
                market_context=market_context,
                event_context_source=event_source,
            )
        return contexts

    async def _fetch_benchmark_data(
        self,
        market: MarketType,
        date_range: tuple[date, date],
    ) -> pd.DataFrame | None:
        """Fetch benchmark data for the active market when the adapter supports it."""
        benchmark_symbol = self._benchmark_symbol(
            market,
            self.config.data.adapter,
        )
        if benchmark_symbol is None:
            return None

        raw = await self._call_optional_data_manager_method(
            "get_index_data",
            benchmark_symbol,
            date_range[0],
            date_range[1],
        )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return None
        if "close" not in raw.columns:
            return None

        benchmark_df = raw.copy()
        if "date" in benchmark_df.columns:
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"]).dt.date
        for column in ("open", "high", "low"):
            if column not in benchmark_df.columns:
                benchmark_df[column] = benchmark_df["close"]
        return cast("pd.DataFrame", benchmark_df.reset_index(drop=True))

    async def _call_optional_data_manager_method(
        self,
        method_name: str,
        *args: Any,
    ) -> Any:
        """Call an optional DataManager method, tolerating missing support."""
        method = getattr(self.data_manager, method_name, None)
        if method is None:
            return None

        try:
            result = method(*args)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception:
            logger.debug(
                "Optional data-manager method %s failed",
                method_name,
                exc_info=True,
            )
            return None

    @staticmethod
    def _build_market_context(benchmark_df: pd.DataFrame | None) -> MarketContext | None:
        """Build a coarse market snapshot from benchmark index data."""
        if benchmark_df is None or benchmark_df.empty or "close" not in benchmark_df.columns:
            return None

        regime = None
        if len(benchmark_df) >= 20:
            regime_source = benchmark_df.copy()
            for column in ("high", "low"):
                if column not in regime_source.columns:
                    regime_source[column] = regime_source["close"]
            regime = RegimeDetector().detect(regime_source.reset_index(drop=True))

        index_change_pct = None
        if len(benchmark_df) >= 2:
            prev_close = float(benchmark_df["close"].iloc[-2])
            last_close = float(benchmark_df["close"].iloc[-1])
            if prev_close > 0:
                index_change_pct = round((last_close - prev_close) / prev_close, 4)

        return MarketContext(
            index_change_pct=index_change_pct,
            regime=regime,
        )

    @staticmethod
    def _merge_market_context(
        benchmark_context: MarketContext | None,
        provider_context: MarketContext | None,
    ) -> MarketContext | None:
        """Merge benchmark-derived regime with provider market breadth/sentiment."""
        if benchmark_context is None:
            return provider_context
        if provider_context is None:
            return benchmark_context

        return MarketContext(
            index_change_pct=(
                provider_context.index_change_pct
                if provider_context.index_change_pct is not None
                else benchmark_context.index_change_pct
            ),
            breadth=provider_context.breadth,
            regime=benchmark_context.regime or provider_context.regime,
            sentiment_index=provider_context.sentiment_index,
            sector_leaders=provider_context.sector_leaders,
            sector_laggards=provider_context.sector_laggards,
        )

    @staticmethod
    def _benchmark_symbol(
        market: MarketType,
        adapter_name: str,
    ) -> str | None:
        """Resolve a best-effort benchmark symbol for the current market."""
        if market == MarketType.US:
            return "^GSPC" if adapter_name == "yfinance" else None
        if market == MarketType.HK:
            return "^HSI" if adapter_name == "yfinance" else None
        if market == MarketType.A_SHARE:
            if adapter_name == "yfinance":
                return "000001.SS"
            if adapter_name == "akshare":
                return "000001"
        return None

    def run_sync(self, strategy_id: str, **kwargs: Any) -> RunResult:
        """Synchronous wrapper for CLI use."""
        return asyncio.run(self.run(strategy_id, **kwargs))

    def ensure_builtin_strategies(self) -> int:
        """Import built-in strategy YAML files if not already in the store."""
        # Resolve relative to the package source tree
        builtin_dir = (
            Path(__file__).resolve().parent.parent.parent.parent / "strategies" / "builtin"
        )
        if not builtin_dir.exists():
            builtin_dir = Path("strategies/builtin")
        if builtin_dir.exists():
            return self.store.import_builtin_strategies(builtin_dir)
        logger.warning("Built-in strategies directory not found")
        return 0
