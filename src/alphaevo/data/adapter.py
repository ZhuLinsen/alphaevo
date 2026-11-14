"""Data adapter abstract base class and DataManager."""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from alphaevo.data.cache import DataCache

if TYPE_CHECKING:
    from datetime import date

    from alphaevo.core.config import AppConfig
    from alphaevo.models.enums import MarketType
    from alphaevo.models.market import (
        EventContextSeries,
        MarketContext,
        MarketSnapshot,
        RealTimeQuote,
        SectorInfo,
        StockInfo,
    )


class DataAdapter(ABC):
    """Abstract interface for market data sources.

    Implement this to plug in any data provider (yfinance, akshare,
    daily_stock_analysis, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""

    @abstractmethod
    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Fetch daily OHLCV data.

        Returns DataFrame with columns:
        date, open, high, low, close, volume, amount (optional)
        """

    @abstractmethod
    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Get list of stocks in a market."""

    async def get_realtime_quote(self, symbol: str) -> RealTimeQuote | None:
        """Get real-time quote (optional, not all adapters support this)."""
        return None

    async def get_sector_data(self, symbol: str) -> SectorInfo | None:
        """Get sector/industry information (optional)."""
        return None

    async def get_snapshot(self, symbol: str, target_date: date) -> MarketSnapshot | None:
        """Build a full MarketSnapshot for a symbol on a given date.

        Default implementation fetches daily data and constructs a snapshot.
        Adapters may override for efficiency.
        """
        return None

    async def get_index_data(self, index_symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch benchmark index data.

        Adapters that do not support indices return an empty DataFrame.
        """
        return cast("pd.DataFrame", pd.DataFrame())

    async def get_event_context(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> EventContextSeries | None:
        """Fetch optional date-aligned event/news context for a symbol.

        Providers that do not support event/news context should return ``None``.
        The returned series is expected to align to trading dates, not just sparse
        event timestamps, so it can be injected directly into backtest data.
        """
        return None

    async def get_market_context(self, market: MarketType) -> MarketContext | None:
        """Fetch optional market-wide context for the active market.

        This is deliberately a coarse snapshot, not a historical series. Providers
        should only return fields they can support without fabricating data.
        """
        return None


class DataSourceHealth(BaseModel):
    """Observable health state for one data adapter in a DataManager chain."""

    name: str
    priority: int
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    disabled: bool = False
    disabled_seconds_remaining: float = 0.0
    last_error: str = ""
    last_success_at: float | None = None
    last_failure_at: float | None = None
    recent_errors: list[str] = Field(default_factory=list)


class DataManager:
    """Unified data manager with multi-source fallback.

    Tries adapters in priority order. If the primary fails, falls
    through to the next available adapter.
    """

    def __init__(
        self,
        adapters: list[DataAdapter],
        *,
        cache: DataCache | None = None,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if not adapters:
            raise ValueError("At least one DataAdapter is required")
        self._adapters = adapters
        self._cache = cache
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = max(0.0, cooldown_seconds)
        self._failure_counts: dict[str, int] = {}
        self._disabled_until: dict[str, float] = {}
        self._success_totals: dict[str, int] = {}
        self._failure_totals: dict[str, int] = {}
        self._last_errors: dict[str, str] = {}
        self._last_success_at: dict[str, float] = {}
        self._last_failure_at: dict[str, float] = {}
        self._recent_errors: dict[str, list[str]] = {}

    @property
    def primary(self) -> DataAdapter:
        return self._adapters[0]

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Fetch daily data with fallback across adapters."""
        errors: list[tuple[str, str]] = []
        for adapter in self._adapters:
            if self._is_disabled(adapter):
                errors.append((adapter.name, "temporarily disabled after repeated failures"))
                continue
            try:
                df = await adapter.get_daily_data(symbol, days)
                if df is not None and not df.empty:
                    self._record_success(adapter)
                    return df
                self._record_failure(adapter, "empty daily data")
                errors.append((adapter.name, "empty daily data"))
            except Exception as e:
                self._record_failure(adapter, str(e))
                errors.append((adapter.name, str(e)))
                continue
        detail = "; ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"All data adapters failed for {symbol}: {detail}")

    async def get_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch history for a date range."""
        if start > end:
            raise ValueError(f"start date ({start}) must be <= end date ({end})")
        cached = self._get_cached_history(symbol, start, end)
        if cached is not None:
            return cached

        days = (end - start).days + 30  # buffer for non-trading days
        df = await self.get_daily_data(symbol, days)
        history = self._prepare_history_frame(symbol, df, start=start, end=end)
        if history.empty:
            raise RuntimeError(f"No historical data returned for {symbol}: {start} → {end}")

        if self._cache is not None:
            self._cache.put(symbol, start, end, history)
        return cast("pd.DataFrame", history.copy())

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Get stock list with fallback."""
        for adapter in self._adapters:
            if self._is_disabled(adapter):
                continue
            try:
                stocks = await adapter.get_stock_list(market)
                if stocks:
                    self._record_success(adapter)
                    return stocks
                self._record_failure(adapter, "empty stock list")
            except Exception as exc:
                self._record_failure(adapter, str(exc))
                continue
        return []

    async def get_snapshot(self, symbol: str, target_date: date) -> MarketSnapshot | None:
        """Get full snapshot with fallback."""
        for adapter in self._adapters:
            if self._is_disabled(adapter):
                continue
            try:
                snapshot = await adapter.get_snapshot(symbol, target_date)
                if snapshot is not None:
                    self._record_success(adapter)
                    return snapshot
            except Exception as exc:
                self._record_failure(adapter, str(exc))
                continue
        return None

    async def get_realtime_quote(self, symbol: str) -> RealTimeQuote | None:
        """Fetch a real-time quote with adapter fallback."""
        for adapter in self._adapters:
            if self._is_disabled(adapter):
                continue
            try:
                quote = await adapter.get_realtime_quote(symbol)
                if quote is not None:
                    self._record_success(adapter)
                    return quote
            except Exception as exc:
                self._record_failure(adapter, str(exc))
                continue
        return None

    async def get_index_data(self, index_symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch benchmark index data with adapter fallback."""
        if start > end:
            raise ValueError(f"start date ({start}) must be <= end date ({end})")

        cache_key = f"__index__{index_symbol}"
        cached = self._get_cached_history(cache_key, start, end)
        if cached is not None:
            return cached

        for adapter in self._adapters:
            if self._is_disabled(adapter):
                continue
            try:
                df = await adapter.get_index_data(index_symbol, start, end)
                if df is not None and not df.empty:
                    history = self._prepare_history_frame(
                        index_symbol,
                        df,
                        start=start,
                        end=end,
                        required_cols={"date", "close"},
                    )
                    if history.empty:
                        self._record_failure(adapter, "empty normalized index data")
                        continue
                    if self._cache is not None:
                        self._cache.put(cache_key, start, end, history)
                    self._record_success(adapter)
                    return cast("pd.DataFrame", history.copy())
                self._record_failure(adapter, "empty index data")
            except Exception as exc:
                self._record_failure(adapter, str(exc))
                continue
        return cast("pd.DataFrame", pd.DataFrame())

    async def get_sector_data(self, symbol: str) -> SectorInfo | None:
        """Fetch sector information with adapter fallback."""
        for adapter in self._adapters:
            if self._is_disabled(adapter):
                continue
            try:
                info = await adapter.get_sector_data(symbol)
                if info is not None:
                    self._record_success(adapter)
                    return info
            except Exception as exc:
                self._record_failure(adapter, str(exc))
                continue
        return None

    async def get_event_context(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> EventContextSeries | None:
        """Fetch event/news context with adapter fallback."""
        for adapter in self._adapters:
            if self._is_disabled(adapter):
                continue
            try:
                context = await adapter.get_event_context(symbol, start, end)
                if context is not None and context.records:
                    self._record_success(adapter)
                    return context
            except Exception as exc:
                self._record_failure(adapter, str(exc))
                continue
        return None

    async def get_market_context(self, market: MarketType) -> MarketContext | None:
        """Fetch market-wide context with adapter fallback."""
        for adapter in self._adapters:
            if self._is_disabled(adapter):
                continue
            try:
                context = await adapter.get_market_context(market)
                if context is not None:
                    self._record_success(adapter)
                    return context
            except Exception as exc:
                self._record_failure(adapter, str(exc))
                continue
        return None

    def _is_disabled(self, adapter: DataAdapter) -> bool:
        """Return whether an adapter is inside its circuit-breaker cooldown."""
        disabled_until = self._disabled_until.get(adapter.name, 0.0)
        if disabled_until <= time.monotonic():
            self._disabled_until.pop(adapter.name, None)
            return False
        return True

    def _record_success(self, adapter: DataAdapter) -> None:
        """Clear source health state after a successful provider call."""
        now = time.monotonic()
        self._success_totals[adapter.name] = self._success_totals.get(adapter.name, 0) + 1
        self._last_success_at[adapter.name] = now
        self._failure_counts.pop(adapter.name, None)
        self._disabled_until.pop(adapter.name, None)

    def _record_failure(self, adapter: DataAdapter, reason: str) -> None:
        """Track repeated provider failures and temporarily skip noisy sources."""
        now = time.monotonic()
        count = self._failure_counts.get(adapter.name, 0) + 1
        self._failure_counts[adapter.name] = count
        self._failure_totals[adapter.name] = self._failure_totals.get(adapter.name, 0) + 1
        self._last_errors[adapter.name] = reason
        self._last_failure_at[adapter.name] = now
        recent = self._recent_errors.setdefault(adapter.name, [])
        recent.append(reason)
        del recent[:-3]
        if count >= self._failure_threshold:
            self._disabled_until[adapter.name] = time.monotonic() + self._cooldown_seconds

    def health_status(self) -> list[DataSourceHealth]:
        """Return current adapter health in priority order for reports/debugging."""
        now = time.monotonic()
        health: list[DataSourceHealth] = []
        for priority, adapter in enumerate(self._adapters):
            disabled_until = self._disabled_until.get(adapter.name, 0.0)
            remaining = max(0.0, disabled_until - now)
            health.append(
                DataSourceHealth(
                    name=adapter.name,
                    priority=priority,
                    success_count=self._success_totals.get(adapter.name, 0),
                    failure_count=self._failure_totals.get(adapter.name, 0),
                    consecutive_failures=self._failure_counts.get(adapter.name, 0),
                    disabled=remaining > 0,
                    disabled_seconds_remaining=remaining,
                    last_error=self._last_errors.get(adapter.name, ""),
                    last_success_at=self._last_success_at.get(adapter.name),
                    last_failure_at=self._last_failure_at.get(adapter.name),
                    recent_errors=list(self._recent_errors.get(adapter.name, [])),
                )
            )
        return health

    def _get_cached_history(
        self,
        cache_key: str,
        start: date,
        end: date,
    ) -> pd.DataFrame | None:
        """Return a normalized cached history frame when available."""
        if self._cache is None:
            return None

        cached = self._cache.get(cache_key, start, end)
        if cached is None or cached.empty:
            return None
        return self._prepare_history_frame(cache_key, cached, start=start, end=end)

    @staticmethod
    def _prepare_history_frame(
        symbol: str,
        df: pd.DataFrame,
        *,
        start: date,
        end: date,
        required_cols: set[str] | None = None,
    ) -> pd.DataFrame:
        """Normalize and clip a history dataframe to the requested date range."""
        required = required_cols or {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Adapter returned DataFrame missing required columns for {symbol}: {missing}"
            )

        normalized = df.copy()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.date
        normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        # Recompute prev_close BEFORE clipping so the first row in the
        # clipped range still has a valid value (not NaN).
        if "prev_close" not in normalized.columns or normalized["prev_close"].isna().any():
            normalized["prev_close"] = normalized["close"].shift(1)

        mask = (normalized["date"] >= start) & (normalized["date"] <= end)
        return cast("pd.DataFrame", normalized.loc[mask].reset_index(drop=True))


def get_adapter_chain(config: AppConfig) -> list[DataAdapter]:
    """Instantiate the configured adapter chain in fallback priority order."""
    adapter_name = config.data.adapter

    if adapter_name == "auto":
        from alphaevo.data.adapters.akshare import AkShareAdapter
        from alphaevo.data.adapters.tencent import TencentAshareAdapter
        from alphaevo.data.adapters.yfinance import YFinanceAdapter

        adapters: list[DataAdapter] = []
        if config.data.dsa_path:
            from alphaevo.data.adapters.dsa import DSAAdapter

            with contextlib.suppress(ImportError):
                adapters.append(DSAAdapter(dsa_path=config.data.dsa_path))
        adapters.extend([TencentAshareAdapter(), AkShareAdapter(), YFinanceAdapter()])
        return adapters

    if adapter_name == "yfinance":
        from alphaevo.data.adapters.yfinance import YFinanceAdapter

        return [YFinanceAdapter()]

    if adapter_name == "akshare":
        from alphaevo.data.adapters.akshare import AkShareAdapter

        return [AkShareAdapter()]

    if adapter_name in {"tencent", "auto"}:
        from alphaevo.data.adapters.tencent import TencentAshareAdapter

        return [TencentAshareAdapter()]

    if adapter_name == "dsa":
        from alphaevo.data.adapters.dsa import DSAAdapter

        try:
            return [DSAAdapter(dsa_path=config.data.dsa_path)]
        except ImportError as err:
            raise ValueError(
                "Adapter 'dsa' is an optional daily_stock_analysis bridge. "
                "Configure ALPHAEVO_DSA_PATH (or data.dsa_path) to enable it."
            ) from err

    raise ValueError(f"Unknown adapter: {adapter_name}")


def get_adapter(config: AppConfig) -> DataAdapter:
    """Instantiate the configured primary adapter for lightweight workflows."""
    return get_adapter_chain(config)[0]
