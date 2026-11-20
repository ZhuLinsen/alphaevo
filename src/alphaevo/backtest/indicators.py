"""Indicator registry and built-in indicator implementations.

Each indicator is a pure function: (df, idx, ctx?) -> float | bool
registered via @IndicatorRegistry.register("name").

Indicators are organized into 3 tiers (see AGENTS.md §9):
  L1 MVP:  Only needs OHLCV data — implemented with real logic
  L2 Ext:  Needs benchmark/sector data — real logic when ctx available, degraded otherwise
  L3 Adv:  Needs news/event APIs — explicit data when present, price/volume proxies otherwise
"""

from __future__ import annotations

import inspect
import re
from bisect import bisect_left, bisect_right
from datetime import date
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from alphaevo.models.market import EventContextSeries, IndicatorContext

# ── Type definitions ─────────────────────────────────────────────────

IndicatorValue = float | bool
_NO_EVENT_DAYS = 999
_EVENT_PROXY_COLUMNS = {
    "negative_news_score",
    "news_sentiment_score",
    "days_since_event",
    "pre_event_close",
    "price_above_pre_event",
    "already_overreacted",
}
_MA_TEMPLATE_NAMES = {
    "maN_above_maM",
    "maN_ge_maM_or_crossing",
    "close_to_maN_pct",
    "close_above_maN",
    "close_below_maN",
    "deviation_from_maN_pct",
    "maN_slope",
}
_WINDOW_TEMPLATE_NAMES = {
    "atr_N",
    "rsi_N",
    "rsi_N_zscore",
    "macd_histogram_fastN_slowM_signalK",
    "macd_cross_bullish_fastN_slowM_signalK",
    "bollinger_band_width_Nd",
    "bollinger_band_width_Nd_stdS",
    "price_above_bollinger_upper_Nd",
    "price_above_bollinger_upper_Nd_stdS",
    "price_below_bollinger_lower_Nd",
    "price_below_bollinger_lower_Nd_stdS",
    "breakout_high_Nd",
    "price_position_Nd",
    "volume_ratio_1d_Nd",
    "momentum_Nd",
    "avg_volume_Nd",
    "days_since_high_Nd",
    "days_since_low_Nd",
    "volatility_Nd",
    "relative_strength_Nd",
}


class IndicatorFn(Protocol):
    def __call__(
        self, df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None
    ) -> IndicatorValue: ...


# ── Registry ─────────────────────────────────────────────────────────


class IndicatorRegistry:
    """Global registry for all computable indicators.

    Each indicator is a pure function that computes a value at a given
    bar index from a DataFrame of OHLCV data (+ optional context).
    """

    _registry: dict[str, IndicatorFn] = {}
    _templated_registry: dict[str, IndicatorFn] = {}
    _takes_ctx: dict[str, bool] = {}

    @classmethod
    def _cache_signature(cls, name: str, fn: IndicatorFn) -> bool:
        """Cache whether *fn* accepts a third ``ctx`` parameter."""
        takes_ctx = len(inspect.signature(fn).parameters) >= 3
        cls._takes_ctx[name] = takes_ctx
        return takes_ctx

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register an indicator function."""

        def decorator(fn: IndicatorFn) -> IndicatorFn:
            cls._registry[name] = fn
            cls._cache_signature(name, fn)
            return fn

        return decorator

    @classmethod
    def compute(
        cls,
        name: str,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> IndicatorValue:
        """Compute indicator value at given index."""
        if name not in cls._registry and name not in cls._templated_registry:
            cls._register_templated(name)
        if name not in cls._registry and name not in cls._templated_registry:
            raise KeyError(f"Unknown indicator: '{name}'. Available: {cls.available()}")
        fn = cls._registry.get(name) or cls._templated_registry[name]
        takes_ctx = cls._takes_ctx.get(name)
        if takes_ctx is None:
            takes_ctx = cls._cache_signature(name, fn)
        if takes_ctx:
            return fn(df, idx, ctx)
        return fn(df, idx)

    @classmethod
    def precompute(
        cls,
        names: Sequence[str],
        df: pd.DataFrame,
        ctx: IndicatorContext | None = None,
    ) -> dict[str, list[IndicatorValue]]:
        """Precompute O(N) vectorized series for a collection of indicator names."""
        unique_names = list(dict.fromkeys(names))
        if not unique_names or df.empty:
            return {name: [] for name in unique_names}

        state = _PrecomputeState(df, ctx)
        result: dict[str, list[IndicatorValue]] = {}
        for name in unique_names:
            if name not in cls._registry and name not in cls._templated_registry:
                cls._register_templated(name)
            if name not in cls._registry and name not in cls._templated_registry:
                raise KeyError(f"Unknown indicator: '{name}'. Available: {cls.available()}")
            if name in cls._dynamic_registry:
                result[name] = [cls.compute(name, df, i, ctx) for i in range(len(df))]
                continue
            series = state.compute(name)
            if series is None:
                series = [cls.compute(name, df, i, ctx) for i in range(len(df))]
            result[name] = series
        return result

    @classmethod
    def available(cls) -> list[str]:
        """List all registered indicator names."""
        names = set(cls._registry.keys())
        names.update(cls._templated_registry.keys())
        names.update(_MA_TEMPLATE_NAMES)
        names.update(_WINDOW_TEMPLATE_NAMES)
        return sorted(names)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        if name in cls._registry or name in cls._templated_registry:
            return True
        return cls._register_templated(name)

    # ── Dynamic registration for Alpha Factory factors ──────────────

    _dynamic_registry: dict[str, IndicatorFn] = {}

    @classmethod
    def register_dynamic(cls, name: str, fn: IndicatorFn) -> None:
        """Register a factor at runtime (e.g., from Alpha Factory).

        Unlike the decorator-based ``register()``, this allows code-generated
        indicators to be hot-loaded without restart.  Dynamic factors can be
        unregistered later via ``unregister_dynamic()``.
        """
        cls._dynamic_registry[name] = fn
        cls._registry[name] = fn
        cls._cache_signature(name, fn)

    @classmethod
    def unregister_dynamic(cls, name: str) -> bool:
        """Remove a dynamically registered factor. Returns True if it existed."""
        removed = cls._dynamic_registry.pop(name, None) is not None
        if removed:
            cls._registry.pop(name, None)
            cls._takes_ctx.pop(name, None)
        return removed

    @classmethod
    def dynamic_names(cls) -> list[str]:
        """List names of all dynamically registered factors."""
        return sorted(cls._dynamic_registry.keys())

    @classmethod
    def _register_templated(cls, name: str) -> bool:
        """Register a lazily generated indicator template on first use."""
        if name in cls._templated_registry:
            return True
        fn = _build_templated_indicator(name)
        if fn is None:
            return False
        cls._templated_registry[name] = fn
        cls._cache_signature(name, fn)
        return True


class _PrecomputeState:
    """Lazy O(N) vectorized series calculator for ``IndicatorRegistry.precompute``."""

    def __init__(self, df: pd.DataFrame, ctx: IndicatorContext | None) -> None:
        self.df = df
        self.ctx = ctx
        self.n = len(df)
        self.close_s = df["close"].astype(float)
        self.open_s = df["open"].astype(float)
        self.high_s = df["high"].astype(float)
        self.low_s = df["low"].astype(float)
        self.vol_s = df["volume"].astype(float)
        self.close = self.close_s.to_numpy(dtype=float)
        self.open = self.open_s.to_numpy(dtype=float)
        self.high = self.high_s.to_numpy(dtype=float)
        self.low = self.low_s.to_numpy(dtype=float)
        self.vol = self.vol_s.to_numpy(dtype=float)
        self._sma_cache: dict[int, np.ndarray] = {}
        self._rsi_cache: dict[int, np.ndarray] = {}
        self._tr: np.ndarray | None = None
        self._event_df: pd.DataFrame | None = None

    def _sma(self, period: int) -> np.ndarray:
        cached = self._sma_cache.get(period)
        if cached is not None:
            return cached
        arr = (
            self.close_s.rolling(period).mean().to_numpy(dtype=float)
            if period > 0
            else np.full(self.n, np.nan, dtype=float)
        )
        self._sma_cache[period] = arr
        return arr

    def _rsi(self, period: int) -> np.ndarray:
        cached = self._rsi_cache.get(period)
        if cached is not None:
            return cached
        out = np.full(self.n, 50.0, dtype=float)
        if period > 0 and self.n > period:
            changes = self.close_s.diff().iloc[1:]
            gains = changes.clip(lower=0.0)
            losses = -changes.clip(upper=0.0)
            ag = gains.ewm(com=period - 1, min_periods=period).mean().to_numpy(dtype=float)
            al = losses.ewm(com=period - 1, min_periods=period).mean().to_numpy(dtype=float)
            ag_tail = ag[period - 1 :]
            al_tail = al[period - 1 :]
            valid = ~np.isnan(ag_tail) & ~np.isnan(al_tail)
            safe_al = np.where(al_tail == 0.0, 1.0, al_tail)
            vals = np.where(al_tail == 0.0, 100.0, 100.0 - (100.0 / (1.0 + (ag_tail / safe_al))))
            out[period:] = np.where(valid, vals, 50.0)
        self._rsi_cache[period] = out
        return out

    def _true_range(self) -> np.ndarray:
        if self._tr is not None:
            return self._tr
        tr: np.ndarray = cast("np.ndarray", np.asarray(self.high - self.low, dtype=float))
        if self.n > 1:
            tr[1:] = np.maximum(
                self.high[1:] - self.low[1:],
                np.maximum(
                    np.abs(self.high[1:] - self.close[:-1]),
                    np.abs(self.low[1:] - self.close[:-1]),
                ),
            )
        self._tr = tr
        return tr

    def _macd(
        self, fast: int, slow: int, signal: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if fast <= 0 or slow <= 0 or signal <= 0 or fast >= slow:
            return None
        ema_fast = self.close_s.ewm(span=fast, adjust=False).mean()
        ema_slow = self.close_s.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return (
            macd_line.to_numpy(dtype=float),
            signal_line.to_numpy(dtype=float),
            histogram.to_numpy(dtype=float),
        )

    def _bollinger(
        self, period: int, std_mult: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if period <= 0 or std_mult <= 0 or self.n < period:
            return None
        mid = self._sma(period)
        std = self.close_s.rolling(period).std(ddof=1).to_numpy(dtype=float)
        return (mid, mid + std_mult * std, mid - std_mult * std)

    def _event_source(self, column: str) -> pd.DataFrame:
        if column in self.df.columns:
            return self.df
        if self._event_df is None:
            self._event_df = enrich_with_event_proxies(self.df)
        return self._event_df

    def compute(self, name: str) -> list[IndicatorValue] | None:
        """Return precomputed list for *name*, or ``None`` if scalar fallback needed."""
        n = self.n
        if n == 0:
            return []

        if match := re.fullmatch(r"ma(\d+)_above_ma(\d+)", name):
            fast, slow = (
                _parse_positive_period(match.group(1)),
                _parse_positive_period(match.group(2)),
            )
            if fast is None or slow is None:
                return None
            fma, sma = self._sma(fast), self._sma(slow)
            return cast(
                "list[IndicatorValue]",
                np.where(~np.isnan(fma) & ~np.isnan(sma), fma > sma, False).tolist(),
            )

        if match := re.fullmatch(r"ma(\d+)_ge_ma(\d+)_or_crossing", name):
            fast, slow = (
                _parse_positive_period(match.group(1)),
                _parse_positive_period(match.group(2)),
            )
            if fast is None or slow is None:
                return None
            fma, sma = self._sma(fast), self._sma(slow)
            pfma = np.roll(fma, 1)
            psma = np.roll(sma, 1)
            pfma[0], psma[0] = np.nan, np.nan
            valid = ~np.isnan(fma) & ~np.isnan(sma) & ~np.isnan(pfma) & ~np.isnan(psma)
            res = (fma >= sma) | ((pfma < psma) & (fma >= sma * 0.998))
            return cast("list[IndicatorValue]", np.where(valid, res, False).tolist())

        if match := re.fullmatch(r"close_to_ma(\d+)_pct", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            ma = self._sma(p)
            valid = ~np.isnan(ma) & (ma != 0.0)
            safe_ma = np.where(valid, ma, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, np.abs(self.close - ma) / safe_ma, 1.0).tolist(),
            )

        if match := re.fullmatch(r"close_above_ma(\d+)", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            ma = self._sma(p)
            return cast(
                "list[IndicatorValue]", np.where(~np.isnan(ma), self.close > ma, False).tolist()
            )

        if match := re.fullmatch(r"close_below_ma(\d+)", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            ma = self._sma(p)
            return cast(
                "list[IndicatorValue]", np.where(~np.isnan(ma), self.close < ma, False).tolist()
            )

        if match := re.fullmatch(r"deviation_from_ma(\d+)_pct", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            ma = self._sma(p)
            valid = ~np.isnan(ma) & (ma != 0.0)
            safe_ma = np.where(valid, ma, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, (self.close - ma) / safe_ma, 0.0).tolist(),
            )

        if match := re.fullmatch(r"ma(\d+)_slope", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            ma = self._sma(p)
            ma_5ago = np.roll(ma, 5)
            ma_5ago[: min(n, 5)] = np.nan
            valid = ~np.isnan(ma) & ~np.isnan(ma_5ago) & (ma_5ago != 0.0)
            safe_prev = np.where(valid, ma_5ago, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, (ma - ma_5ago) / safe_prev, 0.0).tolist(),
            )

        if match := re.fullmatch(r"rsi_(\d+)", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            return cast("list[IndicatorValue]", self._rsi(p).tolist())

        if match := re.fullmatch(r"rsi_(\d+)_zscore", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            rsi_arr = self._rsi(p)
            rsi_s = pd.Series(rsi_arr)
            mean50 = rsi_s.rolling(50).mean().to_numpy(dtype=float)
            std50 = rsi_s.rolling(50).std(ddof=0).to_numpy(dtype=float)
            valid = (np.arange(n) >= 50) & ~np.isnan(mean50) & (std50 >= 1e-6)
            safe_std = np.where(valid, std50, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, (rsi_arr - mean50) / safe_std, 0.0).tolist(),
            )

        if name == "atr":
            atr_p: int | None = 14
        elif atr_match := re.fullmatch(r"atr_(\d+)", name):
            atr_p = _parse_positive_period(atr_match.group(1))
        else:
            atr_p = -1
        if atr_p != -1:
            if atr_p is None:
                return None
            tr = self._true_range()
            out = tr.copy()
            if n > atr_p:
                tr_ma = pd.Series(tr).rolling(atr_p).mean().to_numpy(dtype=float)
                out[atr_p:] = tr_ma[atr_p:]
            return cast("list[IndicatorValue]", out.tolist())

        if name == "macd_histogram":
            fast_h: int | None = 12
            slow_h: int | None = 26
            signal_h: int | None = 9
        elif macd_h_match := re.fullmatch(r"macd_histogram_fast(\d+)_slow(\d+)_signal(\d+)", name):
            fast_h = _parse_positive_period(macd_h_match.group(1))
            slow_h = _parse_positive_period(macd_h_match.group(2))
            signal_h = _parse_positive_period(macd_h_match.group(3))
        else:
            fast_h = slow_h = signal_h = -1
        if fast_h != -1:
            if fast_h is None or slow_h is None or signal_h is None:
                return None
            comps = self._macd(fast_h, slow_h, signal_h)
            if comps is None:
                return [0.0] * n
            out = comps[2].copy()
            out[: min(n, slow_h + signal_h - 2)] = 0.0
            return cast("list[IndicatorValue]", out.tolist())

        if name == "macd_cross_bullish":
            fast_c: int | None = 12
            slow_c: int | None = 26
            signal_c: int | None = 9
        elif macd_c_match := re.fullmatch(
            r"macd_cross_bullish_fast(\d+)_slow(\d+)_signal(\d+)", name
        ):
            fast_c = _parse_positive_period(macd_c_match.group(1))
            slow_c = _parse_positive_period(macd_c_match.group(2))
            signal_c = _parse_positive_period(macd_c_match.group(3))
        else:
            fast_c = slow_c = signal_c = -1
        if fast_c != -1:
            if fast_c is None or slow_c is None or signal_c is None:
                return None
            comps = self._macd(fast_c, slow_c, signal_c)
            if comps is None:
                return [False] * n
            macd_line, signal_line, _ = comps
            prev_m = np.roll(macd_line, 1)
            prev_s = np.roll(signal_line, 1)
            cross = (macd_line > signal_line) & (prev_m <= prev_s)
            cross[: min(n, slow_c + signal_c - 1)] = False
            return cast("list[IndicatorValue]", cross.tolist())

        if name in {
            "bollinger_band_width",
            "price_above_bollinger_upper",
            "price_below_bollinger_lower",
        }:
            period, std_mult = 20, 2.0
            mode = name
        elif match := re.fullmatch(
            r"(bollinger_band_width|price_above_bollinger_upper|price_below_bollinger_lower)_(\d+)d(?:_std([0-9]+(?:p[0-9]+)?))?",
            name,
        ):
            mode = match.group(1)
            period_opt = _parse_positive_period(match.group(2))
            std_opt = _parse_multiplier_token(match.group(3)) if match.group(3) else 2.0
            if period_opt is None or std_opt is None:
                return None
            period, std_mult = period_opt, std_opt
        else:
            mode = ""
            period, std_mult = 0, 0.0

        if mode:
            bands = self._bollinger(period, std_mult)
            if bands is None:
                default: IndicatorValue = 0.1 if mode == "bollinger_band_width" else False
                return [default] * n
            mid, upper, lower = bands
            valid = ~np.isnan(mid) & ~np.isnan(upper)
            if mode == "bollinger_band_width":
                valid_w = valid & (mid != 0.0)
                safe_mid = np.where(valid_w, mid, 1.0)
                return cast(
                    "list[IndicatorValue]",
                    np.where(valid_w, (upper - lower) / safe_mid, 0.1).tolist(),
                )
            if mode == "price_above_bollinger_upper":
                return cast(
                    "list[IndicatorValue]", np.where(valid, self.close > upper, False).tolist()
                )
            return cast("list[IndicatorValue]", np.where(valid, self.close < lower, False).tolist())

        if match := re.fullmatch(r"breakout_high_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            prior_high = self.high_s.shift(1).rolling(p).max().to_numpy(dtype=float)
            return cast(
                "list[IndicatorValue]",
                np.where(~np.isnan(prior_high), self.close > prior_high, False).tolist(),
            )

        if name == "price_position_52w":
            lo = self.close_s.rolling(250, min_periods=20).min().to_numpy(dtype=float)
            hi = self.close_s.rolling(250, min_periods=20).max().to_numpy(dtype=float)
            valid = ~np.isnan(lo) & (hi != lo)
            safe_span = np.where(valid, hi - lo, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, (self.close - lo) / safe_span, 0.5).tolist(),
            )

        if match := re.fullmatch(r"price_position_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            if p < 2:
                return [0.5] * n
            lo = self.close_s.rolling(p, min_periods=2).min().to_numpy(dtype=float)
            hi = self.close_s.rolling(p, min_periods=2).max().to_numpy(dtype=float)
            valid = ~np.isnan(lo) & (hi != lo)
            safe_span = np.where(valid, hi - lo, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, (self.close - lo) / safe_span, 0.5).tolist(),
            )

        if match := re.fullmatch(r"volume_ratio_1d_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            baseline = self.vol_s.shift(1).rolling(p).mean().to_numpy(dtype=float)
            valid = ~np.isnan(baseline) & (baseline > 0.0)
            safe_base = np.where(valid, baseline, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, self.vol / safe_base, 1.0).tolist(),
            )

        if match := re.fullmatch(r"momentum_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            prev = self.close_s.shift(p).to_numpy(dtype=float)
            valid = ~np.isnan(prev) & (prev != 0.0)
            safe_prev = np.where(valid, prev, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, (self.close - prev) / safe_prev, 0.0).tolist(),
            )

        if match := re.fullmatch(r"avg_volume_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            vma = self.vol_s.rolling(p).mean().to_numpy(dtype=float)
            return cast(
                "list[IndicatorValue]",
                np.where(np.isnan(vma), self.vol, vma).tolist(),
            )

        if name == "consecutive_up_days":
            res_days: list[IndicatorValue] = [0] * n
            count = 0
            for i in range(1, n):
                count = count + 1 if self.close[i] > self.close[i - 1] else 0
                res_days[i] = count
            return res_days

        if name == "consecutive_down_days":
            res_days = [0] * n
            count = 0
            for i in range(1, n):
                count = count + 1 if self.close[i] < self.close[i - 1] else 0
                res_days[i] = count
            return res_days

        if match := re.fullmatch(r"days_since_high_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            out_idx = np.zeros(n, dtype=int)
            if n >= p:
                wins = np.lib.stride_tricks.sliding_window_view(self.high, p)
                out_idx[p - 1 :] = (p - 1) - np.argmax(wins, axis=1)
            return cast("list[IndicatorValue]", out_idx.tolist())

        if match := re.fullmatch(r"days_since_low_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            out_idx = np.zeros(n, dtype=int)
            if n >= p:
                wins = np.lib.stride_tricks.sliding_window_view(self.low, p)
                out_idx[p - 1 :] = (p - 1) - np.argmin(wins, axis=1)
            return cast("list[IndicatorValue]", out_idx.tolist())

        if match := re.fullmatch(r"volatility_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            if p < 6 or n <= p:
                return [0.0] * n
            rets = self.close_s.pct_change()
            rstd = rets.rolling(p - 1, min_periods=5).std(ddof=1).to_numpy(dtype=float)
            out_vol = np.zeros(n, dtype=float)
            tail = rstd[p:]
            out_vol[p:] = np.where(~np.isnan(tail), tail * (250.0**0.5), 0.0)
            return cast("list[IndicatorValue]", out_vol.tolist())

        if match := re.fullmatch(r"relative_strength_(\d+)d", name):
            p = _parse_positive_period(match.group(1))
            if p is None:
                return None
            prev = self.close_s.shift(p).to_numpy(dtype=float)
            valid_s = ~np.isnan(prev) & (prev != 0.0)
            safe_prev = np.where(valid_s, prev, 1.0)
            stock_ret = np.where(valid_s, (self.close / safe_prev) - 1.0, 0.0)
            bm_ret = np.zeros(n, dtype=float)
            if (
                self.ctx
                and isinstance(self.ctx.benchmark_df, pd.DataFrame)
                and not self.ctx.benchmark_df.empty
            ):
                bm = self.ctx.benchmark_df
                bm_close = bm["close"].to_numpy(dtype=float)
                sym_dates = _extract_dates(self.df)
                bm_dates = _extract_dates(bm)
                if sym_dates is not None and bm_dates is not None:
                    bm_indices = [bisect_right(bm_dates, d) - 1 for d in sym_dates]
                else:
                    bm_indices = [i if i < len(bm) else -1 for i in range(n)]
                for i in range(p, n):
                    bi = bm_indices[i]
                    if bi >= p and bm_close[bi - p] != 0.0:
                        bm_ret[i] = (bm_close[bi] / bm_close[bi - p]) - 1.0
            return cast(
                "list[IndicatorValue]",
                np.where(valid_s, stock_ret - bm_ret, 0.0).tolist(),
            )

        if name == "has_stop_signal":
            body = np.abs(self.close - self.open)
            lower_shadow = np.minimum(self.open, self.close) - self.low
            hammer = (body > 0.0) & (lower_shadow >= 2.0 * body)
            prev_close = np.roll(self.close, 1)
            prev_open = np.roll(self.open, 1)
            engulf = (
                (prev_close < prev_open)
                & (self.close > self.open)
                & (self.close > prev_open)
                & (self.open < prev_close)
            )
            sig = hammer | engulf
            sig[0] = False
            return cast("list[IndicatorValue]", sig.tolist())

        if name == "volume_shrink_then_rise":
            if n < 4:
                return [False] * n
            res_vsr = np.zeros(n, dtype=bool)
            res_vsr[3:] = (
                (self.vol[:-3] > self.vol[1:-2])
                & (self.vol[1:-2] > self.vol[2:-1])
                & (self.vol[3:] > self.vol[2:-1])
            )
            return cast("list[IndicatorValue]", res_vsr.tolist())

        if name == "gap_up_pct":
            prev_gap = np.roll(self.close, 1)
            prev_gap[0] = 0.0
            valid = (np.arange(n) >= 1) & (prev_gap != 0.0)
            safe_prev = np.where(valid, prev_gap, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, (self.open - prev_gap) / safe_prev, 0.0).tolist(),
            )

        if name == "body_to_range_ratio":
            rng = self.high - self.low
            valid = rng != 0.0
            safe_rng = np.where(valid, rng, 1.0)
            return cast(
                "list[IndicatorValue]",
                np.where(valid, np.abs(self.close - self.open) / safe_rng, 0.0).tolist(),
            )

        if name == "st_flag":
            val = bool(self.ctx.stock_info.is_st) if self.ctx and self.ctx.stock_info else False
            return [val] * n

        if name == "sector_heat_rank":
            val_int = (
                int(self.ctx.sector_info.heat_rank)
                if self.ctx and self.ctx.sector_info and self.ctx.sector_info.heat_rank is not None
                else 1
            )
            return [val_int] * n

        if name == "sector_heat_rising_days":
            val_int = (
                int(self.ctx.sector_info.rising_days)
                if self.ctx
                and self.ctx.sector_info
                and self.ctx.sector_info.rising_days is not None
                else 99
            )
            return [val_int] * n

        if name == "intra_sector_strength_rank_pct":
            val_flt = (
                float(self.ctx.intra_sector_strength_rank_pct)
                if self.ctx and self.ctx.intra_sector_strength_rank_pct is not None
                else 0.0
            )
            return [val_flt] * n

        if name == "sector_fund_flow_positive":
            val_bool = (
                bool(self.ctx.sector_info.net_inflow > 0)
                if self.ctx and self.ctx.sector_info and self.ctx.sector_info.net_inflow is not None
                else True
            )
            return [val_bool] * n

        if name == "sector_risk_flag":
            val_bool = (
                bool(self.ctx.sector_info.risk_flag) if self.ctx and self.ctx.sector_info else False
            )
            return [val_bool] * n

        if name == "sector_net_inflow_days":
            val_int = (
                int(self.ctx.sector_info.net_inflow_days)
                if self.ctx
                and self.ctx.sector_info
                and self.ctx.sector_info.net_inflow_days is not None
                else 99
            )
            return [val_int] * n

        if name in {
            "negative_news_score",
            "news_sentiment_score",
            "days_since_event",
            "price_above_pre_event",
            "already_overreacted",
        }:
            override = getattr(self.ctx, name, None) if self.ctx is not None else None
            if override is not None:
                if name == "days_since_event":
                    return [int(override)] * n
                if name in {"price_above_pre_event", "already_overreacted"}:
                    return [bool(override)] * n
                return [float(override)] * n
            src = self._event_source(name)
            col_vals = src[name].tolist()
            if name == "days_since_event":
                return [_NO_EVENT_DAYS if pd.isna(v) else int(v) for v in col_vals]
            if name in {"price_above_pre_event", "already_overreacted"}:
                return [False if pd.isna(v) else bool(v) for v in col_vals]
            return [0.0 if pd.isna(v) else float(v) for v in col_vals]

        return None


def _parse_positive_period(value: str) -> int | None:
    """Parse a positive integer period from a regex group."""
    try:
        period = int(value)
    except ValueError:
        return None
    return period if period > 0 else None


def _rolling_close_ma(df: pd.DataFrame, idx: int, period: int) -> float | None:
    """Return the close-based moving average ending at ``idx``."""
    if period <= 0 or idx < 0 or idx + 1 < period:
        return None
    window = df["close"].iloc[idx - period + 1 : idx + 1]
    if len(window) < period:
        return None
    return float(window.mean())


def _compute_rsi(df: pd.DataFrame, idx: int, period: int) -> float:
    """Compute RSI using Wilder's smoothing (exponential moving average).

    This matches the standard RSI definition used by TradingView, Wind,
    and other charting platforms.  Wilder's smoothing uses
    ``alpha = 1 / period`` which is equivalent to ``com = period - 1``.
    """
    if period <= 0 or idx < period:
        return 50.0
    changes = df["close"].diff().iloc[1 : idx + 1]  # all available diffs
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    # Wilder's smoothing: EMA with alpha = 1/period (com = period - 1)
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _compute_volume_ratio(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """Today's volume divided by the prior ``lookback`` bars' average volume."""
    if lookback <= 0 or idx < lookback:
        return 1.0
    today_vol = df["volume"].iloc[idx]
    baseline = df["volume"].iloc[idx - lookback : idx].mean()
    if baseline <= 0:
        return 1.0
    return float(today_vol / baseline)


def _compute_momentum(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """Rate of change over ``lookback`` bars."""
    if lookback <= 0 or idx < lookback:
        return 0.0
    prev_price = df["close"].iloc[idx - lookback]
    if prev_price == 0:
        return 0.0
    return float((df["close"].iloc[idx] - prev_price) / prev_price)


def _compute_avg_volume(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """Average volume over the last ``lookback`` bars, inclusive."""
    if lookback <= 0:
        return float(df["volume"].iloc[idx])
    if idx + 1 < lookback:
        return float(df["volume"].iloc[idx])
    return float(df["volume"].iloc[idx - lookback + 1 : idx + 1].mean())


def _compute_days_since_extreme(
    series: pd.Series,
    idx: int,
    lookback: int,
    *,
    use_high: bool,
) -> int:
    """Days since the highest/highest or lowest/low point in the rolling window."""
    if lookback <= 0 or idx + 1 < lookback:
        return 0
    window = series.iloc[idx - lookback + 1 : idx + 1].to_numpy(dtype=float)
    offset = int(np.argmax(window) if use_high else np.argmin(window))
    return int(len(window) - 1 - offset)


def _compute_volatility(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """Annualized volatility over the last ``lookback`` bars."""
    if lookback <= 0 or idx < lookback:
        return 0.0
    returns = df["close"].iloc[idx - lookback + 1 : idx + 1].pct_change().dropna()
    if len(returns) < 5:
        return 0.0
    return float(returns.std() * (250**0.5))


def _compute_price_position(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """Price position in the rolling close range. 0 = low, 1 = high."""
    if lookback <= 0:
        return 0.5
    lookback = min(lookback, idx + 1)
    if lookback < 2:
        return 0.5
    window = df["close"].iloc[idx - lookback + 1 : idx + 1]
    lo, hi = window.min(), window.max()
    if hi == lo:
        return 0.5
    return float((df["close"].iloc[idx] - lo) / (hi - lo))


def _compute_breakout_high(df: pd.DataFrame, idx: int, lookback: int) -> bool:
    """Close breaks above the prior rolling high, excluding the current bar."""
    if lookback <= 0 or idx < lookback:
        return False
    prior_high = df["high"].iloc[idx - lookback : idx].max()
    return bool(float(df["close"].iloc[idx]) > float(prior_high))


def _extract_dates(df: pd.DataFrame) -> list[date] | None:
    """Extract normalized ``date`` objects from ``df['date']`` when available."""
    if "date" not in df.columns or df.empty:
        return None
    raw = df["date"].tolist()
    if all(type(v) is date for v in raw):
        return cast("list[date]", raw)
    return cast("list[date]", pd.to_datetime(df["date"]).dt.date.tolist())


def _align_benchmark_idx(df: pd.DataFrame, idx: int, bm: pd.DataFrame) -> int | None:
    """Resolve the causal benchmark bar index aligned to ``df`` bar ``idx``."""
    if bm.empty or idx < 0 or idx >= len(df):
        return None
    if "date" in df.columns and "date" in bm.columns:
        sym_dates = _extract_dates(df)
        bm_dates = _extract_dates(bm)
        if not sym_dates or not bm_dates:
            return None
        bm_idx = bisect_right(bm_dates, sym_dates[idx]) - 1
        return bm_idx if bm_idx >= 0 else None
    return idx if idx < len(bm) else None


def _compute_relative_strength(
    df: pd.DataFrame,
    idx: int,
    lookback: int,
    ctx: IndicatorContext | None = None,
) -> float:
    """Stock return minus benchmark return over ``lookback`` bars."""
    if lookback <= 0 or idx < lookback:
        return 0.0
    base_price = df["close"].iloc[idx - lookback]
    if base_price == 0:
        return 0.0
    stock_ret = (df["close"].iloc[idx] / base_price) - 1
    benchmark_ret = 0.0
    if ctx and isinstance(ctx.benchmark_df, pd.DataFrame):
        bm = ctx.benchmark_df
        bm_idx = _align_benchmark_idx(df, idx, bm)
        if bm_idx is not None and bm_idx >= lookback:
            bm_base = bm["close"].iloc[bm_idx - lookback]
            if bm_base != 0:
                benchmark_ret = (bm["close"].iloc[bm_idx] / bm_base) - 1
    return float(stock_ret - benchmark_ret)


def _compute_atr(df: pd.DataFrame, idx: int, period: int) -> float:
    """Average True Range for an arbitrary lookback period."""
    if period <= 0:
        return float(df["high"].iloc[idx] - df["low"].iloc[idx])
    if idx < period:
        if idx >= 1:
            hi = df["high"].iloc[idx]
            lo = df["low"].iloc[idx]
            prev_close = df["close"].iloc[idx - 1]
            return float(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
        return float(df["high"].iloc[idx] - df["low"].iloc[idx])

    tr_values = []
    for i in range(idx - period + 1, idx + 1):
        hi = df["high"].iloc[i]
        lo = df["low"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        tr_values.append(tr)
    return float(sum(tr_values) / len(tr_values))


def _compute_bollinger_bands(
    df: pd.DataFrame,
    idx: int,
    period: int,
    std_multiplier: float = 2.0,
) -> tuple[float, float, float] | None:
    """Return the middle/upper/lower Bollinger bands for a rolling window."""
    if period <= 0 or std_multiplier <= 0 or idx + 1 < period:
        return None
    window = df["close"].iloc[idx - period + 1 : idx + 1]
    if len(window) < period:
        return None
    middle = float(window.mean())
    std = float(window.std())
    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std
    return (middle, upper, lower)


def _parse_multiplier_token(value: str) -> float | None:
    """Parse a positive multiplier token like ``2`` or ``1p5``."""
    try:
        multiplier = float(value.replace("p", "."))
    except ValueError:
        return None
    return multiplier if multiplier > 0 else None


def _compute_macd_components(
    df: pd.DataFrame,
    idx: int,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[float, float, float] | None:
    """Compute MACD line, signal line, and histogram for arbitrary periods."""
    if fast <= 0 or slow <= 0 or signal <= 0 or fast >= slow:
        return None
    warmup = slow + signal - 1
    if idx + 1 < warmup:
        return None
    closes = df["close"].iloc[: idx + 1]
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return (
        float(macd_line.iloc[-1]),
        float(signal_line.iloc[-1]),
        float(histogram.iloc[-1]),
    )


def _build_templated_indicator(name: str) -> IndicatorFn | None:
    """Build indicator functions for parameterized moving-average patterns."""
    if match := re.fullmatch(r"ma(\d+)_above_ma(\d+)", name):
        fast = _parse_positive_period(match.group(1))
        slow = _parse_positive_period(match.group(2))
        if fast is None or slow is None:
            return None
        fast_period = fast
        slow_period = slow

        def ma_above(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> bool:
            del ctx
            fast_ma = _rolling_close_ma(df, idx, fast_period)
            slow_ma = _rolling_close_ma(df, idx, slow_period)
            if fast_ma is None or slow_ma is None:
                return False
            return bool(fast_ma > slow_ma)

        return ma_above

    if match := re.fullmatch(r"ma(\d+)_ge_ma(\d+)_or_crossing", name):
        fast = _parse_positive_period(match.group(1))
        slow = _parse_positive_period(match.group(2))
        if fast is None or slow is None:
            return None
        fast_period = fast
        slow_period = slow

        def ma_crossing(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> bool:
            del ctx
            fast_ma = _rolling_close_ma(df, idx, fast_period)
            slow_ma = _rolling_close_ma(df, idx, slow_period)
            prev_fast_ma = _rolling_close_ma(df, idx - 1, fast_period)
            prev_slow_ma = _rolling_close_ma(df, idx - 1, slow_period)
            if fast_ma is None or slow_ma is None or prev_fast_ma is None or prev_slow_ma is None:
                return False
            if fast_ma >= slow_ma:
                return True
            return bool(prev_fast_ma < prev_slow_ma and fast_ma >= slow_ma * 0.998)

        return ma_crossing

    if match := re.fullmatch(r"close_to_ma(\d+)_pct", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        period_days = period

        def close_to_ma(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            ma = _rolling_close_ma(df, idx, period_days)
            if ma is None or ma == 0.0:
                return 1.0
            close = float(df["close"].iloc[idx])
            return float(abs(close - ma) / ma)

        return close_to_ma

    if match := re.fullmatch(r"close_above_ma(\d+)", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        period_days = period

        def close_above_ma(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> bool:
            del ctx
            ma = _rolling_close_ma(df, idx, period_days)
            if ma is None:
                return False
            return bool(float(df["close"].iloc[idx]) > ma)

        return close_above_ma

    if match := re.fullmatch(r"close_below_ma(\d+)", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        period_days = period

        def close_below_ma(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> bool:
            del ctx
            ma = _rolling_close_ma(df, idx, period_days)
            if ma is None:
                return False
            return bool(float(df["close"].iloc[idx]) < ma)

        return close_below_ma

    if match := re.fullmatch(r"deviation_from_ma(\d+)_pct", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        period_days = period

        def deviation_from_ma(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            ma = _rolling_close_ma(df, idx, period_days)
            if ma is None or ma == 0.0:
                return 0.0
            close = float(df["close"].iloc[idx])
            return float((close - ma) / ma)

        return deviation_from_ma

    if match := re.fullmatch(r"ma(\d+)_slope", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        period_days = period

        def ma_slope(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            ma_now = _rolling_close_ma(df, idx, period_days)
            ma_5ago = _rolling_close_ma(df, idx - 5, period_days)
            if ma_now is None or ma_5ago is None or ma_5ago == 0.0:
                return 0.0
            return float((ma_now - ma_5ago) / ma_5ago)

        return ma_slope

    if match := re.fullmatch(r"rsi_(\d+)_zscore", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        period_days = period

        def rsi_zscore(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            lookback = 50
            if idx < lookback:
                return 0.0
            rsi_values = [
                _compute_rsi(df, i, period_days) for i in range(idx - lookback + 1, idx + 1)
            ]
            mean_rsi = sum(rsi_values) / len(rsi_values)
            std_rsi = (sum((v - mean_rsi) ** 2 for v in rsi_values) / len(rsi_values)) ** 0.5
            if std_rsi < 1e-6:
                return 0.0
            return float((rsi_values[-1] - mean_rsi) / std_rsi)

        return rsi_zscore

    if match := re.fullmatch(r"macd_histogram_fast(\d+)_slow(\d+)_signal(\d+)", name):
        fast = _parse_positive_period(match.group(1))
        slow = _parse_positive_period(match.group(2))
        signal = _parse_positive_period(match.group(3))
        if fast is None or slow is None or signal is None or fast >= slow:
            return None
        fast_period = fast
        slow_period = slow
        signal_period = signal

        def macd_histogram_templated(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            components = _compute_macd_components(df, idx, fast_period, slow_period, signal_period)
            if components is None:
                return 0.0
            return components[2]

        return macd_histogram_templated

    if match := re.fullmatch(r"macd_cross_bullish_fast(\d+)_slow(\d+)_signal(\d+)", name):
        fast = _parse_positive_period(match.group(1))
        slow = _parse_positive_period(match.group(2))
        signal = _parse_positive_period(match.group(3))
        if fast is None or slow is None or signal is None or fast >= slow:
            return None
        fast_period = fast
        slow_period = slow
        signal_period = signal

        def macd_cross_bullish_templated(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> bool:
            del ctx
            current = _compute_macd_components(df, idx, fast_period, slow_period, signal_period)
            previous = _compute_macd_components(
                df, idx - 1, fast_period, slow_period, signal_period
            )
            if current is None or previous is None:
                return False
            return bool(current[0] > current[1] and previous[0] <= previous[1])

        return macd_cross_bullish_templated

    if match := re.fullmatch(r"rsi_(\d+)", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        period_days = period

        def rsi(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            return _compute_rsi(df, idx, period_days)

        return rsi

    if match := re.fullmatch(r"atr_(\d+)", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def atr_window(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            return _compute_atr(df, idx, lookback)

        return atr_window

    if match := re.fullmatch(
        r"bollinger_band_width_(\d+)d(?:_std([0-9]+(?:p[0-9]+)?))?",
        name,
    ):
        period = _parse_positive_period(match.group(1))
        std_multiplier = _parse_multiplier_token(match.group(2) or "2")
        if period is None or std_multiplier is None:
            return None
        lookback = period
        band_width = std_multiplier

        def bollinger_width(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            bands = _compute_bollinger_bands(df, idx, lookback, band_width)
            if bands is None:
                return 0.1
            middle, _upper, _lower = bands
            if middle == 0.0:
                return 0.1
            return float((_upper - _lower) / middle)

        return bollinger_width

    if match := re.fullmatch(
        r"price_above_bollinger_upper_(\d+)d(?:_std([0-9]+(?:p[0-9]+)?))?",
        name,
    ):
        period = _parse_positive_period(match.group(1))
        std_multiplier = _parse_multiplier_token(match.group(2) or "2")
        if period is None or std_multiplier is None:
            return None
        lookback = period
        band_width = std_multiplier

        def above_bollinger_upper(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> bool:
            del ctx
            bands = _compute_bollinger_bands(df, idx, lookback, band_width)
            if bands is None:
                return False
            _middle, upper, _lower = bands
            return bool(df["close"].iloc[idx] > upper)

        return above_bollinger_upper

    if match := re.fullmatch(
        r"price_below_bollinger_lower_(\d+)d(?:_std([0-9]+(?:p[0-9]+)?))?",
        name,
    ):
        period = _parse_positive_period(match.group(1))
        std_multiplier = _parse_multiplier_token(match.group(2) or "2")
        if period is None or std_multiplier is None:
            return None
        lookback = period
        band_width = std_multiplier

        def below_bollinger_lower(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> bool:
            del ctx
            bands = _compute_bollinger_bands(df, idx, lookback, band_width)
            if bands is None:
                return False
            _middle, _upper, lower = bands
            return bool(df["close"].iloc[idx] < lower)

        return below_bollinger_lower

    if match := re.fullmatch(r"breakout_high_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def breakout_high(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> bool:
            del ctx
            return _compute_breakout_high(df, idx, lookback)

        return breakout_high

    if match := re.fullmatch(r"price_position_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def price_position(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            return _compute_price_position(df, idx, lookback)

        return price_position

    if match := re.fullmatch(r"volume_ratio_1d_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def volume_ratio(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            return _compute_volume_ratio(df, idx, lookback)

        return volume_ratio

    if match := re.fullmatch(r"momentum_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def momentum(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            return _compute_momentum(df, idx, lookback)

        return momentum

    if match := re.fullmatch(r"avg_volume_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def avg_volume(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            return _compute_avg_volume(df, idx, lookback)

        return avg_volume

    if match := re.fullmatch(r"days_since_high_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def days_since_high(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> int:
            del ctx
            return _compute_days_since_extreme(df["high"], idx, lookback, use_high=True)

        return days_since_high

    if match := re.fullmatch(r"days_since_low_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def days_since_low(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> int:
            del ctx
            return _compute_days_since_extreme(df["low"], idx, lookback, use_high=False)

        return days_since_low

    if match := re.fullmatch(r"volatility_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def volatility(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            del ctx
            return _compute_volatility(df, idx, lookback)

        return volatility

    if match := re.fullmatch(r"relative_strength_(\d+)d", name):
        period = _parse_positive_period(match.group(1))
        if period is None:
            return None
        lookback = period

        def relative_strength(
            df: pd.DataFrame,
            idx: int,
            ctx: IndicatorContext | None = None,
        ) -> float:
            return _compute_relative_strength(df, idx, lookback, ctx)

        return relative_strength

    return None


def enrich_with_event_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Add price/volume-derived event proxy columns when explicit news data is absent.

    The proxies deliberately avoid future leakage:
      - event detection uses only the current/past bar
      - derived sentiment decays over time from the latest detected event
      - ``pre_event_close`` is forward-filled from the latest event anchor
    """
    if df.empty or _EVENT_PROXY_COLUMNS.issubset(df.columns):
        return df

    enriched = df.copy()
    prev_close = enriched["close"].shift(1).fillna(enriched["close"])
    volume_baseline = enriched["volume"].rolling(5, min_periods=1).mean().shift(1)
    volume_ratio = (enriched["volume"] / volume_baseline.replace(0, np.nan)).fillna(1.0)
    gap_pct = ((enriched["open"] - prev_close) / prev_close.replace(0, np.nan)).fillna(0.0)
    day_return = ((enriched["close"] - prev_close) / prev_close.replace(0, np.nan)).fillna(0.0)
    intraday_return = (
        (enriched["close"] - enriched["open"]) / enriched["open"].replace(0, np.nan)
    ).fillna(0.0)
    event_flag = (
        ((gap_pct.abs() >= 0.025) | (day_return.abs() >= 0.045)) & (volume_ratio >= 1.5)
    ).astype(bool)

    negative_scores: list[float] = []
    sentiment_scores: list[float] = []
    days_since_event: list[int] = []
    pre_event_closes: list[float] = []
    price_above_pre_event: list[bool] = []
    already_overreacted: list[bool] = []

    last_event_idx: int | None = None
    last_pre_event_close = float(enriched["close"].iloc[0])
    last_sentiment = 0.5
    last_negative_score = 0.0

    for idx in range(len(enriched)):
        if bool(event_flag.iloc[idx]):
            last_event_idx = idx
            anchor = float(prev_close.iloc[idx])
            last_pre_event_close = anchor if anchor > 0 else float(enriched["close"].iloc[idx])
            raw_sentiment = (
                0.5 + float(gap_pct.iloc[idx]) * 4.0 + float(intraday_return.iloc[idx]) * 2.0
            )
            last_sentiment = min(1.0, max(0.0, raw_sentiment))
            last_negative_score = min(1.0, max(0.0, (0.5 - last_sentiment) * 2.0))

        if last_event_idx is None:
            age = _NO_EVENT_DAYS
            sentiment = 1.0  # No event → fully positive ("don't block")
            negative = 0.0
            pre_event_close = float(enriched["close"].iloc[idx])
        else:
            age = idx - last_event_idx
            decay = max(0.0, 1.0 - age / 5.0)
            sentiment = 0.5 + (last_sentiment - 0.5) * decay
            negative = min(
                1.0,
                last_negative_score * decay + max(0.0, -float(day_return.iloc[idx])) * 2.0,
            )
            pre_event_close = last_pre_event_close

        close = float(enriched["close"].iloc[idx])
        move_from_pre_event = (
            ((close - pre_event_close) / pre_event_close) if pre_event_close else 0.0
        )

        negative_scores.append(round(float(negative), 6))
        sentiment_scores.append(round(float(sentiment), 6))
        days_since_event.append(age)
        pre_event_closes.append(round(float(pre_event_close), 6))
        price_above_pre_event.append(bool(close >= pre_event_close))
        already_overreacted.append(bool(age <= 5 and move_from_pre_event >= 0.12))

    enriched["negative_news_score"] = negative_scores
    enriched["news_sentiment_score"] = sentiment_scores
    enriched["days_since_event"] = days_since_event
    enriched["pre_event_close"] = pre_event_closes
    enriched["price_above_pre_event"] = price_above_pre_event
    enriched["already_overreacted"] = already_overreacted
    return cast("pd.DataFrame", enriched)


def merge_event_context(
    df: pd.DataFrame,
    event_context: EventContextSeries | None = None,
) -> tuple[pd.DataFrame, str]:
    """Merge provider event context onto a proxy-enriched dataframe.

    Provider values overwrite proxy values on matching trading dates. Dates
    without provider coverage retain proxy-derived values so mixed coverage
    degrades gracefully instead of falling back to neutral defaults.

    Returns:
        Tuple of ``(merged_df, source_label)`` where ``source_label`` is one of:
        - ``"proxy"``
        - ``"<provider>"`` for full provider coverage
        - ``"<provider>+proxy"`` for partial / mixed coverage
    """
    enriched = enrich_with_event_proxies(df)
    if event_context is None or not event_context.records:
        return enriched, "proxy"

    provider_rows = [
        {
            "date": record.date,
            "negative_news_score": record.negative_news_score,
            "news_sentiment_score": record.news_sentiment_score,
            "days_since_event": record.days_since_event,
            "pre_event_close": record.pre_event_close,
            "price_above_pre_event": record.price_above_pre_event,
            "already_overreacted": record.already_overreacted,
        }
        for record in event_context.records
    ]
    provider_df = pd.DataFrame(provider_rows)
    if provider_df.empty:
        return enriched, "proxy"

    merged = enriched.copy()
    trading_dates = sorted(pd.to_datetime(merged["date"]).dt.date.unique())
    provider_df["date"] = pd.to_datetime(provider_df["date"]).dt.date
    provider_df["date"] = provider_df["date"].map(
        lambda provider_date: _next_trading_date(provider_date, trading_dates)
    )
    provider_df = provider_df.dropna(subset=["date"])
    if provider_df.empty:
        return merged, "proxy"
    provider_df = provider_df.drop_duplicates(subset=["date"], keep="last")

    provider_lookup = provider_df.set_index("date")
    provider_dates = set(provider_lookup.index)
    matched_dates = set(merged["date"]).intersection(provider_dates)

    if not matched_dates:
        return merged, "proxy"

    for idx, row_date in enumerate(merged["date"]):
        if row_date not in provider_lookup.index:
            continue
        provider_row = provider_lookup.loc[row_date]
        if isinstance(provider_row, pd.DataFrame):
            provider_row = provider_row.iloc[-1]
        for column in _EVENT_PROXY_COLUMNS:
            if column not in provider_row.index:
                continue
            value = provider_row[column]
            if pd.notna(value):
                merged.at[idx, column] = value

    provider_name = (event_context.source or "provider").strip() or "provider"
    if len(matched_dates) == len(merged):
        return merged, provider_name
    return merged, f"{provider_name}+proxy"


def _next_trading_date(
    provider_date: date,
    trading_dates: list[date],
) -> date | None:
    """Map sparse provider news dates to the next available trading date."""
    if not trading_dates:
        return None
    idx = bisect_left(trading_dates, provider_date)
    if idx >= len(trading_dates):
        return None
    return trading_dates[idx]


def _event_proxy_value(
    df: pd.DataFrame,
    idx: int,
    column: str,
) -> float | int | bool:
    """Read an event proxy value from the dataframe, enriching it on demand."""
    source = df if column in df.columns else enrich_with_event_proxies(df)
    value = source[column].iloc[idx]
    if pd.isna(value):
        if column == "days_since_event":
            return _NO_EVENT_DAYS
        if column in {"price_above_pre_event", "already_overreacted"}:
            return False
        return 0.0
    if column == "days_since_event":
        return int(value)
    if column in {"price_above_pre_event", "already_overreacted"}:
        return bool(value)
    return float(value)


# ═══════════════════════════════════════════════════════════════════════
#  L1 MVP Indicators — Only OHLCV needed
# ═══════════════════════════════════════════════════════════════════════


@IndicatorRegistry.register("ma5_above_ma10")
def ma5_above_ma10(df: pd.DataFrame, idx: int) -> bool:
    """MA5 is above MA10 (bullish alignment)."""
    if idx < 9:
        return False
    ma5 = df["close"].iloc[idx - 4 : idx + 1].mean()
    ma10 = df["close"].iloc[idx - 9 : idx + 1].mean()
    return bool(ma5 > ma10)


@IndicatorRegistry.register("close_to_ma10_pct")
def close_to_ma10_pct(df: pd.DataFrame, idx: int) -> float:
    """Absolute percentage distance from close to MA10."""
    if idx < 9:
        return 1.0
    ma10 = df["close"].iloc[idx - 9 : idx + 1].mean()
    if ma10 == 0:
        return 1.0
    return float(abs(df["close"].iloc[idx] - ma10) / ma10)


@IndicatorRegistry.register("close_above_ma20")
def close_above_ma20(df: pd.DataFrame, idx: int) -> bool:
    """Close price is above MA20."""
    if idx < 19:
        return False
    ma20 = df["close"].iloc[idx - 19 : idx + 1].mean()
    return bool(df["close"].iloc[idx] > ma20)


@IndicatorRegistry.register("close_below_ma10")
def close_below_ma10(df: pd.DataFrame, idx: int) -> bool:
    """Close price is below MA10."""
    if idx < 9:
        return False
    ma10 = df["close"].iloc[idx - 9 : idx + 1].mean()
    return bool(df["close"].iloc[idx] < ma10)


@IndicatorRegistry.register("volume_ratio_1d_5d")
def volume_ratio_1d_5d(df: pd.DataFrame, idx: int) -> float:
    """Today's volume / average volume of past 5 days."""
    if idx < 5:
        return 1.0
    today_vol = df["volume"].iloc[idx]
    avg_5d = df["volume"].iloc[idx - 5 : idx].mean()
    if avg_5d <= 0:
        return 1.0
    return float(today_vol / avg_5d)


@IndicatorRegistry.register("rsi_14")
def rsi_14(df: pd.DataFrame, idx: int) -> float:
    """14-period RSI (Wilder's smoothing)."""
    return _compute_rsi(df, idx, 14)


@IndicatorRegistry.register("deviation_from_ma20_pct")
def deviation_from_ma20_pct(df: pd.DataFrame, idx: int) -> float:
    """Percentage deviation of close from MA20 (positive = above, negative = below)."""
    if idx < 19:
        return 0.0
    ma20 = df["close"].iloc[idx - 19 : idx + 1].mean()
    if ma20 == 0:
        return 0.0
    return float((df["close"].iloc[idx] - ma20) / ma20)


@IndicatorRegistry.register("has_stop_signal")
def has_stop_signal(df: pd.DataFrame, idx: int) -> bool:
    """Detect a potential stop/reversal candle pattern.

    Simplified: lower shadow >= 2x body, or bullish engulfing after decline.
    """
    if idx < 1:
        return False
    row = df.iloc[idx]
    o, _hi, lo, c = row["open"], row["high"], row["low"], row["close"]
    body = abs(c - o)
    lower_shadow = min(o, c) - lo

    # Long lower shadow (hammer-like)
    if body > 0 and lower_shadow >= 2 * body:
        return True

    # Bullish engulfing after decline
    prev = df.iloc[idx - 1]
    return bool(
        prev["close"] < prev["open"]  # prev was bearish
        and c > o  # current is bullish
        and c > prev["open"]  # engulfs prev body
        and o < prev["close"]
    )


@IndicatorRegistry.register("volume_shrink_then_rise")
def volume_shrink_then_rise(df: pd.DataFrame, idx: int) -> bool:
    """Volume shrank over past 3 days then rises today.

    Pattern: vol[idx-3] > vol[idx-2] > vol[idx-1] < vol[idx]
    """
    if idx < 3:
        return False
    v = df["volume"].iloc
    shrinking = v[idx - 3] > v[idx - 2] > v[idx - 1]
    rising = v[idx] > v[idx - 1]
    return bool(shrinking and rising)


@IndicatorRegistry.register("ma5_ge_ma10_or_crossing")
def ma5_ge_ma10_or_crossing(df: pd.DataFrame, idx: int) -> bool:
    """MA5 >= MA10, or MA5 is crossing above MA10."""
    if idx < 10:
        return False
    ma5 = df["close"].iloc[idx - 4 : idx + 1].mean()
    ma10 = df["close"].iloc[idx - 9 : idx + 1].mean()
    if ma5 >= ma10:
        return True
    # Check crossing: previous bar MA5 < MA10, current MA5 close to MA10
    prev_ma5 = df["close"].iloc[idx - 5 : idx].mean()
    prev_ma10 = df["close"].iloc[idx - 10 : idx].mean()
    return bool(prev_ma5 < prev_ma10 and ma5 >= ma10 * 0.998)


@IndicatorRegistry.register("atr")
def atr(df: pd.DataFrame, idx: int) -> float:
    """14-period Average True Range."""
    return _compute_atr(df, idx, 14)


# ── Additional L1 Indicators ─────────────────────────────────────────


@IndicatorRegistry.register("macd_histogram")
def macd_histogram(df: pd.DataFrame, idx: int) -> float:
    """MACD histogram (MACD line - Signal line).

    Uses fast=12, slow=26, signal=9 EMAs.
    Positive = bullish momentum, negative = bearish.
    """
    components = _compute_macd_components(df, idx, 12, 26, 9)
    if components is None:
        return 0.0
    return components[2]


@IndicatorRegistry.register("macd_cross_bullish")
def macd_cross_bullish(df: pd.DataFrame, idx: int) -> bool:
    """MACD line just crossed above signal line (bullish crossover)."""
    current = _compute_macd_components(df, idx, 12, 26, 9)
    previous = _compute_macd_components(df, idx - 1, 12, 26, 9)
    if current is None or previous is None:
        return False
    return bool(current[0] > current[1] and previous[0] <= previous[1])


@IndicatorRegistry.register("bollinger_band_width")
def bollinger_band_width(df: pd.DataFrame, idx: int) -> float:
    """Bollinger Band width (20-period, 2 std). Narrower = potential breakout."""
    bands = _compute_bollinger_bands(df, idx, 20, 2.0)
    if bands is None:
        return 0.1
    middle, upper, lower = bands
    if middle == 0.0:
        return 0.1
    return float((upper - lower) / middle)


@IndicatorRegistry.register("price_above_bollinger_upper")
def price_above_bollinger_upper(df: pd.DataFrame, idx: int) -> bool:
    """Price is above upper Bollinger Band (overbought signal)."""
    bands = _compute_bollinger_bands(df, idx, 20, 2.0)
    if bands is None:
        return False
    _middle, upper, _lower = bands
    return bool(df["close"].iloc[idx] > upper)


@IndicatorRegistry.register("price_below_bollinger_lower")
def price_below_bollinger_lower(df: pd.DataFrame, idx: int) -> bool:
    """Price is below lower Bollinger Band (oversold signal)."""
    bands = _compute_bollinger_bands(df, idx, 20, 2.0)
    if bands is None:
        return False
    _middle, _upper, lower = bands
    return bool(df["close"].iloc[idx] < lower)


@IndicatorRegistry.register("ma20_slope")
def ma20_slope(df: pd.DataFrame, idx: int) -> float:
    """Slope of MA20 over last 5 bars (positive = uptrend)."""
    if idx < 24:
        return 0.0
    ma_now = df["close"].iloc[idx - 19 : idx + 1].mean()
    ma_5ago = df["close"].iloc[idx - 24 : idx - 4].mean()
    if ma_5ago == 0:
        return 0.0
    return float((ma_now - ma_5ago) / ma_5ago)


@IndicatorRegistry.register("momentum_10d")
def momentum_10d(df: pd.DataFrame, idx: int) -> float:
    """10-day price momentum (rate of change)."""
    if idx < 10:
        return 0.0
    prev_price = df["close"].iloc[idx - 10]
    if prev_price == 0:
        return 0.0
    return float((df["close"].iloc[idx] - prev_price) / prev_price)


@IndicatorRegistry.register("avg_volume_20d")
def avg_volume_20d(df: pd.DataFrame, idx: int) -> float:
    """20-day average volume. Useful for volume-based filters."""
    if idx < 19:
        return float(df["volume"].iloc[idx])
    return float(df["volume"].iloc[idx - 19 : idx + 1].mean())


@IndicatorRegistry.register("consecutive_up_days")
def consecutive_up_days(df: pd.DataFrame, idx: int) -> int:
    """Number of consecutive up-close days ending at idx."""
    count = 0
    for i in range(idx, 0, -1):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            count += 1
        else:
            break
    return count


@IndicatorRegistry.register("consecutive_down_days")
def consecutive_down_days(df: pd.DataFrame, idx: int) -> int:
    """Number of consecutive down-close days ending at idx."""
    count = 0
    for i in range(idx, 0, -1):
        if df["close"].iloc[i] < df["close"].iloc[i - 1]:
            count += 1
        else:
            break
    return count


@IndicatorRegistry.register("days_since_high_20d")
def days_since_high_20d(df: pd.DataFrame, idx: int) -> int:
    """Days since the 20-day high. 0 = today is the high."""
    if idx < 19:
        return 0
    window = df["high"].iloc[idx - 19 : idx + 1].to_numpy(dtype=float)
    high_offset = int(np.argmax(window))
    return int(len(window) - 1 - high_offset)


@IndicatorRegistry.register("days_since_low_20d")
def days_since_low_20d(df: pd.DataFrame, idx: int) -> int:
    """Days since the 20-day low. 0 = today is the low."""
    if idx < 19:
        return 0
    window = df["low"].iloc[idx - 19 : idx + 1].to_numpy(dtype=float)
    low_offset = int(np.argmin(window))
    return int(len(window) - 1 - low_offset)


@IndicatorRegistry.register("rsi_14_zscore")
def rsi_14_zscore(df: pd.DataFrame, idx: int) -> float:
    """Z-score of RSI-14 relative to its 50-bar history.

    Positive = RSI unusually high, negative = unusually low.
    Helps LLM pick threshold-independent conditions.
    """
    lookback = 50
    if idx < lookback:
        return 0.0
    rsi_values = []
    for i in range(idx - lookback + 1, idx + 1):
        rsi_values.append(rsi_14(df, i))
    mean_rsi = sum(rsi_values) / len(rsi_values)
    std_rsi = (sum((v - mean_rsi) ** 2 for v in rsi_values) / len(rsi_values)) ** 0.5
    if std_rsi < 1e-6:
        return 0.0
    return float((rsi_values[-1] - mean_rsi) / std_rsi)


@IndicatorRegistry.register("volume_ratio_1d_20d")
def volume_ratio_1d_20d(df: pd.DataFrame, idx: int) -> float:
    """Today's volume / 20-day average volume. More stable than 5d ratio."""
    if idx < 20:
        return 1.0
    today_vol = df["volume"].iloc[idx]
    avg_20d = df["volume"].iloc[idx - 20 : idx].mean()
    if avg_20d <= 0:
        return 1.0
    return float(today_vol / avg_20d)


@IndicatorRegistry.register("price_position_52w")
def price_position_52w(df: pd.DataFrame, idx: int) -> float:
    """Price position within 52-week (250-day) range. 0 = at low, 1 = at high."""
    lookback = min(250, idx + 1)
    if lookback < 20:
        return 0.5
    return _compute_price_position(df, idx, lookback)


@IndicatorRegistry.register("volatility_20d")
def volatility_20d(df: pd.DataFrame, idx: int) -> float:
    """20-day annualized volatility (std of daily returns * sqrt(250))."""
    if idx < 20:
        return 0.0
    returns = df["close"].iloc[idx - 19 : idx + 1].pct_change().dropna()
    if len(returns) < 5:
        return 0.0
    return float(returns.std() * (250**0.5))


@IndicatorRegistry.register("gap_up_pct")
def gap_up_pct(df: pd.DataFrame, idx: int) -> float:
    """Today's open gap percentage vs yesterday's close.

    Positive = gap up, negative = gap down.
    """
    if idx < 1:
        return 0.0
    prev_close = df["close"].iloc[idx - 1]
    if prev_close == 0:
        return 0.0
    return float((df["open"].iloc[idx] - prev_close) / prev_close)


@IndicatorRegistry.register("body_to_range_ratio")
def body_to_range_ratio(df: pd.DataFrame, idx: int) -> float:
    """Candle body size / total range. 1.0 = full body (strong), 0 = doji."""
    hi, lo = df["high"].iloc[idx], df["low"].iloc[idx]
    if hi == lo:
        return 0.0
    body = abs(df["close"].iloc[idx] - df["open"].iloc[idx])
    return float(body / (hi - lo))


# ═══════════════════════════════════════════════════════════════════════
#  L2 Extended Indicators — Need benchmark/sector data
#  Real logic when ctx is available; degraded defaults otherwise
# ═══════════════════════════════════════════════════════════════════════


@IndicatorRegistry.register("relative_strength_20d")
def relative_strength_20d(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> float:
    """Stock 20-day return minus benchmark 20-day return."""
    return _compute_relative_strength(df, idx, 20, ctx)


@IndicatorRegistry.register("st_flag")
def st_flag(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> bool:
    """Whether the stock is ST-flagged. Needs StockInfo in ctx."""
    if ctx and ctx.stock_info:
        return ctx.stock_info.is_st
    return False  # Degraded: assume not ST


@IndicatorRegistry.register("sector_heat_rank")
def sector_heat_rank(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> int:
    """Sector heat ranking (1 = hottest). Needs SectorInfo in ctx."""
    if ctx and ctx.sector_info and ctx.sector_info.heat_rank is not None:
        return ctx.sector_info.heat_rank
    return 1  # Degraded: assume top sector (always passes <= threshold)


@IndicatorRegistry.register("sector_heat_rising_days")
def sector_heat_rising_days(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> int:
    """Days sector heat rank has been rising."""
    if ctx and ctx.sector_info and ctx.sector_info.rising_days is not None:
        return ctx.sector_info.rising_days
    return 99


@IndicatorRegistry.register("intra_sector_strength_rank_pct")
def intra_sector_strength_rank_pct(
    df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None
) -> float:
    """Stock's strength rank within sector (0.0 = strongest)."""
    if ctx and ctx.intra_sector_strength_rank_pct is not None:
        return ctx.intra_sector_strength_rank_pct
    return 0.0


# ═══════════════════════════════════════════════════════════════════════
#  L3 Advanced Indicators — Need news/event APIs
#  Use explicit data when present; otherwise fall back to price/volume event proxies
# ═══════════════════════════════════════════════════════════════════════


@IndicatorRegistry.register("negative_news_score")
def negative_news_score(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> float:
    """Negative-news proxy derived from recent event-style downside moves."""
    if ctx is not None and ctx.negative_news_score is not None:
        return float(ctx.negative_news_score)
    return float(_event_proxy_value(df, idx, "negative_news_score"))


@IndicatorRegistry.register("news_sentiment_score")
def news_sentiment_score(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> float:
    """Positive/negative event proxy derived from gap + close-strength confirmation."""
    if ctx is not None and ctx.news_sentiment_score is not None:
        return float(ctx.news_sentiment_score)
    return float(_event_proxy_value(df, idx, "news_sentiment_score"))


@IndicatorRegistry.register("days_since_event")
def days_since_event(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> int:
    """Days since the latest proxy event day."""
    if ctx is not None and ctx.days_since_event is not None:
        return int(ctx.days_since_event)
    return int(_event_proxy_value(df, idx, "days_since_event"))


@IndicatorRegistry.register("price_above_pre_event")
def price_above_pre_event(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> bool:
    """Whether price still holds above the latest event anchor."""
    if ctx is not None and ctx.price_above_pre_event is not None:
        return bool(ctx.price_above_pre_event)
    return bool(_event_proxy_value(df, idx, "price_above_pre_event"))


@IndicatorRegistry.register("sector_fund_flow_positive")
def sector_fund_flow_positive(
    df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None
) -> bool:
    """Whether sector has positive fund flow."""
    if ctx and ctx.sector_info and ctx.sector_info.net_inflow is not None:
        return bool(ctx.sector_info.net_inflow > 0)
    return True


@IndicatorRegistry.register("already_overreacted")
def already_overreacted(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> bool:
    """Whether the post-event move is already stretched relative to the event anchor."""
    if ctx is not None and ctx.already_overreacted is not None:
        return bool(ctx.already_overreacted)
    return bool(_event_proxy_value(df, idx, "already_overreacted"))


@IndicatorRegistry.register("sector_risk_flag")
def sector_risk_flag(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> bool:
    """Whether sector has risk flag."""
    if ctx and ctx.sector_info:
        return ctx.sector_info.risk_flag
    return False


@IndicatorRegistry.register("sector_net_inflow_days")
def sector_net_inflow_days(df: pd.DataFrame, idx: int, ctx: IndicatorContext | None = None) -> int:
    """Days of consecutive sector net inflow."""
    if ctx and ctx.sector_info and ctx.sector_info.net_inflow_days is not None:
        return ctx.sector_info.net_inflow_days
    return 99
