"""Market regime detection from index/price data.

Classifies market environments into five categories based on trend,
volatility, and momentum dimensions — enabling regime-aware strategy
evolution.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from alphaevo.models.enums import MarketRegime


class RegimeDetector:
    """Detect market regimes from OHLCV data.

    Uses three dimensions:
      1. Trend: MA20 slope + price vs MA60
      2. Volatility: ATR(20) / Close
      3. Momentum: Rate of change over 20 days

    Example::

        detector = RegimeDetector()
        regime = detector.detect(index_df)  # MarketRegime.TRENDING_UP
    """

    def __init__(
        self,
        *,
        trend_slope_threshold: float = 0.001,
        crash_slope_threshold: float = -0.003,
        volatility_threshold: float = 0.03,
        ma_short: int = 20,
        ma_long: int = 60,
        atr_period: int = 20,
    ) -> None:
        self.trend_slope_threshold = trend_slope_threshold
        self.crash_slope_threshold = crash_slope_threshold
        self.volatility_threshold = volatility_threshold
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.atr_period = atr_period

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Classify the current market regime from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns: close, high, low.
            Must have at least ``ma_long + 5`` rows.

        Returns
        -------
        MarketRegime
            The detected regime.
        """
        if len(df) < self.ma_long + 5:
            return MarketRegime.RANGE_BOUND

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        # --- Dimension 1: Trend ---
        ma_short = self._sma(close, self.ma_short)
        ma_long = self._sma(close, self.ma_long)

        # MA20 slope: measured as normalized daily change of MA20
        slope = self._slope(ma_short, window=5)
        price_above_ma_long = close[-1] > ma_long[-1] if ma_long[-1] > 0 else False

        # --- Dimension 2: Volatility ---
        atr = self._atr(high, low, close, self.atr_period)
        rel_volatility = atr[-1] / close[-1] if close[-1] > 0 else 0.0

        # --- Dimension 3: Momentum / Drawdown ---
        recent_max = np.max(close[-self.ma_short :])
        drawdown_from_peak = (recent_max - close[-1]) / recent_max if recent_max > 0 else 0.0

        # --- Classification ---
        # Crash: steep negative slope + large drawdown
        if slope < self.crash_slope_threshold and drawdown_from_peak > 0.10:
            return MarketRegime.PANIC

        # High volatility (can overlap with any trend)
        if rel_volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE

        # Trending up
        if slope > self.trend_slope_threshold and price_above_ma_long:
            # Check for euphoria: extended slope + low volatility
            if (
                slope > self.trend_slope_threshold * 4
                and rel_volatility < self.volatility_threshold * 0.5
            ):
                return MarketRegime.EUPHORIA
            return MarketRegime.TRENDING_UP

        # Trending down
        if slope < -self.trend_slope_threshold and not price_above_ma_long:
            return MarketRegime.TRENDING_DOWN

        return MarketRegime.RANGE_BOUND

    def detect_series(self, df: pd.DataFrame) -> list[MarketRegime]:
        """Classify the market regime at every bar causally in O(N) time."""
        n = len(df)
        if n == 0:
            return []
        min_bars = self.ma_long + 5
        if n < min_bars:
            return [MarketRegime.RANGE_BOUND] * n

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        ma_short = self._sma(close, self.ma_short)
        ma_long = self._sma(close, self.ma_long)
        atr = self._atr(high, low, close, self.atr_period)
        recent_max = (
            pd.Series(close).rolling(self.ma_short, min_periods=1).max().to_numpy(dtype=float)
        )

        regimes: list[MarketRegime] = [MarketRegime.RANGE_BOUND] * n
        for i in range(min_bars - 1, n):
            ma_s_now = ma_short[i]
            ma_s_prev = ma_short[i - 4] if i >= 4 else np.nan
            if (
                np.isnan(ma_s_now)
                or np.isnan(ma_s_prev)
                or ma_s_prev == 0
                or not np.isfinite(ma_s_prev)
            ):
                slope = 0.0
            else:
                slope = float((ma_s_now - ma_s_prev) / (ma_s_prev * 5))
                if not np.isfinite(slope):
                    slope = 0.0

            price_above_ma_long = (
                bool(close[i] > ma_long[i])
                if not np.isnan(ma_long[i]) and ma_long[i] > 0
                else False
            )
            rel_volatility = float(atr[i] / close[i]) if close[i] > 0 else 0.0
            rmax = recent_max[i]
            drawdown_from_peak = float((rmax - close[i]) / rmax) if rmax > 0 else 0.0

            if slope < self.crash_slope_threshold and drawdown_from_peak > 0.10:
                regimes[i] = MarketRegime.PANIC
            elif rel_volatility > self.volatility_threshold:
                regimes[i] = MarketRegime.VOLATILE
            elif slope > self.trend_slope_threshold and price_above_ma_long:
                if (
                    slope > self.trend_slope_threshold * 4
                    and rel_volatility < self.volatility_threshold * 0.5
                ):
                    regimes[i] = MarketRegime.EUPHORIA
                else:
                    regimes[i] = MarketRegime.TRENDING_UP
            elif slope < -self.trend_slope_threshold and not price_above_ma_long:
                regimes[i] = MarketRegime.TRENDING_DOWN
            else:
                regimes[i] = MarketRegime.RANGE_BOUND

        return regimes

    def detect_periods(
        self,
        df: pd.DataFrame,
        window: int = 60,
        step: int = 20,
    ) -> list[tuple[date, date, MarketRegime]]:
        """Divide the data into regime periods using a sliding window.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV dataset with a 'date' column.
        window : int
            Window size in rows for each detection.
        step : int
            Step size between windows.

        Returns
        -------
        list of (start_date, end_date, MarketRegime)
        """
        if len(df) < window:
            return []

        periods: list[tuple[date, date, MarketRegime]] = []
        dates = pd.to_datetime(df["date"]).dt.date.values if "date" in df.columns else None

        for start in range(0, len(df) - window + 1, step):
            end_idx = start + window - 1
            chunk = df.iloc[start : end_idx + 1].reset_index(drop=True)
            regime = self.detect(chunk)

            if dates is not None:
                start_date = dates[start]
                end_date = dates[end_idx]
            else:
                start_date = date(2000, 1, 1)
                end_date = date(2000, 1, 1)

            # Merge with previous period if same regime
            if periods and periods[-1][2] == regime:
                periods[-1] = (periods[-1][0], end_date, regime)
            else:
                periods.append((start_date, end_date, regime))

        return periods

    # --- Internal helpers ---

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average using cumsum for efficiency."""
        if len(data) < period:
            return np.full_like(data, np.nan)
        cumsum = np.cumsum(np.insert(data, 0, 0.0))
        sma = (cumsum[period:] - cumsum[:-period]) / period
        # Pad front with NaN
        return np.concatenate([np.full(period - 1, np.nan), sma])

    @staticmethod
    def _slope(ma: np.ndarray, window: int = 5) -> float:
        """Normalized slope of the last ``window`` MA values."""
        valid = ma[~np.isnan(ma)]
        if len(valid) < window:
            return 0.0
        segment = valid[-window:]
        if segment[0] == 0 or not np.isfinite(segment[0]):
            return 0.0
        result = float((segment[-1] - segment[0]) / (segment[0] * window))
        return result if np.isfinite(result) else 0.0

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Average True Range."""
        if len(high) < 2:
            return np.array([0.0])
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        # Prepend first bar's range
        tr = np.insert(tr, 0, high[0] - low[0])
        if len(tr) < period:
            return np.array([np.mean(tr)])
        # Simple rolling average
        cumsum = np.cumsum(np.insert(tr, 0, 0.0))
        atr = (cumsum[period:] - cumsum[:-period]) / period
        return np.concatenate([np.full(period - 1, np.mean(tr[:period])), atr])
