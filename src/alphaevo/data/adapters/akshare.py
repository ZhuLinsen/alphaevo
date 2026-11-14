"""AkShare data adapter for AlphaEvo.

Provides access to full A-share market data via the ``akshare`` library.
Install the optional dependency with::

    pip install alphaevo[data-akshare]
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any, cast

import pandas as pd

from alphaevo.data.adapter import DataAdapter
from alphaevo.models.enums import MarketType
from alphaevo.models.market import StockInfo

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> float | None:
    """Best-effort conversion of provider values to float."""
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _ensure_akshare() -> Any:
    """Import akshare, raising a helpful error if not installed."""
    try:
        import akshare  # noqa: F401

        return akshare
    except ImportError as err:
        raise ImportError(
            "akshare is required for AkShareAdapter. "
            "Install with: pip install alphaevo[data-akshare]"
        ) from err


def _normalize_a_share_symbol(symbol: str) -> str:
    """Normalize A-share symbol to 6-digit format (no suffix).

    Accepts: '000001', '000001.SZ', 'sz000001', 'SZ000001'
    Returns: '000001'
    """
    s = symbol.strip().upper()
    # Remove exchange suffixes
    for suffix in (".SZ", ".SS", ".SH"):
        s = s.removesuffix(suffix)
    # Remove exchange prefixes
    for prefix in ("SZ", "SH", "SS"):
        if s.startswith(prefix) and len(s) > 2:
            s = s[2:]
    return s


class AkShareAdapter(DataAdapter):
    """Data adapter backed by the ``akshare`` library for A-share markets."""

    @property
    def name(self) -> str:  # noqa: D401
        return "akshare"

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Fetch daily OHLCV via akshare (run in a thread)."""
        code = _normalize_a_share_symbol(symbol)
        end_date = date.today()
        start_date = end_date - timedelta(days=int(days * 1.5))  # buffer for non-trading days

        try:
            df = await asyncio.to_thread(self._fetch_hist, code, start_date, end_date)
        except Exception:
            logger.exception("akshare fetch failed for %s", code)
            return self._empty_df()

        if df is None or df.empty:
            logger.warning("No data returned for %s", code)
            return self._empty_df()

        return self._normalize(df)

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Get stock list.

        For A-share markets, dynamically fetches from akshare.
        For other markets, returns empty (use YFinanceAdapter instead).
        """
        if market != MarketType.A_SHARE:
            return []

        try:
            stocks = await asyncio.to_thread(self._fetch_stock_list)
            return stocks
        except Exception:
            logger.exception("Failed to fetch A-share stock list")
            return []

    async def get_index_data(self, index_code: str, start: date, end: date) -> pd.DataFrame:
        """Fetch index data (e.g., '000300' for CSI300, '000001' for SSE Composite)."""
        try:
            df = await asyncio.to_thread(self._fetch_index, index_code, start, end)
            return self._normalize(df) if df is not None and not df.empty else self._empty_df()
        except Exception:
            logger.exception("akshare index fetch failed for %s", index_code)
            return self._empty_df()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_hist(code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Synchronous akshare call."""
        ak = _ensure_akshare()
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust="qfq",  # forward-adjusted
        )
        return cast("pd.DataFrame", df)

    @staticmethod
    def _fetch_stock_list() -> list[StockInfo]:
        """Get full A-share stock list."""
        ak = _ensure_akshare()
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return []

        stocks = []
        for _, row in df.head(500).iterrows():  # Limit for speed
            code = str(row.get("代码", "")).strip()
            name = str(row.get("名称", "")).strip()
            if code and len(code) == 6:
                stocks.append(
                    StockInfo(
                        symbol=code,
                        name=name,
                        market=MarketType.A_SHARE,
                        sector=None,
                        market_cap=_to_float(row.get("总市值")),
                        pe_ttm=_to_float(row.get("市盈率-动态", row.get("市盈率"))),
                    )
                )
        return stocks

    @staticmethod
    def _fetch_index(index_code: str, start: date, end: date) -> pd.DataFrame:
        """Fetch index OHLCV."""
        ak = _ensure_akshare()
        df = ak.stock_zh_index_daily_em(
            symbol=f"sh{index_code}" if index_code.startswith(("0", "9")) else f"sz{index_code}",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        return cast("pd.DataFrame", df)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize akshare DataFrame to AlphaEvo format."""
        df = df.copy()

        # akshare columns are in Chinese; map to standard names
        col_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            # Index data sometimes uses these
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure required columns exist
        for col in ["date", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                if col == "volume":
                    df[col] = 0
                else:
                    logger.warning(
                        "AkShare returned data missing '%s' column — API format may have changed",
                        col,
                    )
                    return AkShareAdapter._empty_df()

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)

        # Add prev_close
        df["prev_close"] = df["close"].shift(1)

        result = df[["date", "open", "high", "low", "close", "volume", "prev_close"]].copy()
        return cast("pd.DataFrame", result)

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Return an empty DataFrame with the standard column schema."""
        return cast(
            "pd.DataFrame",
            pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "prev_close"]),
        )
