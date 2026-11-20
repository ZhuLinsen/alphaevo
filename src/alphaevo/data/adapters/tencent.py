"""Direct Tencent Finance A-share data adapter.

This adapter is a lightweight, dependency-free A-share K-line source inspired by
common A-share data fallback projects: use direct HTTP endpoints for stable daily
OHLCV enrichment, and keep wrapper-heavy libraries (AkShare/DSA) as fallbacks for
richer universes and specialized context.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import date, datetime
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from alphaevo.data.adapter import DataAdapter
from alphaevo.models.enums import MarketType
from alphaevo.models.market import RealTimeQuote, StockInfo

logger = logging.getLogger(__name__)

_TENCENT_DAY_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def normalize_tencent_symbol(symbol: str) -> str:
    """Normalize common A-share symbol formats to Tencent's sh/sz/bj prefix."""
    raw = symbol.strip().lower()
    for suffix in (".xshg", ".sh", ".ss"):
        raw = raw.removesuffix(suffix)
        if raw.isdigit():
            return f"sh{raw}"
    for suffix in (".xshe", ".sz"):
        raw = raw.removesuffix(suffix)
        if raw.isdigit():
            return f"sz{raw}"
    if raw.startswith(("sh", "sz", "bj")) and len(raw) > 2:
        return raw
    if raw.isdigit():
        if raw.startswith(("6", "9")):
            return f"sh{raw}"
        if raw.startswith(("0", "2", "3")):
            return f"sz{raw}"
        if raw.startswith(("4", "8")):
            return f"bj{raw}"
    return raw


class TencentAshareAdapter(DataAdapter):
    """Direct Tencent daily K-line adapter for A-share symbols.

    It intentionally focuses on OHLCV/index history. Stock universes and rich
    context should still come from DSA/AkShare, so this adapter composes well in
    an ``auto`` chain rather than pretending to be a full market-data provider.
    """

    @property
    def name(self) -> str:  # noqa: D401
        return "tencent"

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        code = normalize_tencent_symbol(symbol)
        try:
            payload = await asyncio.to_thread(self._fetch_daily_payload, code, days)
            return self._normalize_daily_payload(payload, code)
        except Exception:
            logger.exception("Tencent daily fetch failed for %s", code)
            return self._empty_df()

    async def get_history_range(self, symbol: str, start: date, end: date) -> pd.DataFrame | None:
        """Fetch daily OHLCV for an explicit calendar range [start, end]."""
        code = normalize_tencent_symbol(symbol)
        count = max(1, (end - start).days + 30)
        try:
            payload = await asyncio.to_thread(
                self._fetch_daily_payload,
                code,
                count,
                start,
                end,
            )
            return self._normalize_daily_payload(payload, code)
        except Exception:
            logger.exception("Tencent range fetch failed for %s", code)
            return self._empty_df()

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Tencent direct K-line endpoint does not provide a full stock universe."""
        return []

    async def get_realtime_quote(self, symbol: str) -> RealTimeQuote | None:
        """Fetch a Tencent real-time quote for an A-share symbol."""
        code = normalize_tencent_symbol(symbol)
        try:
            text = await asyncio.to_thread(self._fetch_quote_text, code)
            return self._parse_quote_text(text, code)
        except Exception:
            logger.exception("Tencent quote fetch failed for %s", code)
            return None

    async def get_index_data(self, index_symbol: str, start: date, end: date) -> pd.DataFrame:
        code = normalize_tencent_symbol(index_symbol)
        df = await self.get_history_range(code, start, end)
        if df is None or df.empty:
            return self._empty_df()
        mask = (df["date"] >= start) & (df["date"] <= end)
        return cast("pd.DataFrame", df.loc[mask].reset_index(drop=True))

    @staticmethod
    def _fetch_daily_payload(
        code: str,
        days: int,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, Any]:
        count = max(1, int(days))
        start_s = start.isoformat() if start is not None else ""
        end_s = end.isoformat() if end is not None else ""
        params = urlencode({"param": f"{code},day,{start_s},{end_s},{count},qfq"})
        request = Request(f"{_TENCENT_DAY_URL}?{params}", headers=_HEADERS)
        with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS endpoint.
            data = response.read().decode("utf-8")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError("Tencent response is not a JSON object")
        return parsed

    @staticmethod
    def _fetch_quote_text(code: str) -> str:
        request = Request(f"{_TENCENT_QUOTE_URL}{code}", headers=_HEADERS)
        with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS endpoint.
            return cast("str", response.read().decode("gbk", errors="ignore"))

    @staticmethod
    def _parse_quote_text(text: str, code: str) -> RealTimeQuote | None:
        if '="' not in text:
            return None
        body = text.split('="', 1)[1].split('"', 1)[0]
        fields = body.split("~")
        if len(fields) < 33 or not fields[3]:
            return None
        timestamp: datetime | None = None
        if len(fields) > 30 and fields[30]:
            with contextlib.suppress(ValueError):
                timestamp = datetime.strptime(fields[30], "%Y%m%d%H%M%S")
        amount: float | None = None
        if len(fields) > 37 and fields[37]:
            with contextlib.suppress(ValueError):
                amount = float(fields[37])
        return RealTimeQuote(
            symbol=code,
            name=fields[1],
            price=float(fields[3]),
            change_pct=float(fields[32]) if fields[32] else 0.0,
            volume=float(fields[6]) if len(fields) > 6 and fields[6] else 0.0,
            amount=amount,
            timestamp=timestamp,
        )

    @staticmethod
    def _normalize_daily_payload(payload: dict[str, Any], code: str) -> pd.DataFrame:
        data = payload.get("data")
        if not isinstance(data, dict):
            return TencentAshareAdapter._empty_df()
        stock_payload = data.get(code)
        if not isinstance(stock_payload, dict):
            return TencentAshareAdapter._empty_df()
        rows = stock_payload.get("qfqday") or stock_payload.get("day")
        if not isinstance(rows, list) or not rows:
            return TencentAshareAdapter._empty_df()

        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list | tuple) or len(row) < 6:
                continue
            records.append(
                {
                    "date": row[0],
                    "open": row[1],
                    "close": row[2],
                    "high": row[3],
                    "low": row[4],
                    "volume": row[5],
                    "amount": row[6] if len(row) > 6 else None,
                }
            )
        if not records:
            return TencentAshareAdapter._empty_df()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close"])
        df = df.sort_values("date").reset_index(drop=True)
        df["prev_close"] = df["close"].shift(1)
        return cast(
            "pd.DataFrame",
            df[["date", "open", "high", "low", "close", "volume", "amount", "prev_close"]].copy(),
        )

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return cast(
            "pd.DataFrame",
            pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "prev_close"]
            ),
        )
