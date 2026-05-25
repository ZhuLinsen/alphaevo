"""Optional Adanos Market Sentiment event-context adapter.

Adanos supplies external sentiment context for US equities. It does not provide
OHLCV history, so AlphaEvo uses this adapter only as a secondary context source
behind a normal market-data adapter such as yfinance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any, cast

import pandas as pd

from alphaevo.data.adapter import DataAdapter
from alphaevo.models.enums import MarketType
from alphaevo.models.market import (
    EventContextRecord,
    EventContextSeries,
    MarketContext,
    StockInfo,
)

logger = logging.getLogger(__name__)


class AdanosSentimentAdapter(DataAdapter):
    """Secondary data adapter for Adanos event/news sentiment context."""

    SOURCE_PATHS = {
        "reddit": "/reddit/stocks/v1",
        "x": "/x/stocks/v1",
        "news": "/news/stocks/v1",
        "polymarket": "/polymarket/stocks/v1",
    }

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.adanos.org",
        sources: tuple[str, ...] = ("reddit", "x", "news", "polymarket"),
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.sources = tuple(dict.fromkeys(source.lower() for source in sources))
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "adanos"

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Adanos does not provide OHLCV data."""
        return pd.DataFrame()

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Adanos is a context adapter and does not provide universe lists."""
        return []

    async def get_event_context(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> EventContextSeries | None:
        """Fetch current Adanos ticker sentiment as a sparse event context row."""
        if start > end or not self._looks_like_us_symbol(symbol):
            return None

        payloads: list[dict[str, Any]] = []
        for source in self.sources:
            if source not in self.SOURCE_PATHS:
                continue
            payload = await asyncio.to_thread(
                self._request,
                source,
                f"/stock/{symbol.upper()}",
                days=(end - start).days + 1,
            )
            if payload:
                payloads.append(payload)

        if not payloads:
            return None

        sentiment = _average([_extract_sentiment_score(payload) for payload in payloads])
        if sentiment is None:
            return None

        negative = _negative_score(sentiment, payloads)
        return EventContextSeries(
            symbol=symbol.upper(),
            source="adanos",
            records=[
                EventContextRecord(
                    date=end,
                    negative_news_score=round(negative, 4),
                    news_sentiment_score=round(sentiment, 4),
                    days_since_event=0,
                )
            ],
        )

    async def get_market_context(self, market: MarketType) -> MarketContext | None:
        """Fetch Adanos market-wide sentiment for US equities."""
        if market != MarketType.US:
            return None

        payloads: list[dict[str, Any]] = []
        for source in self.sources:
            if source not in self.SOURCE_PATHS:
                continue
            payload = await asyncio.to_thread(self._request, source, "/market-sentiment")
            if payload:
                payloads.append(payload)

        sentiment = _average([_extract_sentiment_score(payload) for payload in payloads])
        if sentiment is None:
            return None
        return MarketContext(sentiment_index=round(sentiment, 4))

    def _request(
        self,
        source: str,
        endpoint: str,
        *,
        days: int | None = None,
    ) -> dict[str, Any] | None:
        url = f"{self.base_url}{self.SOURCE_PATHS[source]}{endpoint}"
        if days is not None:
            url = f"{url}?{urllib.parse.urlencode({'days': max(1, min(days, 365))})}"
        request = urllib.request.Request(url, headers={"X-API-Key": self.api_key})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError):
            logger.debug("Adanos sentiment request failed for %s", url, exc_info=True)
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Adanos sentiment response was not JSON for %s", url)
            return None
        return cast("dict[str, Any] | None", payload if isinstance(payload, dict) else None)

    @staticmethod
    def _looks_like_us_symbol(symbol: str) -> bool:
        normalized = symbol.strip().upper()
        return normalized.isalpha() and 1 <= len(normalized) <= 5


def _extract_sentiment_score(payload: dict[str, Any]) -> float | None:
    """Extract and normalize common Adanos sentiment score fields to 0..1."""
    for key in ("sentiment_score", "sentiment"):
        value = _to_float(payload.get(key))
        if value is not None:
            normalized = _from_directional_score(value)
            if normalized is not None:
                return normalized

    for key in ("news_sentiment_score", "score", "bullish_score"):
        value = _to_float(payload.get(key))
        if value is not None:
            normalized = _from_unit_score(value)
            if normalized is not None:
                return normalized
    return None


def _negative_score(sentiment: float, payloads: list[dict[str, Any]]) -> float:
    for payload in payloads:
        value = _to_float(payload.get("negative_news_score") or payload.get("bearish_score"))
        if value is not None:
            normalized = _from_unit_score(value)
            if normalized is not None:
                return normalized
    return 1.0 - sentiment


def _from_directional_score(value: float) -> float | None:
    if -1.0 <= value <= 1.0:
        return (value + 1.0) / 2.0
    if 0.0 <= value <= 100.0:
        return value / 100.0
    return None


def _from_unit_score(value: float) -> float | None:
    if 0.0 <= value <= 1.0:
        return value
    if 0.0 <= value <= 100.0:
        return value / 100.0
    return None


def _average(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
