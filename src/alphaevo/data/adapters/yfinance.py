"""YFinance data adapter for AlphaEvo.

Wraps the ``yfinance`` library to provide OHLCV data for US, HK, and
A-share markets.  Install the optional dependency with::

    pip install alphaevo[data-yfinance]
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from alphaevo.data.adapter import DataAdapter
from alphaevo.models.enums import MarketType
from alphaevo.models.market import EventContextRecord, EventContextSeries, StockInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated stock lists (MVP – hardcoded for speed & reliability)
# ---------------------------------------------------------------------------

_STOCK_LISTS: dict[MarketType, list[dict]] = {
    MarketType.A_SHARE: [
        {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "sector": "银行",
            "market_cap": 210_000_000_000,
            "pe_ttm": 6.5,
        },
        {
            "symbol": "600519.SS",
            "name": "贵州茅台",
            "sector": "白酒",
            "market_cap": 2_100_000_000_000,
            "pe_ttm": 28.0,
        },
        {
            "symbol": "000858.SZ",
            "name": "五粮液",
            "sector": "白酒",
            "market_cap": 520_000_000_000,
            "pe_ttm": 19.0,
        },
        {
            "symbol": "601318.SS",
            "name": "中国平安",
            "sector": "保险",
            "market_cap": 860_000_000_000,
            "pe_ttm": 7.8,
        },
        {
            "symbol": "600036.SS",
            "name": "招商银行",
            "sector": "银行",
            "market_cap": 940_000_000_000,
            "pe_ttm": 7.2,
        },
        {
            "symbol": "000333.SZ",
            "name": "美的集团",
            "sector": "家电",
            "market_cap": 540_000_000_000,
            "pe_ttm": 14.5,
        },
        {
            "symbol": "600276.SS",
            "name": "恒瑞医药",
            "sector": "医药",
            "market_cap": 340_000_000_000,
            "pe_ttm": 42.0,
        },
        {
            "symbol": "601166.SS",
            "name": "兴业银行",
            "sector": "银行",
            "market_cap": 340_000_000_000,
            "pe_ttm": 5.9,
        },
        {
            "symbol": "000651.SZ",
            "name": "格力电器",
            "sector": "家电",
            "market_cap": 220_000_000_000,
            "pe_ttm": 8.6,
        },
        {
            "symbol": "600900.SS",
            "name": "长江电力",
            "sector": "电力",
            "market_cap": 700_000_000_000,
            "pe_ttm": 21.0,
        },
        {
            "symbol": "300750.SZ",
            "name": "宁德时代",
            "sector": "电池",
            "market_cap": 820_000_000_000,
            "pe_ttm": 24.0,
        },
        {
            "symbol": "601398.SS",
            "name": "工商银行",
            "sector": "银行",
            "market_cap": 1_900_000_000_000,
            "pe_ttm": 6.0,
        },
        {
            "symbol": "601288.SS",
            "name": "农业银行",
            "sector": "银行",
            "market_cap": 1_600_000_000_000,
            "pe_ttm": 5.8,
        },
        {
            "symbol": "601939.SS",
            "name": "建设银行",
            "sector": "银行",
            "market_cap": 1_800_000_000_000,
            "pe_ttm": 6.3,
        },
        {
            "symbol": "601988.SS",
            "name": "中国银行",
            "sector": "银行",
            "market_cap": 1_100_000_000_000,
            "pe_ttm": 5.5,
        },
        {
            "symbol": "002594.SZ",
            "name": "比亚迪",
            "sector": "汽车",
            "market_cap": 720_000_000_000,
            "pe_ttm": 23.0,
        },
        {
            "symbol": "600030.SS",
            "name": "中信证券",
            "sector": "券商",
            "market_cap": 350_000_000_000,
            "pe_ttm": 17.5,
        },
        {
            "symbol": "601899.SS",
            "name": "紫金矿业",
            "sector": "有色",
            "market_cap": 420_000_000_000,
            "pe_ttm": 16.0,
        },
        {
            "symbol": "600309.SS",
            "name": "万华化学",
            "sector": "化工",
            "market_cap": 280_000_000_000,
            "pe_ttm": 15.0,
        },
        {
            "symbol": "300760.SZ",
            "name": "迈瑞医疗",
            "sector": "医疗器械",
            "market_cap": 340_000_000_000,
            "pe_ttm": 30.0,
        },
        {
            "symbol": "603259.SS",
            "name": "药明康德",
            "sector": "医药",
            "market_cap": 180_000_000_000,
            "pe_ttm": 20.0,
        },
        {
            "symbol": "601888.SS",
            "name": "中国中免",
            "sector": "消费",
            "market_cap": 160_000_000_000,
            "pe_ttm": 22.0,
        },
        {
            "symbol": "600690.SS",
            "name": "海尔智家",
            "sector": "家电",
            "market_cap": 240_000_000_000,
            "pe_ttm": 14.0,
        },
        {
            "symbol": "600104.SS",
            "name": "上汽集团",
            "sector": "汽车",
            "market_cap": 180_000_000_000,
            "pe_ttm": 9.0,
        },
        {
            "symbol": "600031.SS",
            "name": "三一重工",
            "sector": "机械",
            "market_cap": 160_000_000_000,
            "pe_ttm": 14.0,
        },
    ],
    MarketType.US: [
        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
        {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
        {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial"},
        {"symbol": "V", "name": "Visa Inc.", "sector": "Financial"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
        {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive"},
        {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Defensive"},
        {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Financial"},
        {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Consumer Cyclical"},
        {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Healthcare"},
        {"symbol": "DIS", "name": "Walt Disney Co.", "sector": "Communication"},
        {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
        {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
        {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Communication"},
        {"symbol": "AMD", "name": "Advanced Micro Devices Inc.", "sector": "Technology"},
        {"symbol": "INTC", "name": "Intel Corp.", "sector": "Technology"},
        {"symbol": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technology"},
        {"symbol": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer Defensive"},
        {"symbol": "KO", "name": "Coca-Cola Co.", "sector": "Consumer Defensive"},
        {"symbol": "COST", "name": "Costco Wholesale Corp.", "sector": "Consumer Defensive"},
        {"symbol": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare"},
        {"symbol": "MRK", "name": "Merck & Co. Inc.", "sector": "Healthcare"},
        {"symbol": "LLY", "name": "Eli Lilly & Co.", "sector": "Healthcare"},
        {"symbol": "BAC", "name": "Bank of America Corp.", "sector": "Financial"},
        {"symbol": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
        {"symbol": "ORCL", "name": "Oracle Corp.", "sector": "Technology"},
        {"symbol": "IBM", "name": "IBM Corp.", "sector": "Technology"},
        {"symbol": "QCOM", "name": "Qualcomm Inc.", "sector": "Technology"},
        {"symbol": "TXN", "name": "Texas Instruments Inc.", "sector": "Technology"},
        {"symbol": "CAT", "name": "Caterpillar Inc.", "sector": "Industrials"},
        {"symbol": "CVX", "name": "Chevron Corp.", "sector": "Energy"},
        {"symbol": "MCD", "name": "McDonald's Corp.", "sector": "Consumer Defensive"},
        {"symbol": "NKE", "name": "Nike Inc.", "sector": "Consumer Cyclical"},
        {"symbol": "LOW", "name": "Lowe's Companies Inc.", "sector": "Consumer Cyclical"},
        {"symbol": "UPS", "name": "United Parcel Service Inc.", "sector": "Industrials"},
    ],
    MarketType.HK: [
        {"symbol": "0700.HK", "name": "腾讯控股", "sector": "科技"},
        {"symbol": "9988.HK", "name": "阿里巴巴", "sector": "科技"},
        {"symbol": "0005.HK", "name": "汇丰控股", "sector": "银行"},
        {"symbol": "1299.HK", "name": "友邦保险", "sector": "保险"},
        {"symbol": "0941.HK", "name": "中国移动", "sector": "电信"},
        {"symbol": "3690.HK", "name": "美团", "sector": "科技"},
        {"symbol": "1810.HK", "name": "小米集团", "sector": "科技"},
        {"symbol": "2318.HK", "name": "中国平安", "sector": "保险"},
        {"symbol": "0388.HK", "name": "香港交易所", "sector": "金融"},
        {"symbol": "9618.HK", "name": "京东集团", "sector": "科技"},
    ],
}

# Broaden the built-in A-share universe beyond the original 25-symbol smoke set.
# These liquid large/mid-cap names give the sampler enough breadth when optional
# full-universe providers (AkShare/DSA) are unavailable.  Market caps are coarse
# liquidity proxies used only for universe filtering, not valuation research.
_STOCK_LISTS[MarketType.A_SHARE].extend(
    [
        {
            "symbol": "000002.SZ",
            "name": "万科A",
            "sector": "地产",
            "market_cap": 80_000_000_000,
            "pe_ttm": 9.0,
        },
        {
            "symbol": "000063.SZ",
            "name": "中兴通讯",
            "sector": "通信",
            "market_cap": 150_000_000_000,
            "pe_ttm": 18.0,
        },
        {
            "symbol": "000100.SZ",
            "name": "TCL科技",
            "sector": "电子",
            "market_cap": 80_000_000_000,
            "pe_ttm": 16.0,
        },
        {
            "symbol": "000338.SZ",
            "name": "潍柴动力",
            "sector": "汽车",
            "market_cap": 120_000_000_000,
            "pe_ttm": 12.0,
        },
        {
            "symbol": "000568.SZ",
            "name": "泸州老窖",
            "sector": "白酒",
            "market_cap": 250_000_000_000,
            "pe_ttm": 20.0,
        },
        {
            "symbol": "000625.SZ",
            "name": "长安汽车",
            "sector": "汽车",
            "market_cap": 150_000_000_000,
            "pe_ttm": 16.0,
        },
        {
            "symbol": "000725.SZ",
            "name": "京东方A",
            "sector": "电子",
            "market_cap": 150_000_000_000,
            "pe_ttm": 18.0,
        },
        {
            "symbol": "000776.SZ",
            "name": "广发证券",
            "sector": "券商",
            "market_cap": 100_000_000_000,
            "pe_ttm": 15.0,
        },
        {
            "symbol": "000977.SZ",
            "name": "浪潮信息",
            "sector": "计算机",
            "market_cap": 80_000_000_000,
            "pe_ttm": 30.0,
        },
        {
            "symbol": "002142.SZ",
            "name": "宁波银行",
            "sector": "银行",
            "market_cap": 150_000_000_000,
            "pe_ttm": 7.0,
        },
        {
            "symbol": "002230.SZ",
            "name": "科大讯飞",
            "sector": "计算机",
            "market_cap": 100_000_000_000,
            "pe_ttm": 80.0,
        },
        {
            "symbol": "002371.SZ",
            "name": "北方华创",
            "sector": "半导体",
            "market_cap": 170_000_000_000,
            "pe_ttm": 55.0,
        },
        {
            "symbol": "002415.SZ",
            "name": "海康威视",
            "sector": "计算机",
            "market_cap": 300_000_000_000,
            "pe_ttm": 20.0,
        },
        {
            "symbol": "002475.SZ",
            "name": "立讯精密",
            "sector": "电子",
            "market_cap": 250_000_000_000,
            "pe_ttm": 22.0,
        },
        {
            "symbol": "300014.SZ",
            "name": "亿纬锂能",
            "sector": "电池",
            "market_cap": 90_000_000_000,
            "pe_ttm": 25.0,
        },
        {
            "symbol": "300015.SZ",
            "name": "爱尔眼科",
            "sector": "医疗服务",
            "market_cap": 100_000_000_000,
            "pe_ttm": 45.0,
        },
        {
            "symbol": "300059.SZ",
            "name": "东方财富",
            "sector": "券商",
            "market_cap": 220_000_000_000,
            "pe_ttm": 25.0,
        },
        {
            "symbol": "300124.SZ",
            "name": "汇川技术",
            "sector": "自动化",
            "market_cap": 150_000_000_000,
            "pe_ttm": 30.0,
        },
        {
            "symbol": "300274.SZ",
            "name": "阳光电源",
            "sector": "光伏",
            "market_cap": 150_000_000_000,
            "pe_ttm": 22.0,
        },
        {
            "symbol": "300308.SZ",
            "name": "中际旭创",
            "sector": "通信",
            "market_cap": 120_000_000_000,
            "pe_ttm": 35.0,
        },
        {
            "symbol": "600000.SS",
            "name": "浦发银行",
            "sector": "银行",
            "market_cap": 220_000_000_000,
            "pe_ttm": 6.0,
        },
        {
            "symbol": "600009.SS",
            "name": "上海机场",
            "sector": "机场",
            "market_cap": 90_000_000_000,
            "pe_ttm": 30.0,
        },
        {
            "symbol": "600050.SS",
            "name": "中国联通",
            "sector": "通信",
            "market_cap": 150_000_000_000,
            "pe_ttm": 16.0,
        },
        {
            "symbol": "600150.SS",
            "name": "中国船舶",
            "sector": "军工",
            "market_cap": 160_000_000_000,
            "pe_ttm": 45.0,
        },
        {
            "symbol": "600406.SS",
            "name": "国电南瑞",
            "sector": "电力设备",
            "market_cap": 180_000_000_000,
            "pe_ttm": 22.0,
        },
        {
            "symbol": "600438.SS",
            "name": "通威股份",
            "sector": "光伏",
            "market_cap": 90_000_000_000,
            "pe_ttm": 12.0,
        },
        {
            "symbol": "600585.SS",
            "name": "海螺水泥",
            "sector": "建材",
            "market_cap": 120_000_000_000,
            "pe_ttm": 9.0,
        },
        {
            "symbol": "600887.SS",
            "name": "伊利股份",
            "sector": "消费",
            "market_cap": 180_000_000_000,
            "pe_ttm": 18.0,
        },
        {
            "symbol": "601012.SS",
            "name": "隆基绿能",
            "sector": "光伏",
            "market_cap": 150_000_000_000,
            "pe_ttm": 14.0,
        },
        {
            "symbol": "601088.SS",
            "name": "中国神华",
            "sector": "煤炭",
            "market_cap": 700_000_000_000,
            "pe_ttm": 11.0,
        },
        {
            "symbol": "601328.SS",
            "name": "交通银行",
            "sector": "银行",
            "market_cap": 500_000_000_000,
            "pe_ttm": 6.0,
        },
        {
            "symbol": "601601.SS",
            "name": "中国太保",
            "sector": "保险",
            "market_cap": 250_000_000_000,
            "pe_ttm": 8.0,
        },
        {
            "symbol": "601668.SS",
            "name": "中国建筑",
            "sector": "建筑",
            "market_cap": 220_000_000_000,
            "pe_ttm": 5.0,
        },
        {
            "symbol": "601688.SS",
            "name": "华泰证券",
            "sector": "券商",
            "market_cap": 140_000_000_000,
            "pe_ttm": 13.0,
        },
        {
            "symbol": "601816.SS",
            "name": "京沪高铁",
            "sector": "交通运输",
            "market_cap": 250_000_000_000,
            "pe_ttm": 25.0,
        },
        {
            "symbol": "601857.SS",
            "name": "中国石油",
            "sector": "石油",
            "market_cap": 1_500_000_000_000,
            "pe_ttm": 10.0,
        },
        {
            "symbol": "601919.SS",
            "name": "中远海控",
            "sector": "航运",
            "market_cap": 180_000_000_000,
            "pe_ttm": 8.0,
        },
        {
            "symbol": "603288.SS",
            "name": "海天味业",
            "sector": "消费",
            "market_cap": 200_000_000_000,
            "pe_ttm": 35.0,
        },
        {
            "symbol": "603501.SS",
            "name": "韦尔股份",
            "sector": "半导体",
            "market_cap": 120_000_000_000,
            "pe_ttm": 40.0,
        },
    ]
)

# Mapping from calendar days to yfinance period strings.
_DAYS_TO_PERIOD: list[tuple[int, str]] = [
    (5, "5d"),
    (30, "1mo"),
    (90, "3mo"),
    (180, "6mo"),
    (365, "1y"),
    (730, "2y"),
    (1825, "5y"),
]


def _days_to_yf_period(days: int) -> str:
    """Convert a number of calendar *days* to the closest yfinance period."""
    for threshold, period in _DAYS_TO_PERIOD:
        if days <= threshold:
            return period
    return "max"


def _to_yf_symbol(symbol: str) -> str:
    """Ensure the symbol carries the suffix yfinance expects.

    A-share codes (6-digit, pure numeric) get ``.SS`` (Shanghai) or
    ``.SZ`` (Shenzhen) when missing.  Everything else passes through.
    """
    if "." in symbol:
        return symbol

    # Pure numeric → likely A-share
    if symbol.isdigit() and len(symbol) == 6:
        if symbol.startswith(("6", "9")):
            return f"{symbol}.SS"
        return f"{symbol}.SZ"

    return symbol


class YFinanceAdapter(DataAdapter):
    """Data adapter backed by the ``yfinance`` library."""

    # ------------------------------------------------------------------
    # DataAdapter interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "yfinance"

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Fetch daily OHLCV via *yfinance* (run in a thread)."""
        yf_symbol = _to_yf_symbol(symbol)
        period = _days_to_yf_period(days)

        try:
            df = await asyncio.to_thread(self._fetch_history, yf_symbol, period)
        except Exception:
            logger.exception("yfinance fetch failed for %s", yf_symbol)
            return self._empty_df()

        if df is None or df.empty:
            logger.warning("No data returned for %s (period=%s)", yf_symbol, period)
            return self._empty_df()

        return self._normalize(df)

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Return a curated list of liquid stocks for *market*."""
        entries = _STOCK_LISTS.get(market, [])
        return [
            StockInfo(
                symbol=e["symbol"],
                name=e["name"],
                market=market,
                sector=e.get("sector"),
                market_cap=e.get("market_cap"),
                pe_ttm=e.get("pe_ttm"),
            )
            for e in entries
        ]

    async def get_index_data(self, index_symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch index OHLCV (e.g., '^GSPC' for S&P 500, '^IXIC' for NASDAQ).

        Common symbols: ^GSPC (SP500), ^IXIC (NASDAQ), ^DJI (Dow Jones),
        ^HSI (Hang Seng), 000001.SS (SSE Composite).
        """
        try:
            yf_sym = _to_yf_symbol(index_symbol)
            days = (end - start).days + 30
            period = _days_to_yf_period(days)
            df = await asyncio.to_thread(self._fetch_history, yf_sym, period)
            if df is None or df.empty:
                return self._empty_df()
            result = self._normalize(df)
            result["date"] = pd.to_datetime(result["date"])
            mask = (result["date"].dt.date >= start) & (result["date"].dt.date <= end)
            result = result[mask].reset_index(drop=True)
            result["date"] = result["date"].dt.date
            return cast("pd.DataFrame", result)
        except Exception:
            logger.exception("yfinance index fetch failed for %s", index_symbol)
            return self._empty_df()

    async def get_event_context(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> EventContextSeries | None:
        """Fetch news-based event context from yfinance.

        Uses yfinance ``Ticker.news`` to retrieve recent headlines, applies
        simple keyword-based sentiment scoring, and builds date-aligned
        event context records.  Only returns records within ``[start, end]``.
        """
        yf_symbol = _to_yf_symbol(symbol)
        try:
            news_items = await asyncio.to_thread(self._fetch_news, yf_symbol)
        except Exception:
            logger.debug("yfinance news fetch failed for %s", yf_symbol)
            return None

        if not news_items:
            return None

        records = _build_event_records(news_items, start, end)
        if not records:
            return None

        return EventContextSeries(
            symbol=symbol,
            source="yfinance_news",
            records=records,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_history(yf_symbol: str, period: str) -> pd.DataFrame:
        """Synchronous yfinance call (executed via ``asyncio.to_thread``)."""
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "yfinance is required for YFinanceAdapter. "
                "Install with: pip install alphaevo[data-yfinance]"
            ) from err

        ticker = yf.Ticker(yf_symbol)
        return cast("pd.DataFrame", ticker.history(period=period, auto_adjust=True))

    @staticmethod
    def _fetch_news(yf_symbol: str) -> list[dict]:
        """Synchronous yfinance news call (executed via ``asyncio.to_thread``)."""
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError:
            return []

        ticker = yf.Ticker(yf_symbol)
        try:
            news = ticker.news
            return news if isinstance(news, list) else []
        except Exception:
            return []

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise a yfinance ``history`` DataFrame to AlphaEvo format."""
        df = df.copy()
        index = pd.DatetimeIndex(pd.to_datetime(df.index))

        # yfinance may use timezone-aware index; strip tz for simplicity.
        if index.tz is not None:
            index = index.tz_localize(None)

        df.index = index
        df = df.sort_index()

        result = pd.DataFrame(
            {
                "date": [ts.date() for ts in index],
                "open": df["Open"].values,
                "high": df["High"].values,
                "low": df["Low"].values,
                "close": df["Close"].values,
                "volume": df["Volume"].values,
            }
        )

        result["prev_close"] = result["close"].shift(1)

        result = result.reset_index(drop=True)
        return cast("pd.DataFrame", result)

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Return an empty DataFrame with the standard column schema."""
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "prev_close"]
        )


# ---------------------------------------------------------------------------
# News sentiment helpers
# ---------------------------------------------------------------------------

# Simple keyword lists for rule-based sentiment.  Not meant to be exhaustive —
# the goal is to move L3 indicators from "constant fallback" to "noisy but
# directionally useful" provider data.

_NEGATIVE_KEYWORDS: set[str] = {
    "lawsuit",
    "sue",
    "fraud",
    "scandal",
    "crash",
    "plunge",
    "downgrade",
    "recall",
    "investigation",
    "penalty",
    "fine",
    "layoff",
    "cut",
    "loss",
    "decline",
    "warning",
    "risk",
    "debt",
    "default",
    "bankruptcy",
    "miss",
    "disappointing",
    "weak",
    "bearish",
    "sell",
    "short",
    "probe",
    "subpoena",
    "hack",
    "breach",
    "violation",
    "antitrust",
    "tariff",
    "sanction",
    "halt",
}

_POSITIVE_KEYWORDS: set[str] = {
    "beat",
    "upgrade",
    "surge",
    "rally",
    "record",
    "growth",
    "profit",
    "gain",
    "rise",
    "bullish",
    "buy",
    "outperform",
    "strong",
    "boost",
    "innovation",
    "launch",
    "partnership",
    "deal",
    "acquisition",
    "expand",
    "dividend",
    "buyback",
    "exceed",
    "optimistic",
    "breakthrough",
    "approve",
}


def _score_headline(title: str, summary: str) -> float:
    """Return a sentiment score in [0, 1] (0.5 = neutral).

    Combines title (weighted 2x) and summary keywords.
    """
    text = (title + " " + title + " " + summary).lower()
    words = set(text.split())
    neg = len(words & _NEGATIVE_KEYWORDS)
    pos = len(words & _POSITIVE_KEYWORDS)
    total = neg + pos
    if total == 0:
        return 0.5
    raw = (pos - neg) / total  # range: [-1, 1]
    return max(0.0, min(1.0, 0.5 + raw * 0.5))


def _parse_news_date(value: object) -> date | None:
    """Parse yfinance news timestamps across current and legacy schemas."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date()
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return _parse_news_date(float(stripped))
        try:
            return datetime.fromisoformat(stripped.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _extract_news_fields(item: dict) -> tuple[date | None, str, str]:
    """Extract date/title/summary from supported yfinance news item schemas."""
    content = item.get("content")
    if isinstance(content, dict):
        pub_date = _parse_news_date(content.get("pubDate") or content.get("displayTime"))
        title = str(content.get("title") or item.get("title") or "")
        summary = str(content.get("summary") or item.get("summary") or "")
        return pub_date, title, summary

    # Older yfinance releases expose flat dictionaries with providerPublishTime.
    pub_date = _parse_news_date(
        item.get("providerPublishTime") or item.get("pubDate") or item.get("displayTime")
    )
    title = str(item.get("title") or "")
    summary = str(item.get("summary") or "")
    return pub_date, title, summary


def _build_event_records(
    news_items: list[dict],
    start: date,
    end: date,
) -> list[EventContextRecord]:
    """Convert raw yfinance news into date-aligned event context records."""
    from collections import defaultdict

    date_news: dict[date, list[tuple[str, str]]] = defaultdict(list)

    for item in news_items:
        pub_date, title, summary = _extract_news_fields(item)
        if pub_date is None:
            continue

        if pub_date < start or pub_date > end:
            continue
        if not title and not summary:
            continue
        date_news[pub_date].append((title, summary))

    if not date_news:
        return []

    records: list[EventContextRecord] = []
    sorted_dates = sorted(date_news.keys())
    earliest_event = sorted_dates[0]

    for d in sorted_dates:
        headlines = date_news[d]
        # Aggregate sentiment across all headlines on this date
        scores = [_score_headline(t, s) for t, s in headlines]
        avg_sentiment = sum(scores) / len(scores)
        neg_score = max(0.0, min(1.0, 1.0 - avg_sentiment))

        records.append(
            EventContextRecord(
                date=d,
                negative_news_score=round(neg_score, 3),
                news_sentiment_score=round(avg_sentiment, 3),
                days_since_event=(d - earliest_event).days,
            )
        )

    return records
