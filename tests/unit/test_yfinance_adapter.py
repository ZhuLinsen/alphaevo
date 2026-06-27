"""Unit tests for YFinanceAdapter (all yfinance calls are mocked)."""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import patch

import pandas as pd

from alphaevo.data.adapters.yfinance import (
    YFinanceAdapter,
    _build_event_records,
    _days_to_yf_period,
    _to_yf_symbol,
)
from alphaevo.models.enums import MarketType
from alphaevo.models.market import StockInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_yf_history(rows: int = 5) -> pd.DataFrame:
    """Build a DataFrame that mimics ``yfinance.Ticker.history()`` output."""
    dates = pd.date_range("2024-01-02", periods=rows, freq="B", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(rows)],
            "High": [105.0 + i for i in range(rows)],
            "Low": [99.0 + i for i in range(rows)],
            "Close": [102.0 + i for i in range(rows)],
            "Volume": [1_000_000 + i * 100_000 for i in range(rows)],
            "Dividends": [0.0] * rows,
            "Stock Splits": [0.0] * rows,
        },
        index=dates,
    )


def _run(coro):
    """Convenience wrapper so tests stay synchronous."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests – basic properties
# ---------------------------------------------------------------------------


class TestAdapterProperties:
    def test_name(self):
        adapter = YFinanceAdapter()
        assert adapter.name == "yfinance"


# ---------------------------------------------------------------------------
# Tests – get_daily_data
# ---------------------------------------------------------------------------


class TestGetDailyData:
    @patch("alphaevo.data.adapters.yfinance.YFinanceAdapter._fetch_history")
    def test_returns_proper_columns(self, mock_fetch):
        mock_fetch.return_value = _make_yf_history(10)
        adapter = YFinanceAdapter()
        df = _run(adapter.get_daily_data("AAPL", days=30))

        expected_cols = {"date", "open", "high", "low", "close", "volume", "prev_close"}
        assert set(df.columns) == expected_cols

    @patch("alphaevo.data.adapters.yfinance.YFinanceAdapter._fetch_history")
    def test_prev_close_is_shifted(self, mock_fetch):
        mock_fetch.return_value = _make_yf_history(5)
        adapter = YFinanceAdapter()
        df = _run(adapter.get_daily_data("AAPL"))

        assert pd.isna(df["prev_close"].iloc[0])
        assert df["prev_close"].iloc[1] == df["close"].iloc[0]

    @patch("alphaevo.data.adapters.yfinance.YFinanceAdapter._fetch_history")
    def test_sorted_by_date_ascending(self, mock_fetch):
        mock_fetch.return_value = _make_yf_history(10)
        adapter = YFinanceAdapter()
        df = _run(adapter.get_daily_data("AAPL"))

        dates = list(df["date"])
        assert dates == sorted(dates)

    @patch("alphaevo.data.adapters.yfinance.YFinanceAdapter._fetch_history")
    def test_empty_df_on_no_data(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame()
        adapter = YFinanceAdapter()
        df = _run(adapter.get_daily_data("INVALID"))

        assert df.empty
        expected_cols = {"date", "open", "high", "low", "close", "volume", "prev_close"}
        assert set(df.columns) == expected_cols

    @patch("alphaevo.data.adapters.yfinance.YFinanceAdapter._fetch_history")
    def test_empty_df_on_exception(self, mock_fetch):
        mock_fetch.side_effect = RuntimeError("network error")
        adapter = YFinanceAdapter()
        df = _run(adapter.get_daily_data("AAPL"))

        assert df.empty

    @patch("alphaevo.data.adapters.yfinance.YFinanceAdapter._fetch_history")
    def test_timezone_stripped(self, mock_fetch):
        """Timezone-aware yfinance index should be stripped in output."""
        mock_fetch.return_value = _make_yf_history(3)
        adapter = YFinanceAdapter()
        df = _run(adapter.get_daily_data("AAPL"))

        # date column should contain plain ``datetime.date`` objects
        assert isinstance(df["date"].iloc[0], date)


# ---------------------------------------------------------------------------
# Tests – get_stock_list
# ---------------------------------------------------------------------------


class TestGetStockList:
    def test_us_stocks_return_stock_info(self):
        adapter = YFinanceAdapter()
        stocks = _run(adapter.get_stock_list(MarketType.US))

        assert len(stocks) > 0
        assert all(isinstance(s, StockInfo) for s in stocks)
        assert all(s.market == MarketType.US for s in stocks)

    def test_a_share_stocks(self):
        adapter = YFinanceAdapter()
        stocks = _run(adapter.get_stock_list(MarketType.A_SHARE))

        assert len(stocks) >= 60
        symbols = [s.symbol for s in stocks]
        assert any(s.endswith(".SZ") or s.endswith(".SS") for s in symbols)

    def test_a_share_curated_universe_has_sampler_metadata(self):
        adapter = YFinanceAdapter()
        stocks = _run(adapter.get_stock_list(MarketType.A_SHARE))

        assert len({s.sector for s in stocks if s.sector}) >= 12
        assert sum(s.market_cap is not None for s in stocks) >= 60

    def test_hk_stocks(self):
        adapter = YFinanceAdapter()
        stocks = _run(adapter.get_stock_list(MarketType.HK))

        assert len(stocks) > 0
        assert all(s.market == MarketType.HK for s in stocks)

    def test_stock_info_has_sector(self):
        adapter = YFinanceAdapter()
        stocks = _run(adapter.get_stock_list(MarketType.US))
        assert all(s.sector is not None for s in stocks)


# ---------------------------------------------------------------------------
# Tests – symbol conversion
# ---------------------------------------------------------------------------


class TestSymbolConversion:
    def test_passthrough_with_suffix(self):
        assert _to_yf_symbol("600519.SS") == "600519.SS"

    def test_shanghai_prefix(self):
        assert _to_yf_symbol("600519") == "600519.SS"

    def test_shenzhen_prefix(self):
        assert _to_yf_symbol("000001") == "000001.SZ"

    def test_us_symbol_passthrough(self):
        assert _to_yf_symbol("AAPL") == "AAPL"


# ---------------------------------------------------------------------------
# Tests – period mapping
# ---------------------------------------------------------------------------


class TestPeriodMapping:
    def test_short_period(self):
        assert _days_to_yf_period(3) == "5d"

    def test_one_month(self):
        assert _days_to_yf_period(20) == "1mo"

    def test_three_months(self):
        assert _days_to_yf_period(60) == "3mo"

    def test_six_months(self):
        assert _days_to_yf_period(120) == "6mo"

    def test_one_year(self):
        assert _days_to_yf_period(300) == "1y"

    def test_very_long(self):
        assert _days_to_yf_period(5000) == "max"


# ---------------------------------------------------------------------------
# Tests – event/news context
# ---------------------------------------------------------------------------


class TestEventContext:
    def test_build_event_records_supports_current_content_schema(self):
        records = _build_event_records(
            [
                {
                    "content": {
                        "pubDate": "2026-06-20T01:23:45Z",
                        "title": "Company beats profit expectations",
                        "summary": "Strong growth and optimistic guidance",
                    }
                }
            ],
            date(2026, 6, 1),
            date(2026, 6, 30),
        )

        assert len(records) == 1
        assert records[0].date == date(2026, 6, 20)
        assert records[0].news_sentiment_score is not None
        assert records[0].news_sentiment_score > 0.5
        assert records[0].negative_news_score is not None
        assert records[0].negative_news_score < 0.5

    def test_build_event_records_supports_legacy_provider_publish_time_schema(self):
        records = _build_event_records(
            [
                {
                    "providerPublishTime": 1_781_913_600,
                    "title": "Company faces fraud investigation and lawsuit",
                }
            ],
            date(2026, 6, 1),
            date(2026, 6, 30),
        )

        assert len(records) == 1
        assert records[0].date == date(2026, 6, 20)
        assert records[0].news_sentiment_score is not None
        assert records[0].news_sentiment_score < 0.5
        assert records[0].negative_news_score is not None
        assert records[0].negative_news_score > 0.5

    @patch("alphaevo.data.adapters.yfinance.YFinanceAdapter._fetch_news")
    def test_get_event_context_returns_provider_series_for_legacy_yfinance_news(
        self,
        mock_fetch,
    ):
        mock_fetch.return_value = [
            {
                "providerPublishTime": 1_781_913_600,
                "title": "A-share company reports strong profit growth",
            }
        ]

        context = _run(
            YFinanceAdapter().get_event_context(
                "600519.SS",
                date(2026, 6, 1),
                date(2026, 6, 30),
            )
        )

        assert context is not None
        assert context.symbol == "600519.SS"
        assert context.source == "yfinance_news"
        assert len(context.records) == 1
