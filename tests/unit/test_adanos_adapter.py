"""Unit tests for the optional Adanos sentiment adapter."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any
from unittest.mock import patch

from alphaevo.data.adapters.adanos import AdanosSentimentAdapter
from alphaevo.models.enums import MarketType


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _run(coro):
    return asyncio.run(coro)


def test_get_event_context_fetches_news_sentiment_for_us_symbol():
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, dict(request.header_items()), timeout))
        return _FakeResponse({"sentiment_score": 0.6})

    adapter = AdanosSentimentAdapter(
        "test-key",
        base_url="https://api.example.test",
        sources=("news",),
        timeout=3,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        context = _run(
            adapter.get_event_context(
                "aapl",
                date(2026, 5, 1),
                date(2026, 5, 10),
            )
        )

    assert context is not None
    assert context.symbol == "AAPL"
    assert context.source == "adanos"
    assert context.records[0].date == date(2026, 5, 10)
    assert context.records[0].news_sentiment_score == 0.8
    assert context.records[0].negative_news_score == 0.2
    assert calls == [
        (
            "https://api.example.test/news/stocks/v1/stock/AAPL?days=10",
            {"X-api-key": "test-key"},
            3,
        )
    ]


def test_get_event_context_keeps_bullish_score_on_unit_scale():
    def fake_urlopen(request, timeout):
        return _FakeResponse({"bullish_score": 0.7, "bearish_score": 25})

    adapter = AdanosSentimentAdapter(
        "test-key",
        base_url="https://api.example.test",
        sources=("reddit",),
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        context = _run(
            adapter.get_event_context(
                "MSFT",
                date(2026, 5, 1),
                date(2026, 5, 1),
            )
        )

    assert context is not None
    assert context.records[0].news_sentiment_score == 0.7
    assert context.records[0].negative_news_score == 0.25


def test_get_event_context_skips_non_us_symbols():
    adapter = AdanosSentimentAdapter("test-key", sources=("news",))

    with patch("urllib.request.urlopen") as urlopen:
        context = _run(
            adapter.get_event_context(
                "000001.SZ",
                date(2026, 5, 1),
                date(2026, 5, 10),
            )
        )

    assert context is None
    urlopen.assert_not_called()


def test_get_market_context_returns_us_sentiment_index_only():
    def fake_urlopen(request, timeout):
        return _FakeResponse({"sentiment_score": 75})

    adapter = AdanosSentimentAdapter("test-key", sources=("reddit",))

    with patch("urllib.request.urlopen", fake_urlopen):
        context = _run(adapter.get_market_context(MarketType.US))
        unsupported = _run(adapter.get_market_context(MarketType.A_SHARE))

    assert context is not None
    assert context.sentiment_index == 0.75
    assert unsupported is None
