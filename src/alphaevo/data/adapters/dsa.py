"""DSA (daily_stock_analysis) optional bridge for AlphaEvo.

Bridges AlphaEvo to the ``daily_stock_analysis`` project's
``DataFetcherManager``, reusing its multi-source fallback capability.

This module is intentionally treated as an external enhancement bridge rather
than a core built-in dependency of AlphaEvo.

Install requirements:

1. Clone or install daily_stock_analysis separately.
2. Set the environment variable ``ALPHAEVO_DSA_PATH`` to the project root.
3. Use the ``dsa`` adapter::

       ALPHAEVO_DATA_ADAPTER=dsa alphaevo run <strategy>

This adapter isolates AlphaEvo from DSA internals through the
:class:`DataAdapter` interface — no direct imports leak outside this file.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from alphaevo.data.adapter import DataAdapter
from alphaevo.models.enums import MarketRegime, MarketType
from alphaevo.models.market import (
    EventContextRecord,
    EventContextSeries,
    MarketContext,
    SectorInfo,
    StockInfo,
)

logger = logging.getLogger(__name__)

_NEGATIVE_NEWS_KEYWORDS = (
    "减持",
    "亏损",
    "预亏",
    "下滑",
    "下修",
    "立案",
    "监管",
    "处罚",
    "诉讼",
    "仲裁",
    "风险",
    "违规",
    "问询",
    "退市",
    "暴雷",
    "跌停",
    "质押",
    "冻结",
    "债务",
    "裁员",
)
_POSITIVE_NEWS_KEYWORDS = (
    "增持",
    "回购",
    "盈利",
    "预增",
    "增长",
    "中标",
    "签约",
    "订单",
    "突破",
    "创新高",
    "涨停",
    "利好",
    "获批",
)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _to_ratio(value: Any) -> float | None:
    """Convert provider percentage values to decimal ratios when needed."""
    number = _to_float(value)
    if number is None:
        return None
    return number / 100 if abs(number) > 1 else number


def _get_field(payload: Any, *keys: str) -> Any:
    """Read the first non-empty field from dict/object provider payloads."""
    for key in keys:
        value = payload.get(key) if isinstance(payload, dict) else getattr(payload, key, None)
        if value not in (None, "", "-", "--"):
            return value
    return None


def _records_from_payload(raw: Any) -> list[dict[str, Any]]:
    """Normalize DSA list/DataFrame payloads to row dictionaries."""
    if raw is None:
        return []
    if isinstance(raw, pd.DataFrame):
        rows = cast("list[dict[Any, Any]]", raw.to_dict("records"))
        return [{str(key): value for key, value in row.items()} for row in rows]
    if isinstance(raw, dict):
        nested = raw.get("data") or raw.get("items") or raw.get("records")
        if nested is not None:
            return _records_from_payload(nested)
        return [{str(key): value for key, value in raw.items()}]
    if isinstance(raw, tuple) and raw:
        return _records_from_payload(raw[0])
    if isinstance(raw, list):
        records: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                records.append(dict(item))
                continue
            if hasattr(item, "to_dict"):
                try:
                    converted = item.to_dict()
                    if isinstance(converted, dict):
                        records.append(converted)
                        continue
                except Exception:
                    pass
            attrs = getattr(item, "__dict__", None)
            if isinstance(attrs, dict):
                records.append({k: v for k, v in attrs.items() if not k.startswith("_")})
            else:
                records.append({})
        return records
    return []


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _load_dsa(dsa_path: str | None = None) -> Any:
    """Import the DSA DataFetcherManager, injecting *dsa_path* into sys.path if needed."""
    import os

    path = dsa_path or os.getenv("ALPHAEVO_DSA_PATH", "")
    if not path:
        raise ImportError(
            "ALPHAEVO_DSA_PATH environment variable is required for DSAAdapter. "
            "Point it to the root of your daily_stock_analysis clone."
        )

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ImportError(f"DSA path does not exist: {resolved}")

    path_entries = [str(resolved)]
    if (resolved / "src").is_dir():
        path_entries.append(str(resolved / "src"))
    added_paths: list[str] = []
    for entry in reversed(path_entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)
            added_paths.append(entry)

    candidates = (
        "data_provider",
        "data_provider.base",
        "data_fetcher.manager",  # legacy pre-refactor path
    )
    errors: list[str] = []
    for module_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            manager_cls = getattr(mod, "DataFetcherManager", None)
            if manager_cls is not None:
                return manager_cls
            errors.append(f"{module_name}: DataFetcherManager missing")
        except (ImportError, ModuleNotFoundError) as err:
            errors.append(f"{module_name}: {err}")

    # Clean up sys.path on failure
    for entry in added_paths:
        if entry in sys.path:
            sys.path.remove(entry)
    detail = "; ".join(errors)
    raise ImportError(
        f"Cannot import DataFetcherManager from {resolved}. "
        f"Tried {', '.join(candidates)}. Detail: {detail}"
    )


class DSAAdapter(DataAdapter):
    """Data adapter that delegates to daily_stock_analysis's DataFetcherManager."""

    def __init__(self, dsa_path: str | None = None) -> None:
        manager_cls = _load_dsa(dsa_path)
        self._manager = manager_cls()
        self._sector_rankings_cache: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None

    @property
    def name(self) -> str:  # noqa: D401
        return "dsa"

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Fetch daily OHLCV via DSA's multi-source fetcher."""
        try:
            raw = await asyncio.to_thread(self._fetch_daily_raw, symbol, days)
            df = self._extract_dataframe(raw)
        except Exception:
            logger.exception("DSA fetch failed for %s", symbol)
            return self._empty_df()

        if df is None or df.empty:
            logger.warning("No data returned from DSA for %s", symbol)
            return self._empty_df()

        return self._normalize(df)

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Delegate stock list retrieval to DSA."""
        if market != MarketType.A_SHARE:
            return []

        try:
            raw = await asyncio.to_thread(self._fetch_stock_list_raw)
            records = _records_from_payload(raw)
            if not records:
                return []

            stocks: list[StockInfo] = []
            for item in records[:500]:
                code = str(_get_field(item, "code", "symbol", "代码", "股票代码") or "").strip()
                name = str(_get_field(item, "name", "名称", "股票名称") or "").strip()
                if code:
                    stocks.append(
                        StockInfo(
                            symbol=code,
                            name=name,
                            market=MarketType.A_SHARE,
                            sector=_get_field(item, "sector", "industry", "所属行业", "板块"),
                            market_cap=_to_float(
                                _get_field(item, "market_cap", "总市值", "总市值-元")
                            ),
                            pe_ttm=_to_float(_get_field(item, "pe_ttm", "市盈率", "市盈率-动态")),
                        )
                    )
            return stocks
        except Exception:
            logger.exception("DSA stock list fetch failed")
            return []

    async def get_sector_data(self, symbol: str) -> SectorInfo | None:
        """Fetch DSA board membership and best-effort sector ranking context."""
        try:
            boards = await asyncio.to_thread(self._fetch_belong_boards, symbol)
            top, bottom = await asyncio.to_thread(self._fetch_sector_rankings)
        except Exception:
            logger.debug("DSA sector context fetch failed for %s", symbol, exc_info=True)
            return None

        sector_name = self._select_sector_name(boards)
        matched, rank, in_bottom = self._match_sector_rank(sector_name, top, bottom)
        if not sector_name and matched is not None:
            sector_name = self._sector_name(matched) or ""
        if not sector_name:
            return None

        change_pct = _to_ratio(_get_field(matched or {}, "change_pct", "涨跌幅", "涨幅")) or 0.0
        net_inflow = _to_float(
            _get_field(matched or {}, "net_inflow", "主力净流入", "资金净流入", "净流入")
        )
        rising_days = _to_float(_get_field(matched or {}, "rising_days", "连涨天数"))
        top_stocks = self._top_stocks(matched or {})

        return SectorInfo(
            name=sector_name,
            change_pct=change_pct,
            heat_rank=rank,
            rising_days=int(rising_days) if rising_days is not None else None,
            net_inflow=net_inflow,
            risk_flag=bool(in_bottom),
            top_stocks=top_stocks,
        )

    async def get_event_context(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> EventContextSeries | None:
        """Load cached DSA news intelligence as date-aligned event context.

        This intentionally reads DSA's local news cache instead of doing live web
        search during backtests. If the cache has no dated news in the requested
        window, AlphaEvo falls back to its price/volume event proxy.
        """
        try:
            news_items = await asyncio.to_thread(self._fetch_recent_news, symbol, start, end)
        except Exception:
            logger.debug("DSA news context fetch failed for %s", symbol, exc_info=True)
            return None

        records_by_date: dict[date, list[tuple[float, float]]] = {}
        for item in news_items:
            published_at = _get_field(item, "published_date", "published_at", "fetched_at")
            event_date = self._coerce_date(published_at)
            if event_date is None or event_date < start or event_date > end:
                continue
            negative, sentiment = self._score_news_item(item)
            records_by_date.setdefault(event_date, []).append((negative, sentiment))

        if not records_by_date:
            return None

        records = [
            EventContextRecord(
                date=event_date,
                negative_news_score=round(max(v[0] for v in values), 4),
                news_sentiment_score=round(sum(v[1] for v in values) / len(values), 4),
                days_since_event=0,
            )
            for event_date, values in sorted(records_by_date.items())
        ]
        return EventContextSeries(symbol=symbol, source="dsa_news_cache", records=records)

    async def get_market_context(self, market: MarketType) -> MarketContext | None:
        """Fetch DSA market breadth, index move, and sector leader/laggard context."""
        if market != MarketType.A_SHARE:
            return None

        try:
            stats, indices, rankings = await asyncio.to_thread(self._fetch_market_payload)
        except Exception:
            logger.debug("DSA market context fetch failed", exc_info=True)
            return None

        if not stats and not indices and not rankings:
            return None

        index_change_pct = self._index_change_pct(indices)
        up_count = _to_float(_get_field(stats, "up_count", "上涨家数", "rise_count"))
        down_count = _to_float(_get_field(stats, "down_count", "下跌家数", "fall_count"))
        flat_count = _to_float(_get_field(stats, "flat_count", "平盘家数"))
        limit_up = _to_float(_get_field(stats, "limit_up_count", "涨停家数"))
        limit_down = _to_float(_get_field(stats, "limit_down_count", "跌停家数"))

        breadth = None
        active = (up_count or 0) + (down_count or 0)
        if active > 0:
            breadth = (up_count or 0) / active

        sentiment_index = None
        total = active + (flat_count or 0)
        if breadth is not None:
            limit_pressure = ((limit_up or 0) - (limit_down or 0)) / total if total > 0 else 0.0
            sentiment_index = round(_clamp(0.5 + (breadth - 0.5) * 0.8 + limit_pressure * 2), 4)

        top, bottom = rankings if rankings else ([], [])
        return MarketContext(
            index_change_pct=index_change_pct,
            breadth=round(breadth, 4) if breadth is not None else None,
            regime=self._infer_market_regime(index_change_pct, breadth),
            sentiment_index=sentiment_index,
            sector_leaders=[name for item in top[:5] if (name := self._sector_name(item))],
            sector_laggards=[name for item in bottom[:5] if (name := self._sector_name(item))],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_daily_raw(self, symbol: str, days: int) -> Any:
        if hasattr(self._manager, "get_daily_data"):
            return self._manager.get_daily_data(stock_code=symbol, days=days)
        if hasattr(self._manager, "fetch_daily"):
            return self._manager.fetch_daily(symbol, days)
        raise AttributeError("DSA DataFetcherManager has no get_daily_data/fetch_daily method")

    @staticmethod
    def _extract_dataframe(raw: Any) -> pd.DataFrame | None:
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, tuple) and raw and isinstance(raw[0], pd.DataFrame):
            return raw[0]
        return None

    def _fetch_stock_list_raw(self) -> Any:
        if hasattr(self._manager, "get_stock_list"):
            return self._manager.get_stock_list()

        fetchers = []
        if hasattr(self._manager, "_get_fetchers_snapshot"):
            fetchers = list(self._manager._get_fetchers_snapshot())  # noqa: SLF001
        else:
            fetchers = list(getattr(self._manager, "_fetchers", []) or [])

        for fetcher in fetchers:
            if not hasattr(fetcher, "get_stock_list"):
                continue
            data = fetcher.get_stock_list()
            if data is not None and not (isinstance(data, pd.DataFrame) and data.empty):
                return data
        return []

    def _fetch_belong_boards(self, symbol: str) -> list[dict[str, Any]]:
        if not hasattr(self._manager, "get_belong_boards"):
            return []
        return _records_from_payload(self._manager.get_belong_boards(symbol))

    def _fetch_sector_rankings(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        cached = cast(
            "tuple[list[dict[str, Any]], list[dict[str, Any]]] | None",
            getattr(self, "_sector_rankings_cache", None),
        )
        if cached is not None:
            return cached
        if not hasattr(self._manager, "get_sector_rankings"):
            return ([], [])
        raw = self._manager.get_sector_rankings(n=50)
        rankings: tuple[list[dict[str, Any]], list[dict[str, Any]]]
        if not isinstance(raw, tuple) or len(raw) < 2:
            rankings = ([], [])
        else:
            rankings = (
                _records_from_payload(raw[0]),
                _records_from_payload(raw[1]),
            )
        self._sector_rankings_cache = rankings
        return rankings

    def _fetch_recent_news(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        try:
            storage = importlib.import_module("src.storage")
            db = storage.get_db()
        except (ImportError, ModuleNotFoundError, AttributeError):
            return []

        # DSA's DB method filters by fetched_at. Cover the requested range, but
        # cap the scan so a multi-year backtest does not pull the entire DB.
        today = date.today()
        oldest = min(start, end)
        days = max(7, min(370, (today - oldest).days + 7))
        raw = db.get_recent_news(code=symbol, days=days, limit=100)
        return _records_from_payload(raw)

    def _fetch_market_payload(
        self,
    ) -> tuple[
        dict[str, Any], list[dict[str, Any]], tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ]:
        stats = (
            self._manager.get_market_stats() if hasattr(self._manager, "get_market_stats") else {}
        )
        indices = (
            self._manager.get_main_indices(region="cn")
            if hasattr(self._manager, "get_main_indices")
            else []
        )
        rankings = self._fetch_sector_rankings()
        return (
            stats if isinstance(stats, dict) else {},
            _records_from_payload(indices),
            rankings,
        )

    @staticmethod
    def _select_sector_name(boards: list[dict[str, Any]]) -> str:
        for board in boards:
            board_type = str(_get_field(board, "type", "board_type", "板块类型") or "")
            name = _get_field(board, "name", "board_name", "板块名称", "industry")
            if name and any(token in board_type for token in ("行业", "industry", "INDUSTRY")):
                return str(name).strip()
        for board in boards:
            name = _get_field(board, "name", "board_name", "板块名称", "industry")
            if name:
                return str(name).strip()
        return ""

    @staticmethod
    def _sector_name(item: dict[str, Any]) -> str:
        value = _get_field(item, "name", "sector", "板块名称", "行业", "板块")
        return str(value).strip() if value else ""

    def _match_sector_rank(
        self,
        sector_name: str,
        top: list[dict[str, Any]],
        bottom: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, int | None, bool]:
        if not sector_name:
            return (None, None, False)
        normalized = sector_name.lower()
        for idx, item in enumerate(top, start=1):
            name = self._sector_name(item)
            if name and (name.lower() == normalized or name in sector_name or sector_name in name):
                return (item, idx, False)
        offset = len(top)
        for idx, item in enumerate(bottom, start=1):
            name = self._sector_name(item)
            if name and (name.lower() == normalized or name in sector_name or sector_name in name):
                return (item, offset + idx, True)
        return (None, None, False)

    @staticmethod
    def _top_stocks(item: dict[str, Any]) -> list[str]:
        raw = _get_field(item, "top_stocks", "领涨股", "leader_stocks")
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(x) for x in raw if x]
        return [part.strip() for part in str(raw).replace("，", ",").split(",") if part.strip()]

    @staticmethod
    def _coerce_date(value: Any) -> date | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            parsed = pd.to_datetime(value)
            parsed_date = parsed.date()
            return parsed_date if isinstance(parsed_date, date) else None
        except Exception:
            return None

    @staticmethod
    def _score_news_item(item: dict[str, Any]) -> tuple[float, float]:
        text = " ".join(
            str(_get_field(item, key) or "")
            for key in ("title", "snippet", "summary", "analysis_summary", "dimension")
        )
        neg = sum(1 for keyword in _NEGATIVE_NEWS_KEYWORDS if keyword in text)
        pos = sum(1 for keyword in _POSITIVE_NEWS_KEYWORDS if keyword in text)
        negative_score = _clamp(0.15 + neg * 0.2 - pos * 0.05) if neg else _clamp(0.10 - pos * 0.03)
        sentiment = _clamp(0.5 + pos * 0.12 - neg * 0.15)
        return (round(negative_score, 4), round(sentiment, 4))

    @staticmethod
    def _index_change_pct(indices: list[dict[str, Any]]) -> float | None:
        preferred = None
        for item in indices:
            code = str(_get_field(item, "code", "symbol") or "").lower()
            name = str(_get_field(item, "name", "名称") or "")
            if code in {"sh000001", "000001", "000001.sh"} or "上证" in name:
                preferred = item
                break
        if preferred is None and indices:
            preferred = indices[0]
        if preferred is None:
            return None
        return _to_ratio(_get_field(preferred, "change_pct", "pct_chg", "涨跌幅", "涨幅"))

    @staticmethod
    def _infer_market_regime(
        index_change_pct: float | None,
        breadth: float | None,
    ) -> MarketRegime | None:
        if index_change_pct is None and breadth is None:
            return None
        move = index_change_pct or 0.0
        width = breadth if breadth is not None else 0.5
        if move <= -0.025 and width <= 0.25:
            return MarketRegime.PANIC
        if move >= 0.02 and width >= 0.75:
            return MarketRegime.EUPHORIA
        if move >= 0.008 and width >= 0.55:
            return MarketRegime.TRENDING_UP
        if move <= -0.008 and width <= 0.45:
            return MarketRegime.TRENDING_DOWN
        if abs(move) >= 0.018:
            return MarketRegime.VOLATILE
        return MarketRegime.RANGE_BOUND

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Map DSA output to AlphaEvo's standard OHLCV schema."""
        df = df.copy()

        col_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Amount": "amount",
        }

        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ["date", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                if col == "volume":
                    df[col] = 0
                else:
                    return DSAAdapter._empty_df()

        for col in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.dropna(subset=["date", "open", "high", "low", "close"])
        df = df.sort_values("date").reset_index(drop=True)
        df["prev_close"] = df["close"].shift(1)

        return cast(
            "pd.DataFrame",
            df[["date", "open", "high", "low", "close", "volume", "prev_close"]].copy(),
        )

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Return an empty DataFrame with the standard column schema."""
        return cast(
            "pd.DataFrame",
            pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "prev_close"]),
        )
