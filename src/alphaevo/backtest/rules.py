"""Market rule checker — enforces market-specific trading rules.

Handles A-share T+1, limit-up/down, and suspension rules.
Extensible to HK/US markets via MarketRuleConfig.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from alphaevo.models.strategy import MarketRuleConfig

if TYPE_CHECKING:
    import pandas as pd


class MarketRuleChecker:
    """Check market-specific trading rules (T+1, limit-up/down, suspension)."""

    def can_buy(
        self,
        df: pd.DataFrame,
        idx: int,
        rules: MarketRuleConfig,
        *,
        symbol: str = "",
        stock_name: str = "",
        is_st: bool | None = None,
    ) -> bool:
        """Check if buying is allowed on this bar."""
        if not rules:
            return True

        row = df.iloc[idx]

        # Limit-up: can't buy at limit-up (price already ceiling)
        if rules.limit_up_down and self._is_limit_up(
            row, symbol=symbol, stock_name=stock_name, is_st=is_st
        ):
            return False

        # Suspension: volume == 0 means suspended
        return not (rules.suspension and _get(row, "volume", 0) == 0)

    def can_sell(
        self,
        df: pd.DataFrame,
        idx: int,
        entry_idx: int,
        rules: MarketRuleConfig,
        *,
        symbol: str = "",
        stock_name: str = "",
        is_st: bool | None = None,
    ) -> bool:
        """Check if selling is allowed on this bar."""
        if not rules:
            return True

        row = df.iloc[idx]

        # T+1: cannot sell on the same day as purchase
        if rules.t_plus_1 and idx <= entry_idx:
            return False

        # Limit-down: can't sell at limit-down
        if rules.limit_up_down and self._is_limit_down(
            row, symbol=symbol, stock_name=stock_name, is_st=is_st
        ):
            return False

        # Suspension
        return not (rules.suspension and _get(row, "volume", 0) == 0)

    @staticmethod
    def _is_limit_up(
        row: Any,
        *,
        symbol: str = "",
        stock_name: str = "",
        is_st: bool | None = None,
    ) -> bool:
        """A-share limit-up with board-specific thresholds.

        - Main board: ±10%  (threshold 9.8%)
        - STAR (688xxx) / ChiNext (300xxx): ±20%  (threshold 19.6%)
        - BSE (8xxxxx / 43xxxx): ±30%  (threshold 29.4%)
        - ST stocks: ±5%  (threshold 4.8%)
        """
        prev = _get(row, "prev_close", 0)
        if not prev or prev <= 0:
            return False
        change = (row["close"] - prev) / prev
        threshold = _limit_threshold(row, symbol=symbol, stock_name=stock_name, is_st=is_st)
        return bool(change >= threshold)

    @staticmethod
    def _is_limit_down(
        row: Any,
        *,
        symbol: str = "",
        stock_name: str = "",
        is_st: bool | None = None,
    ) -> bool:
        """A-share limit-down with board-specific thresholds."""
        prev = _get(row, "prev_close", 0)
        if not prev or prev <= 0:
            return False
        change = (row["close"] - prev) / prev
        threshold = _limit_threshold(row, symbol=symbol, stock_name=stock_name, is_st=is_st)
        return bool(change <= -threshold)

    @classmethod
    def default_rules(cls, market: str) -> MarketRuleConfig:
        """Factory: return default rules for a market."""
        defaults = {
            "a_share": MarketRuleConfig(t_plus_1=True, limit_up_down=True, suspension=True),
            "hk": MarketRuleConfig(t_plus_1=False, limit_up_down=False, suspension=True),
            "us": MarketRuleConfig(t_plus_1=False, limit_up_down=False, suspension=True),
        }
        return defaults.get(market, MarketRuleConfig())


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a row (Series or dict-like)."""
    try:
        val = row[key] if key in row.index else default
        return val if val is not None else default
    except (KeyError, AttributeError, TypeError):
        return default


def _limit_threshold(
    row: Any,
    *,
    symbol: str = "",
    stock_name: str = "",
    is_st: bool | None = None,
) -> float:
    """Return the limit-up/down threshold for a stock based on its code.

    Board identification (A-share conventions):
    - 688xxx  → STAR Market (科创板): 20%
    - 300xxx  → ChiNext (创业板): 20%
    - 8xxxxx / 43xxxx → BSE (北交所): 30%
    - ST stocks: 5%  (detected by name containing 'ST' or the ``st``
      column/flag when present in the DataFrame row)
    - Otherwise → Main board: 10%

    A small tolerance (0.2%) is applied to handle rounding.
    """
    raw_symbol = symbol or str(_get(row, "symbol", "") or _get(row, "code", ""))
    name = stock_name or str(_get(row, "name", ""))

    # Strip exchange suffixes (.SZ/.SH/.SS/.BJ) and prefixes (sh/sz/bj)
    code = raw_symbol.strip().split(".")[0]
    code = re.sub(r"^(?:sh|sz|bj)", "", code, flags=re.IGNORECASE)

    # ST detection: check explicit flag first, then row flag, then name string
    if is_st is True:
        return 0.048
    st_flag = _get(row, "st", None)
    if st_flag is not None and bool(st_flag):
        return 0.048  # 5% with tolerance

    if name:
        upper_name = name.upper()
        # Match "*ST", "ST", but avoid false positives like "STRONG", "FAST"
        if "ST " in upper_name or upper_name.startswith("ST") or "*ST" in upper_name:
            return 0.048  # 5% with tolerance

    if code.startswith(("688", "300")):
        return 0.196  # 20% with tolerance

    if code.startswith(("8", "43")):
        return 0.294  # 30% with tolerance

    return 0.098  # 10% with tolerance (main board default)
