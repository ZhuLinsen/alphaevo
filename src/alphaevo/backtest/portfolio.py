"""Portfolio-level backtester — wraps signal-level engine with position sizing.

Adds:
- Initial capital and per-trade allocation
- Maximum concurrent positions
- Portfolio equity curve and drawdown
- Portfolio-level Sharpe ratio
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from alphaevo.models.execution import BacktestResult, TradeSignal

if TYPE_CHECKING:
    import pandas as pd


class PortfolioConfig(BaseModel):
    """Configuration for portfolio-level backtesting."""

    initial_capital: float = Field(default=100_000.0, gt=0)
    max_positions: int = Field(default=5, ge=1)
    position_size_pct: float = Field(default=0.2, gt=0, le=1.0)
    risk_per_trade_pct: float = Field(default=0.02, gt=0, le=0.5)


@dataclass
class PortfolioTrade:
    """A trade with portfolio-level sizing applied."""

    signal: TradeSignal
    shares: float
    capital_allocated: float
    pnl: float
    pnl_pct: float


@dataclass
class PortfolioSnapshot:
    """Daily portfolio snapshot for equity curve."""

    date: date
    equity: float
    cash: float
    positions_open: int
    daily_return: float = 0.0


@dataclass
class PortfolioResult:
    """Portfolio-level backtest result."""

    config: PortfolioConfig
    trades: list[PortfolioTrade] = field(default_factory=list)
    equity_curve: list[PortfolioSnapshot] = field(default_factory=list)
    final_equity: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    max_concurrent_positions: int = 0
    total_trades: int = 0
    capital_utilization: float = 0.0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.pnl > 0)
        return winners / len(self.trades)


class PortfolioBacktester:
    """Wraps signal-level BacktestResult with portfolio-level position sizing.

    Takes the output of BacktestEngine.run() and simulates portfolio-level
    execution with capital constraints and position limits.
    """

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()

    def simulate(
        self,
        backtest_result: BacktestResult,
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> PortfolioResult:
        """Run portfolio simulation on signal-level backtest results.

        Processes signals chronologically, applying position sizing and
        capital constraints. Signals that exceed the max_positions limit
        or available capital are skipped. When *market_data* is provided,
        positions are marked to market on real daily closes.
        """
        cfg = self.config
        cash = cfg.initial_capital
        equity = cfg.initial_capital

        close_map = _build_close_map(market_data)

        # Include closed trades AND unclosed terminal holds that have daily MTM data
        signals = sorted(
            [
                s
                for s in backtest_result.signals
                if s.exit_price is not None or (close_map and s.symbol in close_map)
            ],
            key=lambda s: (s.signal_date, s.symbol),
        )

        if not signals:
            return PortfolioResult(
                config=cfg,
                final_equity=cfg.initial_capital,
            )

        portfolio_trades: list[PortfolioTrade] = []
        open_positions: list[tuple[TradeSignal, float, float]] = []  # (signal, shares, allocated)
        equity_curve: list[PortfolioSnapshot] = []
        max_concurrent = 0
        total_capital_util = 0.0
        total_days = 0

        event_dates: set[date] = {s.signal_date for s in signals}
        for s in signals:
            ex = _resolve_exit_date(s)
            if ex is not None:
                event_dates.add(ex)
        min_d = min(event_dates)
        max_d = max(event_dates)
        if close_map:
            for sym_dates, _ in close_map.values():
                for d in sym_dates:
                    if d >= min_d:
                        event_dates.add(d)
                        max_d = max(max_d, d)

        all_dates = sorted(event_dates)
        signal_idx = 0

        for current_date in all_dates:
            # 1. Size today's morning/open entries using start-of-day equity and prior cash
            start_of_day_equity = equity

            # Close any stale positions whose exit date strictly preceded current_date
            stale_open: list[tuple[TradeSignal, float, float]] = []
            for sig, shares, allocated in open_positions:
                exit_date = _resolve_exit_date(sig)
                if exit_date is not None and exit_date < current_date:
                    pnl_pct = sig.return_pct
                    pnl = allocated * pnl_pct
                    cash += allocated + pnl
                    portfolio_trades.append(
                        PortfolioTrade(
                            signal=sig,
                            shares=shares,
                            capital_allocated=allocated,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )
                else:
                    stale_open.append((sig, shares, allocated))
            open_positions = stale_open

            # 2. Open new positions BEFORE crediting today's exit proceeds or today's close MTM
            while signal_idx < len(signals) and signals[signal_idx].signal_date == current_date:
                sig = signals[signal_idx]
                signal_idx += 1

                if len(open_positions) >= cfg.max_positions:
                    continue  # Skip — position limit reached

                allocation = min(
                    cash,
                    start_of_day_equity * cfg.position_size_pct,
                )
                if allocation < sig.entry_price or sig.entry_price <= 0:
                    continue  # Not enough capital

                shares = allocation / sig.entry_price
                cash -= allocation
                open_positions.append((sig, shares, allocation))

            max_concurrent = max(max_concurrent, len(open_positions))

            # 3. Close positions that reach their exit date at today's close/exit
            still_open: list[tuple[TradeSignal, float, float]] = []
            for sig, shares, allocated in open_positions:
                exit_date = _resolve_exit_date(sig)
                if exit_date is not None and exit_date <= current_date:
                    pnl_pct = sig.return_pct
                    pnl = allocated * pnl_pct
                    cash += allocated + pnl
                    portfolio_trades.append(
                        PortfolioTrade(
                            signal=sig,
                            shares=shares,
                            capital_allocated=allocated,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )
                else:
                    still_open.append((sig, shares, allocated))
            open_positions = still_open

            # 4. Compute end-of-day mark-to-market equity
            invested_mtm = 0.0
            for sig, shares, _alloc in open_positions:
                invested_mtm += _mark_to_market(sig, shares, current_date, close_map=close_map)
            current_equity = cash + invested_mtm
            prev_equity = equity_curve[-1].equity if equity_curve else cfg.initial_capital
            daily_ret = (current_equity / prev_equity - 1) if prev_equity > 0 else 0.0
            equity = current_equity

            equity_curve.append(
                PortfolioSnapshot(
                    date=current_date,
                    equity=current_equity,
                    cash=cash,
                    positions_open=len(open_positions),
                    daily_return=daily_ret,
                )
            )

            day_util = (invested_mtm / current_equity) if current_equity > 0 else 0.0
            total_capital_util += max(0.0, min(1.0, day_util))
            total_days += 1

        # Close or mark any remaining positions at final equity
        end_mark_date = all_dates[-1]
        for sig, shares, allocated in open_positions:
            if sig.exit_price is None and close_map and sig.symbol in close_map:
                final_val = _mark_to_market(sig, shares, end_mark_date, close_map=close_map)
                pnl = final_val - allocated
                pnl_pct = (pnl / allocated) if allocated > 0 else 0.0
            else:
                pnl_pct = sig.return_pct
                pnl = allocated * pnl_pct
            cash += allocated + pnl
            portfolio_trades.append(
                PortfolioTrade(
                    signal=sig,
                    shares=shares,
                    capital_allocated=allocated,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            )

        final_equity = cash
        total_return = (final_equity / cfg.initial_capital - 1) if cfg.initial_capital > 0 else 0.0
        max_dd = _compute_max_drawdown(equity_curve, initial_capital=cfg.initial_capital)
        sharpe = _compute_portfolio_sharpe(equity_curve)
        cap_util = (total_capital_util / total_days) if total_days > 0 else 0.0

        return PortfolioResult(
            config=cfg,
            trades=portfolio_trades,
            equity_curve=equity_curve,
            final_equity=final_equity,
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            max_concurrent_positions=max_concurrent,
            total_trades=len(portfolio_trades),
            capital_utilization=min(1.0, cap_util),
        )


def _build_close_map(
    market_data: dict[str, pd.DataFrame] | None,
) -> dict[str, tuple[list[date], list[float]]]:
    """Build per-symbol date->close lookup tables for real daily MTM."""
    out: dict[str, tuple[list[date], list[float]]] = {}
    if not market_data:
        return out
    for sym, df in market_data.items():
        if df is None or getattr(df, "empty", True):
            continue
        if "date" not in df.columns or "close" not in df.columns:
            continue
        pairs: list[tuple[date, float]] = []
        for d_val, c_val in zip(df["date"].tolist(), df["close"].tolist(), strict=False):
            d_norm = d_val.date() if hasattr(d_val, "date") and type(d_val) is not date else d_val
            if isinstance(d_norm, date) and c_val is not None:
                pairs.append((d_norm, float(c_val)))
        if pairs:
            pairs.sort(key=lambda item: item[0])
            out[sym] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return out


def _resolve_exit_date(signal: TradeSignal) -> date | None:
    """Prefer actual realized exit_date; return None for still-open signals."""
    if signal.exit_price is None:
        return None
    return signal.exit_date or _estimate_exit_date(signal)


def _mark_to_market(
    signal: TradeSignal,
    shares: float,
    current_date: date,
    *,
    close_map: dict[str, tuple[list[date], list[float]]] | None = None,
) -> float:
    """Mark position using actual daily close when available, else linear interpolation."""
    if close_map and signal.symbol in close_map:
        dates_list, closes_list = close_map[signal.symbol]
        pos = bisect_right(dates_list, current_date) - 1
        if pos >= 0:
            return shares * closes_list[pos]

    if signal.holding_days <= 0 or signal.exit_price is None:
        return shares * signal.entry_price  # Fallback to entry value

    days_held = (current_date - signal.signal_date).days
    if days_held <= 0:
        return shares * signal.entry_price

    progress = min(1.0, days_held / max(1, signal.holding_days))
    estimated_price = signal.entry_price + progress * (signal.exit_price - signal.entry_price)
    return shares * estimated_price


def _estimate_exit_date(signal: TradeSignal) -> date | None:
    """Estimate exit date from signal_date + holding_days (trading days).

    Skips weekends (Saturday/Sunday) to approximate real market schedules.
    Does not handle exchange-specific holidays.
    """
    if signal.holding_days <= 0:
        return None

    from datetime import timedelta

    current = signal.signal_date
    remaining = signal.holding_days
    while remaining > 0:
        current += timedelta(days=1)
        # Skip weekends: 5=Saturday, 6=Sunday
        if current.weekday() < 5:
            remaining -= 1
    return current


def _compute_max_drawdown(
    curve: list[PortfolioSnapshot],
    *,
    initial_capital: float | None = None,
) -> float:
    """Compute maximum drawdown from equity curve, anchored to initial_capital when provided."""
    if not curve:
        return 0.0
    peak = max(initial_capital, curve[0].equity) if initial_capital is not None else curve[0].equity
    max_dd = 0.0
    for snap in curve:
        if snap.equity > peak:
            peak = snap.equity
        dd = (peak - snap.equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _compute_portfolio_sharpe(
    curve: list[PortfolioSnapshot],
    risk_free_rate: float = 0.02,
) -> float:
    """Compute annualized Sharpe ratio from all daily portfolio returns."""
    if len(curve) < 2:
        return 0.0
    returns = [snap.daily_return for snap in curve]
    if len(returns) < 2:
        return 0.0
    avg = sum(returns) / len(returns)
    var = sum((r - avg) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    return (avg - daily_rf) / std * math.sqrt(252)
