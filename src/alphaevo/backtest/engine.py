"""Backtest engine — simulates strategy execution on historical data.

Core loop: for each symbol in sample batch, iterate daily bars,
evaluate entry conditions, manage positions, check exits.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

import pandas as pd

from alphaevo.backtest.condition import ConditionEvaluator
from alphaevo.backtest.indicators import (
    IndicatorRegistry,
    IndicatorValue,
    _align_benchmark_idx,
    _extract_dates,
)
from alphaevo.backtest.rules import MarketRuleChecker
from alphaevo.models.enums import ExitReason, MarketRegime, SignalDirection
from alphaevo.models.execution import BacktestResult, SampleBatch, TradeSignal
from alphaevo.sampler.regime import RegimeDetector

if TYPE_CHECKING:
    from alphaevo.models.market import IndicatorContext
    from alphaevo.models.strategy import MarketRuleConfig, StopLossConfig, Strategy

FillPolicy = Literal["conservative", "optimistic", "close_first"]
_FILL_POLICY_ALIASES = {
    "stop_first": "conservative",
    "take_profit_first": "optimistic",
    "tp_first": "optimistic",
}


@dataclass
class _Position:
    """Internal position tracking during backtest."""

    symbol: str
    entry_idx: int
    entry_date: date
    entry_price: float
    regime: MarketRegime | None = None
    sector: str | None = None
    entry_atr: float | None = None
    indicator_snapshot: dict[str, float | bool] | None = None


class BacktestEngine:
    """Signal-level backtest engine.

    Iterates daily bars for each symbol, evaluates entry/exit conditions,
    and records trade signals with outcomes.
    """

    def __init__(
        self,
        condition_evaluator: ConditionEvaluator | None = None,
        rule_checker: MarketRuleChecker | None = None,
        slippage: float = 0.001,
        commission: float = 0.0003,
        min_data_days: int = 30,
        fill_policy: str = "conservative",
    ) -> None:
        if not 0 <= slippage <= 0.1:
            raise ValueError(f"slippage must be in [0, 0.1], got {slippage}")
        if not 0 <= commission <= 0.1:
            raise ValueError(f"commission must be in [0, 0.1], got {commission}")
        normalized_fill_policy = _normalize_fill_policy(fill_policy)
        self.evaluator = condition_evaluator or ConditionEvaluator()
        self.rule_checker = rule_checker or MarketRuleChecker()
        self.slippage = slippage
        self.commission = commission
        self.min_data_days = min_data_days
        self.fill_policy = normalized_fill_policy
        self._evaluator_supports_precomputed = (
            "precomputed" in inspect.signature(self.evaluator.evaluate_entry).parameters
        )
        self._cond_supports_precomputed = (
            "precomputed" in inspect.signature(self.evaluator.evaluate_condition).parameters
        )
        self._rule_checker_supports_meta = (
            "symbol" in inspect.signature(self.rule_checker.can_buy).parameters
        )
        self._regime_cache: dict[tuple[int, int], MarketRegime | None] = {}
        self._regime_series_cache: dict[int, tuple[list[date] | None, list[MarketRegime]]] = {}

    def run(
        self,
        strategy: Strategy,
        data: dict[str, pd.DataFrame],
        batch: SampleBatch,
        contexts: dict[str, IndicatorContext] | None = None,
    ) -> BacktestResult:
        """Run backtest for all symbols in batch.

        Args:
            strategy: Strategy to test.
            data: Dict of symbol → OHLCV DataFrame.
                  DataFrame must have columns: date, open, high, low, close, volume.
                  Optionally: prev_close, amount.
            batch: Sample batch definition.
            contexts: Optional per-symbol IndicatorContext for L2/L3 indicators.

        Returns:
            BacktestResult with all trade signals.
        """
        unknown_indicators = self._collect_unknown_indicators(strategy)
        if unknown_indicators:
            names = ", ".join(unknown_indicators)
            raise ValueError(f"Unknown indicators in strategy: {names}")
        self._regime_cache = {}
        self._regime_series_cache = {}

        # Resolve market rules
        market_key = strategy.meta.market.value
        rules = strategy.market_rules.get(
            market_key,
            MarketRuleChecker.default_rules(market_key),
        )

        # Resolve entry timing and per-strategy slippage override
        timing = "next_open"
        effective_slippage = self.slippage
        if strategy.entry.execution:
            timing = strategy.entry.execution.timing
            if strategy.entry.execution.slippage is not None:
                effective_slippage = strategy.entry.execution.slippage

        all_signals: list[TradeSignal] = []

        for symbol in batch.symbols:
            df = data.get(symbol)
            if df is None or df.empty or len(df) < self.min_data_days:
                continue

            ctx = contexts.get(symbol) if contexts else None
            signals = self._run_symbol(
                strategy,
                df,
                symbol,
                rules,
                ctx,
                timing,
                effective_slippage,
            )
            all_signals.extend(signals)

        executed = [s for s in all_signals if s.exit_price is not None]
        return BacktestResult(
            strategy_id=strategy.meta.id,
            batch_id=batch.batch_id,
            signals=all_signals,
            total_signals=len(all_signals),
            executed_signals=len(executed),
            skipped_signals=len(all_signals) - len(executed),
            date_range=batch.date_range,
        )

    def _can_buy(
        self,
        df: pd.DataFrame,
        idx: int,
        rules: MarketRuleConfig,
        symbol: str,
        stock_name: str,
        is_st: bool | None,
    ) -> bool:
        if self._rule_checker_supports_meta:
            return self.rule_checker.can_buy(
                df,
                idx,
                rules,
                symbol=symbol,
                stock_name=stock_name,
                is_st=is_st,
            )
        return self.rule_checker.can_buy(df, idx, rules)

    def _can_sell(
        self,
        df: pd.DataFrame,
        idx: int,
        entry_idx: int,
        rules: MarketRuleConfig,
        symbol: str,
        stock_name: str,
        is_st: bool | None,
    ) -> bool:
        if self._rule_checker_supports_meta:
            return self.rule_checker.can_sell(
                df,
                idx,
                entry_idx,
                rules,
                symbol=symbol,
                stock_name=stock_name,
                is_st=is_st,
            )
        return self.rule_checker.can_sell(df, idx, entry_idx, rules)

    def _run_symbol(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        symbol: str,
        rules: MarketRuleConfig,
        ctx: IndicatorContext | None,
        timing: str,
        slippage: float,
    ) -> list[TradeSignal]:
        """Run backtest on a single symbol."""
        signals: list[TradeSignal] = []
        position: _Position | None = None
        # Ensure we have enough bars for the largest indicator window
        all_indicators = self._collect_all_indicators(strategy)
        warmup = _indicator_warmup(all_indicators)
        start_idx = max(self.min_data_days, warmup)

        precompute_names = list(all_indicators)
        precompute_names.extend(
            ("rsi_14", "volume_ratio_1d_5d", "atr", "momentum_10d", "ma20_slope")
        )
        sl = strategy.exit.stop_loss
        if sl.type == "atr":
            atr_ind = f"atr_{sl.atr_period}" if sl.atr_period and sl.atr_period != 14 else "atr"
            precompute_names.append(atr_ind)
        precomputed = (
            IndicatorRegistry.precompute(precompute_names, df, ctx)
            if self._evaluator_supports_precomputed
            else None
        )
        stock_name = ctx.stock_info.name if ctx and ctx.stock_info else ""
        is_st = ctx.stock_info.is_st if ctx and ctx.stock_info else None

        for idx in range(start_idx, len(df)):
            # ── Holding a position: check exit ──
            if position is not None:
                if not self._can_sell(
                    df, idx, position.entry_idx, rules, symbol, stock_name, is_st
                ):
                    continue

                exit_result = self._check_exit(
                    position, strategy, df, idx, ctx, precomputed=precomputed
                )
                if exit_result is not None:
                    reason, exit_price = exit_result
                    signal = self._close_position(position, df, idx, exit_price, reason, slippage)
                    signals.append(signal)
                    position = None
                continue  # Don't check entry while holding

            # ── No position: check entry ──
            if timing == "close" and not self._can_buy(df, idx, rules, symbol, stock_name, is_st):
                continue

            if not self._regime_allows_entry(strategy, df, idx, ctx):
                continue

            cond_ok = (
                self.evaluator.evaluate_entry(strategy.entry, df, idx, ctx, precomputed=precomputed)
                if self._evaluator_supports_precomputed
                else self.evaluator.evaluate_entry(strategy.entry, df, idx, ctx)
            )
            if cond_ok:
                position = self._open_position(
                    symbol,
                    df,
                    idx,
                    timing,
                    slippage,
                    ctx,
                    rules=rules,
                    stock_name=stock_name,
                    is_st=is_st,
                    strategy=strategy,
                    precomputed=precomputed,
                )
                if position is None:
                    continue
                # Capture indicator snapshot for failure analysis
                position.indicator_snapshot = self._capture_indicators(
                    strategy,
                    df,
                    idx,
                    ctx,
                    precomputed=precomputed,
                )

        # Force close any remaining position at end
        if position is not None:
            signal = self._close_position(
                position, df, len(df) - 1, df["close"].iloc[-1], ExitReason.MAX_HOLD, slippage
            )
            signals.append(signal)

        return signals

    def _open_position(
        self,
        symbol: str,
        df: pd.DataFrame,
        signal_idx: int,
        timing: str,
        slippage: float,
        ctx: IndicatorContext | None,
        *,
        rules: MarketRuleConfig | None = None,
        stock_name: str = "",
        is_st: bool | None = None,
        strategy: Strategy | None = None,
        precomputed: dict[str, list[IndicatorValue]] | None = None,
    ) -> _Position | None:
        """Create a new position based on entry timing."""
        if timing == "close":
            entry_idx = signal_idx
            entry_price = float(df["close"].iloc[signal_idx])
        elif timing == "breakout_high":
            if signal_idx + 1 >= len(df):
                return None  # no next bar to check breakout
            entry_idx = signal_idx + 1
            breakout_price = float(df["high"].iloc[signal_idx])
            # Confirm breakout: next day must open above signal day's high
            next_open = float(df["open"].iloc[entry_idx])
            if next_open < breakout_price:
                return None
            entry_price = next_open
        else:
            # next_open / default: must have a valid next bar
            if signal_idx + 1 >= len(df):
                return None
            entry_idx = signal_idx + 1
            entry_price = float(df["open"].iloc[entry_idx])

        # Enforce market rules on the actual entry bar
        if rules is not None and not self._can_buy(df, entry_idx, rules, symbol, stock_name, is_st):
            return None

        # Apply slippage (buy slightly higher)
        entry_price *= 1 + slippage

        entry_atr: float | None = None
        if strategy is not None and strategy.exit.stop_loss.type == "atr":
            sl = strategy.exit.stop_loss
            atr_ind = f"atr_{sl.atr_period}" if sl.atr_period and sl.atr_period != 14 else "atr"
            if precomputed and atr_ind in precomputed:
                entry_atr = float(precomputed[atr_ind][signal_idx])
            else:
                entry_atr = float(IndicatorRegistry.compute(atr_ind, df, signal_idx, ctx))

        entry_date = self._get_date(df, entry_idx)
        return _Position(
            symbol=symbol,
            entry_idx=entry_idx,
            entry_date=entry_date,
            entry_price=entry_price,
            regime=self._get_current_regime(df, signal_idx, ctx),
            sector=self._get_sector(ctx),
            entry_atr=entry_atr,
        )

    def _check_exit(
        self,
        position: _Position,
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
        *,
        precomputed: dict[str, list[IndicatorValue]] | None = None,
    ) -> tuple[ExitReason, float] | None:
        """Check all exit conditions. Returns (reason, exit_price) or None."""
        exit_cfg = strategy.exit
        current_price = float(df["close"].iloc[idx])
        holding_days = idx - position.entry_idx

        # 1. Max holding days
        if holding_days >= exit_cfg.max_holding_days:
            return (ExitReason.MAX_HOLD, current_price)

        # 2. Stop loss
        stop_result = self._check_stop_loss(
            position, strategy, df, idx, ctx, precomputed=precomputed
        )

        # 3. Take profit
        take_profit_result = self._check_take_profit(position, strategy, df, idx, ctx)

        if stop_result is not None and take_profit_result is not None:
            return self._resolve_intrabar_conflict(
                stop_result,
                take_profit_result,
                position,
                current_price,
            )
        if stop_result is not None:
            return stop_result
        if take_profit_result is not None:
            return take_profit_result

        # 4. Explicit exit triggers. These are end-of-bar sell signals,
        # so they execute at the current close after intraday risk checks.
        for trigger in exit_cfg.triggers:
            hit = (
                self.evaluator.evaluate_condition(trigger, df, idx, ctx, precomputed=precomputed)
                if self._cond_supports_precomputed
                else self.evaluator.evaluate_condition(trigger, df, idx, ctx)
            )
            if hit:
                return (ExitReason.SIGNAL, current_price)

        return None

    def _check_stop_loss(
        self,
        position: _Position,
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
        *,
        precomputed: dict[str, list[IndicatorValue]] | None = None,
    ) -> tuple[ExitReason, float] | None:
        """Check stop-loss exits for one bar with gap-aware fills."""
        exit_cfg = strategy.exit
        current_price = float(df["close"].iloc[idx])
        open_price = float(df["open"].iloc[idx])
        low_price = float(df["low"].iloc[idx])
        sl = exit_cfg.stop_loss
        if sl.type == "pct" and sl.value is not None:
            stop_price = position.entry_price * (1 - sl.value)
            if open_price <= stop_price:
                return (ExitReason.STOP_LOSS, open_price)
            if low_price <= stop_price:
                return (ExitReason.STOP_LOSS, stop_price)

        elif sl.type == "atr":
            entry_atr = getattr(position, "entry_atr", None)
            if entry_atr is None:
                atr_ind = f"atr_{sl.atr_period}" if sl.atr_period and sl.atr_period != 14 else "atr"
                atr_val = float(IndicatorRegistry.compute(atr_ind, df, idx, ctx))
            else:
                atr_val = float(entry_atr)
            multiplier = sl.multiplier or 2.0
            stop_price = position.entry_price - atr_val * multiplier
            if open_price <= stop_price:
                return (ExitReason.STOP_LOSS, open_price)
            if low_price <= stop_price:
                return (ExitReason.STOP_LOSS, stop_price)

        elif sl.type == "pct_from_low" and sl.value is not None:
            # Causal trailing peak using prior bars since entry
            if idx > position.entry_idx:
                high_since_entry = float(df["high"].iloc[position.entry_idx : idx].max())
            else:
                high_since_entry = float(df["high"].iloc[position.entry_idx])
            if high_since_entry > 0:
                stop_price = high_since_entry * (1 - sl.value)
                if open_price <= stop_price:
                    return (ExitReason.STOP_LOSS, open_price)
                if low_price <= stop_price:
                    return (ExitReason.STOP_LOSS, stop_price)

        elif sl.type == "price_level":
            price_level = self._resolve_price_level(position, sl, df, idx, ctx)
            if price_level is not None:
                if open_price <= price_level:
                    return (ExitReason.STOP_LOSS, open_price)
                if low_price <= price_level:
                    return (ExitReason.STOP_LOSS, price_level)

        elif sl.type == "composite" and sl.conditions:
            # Any condition triggers stop loss
            for cond in sl.conditions:
                hit = (
                    self.evaluator.evaluate_condition(cond, df, idx, ctx, precomputed=precomputed)
                    if self._cond_supports_precomputed
                    else self.evaluator.evaluate_condition(cond, df, idx, ctx)
                )
                if hit:
                    return (ExitReason.STOP_LOSS, current_price)

        return None

    def _check_take_profit(
        self,
        position: _Position,
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> tuple[ExitReason, float] | None:
        """Check take-profit exits for one bar with gap-aware fills."""
        exit_cfg = strategy.exit
        tp = exit_cfg.take_profit
        open_price = float(df["open"].iloc[idx])
        high_price = float(df["high"].iloc[idx])
        low_price = float(df["low"].iloc[idx])
        risk_amount = self._compute_risk(position, exit_cfg.stop_loss, df, idx, ctx)

        if tp.type == "rr" and tp.value is not None:
            target = position.entry_price + risk_amount * tp.value
            if open_price >= target:
                return (ExitReason.TAKE_PROFIT, open_price)
            if high_price >= target:
                return (ExitReason.TAKE_PROFIT, target)

        elif tp.type == "pct" and tp.value is not None:
            target = position.entry_price * (1 + tp.value)
            if open_price >= target:
                return (ExitReason.TAKE_PROFIT, open_price)
            if high_price >= target:
                return (ExitReason.TAKE_PROFIT, target)

        elif tp.type == "target_ma" and tp.target:
            target_idx = max(position.entry_idx, idx - 1)
            target_ma = self._resolve_target_ma(tp.target, df, target_idx)
            if target_ma is not None:
                if open_price >= target_ma:
                    return (ExitReason.TAKE_PROFIT, open_price)
                if high_price >= target_ma:
                    return (ExitReason.TAKE_PROFIT, target_ma)

        elif tp.type == "trailing" and idx > position.entry_idx:
            trigger_pct = tp.trigger_pct or 0.08
            trail_pct = tp.trail_pct or 0.04
            high_since = float(df["high"].iloc[position.entry_idx : idx].max())
            gain_from_entry = (high_since - position.entry_price) / position.entry_price
            if gain_from_entry >= trigger_pct:
                trail_price = high_since * (1 - trail_pct)
                if open_price <= trail_price:
                    return (ExitReason.TAKE_PROFIT, open_price)
                if low_price <= trail_price:
                    return (ExitReason.TAKE_PROFIT, trail_price)

        return None

    def _resolve_intrabar_conflict(
        self,
        stop_result: tuple[ExitReason, float],
        take_profit_result: tuple[ExitReason, float],
        position: _Position,
        close_price: float,
    ) -> tuple[ExitReason, float]:
        """Resolve a bar where stop-loss and take-profit both appear reachable."""
        if self.fill_policy == "optimistic":
            return take_profit_result
        if self.fill_policy == "close_first":
            close_return = (close_price - position.entry_price) / position.entry_price
            return take_profit_result if close_return >= 0 else stop_result
        return stop_result

    def _compute_risk(
        self,
        position: _Position,
        sl_config: StopLossConfig,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> float:
        """Compute per-share risk amount for RR-based take profit."""
        if sl_config.type in ("pct", "pct_from_low") and sl_config.value is not None:
            return float(abs(position.entry_price * sl_config.value))
        elif sl_config.type == "atr":
            entry_atr = getattr(position, "entry_atr", None)
            if entry_atr is not None:
                return float(float(entry_atr) * (sl_config.multiplier or 2.0))
            atr_ind = (
                f"atr_{sl_config.atr_period}"
                if sl_config.atr_period and sl_config.atr_period != 14
                else "atr"
            )
            atr_val = IndicatorRegistry.compute(atr_ind, df, idx, ctx)
            return float(atr_val * (sl_config.multiplier or 2.0))
        # Default: 5% risk (composite or unknown types)
        return float(position.entry_price * 0.05)

    def _close_position(
        self,
        position: _Position,
        df: pd.DataFrame,
        idx: int,
        exit_price: float,
        reason: ExitReason,
        slippage: float,
    ) -> TradeSignal:
        """Close a position and create a TradeSignal."""
        # Apply slippage (sell slightly lower)
        exit_price *= 1 - slippage
        # Apply commission both ways
        net_entry = position.entry_price * (1 + self.commission)
        net_exit = exit_price * (1 - self.commission)
        return_pct = (net_exit - net_entry) / net_entry

        exit_date = self._get_date(df, idx)
        holding_days = idx - position.entry_idx

        return TradeSignal(
            symbol=position.symbol,
            signal_date=position.entry_date,
            direction=SignalDirection.LONG,
            entry_price=position.entry_price,
            exit_price=exit_price,
            exit_date=exit_date,
            exit_reason=reason,
            return_pct=round(return_pct, 6),
            holding_days=holding_days,
            regime=position.regime,
            sector=position.sector,
            indicator_snapshot=position.indicator_snapshot or {},
        )

    @staticmethod
    def _resolve_target_ma(target: str, df: pd.DataFrame, idx: int) -> float | None:
        """Resolve a take-profit moving-average target."""
        target = target.lower().strip()
        periods = {
            "ma5": 5,
            "ma10": 10,
            "ma20": 20,
            "ma60": 60,
        }
        period = periods.get(target)
        if period is None or idx + 1 < period:
            return None
        window = df["close"].iloc[idx - period + 1 : idx + 1]
        return float(window.mean())

    @staticmethod
    def _resolve_price_level(
        position: _Position,
        sl_config: StopLossConfig,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> float | None:
        """Resolve a stop-loss price level from config."""
        if sl_config.value is not None:
            return float(sl_config.value)

        reference = (sl_config.reference or "").strip()
        if not reference:
            return None

        if reference in {"entry", "entry_price"}:
            return float(position.entry_price)
        if reference in {"signal_low", "entry_low"}:
            return float(df["low"].iloc[position.entry_idx])
        if reference in {"signal_close", "entry_close"}:
            return float(df["close"].iloc[position.entry_idx])

        if reference == "pre_event_close" and ctx is not None and ctx.pre_event_close is not None:
            return float(ctx.pre_event_close)

        if reference in df.columns:
            value = df[reference].iloc[min(idx, len(df) - 1)]
            return float(value) if pd.notna(value) else None

        return None

    @staticmethod
    def _get_sector(ctx: IndicatorContext | None) -> str | None:
        """Resolve the current symbol sector from context."""
        if ctx and ctx.stock_info and ctx.stock_info.sector:
            return ctx.stock_info.sector
        if ctx and ctx.sector_info and ctx.sector_info.name:
            return ctx.sector_info.name
        return None

    def _regime_allows_entry(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> bool:
        """Check preferred market regime gating before entry."""
        if not strategy.meta.preferred_regime:
            return True

        regime = self._get_current_regime(df, idx, ctx)
        if regime is None:
            return True
        return regime.value in set(strategy.meta.preferred_regime)

    def _get_regime_series(
        self,
        source_df: pd.DataFrame,
    ) -> tuple[list[date] | None, list[MarketRegime]]:
        """Detect and cache full causal regime series for *source_df*."""
        key = id(source_df)
        cached = self._regime_series_cache.get(key)
        if cached is not None:
            return cached
        work_df = source_df
        if "high" not in work_df.columns or "low" not in work_df.columns:
            work_df = source_df.copy()
            for col in ("high", "low"):
                if col not in work_df.columns:
                    work_df[col] = work_df["close"]
        regimes = RegimeDetector().detect_series(work_df)
        dates = _extract_dates(source_df)
        result = (dates, regimes)
        self._regime_series_cache[key] = result
        return result

    def _get_current_regime(
        self,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> MarketRegime | None:
        """Detect current regime causally from date-aligned benchmark or symbol data."""
        if ctx and isinstance(ctx.benchmark_df, pd.DataFrame) and not ctx.benchmark_df.empty:
            bm = ctx.benchmark_df
            bm_idx = _align_benchmark_idx(df, idx, bm)
            if bm_idx is None or bm_idx + 1 < 20:
                return None
            _, bm_regimes = self._get_regime_series(bm)
            return bm_regimes[bm_idx]

        # Only honor static MarketContext regime when series is shorter than the
        # causal regime warmup window (<65 bars); longer series detect causally.
        if len(df) < 65 and ctx and ctx.market_context and ctx.market_context.regime is not None:
            return ctx.market_context.regime

        if idx + 1 < 20:
            return None
        _, sym_regimes = self._get_regime_series(df)
        return sym_regimes[min(idx, len(sym_regimes) - 1)]

    @staticmethod
    def _collect_unknown_indicators(strategy: Strategy) -> list[str]:
        """Collect indicator names not registered in IndicatorRegistry."""
        names = [c.indicator for c in strategy.entry.conditions]
        names.extend(c.indicator for c in strategy.entry.triggers)
        names.extend(c.indicator for c in strategy.entry.filters)
        names.extend(c.indicator for c in strategy.entry.guards)
        names.extend(c.indicator for c in strategy.exit.triggers)
        if strategy.exit.stop_loss.conditions:
            names.extend(c.indicator for c in strategy.exit.stop_loss.conditions)

        unknown = {name for name in names if not IndicatorRegistry.is_registered(name)}
        return sorted(unknown)

    @staticmethod
    def _collect_all_indicators(strategy: Strategy) -> list[str]:
        """Collect all indicator names used by the strategy."""
        names = [c.indicator for c in strategy.entry.conditions]
        names.extend(c.indicator for c in strategy.entry.triggers)
        names.extend(c.indicator for c in strategy.entry.filters)
        names.extend(c.indicator for c in strategy.entry.guards)
        names.extend(c.indicator for c in strategy.exit.triggers)
        if strategy.exit.stop_loss.conditions:
            names.extend(c.indicator for c in strategy.exit.stop_loss.conditions)
        return sorted(set(names))

    @staticmethod
    def _capture_indicators(
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
        *,
        precomputed: dict[str, list[IndicatorValue]] | None = None,
    ) -> dict[str, float | bool]:
        """Capture all strategy-relevant indicator values at entry.

        This snapshot is attached to TradeSignal so the reflection layer
        can see exactly what indicators looked like when the trade was taken.
        """
        snapshot: dict[str, float | bool] = {}
        indicators_to_capture = {c.indicator for c in strategy.entry.conditions}
        indicators_to_capture.update(c.indicator for c in strategy.entry.triggers)
        indicators_to_capture.update(c.indicator for c in strategy.entry.filters)
        indicators_to_capture.update(c.indicator for c in strategy.entry.guards)
        # Also capture core context indicators the LLM often references
        for extra in ("rsi_14", "volume_ratio_1d_5d", "atr", "momentum_10d", "ma20_slope"):
            indicators_to_capture.add(extra)
        for name in sorted(indicators_to_capture):
            if precomputed is not None and name in precomputed:
                val = precomputed[name][idx]
                snapshot[name] = val if isinstance(val, bool) else round(float(val), 4)
            elif IndicatorRegistry.is_registered(name):
                try:
                    val = IndicatorRegistry.compute(name, df, idx, ctx)
                    if isinstance(val, bool):
                        snapshot[name] = val
                    else:
                        snapshot[name] = round(float(val), 4)
                except Exception:
                    pass
        return snapshot

    @staticmethod
    def _get_date(df: pd.DataFrame, idx: int) -> date:
        """Extract date from DataFrame row."""
        val = df.iloc[idx].get("date", None)
        if val is None:
            return date.today()
        if isinstance(val, date):
            return val
        try:
            parsed = pd.Timestamp(val).date()
            return parsed if isinstance(parsed, date) else date.today()
        except Exception:
            return date.today()


def _indicator_warmup(indicator_names: list[str]) -> int:
    """Estimate the minimum bar index required for all indicators to produce valid values.

    Parses period numbers from indicator names (e.g., ``ma60`` -> 60,
    ``rsi_14`` -> 14, ``bollinger_band_width_30d`` -> 30) and returns
    the maximum period found with a convergence buffer.

    EMA-based indicators (RSI, MACD, Bollinger) need approximately 2x period
    bars for the exponential weighting to stabilise. SMA-based indicators
    only need exactly *period* bars.  We use ``max_period * 2`` as a safe
    lower bound.
    """
    import re

    max_period = 0
    has_ema_indicator = False
    for name in indicator_names:
        # Extract all numbers that look like period lengths
        for m in re.finditer(r"(\d+)", name):
            val = int(m.group(1))
            # Heuristic: reasonable period range 2-500; ignore tiny numbers
            # that are more likely version suffixes (e.g., _v1)
            if 2 <= val <= 500:
                max_period = max(max_period, val)
        # Detect EMA-based indicators that need longer warmup
        lower = name.lower()
        if any(tag in lower for tag in ("rsi", "macd", "ema", "ewm", "bollinger")):
            has_ema_indicator = True

    # MACD needs slow + signal periods of warmup
    for name in indicator_names:
        if "macd" in name.lower():
            max_period = max(max_period, 35)  # 26 + 9 default
            has_ema_indicator = True

    # EMA convergence needs ~2x period; SMA needs exactly period
    buffer = max_period if has_ema_indicator else max(5, max_period // 4)
    return max(30, max_period + buffer)


def _normalize_fill_policy(fill_policy: str) -> FillPolicy:
    normalized = _FILL_POLICY_ALIASES.get(fill_policy.strip().lower(), fill_policy.strip().lower())
    if normalized not in {"conservative", "optimistic", "close_first"}:
        raise ValueError(
            "fill_policy must be one of: conservative, optimistic, close_first, "
            "stop_first, take_profit_first"
        )
    return normalized  # type: ignore[return-value]
