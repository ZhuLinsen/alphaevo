"""Causal Regime-Adaptive Multi-Strategy Risk Budget & Correlation-Aware Allocator."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.core.config import BacktestConfig
from alphaevo.models.enums import MarketRegime, SamplingMethod
from alphaevo.models.execution import SampleBatch
from alphaevo.models.strategy import Strategy
from alphaevo.sampler.regime import RegimeDetector

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


class RiskBudgetConfig(BaseModel):
    """Configuration for causal multi-strategy risk budgeting & drawdown control."""

    initial_capital: float = Field(default=100_000.0, gt=0)
    lookback_days: int = Field(default=20, ge=5)
    vol_floor: float = Field(default=0.005, gt=0)
    target_annual_vol: float = Field(default=0.18, gt=0)
    max_strategy_weight: float = Field(default=0.60, gt=0, le=1.0)
    regime_mismatch_haircut: float = Field(default=0.35, ge=0.0, le=1.0)
    correlation_penalty_strength: float = Field(default=0.50, ge=0.0, le=2.0)
    drawdown_deleveraging_threshold: float = Field(default=0.12, gt=0.0, lt=1.0)
    drawdown_deleveraging_scale: float = Field(default=0.50, gt=0.0, le=1.0)


class StrategyAllocationSummary(BaseModel):
    """Per-strategy contribution and weight diagnostics across the allocation horizon."""

    strategy_id: str
    preferred_regimes: list[str] = Field(default_factory=list)
    mean_weight: float = 0.0
    max_weight: float = 0.0
    standalone_total_return: float = 0.0
    standalone_sharpe: float = 0.0
    standalone_max_drawdown: float = 0.0
    signal_count: int = 0
    return_contribution: float = 0.0


class DailyAllocationPoint(BaseModel):
    """One trading day of causal weights, regime state, exposure, and equity."""

    date: date
    regime: str
    weights: dict[str, float]
    gross_exposure: float
    portfolio_return: float
    equity: float
    drawdown: float


class MultiStrategyAllocationReport(BaseModel):
    """Complete multi-strategy regime-adaptive portfolio allocation report."""

    strategy_ids: list[str]
    date_range: tuple[date, date]
    trading_days: int
    initial_capital: float
    final_equity: float
    combined_total_return: float
    combined_sharpe_ratio: float
    combined_max_drawdown: float
    equal_weight_total_return: float
    equal_weight_max_drawdown: float
    equal_weight_sharpe_ratio: float
    diversification_dd_reduction: float
    regime_day_counts: dict[str, int] = Field(default_factory=dict)
    pairwise_correlations: dict[str, float] = Field(default_factory=dict)
    strategies: list[StrategyAllocationSummary] = Field(default_factory=list)
    daily_curve: list[DailyAllocationPoint] = Field(default_factory=list)


def _normalize_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    vx = sum((x - mx) ** 2 for x in xs[:n])
    vy = sum((y - my) ** 2 for y in ys[:n])
    if vx <= 1e-16 or vy <= 1e-16:
        return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return max(-1.0, min(1.0, cov / math.sqrt(vx * vy)))


def _series_metrics(daily_returns: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(total_return, max_drawdown, annualized_sharpe)`` from daily returns."""
    if not daily_returns:
        return (0.0, 0.0, 0.0)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in daily_returns:
        equity *= max(0.0, 1.0 + r)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    total_ret = equity - 1.0
    if len(daily_returns) < 2:
        return (total_ret, max_dd, 0.0)
    avg_r = mean(daily_returns)
    std_r = stdev(daily_returns)
    sharpe = (avg_r / std_r * math.sqrt(252.0)) if std_r > 0 else 0.0
    return (total_ret, max_dd, sharpe)


def _apply_weight_cap(weights: list[float], cap: float) -> list[float]:
    """Cap weights cleanly, conserving budget across active strategies without reviving inactive ones."""
    n = len(weights)
    if n == 0:
        return []
    out = list(weights)
    if not any(w > cap + 1e-12 for w in out):
        return out

    pool = 0.0
    for i in range(n):
        if out[i] > cap + 1e-12:
            pool += out[i] - cap
            out[i] = cap

    for _ in range(n):
        if pool <= 1e-12:
            break
        under_active = [i for i, w in enumerate(out) if 1e-12 < w < cap - 1e-12]
        if not under_active:
            break
        under_total = sum(out[i] for i in under_active)
        if under_total <= 0:
            break
        used = 0.0
        for i in under_active:
            room = max(0.0, cap - out[i])
            add = min(room, pool * (out[i] / under_total))
            out[i] += add
            used += add
        if used <= 1e-12:
            break
        pool -= used
    return out


def _build_causal_market_proxy(
    all_dates: list[date],
    market_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Construct an equal-weight causal OHLCV index series over *all_dates*."""
    import pandas as pd  # type: ignore[import-untyped]

    date_index = {d: idx for idx, d in enumerate(all_dates)}
    n_days = len(all_dates)
    sum_ret = [0.0] * n_days
    cnt_ret = [0] * n_days
    sum_hi_ratio = [0.0] * n_days
    sum_lo_ratio = [0.0] * n_days
    sum_co_ratio = [0.0] * n_days
    cnt_hl = [0] * n_days
    cnt_co = [0] * n_days

    for df in market_data.values():
        if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
            continue
        by_date: dict[date, tuple[float, float, float, float | None]] = {}
        has_open = "open" in df.columns
        has_high = "high" in df.columns
        has_low = "low" in df.columns
        for idx in range(len(df)):
            ts = pd.to_datetime(df["date"].iloc[idx], errors="coerce")
            if pd.isna(ts):
                continue
            bar_d = ts.date()
            try:
                c_v = float(df["close"].iloc[idx])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(c_v) and c_v > 0):
                continue
            o_v: float | None = None
            if has_open:
                try:
                    raw_o = float(df["open"].iloc[idx])
                    if math.isfinite(raw_o) and raw_o > 0:
                        o_v = raw_o
                except (TypeError, ValueError):
                    pass
            h_v = c_v
            l_v = c_v
            if has_high:
                try:
                    raw_h = float(df["high"].iloc[idx])
                    if math.isfinite(raw_h) and raw_h > 0:
                        h_v = max(raw_h, c_v)
                except (TypeError, ValueError):
                    pass
            if has_low:
                try:
                    raw_l = float(df["low"].iloc[idx])
                    if math.isfinite(raw_l) and raw_l > 0:
                        l_v = min(raw_l, c_v)
                except (TypeError, ValueError):
                    pass
            by_date[bar_d] = (c_v, h_v, l_v, o_v)

        sorted_bars = sorted(by_date.items(), key=lambda item: item[0])
        prev_t_idx: int | None = None
        prev_close: float | None = None
        for bar_d, (c_v, h_v, l_v, o_v) in sorted_bars:
            t_idx = date_index.get(bar_d)
            if t_idx is None:
                continue
            sum_hi_ratio[t_idx] += h_v / c_v
            sum_lo_ratio[t_idx] += l_v / c_v
            cnt_hl[t_idx] += 1
            if o_v is not None and o_v > 0:
                sum_co_ratio[t_idx] += c_v / o_v
                cnt_co[t_idx] += 1
            if (
                prev_t_idx is not None
                and prev_t_idx == t_idx - 1
                and prev_close is not None
                and prev_close > 0
            ):
                sum_ret[t_idx] += (c_v / prev_close) - 1.0
                cnt_ret[t_idx] += 1
            prev_t_idx = t_idx
            prev_close = c_v

    opens: list[float] = []
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    level = 100.0
    for t in range(n_days):
        if cnt_ret[t] > 0:
            level *= max(0.01, 1.0 + sum_ret[t] / cnt_ret[t])
        co_r = (sum_co_ratio[t] / cnt_co[t]) if cnt_co[t] > 0 else 1.0
        open_t = level / max(0.01, co_r)
        hi_r = (sum_hi_ratio[t] / cnt_hl[t]) if cnt_hl[t] > 0 else 1.0
        lo_r = (sum_lo_ratio[t] / cnt_hl[t]) if cnt_hl[t] > 0 else 1.0
        opens.append(open_t)
        closes.append(level)
        highs.append(max(level * hi_r, open_t))
        lows.append(min(level * lo_r, open_t))

    return cast(
        "pd.DataFrame",
        pd.DataFrame(
            {
                "date": all_dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [1_000_000.0] * n_days,
                "observed_count": cnt_hl,
                "return_count": cnt_ret,
            }
        ),
    )


class RegimeAdaptiveAllocator:
    """Strictly causal inverse-vol, regime-gated, correlation-penalized risk allocator."""

    def __init__(self, config: RiskBudgetConfig | None = None) -> None:
        self.config = config or RiskBudgetConfig()

    def allocate(
        self,
        strategies: Sequence[Strategy],
        market_data: dict[str, pd.DataFrame],
        *,
        backtest_config: BacktestConfig | None = None,
    ) -> MultiStrategyAllocationReport:
        """Allocate dynamic daily weights across *strategies* with zero look-ahead."""
        import pandas as pd  # type: ignore[import-untyped]

        if not strategies:
            raise ValueError("At least one strategy is required for allocation")
        if not market_data:
            raise ValueError("market_data cannot be empty")

        date_set: set[date] = set()
        for df in market_data.values():
            if df is None or df.empty or "date" not in df.columns:
                continue
            for item in pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().tolist():
                coerced = _normalize_date(item)
                if coerced is not None:
                    date_set.add(coerced)

        all_dates = sorted(date_set)
        if len(all_dates) < 2:
            raise ValueError("Need at least 2 trading dates across market_data")

        market_proxy = _build_causal_market_proxy(all_dates, market_data)
        regimes = RegimeDetector().detect_series(market_proxy)

        bt_cfg = backtest_config or BacktestConfig(min_data_days=15)
        bt_engine = BacktestEngine(
            slippage=bt_cfg.slippage,
            commission=bt_cfg.commission,
            min_data_days=min(15, bt_cfg.min_data_days),
            fill_policy=bt_cfg.fill_policy,
        )
        batch = SampleBatch(
            batch_id="portfolio_alloc",
            strategy_id="multi_strategy",
            symbols=sorted(market_data.keys()),
            date_range=(all_dates[0], all_dates[-1]),
            sampling_method=SamplingMethod.REPRESENTATIVE,
        )

        close_lookup: dict[str, tuple[list[date], list[float]]] = {}
        for sym, df in market_data.items():
            if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
                continue
            sym_pairs: dict[date, float] = {}
            for idx in range(len(df)):
                ts = pd.to_datetime(df["date"].iloc[idx], errors="coerce")
                if pd.isna(ts):
                    continue
                try:
                    c_v = float(df["close"].iloc[idx])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(c_v) and c_v > 0:
                    sym_pairs[ts.date()] = c_v
            if sym_pairs:
                s_items = sorted(sym_pairs.items(), key=lambda it: it[0])
                close_lookup[sym] = ([it[0] for it in s_items], [it[1] for it in s_items])

        strat_rets: list[list[float]] = []
        standalone_stats: list[tuple[float, float, float, int]] = []
        for strat in strategies:
            bt = bt_engine.run(strat, market_data, batch)
            executed = [s for s in bt.signals if s.exit_price is not None]
            open_sigs = [s for s in bt.signals if s.exit_price is None]
            # Aggregate EVERY position uniformly per-position per day so appending later
            # open positions never alters earlier historical returns.
            from bisect import bisect_left, bisect_right

            day_pos: dict[date, list[float]] = {}
            for sig in executed:
                if sig.symbol not in close_lookup or sig.entry_price <= 0:
                    continue
                s_dates, s_closes = close_lookup[sig.symbol]
                end_d = sig.exit_date or sig.signal_date
                if end_d <= sig.signal_date:
                    day_pos.setdefault(end_d, []).append(float(sig.return_pct))
                    continue
                lo = bisect_left(s_dates, sig.signal_date)
                hi = bisect_right(s_dates, end_d)
                entry_p = float(sig.entry_price)
                prev_mark = entry_p
                target_end = entry_p * (1.0 + float(sig.return_pct))
                booked_exit = False
                for pos in range(lo, hi):
                    bar_d = s_dates[pos]
                    close_p = s_closes[pos]
                    if bar_d == end_d:
                        bar_ret = (target_end / prev_mark) - 1.0 if prev_mark > 0 else 0.0
                        booked_exit = True
                    elif prev_mark > 0:
                        bar_ret = (close_p / prev_mark) - 1.0
                        prev_mark = close_p
                    else:
                        bar_ret = 0.0
                    day_pos.setdefault(bar_d, []).append(bar_ret)
                if not booked_exit:
                    day_pos.setdefault(end_d, []).append(
                        (target_end / prev_mark) - 1.0 if prev_mark > 0 else float(sig.return_pct)
                    )

            for sig in open_sigs:
                if sig.symbol not in close_lookup or sig.entry_price <= 0:
                    continue
                s_dates, s_closes = close_lookup[sig.symbol]
                prev_mark = float(sig.entry_price)
                for bar_d, close_p in zip(s_dates, s_closes, strict=True):
                    if bar_d < sig.signal_date:
                        continue
                    bar_ret = (close_p / prev_mark) - 1.0 if prev_mark > 0 else 0.0
                    prev_mark = close_p
                    day_pos.setdefault(bar_d, []).append(bar_ret)

            ret_by_date = {d: sum(vals) / len(vals) for d, vals in day_pos.items() if vals}

            aligned = [float(ret_by_date.get(d, 0.0)) for d in all_dates]
            strat_rets.append(aligned)
            st_tot, st_dd, st_sh = _series_metrics(aligned)
            standalone_stats.append((st_tot, st_sh, st_dd, len(bt.signals)))

        cfg = self.config
        n_strats = len(strategies)
        n_days = len(all_dates)
        daily_target_vol = cfg.target_annual_vol / math.sqrt(252.0)

        equity = cfg.initial_capital
        peak_equity = cfg.initial_capital
        current_dd = 0.0

        combined_returns: list[float] = []
        equal_weight_returns: list[float] = []
        weight_history: list[list[float]] = [[] for _ in range(n_strats)]
        contrib_sums = [0.0] * n_strats
        daily_curve: list[DailyAllocationPoint] = []
        regime_counts: Counter[str] = Counter()

        for t in range(n_days):
            win_start = max(0, t - cfg.lookback_days)
            causal_regime: MarketRegime = regimes[t - 1] if t > 0 else regimes[0]
            regime_counts[regimes[t].value] += 1

            hists = [strat_rets[i][win_start:t] for i in range(n_strats)]
            active_flags = [
                len(hists[i]) >= 2 and sum(1 for v in hists[i] if abs(v) > 1e-12) >= 2
                for i in range(n_strats)
            ]
            any_active = any(active_flags)

            raw_scores: list[float] = []
            for i in range(n_strats):
                if any_active and not active_flags[i]:
                    raw_scores.append(0.0)
                    continue
                vol_i = max(cfg.vol_floor, stdev(hists[i])) if active_flags[i] else cfg.vol_floor
                score_i = 1.0 / vol_i
                pref = strategies[i].meta.preferred_regime
                if pref:
                    pref_norm = {(r.value if hasattr(r, "value") else str(r)).lower() for r in pref}
                    if causal_regime.value.lower() not in pref_norm:
                        score_i *= cfg.regime_mismatch_haircut
                raw_scores.append(score_i)

            if n_strats > 1 and (t - win_start) >= 5:
                penalized = list(raw_scores)
                for i in range(n_strats):
                    if raw_scores[i] <= 0.0:
                        continue
                    pos_rhos: list[float] = []
                    for j in range(n_strats):
                        if i == j or raw_scores[j] <= 0.0:
                            continue
                        rho = _pearson(strat_rets[i][win_start:t], strat_rets[j][win_start:t])
                        pos_rhos.append(max(0.0, rho))
                    if pos_rhos:
                        mean_pos = sum(pos_rhos) / len(pos_rhos)
                        penalized[i] /= 1.0 + cfg.correlation_penalty_strength * mean_pos
                raw_scores = penalized

            total_score = sum(raw_scores)
            rel_weights = (
                [s / total_score for s in raw_scores]
                if total_score > 0
                else [1.0 / n_strats] * n_strats
            )
            capped_weights = _apply_weight_cap(rel_weights, cfg.max_strategy_weight)

            # Causal portfolio volatility targeting using strictly trailing window [win_start:t]
            if (t - win_start) >= 5:
                trail_port = [
                    sum(capped_weights[i] * strat_rets[i][tau] for i in range(n_strats))
                    for tau in range(win_start, t)
                ]
                port_vol = stdev(trail_port) if len(trail_port) >= 2 else 0.0
                if port_vol > cfg.vol_floor:
                    vol_scale = min(1.0, daily_target_vol / port_vol)
                    capped_weights = [w * vol_scale for w in capped_weights]

            # Causal drawdown deleveraging from prior bar drawdown
            if current_dd >= cfg.drawdown_deleveraging_threshold:
                capped_weights = [w * cfg.drawdown_deleveraging_scale for w in capped_weights]

            port_ret = sum(capped_weights[i] * strat_rets[i][t] for i in range(n_strats))
            eq_ret = sum(strat_rets[i][t] for i in range(n_strats)) / n_strats

            for i in range(n_strats):
                weight_history[i].append(capped_weights[i])
                contrib_sums[i] += capped_weights[i] * strat_rets[i][t]

            equity *= max(0.0, 1.0 + port_ret)
            peak_equity = max(peak_equity, equity)
            current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

            combined_returns.append(port_ret)
            equal_weight_returns.append(eq_ret)

            daily_curve.append(
                DailyAllocationPoint(
                    date=all_dates[t],
                    regime=regimes[t].value,
                    weights={
                        strategies[i].meta.id: round(capped_weights[i], 6) for i in range(n_strats)
                    },
                    gross_exposure=round(sum(capped_weights), 6),
                    portfolio_return=round(port_ret, 6),
                    equity=round(equity, 2),
                    drawdown=round(current_dd, 6),
                )
            )

        comb_tot, comb_dd, comb_sh = _series_metrics(combined_returns)
        eq_tot, eq_dd, eq_sh = _series_metrics(equal_weight_returns)
        dd_reduction = (eq_dd - comb_dd) if eq_dd > 0 else 0.0

        pairwise_corr: dict[str, float] = {}
        for i in range(n_strats):
            for j in range(i + 1, n_strats):
                key = f"{strategies[i].meta.id} vs {strategies[j].meta.id}"
                pairwise_corr[key] = round(_pearson(strat_rets[i], strat_rets[j]), 4)

        summaries: list[StrategyAllocationSummary] = []
        for i, strat in enumerate(strategies):
            st_tot, st_sh, st_dd, sig_cnt = standalone_stats[i]
            ws = weight_history[i]
            summaries.append(
                StrategyAllocationSummary(
                    strategy_id=strat.meta.id,
                    preferred_regimes=[
                        r.value if hasattr(r, "value") else str(r)
                        for r in strat.meta.preferred_regime
                    ],
                    mean_weight=round(mean(ws), 4) if ws else 0.0,
                    max_weight=round(max(ws), 4) if ws else 0.0,
                    standalone_total_return=round(st_tot, 6),
                    standalone_sharpe=round(st_sh, 4),
                    standalone_max_drawdown=round(st_dd, 6),
                    signal_count=sig_cnt,
                    return_contribution=round(contrib_sums[i], 6),
                )
            )

        return MultiStrategyAllocationReport(
            strategy_ids=[s.meta.id for s in strategies],
            date_range=(all_dates[0], all_dates[-1]),
            trading_days=n_days,
            initial_capital=cfg.initial_capital,
            final_equity=round(equity, 2),
            combined_total_return=round(comb_tot, 6),
            combined_sharpe_ratio=round(comb_sh, 4),
            combined_max_drawdown=round(comb_dd, 6),
            equal_weight_total_return=round(eq_tot, 6),
            equal_weight_max_drawdown=round(eq_dd, 6),
            equal_weight_sharpe_ratio=round(eq_sh, 4),
            diversification_dd_reduction=round(dd_reduction, 6),
            regime_day_counts=dict(sorted(regime_counts.items())),
            pairwise_correlations=pairwise_corr,
            strategies=summaries,
            daily_curve=daily_curve,
        )

    @staticmethod
    def render_markdown(report: MultiStrategyAllocationReport) -> str:
        """Render multi-strategy regime-adaptive portfolio report as Markdown."""
        lines = [
            "# AlphaEvo Regime-Adaptive Portfolio Allocation Report",
            "",
            f"- Window: {report.date_range[0]} to {report.date_range[1]} ({report.trading_days} trading days)",
            f"- Initial Capital: ${report.initial_capital:,.0f} -> Final Equity: **${report.final_equity:,.0f}**",
            f"- Combined Total Return: **{report.combined_total_return:.2%}** (Equal-Weight: {report.equal_weight_total_return:.2%})",
            f"- Combined Max Drawdown: **{report.combined_max_drawdown:.1%}** (Equal-Weight: {report.equal_weight_max_drawdown:.1%})",
            f"- Combined Sharpe: **{report.combined_sharpe_ratio:.2f}** (Equal-Weight: {report.equal_weight_sharpe_ratio:.2f})",
            "",
            "## Strategy Risk Budget Breakdown",
            "",
            "| Strategy | Preferred Regimes | Signals | Mean Wt | Max Wt | Standalone Ret | Standalone DD | Contribution |",
            "|----------|-------------------|---------|---------|--------|----------------|---------------|--------------|",
        ]
        for s in report.strategies:
            pref_txt = ", ".join(s.preferred_regimes) if s.preferred_regimes else "any"
            lines.append(
                f"| `{s.strategy_id}` | {pref_txt} | {s.signal_count} | "
                f"{s.mean_weight:.1%} | {s.max_weight:.1%} | {s.standalone_total_return:.2%} | "
                f"{s.standalone_max_drawdown:.1%} | {s.return_contribution:+.2%} |"
            )
        if report.pairwise_correlations:
            lines += ["", "## Pairwise Daily Return Correlations", ""]
            for pair, rho in report.pairwise_correlations.items():
                lines.append(f"- {pair}: `{rho:+.4f}`")
        lines += [
            "",
            "> Research & educational use only. Simulated portfolio allocations do not constitute investment advice.",
        ]
        return "\n".join(lines) + "\n"


__all__ = [
    "DailyAllocationPoint",
    "MultiStrategyAllocationReport",
    "RegimeAdaptiveAllocator",
    "RiskBudgetConfig",
    "StrategyAllocationSummary",
]
