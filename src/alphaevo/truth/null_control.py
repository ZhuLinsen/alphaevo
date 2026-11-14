"""Two-tier Null-Signal Control for L0 Anti-Delusion verification."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from alphaevo.models.execution import TradeSignal
    from alphaevo.models.strategy import Strategy


class NullDistributionStats(BaseModel):
    """Summary of one null permutation generator."""

    method: str
    n_permutations: int
    mean: float = 0.0
    std: float = 0.0
    threshold_percentile: float = 0.95
    threshold_value: float = 0.0
    p_value: float = 1.0


class NullControlReport(BaseModel):
    """Tier-1 candidate validation against block-shuffle, time-shift, and random nulls."""

    observed_avg_return: float
    observed_signal_count: int
    block_shuffle: NullDistributionStats
    time_shift: NullDistributionStats
    random_entry: NullDistributionStats
    worst_null_threshold: float
    worst_p_value: float
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class SystemNullWorldResult(BaseModel):
    """Outcome of running RSI evolution in one shuffled-return null world."""

    world_idx: int
    seed: int
    promotions: int
    notes: str = ""


class SystemNullAuditReport(BaseModel):
    """Tier-2 system audit proving 0 promotions occur on randomized stock returns."""

    n_worlds: int
    false_promotions: int
    passed: bool
    worlds: list[SystemNullWorldResult] = Field(default_factory=list)


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Compute linear-interpolated quantile on a sorted sequence."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize_null_means(
    method: str,
    observed: float,
    null_means: list[float],
    percentile: float,
) -> NullDistributionStats:
    """Compute empirical mean, std, percentile threshold, and one-sided p-value."""
    if not null_means:
        return NullDistributionStats(
            method=method,
            n_permutations=0,
            threshold_percentile=percentile,
        )
    n = len(null_means)
    mean_v = sum(null_means) / n
    var_v = sum((x - mean_v) ** 2 for x in null_means) / max(1, n - 1)
    std_v = math.sqrt(max(0.0, var_v))
    sorted_v = sorted(null_means)
    thresh = _quantile(sorted_v, percentile)
    beats_or_ties = sum(1 for x in null_means if x >= observed)
    p_val = (beats_or_ties + 1) / (n + 1)
    return NullDistributionStats(
        method=method,
        n_permutations=n,
        mean=round(mean_v, 6),
        std=round(std_v, 6),
        threshold_percentile=percentile,
        threshold_value=round(thresh, 6),
        p_value=round(p_val, 4),
    )


def generate_block_shuffled_market(
    market_data: dict[str, pd.DataFrame],
    *,
    block_size: int = 5,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create a null-market world by block-shuffling daily bar returns per symbol.

    Preserves dates, bar count, volatility clustering, and intrabar high/low spread
    ratios while destroying causal indicator -> future-return alignment.
    """
    rng = random.Random(seed)
    effective_block = max(1, int(block_size))
    shuffled_world: dict[str, pd.DataFrame] = {}

    for symbol, df in market_data.items():
        if df.empty or len(df) < 4 or "close" not in df:
            shuffled_world[symbol] = df.copy()
            continue

        frame = (
            df.sort_values("date").reset_index(drop=True)
            if "date" in df
            else df.reset_index(drop=True)
        )
        closes = [max(1e-6, float(x)) for x in frame["close"].tolist()]
        opens = (
            [max(1e-6, float(x)) for x in frame["open"].tolist()]
            if "open" in frame
            else list(closes)
        )
        highs = (
            [max(1e-6, float(x)) for x in frame["high"].tolist()]
            if "high" in frame
            else list(closes)
        )
        lows = (
            [max(1e-6, float(x)) for x in frame["low"].tolist()] if "low" in frame else list(closes)
        )
        volumes = (
            [max(0.0, float(x)) for x in frame["volume"].tolist()]
            if "volume" in frame
            else [1_000_000.0] * len(closes)
        )

        # Build step descriptors starting at index 1
        steps: list[tuple[float, float, float, float, float]] = []
        for idx in range(1, len(closes)):
            prev_c = closes[idx - 1]
            c = closes[idx]
            o = opens[idx]
            h = max(highs[idx], o, c)
            low_v = min(lows[idx], o, c)
            ret_close = (c / prev_c) - 1.0
            open_rel = (o / prev_c) - 1.0
            high_rel = (h / max(o, c)) - 1.0
            low_rel = 1.0 - (low_v / min(o, c))
            steps.append((open_rel, ret_close, high_rel, low_rel, volumes[idx]))

        blocks = [steps[i : i + effective_block] for i in range(0, len(steps), effective_block)]
        rng.shuffle(blocks)
        flat_steps = [step for blk in blocks for step in blk]

        new_rows: list[dict[str, object]] = []
        dates = frame["date"].tolist() if "date" in frame else list(range(len(closes)))
        prev_close = closes[0]
        new_rows.append(
            {
                "date": dates[0],
                "open": round(opens[0], 4),
                "high": round(highs[0], 4),
                "low": round(lows[0], 4),
                "close": round(prev_close, 4),
                "volume": int(volumes[0]),
                "prev_close": round(prev_close, 4),
            }
        )
        for idx, (open_rel, ret_close, high_rel, low_rel, vol) in enumerate(flat_steps, start=1):
            new_close = max(0.01, prev_close * (1.0 + ret_close))
            new_open = max(0.01, prev_close * (1.0 + open_rel))
            new_high = max(new_open, new_close) * (1.0 + max(0.0, high_rel))
            new_low = max(0.01, min(new_open, new_close) * (1.0 - min(0.95, max(0.0, low_rel))))
            new_rows.append(
                {
                    "date": dates[idx],
                    "open": round(new_open, 4),
                    "high": round(new_high, 4),
                    "low": round(new_low, 4),
                    "close": round(new_close, 4),
                    "volume": int(vol),
                    "prev_close": round(prev_close, 4),
                }
            )
            prev_close = new_close

        shuffled_world[symbol] = pd.DataFrame(new_rows)

    return shuffled_world


class NullSignalController:
    """Evaluates candidates against shuffled/shifted/random null distributions."""

    def evaluate_candidate_against_nulls(
        self,
        signals: list[TradeSignal],
        market_data: dict[str, pd.DataFrame] | None = None,
        *,
        n_permutations: int = 80,
        percentile: float = 0.95,
        block_size: int = 5,
        min_signals: int = 5,
        seed: int = 42,
    ) -> NullControlReport:
        """Compare strategy trade returns against three null baselines."""
        executed = [s for s in signals if s.exit_price is not None]
        if len(executed) < min_signals:
            empty_stats = NullDistributionStats(
                method="insufficient",
                n_permutations=0,
                threshold_percentile=percentile,
            )
            return NullControlReport(
                observed_avg_return=0.0,
                observed_signal_count=len(executed),
                block_shuffle=empty_stats.model_copy(update={"method": "block_shuffle"}),
                time_shift=empty_stats.model_copy(update={"method": "time_shift"}),
                random_entry=empty_stats.model_copy(update={"method": "random_entry"}),
                worst_null_threshold=0.0,
                worst_p_value=1.0,
                passed=False,
                reasons=[
                    f"Insufficient executed signals ({len(executed)} < {min_signals}) for null control"
                ],
            )

        rng = random.Random(seed)
        obs_returns = [float(s.return_pct) for s in executed]
        observed_mean = sum(obs_returns) / len(obs_returns)
        avg_hold = max(1, round(sum(max(1, s.holding_days) for s in executed) / len(executed)))
        n_sigs = len(executed)

        # Precompute available market return pools for matched-horizon sampling
        daily_return_pools: dict[str, list[float]] = {}
        horizon_windows: dict[str, list[float]] = {}
        if market_data:
            for sym, df in market_data.items():
                if df.empty or "close" not in df or len(df) < avg_hold + 3:
                    continue
                frame = df.sort_values("date").reset_index(drop=True) if "date" in df else df
                closes = [max(1e-6, float(c)) for c in frame["close"].tolist()]
                daily_return_pools[sym] = [
                    (closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))
                ]
                horizon_windows[sym] = [
                    (closes[i + avg_hold] / closes[i]) - 1.0
                    for i in range(0, len(closes) - avg_hold)
                ]

        block_means: list[float] = []
        shift_means: list[float] = []
        random_means: list[float] = []

        # 1. Block-shuffle null
        if daily_return_pools:
            all_daily_blocks: list[list[float]] = []
            for sym_rets in daily_return_pools.values():
                for i in range(0, len(sym_rets), block_size):
                    blk = sym_rets[i : i + block_size]
                    if blk:
                        all_daily_blocks.append(blk)
            for _ in range(n_permutations):
                synth_trade_rets: list[float] = []
                for _k in range(n_sigs):
                    prod = 1.0
                    steps_needed = avg_hold
                    while steps_needed > 0 and all_daily_blocks:
                        blk = rng.choice(all_daily_blocks)
                        take = blk[:steps_needed]
                        for r in take:
                            prod *= 1.0 + r
                        steps_needed -= len(take)
                    synth_trade_rets.append(prod - 1.0)
                block_means.append(sum(synth_trade_rets) / n_sigs)
        else:
            # Zero-centered sign-flip/block permutation fallback when no OHLCV supplied
            centered = [r - observed_mean for r in obs_returns]
            for _ in range(n_permutations):
                sampled = [rng.choice(centered) * rng.choice((-1.0, 1.0)) for _ in range(n_sigs)]
                block_means.append(sum(sampled) / n_sigs)

        # 2. Time-shift null (shift entry indices away from original dates)
        if horizon_windows:
            eligible_syms = [s for s, wins in horizon_windows.items() if wins]
            for _ in range(n_permutations):
                shifted_rets: list[float] = []
                for sig in executed:
                    wins = horizon_windows.get(sig.symbol)
                    if wins and len(wins) > 1:
                        shifted_rets.append(rng.choice(wins))
                    elif eligible_syms:
                        shifted_rets.append(rng.choice(horizon_windows[rng.choice(eligible_syms)]))
                    else:
                        shifted_rets.append(0.0)
                shift_means.append(sum(shifted_rets) / len(shifted_rets))
        else:
            shift_means = list(block_means)

        # 3. Random-entry null across whole market pool
        flat_horizon_pool = [r for wins in horizon_windows.values() for r in wins]
        if flat_horizon_pool:
            for _ in range(n_permutations):
                picks = [rng.choice(flat_horizon_pool) for _ in range(n_sigs)]
                random_means.append(sum(picks) / n_sigs)
        else:
            random_means = list(block_means)

        block_stats = _summarize_null_means("block_shuffle", observed_mean, block_means, percentile)
        shift_stats = _summarize_null_means("time_shift", observed_mean, shift_means, percentile)
        rand_stats = _summarize_null_means("random_entry", observed_mean, random_means, percentile)

        worst_thresh = max(
            block_stats.threshold_value,
            shift_stats.threshold_value,
            rand_stats.threshold_value,
        )
        worst_p = max(block_stats.p_value, shift_stats.p_value, rand_stats.p_value)

        reasons: list[str] = []
        if observed_mean <= 0.0:
            reasons.append(f"Observed avg return {observed_mean:.2%} is non-positive")
        if observed_mean <= worst_thresh:
            reasons.append(
                f"Observed avg return {observed_mean:.2%} fails to beat null p{int(percentile * 100)} threshold {worst_thresh:.2%}"
            )

        return NullControlReport(
            observed_avg_return=round(observed_mean, 6),
            observed_signal_count=n_sigs,
            block_shuffle=block_stats,
            time_shift=shift_stats,
            random_entry=rand_stats,
            worst_null_threshold=round(worst_thresh, 6),
            worst_p_value=round(worst_p, 4),
            passed=len(reasons) == 0,
            reasons=reasons,
        )

    def verify_zero_promotion_under_shuffled_returns(
        self,
        strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        runner_fn: Callable[[Strategy, dict[str, pd.DataFrame]], int],
        *,
        n_worlds: int = 3,
        block_size: int = 5,
        seed: int = 42,
    ) -> SystemNullAuditReport:
        """Tier-2 audit: verify RSI engine produces 0 promotions on shuffled returns."""
        worlds: list[SystemNullWorldResult] = []
        total_promotions = 0
        for idx in range(max(1, n_worlds)):
            world_seed = seed + idx * 97
            null_market = generate_block_shuffled_market(
                market_data,
                block_size=block_size,
                seed=world_seed,
            )
            promos = int(runner_fn(strategy, null_market))
            total_promotions += max(0, promos)
            worlds.append(
                SystemNullWorldResult(
                    world_idx=idx + 1,
                    seed=world_seed,
                    promotions=promos,
                    notes="passed"
                    if promos == 0
                    else f"LEAK: {promos} promotion(s) on shuffled returns",
                )
            )
        return SystemNullAuditReport(
            n_worlds=len(worlds),
            false_promotions=total_promotions,
            passed=(total_promotions == 0),
            worlds=worlds,
        )


__all__ = [
    "NullControlReport",
    "NullDistributionStats",
    "NullSignalController",
    "SystemNullAuditReport",
    "SystemNullWorldResult",
    "generate_block_shuffled_market",
]
