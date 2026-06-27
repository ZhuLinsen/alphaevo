"""Evaluator — computes multi-dimensional metrics from backtest results.

Produces OverallMetrics, AntiFitMetrics, and the composite confidence_score.
Formula matches AGENTS.md §10 and module_tech_specs.md §6.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from itertools import combinations
from statistics import mean, median, stdev
from typing import Any, Literal

from alphaevo.evaluator.benchmark import BenchmarkComparator
from alphaevo.models.enums import MarketRegime
from alphaevo.models.execution import (
    AntiFitMetrics,
    BacktestResult,
    CPCVMetrics,
    EvaluationReport,
    EventContextMetrics,
    OverallMetrics,
    RegimeHoldoutCase,
    RegimeHoldoutMetrics,
    RegimeMetrics,
    TradeSignal,
    WalkForwardFoldMetrics,
    WalkForwardProtocol,
)
from alphaevo.models.strategy import Strategy
from alphaevo.strategy.tunable import resolve_tunable_target, set_tunable_target

# Minimum signals per split to be meaningful
_MIN_SPLIT_SIGNALS = 5
_MIN_PARAM_SENSITIVITY_SIGNALS = 30
_EVENT_INDICATORS = {
    "negative_news_score",
    "news_sentiment_score",
    "days_since_event",
    "price_above_pre_event",
    "already_overreacted",
}
EvaluationMode = Literal["fast", "full"]


class Evaluator:
    """Compute evaluation metrics from backtest results."""

    def evaluate_fast(
        self,
        result: BacktestResult,
        strategy: Strategy,
        *,
        contexts: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Fast candidate evaluation for optimizer search loops.

        This keeps the trade-level metrics used by qualification gates but skips
        benchmark, stress-window, walk-forward, CPCV, and other heavy diagnostics.
        Full reports should still use :meth:`evaluate`.
        """
        executed = [s for s in result.signals if s.exit_price is not None]
        overall = self.compute_metrics(executed)
        by_regime = self.compute_metrics_by_regime(executed)
        by_sector = self.compute_metrics_by_sector(executed)
        anti_fit = self.compute_anti_overfit(executed)
        sorted_by_return = sorted(executed, key=lambda s: s.return_pct)
        event_context = self.compute_event_context_metrics(strategy, contexts or {})
        confidence = self.compute_confidence_score(
            overall,
            strategy.complexity_score,
            anti_fit=anti_fit,
        )

        return EvaluationReport(
            strategy_id=result.strategy_id,
            batch_id=result.batch_id,
            overall=overall,
            by_regime=by_regime,
            by_sector=by_sector,
            anti_overfit=anti_fit,
            failure_cases=sorted_by_return[:10],
            top_patterns=self.compute_top_patterns(
                by_regime,
                by_sector,
                event_context=event_context,
            ),
            confidence_score=confidence,
            event_context=event_context,
        )

    def evaluate(
        self,
        result: BacktestResult,
        strategy: Strategy,
        *,
        market_data: dict[str, Any] | None = None,
        contexts: dict[str, Any] | None = None,
        backtest_config: Any = None,
    ) -> EvaluationReport:
        """Full evaluation: metrics + confidence score + failure cases."""
        executed = [s for s in result.signals if s.exit_price is not None]

        overall = self.compute_metrics(executed)
        by_regime = self.compute_metrics_by_regime(executed)
        by_sector = self.compute_metrics_by_sector(executed)

        # Anti-overfit: train/val/test split + yearly consistency
        anti_fit = self.compute_anti_overfit(executed)

        requested_folds = int(getattr(backtest_config, "walk_forward_folds", 3))
        train_pct = float(getattr(backtest_config, "walk_forward_train_pct", 0.7))
        pass_gap = float(getattr(backtest_config, "walk_forward_pass_gap", 0.10))
        walk_forward, protocol = self.compute_walk_forward(
            executed,
            requested_folds=requested_folds,
            train_pct=train_pct,
            pass_gap=pass_gap,
        )
        if walk_forward:
            anti_fit.walk_forward_gap = round(
                mean(fold.gap for fold in walk_forward),
                4,
            )
            anti_fit.walk_forward_pass_rate = round(
                sum(1 for fold in walk_forward if fold.gap <= pass_gap) / len(walk_forward),
                4,
            )

        regime_holdout = self.compute_regime_holdout(
            executed,
            preferred_regimes=strategy.meta.preferred_regime,
            pass_gap=pass_gap,
        )

        # CPCV — combinatorial purged cross-validation
        cpcv = self.compute_cpcv(executed)

        # Identify worst failures (bottom 10 by return)
        sorted_by_return = sorted(executed, key=lambda s: s.return_pct)
        failure_cases = sorted_by_return[:10]

        event_context = self.compute_event_context_metrics(strategy, contexts or {})

        benchmark = None
        stress_windows = None
        if market_data:
            comparator = BenchmarkComparator(
                stress_window_days=int(getattr(backtest_config, "stress_window_days", 20)),
                stress_window_top_k=int(getattr(backtest_config, "stress_window_top_k", 3)),
            )
            benchmark_df = self._resolve_benchmark_df(contexts or {})
            benchmark_result = comparator.compare(
                executed,
                market_data,
                benchmark_df=benchmark_df,
            )
            benchmark = benchmark_result.buy_hold
            random_baseline = benchmark_result.random_baseline
            if random_baseline is not None:
                benchmark.random_baseline_mean = random_baseline.mean_return
                benchmark.random_baseline_std = random_baseline.std_return
                benchmark.random_baseline_ci_lower = random_baseline.ci_lower
                benchmark.random_baseline_ci_upper = random_baseline.ci_upper
                benchmark.random_baseline_beat_fraction = random_baseline.beat_fraction
            stress_windows = benchmark_result.stress_windows

        top_patterns = self.compute_top_patterns(
            by_regime,
            by_sector,
            benchmark_excess_return=benchmark.excess_return if benchmark else None,
            event_context=event_context,
        )

        confidence = self.compute_confidence_score(
            overall,
            strategy.complexity_score,
            anti_fit=anti_fit,
        )

        return EvaluationReport(
            strategy_id=result.strategy_id,
            batch_id=result.batch_id,
            overall=overall,
            by_regime=by_regime,
            by_sector=by_sector,
            anti_overfit=anti_fit,
            failure_cases=failure_cases,
            top_patterns=top_patterns,
            confidence_score=confidence,
            benchmark=benchmark,
            event_context=event_context,
            walk_forward=walk_forward,
            walk_forward_protocol=protocol,
            regime_holdout=regime_holdout,
            stress_windows=stress_windows,
            cpcv=cpcv,
        )

    def compute_metrics(self, signals: list[TradeSignal]) -> OverallMetrics:
        """Compute aggregate metrics from executed trade signals."""
        if not signals:
            return OverallMetrics()

        returns = [s.return_pct for s in signals]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        holding_days = [s.holding_days for s in signals if s.holding_days > 0]

        win_rate = len(wins) / len(returns) if returns else 0.0
        avg_return = mean(returns) if returns else 0.0
        median_return = median(returns) if returns else 0.0

        avg_win = mean(wins) if wins else 0.0
        avg_loss = abs(mean(losses)) if losses else 0.001
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        max_dd = self._max_drawdown(returns)
        sharpe = self._sharpe_ratio(returns)
        max_consec_loss = self._max_consecutive_loss(returns)
        total_return = self._total_return(returns)

        return OverallMetrics(
            win_rate=round(win_rate, 4),
            avg_return=round(avg_return, 6),
            avg_win_return=round(avg_win, 6),
            avg_loss_return=round(-avg_loss, 6) if losses else 0.0,
            median_return=round(median_return, 6),
            profit_loss_ratio=round(profit_loss_ratio, 4),
            max_drawdown=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 4),
            signal_count=len(signals),
            avg_holding_days=round(mean(holding_days), 1) if holding_days else 0.0,
            max_consecutive_loss=max_consec_loss,
            total_return=round(total_return, 6),
        )

    def compute_confidence_score(
        self,
        metrics: OverallMetrics,
        complexity_score: float = 0.0,
        anti_fit: AntiFitMetrics | None = None,
    ) -> float:
        """Compute composite confidence score (0.0 ~ 1.0).

        Formula matches AGENTS.md §10:
          0.25 * win_rate + 0.15 * avg_return + 0.15 * pl_ratio
          + 0.15 * drawdown + 0.10 * sharpe + 0.10 * consistency
          + 0.10 * sensitivity - overfit_penalty - complexity_penalty
        """
        # Normalize each dimension to [0, 1]
        wr_score = min(1.0, metrics.win_rate / 0.7)
        ret_score = min(1.0, max(0, metrics.avg_return / 0.05))
        pl_score = min(1.0, metrics.profit_loss_ratio / 2.5)
        dd_score = max(0.0, 1.0 - metrics.max_drawdown / 0.30)
        sharpe_score = min(1.0, max(0, metrics.sharpe_ratio / 2.0))

        # Anti-overfit scores
        consistency = 0.5
        sensitivity = 0.5
        overfit_penalty = 0.0

        if anti_fit:
            consistency = max(0.0, anti_fit.yearly_consistency)
            sensitivity = max(0.0, 1.0 - anti_fit.param_sensitivity)
            if anti_fit.train_val_gap > 0.10:
                overfit_penalty += 0.15
            if anti_fit.val_test_gap > 0.08:
                overfit_penalty += 0.10

        complexity_penalty = complexity_score * 0.10

        # Signal count reliability: < 30 signals are unreliable (AGENTS.md §11)
        # < 10 signals: hard cap at 0.15 to prevent lucky-streak ranking
        signal_reliability = 1.0
        if metrics.signal_count < 10:
            signal_reliability = 0.3
        elif metrics.signal_count < 30:
            signal_reliability = max(0.5, metrics.signal_count / 30)

        score = (
            0.25 * wr_score
            + 0.15 * ret_score
            + 0.15 * pl_score
            + 0.15 * dd_score
            + 0.10 * sharpe_score
            + 0.10 * consistency
            + 0.10 * sensitivity
            - overfit_penalty
            - complexity_penalty
        ) * signal_reliability

        # Hard cap for very small samples to prevent noise from dominating
        if metrics.signal_count < 10:
            score = min(score, 0.15)

        return round(max(0.0, min(1.0, score)), 4)

    def compute_metrics_by_regime(
        self,
        signals: list[TradeSignal],
    ) -> list[RegimeMetrics]:
        """Compute performance segmented by market regime."""
        grouped: dict[MarketRegime, list[TradeSignal]] = defaultdict(list)
        for signal in signals:
            if signal.regime is not None:
                grouped[signal.regime].append(signal)

        metrics: list[RegimeMetrics] = []
        for regime, regime_signals in grouped.items():
            overall = self.compute_metrics(regime_signals)
            metrics.append(
                RegimeMetrics(
                    regime=regime,
                    win_rate=overall.win_rate,
                    avg_return=overall.avg_return,
                    signal_count=overall.signal_count,
                )
            )
        return sorted(metrics, key=lambda item: item.signal_count, reverse=True)

    def compute_metrics_by_sector(
        self,
        signals: list[TradeSignal],
    ) -> dict[str, OverallMetrics]:
        """Compute performance segmented by sector."""
        grouped: dict[str, list[TradeSignal]] = defaultdict(list)
        for signal in signals:
            if signal.sector:
                grouped[signal.sector].append(signal)
        return {
            sector: self.compute_metrics(sector_signals)
            for sector, sector_signals in sorted(grouped.items())
        }

    def compute_walk_forward(
        self,
        signals: list[TradeSignal],
        *,
        requested_folds: int = 3,
        train_pct: float = 0.7,
        pass_gap: float = 0.10,
    ) -> tuple[list[WalkForwardFoldMetrics], WalkForwardProtocol]:
        """Compute an expanding-window walk-forward diagnostic over trade signals."""
        protocol = WalkForwardProtocol(
            requested_folds=requested_folds,
            effective_folds=0,
            train_pct=train_pct,
            test_pct=round(1.0 - train_pct, 4),
            pass_gap=pass_gap,
            min_signals_per_split=_MIN_SPLIT_SIGNALS,
        )
        if len(signals) < _MIN_SPLIT_SIGNALS * 2:
            return [], protocol

        ordered = sorted(signals, key=lambda signal: signal.signal_date)
        min_train = max(_MIN_SPLIT_SIGNALS, int(len(ordered) * train_pct))
        remaining = len(ordered) - min_train
        if remaining < _MIN_SPLIT_SIGNALS:
            return [], protocol

        fold_test_size = max(_MIN_SPLIT_SIGNALS, remaining // max(requested_folds, 1))
        folds: list[WalkForwardFoldMetrics] = []
        train_end = min_train
        for fold_num in range(1, requested_folds + 1):
            if train_end + _MIN_SPLIT_SIGNALS > len(ordered):
                break

            test_end = min(len(ordered), train_end + fold_test_size)
            train_set = ordered[:train_end]
            test_set = ordered[train_end:test_end]
            if len(train_set) < _MIN_SPLIT_SIGNALS or len(test_set) < _MIN_SPLIT_SIGNALS:
                break

            train_wr = self._split_win_rate(train_set)
            test_wr = self._split_win_rate(test_set)
            folds.append(
                WalkForwardFoldMetrics(
                    fold_num=fold_num,
                    train_signal_count=len(train_set),
                    test_signal_count=len(test_set),
                    train_win_rate=round(train_wr, 4),
                    test_win_rate=round(test_wr, 4),
                    gap=round(abs(train_wr - test_wr), 4),
                )
            )
            train_end = test_end

        protocol.effective_folds = len(folds)
        return folds, protocol

    def compute_regime_holdout(
        self,
        signals: list[TradeSignal],
        *,
        preferred_regimes: list[str],
        pass_gap: float = 0.10,
    ) -> RegimeHoldoutMetrics | None:
        """Leave one regime out to assess cross-regime robustness."""
        grouped: dict[MarketRegime, list[TradeSignal]] = defaultdict(list)
        for signal in signals:
            if signal.regime is not None:
                grouped[signal.regime].append(signal)

        if len(grouped) < 2:
            return None

        cases: list[RegimeHoldoutCase] = []
        for regime, holdout in grouped.items():
            in_sample = [
                signal
                for signal in signals
                if signal.regime is not None and signal.regime != regime
            ]
            if len(holdout) < _MIN_SPLIT_SIGNALS or len(in_sample) < _MIN_SPLIT_SIGNALS:
                continue

            in_sample_wr = self._split_win_rate(in_sample)
            holdout_wr = self._split_win_rate(holdout)
            gap = abs(in_sample_wr - holdout_wr)
            cases.append(
                RegimeHoldoutCase(
                    regime=regime,
                    preferred=regime.value in set(preferred_regimes),
                    in_sample_signal_count=len(in_sample),
                    holdout_signal_count=len(holdout),
                    in_sample_win_rate=round(in_sample_wr, 4),
                    holdout_win_rate=round(holdout_wr, 4),
                    gap=round(gap, 4),
                )
            )

        if not cases:
            return None

        worst_case = max(cases, key=lambda case: case.gap)
        pass_rate = sum(1 for case in cases if case.gap <= pass_gap) / len(cases)
        return RegimeHoldoutMetrics(
            preferred_regimes=list(preferred_regimes),
            pass_gap=pass_gap,
            total_cases=len(cases),
            pass_rate=round(pass_rate, 4),
            worst_gap=round(worst_case.gap, 4),
            worst_regime=worst_case.regime,
            holdouts=cases,
        )

    def compute_event_context_metrics(
        self,
        strategy: Strategy,
        contexts: dict[str, object],
    ) -> EventContextMetrics | None:
        """Summarize event/news context coverage for the current run."""
        if not contexts:
            return None

        relevant = sorted(self._strategy_indicators(strategy) & _EVENT_INDICATORS)
        total = len(contexts)
        provider_symbols = 0
        mixed_symbols = 0
        proxy_symbols = 0
        source_breakdown: dict[str, int] = defaultdict(int)

        for ctx in contexts.values():
            source = getattr(ctx, "event_context_source", None) or "proxy"
            source_breakdown[source] += 1
            if source == "proxy":
                proxy_symbols += 1
                continue
            if "+proxy" in source:
                mixed_symbols += 1
                continue
            provider_symbols += 1

        return EventContextMetrics(
            total_symbols=total,
            provider_symbols=provider_symbols,
            mixed_symbols=mixed_symbols,
            proxy_symbols=proxy_symbols,
            provider_coverage=round((provider_symbols + mixed_symbols) / total, 4),
            proxy_only_coverage=round(proxy_symbols / total, 4),
            relevant_indicators=relevant,
            source_breakdown=dict(source_breakdown),
        )

    @staticmethod
    def compute_top_patterns(
        by_regime: list[RegimeMetrics],
        by_sector: dict[str, OverallMetrics],
        *,
        benchmark_excess_return: float | None = None,
        event_context: EventContextMetrics | None = None,
    ) -> list[str]:
        """Generate short reusable research takeaways from the evaluation."""
        patterns: list[str] = []
        if by_regime:
            strongest_regime = max(by_regime, key=lambda item: (item.avg_return, item.win_rate))
            weakest_regime = min(by_regime, key=lambda item: (item.avg_return, item.win_rate))
            patterns.append(
                f"Best regime fit: {strongest_regime.regime.value} "
                f"(win_rate={strongest_regime.win_rate:.1%}, signals={strongest_regime.signal_count})"
            )
            if weakest_regime.regime != strongest_regime.regime:
                patterns.append(
                    f"Weak regime: {weakest_regime.regime.value} "
                    f"(win_rate={weakest_regime.win_rate:.1%}, signals={weakest_regime.signal_count})"
                )
        if by_sector:
            strongest_sector = max(
                by_sector.items(),
                key=lambda item: (item[1].avg_return, item[1].win_rate),
            )
            patterns.append(
                f"Sector edge: {strongest_sector[0]} "
                f"(win_rate={strongest_sector[1].win_rate:.1%}, signals={strongest_sector[1].signal_count})"
            )
        if benchmark_excess_return is not None:
            relation = "outperformed" if benchmark_excess_return > 0 else "lagged"
            patterns.append(f"Buy-and-hold {relation} by {abs(benchmark_excess_return):.2%}")
        if event_context and event_context.relevant_indicators:
            if event_context.is_proxy_dominant:
                patterns.append(
                    "Data-quality remediation should precede event/news mutation: "
                    f"provider coverage {event_context.provider_coverage:.1%}, "
                    f"proxy-only coverage {event_context.proxy_only_coverage:.1%}"
                )
            else:
                patterns.append(
                    f"Event-aware indicators active with {event_context.provider_coverage:.1%} provider coverage"
                )
        return patterns[:5]

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float:
        """Max drawdown from sequential trade returns."""
        if not returns:
            return 0.0
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= 1 + r
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _sharpe_ratio(returns: list[float], risk_free: float = 0.0) -> float:
        """Annualized Sharpe ratio (assuming ~250 trades/year)."""
        if len(returns) < 2:
            return 0.0
        avg = mean(returns) - risk_free
        std = stdev(returns)
        if std == 0:
            return 0.0
        # Scale factor: sqrt(trades per year estimate)
        return avg / std * math.sqrt(min(len(returns), 250))

    @staticmethod
    def _max_consecutive_loss(returns: list[float]) -> int:
        """Maximum consecutive losing trades."""
        max_streak = 0
        current = 0
        for r in returns:
            if r <= 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def _total_return(returns: list[float]) -> float:
        """Cumulative return from sequential trades."""
        equity = 1.0
        for r in returns:
            equity *= 1 + r
        return equity - 1.0

    # ── Anti-overfit analysis ─────────────────────────────────────────

    def compute_anti_overfit(
        self,
        signals: list[TradeSignal],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> AntiFitMetrics:
        """Compute train/val/test split metrics and yearly consistency.

        Signals are split *chronologically* by signal_date:
          - train: first 60% of signals
          - val: next 20%
          - test: last 20%

        If there are too few signals to split meaningfully, returns defaults
        with neutral values that won't trigger false overfit alarms.
        """
        if len(signals) < _MIN_SPLIT_SIGNALS * 3:
            # Not enough data for a meaningful 3-way split
            return AntiFitMetrics(yearly_consistency=0.5)

        # Sort by signal date for chronological split
        sorted_signals = sorted(signals, key=lambda s: s.signal_date)
        n = len(sorted_signals)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_set = sorted_signals[:train_end]
        val_set = sorted_signals[train_end:val_end]
        test_set = sorted_signals[val_end:]

        # Compute win rates for each split
        train_wr = self._split_win_rate(train_set)
        val_wr = self._split_win_rate(val_set)
        test_wr = self._split_win_rate(test_set)

        train_val_gap = abs(train_wr - val_wr)
        val_test_gap = abs(val_wr - test_wr)

        # Yearly consistency: group by year, compute std/mean of win rates
        yearly_consistency = self._yearly_consistency(sorted_signals)

        complexity_penalty = 0.0

        return AntiFitMetrics(
            train_win_rate=round(train_wr, 4),
            val_win_rate=round(val_wr, 4),
            test_win_rate=round(test_wr, 4),
            train_val_gap=round(train_val_gap, 4),
            val_test_gap=round(val_test_gap, 4),
            yearly_consistency=round(yearly_consistency, 4),
            param_sensitivity=0.0,  # Requires perturbation test (future)
            complexity_penalty=round(complexity_penalty, 4),
        )

    @staticmethod
    def _split_win_rate(signals: list[TradeSignal]) -> float:
        """Win rate for a subset of signals."""
        if not signals:
            return 0.0
        wins = sum(1 for s in signals if s.return_pct > 0)
        return wins / len(signals)

    @staticmethod
    def _yearly_consistency(signals: list[TradeSignal]) -> float:
        """Compute yearly consistency: 1 - std/mean of per-year win rates.

        Returns 0.5 (neutral) if there aren't enough years to compare.
        """
        by_year: dict[int, list[TradeSignal]] = defaultdict(list)
        for s in signals:
            by_year[s.signal_date.year].append(s)

        # Need at least 2 years with meaningful sample sizes
        yearly_wr: list[float] = []
        for year_signals in by_year.values():
            if len(year_signals) >= _MIN_SPLIT_SIGNALS:
                wins = sum(1 for s in year_signals if s.return_pct > 0)
                yearly_wr.append(wins / len(year_signals))

        if len(yearly_wr) < 2:
            return 0.5  # neutral — not enough data to assess consistency

        avg = mean(yearly_wr)
        if avg == 0:
            return 0.0
        std = stdev(yearly_wr)
        return max(0.0, min(1.0, 1.0 - std / avg))

    def compute_param_sensitivity(
        self,
        strategy: Strategy,
        signals: list[TradeSignal],
        base_score: float,
        engine: Any,
        data: dict[str, Any],
        batch: Any,
        contexts: dict[str, Any] | None = None,
    ) -> float:
        """Estimate parameter sensitivity by perturbing tunable params ±10%.

        For each tunable parameter, creates a variant strategy with ±10% change,
        re-runs backtest, and measures score degradation. Returns the average
        degradation ratio (0.0 = robust, 1.0 = extremely fragile).

        This is opt-in and called from the evolution pipeline rather than
        evaluate() to avoid circular imports and doubled backtest cost.
        """
        if not strategy.params.tunable or base_score <= 0:
            return 0.0

        # Sparse-signal strategies are too noisy for a meaningful perturbation
        # study, and rerunning the backtest dozens of times only slows real LLM
        # evolution without adding reliable evidence.
        if len(signals) < _MIN_PARAM_SENSITIVITY_SIGNALS:
            return 0.0

        from alphaevo.backtest.engine import BacktestEngine

        if not isinstance(engine, BacktestEngine):
            return 0.0

        # Recompute base_score without anti_fit for consistent comparison
        # (anti_fit doesn't change with parameter perturbation)
        base_score_raw = self.compute_confidence_score(
            self.compute_metrics([s for s in signals if s.exit_price is not None]),
            strategy.complexity_score,
        )
        if base_score_raw <= 0:
            return 0.0

        degradations: list[float] = []
        for param in strategy.params.tunable:
            current = self._resolve_tunable(strategy, param)
            if current is None or not isinstance(current, (int, float)):
                continue

            for factor in (0.9, 1.1):  # ±10%
                perturbed_val = round(current * factor, 6)
                # Clamp to param range
                lo, hi = param.range
                perturbed_val = max(lo, min(hi, perturbed_val))
                if perturbed_val == current:
                    continue

                # Create variant
                variant = strategy.model_copy(deep=True)
                if not self._set_tunable(variant, param, perturbed_val):
                    continue

                try:
                    result = engine.run(variant, data, batch, contexts=contexts)
                    executed = [s for s in result.signals if s.exit_price is not None]
                    metrics = self.compute_metrics(executed)
                    variant_score = self.compute_confidence_score(
                        metrics,
                        variant.complexity_score,
                    )
                    degradation = max(0, base_score_raw - variant_score) / base_score_raw
                    degradations.append(degradation)
                except Exception:
                    # If the variant crashes, treat as fragile
                    degradations.append(0.5)

        if not degradations:
            return 0.0
        return round(mean(degradations), 4)

    # ── Combinatorial Purged Cross-Validation (CPCV) ────────────────

    def compute_cpcv(
        self,
        signals: list[TradeSignal],
        *,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_days: int = 5,
    ) -> CPCVMetrics | None:
        """Combinatorial Purged Cross-Validation (CPCV).

        Splits signals chronologically into *n_groups* groups.  For every
        ``C(n_groups, n_test_groups)`` combination, removes signals in the test
        groups PLUS a purge window around the train/test boundary to eliminate
        look-ahead contamination, then measures train vs test win-rate gap.

        Returns aggregated metrics, or ``None`` when data is insufficient.
        """
        if len(signals) < n_groups * _MIN_SPLIT_SIGNALS:
            return None

        ordered = sorted(signals, key=lambda s: s.signal_date)
        # Split into chronological groups
        group_size = len(ordered) // n_groups
        if group_size < _MIN_SPLIT_SIGNALS:
            return None

        groups: list[list[TradeSignal]] = []
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else len(ordered)
            groups.append(ordered[start:end])

        # Pre-compute group date boundaries for purging
        group_boundaries: list[tuple[date, date]] = []
        for g in groups:
            group_boundaries.append((g[0].signal_date, g[-1].signal_date))

        gaps: list[float] = []
        test_win_rates: list[float] = []
        for test_indices in combinations(range(n_groups), n_test_groups):
            test_set: list[TradeSignal] = []
            for ti in test_indices:
                test_set.extend(groups[ti])

            # Build train set with purging
            test_date_ranges = [group_boundaries[ti] for ti in test_indices]
            train_set: list[TradeSignal] = []
            for i, g in enumerate(groups):
                if i in test_indices:
                    continue
                for sig in g:
                    if self._is_purged(sig.signal_date, test_date_ranges, purge_days):
                        continue
                    train_set.append(sig)

            if len(train_set) < _MIN_SPLIT_SIGNALS or len(test_set) < _MIN_SPLIT_SIGNALS:
                continue

            train_wr = self._split_win_rate(train_set)
            test_wr = self._split_win_rate(test_set)
            gap = abs(train_wr - test_wr)
            gaps.append(gap)
            test_win_rates.append(test_wr)

        if not gaps:
            return None

        return CPCVMetrics(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_days=purge_days,
            n_paths=len(gaps),
            mean_gap=round(mean(gaps), 4),
            max_gap=round(max(gaps), 4),
            mean_test_win_rate=round(mean(test_win_rates), 4),
            std_test_win_rate=round(stdev(test_win_rates), 4) if len(test_win_rates) > 1 else 0.0,
        )

    @staticmethod
    def _is_purged(
        sig_date: date,
        test_ranges: list[tuple[date, date]],
        purge_days: int,
    ) -> bool:
        """Return True if *sig_date* falls within the purge window around any test range."""
        purge = timedelta(days=purge_days)
        return any(start - purge <= sig_date <= end + purge for start, end in test_ranges)

    # ── Strategy Fingerprint (deduplication) ──────────────────────────

    @staticmethod
    def compute_strategy_fingerprint(strategy: Strategy) -> str:
        """Compute a content-based fingerprint for strategy deduplication.

        Two strategies with the same entry conditions, exit rules, and
        universe (ignoring meta.id/name/version) produce the same hash.
        """
        import hashlib

        parts: list[str] = []
        # Entry conditions (sorted for stability)
        parts.append(f"logic={strategy.entry.logic}")
        for c in sorted(strategy.entry.triggers, key=lambda x: x.indicator):
            parts.append(f"et:{c.indicator}{c.op}{c.value}")
        for c in sorted(strategy.entry.conditions, key=lambda x: x.indicator):
            parts.append(f"ec:{c.indicator}{c.op}{c.value}")
        for g in sorted(strategy.entry.guards, key=lambda x: x.indicator):
            parts.append(f"eg:{g.indicator}{g.op}{g.value}")
        for f in sorted(strategy.entry.filters, key=lambda x: x.indicator):
            parts.append(f"ef:{f.indicator}{f.op}{f.value}")
        # Exit
        for trigger in sorted(strategy.exit.triggers, key=lambda x: x.indicator):
            parts.append(f"xt:{trigger.indicator}{trigger.op}{trigger.value}")
        sl = strategy.exit.stop_loss
        parts.append(f"sl:{sl.type}:{sl.value}:{sl.multiplier}")
        tp = strategy.exit.take_profit
        parts.append(f"tp:{tp.type}:{tp.value}:{tp.target}")
        parts.append(f"mhd:{strategy.exit.max_holding_days}")
        # Universe
        for m in sorted(strategy.universe.market):
            parts.append(f"u:{m}")
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _resolve_benchmark_df(contexts: dict[str, Any]) -> Any:
        """Pick the first available benchmark dataframe from indicator contexts."""
        for ctx in contexts.values():
            benchmark_df = getattr(ctx, "benchmark_df", None)
            if benchmark_df is not None:
                return benchmark_df
        return None

    @staticmethod
    def _strategy_indicators(strategy: Strategy) -> set[str]:
        """Collect indicator names referenced by a strategy."""
        names = {cond.indicator for cond in strategy.entry.conditions}
        names.update(cond.indicator for cond in strategy.entry.triggers)
        names.update(cond.indicator for cond in strategy.entry.filters)
        names.update(cond.indicator for cond in strategy.entry.guards)
        names.update(cond.indicator for cond in strategy.exit.triggers)
        if strategy.exit.stop_loss.conditions:
            names.update(cond.indicator for cond in strategy.exit.stop_loss.conditions)
        return names

    @staticmethod
    def _resolve_tunable(strategy: Strategy, param: Any) -> Any:
        """Resolve current value of a tunable parameter."""
        return resolve_tunable_target(strategy, param.target)

    @staticmethod
    def _set_tunable(strategy: Strategy, param: Any, new_value: Any) -> bool:
        """Set the value of a tunable parameter on a (copy of) strategy."""
        return set_tunable_target(strategy, param.target, new_value)
