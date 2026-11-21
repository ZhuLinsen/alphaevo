"""Exit/risk optimization for executable strategies."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.evaluator.metrics import EvaluationMode, Evaluator
from alphaevo.models.enums import ExitReason, StrategyStatus
from alphaevo.models.execution import BacktestResult, EvaluationReport, SampleBatch, TradeSignal
from alphaevo.models.market import IndicatorContext
from alphaevo.models.strategy import Strategy, StrategyCondition
from alphaevo.optimizer.scoring import (
    OptimizationObjective,
    candidate_sort_key,
    normalize_objective,
    objective_value,
    record_optimizer_trials,
)
from alphaevo.optimizer.summary import (
    format_best_candidate_markdown,
    select_high_win_return_candidate,
)
from alphaevo.strategy.dsl.serializer import StrategySerializer

_DEFAULT_SPACES = ("exit", "stoploss", "takeprofit", "holding")
_SPACE_ALIASES = {
    "sell": "exit",
    "sellsignal": "exit",
    "sell_signal": "exit",
    "roi": "takeprofit",
    "tp": "takeprofit",
    "take_profit": "takeprofit",
    "stop": "stoploss",
    "sl": "stoploss",
    "stop_loss": "stoploss",
    "hold": "holding",
    "maxhold": "holding",
    "max_holding": "holding",
}
_STOP_LOSS_GRID = (0.02, 0.03, 0.05, 0.08)
_TAKE_PROFIT_RR_GRID = (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
_TAKE_PROFIT_TRAILING_GRID = (
    (0.04, 0.02),
    (0.06, 0.03),
    (0.08, 0.04),
    (0.10, 0.05),
)
_HOLDING_DAYS_GRID = (2, 3, 5, 10, 20)
_EARLY_EXIT_LOOKAHEAD_DAYS = 5
_MEANINGFUL_MOVE = 0.03
_LATE_GIVEBACK = 0.05
TakeProfitOption = tuple[str, float | None, float | None, float | None]


class ExitDiagnosticSummary(BaseModel):
    """Post-trade diagnostics focused on exit quality."""

    total_trades: int = 0
    exit_reason_counts: dict[str, int] = Field(default_factory=dict)
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    avg_giveback: float = 0.0
    sold_early_count: int = 0
    sold_late_count: int = 0
    stop_loss_effective_count: int = 0
    take_profit_truncated_count: int = 0


class ExitOptimizationCandidate(BaseModel):
    """One tested exit/risk candidate."""

    candidate_id: str
    changes: list[str] = Field(default_factory=list)
    strategy: Strategy
    evaluation: EvaluationReport
    evaluation_mode: EvaluationMode = "fast"
    diagnostics: ExitDiagnosticSummary = Field(default_factory=ExitDiagnosticSummary)
    passed_gate: bool = True
    gate_reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _ExitCandidateWorkItem:
    candidate_strategy: Strategy
    changes: list[str]


class ExitOptimizationResult(BaseModel):
    """Ranked result of an exit/risk optimization run."""

    base_strategy_id: str
    spaces: list[str]
    objective: OptimizationObjective = "confidence"
    evaluation_mode: EvaluationMode = "fast"
    full_eval_top_n: int = 0
    min_win_rate: float | None = None
    min_avg_return: float | None = None
    min_total_return: float | None = None
    min_profit_loss_ratio: float | None = None
    max_drawdown: float | None = None
    min_signals: int | None = None
    reject_overfit: bool = False
    max_train_val_gap: float | None = None
    max_val_test_gap: float | None = None
    max_walk_forward_gap: float | None = None
    min_walk_forward_pass_rate: float | None = None
    candidates: list[ExitOptimizationCandidate] = Field(default_factory=list)
    best_candidate_id: str | None = None

    @property
    def qualified_count(self) -> int:
        """Return the number of candidates that passed all configured gates."""
        return sum(1 for candidate in self.candidates if candidate.passed_gate)

    @property
    def best_candidate(self) -> ExitOptimizationCandidate | None:
        """Return the highest-ranked candidate if available."""
        if not self.best_candidate_id:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == self.best_candidate_id:
                return candidate
        return None


class ExitOptimizer:
    """Batch-search exit, stop, take-profit, and holding configurations."""

    def __init__(
        self,
        *,
        slippage: float = 0.001,
        commission: float = 0.0003,
        min_data_days: int = 30,
        fill_policy: str = "conservative",
        backtest_config: Any = None,
    ) -> None:
        self.slippage = slippage
        self.commission = commission
        self.min_data_days = min_data_days
        self.fill_policy = fill_policy
        self.backtest_config = backtest_config

    def optimize(
        self,
        strategy: Strategy,
        data: dict[str, Any],
        batch: SampleBatch,
        *,
        contexts: dict[str, IndicatorContext] | None = None,
        spaces: Iterable[str] | None = None,
        max_candidates: int = 128,
        objective: OptimizationObjective = "confidence",
        evaluation_mode: EvaluationMode = "fast",
        full_eval_top_n: int = 0,
        min_win_rate: float | None = None,
        min_avg_return: float | None = None,
        min_total_return: float | None = None,
        min_profit_loss_ratio: float | None = None,
        max_drawdown: float | None = None,
        min_signals: int | None = None,
        reject_overfit: bool = False,
        max_train_val_gap: float | None = None,
        max_val_test_gap: float | None = None,
        max_walk_forward_gap: float | None = None,
        min_walk_forward_pass_rate: float | None = None,
        parallel_workers: int = 1,
        trial_ledger: Any = None,
    ) -> ExitOptimizationResult:
        """Evaluate a bounded grid of exit/risk candidates on existing data."""
        normalized_spaces = _normalize_spaces(spaces)
        normalized_objective = _normalize_objective(objective)
        normalized_evaluation_mode = _normalize_evaluation_mode(evaluation_mode)
        work_items: list[_ExitCandidateWorkItem] = []

        for idx, (candidate_strategy, changes) in enumerate(
            self._generate_candidates(strategy, normalized_spaces),
            start=1,
        ):
            if idx > max_candidates:
                break
            candidate_strategy.meta.id = f"{strategy.meta.id}_opt_{idx:03d}"
            candidate_strategy.meta.parent_id = strategy.meta.id
            candidate_strategy.meta.version = strategy.meta.version + 1
            candidate_strategy.meta.status = StrategyStatus.DRAFT

            work_items.append(
                _ExitCandidateWorkItem(
                    candidate_strategy=candidate_strategy,
                    changes=changes,
                )
            )

        runner = partial(
            _evaluate_work_item,
            data=data,
            batch=batch,
            contexts=contexts,
            slippage=self.slippage,
            commission=self.commission,
            min_data_days=self.min_data_days,
            fill_policy=self.fill_policy,
            backtest_config=self.backtest_config,
            mode=normalized_evaluation_mode,
            min_win_rate=min_win_rate,
            min_avg_return=min_avg_return,
            min_total_return=min_total_return,
            min_profit_loss_ratio=min_profit_loss_ratio,
            max_drawdown=max_drawdown,
            min_signals=min_signals,
            reject_overfit=reject_overfit,
            max_train_val_gap=max_train_val_gap,
            max_val_test_gap=max_val_test_gap,
            max_walk_forward_gap=max_walk_forward_gap,
            min_walk_forward_pass_rate=min_walk_forward_pass_rate,
            require_full_for_robust_gates=(
                normalized_evaluation_mode == "fast"
                and _robust_gates_configured(
                    reject_overfit=reject_overfit,
                    max_train_val_gap=max_train_val_gap,
                    max_val_test_gap=max_val_test_gap,
                    max_walk_forward_gap=max_walk_forward_gap,
                    min_walk_forward_pass_rate=min_walk_forward_pass_rate,
                )
            ),
        )
        workers = max(1, int(parallel_workers))
        if workers > 1 and len(work_items) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                evaluated = list(executor.map(runner, work_items))
        else:
            evaluated = [runner(item) for item in work_items]

        candidates = [candidate for candidate, _result in evaluated]
        backtest_results = {candidate.candidate_id: result for candidate, result in evaluated}

        ranked = sorted(
            candidates,
            key=lambda candidate: _candidate_sort_key(candidate, normalized_objective),
            reverse=True,
        )
        full_eval_count = 0
        if normalized_evaluation_mode == "fast" and full_eval_top_n > 0:
            evaluator = Evaluator()
            for candidate in ranked[:full_eval_top_n]:
                backtest_result = backtest_results.get(candidate.candidate_id)
                if backtest_result is None:
                    continue
                candidate.evaluation = evaluator.evaluate(
                    backtest_result,
                    candidate.strategy,
                    market_data=data,
                    contexts=contexts,
                    backtest_config=self.backtest_config,
                    trial_ledger=trial_ledger,
                )
                candidate.evaluation_mode = "full"
                candidate.gate_reasons = _gate_reasons(
                    candidate.evaluation,
                    min_win_rate=min_win_rate,
                    min_avg_return=min_avg_return,
                    min_total_return=min_total_return,
                    min_profit_loss_ratio=min_profit_loss_ratio,
                    max_drawdown=max_drawdown,
                    min_signals=min_signals,
                    reject_overfit=reject_overfit,
                    max_train_val_gap=max_train_val_gap,
                    max_val_test_gap=max_val_test_gap,
                    max_walk_forward_gap=max_walk_forward_gap,
                    min_walk_forward_pass_rate=min_walk_forward_pass_rate,
                    require_full_for_robust_gates=False,
                )
                candidate.passed_gate = not candidate.gate_reasons
                full_eval_count += 1
            ranked = sorted(
                ranked,
                key=lambda candidate: _candidate_sort_key(candidate, normalized_objective),
                reverse=True,
            )
        best_id = ranked[0].candidate_id if ranked else None
        if trial_ledger is not None:
            record_optimizer_trials(
                trial_ledger,
                base_strategy=strategy,
                candidates=ranked,
                backtest_results=backtest_results,
                best_candidate_id=best_id,
                data=data,
                notes="exit_optimizer",
            )
        return ExitOptimizationResult(
            base_strategy_id=strategy.meta.id,
            spaces=list(normalized_spaces),
            objective=normalized_objective,
            evaluation_mode=normalized_evaluation_mode,
            full_eval_top_n=full_eval_count,
            min_win_rate=min_win_rate,
            min_avg_return=min_avg_return,
            min_total_return=min_total_return,
            min_profit_loss_ratio=min_profit_loss_ratio,
            max_drawdown=max_drawdown,
            min_signals=min_signals,
            reject_overfit=reject_overfit,
            max_train_val_gap=max_train_val_gap,
            max_val_test_gap=max_val_test_gap,
            max_walk_forward_gap=max_walk_forward_gap,
            min_walk_forward_pass_rate=min_walk_forward_pass_rate,
            candidates=ranked,
            best_candidate_id=best_id,
        )

    def _generate_candidates(
        self,
        strategy: Strategy,
        spaces: tuple[str, ...],
    ) -> Iterable[tuple[Strategy, list[str]]]:
        stop_options = _stop_loss_options(strategy, "stoploss" in spaces)
        tp_options = _take_profit_options(strategy, "takeprofit" in spaces)
        holding_options = _holding_options(strategy, "holding" in spaces)
        trigger_options = _exit_trigger_options(strategy, "exit" in spaces)

        seen_signatures: set[str] = set()
        for stop_value, holding_days, triggers, tp_option in product(
            stop_options,
            holding_options,
            trigger_options,
            tp_options,
        ):
            candidate = strategy.model_copy(deep=True)
            changes: list[str] = []

            if stop_value is not None:
                old = candidate.exit.stop_loss.value
                candidate.exit.stop_loss.type = "pct"
                candidate.exit.stop_loss.value = stop_value
                if old != stop_value:
                    changes.append(f"stop_loss={stop_value:.1%}")

            old_tp_signature = _take_profit_signature(candidate.exit.take_profit)
            tp_type, tp_value, trigger_pct, trail_pct = tp_option
            candidate.exit.take_profit.type = tp_type
            candidate.exit.take_profit.value = tp_value
            candidate.exit.take_profit.target = None
            candidate.exit.take_profit.trigger_pct = trigger_pct
            candidate.exit.take_profit.trail_pct = trail_pct
            _sync_take_profit_tunables(candidate)
            new_tp_signature = _take_profit_signature(candidate.exit.take_profit)
            if old_tp_signature != new_tp_signature:
                changes.append(_format_take_profit_change(tp_option))

            old_holding = candidate.exit.max_holding_days
            candidate.exit.max_holding_days = holding_days
            if old_holding != holding_days:
                changes.append(f"max_holding_days={holding_days}")

            old_triggers = _condition_signature(candidate.exit.triggers)
            candidate.exit.triggers = list(triggers)
            new_triggers = _condition_signature(candidate.exit.triggers)
            if old_triggers != new_triggers:
                label = new_triggers or "none"
                changes.append(f"exit_triggers={label}")

            signature = _strategy_exit_signature(candidate)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            final_changes = changes or ["baseline"]
            _append_optimization_notes(candidate, final_changes, "exit optimization")
            yield candidate, final_changes


def render_exit_optimization_report(
    result: ExitOptimizationResult,
    *,
    top_n: int = 10,
) -> str:
    """Render a concise Markdown report for an optimization result."""
    lines = [
        f"# Exit Optimization Report: {result.base_strategy_id}",
        "",
        f"- Spaces: {', '.join(result.spaces)}",
        f"- Objective: {result.objective}",
        f"- Evaluation mode: {result.evaluation_mode}",
        f"- Full re-evaluated top candidates: {result.full_eval_top_n}",
        "- Gates: "
        f"{_format_gates(result.min_win_rate, result.min_avg_return, result.min_total_return, result.min_profit_loss_ratio, result.max_drawdown, result.min_signals, result.reject_overfit, result.max_train_val_gap, result.max_val_test_gap, result.max_walk_forward_gap, result.min_walk_forward_pass_rate)}",
        f"- Candidates tested: {len(result.candidates)}",
        f"- Qualified candidates: {result.qualified_count}",
    ]
    best = result.best_candidate
    if best is not None:
        lines.extend(["", *format_best_candidate_markdown(best), ""])
    else:
        lines.append("")
    showcase = select_high_win_return_candidate(result.candidates)
    if showcase is not None and (best is None or showcase.candidate_id != best.candidate_id):
        lines.extend(
            [
                "",
                *format_best_candidate_markdown(
                    showcase,
                    title="## Best High-Win/High-Return Candidate",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "| Rank | Candidate | Confidence | Win Rate | Avg Return | Avg Win | Avg Loss | Total Return | P/L | Max DD | Signals | Changes |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, candidate in enumerate(result.candidates[:top_n], start=1):
        ev = candidate.evaluation.overall
        lines.append(
            "| "
            f"{rank} | `{candidate.candidate_id}` | "
            f"{candidate.evaluation.confidence_score:.1%} | "
            f"{ev.win_rate:.1%} | "
            f"{ev.avg_return:.2%} | "
            f"{ev.avg_win_return:.2%} | "
            f"{ev.avg_loss_return:.2%} | "
            f"{ev.total_return:.2%} | "
            f"{ev.profit_loss_ratio:.2f} | "
            f"{ev.max_drawdown:.1%} | "
            f"{ev.signal_count} | "
            f"{'; '.join(candidate.changes)}{_format_gate_suffix(candidate)} |"
        )
    lines.append("")
    if best is not None:
        diag = best.diagnostics
        lines.extend(
            [
                "## Best Candidate Exit Diagnostics",
                "",
                f"- Exit reason mix: {_format_reason_counts(diag.exit_reason_counts)}",
                f"- Avg MFE: {diag.avg_mfe:.2%}",
                f"- Avg MAE: {diag.avg_mae:.2%}",
                f"- Avg giveback from best intratrade price: {diag.avg_giveback:.2%}",
                f"- Potentially sold early: {diag.sold_early_count}/{diag.total_trades}",
                f"- Potentially sold late: {diag.sold_late_count}/{diag.total_trades}",
                f"- Stop-loss exits that avoided further drawdown: {diag.stop_loss_effective_count}",
                f"- Take-profit exits with further upside soon after exit: {diag.take_profit_truncated_count}",
                "",
            ]
        )
    lines.append(
        "Note: candidates are ranked by the selected objective, confidence score, "
        "average return, P/L ratio, signal count, and lower drawdown. "
        "This is a research screen, not an investment recommendation."
    )
    return "\n".join(lines)


def analyze_exit_points(
    signals: list[TradeSignal],
    data: dict[str, Any],
    *,
    lookahead_days: int = _EARLY_EXIT_LOOKAHEAD_DAYS,
) -> ExitDiagnosticSummary:
    """Analyze whether exits look early, late, protective, or truncating."""
    executed = [signal for signal in signals if signal.exit_price is not None and signal.exit_date]
    if not executed:
        return ExitDiagnosticSummary()

    reason_counts: dict[str, int] = {}
    mfe_values: list[float] = []
    mae_values: list[float] = []
    givebacks: list[float] = []
    sold_early = 0
    sold_late = 0
    stop_effective = 0
    tp_truncated = 0

    for signal in executed:
        reason = signal.exit_reason.value if signal.exit_reason else "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        df = data.get(signal.symbol)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        entry_idx = _find_date_index(df, signal.signal_date)
        exit_idx = _find_date_index(df, signal.exit_date)
        if entry_idx is None or exit_idx is None or exit_idx < entry_idx:
            continue
        trade_window = df.iloc[entry_idx : exit_idx + 1]
        if trade_window.empty or signal.entry_price <= 0:
            continue

        mfe = (float(trade_window["high"].max()) - signal.entry_price) / signal.entry_price
        mae = (float(trade_window["low"].min()) - signal.entry_price) / signal.entry_price
        giveback = max(0.0, mfe - signal.return_pct)
        mfe_values.append(mfe)
        mae_values.append(mae)
        givebacks.append(giveback)

        post_window = df.iloc[exit_idx + 1 : exit_idx + 1 + lookahead_days]
        if post_window.empty or signal.exit_price is None or signal.exit_price <= 0:
            continue
        post_high_gain = (float(post_window["high"].max()) - signal.exit_price) / signal.exit_price
        post_low_drawdown = (
            signal.exit_price - float(post_window["low"].min())
        ) / signal.exit_price

        if (
            signal.exit_reason in {ExitReason.TAKE_PROFIT, ExitReason.SIGNAL, ExitReason.MAX_HOLD}
            and post_high_gain >= _MEANINGFUL_MOVE
        ):
            sold_early += 1
        if mfe >= _MEANINGFUL_MOVE and giveback >= _LATE_GIVEBACK:
            sold_late += 1
        if signal.exit_reason == ExitReason.STOP_LOSS and post_low_drawdown >= _MEANINGFUL_MOVE:
            stop_effective += 1
        if signal.exit_reason == ExitReason.TAKE_PROFIT and post_high_gain >= _MEANINGFUL_MOVE:
            tp_truncated += 1

    return ExitDiagnosticSummary(
        total_trades=len(executed),
        exit_reason_counts=reason_counts,
        avg_mfe=round(mean(mfe_values), 6) if mfe_values else 0.0,
        avg_mae=round(mean(mae_values), 6) if mae_values else 0.0,
        avg_giveback=round(mean(givebacks), 6) if givebacks else 0.0,
        sold_early_count=sold_early,
        sold_late_count=sold_late,
        stop_loss_effective_count=stop_effective,
        take_profit_truncated_count=tp_truncated,
    )


def export_best_strategy(
    result: ExitOptimizationResult,
    output_dir: Path,
) -> Path | None:
    """Write the best strategy YAML to *output_dir* and return its path."""
    best = result.best_candidate
    if best is None:
        return None
    if (
        result.min_win_rate is not None
        or result.min_avg_return is not None
        or result.min_total_return is not None
        or result.min_profit_loss_ratio is not None
        or result.max_drawdown is not None
        or result.min_signals is not None
        or result.reject_overfit
        or result.max_train_val_gap is not None
        or result.max_val_test_gap is not None
        or result.max_walk_forward_gap is not None
        or result.min_walk_forward_pass_rate is not None
    ) and not best.passed_gate:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{best.candidate_id}.yaml"
    StrategySerializer().to_file(best.strategy, path)
    return path


def _normalize_spaces(spaces: Iterable[str] | None) -> tuple[str, ...]:
    if spaces is None:
        return _DEFAULT_SPACES
    normalized: list[str] = []
    for space in spaces:
        key = space.strip().lower().replace("-", "_")
        key = _SPACE_ALIASES.get(key, key)
        if key == "all":
            for default in _DEFAULT_SPACES:
                if default not in normalized:
                    normalized.append(default)
            continue
        if key not in _DEFAULT_SPACES:
            raise ValueError(f"Unsupported optimization space: {space}")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized or _DEFAULT_SPACES)


def _normalize_objective(objective: str) -> OptimizationObjective:
    return normalize_objective(objective)


def _normalize_evaluation_mode(mode: str) -> EvaluationMode:
    key = mode.strip().lower().replace("-", "_")
    aliases = {
        "quick": "fast",
        "search": "fast",
        "complete": "full",
    }
    key = aliases.get(key, key)
    if key not in {"fast", "full"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    return key  # type: ignore[return-value]


def _evaluate_candidate(
    evaluator: Evaluator,
    result: BacktestResult,
    strategy: Strategy,
    *,
    data: dict[str, Any],
    contexts: dict[str, IndicatorContext] | None,
    backtest_config: Any,
    mode: EvaluationMode,
) -> EvaluationReport:
    if mode == "fast":
        return evaluator.evaluate_fast(result, strategy, contexts=contexts)
    return evaluator.evaluate(
        result,
        strategy,
        market_data=data,
        contexts=contexts,
        backtest_config=backtest_config,
    )


def _evaluate_work_item(
    item: _ExitCandidateWorkItem,
    *,
    data: dict[str, Any],
    batch: SampleBatch,
    contexts: dict[str, IndicatorContext] | None,
    slippage: float,
    commission: float,
    min_data_days: int,
    fill_policy: str,
    backtest_config: Any,
    mode: EvaluationMode,
    min_win_rate: float | None,
    min_avg_return: float | None,
    min_total_return: float | None,
    min_profit_loss_ratio: float | None,
    max_drawdown: float | None,
    min_signals: int | None,
    reject_overfit: bool,
    max_train_val_gap: float | None,
    max_val_test_gap: float | None,
    max_walk_forward_gap: float | None,
    min_walk_forward_pass_rate: float | None,
    require_full_for_robust_gates: bool,
) -> tuple[ExitOptimizationCandidate, BacktestResult]:
    engine = BacktestEngine(
        slippage=slippage,
        commission=commission,
        min_data_days=min_data_days,
        fill_policy=fill_policy,
    )
    evaluator = Evaluator()
    result = engine.run(item.candidate_strategy, data, batch, contexts=contexts)
    diagnostics = analyze_exit_points(result.signals, data)
    evaluation = _evaluate_candidate(
        evaluator,
        result,
        item.candidate_strategy,
        data=data,
        contexts=contexts,
        backtest_config=backtest_config,
        mode=mode,
    )
    gate_reasons = _gate_reasons(
        evaluation,
        min_win_rate=min_win_rate,
        min_avg_return=min_avg_return,
        min_total_return=min_total_return,
        min_profit_loss_ratio=min_profit_loss_ratio,
        max_drawdown=max_drawdown,
        min_signals=min_signals,
        reject_overfit=reject_overfit,
        max_train_val_gap=max_train_val_gap,
        max_val_test_gap=max_val_test_gap,
        max_walk_forward_gap=max_walk_forward_gap,
        min_walk_forward_pass_rate=min_walk_forward_pass_rate,
        require_full_for_robust_gates=require_full_for_robust_gates,
    )
    return (
        ExitOptimizationCandidate(
            candidate_id=item.candidate_strategy.meta.id,
            changes=item.changes,
            strategy=item.candidate_strategy,
            evaluation=evaluation,
            evaluation_mode=mode,
            diagnostics=diagnostics,
            passed_gate=not gate_reasons,
            gate_reasons=gate_reasons,
        ),
        result,
    )


def _stop_loss_options(strategy: Strategy, enabled: bool) -> list[float | None]:
    if not enabled:
        return [strategy.exit.stop_loss.value]
    current = strategy.exit.stop_loss.value if strategy.exit.stop_loss.type == "pct" else None
    return _dedupe_numbers([current, *_STOP_LOSS_GRID])


def _take_profit_options(strategy: Strategy, enabled: bool) -> list[TakeProfitOption]:
    current = _current_take_profit_option(strategy)
    if not enabled:
        return [current]
    options: list[TakeProfitOption] = [current]
    options.extend(("rr", value, None, None) for value in _TAKE_PROFIT_RR_GRID)
    options.extend(
        ("trailing", None, trigger_pct, trail_pct)
        for trigger_pct, trail_pct in _TAKE_PROFIT_TRAILING_GRID
    )
    return _dedupe_take_profit_options(options)


def _holding_options(strategy: Strategy, enabled: bool) -> list[int]:
    if not enabled:
        return [strategy.exit.max_holding_days]
    return _dedupe_ints([strategy.exit.max_holding_days, *_HOLDING_DAYS_GRID])


def _exit_trigger_options(
    strategy: Strategy,
    enabled: bool,
) -> list[tuple[StrategyCondition, ...]]:
    current = tuple(strategy.exit.triggers)
    if not enabled:
        return [current]
    options = [
        current,
        tuple(),
        (StrategyCondition(indicator="close_below_ma20", op="==", value=True),),
        (StrategyCondition(indicator="close_below_ma10", op="==", value=True),),
        (StrategyCondition(indicator="close_below_ma5", op="==", value=True),),
        (StrategyCondition(indicator="rsi_14", op=">", value=75),),
    ]
    deduped: list[tuple[StrategyCondition, ...]] = []
    seen: set[str] = set()
    for option in options:
        signature = _condition_signature(option)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(option)
    return deduped


def _dedupe_numbers(values: Iterable[float | None]) -> list[float | None]:
    seen: set[float | None] = set()
    result: list[float | None] = []
    for value in values:
        key = None if value is None else round(float(value), 6)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _dedupe_ints(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        key = int(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _current_take_profit_option(strategy: Strategy) -> TakeProfitOption:
    tp = strategy.exit.take_profit
    return (tp.type, tp.value, tp.trigger_pct, tp.trail_pct)


def _dedupe_take_profit_options(values: Iterable[TakeProfitOption]) -> list[TakeProfitOption]:
    seen: set[TakeProfitOption] = set()
    result: list[TakeProfitOption] = []
    for value in values:
        key = (
            value[0],
            None if value[1] is None else round(value[1], 6),
            None if value[2] is None else round(value[2], 6),
            None if value[3] is None else round(value[3], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _take_profit_signature(take_profit: Any) -> str:
    return "|".join(
        [
            str(take_profit.type),
            str(take_profit.value),
            str(take_profit.target),
            str(take_profit.trigger_pct),
            str(take_profit.trail_pct),
        ]
    )


def _sync_take_profit_tunables(strategy: Strategy) -> None:
    """Drop take-profit tunables that no longer apply after changing TP type."""
    keep_by_type = {
        "rr": {"exit.take_profit.value"},
        "pct": {"exit.take_profit.value"},
        "target_ma": {"exit.take_profit.target"},
        "trailing": {"exit.take_profit.trigger_pct", "exit.take_profit.trail_pct"},
    }
    take_profit_targets = {
        "exit.take_profit.value",
        "exit.take_profit.target",
        "exit.take_profit.trigger_pct",
        "exit.take_profit.trail_pct",
    }
    keep_targets = keep_by_type.get(strategy.exit.take_profit.type, set())
    strategy.params.tunable = [
        param
        for param in strategy.params.tunable
        if param.target not in take_profit_targets or param.target in keep_targets
    ]


def _format_take_profit_change(option: TakeProfitOption) -> str:
    tp_type, value, trigger_pct, trail_pct = option
    if tp_type == "trailing":
        trigger_label = "n/a" if trigger_pct is None else f"{trigger_pct:.1%}"
        trail_label = "n/a" if trail_pct is None else f"{trail_pct:.1%}"
        return f"take_profit_trailing={trigger_label}/{trail_label}"
    if value is not None:
        return f"take_profit_{tp_type}={value:.1f}"
    return f"take_profit={tp_type}"


def _append_optimization_notes(strategy: Strategy, changes: list[str], source: str) -> None:
    if not changes or changes == ["baseline"]:
        return
    note = f"Optimization notes ({source}): " + "; ".join(changes)
    description = strategy.description.rstrip()
    if note in description:
        return
    strategy.description = f"{description}\n\n{note}"


def _condition_signature(conditions: Iterable[StrategyCondition]) -> str:
    return ",".join(f"{c.indicator}{c.op}{c.value}" for c in conditions)


def _strategy_exit_signature(strategy: Strategy) -> str:
    sl = strategy.exit.stop_loss
    tp = strategy.exit.take_profit
    return "|".join(
        [
            f"triggers={_condition_signature(strategy.exit.triggers)}",
            f"sl={sl.type}:{sl.value}:{sl.multiplier}:{sl.atr_period}:{sl.reference}",
            f"tp={tp.type}:{tp.value}:{tp.target}:{tp.trigger_pct}:{tp.trail_pct}",
            f"hold={strategy.exit.max_holding_days}",
        ]
    )


def _candidate_sort_key(
    candidate: ExitOptimizationCandidate,
    objective: OptimizationObjective,
) -> tuple[float, float, float, float, float, float, float, float, float, int, float]:
    return candidate_sort_key(
        candidate.evaluation,
        passed_gate=candidate.passed_gate,
        objective=objective,
    )


def _objective_value(
    candidate: ExitOptimizationCandidate, objective: OptimizationObjective
) -> float:
    return objective_value(candidate.evaluation, objective)


def _gate_reasons(
    evaluation: EvaluationReport,
    *,
    min_win_rate: float | None,
    min_avg_return: float | None,
    min_total_return: float | None,
    min_profit_loss_ratio: float | None,
    max_drawdown: float | None,
    min_signals: int | None,
    reject_overfit: bool,
    max_train_val_gap: float | None,
    max_val_test_gap: float | None,
    max_walk_forward_gap: float | None,
    min_walk_forward_pass_rate: float | None,
    require_full_for_robust_gates: bool,
) -> list[str]:
    reasons: list[str] = []
    if min_win_rate is not None and evaluation.overall.win_rate < min_win_rate:
        reasons.append(f"win_rate<{min_win_rate:.1%}")
    if min_avg_return is not None and evaluation.overall.avg_return < min_avg_return:
        reasons.append(f"avg_return<{min_avg_return:.2%}")
    if min_total_return is not None and evaluation.overall.total_return < min_total_return:
        reasons.append(f"total_return<{min_total_return:.2%}")
    if (
        min_profit_loss_ratio is not None
        and evaluation.overall.profit_loss_ratio < min_profit_loss_ratio
    ):
        reasons.append(f"profit_loss_ratio<{min_profit_loss_ratio:.2f}")
    if max_drawdown is not None and evaluation.overall.max_drawdown > max_drawdown:
        reasons.append(f"max_drawdown>{max_drawdown:.1%}")
    if min_signals is not None and evaluation.overall.signal_count < min_signals:
        reasons.append(f"signals<{min_signals}")
    robust_configured = _robust_gates_configured(
        reject_overfit=reject_overfit,
        max_train_val_gap=max_train_val_gap,
        max_val_test_gap=max_val_test_gap,
        max_walk_forward_gap=max_walk_forward_gap,
        min_walk_forward_pass_rate=min_walk_forward_pass_rate,
    )
    if robust_configured and require_full_for_robust_gates:
        reasons.append("full_eval_required_for_robust_gates")
        return reasons

    anti = evaluation.anti_overfit
    if reject_overfit and anti.is_overfit:
        reasons.append("overfit_detected")
    if max_train_val_gap is not None and anti.train_val_gap > max_train_val_gap:
        reasons.append(f"train_val_gap>{max_train_val_gap:.1%}")
    if max_val_test_gap is not None and anti.val_test_gap > max_val_test_gap:
        reasons.append(f"val_test_gap>{max_val_test_gap:.1%}")
    if max_walk_forward_gap is not None and anti.walk_forward_gap > max_walk_forward_gap:
        reasons.append(f"walk_forward_gap>{max_walk_forward_gap:.1%}")
    if (
        min_walk_forward_pass_rate is not None
        and anti.walk_forward_pass_rate < min_walk_forward_pass_rate
    ):
        reasons.append(f"walk_forward_pass_rate<{min_walk_forward_pass_rate:.1%}")
    return reasons


def _robust_gates_configured(
    *,
    reject_overfit: bool,
    max_train_val_gap: float | None,
    max_val_test_gap: float | None,
    max_walk_forward_gap: float | None,
    min_walk_forward_pass_rate: float | None,
) -> bool:
    return (
        reject_overfit
        or max_train_val_gap is not None
        or max_val_test_gap is not None
        or max_walk_forward_gap is not None
        or min_walk_forward_pass_rate is not None
    )


def _format_gates(
    min_win_rate: float | None,
    min_avg_return: float | None,
    min_total_return: float | None,
    min_profit_loss_ratio: float | None,
    max_drawdown: float | None,
    min_signals: int | None,
    reject_overfit: bool,
    max_train_val_gap: float | None,
    max_val_test_gap: float | None,
    max_walk_forward_gap: float | None,
    min_walk_forward_pass_rate: float | None,
) -> str:
    gates: list[str] = []
    if min_win_rate is not None:
        gates.append(f"win_rate >= {min_win_rate:.1%}")
    if min_avg_return is not None:
        gates.append(f"avg_return >= {min_avg_return:.2%}")
    if min_total_return is not None:
        gates.append(f"total_return >= {min_total_return:.2%}")
    if min_profit_loss_ratio is not None:
        gates.append(f"profit_loss_ratio >= {min_profit_loss_ratio:.2f}")
    if max_drawdown is not None:
        gates.append(f"max_drawdown <= {max_drawdown:.1%}")
    if min_signals is not None:
        gates.append(f"signals >= {min_signals}")
    if reject_overfit:
        gates.append("not overfit")
    if max_train_val_gap is not None:
        gates.append(f"train_val_gap <= {max_train_val_gap:.1%}")
    if max_val_test_gap is not None:
        gates.append(f"val_test_gap <= {max_val_test_gap:.1%}")
    if max_walk_forward_gap is not None:
        gates.append(f"walk_forward_gap <= {max_walk_forward_gap:.1%}")
    if min_walk_forward_pass_rate is not None:
        gates.append(f"walk_forward_pass_rate >= {min_walk_forward_pass_rate:.1%}")
    return ", ".join(gates) if gates else "none"


def _format_gate_suffix(candidate: ExitOptimizationCandidate) -> str:
    if candidate.passed_gate:
        return ""
    return f" (gate fail: {', '.join(candidate.gate_reasons)})"


def _find_date_index(df: pd.DataFrame, target: Any) -> int | None:
    if "date" not in df.columns:
        return None
    target_date = pd.Timestamp(target).date()
    for idx, value in enumerate(df["date"]):
        try:
            if pd.Timestamp(value).date() == target_date:
                return idx
        except Exception:
            continue
    return None


def _format_reason_counts(reason_counts: dict[str, int]) -> str:
    if not reason_counts:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
