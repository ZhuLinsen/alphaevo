"""Parameter optimization over executable strategy tunables."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.evaluator.metrics import EvaluationMode, Evaluator
from alphaevo.models.enums import StrategyStatus
from alphaevo.models.execution import BacktestResult, EvaluationReport, SampleBatch
from alphaevo.models.market import IndicatorContext
from alphaevo.models.strategy import Strategy, StrategyCondition, TunableParam
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
from alphaevo.strategy.tunable import (
    is_integer_tunable_target,
    is_period_tunable_target,
    resolve_tunable_target,
    set_tunable_target,
)

OptimizationValue = float | int | str | bool | None
OptimizationOption = tuple[TunableParam, OptimizationValue, OptimizationValue]

_DEFAULT_SPACES = ("entry", "exit", "indicator")
_SPACE_ALIASES = {
    "all": "all",
    "param": "all",
    "params": "all",
    "parameter": "all",
    "parameters": "all",
    "tunable": "all",
    "tunables": "all",
    "buy": "entry",
    "buy_signal": "entry",
    "buytrigger": "entry",
    "trigger": "entry",
    "triggers": "entry",
    "guard": "entry",
    "guards": "entry",
    "filter": "entry",
    "filters": "entry",
    "sell": "exit",
    "sell_signal": "exit",
    "stop": "exit",
    "stoploss": "exit",
    "stop_loss": "exit",
    "takeprofit": "exit",
    "take_profit": "exit",
    "tp": "exit",
    "period": "indicator",
    "window": "indicator",
    "lookback": "indicator",
}
_ENTRY_GUARD_CANDIDATES = (
    StrategyCondition(indicator="ma20_slope", op=">", value=0),
    StrategyCondition(indicator="momentum_10d", op=">", value=0),
    StrategyCondition(indicator="relative_strength_20d", op=">", value=0),
    StrategyCondition(indicator="volume_ratio_1d_20d", op=">=", value=1.0),
    StrategyCondition(indicator="volatility_20d", op="<=", value=0.04),
    StrategyCondition(indicator="volatility_20d", op="<=", value=0.06),
    StrategyCondition(indicator="rsi_14", op="<", value=65),
    StrategyCondition(indicator="rsi_14_zscore", op="<=", value=1.5),
    StrategyCondition(indicator="price_position_120d", op=">=", value=0.65),
    StrategyCondition(indicator="price_position_52w", op=">=", value=0.5),
    StrategyCondition(indicator="bollinger_band_width_20d", op="<=", value=0.16),
    StrategyCondition(indicator="body_to_range_ratio", op=">=", value=0.35),
    StrategyCondition(indicator="gap_up_pct", op="<=", value=0.04),
    StrategyCondition(indicator="days_since_high_20d", op="<=", value=10),
)


class ParamOptimizationCandidate(BaseModel):
    """One tested strategy parameter candidate."""

    candidate_id: str
    target: str | None = None
    from_value: OptimizationValue = None
    to_value: OptimizationValue = None
    changes: list[str] = Field(default_factory=list)
    strategy: Strategy
    evaluation: EvaluationReport
    evaluation_mode: EvaluationMode = "fast"
    passed_gate: bool = True
    gate_reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _ParamCandidateWorkItem:
    candidate_strategy: Strategy
    target: str | None
    from_value: OptimizationValue
    to_value: OptimizationValue
    changes: list[str]


class ParamOptimizationResult(BaseModel):
    """Ranked result of a tunable-parameter optimization run."""

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
    tunables_considered: int = 0
    candidates: list[ParamOptimizationCandidate] = Field(default_factory=list)
    best_candidate_id: str | None = None

    @property
    def qualified_count(self) -> int:
        """Return the number of candidates that passed all configured gates."""
        return sum(1 for candidate in self.candidates if candidate.passed_gate)

    @property
    def best_candidate(self) -> ParamOptimizationCandidate | None:
        """Return the highest-ranked candidate if available."""
        if not self.best_candidate_id:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == self.best_candidate_id:
                return candidate
        return None


class ParamOptimizer:
    """Search a bounded grid over ``strategy.params.tunable`` values."""

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
        max_values_per_param: int = 5,
        max_changes: int = 1,
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
    ) -> ParamOptimizationResult:
        """Evaluate bounded single-parameter mutations on existing data."""
        normalized_spaces = _normalize_spaces(spaces)
        normalized_objective = _normalize_objective(objective)
        normalized_evaluation_mode = _normalize_evaluation_mode(evaluation_mode)

        tunables = [
            param
            for param in strategy.params.tunable
            if _target_matches_spaces(param.target, normalized_spaces)
        ]
        work_items: list[_ParamCandidateWorkItem] = []

        for idx, (candidate_strategy, target, from_value, to_value, changes) in enumerate(
            self._generate_candidates(
                strategy,
                tunables,
                max_values_per_param=max_values_per_param,
                max_changes=max_changes,
                include_entry_guards="entry" in normalized_spaces,
            ),
            start=1,
        ):
            if idx > max_candidates:
                break

            candidate_strategy.meta.id = f"{strategy.meta.id}_param_{idx:03d}"
            candidate_strategy.meta.parent_id = strategy.meta.id
            candidate_strategy.meta.version = strategy.meta.version + 1
            candidate_strategy.meta.status = StrategyStatus.DRAFT

            work_items.append(
                _ParamCandidateWorkItem(
                    candidate_strategy=candidate_strategy,
                    target=target,
                    from_value=from_value,
                    to_value=to_value,
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
                notes="param_optimizer",
            )
        return ParamOptimizationResult(
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
            tunables_considered=len(tunables),
            candidates=ranked,
            best_candidate_id=best_id,
        )

    def _generate_candidates(
        self,
        strategy: Strategy,
        tunables: list[TunableParam],
        *,
        max_values_per_param: int,
        max_changes: int,
        include_entry_guards: bool,
    ) -> Iterable[
        tuple[
            Strategy,
            str | None,
            OptimizationValue,
            OptimizationValue,
            list[str],
        ]
    ]:
        seen_signatures: set[str] = set()
        baseline = strategy.model_copy(deep=True)
        baseline_signature = _strategy_param_signature(baseline)
        seen_signatures.add(baseline_signature)
        yield baseline, None, None, None, ["baseline"]

        options: list[OptimizationOption] = []
        for param in tunables:
            current = resolve_tunable_target(strategy, param.target)
            for value in _candidate_values(param, current, max_values=max_values_per_param):
                if _values_equal(value, current):
                    continue
                options.append((param, _coerce_report_value(current), value))

        for change_count in range(1, max(1, max_changes) + 1):
            for option_group in combinations(options, change_count):
                targets = [param.target for param, _current, _value in option_group]
                if len(set(targets)) != len(targets):
                    continue
                candidate = strategy.model_copy(deep=True)
                target: str | None = None
                from_value: OptimizationValue = None
                to_value: OptimizationValue = None
                applied_changes = _apply_option_group(candidate, option_group)
                if applied_changes is None:
                    continue
                for param, current, value in option_group:
                    if target is None:
                        target = param.target
                        from_value = current
                        to_value = value
                signature = _strategy_param_signature(candidate)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                _append_optimization_notes(candidate, applied_changes, "parameter optimization")
                yield (
                    candidate,
                    target,
                    from_value,
                    to_value,
                    applied_changes,
                )

        if include_entry_guards:
            for guard in _entry_guard_candidates(strategy):
                candidate = strategy.model_copy(deep=True)
                candidate.entry.guards.append(guard)
                signature = _strategy_param_signature(candidate)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                label = _format_condition(guard)
                changes = [f"Add entry guard: {label}"]
                _append_optimization_notes(candidate, changes, "parameter optimization")
                yield (
                    candidate,
                    "entry.guards",
                    None,
                    label,
                    changes,
                )

        if include_entry_guards and max_changes > 1:
            for guard in _entry_guard_candidates(strategy):
                guard_label = _format_condition(guard)
                for change_count in range(1, max(1, max_changes)):
                    for option_group in combinations(options, change_count):
                        candidate = strategy.model_copy(deep=True)
                        candidate.entry.guards.append(guard.model_copy(deep=True))
                        changes = [f"Add entry guard: {guard_label}"]
                        applied_changes = _apply_option_group(candidate, option_group)
                        if applied_changes is None:
                            continue
                        changes.extend(applied_changes)
                        signature = _strategy_param_signature(candidate)
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        _append_optimization_notes(
                            candidate,
                            changes,
                            "parameter optimization",
                        )
                        yield (
                            candidate,
                            "entry.guards",
                            None,
                            guard_label,
                            changes,
                        )


def render_param_optimization_report(
    result: ParamOptimizationResult,
    *,
    top_n: int = 10,
) -> str:
    """Render a concise Markdown report for parameter optimization."""
    lines = [
        f"# Parameter Optimization Report: {result.base_strategy_id}",
        "",
        f"- Spaces: {', '.join(result.spaces)}",
        f"- Objective: {result.objective}",
        f"- Evaluation mode: {result.evaluation_mode}",
        f"- Full re-evaluated top candidates: {result.full_eval_top_n}",
        "- Gates: "
        f"{_format_gates(result.min_win_rate, result.min_avg_return, result.min_total_return, result.min_profit_loss_ratio, result.max_drawdown, result.min_signals, result.reject_overfit, result.max_train_val_gap, result.max_val_test_gap, result.max_walk_forward_gap, result.min_walk_forward_pass_rate)}",
        f"- Tunables considered: {result.tunables_considered}",
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
        changes = "; ".join(candidate.changes)
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
            f"{changes}{_format_gate_suffix(candidate)} |"
        )
    lines.append("")
    lines.append(
        "Note: each candidate mutates one or more DSL tunables and is ranked by "
        "the selected objective, confidence score, average return, P/L ratio, "
        "signal count, and lower drawdown. "
        "This is a research screen, not an investment recommendation."
    )
    return "\n".join(lines)


def export_best_param_strategy(
    result: ParamOptimizationResult,
    output_dir: Path,
) -> Path | None:
    """Write the best parameter-optimized strategy YAML to *output_dir*."""
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
            raise ValueError(f"Unsupported parameter optimization space: {space}")
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
    item: _ParamCandidateWorkItem,
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
) -> tuple[ParamOptimizationCandidate, BacktestResult]:
    engine = BacktestEngine(
        slippage=slippage,
        commission=commission,
        min_data_days=min_data_days,
        fill_policy=fill_policy,
    )
    evaluator = Evaluator()
    result = engine.run(item.candidate_strategy, data, batch, contexts=contexts)
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
        ParamOptimizationCandidate(
            candidate_id=item.candidate_strategy.meta.id,
            target=item.target,
            from_value=item.from_value,
            to_value=item.to_value,
            changes=item.changes,
            strategy=item.candidate_strategy,
            evaluation=evaluation,
            evaluation_mode=mode,
            passed_gate=not gate_reasons,
            gate_reasons=gate_reasons,
        ),
        result,
    )


def _target_matches_spaces(target: str, spaces: tuple[str, ...]) -> bool:
    return (
        ("entry" in spaces and target.startswith("entry."))
        or ("exit" in spaces and target.startswith("exit."))
        or ("indicator" in spaces and is_period_tunable_target(target))
    )


def _candidate_values(
    param: TunableParam,
    current: Any,
    *,
    max_values: int,
) -> list[OptimizationValue]:
    if current is None or isinstance(current, bool) or not isinstance(current, (int, float)):
        return []
    if param.step <= 0:
        return []

    lo, hi = param.range
    if lo > hi:
        lo, hi = hi, lo
    current_float = float(current)
    if current_float < lo or current_float > hi:
        return []

    raw_values: list[float] = [
        current_float,
        current_float - param.step,
        current_float + param.step,
        lo,
        hi,
        current_float - param.step * 2,
        current_float + param.step * 2,
        (lo + hi) / 2,
    ]

    values: list[OptimizationValue] = []
    seen: set[str] = set()
    for raw in raw_values:
        if len(values) >= max(1, max_values):
            break
        if raw < lo or raw > hi:
            continue
        value = _normalize_candidate_value(param.target, raw)
        key = _value_key(value)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def _normalize_candidate_value(target: str, value: float) -> OptimizationValue:
    if is_integer_tunable_target(target):
        return int(round(value))
    if is_period_tunable_target(target):
        return round(float(value), 4)
    return round(float(value), 6)


def _coerce_report_value(value: Any) -> OptimizationValue:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _format_change(param: TunableParam, current: Any, value: OptimizationValue) -> str:
    label = param.label or param.target
    return f"{label}: {current} -> {value}"


def _apply_option_group(
    candidate: Strategy,
    option_group: Iterable[OptimizationOption],
) -> list[str] | None:
    targets: set[str] = set()
    changes: list[str] = []
    for param, current, value in option_group:
        if param.target in targets:
            return None
        targets.add(param.target)
        if not set_tunable_target(candidate, param.target, value):
            return None
        changes.append(_format_change(param, current, value))
    return changes


def _entry_guard_candidates(strategy: Strategy) -> list[StrategyCondition]:
    existing = _entry_condition_keys(strategy)
    candidates: list[StrategyCondition] = []
    for condition in _ENTRY_GUARD_CANDIDATES:
        if _condition_key(condition) in existing:
            continue
        candidates.append(condition.model_copy(deep=True))
    return candidates


def _entry_condition_keys(strategy: Strategy) -> set[str]:
    keys: set[str] = set()
    for group in (
        strategy.entry.triggers,
        strategy.entry.guards,
        strategy.entry.conditions,
        strategy.entry.filters,
    ):
        keys.update(_condition_key(condition) for condition in group)
    return keys


def _condition_key(condition: StrategyCondition) -> str:
    return f"{condition.indicator}|{condition.op}|{condition.value}"


def _format_condition(condition: StrategyCondition) -> str:
    return f"{condition.indicator} {condition.op} {condition.value}"


def _append_optimization_notes(strategy: Strategy, changes: list[str], source: str) -> None:
    if not changes or changes == ["baseline"]:
        return
    note = f"Optimization notes ({source}): " + "; ".join(changes)
    description = strategy.description.rstrip()
    if note in description:
        return
    strategy.description = f"{description}\n\n{note}"


def _strategy_param_signature(strategy: Strategy) -> str:
    return "|".join(
        [
            repr(strategy.entry.model_dump()),
            repr(strategy.exit.model_dump()),
            repr(strategy.params.model_dump()),
        ]
    )


def _candidate_sort_key(
    candidate: ParamOptimizationCandidate,
    objective: OptimizationObjective,
) -> tuple[float, float, float, float, float, float, float, float, float, int, float]:
    return candidate_sort_key(
        candidate.evaluation,
        passed_gate=candidate.passed_gate,
        objective=objective,
    )


def _objective_value(
    candidate: ParamOptimizationCandidate, objective: OptimizationObjective
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


def _format_gate_suffix(candidate: ParamOptimizationCandidate) -> str:
    if candidate.passed_gate:
        return ""
    return f" (gate fail: {', '.join(candidate.gate_reasons)})"


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return round(float(left), 8) == round(float(right), 8)
    return bool(left == right)


def _value_key(value: OptimizationValue) -> str:
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)
