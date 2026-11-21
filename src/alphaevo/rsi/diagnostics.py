"""Pure structural condition pass-rate diagnostics for failure-driven Stock RSI evolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from alphaevo.backtest.condition import ConditionEvaluator
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.models.strategy import Strategy, StrategyCondition

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

ConditionGroupName = Literal["triggers", "conditions", "guards", "filters"]


class ConditionPassRate(BaseModel):
    """Per-condition structural bar pass rate across sampled market data (zero return peeking)."""

    group: ConditionGroupName
    indicator: str
    op: str
    value: Any
    passed_bars: int = 0
    total_bars: int = 0
    pass_rate: float = 0.0
    matching_tunable_targets: list[str] = Field(default_factory=list)


class StrategyBottleneckReport(BaseModel):
    """Structural signal-flow X-Ray identifying which conditions block strategy entry."""

    strategy_id: str
    entry_logic: str
    total_evaluated_bars: int = 0
    trigger_group_pass_rate: float = 0.0
    guard_group_pass_rate: float = 0.0
    combined_pass_rate: float = 0.0
    conditions: list[ConditionPassRate] = Field(default_factory=list)
    tightest_bottleneck: ConditionPassRate | None = None
    diagnosis_summary: str = ""


def _matching_tunables(strategy: Strategy, group: ConditionGroupName, indicator: str) -> list[str]:
    prefix = f"entry.{group}[indicator={indicator}]"
    fallback_tok = f"indicator={indicator}]"
    exact = [p.target for p in strategy.params.tunable if p.target.startswith(prefix)]
    if exact:
        return exact
    return [p.target for p in strategy.params.tunable if fallback_tok in p.target]


def analyze_entry_bottlenecks(
    strategy: Strategy,
    market_data: dict[str, pd.DataFrame],
) -> StrategyBottleneckReport:
    """Measure per-condition boolean pass rates without peeking at future returns or TrialLedger."""
    entry = strategy.entry
    grouped_conds: list[tuple[ConditionGroupName, StrategyCondition]] = [
        ("triggers", c) for c in entry.triggers
    ]
    if not entry.triggers:
        grouped_conds.extend(("conditions", c) for c in entry.conditions)
    grouped_conds.extend(("guards", c) for c in entry.guards)
    grouped_conds.extend(("filters", c) for c in entry.filters)

    if not grouped_conds or not market_data:
        return StrategyBottleneckReport(
            strategy_id=strategy.meta.id,
            entry_logic=entry.logic,
            diagnosis_summary="No entry conditions or market bars to diagnose.",
        )

    all_inds = list(dict.fromkeys(c.indicator for _, c in grouped_conds))
    registered_inds = [name for name in all_inds if IndicatorRegistry.is_registered(name)]

    cond_evaluator = ConditionEvaluator()
    pass_counts = [0] * len(grouped_conds)
    total_bars = 0
    trigger_pass_bars = 0
    guard_pass_bars = 0
    combined_pass_bars = 0

    trigger_group = entry.triggers or entry.conditions
    guard_group = [*entry.guards, *entry.filters]

    for df in market_data.values():
        if df is None or df.empty:
            continue
        n_bars = len(df)
        try:
            precomp = IndicatorRegistry.precompute(registered_inds, df) if registered_inds else {}
        except Exception:
            precomp = {}

        for idx in range(n_bars):
            total_bars += 1
            for c_pos, (_grp, cond) in enumerate(grouped_conds):
                if (
                    cond.indicator in precomp or IndicatorRegistry.is_registered(cond.indicator)
                ) and cond_evaluator.evaluate_condition(cond, df, idx, precomputed=precomp):
                    pass_counts[c_pos] += 1
            trig_ok = cond_evaluator.evaluate_group(
                trigger_group, entry.logic, df, idx, precomputed=precomp
            )
            guard_ok = cond_evaluator.evaluate_group(
                guard_group, "and", df, idx, precomputed=precomp
            )
            if trig_ok:
                trigger_pass_bars += 1
            if guard_ok:
                guard_pass_bars += 1
            if trig_ok and guard_ok:
                combined_pass_bars += 1

    cond_reports: list[ConditionPassRate] = []
    for c_pos, (grp, cond) in enumerate(grouped_conds):
        passed = pass_counts[c_pos]
        rate = round(passed / total_bars, 6) if total_bars > 0 else 0.0
        cond_reports.append(
            ConditionPassRate(
                group=grp,
                indicator=cond.indicator,
                op=cond.op,
                value=cond.value,
                passed_bars=passed,
                total_bars=total_bars,
                pass_rate=rate,
                matching_tunable_targets=_matching_tunables(strategy, grp, cond.indicator),
            )
        )

    tightest: ConditionPassRate | None = None
    if cond_reports:
        restrictive_guards = [
            c for c in cond_reports if c.group in {"guards", "filters"} and c.pass_rate < 0.25
        ]
        if entry.logic == "or" and restrictive_guards:
            tightest = min(restrictive_guards, key=lambda c: (c.pass_rate, c.indicator))
        else:
            tightest = min(
                cond_reports,
                key=lambda c: (
                    c.pass_rate,
                    0 if c.group in {"guards", "filters"} else 1,
                    c.indicator,
                ),
            )

    trig_rate = round(trigger_pass_bars / total_bars, 6) if total_bars > 0 else 0.0
    guard_rate = round(guard_pass_bars / total_bars, 6) if total_bars > 0 else 0.0
    comb_rate = round(combined_pass_bars / total_bars, 6) if total_bars > 0 else 0.0

    if tightest is not None:
        summary = (
            f"Combined entry pass rate={comb_rate:.2%} across {total_bars} bars "
            f"(triggers={trig_rate:.2%}, guards/filters={guard_rate:.2%}). "
            f"Tightest bottleneck: {tightest.group}.{tightest.indicator} "
            f"{tightest.op} {tightest.value} ({tightest.pass_rate:.2%})."
        )
    else:
        summary = f"Combined entry pass rate={comb_rate:.2%} across {total_bars} bars."

    return StrategyBottleneckReport(
        strategy_id=strategy.meta.id,
        entry_logic=entry.logic,
        total_evaluated_bars=total_bars,
        trigger_group_pass_rate=trig_rate,
        guard_group_pass_rate=guard_rate,
        combined_pass_rate=comb_rate,
        conditions=cond_reports,
        tightest_bottleneck=tightest,
        diagnosis_summary=summary,
    )


__all__ = [
    "ConditionGroupName",
    "ConditionPassRate",
    "StrategyBottleneckReport",
    "analyze_entry_bottlenecks",
]
