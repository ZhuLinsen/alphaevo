"""Heuristic lookahead/repainting checks for strategy DSL definitions."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from alphaevo.models.strategy import Strategy, StrategyCondition

BiasSeverity = Literal["info", "warning", "error"]

_FUTURE_LOOKING_RE = re.compile(
    r"(?:^|[_\-.])(?:future|forward|lead|lookahead|tomorrow|next)(?:[_\-.]|$)|"
    r"(?:shift[_\-.]?-\d+)|(?:return[_\-.]?\+\d+)",
    re.IGNORECASE,
)
_EVENT_INDICATORS = {
    "negative_news_score",
    "news_sentiment_score",
    "days_since_event",
    "price_above_pre_event",
    "sector_fund_flow_positive",
    "already_overreacted",
}


class BiasFinding(BaseModel):
    """One lookahead/repainting validation finding."""

    severity: BiasSeverity
    category: str
    location: str
    message: str


class BiasValidationReport(BaseModel):
    """Lookahead/repainting validation report for one strategy."""

    strategy_id: str
    lookahead_checked: bool = True
    findings: list[BiasFinding] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True when any finding should block a strategy."""
        return any(finding.severity == "error" for finding in self.findings)

    @property
    def warning_count(self) -> int:
        """Return the number of warning findings."""
        return sum(1 for finding in self.findings if finding.severity == "warning")

    @property
    def risk_level(self) -> Literal["low", "medium", "high"]:
        """Coarse risk level for report rendering."""
        if self.has_errors:
            return "high"
        if self.warning_count:
            return "medium"
        return "low"


def analyze_strategy_bias(strategy: Strategy) -> BiasValidationReport:
    """Run heuristic lookahead/repainting checks against a strategy DSL.

    The checks are intentionally conservative and DSL-level only. They do not prove
    a backtest is bias-free, but they catch common red flags before a strategy is
    allowed to enter optimization or evolution loops.
    """
    findings: list[BiasFinding] = []

    _collect_condition_findings(findings, strategy.entry.triggers, "entry.triggers")
    _collect_condition_findings(findings, strategy.entry.conditions, "entry.conditions")
    _collect_condition_findings(findings, strategy.entry.guards, "entry.guards")
    _collect_condition_findings(findings, strategy.entry.filters, "entry.filters")
    _collect_condition_findings(findings, strategy.exit.triggers, "exit.triggers")
    _collect_condition_findings(
        findings,
        strategy.exit.stop_loss.conditions or [],
        "exit.stop_loss.conditions",
    )

    execution = strategy.entry.execution
    if execution is not None and execution.timing == "close":
        findings.append(
            BiasFinding(
                severity="warning",
                category="same_bar_execution",
                location="entry.execution.timing",
                message=(
                    "Entry timing uses the current close. Confirm reports describe this as "
                    "same-bar execution and do not compare it directly with next-open strategies."
                ),
            )
        )

    if _uses_event_indicators(strategy):
        findings.append(
            BiasFinding(
                severity="info",
                category="event_effective_date",
                location="event/news indicators",
                message=(
                    "Event/news indicators require provider timestamps to be aligned to an "
                    "effective trading date or the next available trading bar before backtesting."
                ),
            )
        )

    for idx, param in enumerate(strategy.params.tunable):
        if _FUTURE_LOOKING_RE.search(param.target):
            findings.append(
                BiasFinding(
                    severity="error",
                    category="future_tunable_target",
                    location=f"params.tunable[{idx}].target",
                    message=f"Tunable target appears future-looking: {param.target}",
                )
            )

    return BiasValidationReport(strategy_id=strategy.meta.id, findings=findings)


def _collect_condition_findings(
    findings: list[BiasFinding],
    conditions: list[StrategyCondition],
    base_location: str,
) -> None:
    for idx, condition in enumerate(conditions):
        location = f"{base_location}[{idx}].indicator"
        if _FUTURE_LOOKING_RE.search(condition.indicator):
            findings.append(
                BiasFinding(
                    severity="error",
                    category="future_indicator",
                    location=location,
                    message=f"Indicator appears future-looking: {condition.indicator}",
                )
            )


def _uses_event_indicators(strategy: Strategy) -> bool:
    conditions = [
        *strategy.entry.triggers,
        *strategy.entry.conditions,
        *strategy.entry.guards,
        *strategy.entry.filters,
        *strategy.exit.triggers,
        *(strategy.exit.stop_loss.conditions or []),
    ]
    return any(condition.indicator in _EVENT_INDICATORS for condition in conditions)
