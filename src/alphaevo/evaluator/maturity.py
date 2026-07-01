"""Research maturity checks inspired by mature open-source quant systems."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from alphaevo.models.execution import EvaluationReport
from alphaevo.models.strategy import Strategy

MaturityStatus = Literal["pass", "watch", "fail"]
MaturityAction = Literal[
    "promote_to_validation",
    "expand_sample",
    "add_baseline",
    "run_robustness",
    "repair_data",
    "simplify_strategy",
    "optimize_strategy",
]


class MaturityCheck(BaseModel):
    """One research-readiness check for a strategy evaluation."""

    check_id: str
    title: str
    status: MaturityStatus
    evidence: list[str] = Field(default_factory=list)
    recommendation: str
    inspired_by: list[str] = Field(default_factory=list)


class MaturityNextAction(BaseModel):
    """The single highest-priority next step implied by maturity checks."""

    action: MaturityAction
    priority: Literal["high", "medium", "low"] = "medium"
    title: str
    rationale: str
    commands: list[str] = Field(default_factory=list)


class ResearchMaturityReport(BaseModel):
    """Aggregate maturity report for deciding the next research action."""

    status: MaturityStatus
    score: float
    checks: list[MaturityCheck] = Field(default_factory=list)
    next_action: MaturityNextAction

    @property
    def failed_checks(self) -> list[MaturityCheck]:
        """Return checks that block promotion."""
        return [check for check in self.checks if check.status == "fail"]

    @property
    def watch_checks(self) -> list[MaturityCheck]:
        """Return checks that require caution but do not block local research."""
        return [check for check in self.checks if check.status == "watch"]


def build_research_maturity_report(
    report: EvaluationReport,
    strategy: Strategy | None = None,
    *,
    min_signal_count: int = 30,
) -> ResearchMaturityReport:
    """Build a deterministic research-readiness checklist.

    The checks deliberately mirror common practices visible in mature quant
    projects: compare against baselines, separate research/validation windows,
    treat data quality as a first-class gate, and avoid promoting complex,
    fragile parameter fits.
    """
    checks = [
        _sample_evidence_check(report, min_signal_count=min_signal_count),
        _baseline_check(report),
        _robustness_check(report),
        _data_quality_check(report),
        _complexity_check(strategy),
        _optimization_readiness_check(
            report,
            strategy,
            min_signal_count=min_signal_count,
        ),
    ]
    status = _aggregate_status(checks)
    score = _aggregate_score(checks)
    next_action = _select_next_action(
        checks,
        report,
        strategy,
        min_signal_count=min_signal_count,
    )
    return ResearchMaturityReport(
        status=status,
        score=score,
        checks=checks,
        next_action=next_action,
    )


def render_research_maturity_markdown(maturity: ResearchMaturityReport) -> list[str]:
    """Render a maturity checklist as Markdown lines."""
    lines = [
        "",
        "## Research Maturity Checklist",
        "",
        f"- Overall Status: **{maturity.status.upper()}**",
        f"- Maturity Score: {maturity.score:.0%}",
        f"- Recommended Action: **{maturity.next_action.title}**",
        f"- Action Priority: {maturity.next_action.priority}",
        f"- Rationale: {maturity.next_action.rationale}",
    ]
    if maturity.next_action.commands:
        lines.append("- Suggested Commands:")
        for command in maturity.next_action.commands:
            lines.append(f"  - `{command}`")
    lines += [
        "",
        "| Gate | Status | Evidence | Next Action | Inspired By |",
        "|------|--------|----------|-------------|-------------|",
    ]
    for check in maturity.checks:
        evidence = "<br>".join(check.evidence) if check.evidence else "n/a"
        inspired_by = ", ".join(check.inspired_by) if check.inspired_by else "internal"
        lines.append(
            f"| {check.title} | {check.status.upper()} | {evidence} "
            f"| {check.recommendation} | {inspired_by} |"
        )
    return lines


def _sample_evidence_check(
    report: EvaluationReport,
    *,
    min_signal_count: int,
) -> MaturityCheck:
    signals = report.overall.signal_count
    if signals >= min_signal_count:
        status: MaturityStatus = "pass"
        recommendation = "Use the result for local research ranking, then validate out of sample."
    elif signals >= max(1, min_signal_count // 2):
        status = "watch"
        recommendation = "Expand symbols or date range before treating metrics as stable."
    else:
        status = "fail"
        recommendation = "Do not promote; broaden sampling before mutation or ranking."
    return MaturityCheck(
        check_id="sample_evidence",
        title="Sample Evidence",
        status=status,
        evidence=[f"signals={signals}", f"minimum={min_signal_count}"],
        recommendation=recommendation,
        inspired_by=["Qlib", "vectorbt", "freqtrade"],
    )


def _baseline_check(report: EvaluationReport) -> MaturityCheck:
    benchmark = report.benchmark
    if benchmark is None:
        status: MaturityStatus = "watch"
        evidence = ["benchmark=missing", "random_baseline=missing"]
        recommendation = "Add buy-and-hold and random baseline comparison before promotion."
    elif benchmark.random_baseline_mean is None:
        status = "watch"
        evidence = [
            f"buy_hold_alpha={benchmark.excess_return:+.2%}",
            "random_baseline=missing",
        ]
        recommendation = "Keep buy-and-hold comparison and add random baseline simulation."
    else:
        beat_fraction = benchmark.random_baseline_beat_fraction
        if benchmark.excess_return > 0 and (beat_fraction is None or beat_fraction >= 0.50):
            status = "pass"
            recommendation = "Baseline evidence is usable; keep it in reports."
        else:
            status = "watch"
            recommendation = "Treat the edge as provisional until it beats baseline more clearly."
        evidence = [
            f"buy_hold_alpha={benchmark.excess_return:+.2%}",
            f"random_mean={benchmark.random_baseline_mean:.2%}",
        ]
        if beat_fraction is not None:
            evidence.append(f"random_beat_fraction={beat_fraction:.0%}")
    return MaturityCheck(
        check_id="baseline_protocol",
        title="Baseline Protocol",
        status=status,
        evidence=evidence,
        recommendation=recommendation,
        inspired_by=["vectorbt", "backtrader", "freqtrade"],
    )


def _robustness_check(report: EvaluationReport) -> MaturityCheck:
    anti = report.anti_overfit
    has_holdout = bool(report.walk_forward or report.regime_holdout or report.stress_windows)
    if anti.is_overfit:
        status: MaturityStatus = "fail"
        recommendation = "Simplify or retune; do not promote overfit candidates."
    elif report.walk_forward and anti.walk_forward_pass_rate >= 0.50:
        status = "pass"
        recommendation = "Robustness evidence is present; continue with larger validation."
    elif has_holdout:
        status = "watch"
        recommendation = "Add walk-forward pass-rate evidence before promotion."
    else:
        status = "watch"
        recommendation = "Run walk-forward or stress-window validation."
    evidence = [
        f"train_val_gap={anti.train_val_gap:.1%}",
        f"val_test_gap={anti.val_test_gap:.1%}",
        f"walk_forward_pass_rate={anti.walk_forward_pass_rate:.1%}",
    ]
    return MaturityCheck(
        check_id="robustness_protocol",
        title="Robustness Protocol",
        status=status,
        evidence=evidence,
        recommendation=recommendation,
        inspired_by=["Qlib", "FinRL", "TradeMaster"],
    )


def _data_quality_check(report: EvaluationReport) -> MaturityCheck:
    event_context = report.event_context
    if event_context is None or not event_context.uses_event_indicators:
        return MaturityCheck(
            check_id="data_quality",
            title="Data Quality",
            status="pass",
            evidence=["event_indicators=none"],
            recommendation="OHLCV-only evidence can proceed through normal validation.",
            inspired_by=["OpenBB", "Qlib"],
        )

    evidence = [
        f"provider_coverage={event_context.provider_coverage:.1%}",
        f"proxy_only_coverage={event_context.proxy_only_coverage:.1%}",
        f"indicators={', '.join(event_context.relevant_indicators)}",
    ]
    if event_context.is_proxy_dominant:
        status: MaturityStatus = "fail"
        recommendation = "Fix provider coverage or switch to OHLCV-only variant before mutation."
    elif event_context.has_proxy_caveat:
        status = "watch"
        recommendation = "Keep proxy caveat visible and validate with provider-backed context."
    else:
        status = "pass"
        recommendation = "Provider-backed event context is acceptable for local research."
    return MaturityCheck(
        check_id="data_quality",
        title="Data Quality",
        status=status,
        evidence=evidence,
        recommendation=recommendation,
        inspired_by=["OpenBB", "Qlib", "AI hedge fund"],
    )


def _complexity_check(strategy: Strategy | None) -> MaturityCheck:
    if strategy is None:
        return MaturityCheck(
            check_id="complexity",
            title="Strategy Complexity",
            status="watch",
            evidence=["strategy=missing"],
            recommendation="Attach strategy DSL to report so complexity can be audited.",
            inspired_by=["freqtrade", "Qlib"],
        )

    score = strategy.complexity_score
    condition_count = (
        len(strategy.entry.triggers)
        + len(strategy.entry.conditions)
        + len(strategy.entry.guards)
        + len(strategy.entry.filters)
        + len(strategy.exit.triggers)
        + len(strategy.exit.stop_loss.conditions or [])
    )
    if score <= 0.50:
        status: MaturityStatus = "pass"
        recommendation = "Complexity is within the local research guardrail."
    elif score <= 0.80:
        status = "watch"
        recommendation = "Prefer pruning weak conditions before adding new filters."
    else:
        status = "fail"
        recommendation = "Simplify before promotion; complexity penalty is too high."
    return MaturityCheck(
        check_id="complexity",
        title="Strategy Complexity",
        status=status,
        evidence=[f"complexity_score={score:.1%}", f"conditions={condition_count}"],
        recommendation=recommendation,
        inspired_by=["freqtrade", "vectorbt"],
    )


def _optimization_readiness_check(
    report: EvaluationReport,
    strategy: Strategy | None,
    *,
    min_signal_count: int,
) -> MaturityCheck:
    has_enough_signals = report.overall.signal_count >= min_signal_count
    data_blocked = (
        report.event_context is not None
        and report.event_context.uses_event_indicators
        and report.event_context.is_proxy_dominant
    )
    complexity_blocked = strategy is not None and strategy.complexity_score > 0.80
    if data_blocked or report.anti_overfit.is_overfit or complexity_blocked:
        status: MaturityStatus = "fail"
        recommendation = "Resolve blocking gates before running broader optimization."
    elif has_enough_signals:
        status = "pass"
        recommendation = "Use robust_profit_quality or quality objective for the next search."
    else:
        status = "watch"
        recommendation = "Expand evidence before optimizing thresholds."
    evidence = [
        f"signals={report.overall.signal_count}",
        f"overfit={report.anti_overfit.is_overfit}",
        f"avg_return={report.overall.avg_return:.2%}",
    ]
    return MaturityCheck(
        check_id="optimization_readiness",
        title="Optimization Readiness",
        status=status,
        evidence=evidence,
        recommendation=recommendation,
        inspired_by=["FinRL", "TradeMaster", "freqtrade"],
    )


def _aggregate_status(checks: list[MaturityCheck]) -> MaturityStatus:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "watch" for check in checks):
        return "watch"
    return "pass"


def _aggregate_score(checks: list[MaturityCheck]) -> float:
    if not checks:
        return 0.0
    values = {"pass": 1.0, "watch": 0.5, "fail": 0.0}
    return round(sum(values[check.status] for check in checks) / len(checks), 4)


def _select_next_action(
    checks: list[MaturityCheck],
    report: EvaluationReport,
    strategy: Strategy | None,
    *,
    min_signal_count: int,
) -> MaturityNextAction:
    check_by_id = {check.check_id: check for check in checks}
    strategy_id = strategy.meta.id if strategy is not None else report.strategy_id

    sample = check_by_id.get("sample_evidence")
    if sample is not None and sample.status == "fail":
        return MaturityNextAction(
            action="expand_sample",
            priority="high",
            title="Expand the evidence sample",
            rationale=(
                f"Only {report.overall.signal_count} signals are available, below the "
                f"{min_signal_count}-signal research threshold."
            ),
            commands=[
                f"alphaevo run {strategy_id} --samples 120 --sampling strategy_scoped",
            ],
        )

    data_quality = check_by_id.get("data_quality")
    if data_quality is not None and data_quality.status == "fail":
        return MaturityNextAction(
            action="repair_data",
            priority="high",
            title="Repair data/event coverage before strategy mutation",
            rationale=(
                "Event/news evidence is not provider-backed enough for reliable "
                "optimization or evolution."
            ),
            commands=[
                f"alphaevo run {strategy_id} --sampling strategy_scoped",
                (
                    f"alphaevo strategy revise {strategy_id} "
                    '"switch to OHLCV-only event logic or remove proxy-dominant conditions"'
                ),
            ],
        )

    complexity = check_by_id.get("complexity")
    if complexity is not None and complexity.status == "fail":
        return MaturityNextAction(
            action="simplify_strategy",
            priority="high",
            title="Simplify strategy structure",
            rationale="The DSL is too complex to promote or optimize safely.",
            commands=[
                (
                    f"alphaevo strategy revise {strategy_id} "
                    '"remove low-contribution conditions and keep core triggers plus risk controls"'
                ),
            ],
        )

    baseline = check_by_id.get("baseline_protocol")
    if baseline is not None and baseline.status == "watch":
        return MaturityNextAction(
            action="add_baseline",
            priority="medium",
            title="Add stronger baseline evidence",
            rationale="The run is missing either buy-and-hold or random baseline context.",
            commands=[
                f"alphaevo run {strategy_id} --samples 120 --sampling strategy_scoped",
            ],
        )

    robustness = check_by_id.get("robustness_protocol")
    if robustness is not None and robustness.status == "watch":
        return MaturityNextAction(
            action="run_robustness",
            priority="medium",
            title="Run robustness validation",
            rationale="The strategy has enough raw evidence but lacks walk-forward or stress evidence.",
            commands=[
                f"alphaevo run {strategy_id} --sampling strategy_scoped --wf-folds 5",
            ],
        )

    optimization = check_by_id.get("optimization_readiness")
    if optimization is not None and optimization.status == "pass":
        return MaturityNextAction(
            action="optimize_strategy",
            priority="medium",
            title="Optimize with robust objective",
            rationale="Blocking gates passed, so the next search can focus on robust profit quality.",
            commands=[
                (
                    f"alphaevo optimize {strategy_id} --objective robust_profit_quality "
                    "--spaces entry,params,exit"
                ),
            ],
        )

    return MaturityNextAction(
        action="promote_to_validation",
        priority="low",
        title="Promote to broader validation",
        rationale="All maturity gates passed; validate on a larger universe before showcasing.",
        commands=[
            f"alphaevo run {strategy_id} --samples 120 --sampling strategy_scoped",
        ],
    )
