"""Deterministic analyst rules for AlphaEvo research committee."""

from __future__ import annotations

from alphaevo.models.execution import EvaluationReport, StrategyChange
from alphaevo.models.strategy import Strategy
from alphaevo.research_committee.models import AnalystVerdict


def technical_verdict(strategy: Strategy, report: EvaluationReport) -> AnalystVerdict:
    """Assess signal quality and payoff from the technical strategy view."""
    metrics = report.overall
    evidence = [
        f"signals={metrics.signal_count}",
        f"win_rate={metrics.win_rate:.1%}",
        f"avg_return={metrics.avg_return:.2%}",
        f"p/l={metrics.profit_loss_ratio:.2f}",
    ]
    recommendations: list[str] = []

    if metrics.signal_count < 30:
        status = "fail"
        summary = "Entry stack is too strict for a reliable research sample."
        recommendations.append(
            "Relax one entry gate or switch entry logic only if retest improves."
        )
    elif metrics.avg_return <= 0:
        status = "fail"
        summary = "Signals are tradable but expected return is not yet positive."
        recommendations.append("Prefer exit/risk changes before adding more entry conditions.")
    elif metrics.profit_loss_ratio < 1.0:
        status = "watch"
        summary = "Hit rate exists, but payoff asymmetry is weak."
        recommendations.append("Review take-profit and stop-loss balance.")
    else:
        status = "pass"
        summary = "Technical signal has enough trades and positive payoff shape."

    if strategy.entry.logic == "and" and metrics.signal_count < 30:
        recommendations.append("The current AND entry logic may be over-confirming the setup.")

    return AnalystVerdict(
        analyst="Technical Analyst",
        status=status,  # type: ignore[arg-type]
        summary=summary,
        evidence=evidence,
        recommendations=recommendations,
    )


def risk_verdict(report: EvaluationReport) -> AnalystVerdict:
    """Assess drawdown and loss clustering."""
    metrics = report.overall
    evidence = [
        f"max_drawdown={metrics.max_drawdown:.1%}",
        f"max_consecutive_loss={metrics.max_consecutive_loss}",
        f"avg_loss={metrics.avg_loss_return:.2%}",
    ]
    recommendations: list[str] = []

    if metrics.max_drawdown > 0.45:
        status = "fail"
        summary = "Drawdown is too high for promotion."
        recommendations.append(
            "Widening stop loss alone is not enough; retest holding period and exits."
        )
    elif metrics.max_drawdown > 0.30 or metrics.max_consecutive_loss >= 8:
        status = "watch"
        summary = "Risk is usable for research but still needs exit discipline."
        recommendations.append(
            "Retest stop loss and max holding days before accepting the version."
        )
    else:
        status = "pass"
        summary = "Drawdown and loss clustering are within the research tolerance band."

    return AnalystVerdict(
        analyst="Risk Analyst",
        status=status,  # type: ignore[arg-type]
        summary=summary,
        evidence=evidence,
        recommendations=recommendations,
    )


def overfit_verdict(report: EvaluationReport) -> AnalystVerdict:
    """Assess overfit and sample adequacy risk."""
    anti = report.anti_overfit
    metrics = report.overall
    evidence = [
        f"signal_count={metrics.signal_count}",
        f"train_val_gap={anti.train_val_gap:.1%}",
        f"val_test_gap={anti.val_test_gap:.1%}",
        f"param_sensitivity={anti.param_sensitivity:.1%}",
    ]
    recommendations: list[str] = []

    if metrics.signal_count < 30:
        status = "fail"
        summary = "Sample is too small to trust improvement claims."
        recommendations.append("Treat this as diagnosis only until retest reaches 30+ signals.")
    elif anti.is_overfit:
        status = "fail"
        summary = "Anti-overfit checks reject promotion."
        recommendations.append("Require walk-forward or holdout improvement before promotion.")
    elif metrics.signal_count < 60:
        status = "watch"
        summary = "Sample is acceptable for showcase, but still thin."
        recommendations.append(
            "Use a larger basket before treating this as reusable research evidence."
        )
    else:
        status = "pass"
        summary = "No immediate overfit gate failed."

    return AnalystVerdict(
        analyst="Overfit Critic",
        status=status,  # type: ignore[arg-type]
        summary=summary,
        evidence=evidence,
        recommendations=recommendations,
    )


def data_quality_verdict(
    report: EvaluationReport,
    *,
    data_source: str,
    symbols: list[str],
) -> AnalystVerdict:
    """Assess whether the data context is explicit enough for a showcase result."""
    evidence = [
        f"source={data_source}",
        f"symbols={len(symbols)}",
        f"symbol_list={', '.join(symbols[:8])}",
    ]
    if report.event_context is not None:
        event = report.event_context
        evidence.extend(
            [
                f"event_provider_coverage={event.provider_coverage:.1%}",
                f"event_proxy_only_coverage={event.proxy_only_coverage:.1%}",
                f"event_indicators={', '.join(event.relevant_indicators) or 'none'}",
            ]
        )
    recommendations: list[str] = []

    if len(symbols) < 5:
        status = "watch"
        summary = "Data sample is small; useful for demo but not for broad claims."
        recommendations.append("Keep README claims scoped to the named showcase basket.")
    elif report.event_context and report.event_context.is_proxy_dominant:
        status = "fail"
        summary = "Event/news filters are proxy-dominant, so mutation evidence is not trustworthy."
        recommendations.append(
            "Prioritize event/news provider coverage or switch to an OHLCV-only variant before tuning thresholds."
        )
    elif report.event_context and report.event_context.has_proxy_caveat:
        status = "watch"
        summary = "Some event/news fields rely on proxy context."
        recommendations.append("Label event/news semantics as proxy-backed in reports.")
    else:
        status = "pass"
        summary = "Data source and symbol basket are explicit enough for a showcase."

    return AnalystVerdict(
        analyst="Data Quality Auditor",
        status=status,  # type: ignore[arg-type]
        summary=summary,
        evidence=evidence,
        recommendations=recommendations,
    )


def mutation_planner_verdict(
    report: EvaluationReport,
    mutation_plan: list[StrategyChange],
) -> AnalystVerdict:
    """Explain the deterministic mutation plan being tested."""
    evidence = [
        f"planned_changes={len(mutation_plan)}",
        *[f"{change.target}: {change.from_value} -> {change.to_value}" for change in mutation_plan],
    ]
    recommendations: list[str] = []

    if not mutation_plan:
        status = "watch"
        summary = "No mutation plan is available; evaluate current version only."
    elif report.overall.signal_count < 30:
        status = "pass"
        summary = "Plan starts by unlocking enough signals for a valid retest."
        recommendations.append(
            "Accept only if the retest improves and reaches a usable sample size."
        )
    else:
        status = "pass"
        summary = "Plan changes one controlled lever at a time."
        recommendations.append("Keep each candidate tied to a measured failure mode.")

    return AnalystVerdict(
        analyst="Mutation Planner",
        status=status,  # type: ignore[arg-type]
        summary=summary,
        evidence=evidence,
        recommendations=recommendations,
    )
