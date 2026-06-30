"""Data quality diagnostics shared by run, evolution, and reporting flows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from alphaevo.data.adapter import DataSourceHealth
from alphaevo.models.execution import EventContextMetrics

DataQualitySeverity = Literal["info", "warning", "error"]


class DataQualityFinding(BaseModel):
    """One data-quality finding from provider health or context coverage."""

    severity: DataQualitySeverity
    category: str
    message: str
    evidence: list[str] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    """Run-level data quality summary used before strategy mutation."""

    checked: bool = True
    findings: list[DataQualityFinding] = Field(default_factory=list)
    source_count: int = 0
    unhealthy_sources: int = 0
    disabled_sources: int = 0
    event_provider_coverage: float | None = None
    event_proxy_only_coverage: float | None = None
    event_relevant_indicators: list[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True when a finding should hard-fail downstream use."""
        return any(finding.severity == "error" for finding in self.findings)

    @property
    def warning_count(self) -> int:
        """Return the number of warning findings."""
        return sum(1 for finding in self.findings if finding.severity == "warning")

    @property
    def should_prioritize_data_quality(self) -> bool:
        """Whether reflection should diagnose data quality before strategy rules."""
        priority_categories = {
            "source_disabled",
            "source_unhealthy",
            "source_recovered_errors",
            "primary_source_unhealthy",
            "low_event_provider_coverage",
            "proxy_dominant_event_context",
            "event_proxy_caveat",
        }
        return any(finding.category in priority_categories for finding in self.findings)

    @property
    def blocks_strategy_iteration(self) -> bool:
        """Whether strategy mutation should pause until data quality is remediated."""
        blocking_categories = {
            "low_event_provider_coverage",
            "proxy_dominant_event_context",
        }
        return self.has_errors or any(
            finding.category in blocking_categories for finding in self.findings
        )

    @property
    def risk_level(self) -> Literal["low", "medium", "high"]:
        """Coarse risk label for CLI and LLM context."""
        if self.blocks_strategy_iteration:
            return "high"
        if self.warning_count or self.should_prioritize_data_quality:
            return "medium"
        return "low"

    @property
    def summary(self) -> str:
        """Short human-readable summary."""
        if not self.findings:
            return "No material data-quality findings."
        first = self.findings[0]
        return f"{self.risk_level}: {first.category} ({len(self.findings)} finding(s))"

    def format_for_prompt(self) -> str:
        """Render concise context for strategy reflection prompts."""
        if not self.findings:
            return ""

        lines = [
            "### Data Quality Gate",
            f"- Risk: {self.risk_level}",
            f"- Blocks strategy iteration: {self.blocks_strategy_iteration}",
        ]
        if self.event_relevant_indicators:
            lines.append(f"- Event/news indicators: {', '.join(self.event_relevant_indicators)}")
        if self.event_provider_coverage is not None:
            lines.append(f"- Event provider coverage: {self.event_provider_coverage:.1%}")
        if self.event_proxy_only_coverage is not None:
            lines.append(f"- Event proxy-only coverage: {self.event_proxy_only_coverage:.1%}")

        for finding in self.findings[:5]:
            evidence = f" ({'; '.join(finding.evidence[:3])})" if finding.evidence else ""
            lines.append(f"- [{finding.severity}] {finding.category}: {finding.message}{evidence}")

        return "\n".join(lines)


def build_data_quality_report(
    *,
    event_context: EventContextMetrics | None = None,
    data_source_health: list[DataSourceHealth] | None = None,
    min_event_provider_coverage: float = 0.30,
    max_proxy_only_coverage: float = 0.50,
) -> DataQualityReport:
    """Build a run-level data quality report from source and context evidence."""
    findings: list[DataQualityFinding] = []
    health = list(data_source_health or [])

    disabled_sources = [source for source in health if source.disabled]
    unhealthy_sources = [
        source
        for source in health
        if source.disabled or source.consecutive_failures > 0 or bool(source.last_error)
    ]

    if disabled_sources:
        findings.append(
            DataQualityFinding(
                severity="warning",
                category="source_disabled",
                message="One or more data sources are temporarily disabled after repeated failures.",
                evidence=[_format_source_evidence(source) for source in disabled_sources],
            )
        )

    repeated_failures = [
        source for source in health if not source.disabled and source.consecutive_failures >= 2
    ]
    if repeated_failures:
        findings.append(
            DataQualityFinding(
                severity="warning",
                category="source_unhealthy",
                message="One or more data sources are failing repeatedly during fallback.",
                evidence=[_format_source_evidence(source) for source in repeated_failures],
            )
        )

    if health:
        primary = min(health, key=lambda source: source.priority)
        if primary.failure_count > 0 and primary.success_count == 0:
            findings.append(
                DataQualityFinding(
                    severity="warning",
                    category="primary_source_unhealthy",
                    message="The highest-priority data source has failures but no successes.",
                    evidence=[_format_source_evidence(primary)],
                )
            )

        recovered = [
            source
            for source in health
            if source.failure_count > 0
            and source.success_count > 0
            and source not in disabled_sources
            and source not in repeated_failures
        ]
        if recovered:
            findings.append(
                DataQualityFinding(
                    severity="info",
                    category="source_recovered_errors",
                    message="Data source fallback recovered from provider errors during the run.",
                    evidence=[_format_source_evidence(source) for source in recovered[:3]],
                )
            )

    if event_context is not None and event_context.uses_event_indicators:
        provider_coverage = event_context.provider_coverage
        proxy_only_coverage = event_context.proxy_only_coverage
        provider_symbols = event_context.provider_symbols + event_context.mixed_symbols

        if provider_coverage < min_event_provider_coverage:
            findings.append(
                DataQualityFinding(
                    severity="warning",
                    category="low_event_provider_coverage",
                    message=(
                        "Event/news provider coverage is too low for reliable strategy mutation."
                    ),
                    evidence=[
                        f"provider_coverage={provider_coverage:.1%}",
                        f"provider_or_mixed_symbols={provider_symbols}/{event_context.total_symbols}",
                    ],
                )
            )
        if proxy_only_coverage > max_proxy_only_coverage:
            findings.append(
                DataQualityFinding(
                    severity="warning",
                    category="proxy_dominant_event_context",
                    message=(
                        "Event/news context is proxy-dominant; diagnose data coverage before "
                        "tuning event thresholds."
                    ),
                    evidence=[
                        f"proxy_only_coverage={proxy_only_coverage:.1%}",
                        f"proxy_symbols={event_context.proxy_symbols}/{event_context.total_symbols}",
                    ],
                )
            )
        elif event_context.has_proxy_caveat:
            findings.append(
                DataQualityFinding(
                    severity="info",
                    category="event_proxy_caveat",
                    message="Some active event/news values are price/volume proxy-derived.",
                    evidence=[
                        f"proxy_only_coverage={proxy_only_coverage:.1%}",
                        f"source_breakdown={event_context.source_breakdown}",
                    ],
                )
            )

    return DataQualityReport(
        findings=findings,
        source_count=len(health),
        unhealthy_sources=len(unhealthy_sources),
        disabled_sources=len(disabled_sources),
        event_provider_coverage=(
            event_context.provider_coverage
            if event_context is not None and event_context.uses_event_indicators
            else None
        ),
        event_proxy_only_coverage=(
            event_context.proxy_only_coverage
            if event_context is not None and event_context.uses_event_indicators
            else None
        ),
        event_relevant_indicators=(
            list(event_context.relevant_indicators)
            if event_context is not None and event_context.uses_event_indicators
            else []
        ),
    )


def _format_source_evidence(source: DataSourceHealth) -> str:
    """Render one source-health row for logs and prompts."""
    parts = [
        f"{source.name}",
        f"priority={source.priority}",
        f"success={source.success_count}",
        f"failure={source.failure_count}",
        f"consecutive={source.consecutive_failures}",
    ]
    if source.disabled:
        parts.append("disabled=true")
    if source.last_error:
        parts.append(f"last_error={source.last_error}")
    return ", ".join(parts)
