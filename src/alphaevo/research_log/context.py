"""Tiered context builder — manages LLM prompt context with three tiers.

Inspired by Hermes-Agent's context_compressor: instead of dumping all
research history into a single prompt block, we provide three tiers:

  Tier 1 — Compressed summary (~200 tokens): Always injected. Contains
    the strategy's current state, key metrics, and 1-line problem statement.

  Tier 2 — Retrieval context (~500 tokens): Injected when relevant.
    Matched playbooks, family-specific lessons, pattern library entries.
    Selected by problem category, not dumped wholesale.

  Tier 3 — Full detail (on demand): Only injected for specific deep dives.
    Complete history text, raw experience records, all failure cases.

This prevents prompt bloat as evolution sessions grow longer and ensures
the LLM sees the most relevant information first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaevo.data.quality import DataQualityReport
    from alphaevo.models.execution import EvaluationReport
    from alphaevo.models.strategy import Strategy
    from alphaevo.orchestrator.evolution import EvolutionRound
    from alphaevo.reflection.experience import ExperienceStore
    from alphaevo.reflection.meta_learner import EvolutionProfile
    from alphaevo.reflection.playbook import PlaybookStore
    from alphaevo.research_log.logger import ResearchLogger
    from alphaevo.strategy.library import PatternLibrary

logger = logging.getLogger(__name__)


@dataclass
class ContextTier:
    """A single tier of context with a token budget."""

    name: str
    content: str = ""
    max_chars: int = 2000  # Rough proxy for tokens (~4 chars/token)

    @property
    def is_empty(self) -> bool:
        return not self.content.strip()

    def truncate(self) -> str:
        """Return content truncated to budget."""
        if len(self.content) <= self.max_chars:
            return self.content
        return self.content[: self.max_chars - 20] + "\n... (truncated)"


@dataclass
class ResearchContext:
    """Three-tier research context for LLM prompt injection."""

    tier1_summary: ContextTier = field(
        default_factory=lambda: ContextTier(name="summary", max_chars=800)
    )
    tier2_retrieval: ContextTier = field(
        default_factory=lambda: ContextTier(name="retrieval", max_chars=2000)
    )
    tier3_detail: ContextTier = field(
        default_factory=lambda: ContextTier(name="detail", max_chars=4000)
    )
    # Tracking: which playbooks were injected for downstream recording
    playbooks_used: list[str] = field(default_factory=list)

    def to_prompt(self, *, include_detail: bool = False) -> str:
        """Assemble the final prompt context.

        By default only Tier 1 + Tier 2 are included.
        Set include_detail=True for the first round or complex diagnoses.
        """
        parts: list[str] = []

        if not self.tier1_summary.is_empty:
            parts.append("## Current State")
            parts.append(self.tier1_summary.truncate())

        if not self.tier2_retrieval.is_empty:
            parts.append("\n## Relevant Research Context")
            parts.append(self.tier2_retrieval.truncate())

        if include_detail and not self.tier3_detail.is_empty:
            parts.append("\n## Detailed History")
            parts.append(self.tier3_detail.truncate())

        return "\n".join(parts)


class ContextBuilder:
    """Builds tiered context from AlphaEvo's research memory modules.

    Replaces the ad-hoc context assembly in EvolutionPipeline with a
    structured approach that prioritizes relevance over recency.
    """

    def __init__(
        self,
        experience_store: ExperienceStore | None = None,
        playbook_store: PlaybookStore | None = None,
        pattern_library: PatternLibrary | None = None,
        research_logger: ResearchLogger | None = None,
    ) -> None:
        self._experience = experience_store
        self._playbooks = playbook_store
        self._patterns = pattern_library
        self._logger = research_logger

    def build(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        rounds: list[EvolutionRound] | None = None,
        meta_profile: EvolutionProfile | None = None,
        data_quality_report: DataQualityReport | None = None,
        family_id: str | None = None,
        round_num: int = 1,
    ) -> ResearchContext:
        """Build three-tier context for the current evolution state."""
        ctx = ResearchContext()

        # ── Tier 1: Compressed state summary ──────────────────────
        ctx.tier1_summary.content = self._build_summary(
            strategy,
            evaluation,
            round_num=round_num,
            data_quality_report=data_quality_report,
        )

        # ── Tier 2: Retrieved relevant context ────────────────────
        tier2_parts: list[str] = []

        # 2a. Classify current problems
        problems = self._classify_problems(
            evaluation,
            data_quality_report=data_quality_report,
        )

        if data_quality_report is not None and data_quality_report.should_prioritize_data_quality:
            data_quality_text = data_quality_report.format_for_prompt()
            if data_quality_text:
                tier2_parts.append(data_quality_text)

        # 2b. Inject matching playbooks
        if self._playbooks and problems:
            matched_ids: list[str] = []
            seen: set[str] = set()
            for cat in problems:
                for pb in self._playbooks.match(cat, limit=2):
                    if pb.playbook_id not in seen:
                        seen.add(pb.playbook_id)
                        matched_ids.append(pb.playbook_id)
            ctx.playbooks_used = matched_ids
            playbook_text = self._playbooks.format_for_prompt(problems, limit=2)
            if playbook_text:
                tier2_parts.append(playbook_text)

        # 2c. Family-specific experience (concise)
        if self._experience and family_id:
            exp_text = self._experience.format_for_prompt(
                family_id=family_id,
                limit=5,
                exclude_test_sources=True,
            )
            if exp_text:
                tier2_parts.append("### Recent Family Lessons")
                tier2_parts.append(exp_text)

        # 2d. Pattern library (only relevant category)
        if self._patterns:
            pattern_text = self._patterns.format_for_prompt(
                category=strategy.meta.category,
                limit=3,
                exclude_test_sources=True,
            )
            if pattern_text:
                tier2_parts.append(pattern_text)

        # 2e. Meta-learner recommendations
        if meta_profile:
            from alphaevo.reflection.meta_learner import MetaLearner

            meta_text = MetaLearner.format_meta_insights_static(meta_profile)
            if meta_text:
                tier2_parts.append(meta_text)

        ctx.tier2_retrieval.content = "\n\n".join(tier2_parts)

        # ── Tier 3: Full detail (only used for round 1 / deep analysis) ──
        tier3_parts: list[str] = []

        # 3a. Full history text
        if rounds:
            tier3_parts.append(self._build_full_history(rounds))

        # 3b. Research log summary
        if self._logger:
            summary = self._logger.get_summary()
            if summary and summary != "No research events recorded yet.":
                tier3_parts.append("### Research Log")
                tier3_parts.append(summary)

        # 3c. Extended experience (more records)
        if self._experience and family_id:
            ext_text = self._experience.format_for_prompt(
                family_id=family_id,
                limit=15,
                exclude_test_sources=True,
            )
            if ext_text:
                tier3_parts.append("### Extended Lessons")
                tier3_parts.append(ext_text)

        ctx.tier3_detail.content = "\n\n".join(tier3_parts)

        return ctx

    # ── Private builders ─────────────────────────────────────────────

    def _build_summary(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        round_num: int = 1,
        data_quality_report: DataQualityReport | None = None,
    ) -> str:
        """Tier 1: Compressed state (~200 tokens)."""
        m = evaluation.overall
        lines = [
            f"Strategy: {strategy.meta.id} v{strategy.meta.version}",
            f"Round: {round_num}",
            f"Score: {evaluation.confidence_score:.1%}",
            f"Win rate: {m.win_rate:.1%} | Signals: {m.signal_count}",
            f"P/L ratio: {m.profit_loss_ratio:.2f} | Drawdown: {m.max_drawdown:.1%}",
            f"Sharpe: {m.sharpe_ratio:.2f}",
            f"Entry logic: {strategy.entry.logic} with {len(strategy.entry.conditions)} conditions",
        ]

        # One-line problem statement
        problems = self._classify_problems(
            evaluation,
            data_quality_report=data_quality_report,
        )
        if problems:
            lines.append(f"Key problems: {', '.join(problems)}")
        if data_quality_report is not None and data_quality_report.risk_level != "low":
            lines.append(f"Data quality: {data_quality_report.summary}")

        return "\n".join(lines)

    def _classify_problems(
        self,
        evaluation: EvaluationReport,
        *,
        data_quality_report: DataQualityReport | None = None,
    ) -> list[str]:
        """Classify current strategy problems into categories."""
        problems: list[str] = []
        m = evaluation.overall

        if m.signal_count < 25:
            problems.append("low_signals")
        if m.win_rate < 0.45:
            problems.append("low_win_rate")
        if m.max_drawdown > 0.20:
            problems.append("high_drawdown")
        if m.profit_loss_ratio < 1.5 and m.win_rate >= 0.45:
            problems.append("low_pl_ratio")

        # Anti-overfit check
        if hasattr(evaluation, "anti_overfit") and evaluation.anti_overfit:
            af = evaluation.anti_overfit
            if af.train_val_gap > 0.10:
                problems.append("overfit")

        if evaluation.event_context is not None:
            event = evaluation.event_context
            if event.relevant_indicators and (
                event.provider_coverage < 0.30 or event.proxy_only_coverage > 0.50
            ):
                problems.append("data_quality")

        if (
            data_quality_report is not None
            and data_quality_report.should_prioritize_data_quality
            and "data_quality" not in problems
        ):
            problems.append("data_quality")

        return problems

    def _build_full_history(self, rounds: list[EvolutionRound]) -> str:
        """Tier 3: Full round-by-round history."""
        if not rounds:
            return ""

        lines = ["### Complete Evolution History"]
        for r in rounds:
            outcome = "improved" if r.improved else "no improvement"
            lines.append(
                f"Round {r.round_num}: score={r.evaluation.confidence_score:.1%} "
                f"(win_rate={r.evaluation.overall.win_rate:.1%}, "
                f"signals={r.evaluation.overall.signal_count}) — {outcome}"
            )
            if r.reflection and r.reflection.proposed_changes:
                for ch in r.reflection.proposed_changes:
                    lines.append(
                        f"  • {ch.change_type.value} {ch.target}: {ch.from_value} → {ch.to_value}"
                    )
        return "\n".join(lines)
