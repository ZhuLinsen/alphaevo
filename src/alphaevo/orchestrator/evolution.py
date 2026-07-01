"""Evolution pipeline — multi-round strategy improvement loop.

Orchestrates: run → reflect → mutate → save → repeat.

Supports three methods:
- LLM: Full LLM-driven reflection and mutation
- param_search: Grid search over tunable parameters (no LLM)
- hybrid: LLM reflection + param search fine-tuning
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, TypedDict

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.core.config import AppConfig
from alphaevo.core.llm import LLMClient, LLMNotAvailableError
from alphaevo.models.enums import ChangeType, EvolutionMethod, SamplingMethod
from alphaevo.models.execution import (
    CandidateExperiment,
    EvaluationReport,
    ReflectionResult,
    SampleBatch,
    StrategyChange,
)
from alphaevo.models.strategy import MarketHypothesisAssessment, Strategy, StrategyCondition
from alphaevo.orchestrator.pipeline import RunPipeline, RunResult
from alphaevo.reflection.analyzer import ReflectionAnalyzer
from alphaevo.reflection.critic import SelfCritic
from alphaevo.reflection.experience import ExperienceQuery, ExperienceRecord, ExperienceStore
from alphaevo.reflection.meta_learner import EvolutionProfile, MetaLearner
from alphaevo.reflection.mutator import MutationError, StrategyMutator
from alphaevo.reflection.playbook import PlaybookStore
from alphaevo.research_log import ResearchLogger
from alphaevo.research_log.context import ContextBuilder
from alphaevo.research_log.trajectory import TrajectoryCollector, TrajectoryStep
from alphaevo.strategy.library import PatternLibrary
from alphaevo.strategy.tunable import (
    is_integer_tunable_target,
    is_period_tunable_target,
    resolve_tunable_target,
    tune_period_value,
)

if TYPE_CHECKING:
    import pandas as pd

    from alphaevo.alpha_factory.factory import AlphaFactory
    from alphaevo.strategy.store import StrategyStore

logger = logging.getLogger(__name__)


class _CrossStrategyBucket(TypedDict):
    total: int
    succeeded: int
    families: set[str]
    strategies: list[str]


@dataclass
class _ScreenedMutationCandidate:
    """One mutation candidate that was validated on the current batch."""

    strategy: Strategy
    changes: list[StrategyChange]
    candidate: CandidateExperiment | None = None
    evaluation: EvaluationReport | None = None
    score_delta: float = 0.0
    signal_delta: int = 0


@dataclass
class EvolutionRound:
    """Result of a single evolution round."""

    round_num: int
    strategy: Strategy
    evaluation: EvaluationReport
    batch: SampleBatch | None = None
    reflection: ReflectionResult | None = None
    improved: bool = False
    hypothesis_status: str = ""
    hypothesis_rationale: str = ""
    hypothesis_next_step: str = ""
    meta_insights: list[str] = field(default_factory=list)
    experience_lessons: list[str] = field(default_factory=list)
    pattern_context: list[str] = field(default_factory=list)
    cross_strategy_memory: list[str] = field(default_factory=list)
    recommended_method: str | None = None
    recommended_intensity: float | None = None
    recommended_max_changes: int | None = None


@dataclass
class EvolutionResult:
    """Result of the full evolution pipeline."""

    original_strategy_id: str
    rounds: list[EvolutionRound] = field(default_factory=list)
    champion_id: str | None = None
    champion_score: float = 0.0
    total_rounds: int = 0
    early_stopped: bool = False
    stop_reason: str = ""
    research_log: ResearchLogger | None = field(default=None, repr=False)
    baseline_param_search_score: float | None = None
    trajectory: object | None = field(default=None, repr=False)  # EvolutionTrajectory

    @property
    def improvement(self) -> float:
        """Score improvement from first to best round."""
        if not self.rounds:
            return 0.0
        first = self.rounds[0].evaluation.confidence_score
        return self.champion_score - first

    @property
    def velocity(self) -> list[float]:
        """Per-round score deltas."""
        if len(self.rounds) < 2:
            return []
        scores = [r.evaluation.confidence_score for r in self.rounds]
        return [scores[i] - scores[i - 1] for i in range(1, len(scores))]

    @property
    def efficiency(self) -> float:
        """Fraction of rounds that improved."""
        improving = sum(1 for r in self.rounds if r.improved)
        return improving / len(self.rounds) if self.rounds else 0.0


class EvolutionPipeline:
    """Multi-round strategy evolution loop.

    Flow per round:
    1. Run backtest (via RunPipeline)
    2. Reflect on results (via ReflectionAnalyzer / heuristics)
    3. Mutate strategy (via StrategyMutator)
    4. Save new version
    5. Check safety guardrails
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._run_pipeline = RunPipeline(config)
        self._mutator = StrategyMutator(
            max_changes=config.evolution.max_changes_per_round,
            complexity_limit=config.evolution.complexity_limit,
        )
        self._llm: LLMClient | None = None
        self._analyzer: ReflectionAnalyzer | None = None
        self._experience_store = ExperienceStore(config.db_path)
        self._meta_learner = MetaLearner(self._experience_store)
        self._pattern_library = PatternLibrary(db_path=config.db_path)
        self._critic = SelfCritic(
            experience_store=self._experience_store,
            complexity_limit=config.evolution.complexity_limit,
        )
        self._alpha_factory: AlphaFactory | None = None
        self._playbook_store = PlaybookStore(db_path=config.db_path)
        self._failed_change_threshold = 2
        self.research_log = ResearchLogger()
        self._context_builder = ContextBuilder(
            experience_store=self._experience_store,
            playbook_store=self._playbook_store,
            pattern_library=self._pattern_library,
            research_logger=self.research_log,
        )
        self._recovery_attempted = False
        self._recovery_round: int | None = None
        self._recovery_mode: str | None = None

    @property
    def store(self) -> StrategyStore:
        return self._run_pipeline.store

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient.from_config(self.config)
        return self._llm

    def _ensure_analyzer(self) -> ReflectionAnalyzer:
        if self._analyzer is None:
            llm = self._ensure_llm()
            self._analyzer = ReflectionAnalyzer(
                llm,
                max_changes=self.config.evolution.max_changes_per_round,
                num_candidates=self.config.evolution.num_candidates,
            )
        return self._analyzer

    def _ensure_alpha_factory(self) -> AlphaFactory:
        """Lazily initialize the AlphaFactory for factor discovery."""
        if self._alpha_factory is None:
            from alphaevo.alpha_factory.factory import AlphaFactory

            llm = self._ensure_llm()
            self._alpha_factory = AlphaFactory(
                llm,
                db_path=self.config.db_path,
            )
        return self._alpha_factory

    def evolve(
        self,
        strategy_id: str,
        *,
        rounds: int = 5,
        method: EvolutionMethod = EvolutionMethod.HYBRID,
        max_symbols: int = 60,
        sampling_method: SamplingMethod = SamplingMethod.REPRESENTATIVE,
        date_range: tuple[date, date] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> EvolutionResult:
        """Run the evolution loop synchronously."""
        import asyncio

        return asyncio.run(
            self.evolve_async(
                strategy_id,
                rounds=rounds,
                method=method,
                max_symbols=max_symbols,
                sampling_method=sampling_method,
                date_range=date_range,
                on_progress=on_progress,
            )
        )

    async def evolve_async(
        self,
        strategy_id: str,
        *,
        rounds: int = 5,
        method: EvolutionMethod = EvolutionMethod.HYBRID,
        max_symbols: int = 60,
        sampling_method: SamplingMethod = SamplingMethod.REPRESENTATIVE,
        date_range: tuple[date, date] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> EvolutionResult:
        """Run the evolution loop."""

        def _progress(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        # Ensure builtin strategies are loaded
        self._run_pipeline.ensure_builtin_strategies()

        if date_range is None:
            end = date.today()
            start = end - timedelta(days=365)
            date_range = (start, end)

        result = EvolutionResult(
            original_strategy_id=strategy_id,
            total_rounds=rounds,
        )

        # Derive the strategy family from the original id (strip version suffix)
        family_id = strategy_id.rsplit("_v", 1)[0] if "_v" in strategy_id else strategy_id

        current_id = strategy_id
        best_score: float | None = None
        best_id = strategy_id

        # Pending experience: track changes applied at round N-1 to record
        # their outcome after round N's score is known.
        # Format: (change, score_before, strategy_id, round_num, hypothesis, source, regime)
        pending_exp: list[tuple[StrategyChange, float, str, int, str, str, str]] = []
        self._recovery_attempted = False
        self._recovery_round = None
        self._recovery_mode = None

        # Log initial hypothesis
        self.research_log.log(
            "hypothesis",
            f"Evolving {strategy_id} for {rounds} rounds using {method.value} method",
            strategy_id=strategy_id,
            data={"rounds": rounds, "method": method.value, "max_symbols": max_symbols},
        )

        # Initialize trajectory collector for training data export
        _trajectory = TrajectoryCollector(
            trajectory_id=f"{family_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            strategy_family=family_id,
        )
        _trajectory.set_metadata("method", method.value)
        _trajectory.set_metadata("rounds", rounds)
        _trajectory.set_metadata("max_symbols", max_symbols)

        for round_num in range(1, rounds + 1):
            _progress(f"Round {round_num}/{rounds}: Running {current_id}...")

            # Step 1: Run backtest
            try:
                run_result = await self._run_pipeline.run(
                    current_id,
                    max_symbols=max_symbols,
                    sampling_method=sampling_method,
                    date_range=date_range,
                    on_progress=lambda msg: _progress(f"  {msg}"),
                )
            except Exception as e:
                _progress(f"Round {round_num} failed: {e}")
                if round_num == 1:
                    # First round failure is fatal (original strategy is broken)
                    result.early_stopped = True
                    result.stop_reason = f"Backtest failed: {e}"
                    break
                # Later rounds: LLM mutation may have introduced invalid
                # indicators/conditions. Revert to best known strategy and
                # continue instead of aborting the whole evolution.
                _progress(f"  Reverting to last champion {best_id} (mutated strategy was invalid)")
                # Record pending changes as failed with neutral score delta,
                # because this round had no valid evaluation output.
                failed_score = pending_exp[0][1] if pending_exp else (best_score or 0.0)
                self._record_pending_experience(
                    pending_exp,
                    family_id,
                    score_after=failed_score,
                    worked=False,
                    failure_reason=str(e),
                )
                pending_exp = []
                current_id = best_id
                continue

            score = run_result.evaluation.confidence_score
            improved = best_score is not None and score > best_score
            signal_count = run_result.evaluation.overall.signal_count
            min_signal_count = self.config.evolution.min_signal_count
            sparse_signal_block = (
                run_result.batch.insufficient_signals or signal_count < min_signal_count
            )
            hypothesis_assessment = run_result.strategy.assess_market_hypothesis(
                run_result.evaluation
            )
            data_quality_report = run_result.data_quality
            data_quality_block = data_quality_report.blocks_strategy_iteration
            if data_quality_report.should_prioritize_data_quality:
                _progress(f"  Data quality gate: {data_quality_report.summary}")
                self.research_log.log(
                    "insight",
                    f"Data quality gate triggered: {data_quality_report.summary}",
                    round_num=round_num,
                    strategy_id=current_id,
                    data={
                        "risk_level": data_quality_report.risk_level,
                        "blocks_strategy_iteration": data_quality_block,
                        "findings": [
                            finding.model_dump(mode="json")
                            for finding in data_quality_report.findings
                        ],
                    },
                )

            # Compute param sensitivity if engine and data are available
            if (
                run_result._engine is not None
                and run_result._data is not None
                and score > 0
                and not sparse_signal_block
            ):
                try:
                    from alphaevo.evaluator.metrics import Evaluator as _Eval

                    sensitivity = _Eval().compute_param_sensitivity(
                        run_result.strategy,
                        [s for s in run_result.backtest_result.signals if s.exit_price is not None],
                        score,
                        run_result._engine,
                        run_result._data,
                        run_result.batch,
                        contexts=run_result._contexts,
                    )
                    run_result.evaluation.anti_overfit.param_sensitivity = sensitivity
                    # Recompute confidence with real sensitivity
                    run_result.evaluation.confidence_score = _Eval().compute_confidence_score(
                        run_result.evaluation.overall,
                        run_result.strategy.complexity_score,
                        anti_fit=run_result.evaluation.anti_overfit,
                    )
                    score = run_result.evaluation.confidence_score
                    improved = best_score is not None and score > best_score
                    if sensitivity > 0.3:
                        _progress(
                            f"  ⚠️ Param sensitivity={sensitivity:.1%} "
                            f"(fragile — small param changes degrade performance)"
                        )
                except Exception as e:
                    logger.debug("Param sensitivity skipped: %s", e)

            # Record pending experience from previous round now that we know the outcome
            if pending_exp:
                if sparse_signal_block:
                    self._record_pending_experience(
                        pending_exp,
                        family_id,
                        score_after=score,
                        worked=False,
                        failure_reason=(f"insufficient_signals:{signal_count}/{min_signal_count}"),
                    )
                else:
                    self._record_pending_experience(
                        pending_exp,
                        family_id,
                        score_after=score,
                        worked=None,
                    )
                pending_exp = []

            if best_score is None:
                # Always seed the baseline with round 1's actual score, even if it
                # is overfit, so the CLI/result summary stays grounded in reality.
                best_score = score
                best_id = current_id
                improved = round_num == 1
            elif improved:
                gate_reason = self._promotion_guard_reason(run_result.evaluation)
                if gate_reason is not None:
                    _progress(f"  ⚠️ {gate_reason} — score {score:.1%} NOT accepted as champion")
                    improved = False
                else:
                    best_score = score
                    best_id = current_id

            # Log observation: backtest result
            self.research_log.log(
                "observation",
                (
                    f"Score={score:.1%}, win_rate={run_result.evaluation.overall.win_rate:.1%}, "
                    f"avg_return={run_result.evaluation.overall.avg_return:.2%}, "
                    f"signals={run_result.evaluation.overall.signal_count}"
                ),
                round_num=round_num,
                strategy_id=current_id,
                data={
                    "score": round(score, 4),
                    "win_rate": round(run_result.evaluation.overall.win_rate, 4),
                    "avg_return": round(run_result.evaluation.overall.avg_return, 4),
                    "signal_count": run_result.evaluation.overall.signal_count,
                    "improved": improved,
                    "min_signal_count": min_signal_count,
                    "insufficient_signals": sparse_signal_block,
                    "sampling_attempts": run_result.batch.sampling_attempt,
                },
            )
            self.research_log.log(
                "insight",
                (
                    f"Hypothesis assessment={hypothesis_assessment.status}: "
                    f"{hypothesis_assessment.rationale}"
                ),
                round_num=round_num,
                strategy_id=current_id,
                data={
                    "hypothesis_status": hypothesis_assessment.status,
                    "next_step": hypothesis_assessment.next_step,
                },
            )

            if sparse_signal_block:
                self.research_log.log(
                    "decision",
                    (
                        f"Signal count {signal_count}/{min_signal_count} remained too low "
                        f"after {run_result.batch.sampling_attempt} sampling attempt(s); "
                        "blocking further evolution"
                    ),
                    round_num=round_num,
                    strategy_id=current_id,
                    data={
                        "signal_count": signal_count,
                        "min_signal_count": min_signal_count,
                        "sampling_attempts": run_result.batch.sampling_attempt,
                    },
                )

            _progress(
                f"Round {round_num}: score={score:.1%} "
                f"{'↑ improved' if improved else '→ no improvement'}"
            )

            # Step 2: Reflect — build history and experience context for LLM
            reflection: ReflectionResult | None = None
            meta_profile: EvolutionProfile | None = None
            family_lessons: list[ExperienceRecord] = []
            # Track critic and playbook info for trajectory recording
            _round_verdict_summary: str = ""
            _round_playbooks: str = ""
            _round_llm_telemetry: dict[str, object] = {}
            if (
                round_num < rounds and not sparse_signal_block and not data_quality_block
            ):  # Don't reflect on last round
                family_lessons = self._experience_store.get_family_lessons(
                    family_id,
                    limit=10,
                    exclude_test_sources=True,
                )

                # MetaLearner: adaptive intensity and insights
                meta_profile = self._meta_learner.analyze(
                    family_id=family_id,
                    evaluation=run_result.evaluation,
                )

                # Build tiered context (replaces ad-hoc assembly)
                research_ctx = self._context_builder.build(
                    run_result.strategy,
                    run_result.evaluation,
                    rounds=result.rounds,
                    meta_profile=meta_profile,
                    data_quality_report=data_quality_report,
                    family_id=family_id,
                    round_num=round_num,
                )
                # Tier 1+2 always; Tier 3 for first round or deep analysis
                include_detail = round_num <= 2
                assembled_context = research_ctx.to_prompt(include_detail=include_detail)

                # Record which playbooks were injected for trajectory
                if research_ctx.playbooks_used:
                    _round_playbooks = ", ".join(research_ctx.playbooks_used)

                # Use MetaLearner-recommended intensity, blended with
                # the default annealing schedule.
                default_intensity = max(
                    0.4,
                    1.5 - (round_num - 1) * (1.0 / max(1, rounds - 1)),
                )
                recovery_mode = self._recovery_mode
                self._recovery_mode = None
                intensity = self._guided_intensity(
                    default_intensity=default_intensity,
                    meta_intensity=meta_profile.recommended_intensity,
                    assessment_status=hypothesis_assessment.status,
                    recovery_mode=recovery_mode,
                )

                # Research log summary for LLM memory continuity
                research_summary = self.research_log.get_summary()

                # assembled_context already contains playbooks + experience +
                # patterns + meta-insights from ContextBuilder.build().
                # No need to inject playbook text separately.

                reflection = self._do_reflection(
                    run_result.strategy,
                    run_result.evaluation,
                    method,
                    history_text="",  # history is already in tiered context
                    experience_text="",  # experience is already in tiered context
                    meta_text=assembled_context,
                    research_summary=research_summary,
                    intensity=intensity,
                )
                if reflection:
                    refined_assessment = self._refine_hypothesis_assessment(
                        hypothesis_assessment,
                        reflection,
                    )
                    if refined_assessment.status != hypothesis_assessment.status:
                        _progress(
                            "  Hypothesis lens escalated "
                            f"{hypothesis_assessment.status.replace('_', '-')} → "
                            f"{refined_assessment.status.replace('_', '-')}"
                        )
                        self.research_log.log(
                            "insight",
                            (
                                "Reflection evidence upgraded the diagnosis from "
                                f"{hypothesis_assessment.status} to {refined_assessment.status}"
                            ),
                            round_num=round_num,
                            strategy_id=current_id,
                            data={
                                "from_status": hypothesis_assessment.status,
                                "to_status": refined_assessment.status,
                            },
                        )
                        hypothesis_assessment = refined_assessment

                    if reflection.llm_telemetry is not None:
                        _round_llm_telemetry = reflection.llm_telemetry.model_dump(mode="json")
                        timeout_failures = sum(
                            1
                            for call in reflection.llm_telemetry.calls
                            if not call.success
                            and any(
                                token in call.error.lower() for token in ("timeout", "timed out")
                            )
                        )
                        reflection_content = (
                            f"path={reflection.llm_telemetry.path}, "
                            f"calls={len(reflection.llm_telemetry.calls)}, "
                            f"total={reflection.llm_telemetry.total_duration_ms}ms"
                        )
                        if timeout_failures:
                            reflection_content += f", timeout_failures={timeout_failures}"
                        if reflection.llm_telemetry.fallback_trigger:
                            reflection_content += (
                                f", fallback={reflection.llm_telemetry.fallback_trigger}"
                            )
                        self.research_log.log(
                            "reflection",
                            reflection_content,
                            round_num=round_num,
                            strategy_id=current_id,
                            data={
                                "path": reflection.llm_telemetry.path,
                                "calls": len(reflection.llm_telemetry.calls),
                                "total_duration_ms": reflection.llm_telemetry.total_duration_ms,
                                "timeout_failures": timeout_failures,
                                "fallback_trigger": reflection.llm_telemetry.fallback_trigger,
                            },
                        )

                    # Log diagnosis from two-step reflection
                    if reflection.diagnosis:
                        self.research_log.log(
                            "diagnosis",
                            reflection.diagnosis,
                            round_num=round_num,
                            strategy_id=current_id,
                            data={"failure_patterns": reflection.failure_patterns},
                        )
                    elif reflection.failure_patterns:
                        self.research_log.log(
                            "diagnosis",
                            "; ".join(reflection.failure_patterns[:3]),
                            round_num=round_num,
                            strategy_id=current_id,
                            data={"failure_patterns": reflection.failure_patterns},
                        )

                    # Log candidate experiments if available
                    if reflection.candidates:
                        for i, cand in enumerate(reflection.candidates):
                            self.research_log.log(
                                "experiment",
                                (
                                    f"Candidate {i + 1}: {cand.hypothesis.hypothesis} "
                                    f"(priority={cand.priority_score:.2f})"
                                ),
                                round_num=round_num,
                                strategy_id=current_id,
                                data={
                                    "hypothesis": cand.hypothesis.problem,
                                    "expected": cand.hypothesis.expected_outcome,
                                    "num_changes": len(cand.proposed_changes),
                                },
                            )

                    # Rank candidates through critic (rule-based scoring)
                    if reflection.candidates:
                        ranked = self._critic.rank_candidates(
                            run_result.strategy,
                            run_result.evaluation,
                            reflection.candidates,
                        )
                        reflection.candidates = ranked
                        # Update proposed_changes to top candidate
                        if ranked:
                            reflection.proposed_changes = ranked[0].proposed_changes[
                                : self.config.evolution.max_changes_per_round
                            ]
                            aligned_candidate = self._select_hypothesis_aligned_candidate(
                                ranked,
                                hypothesis_assessment.status,
                            )
                            if (
                                aligned_candidate is not None
                                and aligned_candidate.proposed_changes
                                != reflection.proposed_changes
                            ):
                                reflection.proposed_changes = aligned_candidate.proposed_changes[
                                    : self.config.evolution.max_changes_per_round
                                ]
                                _progress(
                                    "  Hypothesis lens promoted a more aligned experiment bundle"
                                )
                                self.research_log.log(
                                    "decision",
                                    (
                                        "Hypothesis lens promoted a candidate aligned with "
                                        f"{hypothesis_assessment.status}"
                                    ),
                                    round_num=round_num,
                                    strategy_id=current_id,
                                    data={"hypothesis_status": hypothesis_assessment.status},
                                )

                    reflection.proposed_changes = self._apply_hypothesis_guidance(
                        run_result.strategy,
                        run_result.evaluation,
                        hypothesis_assessment.status,
                        reflection.proposed_changes,
                    )

                    # SelfCritic: validate changes before mutation
                    verdict = self._critic.critique(
                        run_result.strategy,
                        run_result.evaluation,
                        reflection,
                    )
                    if verdict.rejected:
                        for change, reason in verdict.rejected:
                            _progress(
                                f"  Critic rejected: {change.change_type.value} "
                                f"on {change.target} — {reason}"
                            )
                    if verdict.warnings:
                        for w in verdict.warnings:
                            _progress(f"  Critic warning: {w}")
                    reflection.proposed_changes = verdict.approved

                    # Serialize verdict summary for trajectory recording
                    _rejected_summary = (
                        "; ".join(
                            f"{ch.change_type.value}({ch.target}): {reason}"
                            for ch, reason in verdict.rejected
                        )
                        if verdict.rejected
                        else ""
                    )
                    _round_verdict_summary = (
                        f"approved={len(verdict.approved)}, rejected={len(verdict.rejected)}"
                    )
                    if verdict.warnings:
                        _round_verdict_summary += f", warnings={len(verdict.warnings)}"
                    if _rejected_summary:
                        _round_verdict_summary += f" [{_rejected_summary}]"

                    # Additionally filter out repeated failures
                    reflection.proposed_changes = self._filter_failed_repeated_changes(
                        family_id,
                        reflection.proposed_changes,
                    )
                    _progress(f"  Reflection: {len(reflection.proposed_changes)} changes proposed")

                    # ── Factor Discovery ──
                    # When reflection has few actionable changes and method
                    # uses LLM, attempt to discover new indicators via
                    # AlphaFactory and inject them as DISCOVER_FACTOR changes.
                    if (
                        len(reflection.proposed_changes) == 0
                        and method != EvolutionMethod.PARAM_SEARCH
                        and run_result._data
                    ):
                        _progress("  No conventional changes — attempting factor discovery...")
                        discovered = await self._try_factor_discovery(
                            run_result.strategy,
                            run_result.evaluation,
                            run_result._data,
                        )
                        if discovered:
                            reflection.proposed_changes.extend(discovered)
                            _progress(f"  Discovered {len(discovered)} new factor(s) for injection")

                    if len(reflection.proposed_changes) == 0:
                        exploration_changes = self._exploration_changes(
                            run_result.strategy,
                            run_result.evaluation,
                        )
                        if exploration_changes:
                            reflection.proposed_changes.extend(
                                exploration_changes[: self.config.evolution.max_changes_per_round]
                            )
                            _progress(
                                "  Exploration mode injected "
                                f"{len(reflection.proposed_changes)} structural change(s)"
                            )
                            self.research_log.log(
                                "experiment",
                                "Exploration mode activated to escape stagnation",
                                round_num=round_num,
                                strategy_id=current_id,
                                data={"changes": len(reflection.proposed_changes)},
                            )

                    # Log decision: approved changes
                    if reflection.proposed_changes:
                        change_descs = [
                            f"{ch.change_type.value} {ch.target}: {ch.from_value}→{ch.to_value}"
                            for ch in reflection.proposed_changes
                        ]
                        self.research_log.log(
                            "decision",
                            f"Approved {len(reflection.proposed_changes)} changes: "
                            + "; ".join(change_descs),
                            round_num=round_num,
                            strategy_id=current_id,
                            data={
                                "approved_count": len(reflection.proposed_changes),
                                "rejected_count": len(verdict.rejected) if verdict.rejected else 0,
                                "intensity": round(intensity, 2),
                            },
                        )

            # Extract experience lessons for reporting
            _exp_lessons: list[str] = []
            for rec in family_lessons:
                outcome = "worked" if rec.worked else "failed"
                summary = (
                    f"{rec.change_type.value} on {rec.target} ({outcome}, "
                    f"score {rec.score_before:.1%}→{rec.score_after:.1%})"
                )
                if rec.lesson:
                    summary += f" — {rec.lesson}"
                _exp_lessons.append(summary)

            # Extract pattern context from meta profile and pattern library
            _patterns: list[str] = []
            if meta_profile is not None:
                for ins in meta_profile.insights:
                    if ins.insight_type == "problem_pattern" and ins.data:
                        patterns_data = ins.data.get("patterns", {})
                        for pat, count in list(patterns_data.items())[:3]:
                            _patterns.append(f"{pat} (succeeded {count}x)")
                    elif ins.insight_type == "change_effectiveness" and ins.data:
                        ranked = ins.data.get("ranked", [])
                        if ranked:
                            _patterns.append(f"Most effective change type: {ranked[0]}")
            # Add reusable patterns from pattern library
            library_patterns = self._pattern_library.get_best_patterns(
                category=run_result.strategy.meta.category,
                exclude_test_sources=True,
                limit=3,
            )
            for lp in library_patterns:
                _patterns.append(
                    f"[{lp.pattern_type}] {lp.name} "
                    f"(score={lp.confidence_score:.0%}, wr={lp.win_rate:.0%})"
                )

            evolution_round = EvolutionRound(
                round_num=round_num,
                strategy=run_result.strategy,
                evaluation=run_result.evaluation,
                batch=run_result.batch,
                reflection=reflection,
                improved=improved,
                hypothesis_status=(
                    "data_quality_blocked" if data_quality_block else hypothesis_assessment.status
                ),
                hypothesis_rationale=(
                    data_quality_report.summary
                    if data_quality_block
                    else hypothesis_assessment.rationale
                ),
                hypothesis_next_step=(
                    "Remediate data source/event coverage before mutating strategy rules."
                    if data_quality_block
                    else hypothesis_assessment.next_step
                ),
                meta_insights=[
                    ins.description for ins in meta_profile.insights if ins.confidence > 0.2
                ]
                if meta_profile is not None
                else [],
                experience_lessons=_exp_lessons,
                pattern_context=_patterns,
                cross_strategy_memory=self._build_cross_strategy_memory(
                    family_id,
                    run_result.strategy,
                ),
                recommended_method=(
                    meta_profile.recommended_method if meta_profile is not None else None
                ),
                recommended_intensity=(
                    meta_profile.recommended_intensity if meta_profile is not None else None
                ),
                recommended_max_changes=(
                    meta_profile.recommended_max_changes if meta_profile is not None else None
                ),
            )
            result.rounds.append(evolution_round)

            # Record trajectory step for training data export
            _traj_changes = []
            if reflection and reflection.proposed_changes:
                _traj_changes = [
                    {
                        "change_type": ch.change_type.value,
                        "target": ch.target,
                        "from_value": ch.from_value,
                        "to_value": ch.to_value,
                    }
                    for ch in reflection.proposed_changes
                ]
            _traj_step = TrajectoryStep(
                round_num=round_num,
                strategy_id=current_id,
                strategy_version=run_result.strategy.meta.version,
                score_before=score,
                win_rate_before=run_result.evaluation.overall.win_rate,
                signal_count_before=run_result.evaluation.overall.signal_count,
                failure_patterns=reflection.failure_patterns if reflection else [],
                diagnosis=reflection.diagnosis if reflection else "",
                hypothesis=(
                    reflection.candidates[0].hypothesis.hypothesis
                    if reflection and reflection.candidates
                    else ""
                ),
                expected_outcome=(
                    reflection.candidates[0].hypothesis.expected_outcome
                    if reflection and reflection.candidates
                    else ""
                ),
                changes=_traj_changes,
                method=method.value,
                improved=improved,
                critic_verdict=_round_verdict_summary,
                playbook_used=_round_playbooks,
                llm_telemetry=_round_llm_telemetry,
            )
            if round_num == 1:
                _trajectory.set_initial_score(score)
            _trajectory.record_step(_traj_step)

            if sparse_signal_block:
                _progress(
                    "Early stop: insufficient signals after sample expansion "
                    f"({signal_count}/{min_signal_count})"
                )
                result.early_stopped = True
                result.stop_reason = (
                    "Insufficient signals after sample expansion "
                    f"({signal_count}/{min_signal_count})"
                )
                break

            if data_quality_block:
                _progress(
                    "Early stop: data quality gate blocked strategy mutation "
                    f"({data_quality_report.summary})"
                )
                result.early_stopped = True
                result.stop_reason = f"Data quality gate: {data_quality_report.summary}"
                break

            # Step 3: Safety guardrails
            if round_num >= 2 and not improved:
                # Skip stagnation check for the round right after recovery,
                # because that round re-ran the champion (same score) and
                # needs one more round to test the new mutation.
                if getattr(self, "_recovery_round", None) == round_num - 1:
                    pass  # Suppress — let the mutation attempt proceed
                elif (
                    len(result.rounds) >= 2
                    and all(not r.improved for r in result.rounds[-2:])
                    and not self._recovery_attempted
                ):
                    recovery_mode = self._recovery_mode_for_assessment(hypothesis_assessment.status)
                    _progress(
                        "  Stagnation detected — reverting to champion "
                        f"and switching to {recovery_mode.replace('_', '-')} mode"
                    )
                    current_id = best_id
                    self._recovery_attempted = True
                    self._recovery_round = round_num
                    self._recovery_mode = recovery_mode
                    continue
                elif (
                    len(result.rounds) >= 2
                    and all(not r.improved for r in result.rounds[-2:])
                    and self._recovery_attempted
                ):
                    _progress("Early stop: no improvement after recovery attempt")
                    result.early_stopped = True
                    result.stop_reason = "No improvement after recovery attempt"
                    break

            # Check overfit — but allow round 1 to be evolved (the whole
            # point of `builtin_overfit` playbook is to FIX overfit strategies).
            if round_num >= 2 and run_result.evaluation.anti_overfit.is_overfit:
                _progress("Early stop: overfitting detected")
                result.early_stopped = True
                result.stop_reason = "Overfitting detected"
                break

            # Step 4: Mutate for next round
            if round_num < rounds and reflection and reflection.proposed_changes:
                mutated = False
                screening_enabled = self._can_screen_mutation_candidates(run_result)
                screened_successes: list[_ScreenedMutationCandidate] = []
                # Try the final decision bundle first (it may have been adjusted by
                # critic, hypothesis guidance, or repeated-failure filtering), then
                # fall back to any remaining ranked candidates.
                candidates_to_try: list[CandidateExperiment | None] = [None]
                if reflection.candidates:
                    for candidate in reflection.candidates:
                        if candidate.proposed_changes == reflection.proposed_changes:
                            continue
                        candidates_to_try.append(candidate)

                for cand_idx, candidate_option in enumerate(candidates_to_try):
                    if candidate_option is not None:
                        changes_to_apply = candidate_option.proposed_changes[
                            : self.config.evolution.max_changes_per_round
                        ]
                    else:
                        changes_to_apply = reflection.proposed_changes

                    if not changes_to_apply:
                        continue

                    try:
                        new_strategy = self._mutator.mutate(
                            run_result.strategy,
                            changes_to_apply,
                            atomic=True,
                        )
                        if screening_enabled:
                            screened = self._screen_mutation_candidate(
                                run_result,
                                new_strategy,
                                changes_to_apply,
                                candidate_option,
                            )
                            screened_successes.append(screened)
                            if screened.evaluation is not None:
                                gate_reason = self._promotion_guard_reason(screened.evaluation)
                                gate_note = "promotable" if gate_reason is None else gate_reason
                                _progress(
                                    "  Candidate screen: "
                                    f"{new_strategy.meta.id} -> {screened.evaluation.confidence_score:.1%} "
                                    f"(Δ {screened.score_delta:+.1%}, "
                                    f"signals {screened.evaluation.overall.signal_count}, {gate_note})"
                                )
                            else:
                                _progress(
                                    f"  Candidate screen unavailable for {new_strategy.meta.id}; "
                                    "keeping it as a fallback option"
                                )
                            continue

                        current_id, pending_exp = self._commit_mutated_strategy(
                            run_result=run_result,
                            reflection=reflection,
                            new_strategy=new_strategy,
                            changes_to_apply=changes_to_apply,
                            candidate_option=candidate_option,
                            method=method,
                            round_num=round_num,
                            score=score,
                            progress=_progress,
                        )
                        mutated = True
                        break  # Successfully mutated — stop trying candidates
                    except MutationError as e:
                        logger.info(
                            "Mutation failed for candidate %d: %s — trying next",
                            cand_idx + 1,
                            e,
                        )
                        continue

                if screening_enabled and screened_successes:
                    chosen = self._select_screened_candidate(run_result, screened_successes)
                    current_id, pending_exp = self._commit_mutated_strategy(
                        run_result=run_result,
                        reflection=reflection,
                        new_strategy=chosen.strategy,
                        changes_to_apply=chosen.changes,
                        candidate_option=chosen.candidate,
                        method=method,
                        round_num=round_num,
                        score=score,
                        progress=_progress,
                    )
                    mutated = True

                if not mutated:
                    _progress("  All candidate mutations failed")
                    result.early_stopped = True
                    result.stop_reason = "All candidate mutations failed"
                    break

        best_score_value = best_score if best_score is not None else 0.0
        result.champion_id = best_id
        result.champion_score = best_score_value

        # Extract and save reusable patterns from the champion strategy
        if best_score_value > 0 and result.rounds:
            champion_round = next(
                (r for r in result.rounds if r.strategy.meta.id == best_id),
                None,
            )
            if champion_round is not None:
                extracted = self._pattern_library.extract_patterns_from_strategy(
                    champion_round.strategy,
                    champion_round.evaluation,
                )
                for pat in extracted:
                    self._pattern_library.save(pat)
                if extracted:
                    _progress(f"  Extracted {len(extracted)} reusable patterns from champion")

                # Track pattern usage for champion patterns only
                for pat in extracted:
                    with contextlib.suppress(Exception):
                        self._pattern_library.record_usage(
                            pat.pattern_id,
                            succeeded=True,
                        )

        # ── Auto-discover playbooks from accumulated experience ──
        with contextlib.suppress(Exception):
            discovered_playbooks = self._playbook_store.discover_from_experience(
                self._experience_store, min_occurrences=3
            )
            if discovered_playbooks:
                _progress(f"  Discovered {len(discovered_playbooks)} new research playbook(s)")

        # ── Finalize trajectory for training data export ──
        # Backfill actual outcomes BEFORE finalize() so that final_score
        # and per-step metadata are consistent.
        #
        # Each step's outcome is the *next* step's score/metrics (since that
        # reflects the backtest of the mutated strategy). The final step's
        # outcome is the best_score / last round's own metrics.
        traj_steps = _trajectory._steps
        for i, step in enumerate(traj_steps):
            if i + 1 < len(traj_steps):
                nxt = traj_steps[i + 1]
                step.score_after = nxt.score_before
                step.win_rate_after = nxt.win_rate_before
                step.signal_count_after = nxt.signal_count_before
            else:
                step.score_after = best_score_value
                # For the last step, use its own metrics (no future round)
                step.win_rate_after = step.win_rate_before
                step.signal_count_after = step.signal_count_before
            step.score_delta = step.score_after - step.score_before
            # Recompute `improved` from the actual effect of changes
            # (positive delta means the changes helped)
            step.improved = step.score_delta > 0

        result_trajectory = _trajectory.finalize()
        # Attach to result for downstream export
        result.trajectory = result_trajectory  # type: ignore[attr-defined]

        # ── Baseline comparison: param_search-only benchmark ──
        # When using LLM or hybrid, run one round of param_search on the
        # original strategy to show the incremental value of LLM reflection.
        if method != EvolutionMethod.PARAM_SEARCH and result.rounds:
            try:
                ps_reflection = self._param_search_reflection(
                    result.rounds[0].strategy,
                    result.rounds[0].evaluation,
                )
                if ps_reflection and ps_reflection.proposed_changes:
                    ps_strategy = self._mutator.mutate(
                        result.rounds[0].strategy,
                        ps_reflection.proposed_changes,
                        atomic=True,
                    )
                    # Give the baseline strategy a distinct ID so it cannot
                    # collide with real evolved versions (e.g. foo_v2 from
                    # round-2 LLM mutation).  The mutator produces foo_v2;
                    # we rename it to foo_ps_baseline.
                    base_name = result.rounds[0].strategy.meta.id.rsplit("_v", 1)[0]
                    ps_strategy.meta.id = f"{base_name}_ps_baseline"
                    ps_strategy.meta.parent_id = result.rounds[0].strategy.meta.id
                    # Save before running so RunPipeline.run() can find it
                    self.store.save(ps_strategy)
                    ps_run = await self._run_pipeline.run(
                        ps_strategy.meta.id,
                        max_symbols=max_symbols,
                        sampling_method=sampling_method,
                        date_range=date_range,
                    )
                    result.baseline_param_search_score = ps_run.evaluation.confidence_score
                    _progress(
                        f"  Baseline (param_search only): {result.baseline_param_search_score:.1%} "
                        f"vs champion (LLM): {best_score_value:.1%}"
                    )
            except Exception as e:
                logger.debug("Baseline param_search comparison skipped: %s", e)

        # Log final result
        self.research_log.log(
            "result",
            (
                f"Evolution complete: champion={best_id}, "
                f"score={best_score_value:.1%}, improvement={result.improvement:+.1%}"
            ),
            strategy_id=best_id,
            data={
                "champion_id": best_id,
                "champion_score": round(best_score_value, 4),
                "improvement": round(result.improvement, 4),
                "total_rounds": len(result.rounds),
                "early_stopped": result.early_stopped,
                "stop_reason": result.stop_reason or "",
            },
        )
        # Attach research log to result for downstream reporting
        result.research_log = self.research_log

        _progress(
            f"Evolution complete: champion={best_id} "
            f"score={best_score_value:.1%} "
            f"improvement={result.improvement:+.1%}"
        )
        return result

    def _exploration_changes(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
    ) -> list[StrategyChange]:
        """Inject structural or cross-strategy changes when normal reflection stalls."""
        changes: list[StrategyChange] = []
        entry_signals = _entry_signal_conditions(strategy)
        entry_signal_bucket = _entry_signal_bucket_name(strategy)
        existing = {
            condition.indicator
            for condition in (
                strategy.entry.triggers
                + strategy.entry.conditions
                + strategy.entry.guards
                + strategy.entry.filters
            )
        }

        if (
            evaluation.overall.signal_count < self.config.evolution.min_signal_count
            and strategy.entry.logic == "and"
            and len(entry_signals) >= 2
        ):
            changes.append(
                StrategyChange(
                    change_type=ChangeType.CHANGE_LOGIC,
                    target="entry.logic",
                    from_value="and",
                    to_value="or",
                    reason="Exploration mode: AND logic may be suppressing too many signals",
                )
            )

        if (
            evaluation.overall.signal_count < self.config.evolution.min_signal_count
            and len(entry_signals) >= 3
            and len(changes) < self.config.evolution.max_changes_per_round
        ):
            removable = entry_signals[-1].indicator
            changes.append(
                StrategyChange(
                    change_type=ChangeType.REMOVE_CONDITION,
                    target=f"entry.{entry_signal_bucket}[indicator={removable}]",
                    from_value=True,
                    to_value=None,
                    reason="Exploration mode: remove one entry condition to broaden the search space",
                )
            )

        if (
            evaluation.overall.win_rate < 0.45
            and strategy.entry.logic == "or"
            and len(entry_signals) >= 2
            and len(changes) < self.config.evolution.max_changes_per_round
        ):
            changes.append(
                StrategyChange(
                    change_type=ChangeType.CHANGE_LOGIC,
                    target="entry.logic",
                    from_value="or",
                    to_value="and",
                    reason="Exploration mode: OR logic may be admitting too many noisy signals",
                )
            )

        if len(changes) < self.config.evolution.max_changes_per_round:
            for pattern in self._pattern_library.get_best_patterns(
                category=strategy.meta.category,
                exclude_test_sources=True,
                limit=5,
                min_score=0.0,
            ):
                for condition in pattern.conditions:
                    indicator = str(condition.get("indicator", "")).strip()
                    if not indicator or indicator in existing:
                        continue
                    changes.append(
                        StrategyChange(
                            change_type=ChangeType.ADD_CONDITION,
                            target=f"entry.{entry_signal_bucket}",
                            from_value=None,
                            to_value=condition,
                            reason=(
                                "Exploration mode: inject reusable pattern from "
                                f"{pattern.source_strategy}"
                            ),
                        )
                    )
                    existing.add(indicator)
                    break
                if len(changes) >= self.config.evolution.max_changes_per_round:
                    break

        deduped: list[StrategyChange] = []
        seen: set[tuple[ChangeType, str]] = set()
        for change in changes:
            key = (change.change_type, change.target)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(change)
        return deduped[: self.config.evolution.max_changes_per_round]

    def _guided_intensity(
        self,
        *,
        default_intensity: float,
        meta_intensity: float,
        assessment_status: str,
        recovery_mode: str | None = None,
    ) -> float:
        """Blend annealing, meta-learning, and hypothesis-aware guidance."""
        intensity = (default_intensity + meta_intensity) / 2

        if assessment_status == "parameter_misaligned":
            intensity = min(intensity, 0.85)
        elif assessment_status == "execution_misaligned":
            intensity = max(intensity, 1.0)
        elif assessment_status == "thesis_misaligned":
            intensity = max(intensity, 1.65)

        if recovery_mode == "fine_tune":
            intensity = min(intensity, 0.65)
        elif recovery_mode == "exploration":
            intensity = max(intensity, 1.8)

        return intensity

    @staticmethod
    def _refine_hypothesis_assessment(
        assessment: MarketHypothesisAssessment,
        reflection: ReflectionResult | None,
    ) -> MarketHypothesisAssessment:
        """Upgrade a coarse heuristic assessment using explicit reflection evidence."""
        if reflection is None or assessment.status == "unproven_small_sample":
            return assessment

        evidence_parts = [
            reflection.diagnosis,
            reflection.reflection_summary,
            *reflection.failure_patterns,
        ]
        evidence = " ".join(part for part in evidence_parts if part).lower()
        if not evidence:
            return assessment

        thesis_tokens = (
            "core hypothesis",
            "underlying thesis",
            "market belief",
            "not supported by the data",
            "fails in its intended regime",
            "strategy fails in its intended regime",
            "thesis is not holding",
        )
        execution_tokens = (
            "entry timing",
            "execution timing",
            "next-day execution",
            "next open",
            "overnight gap risk",
            "stop-loss",
            "take-profit",
            "exit framework",
            "payoff profile",
        )

        if assessment.status != "thesis_misaligned" and any(
            token in evidence for token in thesis_tokens
        ):
            return assessment.model_copy(
                update={
                    "status": "thesis_misaligned",
                    "rationale": (
                        f"{assessment.rationale} Reflection also says the core hypothesis may no "
                        "longer be supported by the data, so this round should favor structural "
                        "rewrites over more threshold tuning."
                    ),
                    "next_step": (
                        "Favor structural rewrites, regime shifts, or factor discovery before "
                        "trying more parameter tuning."
                    ),
                }
            )

        if assessment.status == "parameter_misaligned" and any(
            token in evidence for token in execution_tokens
        ):
            return assessment.model_copy(
                update={
                    "status": "execution_misaligned",
                    "rationale": (
                        f"{assessment.rationale} Reflection also points to execution or exit "
                        "design as the nearer-term bottleneck."
                    ),
                    "next_step": (
                        "Rework execution timing, stop placement, or take-profit design before "
                        "continuing broader parameter search."
                    ),
                }
            )

        return assessment

    @staticmethod
    def _has_structural_change(changes: list[StrategyChange]) -> bool:
        """Whether a bundle changes structure rather than only thresholds."""
        structural_types = {
            ChangeType.ADD_CONDITION,
            ChangeType.REMOVE_CONDITION,
            ChangeType.CHANGE_LOGIC,
            ChangeType.CHANGE_UNIVERSE,
            ChangeType.DISCOVER_FACTOR,
        }
        return any(change.change_type in structural_types for change in changes)

    @staticmethod
    def _has_exit_change(changes: list[StrategyChange]) -> bool:
        """Whether a bundle touches exit logic."""
        return any(change.change_type == ChangeType.ADJUST_EXIT for change in changes)

    def _select_hypothesis_aligned_candidate(
        self,
        candidates: list[CandidateExperiment],
        assessment_status: str,
    ) -> CandidateExperiment | None:
        """Promote candidate bundles that match the current diagnosis lens."""
        if not candidates:
            return None
        if assessment_status == "thesis_misaligned":
            return next(
                (
                    candidate
                    for candidate in candidates
                    if self._has_structural_change(candidate.proposed_changes)
                ),
                None,
            )
        if assessment_status == "execution_misaligned":
            return next(
                (
                    candidate
                    for candidate in candidates
                    if self._has_exit_change(candidate.proposed_changes)
                ),
                None,
            )
        if assessment_status == "parameter_misaligned":
            return next(
                (
                    candidate
                    for candidate in candidates
                    if not self._has_structural_change(candidate.proposed_changes)
                ),
                None,
            )
        return None

    def _execution_recovery_changes(self, strategy: Strategy) -> list[StrategyChange]:
        """Suggest exit-focused fixes when signal quality exists but payoff does not."""
        changes: list[StrategyChange] = []

        if strategy.exit.stop_loss.type != "atr":
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.stop_loss.type",
                    from_value=strategy.exit.stop_loss.type,
                    to_value="atr",
                    reason=(
                        "Hypothesis lens: move to volatility-aware stops when entries work but "
                        "payoff quality is weak"
                    ),
                )
            )

        if (
            strategy.exit.stop_loss.type == "atr" or changes
        ) and strategy.exit.stop_loss.atr_period != 14:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.stop_loss.atr_period",
                    from_value=strategy.exit.stop_loss.atr_period,
                    to_value=14,
                    reason="Hypothesis lens: standardize ATR stop calibration before rewriting entry logic",
                )
            )

        if strategy.exit.take_profit.type != "trailing":
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.take_profit.type",
                    from_value=strategy.exit.take_profit.type,
                    to_value="trailing",
                    reason=(
                        "Hypothesis lens: entries may be directionally right, but the exit logic "
                        "is capturing payoff too mechanically"
                    ),
                )
            )

        if strategy.exit.take_profit.trail_pct is None:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.take_profit.trail_pct",
                    from_value=strategy.exit.take_profit.trail_pct,
                    to_value=0.04,
                    reason="Hypothesis lens: add a trailing-profit leash so winners can breathe before exiting",
                )
            )

        return changes[: self.config.evolution.max_changes_per_round]

    def _thesis_recovery_changes(self, strategy: Strategy) -> list[StrategyChange]:
        """Force structural exploration when the market thesis itself looks wrong."""
        changes: list[StrategyChange] = []
        entry_signals = _entry_signal_conditions(strategy)
        entry_signal_bucket = _entry_signal_bucket_name(strategy)

        if strategy.entry.logic == "and" and len(entry_signals) >= 2:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.CHANGE_LOGIC,
                    target="entry.logic",
                    from_value="and",
                    to_value="or",
                    reason=(
                        "Hypothesis lens: the current thesis may be too narrowly encoded, so "
                        "allow alternative entry paths before more threshold tuning"
                    ),
                )
            )
        elif strategy.entry.logic == "or" and len(entry_signals) >= 2:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.CHANGE_LOGIC,
                    target="entry.logic",
                    from_value="or",
                    to_value="and",
                    reason=(
                        "Hypothesis lens: the current thesis may be too broad, so require stronger "
                        "alignment before committing capital"
                    ),
                )
            )

        if len(entry_signals) >= 3:
            removable = entry_signals[-1].indicator
            changes.append(
                StrategyChange(
                    change_type=ChangeType.REMOVE_CONDITION,
                    target=f"entry.{entry_signal_bucket}[indicator={removable}]",
                    from_value=True,
                    to_value=None,
                    reason=(
                        "Hypothesis lens: remove one branch and reopen the structural search space "
                        "instead of only tuning existing thresholds"
                    ),
                )
            )

        return changes[: self.config.evolution.max_changes_per_round]

    def _parameter_recovery_changes(self, strategy: Strategy) -> list[StrategyChange]:
        """Simplify brittle strategies before trying more tuning."""
        changes: list[StrategyChange] = []
        guard_bucket_name = _entry_guard_bucket_name(strategy)
        guard_bucket = _entry_guard_conditions(strategy)
        entry_signals = _entry_signal_conditions(strategy)
        entry_signal_bucket = _entry_signal_bucket_name(strategy)
        if guard_bucket:
            removable = guard_bucket[-1].indicator
            changes.append(
                StrategyChange(
                    change_type=ChangeType.REMOVE_CONDITION,
                    target=f"entry.{guard_bucket_name}[indicator={removable}]",
                    from_value=True,
                    to_value=None,
                    reason="Hypothesis lens: simplify the most recent filter to reduce brittle overfitting",
                )
            )
        elif len(entry_signals) > 1:
            removable = entry_signals[-1].indicator
            changes.append(
                StrategyChange(
                    change_type=ChangeType.REMOVE_CONDITION,
                    target=f"entry.{entry_signal_bucket}[indicator={removable}]",
                    from_value=True,
                    to_value=None,
                    reason="Hypothesis lens: remove one gating condition before stacking more parameter tweaks",
                )
            )
        return changes

    def _merge_unique_changes(
        self,
        primary: list[StrategyChange],
        secondary: list[StrategyChange],
    ) -> list[StrategyChange]:
        """Merge two bundles without repeating the same target/type pair."""
        merged: list[StrategyChange] = []
        seen: set[tuple[ChangeType, str]] = set()
        for change in [*primary, *secondary]:
            key = (change.change_type, change.target)
            if key in seen:
                continue
            seen.add(key)
            merged.append(change)
            if len(merged) >= self.config.evolution.max_changes_per_round:
                break
        return merged

    def _apply_hypothesis_guidance(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        assessment_status: str,
        proposed_changes: list[StrategyChange],
    ) -> list[StrategyChange]:
        """Promote thesis-aware recovery actions when reflection is too conservative."""
        if assessment_status == "execution_misaligned":
            if self._has_exit_change(proposed_changes):
                return proposed_changes[: self.config.evolution.max_changes_per_round]
            guided = self._execution_recovery_changes(strategy)
            return self._merge_unique_changes(guided, proposed_changes)

        if assessment_status == "parameter_misaligned":
            if proposed_changes and not self._has_structural_change(proposed_changes):
                return proposed_changes[: self.config.evolution.max_changes_per_round]
            guided = self._parameter_recovery_changes(strategy)
            return self._merge_unique_changes(guided, proposed_changes)

        if assessment_status == "thesis_misaligned":
            guided = self._merge_unique_changes(
                self._thesis_recovery_changes(strategy),
                self._exploration_changes(strategy, evaluation),
            )
            if not proposed_changes:
                return guided
            if self._has_structural_change(proposed_changes):
                return proposed_changes[: self.config.evolution.max_changes_per_round]
            return self._merge_unique_changes(guided, proposed_changes)

        return proposed_changes[: self.config.evolution.max_changes_per_round]

    @staticmethod
    def _recovery_mode_for_assessment(assessment_status: str) -> str:
        """Choose fine-tuning vs exploration after stagnation."""
        if assessment_status == "thesis_misaligned":
            return "exploration"
        return "fine_tune"

    @staticmethod
    def _can_screen_mutation_candidates(run_result: RunResult) -> bool:
        """Return True when we can cheaply re-score candidate mutations."""
        return bool(
            run_result._engine is not None
            and run_result._data is not None
            and run_result.batch is not None
        )

    def _screen_mutation_candidate(
        self,
        run_result: RunResult,
        mutated_strategy: Strategy,
        changes: list[StrategyChange],
        candidate_option: CandidateExperiment | None,
    ) -> _ScreenedMutationCandidate:
        """Evaluate one mutated strategy on the current batch before committing it."""
        screened = _ScreenedMutationCandidate(
            strategy=mutated_strategy,
            changes=changes,
            candidate=candidate_option,
        )
        engine = run_result._engine
        data = run_result._data
        if engine is None or data is None or not isinstance(engine, BacktestEngine):
            return screened

        try:
            backtest_result = engine.run(
                mutated_strategy,
                data,
                run_result.batch,
                contexts=run_result._contexts or None,
            )
            from alphaevo.evaluator.metrics import Evaluator as _Eval

            evaluation = _Eval().evaluate(
                backtest_result,
                mutated_strategy,
                market_data=data,
                contexts=run_result._contexts or None,
                backtest_config=self.config.backtest,
            )
            screened.evaluation = evaluation
            screened.score_delta = (
                evaluation.confidence_score - run_result.evaluation.confidence_score
            )
            screened.signal_delta = (
                evaluation.overall.signal_count - run_result.evaluation.overall.signal_count
            )
        except Exception as exc:
            logger.debug(
                "Candidate screening failed for %s: %s",
                mutated_strategy.meta.id,
                exc,
                exc_info=True,
            )
        return screened

    def _select_screened_candidate(
        self,
        run_result: RunResult,
        candidates: list[_ScreenedMutationCandidate],
    ) -> _ScreenedMutationCandidate:
        """Choose the most promising screened candidate for the next round."""
        min_signals = self.config.evolution.min_signal_count
        base_signals = run_result.evaluation.overall.signal_count

        def _rank(candidate: _ScreenedMutationCandidate) -> tuple[float, ...]:
            if candidate.evaluation is None:
                priority = candidate.candidate.priority_score if candidate.candidate else 0.0
                return (-1.0, priority)

            evaluation = candidate.evaluation
            adequate_signals = evaluation.overall.signal_count >= min_signals
            not_overfit = not evaluation.anti_overfit.is_overfit
            signal_retention = evaluation.overall.signal_count / max(1, base_signals)
            if base_signals < min_signals:
                return (
                    1.0 if adequate_signals else 0.0,
                    1.0 if candidate.signal_delta > 0 else 0.0,
                    float(evaluation.overall.signal_count),
                    1.0 if not_overfit else 0.0,
                    candidate.score_delta,
                    evaluation.confidence_score,
                )
            return (
                1.0 if adequate_signals else 0.0,
                1.0 if not_overfit else 0.0,
                1.0 if signal_retention >= 0.70 else 0.0,
                signal_retention,
                candidate.score_delta,
                evaluation.confidence_score,
                float(evaluation.overall.signal_count),
            )

        return max(candidates, key=_rank)

    def _promotion_guard_reason(self, evaluation: EvaluationReport) -> str | None:
        """Return a user-facing reason when a version should not become champion."""
        min_signal_count = self.config.evolution.min_signal_count
        if evaluation.overall.signal_count < min_signal_count:
            return f"Sparse-signal result ({evaluation.overall.signal_count}/{min_signal_count})"
        if evaluation.anti_overfit.is_overfit:
            return f"Overfit detected (train_val_gap={evaluation.anti_overfit.train_val_gap:.1%})"
        if evaluation.overall.avg_return <= 0 and evaluation.overall.total_return <= 0:
            return (
                "Non-positive trade payoff "
                f"(avg_return={evaluation.overall.avg_return:.2%}, "
                f"total_return={evaluation.overall.total_return:.2%})"
            )
        return None

    def _commit_mutated_strategy(
        self,
        *,
        run_result: RunResult,
        reflection: ReflectionResult,
        new_strategy: Strategy,
        changes_to_apply: list[StrategyChange],
        candidate_option: CandidateExperiment | None,
        method: EvolutionMethod,
        round_num: int,
        score: float,
        progress: Callable[[str], None],
    ) -> tuple[str, list[tuple[StrategyChange, float, str, int, str, str, str]]]:
        """Persist the selected mutation and queue its experience outcome tracking."""
        self.store.save(new_strategy)
        current_id = new_strategy.meta.id
        progress(f"  Created {current_id} (v{new_strategy.meta.version})")

        hyp = ""
        source = "heuristic"
        if candidate_option is not None:
            hyp = candidate_option.hypothesis.hypothesis
            source = "llm"
        elif method == EvolutionMethod.PARAM_SEARCH:
            source = "param_search"

        regime = ""
        if run_result.batch.market_regimes:
            regime = run_result.batch.market_regimes[0].value

        reflection.proposed_changes = changes_to_apply
        pending_exp = [
            (change, score, current_id, round_num, hyp, source, regime)
            for change in changes_to_apply
        ]
        return current_id, pending_exp

    def _do_reflection(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        method: EvolutionMethod,
        *,
        history_text: str = "",
        experience_text: str = "",
        meta_text: str = "",
        research_summary: str = "",
        intensity: float = 1.0,
    ) -> ReflectionResult | None:
        """Perform reflection based on the chosen method."""
        if method == EvolutionMethod.PARAM_SEARCH:
            return self._param_search_reflection(strategy, evaluation)

        # LLM or HYBRID: try LLM first
        try:
            analyzer = self._ensure_analyzer()
            return analyzer.reflect(
                strategy,
                evaluation,
                history_text=history_text,
                experience_text=experience_text,
                meta_text=meta_text,
                research_summary=research_summary,
                intensity=intensity,
            )
        except LLMNotAvailableError:
            if method == EvolutionMethod.LLM:
                raise
            # HYBRID fallback to heuristic
            logger.info("LLM not available, falling back to heuristics")
            return ReflectionAnalyzer.heuristic_only(
                max_changes=self.config.evolution.max_changes_per_round,
            ).reflect(
                strategy,
                evaluation,
                intensity=intensity,
            )

    async def _try_factor_discovery(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        data: dict[str, pd.DataFrame],
    ) -> list[StrategyChange]:
        """Attempt to discover new indicators via AlphaFactory.

        Returns a list of DISCOVER_FACTOR changes that can be applied by the mutator.
        Called when conventional reflection yields no actionable changes.

        Falls back to heuristic indicator suggestion when LLM is unavailable.
        """
        try:
            factory = self._ensure_alpha_factory()
        except LLMNotAvailableError:
            logger.info("Factor discovery: LLM not available, trying heuristic fallback")
            return self._heuristic_indicator_suggestion(strategy, evaluation)

        # Build a context string from the evaluation for the LLM
        context = (
            f"Strategy '{strategy.meta.name}' has win_rate={evaluation.overall.win_rate:.1%}, "
            f"avg_return={evaluation.overall.avg_return:.2%}, "
            f"signals={evaluation.overall.signal_count}. "
            f"Existing indicators: {_entry_indicator_names(strategy)}. "
            f"Goal: discover a new indicator that improves signal selectivity."
        )

        # Prepare combined OHLCV data and forward returns for validation
        combined_frames = []
        for sym, df in list(data.items())[:8]:  # Keep the validation basket modest but useful
            if len(df) < 30:
                continue
            frame = df.copy()
            frame["_symbol"] = sym
            combined_frames.append(frame)

        if not combined_frames:
            return self._heuristic_indicator_suggestion(strategy, evaluation)

        import pandas as pd

        combined = pd.concat(combined_frames, ignore_index=True)
        sort_cols = ["_symbol"]
        if "date" in combined.columns:
            sort_cols.append("date")
        combined = combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        # Forward returns: next-day close / close - 1
        if "close" in combined.columns:
            combined["_fwd_return"] = (
                combined.groupby("_symbol", sort=False)["close"].shift(-1) / combined["close"] - 1
            )
            combined["_fwd_return"] = combined["_fwd_return"].fillna(0)
            fwd_returns = combined["_fwd_return"]
        else:
            return self._heuristic_indicator_suggestion(strategy, evaluation)

        try:
            result = await factory.discover(
                context=context,
                ohlcv_data=combined,
                forward_returns=fwd_returns,
                max_candidates=2,
                dates=combined["date"] if "date" in combined.columns else None,
                register=True,
            )
        except Exception as e:
            logger.warning("Factor discovery failed: %s — trying heuristic fallback", e)
            return self._heuristic_indicator_suggestion(strategy, evaluation)

        changes: list[StrategyChange] = []
        for factor_name in result.registered:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.DISCOVER_FACTOR,
                    target=f"entry.{_entry_guard_bucket_name(strategy)}",
                    from_value=None,
                    to_value={
                        "indicator": factor_name,
                        "op": ">",
                        "value": 0.0,
                    },
                    reason=f"AlphaFactory discovered new factor '{factor_name}'",
                )
            )

        # If LLM discovered nothing useful, fall back to heuristics
        if not changes:
            return self._heuristic_indicator_suggestion(strategy, evaluation)
        return changes

    def _heuristic_indicator_suggestion(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
    ) -> list[StrategyChange]:
        """Suggest adding an existing registered indicator not yet in the strategy.

        Heuristic fallback when LLM-based factor discovery is unavailable.
        Picks indicators that are complementary to the current set based on
        the strategy's weakness (low win rate → add filter, few signals → skip).
        """
        from alphaevo.backtest.indicators import IndicatorRegistry

        # Don't add conditions if the problem is too few signals
        if evaluation.overall.signal_count < 10:
            return []

        existing = set(_entry_indicator_names(strategy))

        # Candidate indicators by weakness type, with sensible defaults
        candidates_for_low_wr: list[tuple[str, str, float | bool]] = [
            ("rsi_14", "<", 70),
            ("volatility_20d", "<", 0.04),
            ("momentum_10d", ">", 0.0),
            ("ma20_slope", ">", 0.0),
        ]
        candidates_for_low_pl: list[tuple[str, str, float | bool]] = [
            ("atr", ">", 0.5),
            ("volume_ratio_1d_5d", ">", 1.2),
        ]

        m = evaluation.overall
        if m.win_rate < 0.5:
            pool = candidates_for_low_wr
        elif m.profit_loss_ratio < 1.5:
            pool = candidates_for_low_pl
        else:
            return []  # Strategy is OK — don't add complexity

        for indicator, op, value in pool:
            if indicator in existing:
                continue
            if not IndicatorRegistry.is_registered(indicator):
                continue
            return [
                StrategyChange(
                    change_type=ChangeType.ADD_CONDITION,
                    target=f"entry.{_entry_guard_bucket_name(strategy)}",
                    from_value=None,
                    to_value={"indicator": indicator, "op": op, "value": value},
                    reason=f"Heuristic: add '{indicator}' filter to improve signal quality",
                )
            ]
        return []

    def _record_pending_experience(
        self,
        pending_exp: list[tuple[StrategyChange, float, str, int, str, str, str]],
        family_id: str,
        score_after: float,
        worked: bool | None,
        failure_reason: str = "",
    ) -> None:
        """Persist outcomes for changes applied at a previous round.

        Each entry in pending_exp is:
        (change, score_before, strategy_id, round_num, hypothesis, source, regime)
        """
        records: list[ExperienceRecord] = []
        change_count = max(1, len(pending_exp))
        for change, score_before, strategy_id, round_num, hypothesis, source, regime in pending_exp:
            delta = (score_after - score_before) / change_count
            change_worked = worked if worked is not None else (delta > 0)
            lesson = ""
            if failure_reason:
                lesson = (
                    f"{change.change_type.value} on {change.target} failed to "
                    f"execute: {failure_reason}"
                )
            elif change_worked:
                lesson = (
                    f"{change.change_type.value} on {change.target} improved score "
                    f"by {delta:+.1%}: {change.reason}"
                )
            else:
                lesson = (
                    f"{change.change_type.value} on {change.target} did NOT improve "
                    f"score ({delta:+.1%}): consider reverting or adjusting further"
                )
            records.append(
                ExperienceRecord(
                    strategy_family=family_id,
                    strategy_id=strategy_id,
                    round_num=round_num,
                    change_type=change.change_type,
                    target=change.target,
                    from_value=change.from_value,
                    to_value=change.to_value,
                    reason=change.reason,
                    score_before=score_before,
                    score_after=score_after,
                    score_delta=delta,
                    worked=change_worked,
                    lesson=lesson,
                    hypothesis=hypothesis,
                    action_type=source,
                    source=source,
                    regime=regime,
                )
            )
        if records:
            try:
                self._experience_store.record_batch(records)
                logger.debug("Recorded %d experience entries", len(records))
            except Exception as e:
                logger.warning("Failed to record experience: %s", e)

    def _build_cross_strategy_memory(
        self,
        family_id: str,
        strategy: Strategy,
        *,
        limit: int = 4,
    ) -> list[str]:
        """Summarize reusable cross-strategy memory for user-facing reports."""
        memory: list[str] = []

        with contextlib.suppress(Exception):
            global_records = [
                record
                for record in self._experience_store.query(
                    ExperienceQuery(limit=120, exclude_test_sources=True)
                )
                if record.strategy_family != family_id
            ]
            grouped: dict[tuple[str, str], _CrossStrategyBucket] = {}
            for record in global_records:
                key = (record.change_type.value, record.target)
                bucket = grouped.setdefault(
                    key,
                    {
                        "total": 0,
                        "succeeded": 0,
                        "families": set(),
                        "strategies": [],
                    },
                )
                bucket["total"] += 1
                if record.worked:
                    bucket["succeeded"] += 1
                bucket["families"].add(record.strategy_family)
                if record.strategy_id not in bucket["strategies"]:
                    bucket["strategies"].append(record.strategy_id)

            ranked_memory = sorted(
                grouped.items(),
                key=lambda item: (
                    item[1]["succeeded"],
                    item[1]["total"],
                ),
                reverse=True,
            )
            for (change_type, target), bucket in ranked_memory:
                total = bucket["total"]
                succeeded = bucket["succeeded"]
                if total == 0:
                    continue
                success_rate = succeeded / total
                if success_rate <= 0:
                    continue
                family_count = len(bucket["families"])
                strategies = bucket["strategies"]
                example_sources = ", ".join(strategies[:2]) if strategies else "earlier strategies"
                memory.append(
                    f"Across {total} prior cross-strategy experiments from "
                    f"{family_count} other family/families, "
                    f"`{change_type}` on `{target}` succeeded {success_rate:.0%} "
                    f"({succeeded}/{total}); recently seen in `{example_sources}`."
                )
                if len(memory) >= max(1, limit // 2):
                    break

        with contextlib.suppress(Exception):
            patterns = self._pattern_library.get_best_patterns(
                category=strategy.meta.category,
                exclude_test_sources=True,
                limit=8,
                min_score=0.0,
            )
            for pattern in patterns:
                source_strategy = pattern.source_strategy.strip()
                source_family = (
                    source_strategy.rsplit("_v", 1)[0]
                    if "_v" in source_strategy
                    else source_strategy
                )
                if not source_strategy or source_family == family_id:
                    continue
                usage = (
                    f"used {pattern.times_used}x with {pattern.success_rate:.0%} success"
                    if pattern.times_used > 0
                    else f"source score {pattern.confidence_score:.0%}, "
                    f"win rate {pattern.win_rate:.0%}"
                )
                memory.append(
                    f"Borrowed [{pattern.pattern_type}] from `{source_strategy}`: "
                    f"{pattern.description}; {usage}."
                )
                if len(memory) >= limit:
                    break

        return memory[:limit]

    def _filter_failed_repeated_changes(
        self,
        family_id: str,
        changes: list[StrategyChange],
    ) -> list[StrategyChange]:
        """Drop changes that previously failed for the same strategy family."""
        if not changes:
            return []

        try:
            failed_signatures = self._experience_store.get_failed_signatures(
                family_id,
                min_failures=self._failed_change_threshold,
                limit=500,
                exclude_test_sources=True,
            )
        except Exception as e:
            logger.warning("Failed to load experience for filtering: %s", e)
            return changes

        if not failed_signatures:
            return changes

        kept: list[StrategyChange] = []
        for change in changes:
            sig = self._change_signature(
                change.change_type,
                change.target,
                change.to_value,
            )
            if sig in failed_signatures:
                logger.info(
                    "Skipping repeated failed change for %s: %s %s -> %s",
                    family_id,
                    change.change_type.value,
                    change.target,
                    change.to_value,
                )
                continue
            kept.append(change)
        return kept

    @staticmethod
    def _change_signature(
        change_type: ChangeType,
        target: str,
        to_value: object,
    ) -> tuple[str, str, str]:
        """Create a stable signature for deduping change attempts.

        Uses fuzzy matching: numeric values are rounded to 2 decimal places
        and targets are normalized to category level, so that similar changes
        (e.g., stop_loss 4%→3% vs 4%→2.9%) share the same signature.
        """
        # Normalize to_value: round floats for fuzzy matching
        normalized_value = to_value
        if isinstance(to_value, float):
            normalized_value = round(to_value, 2)
        elif isinstance(to_value, str):
            with contextlib.suppress(ValueError, TypeError):
                normalized_value = round(float(to_value), 2)

        return (
            change_type.value,
            target.strip(),
            json.dumps(normalized_value, sort_keys=True, default=str),
        )

    def _param_search_reflection(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
    ) -> ReflectionResult | None:
        """Simple param search: try adjusting tunable params."""
        if not strategy.params.tunable:
            return None

        changes: list[StrategyChange] = []
        for param in strategy.params.tunable[: self.config.evolution.max_changes_per_round]:
            lo, hi = param.range
            current = resolve_tunable_target(strategy, param.target)
            is_exit_target = param.target.startswith("exit.")

            if current is None:
                continue

            # Simple heuristic: if win rate low, tighten; if too few signals, loosen
            if isinstance(current, (int, float)):
                is_period_tunable = is_period_tunable_target(param.target)
                if evaluation.overall.win_rate < 0.5:
                    new_val = (
                        tune_period_value(
                            current,
                            param.step,
                            param.target,
                            tighten=True,
                            lo=lo,
                            hi=hi,
                        )
                        if is_period_tunable
                        else current + param.step
                    )
                    change_type = (
                        ChangeType.ADJUST_EXIT if is_exit_target else ChangeType.TIGHTEN_FILTER
                    )
                else:
                    new_val = (
                        tune_period_value(
                            current,
                            param.step,
                            param.target,
                            tighten=False,
                            lo=lo,
                            hi=hi,
                        )
                        if is_period_tunable
                        else current - param.step
                    )
                    change_type = (
                        ChangeType.ADJUST_EXIT if is_exit_target else ChangeType.LOOSEN_FILTER
                    )

                new_val = max(lo, min(hi, new_val))
                if is_period_tunable:
                    if is_integer_tunable_target(param.target):
                        new_val = int(round(new_val))
                    else:
                        new_val = round(float(new_val), 4)
                if new_val != current:
                    changes.append(
                        StrategyChange(
                            change_type=change_type,
                            target=param.target,
                            from_value=current,
                            to_value=new_val,
                            reason=f"Param search: adjust {param.label or param.target}",
                        )
                    )

        if not changes:
            return None

        return ReflectionResult(
            strategy_id=strategy.meta.id,
            evaluation_id=evaluation.evaluation_id,
            failure_patterns=["Param search optimization"],
            proposed_changes=changes,
            reflection_summary="Parameter grid search adjustment",
        )


def _entry_indicator_names(strategy: Strategy) -> list[str]:
    """Return unique entry indicator names across new and legacy buckets."""
    names: list[str] = []
    for condition in (
        strategy.entry.triggers
        + strategy.entry.conditions
        + strategy.entry.guards
        + strategy.entry.filters
    ):
        if condition.indicator not in names:
            names.append(condition.indicator)
    return names


def _entry_signal_bucket_name(strategy: Strategy) -> str:
    """Return the preferred entry signal bucket name for this strategy."""
    return "triggers" if strategy.entry.triggers else "conditions"


def _entry_signal_conditions(strategy: Strategy) -> list[StrategyCondition]:
    """Return the active buy-signal bucket, preserving legacy compatibility."""
    return strategy.entry.triggers if strategy.entry.triggers else strategy.entry.conditions


def _entry_guard_bucket_name(strategy: Strategy) -> str:
    """Return the preferred hard-filter bucket name for this strategy."""
    return "guards" if strategy.entry.guards or strategy.entry.triggers else "filters"


def _entry_guard_conditions(strategy: Strategy) -> list[StrategyCondition]:
    """Return the active hard-filter bucket, preserving legacy compatibility."""
    return (
        strategy.entry.guards
        if strategy.entry.guards or strategy.entry.triggers
        else strategy.entry.filters
    )
