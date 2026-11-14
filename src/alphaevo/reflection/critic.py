"""Self-critique module — validates proposed changes before mutation.

Inspired by:
- Self-Refine (CMU): Iterative critique → refine loop
- Reflexion (Shinn et al.): Verbal reinforcement from self-reflection

Before applying mutations, the critic checks:
1. Consistency: proposed changes don't contradict each other
2. History: changes haven't failed before in this family
3. Feasibility: target indicators exist and values are in range
4. Anti-regression: changes are unlikely to degrade key metrics
5. Complexity: changes don't push complexity beyond limits
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.models.enums import ChangeType
from alphaevo.models.strategy import StopLossConfig, TakeProfitConfig
from alphaevo.reflection.experience import ExperienceQuery

if TYPE_CHECKING:
    from alphaevo.models.execution import (
        CandidateExperiment,
        EvaluationReport,
        ReflectionResult,
        StrategyChange,
    )
    from alphaevo.models.strategy import Strategy
    from alphaevo.reflection.experience import ExperienceStore

logger = logging.getLogger(__name__)

_SPARSE_SIGNAL_COUNT = 30
_VERY_LOW_SIGNAL_COUNT = 10


class CritiqueVerdict:
    """Result of self-critique analysis."""

    def __init__(self) -> None:
        self.approved: list[StrategyChange] = []
        self.rejected: list[tuple[StrategyChange, str]] = []
        self.warnings: list[str] = []

    @property
    def approval_rate(self) -> float:
        total = len(self.approved) + len(self.rejected)
        return len(self.approved) / total if total > 0 else 1.0


class SelfCritic:
    """Validates proposed strategy changes before mutation.

    Acts as a quality gate between the ReflectionAnalyzer and StrategyMutator.
    Filters out changes that are likely to fail or regress performance.
    """

    def __init__(
        self,
        experience_store: ExperienceStore | None = None,
        complexity_limit: int = 8,
        scars: list[Any] | None = None,
    ) -> None:
        self._experience_store = experience_store
        self._complexity_limit = complexity_limit
        self._scars: list[Any] = list(scars or [])

    def set_scars(self, scars: list[Any]) -> None:
        """Update active executable pinned scars checked on every mutation proposal."""
        self._scars = list(scars)

    def critique(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        reflection: ReflectionResult,
    ) -> CritiqueVerdict:
        """Analyze and filter proposed changes.

        Returns a CritiqueVerdict with approved changes and rejected ones with reasons.
        """
        verdict = CritiqueVerdict()

        if not reflection.proposed_changes:
            return verdict

        exit_bundle_reasons = self._check_exit_bundle(strategy, reflection.proposed_changes)
        for change in reflection.proposed_changes:
            reasons = self._check_change(strategy, evaluation, change, reflection.proposed_changes)
            if exit_bundle_reasons and change.change_type == ChangeType.ADJUST_EXIT:
                reasons.extend(exit_bundle_reasons)
            if reasons:
                verdict.rejected.append((change, "; ".join(reasons)))
                logger.info(
                    "Critic rejected %s on %s: %s", change.change_type.value, change.target, reasons
                )
            else:
                verdict.approved.append(change)

        # Cross-change consistency checks
        self._check_consistency(verdict, evaluation)

        return verdict

    def _check_change(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        change: StrategyChange,
        all_changes: list[StrategyChange],
    ) -> list[str]:
        """Return list of rejection reasons (empty = approved)."""
        reasons: list[str] = []
        signal_count = evaluation.overall.signal_count
        sparse_signals = signal_count < _SPARSE_SIGNAL_COUNT
        very_low_signals = signal_count < _VERY_LOW_SIGNAL_COUNT

        # 1. Check indicator exists for ADD_CONDITION
        if change.change_type == ChangeType.ADD_CONDITION:
            if isinstance(change.to_value, dict):
                ind = change.to_value.get("indicator", "")
                if ind and not IndicatorRegistry.is_registered(ind):
                    reasons.append(f"Unknown indicator: {ind}")
            else:
                reasons.append("ADD_CONDITION requires dict to_value")

        if change.change_type == ChangeType.ADD_CONDITION and very_low_signals:
            reasons.append(
                "Adding conditions when signal_count is very low (<10) "
                "would likely make a sparse strategy even harder to trigger"
            )

        # 2. Check complexity limit for ADD_CONDITION
        if change.change_type == ChangeType.ADD_CONDITION:
            n_current = (
                len(strategy.entry.triggers)
                + len(strategy.entry.conditions)
                + len(strategy.entry.guards)
                + len(strategy.entry.filters)
            )
            n_adding = sum(1 for c in all_changes if c.change_type == ChangeType.ADD_CONDITION)
            n_removing = sum(1 for c in all_changes if c.change_type == ChangeType.REMOVE_CONDITION)
            projected = n_current + n_adding - n_removing
            if projected > self._complexity_limit:
                reasons.append(
                    f"Would exceed complexity limit ({projected} > {self._complexity_limit})"
                )

        # 3. Check value ranges for param adjustments
        if (
            change.change_type in (ChangeType.TIGHTEN_FILTER, ChangeType.LOOSEN_FILTER)
            and change.to_value is not None
            and strategy.params.tunable
        ):
            for param in strategy.params.tunable:
                if param.target == change.target:
                    lo, hi = param.range
                    try:
                        val = float(change.to_value)
                        if val < lo or val > hi:
                            reasons.append(f"Value {val} outside tunable range [{lo}, {hi}]")
                    except (TypeError, ValueError):
                        pass

        # 4. Check for contradictory signal count / win rate adjustments
        if (
            change.change_type == ChangeType.REMOVE_CONDITION
            and evaluation.overall.win_rate < 0.35
            and not sparse_signals
        ):
            # Removing conditions when win rate is already low = risky
            reasons.append(
                "Removing conditions when win_rate is already low (<35%) "
                "would likely produce more losing trades"
            )

        # 5. Stop loss sanity checks
        if (
            change.change_type == ChangeType.ADJUST_EXIT
            and "stop_loss.value" in change.target
            and change.to_value is not None
        ):
            try:
                sl_val = float(change.to_value)
                if sl_val <= 0.005:
                    reasons.append("Stop loss too tight (<0.5%) — will exit on noise")
                if sl_val > 0.20:
                    reasons.append("Stop loss too wide (>20%) — excessive risk per trade")
            except (TypeError, ValueError):
                pass

        if (
            change.change_type == ChangeType.ADJUST_EXIT
            and "stop_loss.multiplier" in change.target
            and change.to_value is not None
        ):
            try:
                multiplier = float(change.to_value)
                if multiplier <= 0:
                    reasons.append("ATR stop loss multiplier must be > 0")
                if multiplier > 10:
                    reasons.append("ATR stop loss multiplier is implausibly large (>10)")
            except (TypeError, ValueError):
                pass

        # 6. Check experience store for repeated failures
        if self._experience_store is not None:
            try:
                failed = self._experience_store.get_failed_signatures(
                    strategy.meta.family_id,
                    min_failures=2,
                    limit=200,
                    exclude_test_sources=True,
                )
                sig = (
                    change.change_type.value,
                    change.target.strip(),
                    json.dumps(change.to_value, sort_keys=True, default=str),
                )
                if sig in failed:
                    reasons.append("This exact change has failed 2+ times before")
            except Exception as e:
                logger.debug("Failed signature check error: %s", e)

        # 7. Avoid immediately undoing a recently successful change
        if self._experience_store is not None:
            try:
                recent_successes = self._experience_store.query(
                    ExperienceQuery(
                        strategy_family=strategy.meta.family_id,
                        only_worked=True,
                        limit=10,
                        exclude_test_sources=True,
                    )
                )
                for rec in recent_successes:
                    if rec.target.strip() != change.target.strip():
                        continue
                    if rec.change_type != change.change_type:
                        continue
                    if rec.to_value != change.from_value:
                        continue
                    if rec.from_value != change.to_value:
                        continue
                    if rec.score_delta <= 0:
                        continue
                    reasons.append(
                        "This change directly reverts a recently successful change "
                        f"(round {rec.round_num}, delta {rec.score_delta:+.1%})"
                    )
                    break
            except Exception as e:
                logger.debug("Recent success reversal check error: %s", e)

        # 8. Check executable pinned scars (Meta-RSI guardrails)
        for scar in self._scars:
            check_fn = getattr(scar, "matches", None)
            if callable(check_fn) and check_fn(strategy.meta.family_id, change):
                reasons.append(f"Blocked by pinned scar [{scar.scar_id}]: {scar.failure_mechanism}")

        return reasons

    def _check_consistency(self, verdict: CritiqueVerdict, evaluation: EvaluationReport) -> None:
        """Check for logical inconsistencies among approved changes."""
        # Check for tighten + loosen on same target
        tighten_targets = {
            c.target for c in verdict.approved if c.change_type == ChangeType.TIGHTEN_FILTER
        }
        loosen_targets = {
            c.target for c in verdict.approved if c.change_type == ChangeType.LOOSEN_FILTER
        }
        conflicting = tighten_targets & loosen_targets
        if conflicting:
            # Dynamic resolution: high win_rate → prefer loosen (more signals);
            # low win_rate → prefer tighten (higher quality).
            # But when signals are sparse, prioritize getting enough trades
            # to evaluate the strategy at all.
            win_rate = evaluation.overall.win_rate
            if evaluation.overall.signal_count < _SPARSE_SIGNAL_COUNT or win_rate >= 0.55:
                drop, label = ChangeType.TIGHTEN_FILTER, "loosen"
            else:
                drop, label = ChangeType.LOOSEN_FILTER, "tighten"
            verdict.warnings.append(
                f"Contradictory changes on: {conflicting} — keeping {label} "
                f"(win_rate={win_rate:.0%})"
            )
            verdict.approved = [
                c
                for c in verdict.approved
                if not (c.change_type == drop and c.target in conflicting)
            ]

        # Check for adding and removing same indicator
        add_indicators = set()
        remove_indicators = set()
        for c in verdict.approved:
            if c.change_type == ChangeType.ADD_CONDITION and isinstance(c.to_value, dict):
                add_indicators.add(c.to_value.get("indicator", ""))
            elif c.change_type == ChangeType.REMOVE_CONDITION:
                ind = c.target.split("indicator=")[-1].split("]")[0]
                remove_indicators.add(ind)

        conflicting_inds = add_indicators & remove_indicators
        if conflicting_inds:
            verdict.warnings.append(f"Adding and removing same indicator: {conflicting_inds}")

    def rank_candidates(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        candidates: list[CandidateExperiment],
    ) -> list[CandidateExperiment]:
        """Rank and filter candidate experiments using rule-based scoring.

        Applies hard constraints (indicator validity, complexity) to remove
        invalid candidates, then scores and sorts the valid ones by:
        - LLM-assigned priority_score (if available)
        - Number of valid changes after filtering
        - Hypothesis confidence
        - Novelty bonus (penalize if experience store shows repeated failures)

        Returns candidates sorted by adjusted score (best first).
        """
        if not candidates:
            return []

        scored: list[tuple[float, CandidateExperiment]] = []

        for candidate in candidates:
            exit_bundle_reasons = self._check_exit_bundle(strategy, candidate.proposed_changes)
            # Apply rule-based checks to each change
            valid_changes: list[StrategyChange] = []
            rejected_count = 0
            for change in candidate.proposed_changes:
                reasons = self._check_change(
                    strategy,
                    evaluation,
                    change,
                    candidate.proposed_changes,
                )
                if exit_bundle_reasons and change.change_type == ChangeType.ADJUST_EXIT:
                    reasons.extend(exit_bundle_reasons)
                if reasons:
                    logger.debug(
                        "Candidate ranking: rejected change %s — %s",
                        change.target,
                        reasons,
                    )
                    rejected_count += 1
                else:
                    valid_changes.append(change)

            # Multi-step hypotheses are treated as bundles: partial approval
            # tends to break the LLM's intended experiment design.
            if len(candidate.proposed_changes) > 1 and rejected_count > 0:
                logger.debug(
                    "Candidate ranking: dropping bundled candidate with %d/%d valid changes",
                    len(valid_changes),
                    len(candidate.proposed_changes),
                )
                continue

            if not valid_changes:
                # All changes were invalid — skip this candidate
                continue

            # Score components
            base_score = candidate.priority_score  # LLM-assigned
            hypothesis_bonus = candidate.hypothesis.confidence * 0.1
            validity_ratio = len(valid_changes) / max(1, len(candidate.proposed_changes))
            validity_bonus = validity_ratio * 0.2

            # Novelty penalty: penalize if many changes overlap with past failures
            novelty_penalty = 0.0
            if self._experience_store is not None:
                try:
                    failed_sigs = self._experience_store.get_failed_signatures(
                        strategy.meta.family_id,
                        min_failures=2,
                        limit=200,
                        exclude_test_sources=True,
                    )
                    overlap = sum(
                        1
                        for c in valid_changes
                        if (
                            c.change_type.value,
                            c.target.strip(),
                            json.dumps(c.to_value, sort_keys=True, default=str),
                        )
                        in failed_sigs
                    )
                    novelty_penalty = overlap * 0.15
                except Exception:
                    pass

            final_score = base_score + hypothesis_bonus + validity_bonus - novelty_penalty
            # Update candidate in-place
            candidate.proposed_changes = valid_changes
            candidate.priority_score = round(max(0.0, min(1.0, final_score)), 3)
            scored.append((final_score, candidate))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def _check_exit_bundle(
        self,
        strategy: Strategy,
        changes: list[StrategyChange],
    ) -> list[str]:
        """Validate structural exit changes as one coherent bundle."""
        exit_changes = [c for c in changes if c.change_type == ChangeType.ADJUST_EXIT]
        if not exit_changes:
            return []

        from alphaevo.reflection.mutator import MutationError, StrategyMutator

        mutator = StrategyMutator(
            max_changes=max(len(exit_changes), 1),
            complexity_limit=max(
                self._complexity_limit,
                len(strategy.entry.triggers)
                + len(strategy.entry.conditions)
                + len(strategy.entry.guards)
                + len(strategy.entry.filters),
            ),
        )
        try:
            simulated = mutator.mutate(strategy, exit_changes, atomic=True)
        except MutationError as e:
            return [f"Invalid exit bundle: {e}"]

        reasons: list[str] = []
        reasons.extend(self._validate_stop_loss(simulated.exit.stop_loss))
        reasons.extend(self._validate_take_profit(simulated.exit.take_profit))
        if simulated.exit.max_holding_days <= 0:
            reasons.append("max_holding_days must be > 0")
        return reasons

    @staticmethod
    def _validate_stop_loss(stop_loss: StopLossConfig) -> list[str]:
        """Semantic validation for stop-loss configs after bundled mutation."""
        reasons: list[str] = []
        sl_type = (stop_loss.type or "").strip()

        if sl_type == "pct":
            if stop_loss.value is None or stop_loss.value <= 0:
                reasons.append("pct stop loss requires a positive value")
        elif sl_type == "atr":
            if stop_loss.multiplier is not None and stop_loss.multiplier <= 0:
                reasons.append("ATR stop loss multiplier must be > 0")
            if stop_loss.atr_period is not None and stop_loss.atr_period < 1:
                reasons.append("ATR stop loss period must be >= 1")
        elif sl_type == "pct_from_low":
            if stop_loss.value is None or stop_loss.value <= 0:
                reasons.append("pct_from_low stop loss requires a positive value")
        elif sl_type == "price_level":
            if stop_loss.value is None and not stop_loss.reference:
                reasons.append("price_level stop loss requires value or reference")
        elif sl_type == "composite" and not stop_loss.conditions:
            reasons.append("composite stop loss requires conditions")

        return reasons

    @staticmethod
    def _validate_take_profit(take_profit: TakeProfitConfig) -> list[str]:
        """Semantic validation for take-profit configs after bundled mutation."""
        reasons: list[str] = []
        tp_type = (take_profit.type or "").strip()

        if tp_type in {"rr", "pct"}:
            if take_profit.value is None or take_profit.value <= 0:
                reasons.append(f"{tp_type} take profit requires a positive value")
        elif tp_type == "target_ma":
            if not take_profit.target:
                reasons.append("target_ma take profit requires a target moving average")
        elif tp_type == "trailing":
            if take_profit.trigger_pct is not None and take_profit.trigger_pct <= 0:
                reasons.append("trailing take profit trigger_pct must be > 0")
            if take_profit.trail_pct is not None and take_profit.trail_pct <= 0:
                reasons.append("trailing take profit trail_pct must be > 0")

        return reasons
