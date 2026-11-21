"""Semantic DSL Genetic Crossover Breeder for balanced L1+L2 population evolution."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import StrategyChange
from alphaevo.models.strategy import Strategy, StrategyCondition

CrossoverOperator = Literal["exit_graft", "filter_graft", "trigger_blend"]


class CrossoverCandidate(BaseModel):
    """Candidate strategy produced by recombining two Pareto/niche parents."""

    child_strategy: Strategy
    primary_parent_id: str
    secondary_parent_id: str
    operator: CrossoverOperator
    rationale: str
    summary_changes: list[StrategyChange] = Field(default_factory=list)


class SemanticCrossoverBreeder:
    """Structure-preserving genetic crossover over YAML DSL entry/exit components."""

    def __init__(self, *, complexity_limit: int = 8) -> None:
        self._complexity_limit = max(2, int(complexity_limit))

    @staticmethod
    def _count_conditions(strategy: Strategy) -> int:
        return (
            len(strategy.entry.triggers)
            + len(strategy.entry.conditions)
            + len(strategy.entry.guards)
            + len(strategy.entry.filters)
            + len(strategy.exit.triggers)
        )

    @staticmethod
    def _used_indicators(strategy: Strategy) -> set[str]:
        return {
            c.indicator
            for c in (
                strategy.entry.triggers
                + strategy.entry.conditions
                + strategy.entry.guards
                + strategy.entry.filters
            )
        }

    @staticmethod
    def _canonical_hash(strategy: Strategy) -> str:
        payload = strategy.model_dump(mode="json", exclude={"meta", "complexity_score"})
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _summarize_exit_changes(primary: Strategy, secondary: Strategy) -> list[StrategyChange]:
        """Build StrategyMutator-compatible exit change descriptors for Prosecutor/Critic."""
        changes: list[StrategyChange] = []
        p_sl, s_sl = primary.exit.stop_loss, secondary.exit.stop_loss
        p_tp, s_tp = primary.exit.take_profit, secondary.exit.take_profit
        if s_sl.type != p_sl.type:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.stop_loss.type",
                    from_value=p_sl.type,
                    to_value=s_sl.type,
                    reason=f"Crossover stop_loss.type from {secondary.meta.id}",
                )
            )
        if s_sl.value is not None and s_sl.value != p_sl.value:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.stop_loss.value",
                    from_value=p_sl.value,
                    to_value=s_sl.value,
                    reason=f"Crossover stop_loss.value from {secondary.meta.id}",
                )
            )
        if s_sl.multiplier is not None and s_sl.multiplier != p_sl.multiplier:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.stop_loss.multiplier",
                    from_value=p_sl.multiplier,
                    to_value=s_sl.multiplier,
                    reason=f"Crossover stop_loss.multiplier from {secondary.meta.id}",
                )
            )
        if s_tp.type != p_tp.type:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.take_profit.type",
                    from_value=p_tp.type,
                    to_value=s_tp.type,
                    reason=f"Crossover take_profit.type from {secondary.meta.id}",
                )
            )
        if s_tp.value is not None and s_tp.value != p_tp.value:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.take_profit.value",
                    from_value=p_tp.value,
                    to_value=s_tp.value,
                    reason=f"Crossover take_profit.value from {secondary.meta.id}",
                )
            )
        if s_tp.trigger_pct is not None and s_tp.trigger_pct != p_tp.trigger_pct:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.take_profit.trigger_pct",
                    from_value=p_tp.trigger_pct,
                    to_value=s_tp.trigger_pct,
                    reason=f"Crossover take_profit.trigger_pct from {secondary.meta.id}",
                )
            )
        if s_tp.trail_pct is not None and s_tp.trail_pct != p_tp.trail_pct:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.take_profit.trail_pct",
                    from_value=p_tp.trail_pct,
                    to_value=s_tp.trail_pct,
                    reason=f"Crossover take_profit.trail_pct from {secondary.meta.id}",
                )
            )
        if secondary.exit.max_holding_days != primary.exit.max_holding_days:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.max_holding_days",
                    from_value=primary.exit.max_holding_days,
                    to_value=secondary.exit.max_holding_days,
                    reason=f"Crossover max_holding_days from {secondary.meta.id}",
                )
            )
        if not changes:
            changes.append(
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.max_holding_days",
                    from_value=primary.exit.max_holding_days,
                    to_value=secondary.exit.max_holding_days,
                    reason=f"Crossover exit structure from {secondary.meta.id}",
                )
            )
        return changes[:3]

    @staticmethod
    def _filter_target_bucket(primary: Strategy) -> tuple[str, list[StrategyCondition]]:
        if primary.entry.triggers:
            return "entry.guards", primary.entry.guards
        if primary.entry.logic == "or":
            return "entry.filters", primary.entry.filters
        return "entry.conditions", primary.entry.conditions

    def breed(
        self,
        primary: Strategy,
        secondary: Strategy,
        *,
        max_children: int = 2,
    ) -> list[CrossoverCandidate]:
        """Recombine *primary* and *secondary* into valid, deduplicated child strategies."""
        if max_children <= 0 or primary.meta.id == secondary.meta.id:
            return []

        seen_hashes: set[str] = {
            self._canonical_hash(primary),
            self._canonical_hash(secondary),
        }
        candidates: list[CrossoverCandidate] = []

        def _try_accept(
            child: Strategy,
            operator: CrossoverOperator,
            rationale: str,
            changes: list[StrategyChange],
        ) -> None:
            if len(candidates) >= max_children:
                return
            if self._count_conditions(child) > self._complexity_limit:
                return
            chash = self._canonical_hash(child)
            if chash in seen_hashes:
                return
            seen_hashes.add(chash)
            candidates.append(
                CrossoverCandidate(
                    child_strategy=child,
                    primary_parent_id=primary.meta.id,
                    secondary_parent_id=secondary.meta.id,
                    operator=operator,
                    rationale=rationale,
                    summary_changes=changes,
                )
            )

        # Operator 1: Exit Grafting — combine primary's entry edge with secondary's exit discipline.
        if secondary.exit.model_dump(mode="json") != primary.exit.model_dump(mode="json"):
            child_exit = primary.model_copy(deep=True)
            child_exit.exit = secondary.exit.model_copy(deep=True)
            _try_accept(
                child_exit,
                "exit_graft",
                f"Semantic crossover exit_graft: {primary.meta.id} entry × {secondary.meta.id} exit",
                self._summarize_exit_changes(primary, secondary),
            )

        # Operator 2: Filter / Guard Grafting — transfer orthogonal guard/L2 factor into primary.
        used_primary = self._used_indicators(primary)
        dynamic_set = set(IndicatorRegistry.dynamic_names())
        donor_guards = [
            c
            for c in (secondary.entry.guards + secondary.entry.filters + secondary.entry.conditions)
            if c.indicator not in used_primary
        ]
        donor_guards.sort(key=lambda c: (0 if c.indicator in dynamic_set else 1, c.indicator))
        if donor_guards and self._count_conditions(primary) + 1 <= self._complexity_limit:
            graft_cond = donor_guards[0].model_copy(deep=True)
            child_flt = primary.model_copy(deep=True)
            target_field, bucket = self._filter_target_bucket(child_flt)
            bucket.append(graft_cond)
            _try_accept(
                child_flt,
                "filter_graft",
                f"Semantic crossover filter_graft: '{graft_cond.indicator}' from {secondary.meta.id}",
                [
                    StrategyChange(
                        change_type=ChangeType.ADD_CONDITION,
                        target=target_field,
                        from_value=None,
                        to_value=graft_cond.model_dump(mode="json"),
                        reason=(
                            f"Crossover filter_graft ({graft_cond.indicator}) from "
                            f"{secondary.meta.id} into {target_field}"
                        ),
                    )
                ],
            )

        # Operator 3: Trigger Blending — unite orthogonal entry triggers via OR for same market.
        if (
            primary.meta.market == secondary.meta.market
            and self._count_conditions(primary) + 1 <= self._complexity_limit
        ):
            donor_trigs = [
                c
                for c in (secondary.entry.triggers or secondary.entry.conditions)
                if c.indicator not in used_primary
            ]
            if donor_trigs:
                blend_cond = donor_trigs[0].model_copy(deep=True)
                child_blend = primary.model_copy(deep=True)
                if child_blend.entry.triggers:
                    trig_target = "entry.triggers"
                    child_blend.entry.triggers.append(blend_cond)
                else:
                    trig_target = "entry.conditions"
                    child_blend.entry.conditions.append(blend_cond)
                prev_logic = child_blend.entry.logic
                child_blend.entry.logic = "or"
                changes_blend = [
                    StrategyChange(
                        change_type=ChangeType.ADD_CONDITION,
                        target=trig_target,
                        from_value=None,
                        to_value=blend_cond.model_dump(mode="json"),
                        reason=(
                            f"Crossover trigger_blend ({blend_cond.indicator}) from "
                            f"{secondary.meta.id}"
                        ),
                    )
                ]
                if prev_logic != "or":
                    changes_blend.append(
                        StrategyChange(
                            change_type=ChangeType.CHANGE_LOGIC,
                            target="entry.logic",
                            from_value=prev_logic,
                            to_value="or",
                            reason="Unite blended crossover entry triggers under OR logic",
                        )
                    )
                _try_accept(
                    child_blend,
                    "trigger_blend",
                    f"Semantic crossover trigger_blend: +'{blend_cond.indicator}' (OR) from {secondary.meta.id}",
                    changes_blend,
                )

        return candidates


__all__ = [
    "CrossoverCandidate",
    "CrossoverOperator",
    "SemanticCrossoverBreeder",
]
