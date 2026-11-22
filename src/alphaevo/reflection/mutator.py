"""Strategy mutator — applies proposed changes to produce a new strategy version.

Pure logic module (no LLM). Takes a Strategy and a list of StrategyChange objects
and produces a new Strategy with incremented version and updated parent_id.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import StrategyChange
from alphaevo.models.strategy import (
    Strategy,
    StrategyCondition,
)
from alphaevo.strategy.tunable import set_tunable_target

logger = logging.getLogger(__name__)


class MutationError(Exception):
    """Raised when a mutation cannot be applied."""


class StrategyMutator:
    """Apply StrategyChange objects to mutate a strategy into a new version.

    Safety guardrails (from AGENTS.md §11):
    - max_changes: maximum changes per round (default 3)
    - complexity_limit: max entry conditions (default 8)
    """

    def __init__(
        self,
        max_changes: int = 3,
        complexity_limit: int = 8,
    ) -> None:
        if complexity_limit < 1:
            raise ValueError(f"complexity_limit must be >= 1, got {complexity_limit}")
        self.max_changes = max_changes
        self.complexity_limit = complexity_limit

    def mutate(
        self,
        strategy: Strategy,
        changes: list[StrategyChange],
        *,
        atomic: bool = False,
    ) -> Strategy:
        """Apply changes to strategy, returning a new version.

        When ``atomic`` is True, every change in the bundle must apply
        successfully or the whole mutation is rejected.

        Raises MutationError if changes violate safety guardrails.
        """
        if len(changes) > self.max_changes:
            if atomic:
                raise MutationError(
                    f"Atomic mutation requires <= {self.max_changes} changes, got {len(changes)}"
                )
            logger.warning("Truncating %d changes to max %d", len(changes), self.max_changes)
            changes = changes[: self.max_changes]

        # Deep copy to avoid modifying original
        new = strategy.model_copy(deep=True)

        applied: list[StrategyChange] = []
        for change in changes:
            try:
                self._apply_change(new, change)
                applied.append(change)
            except Exception as e:
                if atomic:
                    raise MutationError(f"Atomic mutation failed on {change.target}: {e}") from e
                logger.warning("Skipping change %s: %s", change.target, e)

        if not applied:
            raise MutationError("No changes could be applied")

        if any(c.change_type == ChangeType.ADJUST_EXIT for c in applied):
            prune_dangling_exit_tunables(new)

        # Enforce complexity limit on total entry trigger/filter rules.
        n_conditions = _entry_rule_count(new)
        if n_conditions > self.complexity_limit:
            if atomic:
                raise MutationError(
                    f"Atomic mutation would exceed complexity limit "
                    f"({n_conditions} > {self.complexity_limit})"
                )
            logger.warning(
                "Complexity %d exceeds limit %d, trimming",
                n_conditions,
                self.complexity_limit,
            )
            _trim_entry_rules(new, self.complexity_limit)

        # Bump version
        old_id = new.meta.id
        new_version = new.meta.version + 1
        base_name = old_id.rsplit("_v", 1)[0]
        new_id = f"{base_name}_v{new_version}"

        new.meta.id = new_id
        new.meta.version = new_version
        new.meta.parent_id = old_id
        new.meta.created_at = datetime.now(timezone.utc)

        return new

    def _apply_change(self, strategy: Strategy, change: StrategyChange) -> None:
        """Apply a single change to a strategy (in-place)."""
        handler = _CHANGE_HANDLERS.get(change.change_type)
        if handler is None:
            raise MutationError(f"Unknown change type: {change.change_type}")
        handler(strategy, change)


# ── Change handlers ──────────────────────────────────────────────────


def _tighten_filter(strategy: Strategy, change: StrategyChange) -> None:
    """Make a condition more restrictive (e.g., lower threshold)."""
    _update_condition_value(strategy, change)


def _loosen_filter(strategy: Strategy, change: StrategyChange) -> None:
    """Make a condition less restrictive (e.g., raise threshold)."""
    _update_condition_value(strategy, change)


def promote_legacy_conditions_to_triggers(
    strategy: Strategy,
    new_trigger: StrategyCondition,
) -> None:
    """Promote legacy entry.conditions into entry.triggers (+ guards for multi-AND) and retarget tunables."""
    if strategy.entry.logic == "and" and len(strategy.entry.conditions) >= 2:
        orig_conds = list(strategy.entry.conditions)
        guard_offset = len(strategy.entry.guards)
        promoted_first = orig_conds[0]
        moved_guards = orig_conds[1:]
        strategy.entry.triggers = [promoted_first, new_trigger]
        strategy.entry.guards.extend(moved_guards)
        strategy.entry.conditions = []
        first_ind = promoted_first.indicator
        for param in strategy.params.tunable:
            if not param.target.startswith("entry.conditions["):
                continue
            m = re.match(r"^entry\.conditions\[(\d+)\](.*)$", param.target)
            if m is not None:
                idx = int(m.group(1))
                suffix = m.group(2)
                if idx == 0:
                    param.target = f"entry.triggers[0]{suffix}"
                elif 1 <= idx < len(orig_conds):
                    param.target = f"entry.guards[{guard_offset + idx - 1}]{suffix}"
            else:
                replacement_bucket = (
                    "entry.triggers["
                    if f"[indicator={first_ind}]" in param.target
                    else "entry.guards["
                )
                param.target = param.target.replace("entry.conditions[", replacement_bucket, 1)
    else:
        strategy.entry.triggers = [*strategy.entry.conditions, new_trigger]
        strategy.entry.conditions = []
        for param in strategy.params.tunable:
            if param.target.startswith("entry.conditions["):
                param.target = param.target.replace("entry.conditions[", "entry.triggers[", 1)


def _add_condition(strategy: Strategy, change: StrategyChange) -> None:
    """Add a new entry condition."""
    if not isinstance(change.to_value, dict):
        raise MutationError(f"ADD_CONDITION requires to_value as dict, got {type(change.to_value)}")
    new_cond = StrategyCondition(**change.to_value)

    # Validate that the indicator exists in the registry before adding,
    # so LLM-invented indicators are caught before they reach backtest.
    from alphaevo.backtest.indicators import IndicatorRegistry

    if not IndicatorRegistry.is_registered(new_cond.indicator):
        raise MutationError(
            f"Unknown indicator '{new_cond.indicator}' proposed by LLM. "
            f"Available: {IndicatorRegistry.available()}"
        )

    if (
        change.target.startswith("entry.triggers")
        and not strategy.entry.triggers
        and strategy.entry.conditions
    ):
        promote_legacy_conditions_to_triggers(strategy, new_cond)
        return

    _entry_bucket(strategy, change.target).append(new_cond)


def prune_dangling_exit_tunables(strategy: Strategy) -> None:
    """Drop tunable targets referencing exit fields no longer active after exit changes."""
    sl = strategy.exit.stop_loss
    tp = strategy.exit.take_profit
    kept = []
    for param in strategy.params.tunable:
        tgt = param.target.strip()
        if tgt == "exit.stop_loss.value" and sl.value is None:
            continue
        if tgt in {"exit.stop_loss.multiplier", "exit.stop_loss.atr_period"} and (sl.type != "atr"):
            continue
        if tgt == "exit.take_profit.value" and tp.value is None:
            continue
        if tgt == "exit.take_profit.target" and tp.type != "target_ma":
            continue
        if tgt in {"exit.take_profit.trigger_pct", "exit.take_profit.trail_pct"} and (
            tp.type != "trailing"
        ):
            continue
        kept.append(param)
    strategy.params.tunable = kept


def _remove_condition(strategy: Strategy, change: StrategyChange) -> None:
    """Remove an entry condition by indicator name."""
    indicator = change.target.split("indicator=")[-1].split("]")[0].strip()
    if not indicator:
        raise MutationError(f"Invalid target format (empty indicator name): {change.target}")

    for bucket in (
        strategy.entry.triggers,
        strategy.entry.conditions,
        strategy.entry.guards,
        strategy.entry.filters,
    ):
        for i, condition in enumerate(bucket):
            if condition.indicator == indicator:
                bucket.pop(i)
                return

    logger.warning("Condition with indicator=%s not found, skipping", indicator)


def _adjust_exit(strategy: Strategy, change: StrategyChange) -> None:
    """Modify exit parameters (stop loss, take profit, max holding days)."""
    target = change.target
    raw_val = change.to_value
    if "stop_loss.conditions" in target:
        if raw_val is None:
            strategy.exit.stop_loss.conditions = None
        elif isinstance(raw_val, list):
            strategy.exit.stop_loss.conditions = [
                c if isinstance(c, StrategyCondition) else StrategyCondition(**c) for c in raw_val
            ]
        else:
            raise MutationError(f"Invalid stop_loss.conditions value: {raw_val}")
        return
    if "exit.triggers" in target or target == "triggers":
        if raw_val is None:
            strategy.exit.triggers = []
        elif isinstance(raw_val, list):
            strategy.exit.triggers = [
                c if isinstance(c, StrategyCondition) else StrategyCondition(**c) for c in raw_val
            ]
        else:
            raise MutationError(f"Invalid exit.triggers value: {raw_val}")
        return
    val = _sanitize_condition_value(raw_val)
    if "stop_loss.value" in target:
        strategy.exit.stop_loss.value = None if val is None else float(val)
    elif "stop_loss.type" in target and val is not None:
        new_type = str(val)
        strategy.exit.stop_loss.type = new_type
        if new_type == "atr":
            strategy.exit.stop_loss.value = None
            strategy.exit.stop_loss.reference = None
            strategy.exit.stop_loss.conditions = None
        elif new_type == "price_level":
            strategy.exit.stop_loss.value = None
            strategy.exit.stop_loss.multiplier = None
            strategy.exit.stop_loss.atr_period = None
            strategy.exit.stop_loss.conditions = None
        elif new_type == "composite":
            strategy.exit.stop_loss.value = None
            strategy.exit.stop_loss.multiplier = None
            strategy.exit.stop_loss.atr_period = None
            strategy.exit.stop_loss.reference = None
        else:
            strategy.exit.stop_loss.multiplier = None
            strategy.exit.stop_loss.atr_period = None
            strategy.exit.stop_loss.reference = None
            strategy.exit.stop_loss.conditions = None
    elif "stop_loss.reference" in target:
        strategy.exit.stop_loss.reference = None if val is None else str(val)
    elif "stop_loss.multiplier" in target:
        strategy.exit.stop_loss.multiplier = None if val is None else float(val)
    elif "stop_loss.atr_period" in target:
        if val is None:
            strategy.exit.stop_loss.atr_period = None
        elif not set_tunable_target(strategy, target, val):
            raise MutationError(f"Invalid stop_loss.atr_period update: {change.target}")
    elif "take_profit.value" in target:
        strategy.exit.take_profit.value = None if val is None else float(val)
    elif "take_profit.target" in target:
        if val is None:
            strategy.exit.take_profit.target = None
        elif not set_tunable_target(strategy, target, val):
            if isinstance(val, str) and val.strip():
                strategy.exit.take_profit.target = val.strip()
            else:
                raise MutationError(f"Invalid take_profit.target update: {change.target}")
    elif "take_profit.type" in target and val is not None:
        new_type = str(val)
        strategy.exit.take_profit.type = new_type
        if new_type == "target_ma":
            strategy.exit.take_profit.value = None
            strategy.exit.take_profit.trigger_pct = None
            strategy.exit.take_profit.trail_pct = None
        elif new_type == "trailing":
            strategy.exit.take_profit.value = None
            strategy.exit.take_profit.target = None
        else:
            strategy.exit.take_profit.target = None
            strategy.exit.take_profit.trigger_pct = None
            strategy.exit.take_profit.trail_pct = None
    elif "take_profit.trigger_pct" in target:
        strategy.exit.take_profit.trigger_pct = None if val is None else float(val)
    elif "take_profit.trail_pct" in target:
        strategy.exit.take_profit.trail_pct = None if val is None else float(val)
    elif "max_holding_days" in target and val is not None:
        strategy.exit.max_holding_days = int(val)
    else:
        raise MutationError(f"Unknown exit target: {target}")


def _change_universe(strategy: Strategy, change: StrategyChange) -> None:
    """Modify universe filters."""
    if "market" in change.target and isinstance(change.to_value, list):
        strategy.universe.market = change.to_value
    else:
        logger.warning("Universe change target '%s' not supported", change.target)


def _change_logic(strategy: Strategy, change: StrategyChange) -> None:
    """Switch entry logic between 'and' and 'or'."""
    new_logic = str(change.to_value).lower()
    if new_logic not in ("and", "or"):
        raise MutationError(f"Invalid logic value: {change.to_value}, expected 'and' or 'or'")
    strategy.entry.logic = new_logic  # type: ignore[assignment]


def _discover_factor(strategy: Strategy, change: StrategyChange) -> None:
    """Add a newly discovered factor as a condition.

    The factor must already be registered in IndicatorRegistry by the
    AlphaFactory pipeline before this handler is called.
    """
    if not isinstance(change.to_value, dict):
        raise MutationError(
            f"DISCOVER_FACTOR requires to_value as dict with condition fields, "
            f"got {type(change.to_value)}"
        )
    new_cond = StrategyCondition(**change.to_value)

    from alphaevo.backtest.indicators import IndicatorRegistry

    if not IndicatorRegistry.is_registered(new_cond.indicator):
        raise MutationError(
            f"Discovered factor '{new_cond.indicator}' not registered in IndicatorRegistry. "
            f"Ensure AlphaFactory.discover() ran successfully first."
        )

    _entry_bucket(strategy, change.target).append(new_cond)


def _sanitize_condition_value(raw: Any) -> Any:
    """Strip operator prefixes LLMs may include (e.g. '< 65.0' → 65.0).

    LLMs sometimes output to_value as '"< 65.0"' or '">= 1.5"' instead of
    the bare number.  We detect the pattern and extract the numeric part.
    Boolean strings ('true'/'false') are also normalised.  Values may arrive
    with YAML/LLM quote characters preserved as part of the string, e.g.
    "'0.01'"; strip those before coercion so numeric thresholds remain numeric
    after mutation and serialization round-trips.
    """
    if not isinstance(raw, str):
        return raw
    # Strip leading comparison operators: <, <=, >, >=, ==, !=
    stripped = re.sub(r"^[<>=!]+\s*", "", raw).strip()
    if not stripped:
        return raw
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        stripped = stripped[1:-1].strip()
    # Boolean
    if stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"
    # Numeric
    try:
        return int(stripped) if "." not in stripped else float(stripped)
    except ValueError:
        return raw


def _update_condition_value(strategy: Strategy, change: StrategyChange) -> None:
    """Update the value of an existing condition by indicator name."""
    sanitized_value = _sanitize_condition_value(change.to_value)
    if set_tunable_target(strategy, change.target, sanitized_value):
        return

    indicator = change.target.split("indicator=")[-1].split("]")[0]

    for cond in (
        strategy.entry.triggers
        + strategy.entry.conditions
        + strategy.entry.guards
        + strategy.entry.filters
    ):
        if cond.indicator == indicator:
            if sanitized_value is not None:
                cond.value = sanitized_value
            return

    raise MutationError(f"Condition with indicator={indicator} not found")


def _entry_bucket(strategy: Strategy, target: str) -> list[StrategyCondition]:
    """Return the entry condition bucket implied by a mutation target."""
    if target.startswith("entry.guards"):
        return strategy.entry.guards
    if target.startswith("entry.filters"):
        return strategy.entry.filters
    if target.startswith("entry.triggers"):
        return strategy.entry.triggers
    return strategy.entry.conditions


def _entry_rule_count(strategy: Strategy) -> int:
    return (
        len(strategy.entry.triggers)
        + len(strategy.entry.conditions)
        + len(strategy.entry.guards)
        + len(strategy.entry.filters)
    )


def _trim_entry_rules(strategy: Strategy, limit: int) -> None:
    """Trim entry buckets in priority order until the total fits the limit."""
    remaining = limit
    for bucket in (
        strategy.entry.triggers,
        strategy.entry.conditions,
        strategy.entry.guards,
        strategy.entry.filters,
    ):
        if remaining <= 0:
            bucket.clear()
            continue
        del bucket[remaining:]
        remaining -= len(bucket)


_CHANGE_HANDLERS = {
    ChangeType.TIGHTEN_FILTER: _tighten_filter,
    ChangeType.LOOSEN_FILTER: _loosen_filter,
    ChangeType.ADD_CONDITION: _add_condition,
    ChangeType.REMOVE_CONDITION: _remove_condition,
    ChangeType.ADJUST_EXIT: _adjust_exit,
    ChangeType.CHANGE_UNIVERSE: _change_universe,
    ChangeType.CHANGE_LOGIC: _change_logic,
    ChangeType.DISCOVER_FACTOR: _discover_factor,
}
