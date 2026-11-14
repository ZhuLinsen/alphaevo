"""Strategy DSL parser — loads YAML strategy files into Strategy models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.models.strategy import (
    ExecutionConfig,
    MarketRuleConfig,
    PortfolioProfile,
    SetupSpec,
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    TunableParam,
    UniverseConfig,
    UniverseFilter,
)
from alphaevo.strategy.tunable import resolve_tunable_target

if TYPE_CHECKING:
    from pathlib import Path


class StrategyParseError(Exception):
    """Raised when a strategy YAML cannot be parsed."""


@dataclass
class ValidationDiagnostics:
    """Semantic validation diagnostics for a parsed strategy."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def problems(self) -> list[str]:
        return [*self.errors, *self.warnings]


class StrategyParser:
    """Parse YAML DSL files into validated Strategy objects."""

    _SUPPORTED_STOP_LOSS = {"pct", "atr", "price_level", "pct_from_low", "composite"}
    _SUPPORTED_TAKE_PROFIT = {"rr", "pct", "target_ma", "trailing"}
    _SUPPORTED_EXIT_TARGETS = {
        "exit.stop_loss.value",
        "exit.stop_loss.type",
        "exit.stop_loss.reference",
        "exit.stop_loss.multiplier",
        "exit.stop_loss.atr_period",
        "exit.take_profit.value",
        "exit.take_profit.type",
        "exit.take_profit.target",
        "exit.take_profit.trigger_pct",
        "exit.take_profit.trail_pct",
        "exit.max_holding_days",
    }

    def parse_file(self, path: Path) -> Strategy:
        """Load and parse a strategy YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return self.parse_yaml(content, source=str(path))

    def parse_yaml(self, content: str, source: str = "<string>") -> Strategy:
        """Parse YAML string into a Strategy."""
        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise StrategyParseError(f"Invalid YAML in {source}: {e}") from e

        if not isinstance(raw, dict):
            raise StrategyParseError(f"Expected dict at top level in {source}")

        try:
            return self._build_strategy(raw)
        except (PydanticValidationError, KeyError, TypeError) as e:
            raise StrategyParseError(f"Strategy validation failed in {source}: {e}") from e

    def parse_directory(self, directory: Path) -> list[Strategy]:
        """Parse all .yaml files in a directory."""
        strategies: list[Strategy] = []
        if not directory.is_dir():
            return strategies
        for path in sorted(directory.glob("*.yaml")):
            try:
                strategies.append(self.parse_file(path))
            except (StrategyParseError, FileNotFoundError):
                continue
        return strategies

    def validate(self, strategy: Strategy) -> list[str]:
        """Validate a strategy and return all semantic problems."""
        diagnostics = self.diagnose(strategy)
        return diagnostics.problems

    def diagnose(self, strategy: Strategy) -> ValidationDiagnostics:
        """Return semantic validation errors and warnings separately."""
        diagnostics = ValidationDiagnostics()

        if not strategy.entry.conditions and not strategy.entry.triggers:
            diagnostics.warnings.append("Strategy has no entry conditions or triggers")

        if strategy.exit.max_holding_days < 1:
            diagnostics.errors.append("max_holding_days must be >= 1")

        if strategy.exit.max_holding_days > 60:
            diagnostics.warnings.append("max_holding_days > 60 is unusually long")

        if strategy.exit.stop_loss.type not in self._SUPPORTED_STOP_LOSS:
            diagnostics.errors.append(f"Unsupported stop_loss.type: {strategy.exit.stop_loss.type}")

        if strategy.exit.take_profit.type not in self._SUPPORTED_TAKE_PROFIT:
            diagnostics.errors.append(
                f"Unsupported take_profit.type: {strategy.exit.take_profit.type}"
            )

        if strategy.exit.stop_loss.type == "price_level" and (
            strategy.exit.stop_loss.value is None and not strategy.exit.stop_loss.reference
        ):
            diagnostics.errors.append("price_level stop loss requires either value or reference")

        if (
            strategy.exit.take_profit.type in {"rr", "pct"}
            and strategy.exit.take_profit.value is None
        ):
            diagnostics.errors.append(
                f"{strategy.exit.take_profit.type} take profit requires a numeric value"
            )

        tp_target = strategy.exit.take_profit.target
        if strategy.exit.take_profit.type == "target_ma" and not tp_target:
            diagnostics.errors.append("target_ma take profit requires a target moving average")
        elif strategy.exit.take_profit.type == "target_ma":
            assert tp_target is not None
            target = tp_target.lower().strip()
            if not re.fullmatch(r"ma\d+", target):
                diagnostics.errors.append(
                    "target_ma take profit target must look like ma20 or ma180"
                )

        all_conditions = (
            strategy.entry.conditions
            + strategy.entry.triggers
            + strategy.entry.filters
            + strategy.entry.guards
            + strategy.exit.triggers
            + (strategy.exit.stop_loss.conditions or [])
        )
        for condition in all_conditions:
            if not IndicatorRegistry.is_registered(condition.indicator):
                diagnostics.errors.append(f"Unknown indicator: {condition.indicator}")

        for param in strategy.params.tunable:
            lo, hi = param.range
            if lo >= hi:
                diagnostics.errors.append(f"Tunable param '{param.target}': range min >= max")
            if param.step <= 0:
                diagnostics.errors.append(f"Tunable param '{param.target}': step must be > 0")
            if not self._is_supported_tunable_target(param.target):
                diagnostics.errors.append(
                    f"Tunable param '{param.target}': unsupported target path"
                )
                continue
            if resolve_tunable_target(strategy, param.target) is None:
                diagnostics.errors.append(
                    f"Tunable param '{param.target}': target does not resolve to a tunable value"
                )

        entry_rule_count = (
            len(strategy.entry.triggers)
            + len(strategy.entry.conditions)
            + len(strategy.entry.guards)
            + len(strategy.entry.filters)
        )
        if entry_rule_count > 8:
            diagnostics.warnings.append("Strategy has > 8 entry rules — high overfitting risk")

        if (
            strategy.setup is not None
            and strategy.setup.target_rr is not None
            and strategy.setup.target_rr < strategy.setup.min_rr
        ):
            diagnostics.errors.append("setup.target_rr must be >= setup.min_rr")

        return diagnostics

    def assert_valid(
        self,
        strategy: Strategy,
        *,
        strict: bool = False,
    ) -> ValidationDiagnostics:
        """Raise StrategyParseError when semantic validation fails."""
        diagnostics = self.diagnose(strategy)
        if diagnostics.errors:
            raise StrategyParseError("; ".join(diagnostics.errors))
        if strict and diagnostics.warnings:
            raise StrategyParseError("; ".join(diagnostics.warnings))
        return diagnostics

    def _is_supported_tunable_target(self, target: str) -> bool:
        """Return True when the target path matches implemented mutation hooks."""
        if target in self._SUPPORTED_EXIT_TARGETS:
            return True
        if re.fullmatch(
            r"entry\.(conditions|filters|triggers|guards)\[\d+\]\."
            r"(value|indicator(?:\.(?:fast|slow|signal|std))?)",
            target,
        ):
            return True
        return bool(
            re.fullmatch(
                r"entry\.(conditions|filters|triggers|guards)\[indicator=[^]]+\]\."
                r"(value|indicator(?:\.(?:fast|slow|signal|std))?)",
                target,
            )
        )

    # ── internal builders ─────────────────────────────────────────────

    def _build_strategy(self, raw: dict[str, Any]) -> Strategy:
        for required in ("meta", "entry", "exit"):
            if required not in raw:
                raise StrategyParseError(f"Missing required top-level key: '{required}'")

        meta = StrategyMeta(**raw["meta"])
        description = raw.get("description", "")

        universe = self._build_universe(raw.get("universe", {}))
        entry = self._build_entry(raw["entry"])
        exit_ = self._build_exit(raw["exit"])
        params = self._build_params(raw.get("params", {}))
        market_rules = self._build_market_rules(raw.get("market_rules", {}))
        setup = (
            SetupSpec(**raw["setup"]) if "setup" in raw and isinstance(raw["setup"], dict) else None
        )
        portfolio_profile = (
            PortfolioProfile(**raw["portfolio_profile"])
            if "portfolio_profile" in raw and isinstance(raw["portfolio_profile"], dict)
            else None
        )

        return Strategy(
            meta=meta,
            description=description,
            universe=universe,
            entry=entry,
            exit=exit_,
            params=params,
            market_rules=market_rules,
            setup=setup,
            portfolio_profile=portfolio_profile,
        )

    def _build_universe(self, raw: dict[str, Any]) -> UniverseConfig:
        filters = [UniverseFilter(**f) for f in raw.get("filters", [])]
        return UniverseConfig(
            market=raw.get("market", ["a_share_main"]),
            filters=filters,
        )

    def _build_entry(self, raw: dict[str, Any]) -> StrategyEntry:
        conditions = [StrategyCondition(**c) for c in raw.get("conditions", [])]
        triggers = [StrategyCondition(**c) for c in raw.get("triggers", [])]
        filters = [StrategyCondition(**f) for f in raw.get("filters", [])]
        guards = [StrategyCondition(**f) for f in raw.get("guards", [])]
        logic = raw.get("logic", "and")
        execution = None
        if "execution" in raw and isinstance(raw["execution"], dict):
            execution = ExecutionConfig(**raw["execution"])
        return StrategyEntry(
            logic=logic,
            triggers=triggers,
            guards=guards,
            conditions=conditions,
            filters=filters,
            execution=execution,
        )

    def _build_exit(self, raw: dict[str, Any]) -> StrategyExit:
        sl_raw = raw.get("stop_loss", {})
        tp_raw = raw.get("take_profit", {})

        # Normalize composite stop loss conditions to StrategyCondition
        if sl_raw.get("type") == "composite" and "conditions" in sl_raw:
            normalized: list[dict[str, Any]] = []
            for c in sl_raw["conditions"]:
                if "indicator" in c:
                    normalized.append(c)
                elif "type" in c:
                    # Legacy format: {type: "sector_rank_exit", threshold: 10}
                    # Convert to standard StrategyCondition
                    normalized.append(
                        {
                            "indicator": c["type"],
                            "op": c.get("op", "=="),
                            "value": c.get("threshold", c.get("value", True)),
                        }
                    )
            sl_raw = {**sl_raw, "conditions": normalized}

        return StrategyExit(
            triggers=[StrategyCondition(**c) for c in raw.get("triggers", [])],
            stop_loss=StopLossConfig(**sl_raw),
            take_profit=TakeProfitConfig(**tp_raw),
            max_holding_days=raw.get("max_holding_days", 10),
        )

    def _build_params(self, raw: dict[str, Any]) -> StrategyParams:
        tunable = [TunableParam(**p) for p in raw.get("tunable", [])]
        return StrategyParams(tunable=tunable)

    def _build_market_rules(self, raw: dict[str, Any]) -> dict[str, MarketRuleConfig]:
        rules: dict[str, MarketRuleConfig] = {}
        for market_key, rule_data in raw.items():
            if isinstance(rule_data, dict):
                rules[market_key] = MarketRuleConfig(**rule_data)
        return rules
