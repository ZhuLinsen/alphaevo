"""AlphaEvo — Open-Source Recursive Self-Improvement (RSI) Agent for Stock Strategy Research."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Strategy",
    "StrategyCondition",
    "ExecutionConfig",
    "EvaluationReport",
    "BacktestResult",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BacktestResult": ("alphaevo.models.execution", "BacktestResult"),
    "EvaluationReport": ("alphaevo.models.execution", "EvaluationReport"),
    "ExecutionConfig": ("alphaevo.models.strategy", "ExecutionConfig"),
    "Strategy": ("alphaevo.models.strategy", "Strategy"),
    "StrategyCondition": ("alphaevo.models.strategy", "StrategyCondition"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'alphaevo' has no attribute {name!r}")
    module_name, attr_name = target
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
