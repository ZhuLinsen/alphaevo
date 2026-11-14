"""AlphaEvo — Open-Source Recursive Self-Improvement (RSI) Agent for Stock Strategy Research"""

__version__ = "0.1.0"

from alphaevo.models.execution import BacktestResult, EvaluationReport
from alphaevo.models.strategy import ExecutionConfig, Strategy, StrategyCondition

__all__ = [
    "__version__",
    "Strategy",
    "StrategyCondition",
    "ExecutionConfig",
    "EvaluationReport",
    "BacktestResult",
]
