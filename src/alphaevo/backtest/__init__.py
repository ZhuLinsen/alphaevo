"""Backtest engine: indicators, condition evaluation, market rules, and execution."""

from alphaevo.backtest.allocator import (
    DailyAllocationPoint,
    MultiStrategyAllocationReport,
    RegimeAdaptiveAllocator,
    RiskBudgetConfig,
    StrategyAllocationSummary,
)
from alphaevo.backtest.condition import ConditionEvaluator
from alphaevo.backtest.engine import BacktestEngine
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.backtest.portfolio import PortfolioBacktester, PortfolioConfig, PortfolioResult
from alphaevo.backtest.rules import MarketRuleChecker

__all__ = [
    "BacktestEngine",
    "ConditionEvaluator",
    "DailyAllocationPoint",
    "IndicatorRegistry",
    "MarketRuleChecker",
    "MultiStrategyAllocationReport",
    "PortfolioBacktester",
    "PortfolioConfig",
    "PortfolioResult",
    "RegimeAdaptiveAllocator",
    "RiskBudgetConfig",
    "StrategyAllocationSummary",
]
