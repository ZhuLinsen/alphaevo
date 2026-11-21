"""Joint entry/parameter seed plus exit/risk optimization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from alphaevo.evaluator.metrics import EvaluationMode
from alphaevo.models.execution import SampleBatch
from alphaevo.models.market import IndicatorContext
from alphaevo.models.strategy import Strategy
from alphaevo.optimizer.exit import ExitOptimizationResult, ExitOptimizer
from alphaevo.optimizer.scoring import (
    OptimizationObjective,
    candidate_sort_key,
    normalize_objective,
)


class JointOptimizer:
    """Refine ranked strategy seeds with a bounded exit/risk search."""

    def __init__(
        self,
        *,
        slippage: float = 0.001,
        commission: float = 0.0003,
        min_data_days: int = 30,
        fill_policy: str = "conservative",
        backtest_config: Any = None,
    ) -> None:
        self.exit_optimizer = ExitOptimizer(
            slippage=slippage,
            commission=commission,
            min_data_days=min_data_days,
            fill_policy=fill_policy,
            backtest_config=backtest_config,
        )

    def optimize(
        self,
        base_strategy_id: str,
        seed_strategies: Iterable[Strategy],
        data: dict[str, Any],
        batch: SampleBatch,
        *,
        contexts: dict[str, IndicatorContext] | None = None,
        spaces: Iterable[str] | None = None,
        max_candidates_per_seed: int = 64,
        objective: OptimizationObjective = "quality",
        evaluation_mode: EvaluationMode = "fast",
        full_eval_top_n: int = 0,
        min_win_rate: float | None = None,
        min_avg_return: float | None = None,
        min_total_return: float | None = None,
        min_profit_loss_ratio: float | None = None,
        max_drawdown: float | None = None,
        min_signals: int | None = None,
        reject_overfit: bool = False,
        max_train_val_gap: float | None = None,
        max_val_test_gap: float | None = None,
        max_walk_forward_gap: float | None = None,
        min_walk_forward_pass_rate: float | None = None,
        parallel_workers: int = 1,
        trial_ledger: Any = None,
    ) -> ExitOptimizationResult:
        """Run exit/risk optimization on each seed and return one combined ranking."""
        normalized_objective = normalize_objective(objective)
        results: list[ExitOptimizationResult] = []
        for seed in seed_strategies:
            results.append(
                self.exit_optimizer.optimize(
                    seed,
                    data,
                    batch,
                    contexts=contexts,
                    spaces=spaces,
                    max_candidates=max_candidates_per_seed,
                    objective=normalized_objective,
                    evaluation_mode=evaluation_mode,
                    full_eval_top_n=full_eval_top_n,
                    min_win_rate=min_win_rate,
                    min_avg_return=min_avg_return,
                    min_total_return=min_total_return,
                    min_profit_loss_ratio=min_profit_loss_ratio,
                    max_drawdown=max_drawdown,
                    min_signals=min_signals,
                    reject_overfit=reject_overfit,
                    max_train_val_gap=max_train_val_gap,
                    max_val_test_gap=max_val_test_gap,
                    max_walk_forward_gap=max_walk_forward_gap,
                    min_walk_forward_pass_rate=min_walk_forward_pass_rate,
                    parallel_workers=parallel_workers,
                    trial_ledger=trial_ledger,
                )
            )

        candidates = [candidate for result in results for candidate in result.candidates]
        ranked = sorted(
            candidates,
            key=lambda candidate: candidate_sort_key(
                candidate.evaluation,
                passed_gate=candidate.passed_gate,
                objective=normalized_objective,
            ),
            reverse=True,
        )
        return ExitOptimizationResult(
            base_strategy_id=f"{base_strategy_id}_joint",
            spaces=list(results[0].spaces) if results else list(spaces or []),
            objective=normalized_objective,
            evaluation_mode=results[0].evaluation_mode if results else "fast",
            full_eval_top_n=sum(result.full_eval_top_n for result in results),
            min_win_rate=min_win_rate,
            min_avg_return=min_avg_return,
            min_total_return=min_total_return,
            min_profit_loss_ratio=min_profit_loss_ratio,
            max_drawdown=max_drawdown,
            min_signals=min_signals,
            reject_overfit=reject_overfit,
            max_train_val_gap=max_train_val_gap,
            max_val_test_gap=max_val_test_gap,
            max_walk_forward_gap=max_walk_forward_gap,
            min_walk_forward_pass_rate=min_walk_forward_pass_rate,
            candidates=ranked,
            best_candidate_id=ranked[0].candidate_id if ranked else None,
        )
