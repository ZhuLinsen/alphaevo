"""Shared scoring utilities for strategy optimization candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from alphaevo.models.enums import RSILevel
from alphaevo.models.execution import BacktestResult, EvaluationReport

if TYPE_CHECKING:
    from alphaevo.models.strategy import Strategy

OptimizationObjective = Literal[
    "confidence",
    "win_rate",
    "avg_return",
    "drawdown",
    "quality",
    "profit_quality",
    "robust_profit_quality",
]


def normalize_objective(objective: str) -> OptimizationObjective:
    """Normalize objective aliases used by CLI and optimizer APIs."""
    key = objective.strip().lower().replace("-", "_")
    aliases = {
        "score": "confidence",
        "confidence_score": "confidence",
        "wr": "win_rate",
        "winrate": "win_rate",
        "return": "avg_return",
        "avg": "avg_return",
        "mdd": "drawdown",
        "max_drawdown": "drawdown",
        "balanced": "quality",
        "edge": "quality",
        "expectancy": "profit_quality",
        "profit": "profit_quality",
        "profitability": "profit_quality",
        "payoff": "profit_quality",
        "return_quality": "profit_quality",
        "robust": "robust_profit_quality",
        "robust_profit": "robust_profit_quality",
        "robust_expectancy": "robust_profit_quality",
        "stable_profit": "robust_profit_quality",
    }
    key = aliases.get(key, key)
    valid = {
        "confidence",
        "win_rate",
        "avg_return",
        "drawdown",
        "quality",
        "profit_quality",
        "robust_profit_quality",
    }
    if key not in valid:
        raise ValueError(f"Unsupported optimization objective: {objective}")
    return key  # type: ignore[return-value]


def objective_value(evaluation: EvaluationReport, objective: OptimizationObjective) -> float:
    """Return the scalar value for one optimization objective."""
    ev = evaluation.overall
    if objective == "win_rate":
        return ev.win_rate
    if objective == "avg_return":
        return ev.avg_return
    if objective == "drawdown":
        return -ev.max_drawdown
    if objective == "quality":
        return quality_score(evaluation)
    if objective == "profit_quality":
        return profit_quality_score(evaluation)
    if objective == "robust_profit_quality":
        return robust_profit_quality_score(evaluation)
    return evaluation.confidence_score


def quality_score(evaluation: EvaluationReport) -> float:
    """Blend win rate and payoff quality into a bounded optimization score.

    The score intentionally gives no credit for negative average return or a
    profit/loss ratio below 1.0. This prevents high-win-rate candidates with
    poor payoff asymmetry from dominating the search.
    """
    ev = evaluation.overall
    win_rate_score = _clamp((ev.win_rate - 0.45) / 0.15)
    avg_return_score = _clamp(ev.avg_return / 0.02)
    profit_loss_score = _clamp((ev.profit_loss_ratio - 1.0) / 1.0)
    drawdown_score = _clamp(1.0 - ev.max_drawdown / 0.40)
    signal_reliability = _clamp(ev.signal_count / 30.0)
    raw_score = (
        0.30 * win_rate_score
        + 0.30 * avg_return_score
        + 0.20 * profit_loss_score
        + 0.20 * drawdown_score
    )
    return round(raw_score * signal_reliability, 6)


def profit_quality_score(evaluation: EvaluationReport) -> float:
    """Return-focused score inspired by hyperopt loss functions.

    Compared with :func:`quality_score`, this gives more weight to average
    return, compounded total return, and payoff ratio while still requiring a
    usable win rate and drawdown profile.
    """
    ev = evaluation.overall
    avg_return_score = _clamp(ev.avg_return / 0.015)
    total_return_score = _clamp(ev.total_return / 0.20)
    profit_loss_score = _clamp((ev.profit_loss_ratio - 1.0) / 1.5)
    win_rate_score = _clamp((ev.win_rate - 0.45) / 0.15)
    drawdown_score = _clamp(1.0 - ev.max_drawdown / 0.35)
    signal_reliability = _clamp(ev.signal_count / 30.0)
    raw_score = (
        0.35 * avg_return_score
        + 0.20 * total_return_score
        + 0.20 * profit_loss_score
        + 0.15 * win_rate_score
        + 0.10 * drawdown_score
    )
    if ev.avg_return <= 0 or ev.profit_loss_ratio < 1.0:
        raw_score *= 0.25
    return round(raw_score * signal_reliability, 6)


def robustness_score(evaluation: EvaluationReport) -> float:
    """Score train/validation/test and walk-forward stability.

    Fast evaluation includes train/validation/test anti-overfit metrics but not
    walk-forward folds, so walk-forward dimensions are neutral until a full
    candidate re-evaluation populates them.
    """
    anti = evaluation.anti_overfit
    train_val_score = _clamp(1.0 - anti.train_val_gap / 0.20)
    val_test_score = _clamp(1.0 - anti.val_test_gap / 0.15)
    if evaluation.walk_forward:
        walk_forward_gap_score = _clamp(1.0 - anti.walk_forward_gap / 0.25)
        walk_forward_pass_score = _clamp(anti.walk_forward_pass_rate)
    else:
        walk_forward_gap_score = 0.65
        walk_forward_pass_score = 0.65
    overfit_score = 0.0 if anti.is_overfit else 1.0
    raw_score = (
        0.30 * train_val_score
        + 0.25 * val_test_score
        + 0.20 * walk_forward_gap_score
        + 0.15 * walk_forward_pass_score
        + 0.10 * overfit_score
    )
    return round(raw_score, 6)


def robust_profit_quality_score(evaluation: EvaluationReport) -> float:
    """Blend return quality with anti-overfit stability."""
    return round(
        0.65 * profit_quality_score(evaluation) + 0.35 * robustness_score(evaluation),
        6,
    )


def candidate_sort_key(
    evaluation: EvaluationReport,
    *,
    passed_gate: bool,
    objective: OptimizationObjective,
) -> tuple[float, float, float, float, float, float, float, float, float, int, float]:
    """Return a stable sort key that keeps payoff quality in tie-breaks."""
    ev = evaluation.overall
    return (
        1.0 if passed_gate else 0.0,
        objective_value(evaluation, objective),
        robustness_score(evaluation),
        profit_quality_score(evaluation),
        quality_score(evaluation),
        evaluation.confidence_score,
        ev.avg_return,
        ev.total_return,
        ev.profit_loss_ratio,
        ev.signal_count,
        -ev.max_drawdown,
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def record_optimizer_trials(
    trial_ledger: Any,
    *,
    base_strategy: Strategy,
    candidates: Sequence[Any],
    backtest_results: Mapping[str, BacktestResult],
    best_candidate_id: str | None,
    data: Mapping[str, Any],
    notes: str,
) -> None:
    """Persist optimizer evaluations into ``TrialLedger`` without duplicating loop code."""
    if trial_ledger is None or not candidates:
        return

    from alphaevo.evaluator.metrics import Evaluator
    from alphaevo.truth.ledger import compute_data_fingerprint

    family_id = trial_ledger.resolve_trial_family_id(base_strategy)
    data_fp = compute_data_fingerprint(dict(data))

    for cand in candidates:
        changes = getattr(cand, "changes", [])
        if changes == ["baseline"]:
            continue
        ev = cand.evaluation
        bt_res = backtest_results.get(cand.candidate_id)
        executed = [s for s in bt_res.signals if s.exit_price is not None] if bt_res else []
        returns: list[float]
        if ev.overall.metrics_basis == "daily_mtm" and data and executed:
            returns = Evaluator.compute_daily_mtm_returns(executed, dict(data))[0]
        else:
            returns = [s.return_pct for s in executed]
        is_accepted = bool(cand.candidate_id == best_candidate_id and cand.passed_gate)
        trial_rec = trial_ledger.record_trial(
            strategy=cand.strategy,
            trial_family_id=family_id,
            data_fingerprint=data_fp,
            rsi_level=RSILevel.L1_STRATEGY,
            round_num=0,
            signal_count=ev.overall.signal_count,
            sharpe_ratio=ev.overall.sharpe_ratio,
            returns=returns,
            confidence_score=ev.confidence_score,
            passed_null_control=ev.null_control_passed,
            accepted=is_accepted,
            notes=notes,
        )
        ev.family_trial_count = trial_ledger.count_family_trials(family_id)
        ev.deflated_sharpe_prob = trial_rec.deflated_sharpe_prob
