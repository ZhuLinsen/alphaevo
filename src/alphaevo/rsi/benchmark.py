"""Reproducible Multi-Strategy L1 / L2 / Crossover Ablation Benchmark Suite."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from statistics import mean
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from alphaevo.core.config import BacktestConfig
from alphaevo.models.enums import RSILevel
from alphaevo.models.strategy import Strategy
from alphaevo.rsi.engine import RSIRunResult, StockRSIEngine, _compute_rsi_fitness
from alphaevo.rsi.program import RSICharter
from alphaevo.truth.kernel import (
    EvaluationKernel,
    split_market_data_with_holdout,
)
from alphaevo.truth.ledger import compute_kernel_fingerprint
from alphaevo.truth.null_control import NullSignalController

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


class AblationArmMetrics(BaseModel):
    """Metrics captured for one ablation arm of one seed strategy."""

    arm: Literal["baseline", "l1_only", "l1_l2_crossover"]
    champion_id: str
    signals: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    fitness_score: float = 0.0
    confidence_score: float = 0.0
    holdout_avg_return: float | None = None
    holdout_signals: int | None = None
    promotions: int = 0
    vetoes: int = 0
    map_elites_cells: int = 1
    l2_factors: int = 0
    crossovers_kept: int = 0


class BenchmarkStrategyRow(BaseModel):
    """Paired Baseline vs L1-only vs L1+L2+Crossover comparison for one strategy."""

    strategy_id: str
    baseline: AblationArmMetrics
    l1_only: AblationArmMetrics
    l1_l2_crossover: AblationArmMetrics
    fitness_gain_l1: float = 0.0
    fitness_gain_full: float = 0.0


class EvolutionBenchmarkReport(BaseModel):
    """Reproducible ablation scorecard and Tier-2 Null Audit verification."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data_source: str = "bundled_snapshot"
    symbols: list[str] = Field(default_factory=list)
    rounds: int = 2
    locked_holdout_pct: float = 0.20
    rows: list[BenchmarkStrategyRow] = Field(default_factory=list)
    mean_baseline_fitness: float = 0.0
    mean_l1_fitness: float = 0.0
    mean_full_fitness: float = 0.0
    total_l2_factors: int = 0
    total_crossovers_kept: int = 0
    null_audit_passed: bool | None = None
    null_false_promotions: int = 0
    kernel_intact: bool = True


def _metrics_from_rsi_result(
    arm: Literal["l1_only", "l1_l2_crossover"],
    res: RSIRunResult,
    *,
    min_signals: int,
    max_drawdown: float,
) -> AblationArmMetrics:
    champ_eval = res.champion_evaluation
    ov = champ_eval.overall
    veto_cnt = sum(1 for d in res.deliberations if d.l0_outcome in {"prosecutor_vetoed", "revert"})
    fit = _compute_rsi_fitness(
        champ_eval,
        objective="robust_expectancy",
        min_signals=min_signals,
        max_drawdown=max_drawdown,
    )
    ho = res.holdout_evaluation
    return AblationArmMetrics(
        arm=arm,
        champion_id=res.champion_id,
        signals=int(ov.signal_count),
        win_rate=round(float(ov.win_rate), 4),
        avg_return=round(float(ov.avg_return), 6),
        max_drawdown=round(float(ov.max_drawdown), 4),
        sharpe_ratio=round(float(ov.sharpe_ratio), 4),
        fitness_score=round(float(fit), 4),
        confidence_score=round(float(champ_eval.confidence_score), 4),
        holdout_avg_return=(round(float(ho.overall.avg_return), 6) if ho is not None else None),
        holdout_signals=int(ho.overall.signal_count) if ho is not None else None,
        promotions=int(res.promotions),
        vetoes=veto_cnt,
        map_elites_cells=int(res.archive_summary.total_cells_occupied),
        l2_factors=len(res.persisted_factors),
        crossovers_kept=int(res.crossovers_kept),
    )


def run_evolution_benchmark(
    strategies: Sequence[Strategy],
    market_data: dict[str, pd.DataFrame],
    *,
    rounds: int = 2,
    locked_holdout_pct: float = 0.20,
    run_null_audit: bool = True,
    null_worlds: int = 1,
    seed: int = 42,
    data_source: str = "bundled_snapshot",
    backtest_config: BacktestConfig | None = None,
) -> EvolutionBenchmarkReport:
    """Run controlled Baseline vs L1-only vs L1+L2+Crossover ablation benchmark."""
    if not strategies:
        raise ValueError("Provide at least one strategy for ablation benchmarking")

    bt_cfg = backtest_config or BacktestConfig(min_data_days=15)
    min_sigs = min(15, bt_cfg.min_data_days)
    kernel_before = compute_kernel_fingerprint()

    rows: list[BenchmarkStrategyRow] = []
    l2_factor_names: set[str] = set()
    total_xover_kept = 0

    for idx, strat in enumerate(strategies):
        # Baseline on search split (+ locked holdout when enabled) for apples-to-apples comparison
        vault = None
        search_market = market_data
        if locked_holdout_pct > 0.0:
            try:
                vault, _split = split_market_data_with_holdout(
                    market_data, holdout_pct=locked_holdout_pct
                )
                search_market = vault.search_data
            except ValueError:
                vault = None
                search_market = market_data

        base_kernel = EvaluationKernel(
            backtest_config=bt_cfg,
            db_path=":memory:",
            min_data_days=min(15, bt_cfg.min_data_days),
        )
        base_out = base_kernel.evaluate(
            strat,
            search_market,
            mode="fast",
            record_trial=False,
            run_null_control=True,
            null_permutations=20,
            seed=seed + idx,
        )
        base_eval = base_out.evaluation
        base_ho = (
            vault.evaluate_locked_holdout(strat, base_kernel, mode="fast", record_trial=False)
            if vault is not None
            else None
        )
        base_fit = _compute_rsi_fitness(
            base_eval,
            objective="robust_expectancy",
            min_signals=min_sigs,
            max_drawdown=0.25,
        )
        baseline_arm = AblationArmMetrics(
            arm="baseline",
            champion_id=strat.meta.id,
            signals=int(base_eval.overall.signal_count),
            win_rate=round(float(base_eval.overall.win_rate), 4),
            avg_return=round(float(base_eval.overall.avg_return), 6),
            max_drawdown=round(float(base_eval.overall.max_drawdown), 4),
            sharpe_ratio=round(float(base_eval.overall.sharpe_ratio), 4),
            fitness_score=round(float(base_fit), 4),
            confidence_score=round(float(base_eval.confidence_score), 4),
            holdout_avg_return=(
                round(float(base_ho.evaluation.overall.avg_return), 6)
                if base_ho is not None
                else None
            ),
            holdout_signals=int(base_ho.evaluation.overall.signal_count)
            if base_ho is not None
            else None,
        )

        # Arm 2: L1-only (no L2 factor synthesis, no semantic crossover)
        l1_engine = StockRSIEngine(
            db_path=":memory:",
            backtest_config=bt_cfg,
            enable_crossover=False,
        )
        l1_charter = RSICharter(
            name=f"bench_l1_{strat.meta.id}",
            seed_strategy_id=strat.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=max(1, rounds),
            max_candidates_per_round=2,
            max_trials_budget=max(10, rounds * 8 + 4),
            min_signals=min_sigs,
            locked_holdout_pct=locked_holdout_pct,
            require_null_control=True,
            null_permutations=20,
        )
        l1_res = l1_engine.run(
            strat,
            market_data,
            charter=l1_charter,
            persist_champion=False,
            seed=seed + idx,
        )
        l1_arm = _metrics_from_rsi_result(
            "l1_only",
            l1_res,
            min_signals=l1_charter.min_signals,
            max_drawdown=l1_charter.max_drawdown,
        )

        # Arm 3: Full L1 + L2 Co-STEER + Semantic Crossover
        full_engine = StockRSIEngine(
            db_path=":memory:",
            backtest_config=bt_cfg,
            enable_crossover=True,
        )
        full_charter = RSICharter(
            name=f"bench_full_{strat.meta.id}",
            seed_strategy_id=strat.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
            max_rounds=max(1, rounds),
            max_candidates_per_round=2,
            max_trials_budget=max(10, rounds * 8 + 4),
            min_signals=min_sigs,
            locked_holdout_pct=locked_holdout_pct,
            require_null_control=True,
            null_permutations=20,
        )
        full_res = full_engine.run(
            strat,
            market_data,
            charter=full_charter,
            persist_champion=False,
            seed=seed + idx,
        )
        full_arm = _metrics_from_rsi_result(
            "l1_l2_crossover",
            full_res,
            min_signals=full_charter.min_signals,
            max_drawdown=full_charter.max_drawdown,
        )
        l2_factor_names.update(full_res.persisted_factors)
        total_xover_kept += full_res.crossovers_kept

        rows.append(
            BenchmarkStrategyRow(
                strategy_id=strat.meta.id,
                baseline=baseline_arm,
                l1_only=l1_arm,
                l1_l2_crossover=full_arm,
                fitness_gain_l1=round(l1_arm.fitness_score - baseline_arm.fitness_score, 4),
                fitness_gain_full=round(full_arm.fitness_score - baseline_arm.fitness_score, 4),
            )
        )

    null_passed: bool | None = None
    null_fp = 0
    if run_null_audit:

        def _run_null_world(seed_strat: Strategy, null_market: dict[str, pd.DataFrame]) -> int:
            eng = StockRSIEngine(
                db_path=":memory:",
                backtest_config=bt_cfg,
                enable_crossover=True,
            )
            ch = RSICharter(
                name="bench_null_audit",
                seed_strategy_id=seed_strat.meta.id,
                allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
                max_rounds=max(1, min(2, rounds)),
                max_candidates_per_round=2,
                min_signals=min_sigs,
                require_null_control=True,
                null_permutations=25,
            )
            return eng.run(seed_strat, null_market, charter=ch, persist_champion=False).promotions

        audit = NullSignalController().verify_zero_promotion_under_shuffled_returns(
            strategies[0],
            market_data,
            _run_null_world,
            n_worlds=max(1, null_worlds),
            seed=seed,
        )
        null_passed = bool(audit.passed)
        null_fp = int(audit.false_promotions)

    kernel_after = compute_kernel_fingerprint()
    return EvolutionBenchmarkReport(
        data_source=data_source,
        symbols=sorted(market_data.keys()),
        rounds=max(1, rounds),
        locked_holdout_pct=locked_holdout_pct,
        rows=rows,
        mean_baseline_fitness=round(mean(r.baseline.fitness_score for r in rows), 4),
        mean_l1_fitness=round(mean(r.l1_only.fitness_score for r in rows), 4),
        mean_full_fitness=round(mean(r.l1_l2_crossover.fitness_score for r in rows), 4),
        total_l2_factors=len(l2_factor_names),
        total_crossovers_kept=total_xover_kept,
        null_audit_passed=null_passed,
        null_false_promotions=null_fp,
        kernel_intact=(kernel_before == kernel_after),
    )


def render_benchmark_markdown(report: EvolutionBenchmarkReport) -> str:
    """Render concise Markdown ablation scorecard for CLI, Web Studio, and reports."""
    null_badge = (
        f"**{'PASS' if report.null_audit_passed else 'FAIL'}** (false_promotions={report.null_false_promotions})"
        if report.null_audit_passed is not None
        else "SKIPPED"
    )
    lines = [
        "# AlphaEvo Self-Evolution Ablation Benchmark",
        "",
        f"- Source: `{report.data_source}` | Symbols: {len(report.symbols)} | Rounds/Arm: {report.rounds} | Locked Holdout: {report.locked_holdout_pct:.0%}",
        (
            f"- Mean Fitness: Baseline `{report.mean_baseline_fitness:.3f}` → "
            f"L1-only `{report.mean_l1_fitness:.3f}` → **L1+L2+Crossover `{report.mean_full_fitness:.3f}`**"
        ),
        (
            f"- L2 Factors Synthesized: **{report.total_l2_factors}** | "
            f"Semantic Crossovers Kept: **{report.total_crossovers_kept}** | "
            f"Tier-2 Null Audit: {null_badge} | Kernel Intact: **{report.kernel_intact}**"
        ),
        "",
        "| Strategy | Baseline Fit (Sig / AvgRet) | L1-Only Fit (Sig / AvgRet / Holdout) | L1+L2+XOver Fit (Sig / AvgRet / Holdout) | Niches |",
        "|----------|------------------------------|--------------------------------------|------------------------------------------|--------|",
    ]
    for r in report.rows:
        b = r.baseline
        l1 = r.l1_only
        f = r.l1_l2_crossover
        l1_ho = f"{l1.holdout_avg_return:+.2%}" if l1.holdout_avg_return is not None else "n/a"
        f_ho = f"{f.holdout_avg_return:+.2%}" if f.holdout_avg_return is not None else "n/a"
        lines.append(
            f"| `{r.strategy_id}` "
            f"| {b.fitness_score:.3f} ({b.signals} / {b.avg_return:+.2%}) "
            f"| {l1.fitness_score:.3f} ({l1.signals} / {l1.avg_return:+.2%} / {l1_ho}) "
            f"| **{f.fitness_score:.3f}** ({f.signals} / {f.avg_return:+.2%} / {f_ho}) "
            f"| {f.map_elites_cells} |"
        )
    lines += [
        "",
        "*Research & educational verification only — not investment advice or return guarantees.*",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "AblationArmMetrics",
    "BenchmarkStrategyRow",
    "EvolutionBenchmarkReport",
    "render_benchmark_markdown",
    "run_evolution_benchmark",
]
