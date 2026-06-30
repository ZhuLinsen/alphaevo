"""Evolution CLI commands — run, evolve, evolve-islands, evolve-curriculum."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from alphaevo.cli._helpers import _get_config

if TYPE_CHECKING:
    from alphaevo.models.execution import EvaluationReport
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
OptimizationEvaluationMode = Literal["fast", "full"]

logger = logging.getLogger(__name__)
console = Console()


class _OptimizationCandidateLike(Protocol):
    candidate_id: str
    strategy: Strategy
    evaluation: EvaluationReport
    passed_gate: bool


def _write_strategy_yaml_artifacts(
    result: object,
    *,
    strategy_id: str,
    out_path: Path,
) -> tuple[Path | None, Path | None, int]:
    """Export per-round strategy YAMLs plus a champion shortcut file."""
    from alphaevo.strategy.dsl.serializer import StrategySerializer

    serializer = StrategySerializer()
    strategy_dir: Path | None = None
    champion_path: Path | None = None
    champion_id = getattr(result, "champion_id", None)
    written: set[str] = set()

    for round_result in getattr(result, "rounds", []):
        strategy = getattr(round_result, "strategy", None)
        meta = getattr(strategy, "meta", None)
        round_strategy_id = getattr(meta, "id", None)
        if strategy is None or not round_strategy_id or round_strategy_id in written:
            continue
        if strategy_dir is None:
            strategy_dir = out_path / f"{strategy_id}_strategies"
            strategy_dir.mkdir(parents=True, exist_ok=True)
        try:
            serializer.to_file(strategy, strategy_dir / f"{round_strategy_id}.yaml")
        except Exception:
            continue
        written.add(round_strategy_id)
        if round_strategy_id == champion_id:
            champion_path = out_path / f"{strategy_id}_champion.yaml"
            try:
                serializer.to_file(strategy, champion_path)
            except Exception:
                champion_path = None

    return strategy_dir, champion_path, len(written)


def _format_applied_change_line(
    change_type: str, target: str, from_value: object, to_value: object
) -> Text:
    """Render a mutation line without Rich eating target selectors."""
    from alphaevo.reflection.mutator import _sanitize_condition_value

    line = Text("    • ")
    line.append(change_type)
    line.append(" on ")
    line.append(str(target), style="dim")
    line.append(" (")
    line.append(str(_sanitize_condition_value(from_value)))
    line.append(" -> ")
    line.append(str(_sanitize_condition_value(to_value)))
    line.append(")")
    return line


def run_command(
    strategy_id: str = typer.Argument(..., help="Strategy ID to run"),
    samples: int = typer.Option(60, "--samples", "-n", help="Max stock samples"),
    start_date: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    report_dir: str = typer.Option("reports", "--output", "-o", help="Report output directory"),
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        help="Data adapter override; core: yfinance/akshare, optional bridge: dsa",
    ),
    sampling: str | None = typer.Option(None, "--sampling", help="Sampling method"),
    wf_folds: int | None = typer.Option(None, "--wf-folds", help="Walk-forward folds"),
    wf_train_pct: float | None = typer.Option(None, "--wf-train-pct", help="Walk-forward train %"),
    wf_pass_gap: float | None = typer.Option(None, "--wf-pass-gap", help="Walk-forward pass gap"),
    fill_policy: str | None = typer.Option(
        None,
        "--fill-policy",
        help="Intrabar stop/take-profit conflict policy: conservative/optimistic/close_first",
    ),
    stress_window_days: int | None = typer.Option(
        None, "--stress-window-days", help="Stress window days"
    ),
    stress_window_top_k: int | None = typer.Option(
        None, "--stress-window-top-k", help="Stress window top-k"
    ),
) -> None:
    """Run the full research loop: sample → backtest → evaluate → report."""
    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    bt_overrides: dict = {}
    if wf_folds is not None:
        bt_overrides["walk_forward_folds"] = wf_folds
    if wf_train_pct is not None:
        bt_overrides["walk_forward_train_pct"] = wf_train_pct
    if wf_pass_gap is not None:
        bt_overrides["walk_forward_pass_gap"] = wf_pass_gap
    if fill_policy is not None:
        bt_overrides["fill_policy"] = fill_policy
    if stress_window_days is not None:
        bt_overrides["stress_window_days"] = stress_window_days
    if stress_window_top_k is not None:
        bt_overrides["stress_window_top_k"] = stress_window_top_k
    if bt_overrides:
        overrides["backtest"] = bt_overrides

    config = _get_config(overrides or None)

    from alphaevo.models.enums import SamplingMethod

    sampling_method: SamplingMethod | None = None
    if sampling:
        try:
            sampling_method = SamplingMethod(sampling)
        except ValueError:
            console.print(f"[red]Unknown sampling method: {sampling}[/red]")
            raise typer.Exit(1) from None

    dr: tuple[date, date] | None = None
    if start_date:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else date.today()
        if s > e:
            console.print("[red]Error: --start date must be before --end date[/red]")
            raise typer.Exit(1)
        dr = (s, e)
    elif end_date:
        console.print("[red]Error: --end requires --start[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"🚀 Running research loop for [cyan]{strategy_id}[/cyan]",
            style="bold green",
        )
    )

    from alphaevo.orchestrator.pipeline import RunPipeline

    pipeline = RunPipeline(config)
    pipeline.ensure_builtin_strategies()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=None)

        def on_progress(msg: str) -> None:
            progress.update(task, description=msg)

        run_kwargs: dict = {
            "max_symbols": samples,
            "date_range": dr,
            "report_dir": Path(report_dir),
            "on_progress": on_progress,
        }
        if sampling_method is not None:
            run_kwargs["sampling_method"] = sampling_method

        try:
            result = asyncio.run(pipeline.run(strategy_id, **run_kwargs))
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from None
        except Exception as e:
            logger.exception("Pipeline failed")
            console.print(f"[red]Pipeline failed: {e}[/red]")
            console.print("[dim]Set ALPHAEVO_LOG_LEVEL=DEBUG for full traceback[/dim]")
            raise typer.Exit(1) from None

    ev = result.evaluation
    table = Table(title=f"📊 Results: {strategy_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Win Rate", f"{ev.overall.win_rate:.1%}")
    table.add_row("Avg Return", f"{ev.overall.avg_return:.2%}")
    table.add_row("P/L Ratio", f"{ev.overall.profit_loss_ratio:.2f}")
    table.add_row("Max Drawdown", f"{ev.overall.max_drawdown:.1%}")
    table.add_row("Sharpe Ratio", f"{ev.overall.sharpe_ratio:.2f}")
    table.add_row("Total Signals", str(ev.overall.signal_count))
    table.add_row("Avg Holding Days", f"{ev.overall.avg_holding_days:.1f}")
    table.add_row("Total Return", f"{ev.overall.total_return:.2%}")
    table.add_row("Confidence Score", f"[bold green]{ev.confidence_score:.1%}[/bold green]")
    console.print(table)
    _print_data_quality_summary(getattr(result, "data_quality", None))

    # Equity curve and charts (plotext)
    if ev.overall.signal_count > 0:
        from alphaevo.evaluator.reporter import Reporter

        equity_chart = Reporter.plot_equity_curve(
            result.backtest_result.signals,
            title=f"📈 Equity Curve — {strategy_id}",
        )
        console.print(Text.from_ansi(f"\n{equity_chart}"))

        dist_chart = Reporter.plot_return_distribution(
            result.backtest_result.signals,
            title=f"📊 Return Distribution — {strategy_id}",
        )
        console.print(Text.from_ansi(f"\n{dist_chart}"))

        dd_chart = Reporter.plot_drawdown_curve(
            result.backtest_result.signals,
            title=f"📉 Drawdown — {strategy_id}",
        )
        console.print(Text.from_ansi(f"\n{dd_chart}"))

    # Portfolio-level simulation
    if ev.overall.signal_count > 0:
        try:
            from alphaevo.backtest.portfolio import PortfolioBacktester, PortfolioConfig

            pf = PortfolioBacktester(PortfolioConfig())
            pf_result = pf.simulate(result.backtest_result)
            if pf_result.total_trades > 0:
                pf_table = Table(title="💼 Portfolio Simulation ($100K, max 5 positions)")
                pf_table.add_column("Metric", style="cyan")
                pf_table.add_column("Value", justify="right")
                pf_table.add_row("Final Equity", f"${pf_result.final_equity:,.0f}")
                pf_table.add_row("Total Return", f"{pf_result.total_return:.2%}")
                pf_table.add_row("Max Drawdown", f"{pf_result.max_drawdown:.1%}")
                pf_table.add_row("Sharpe Ratio", f"{pf_result.sharpe_ratio:.2f}")
                pf_table.add_row("Max Concurrent", str(pf_result.max_concurrent_positions))
                pf_table.add_row("Capital Utilization", f"{pf_result.capital_utilization:.0%}")
                pf_table.add_row("Win Rate", f"{pf_result.win_rate:.1%}")
                console.print(pf_table)
        except Exception:
            pass  # Portfolio simulation is optional

    if ev.overall.signal_count == 0:
        console.print(
            "\n[yellow]⚠ No trades fired in this run.[/yellow] "
            "That usually means the strategy was too strict for the sampled window,"
            " not that the backtest crashed."
        )
        console.print(
            f"[dim]Adapter: {config.data.adapter} | "
            f"Strategy market: {result.strategy.meta.market.value} | "
            f"Sampled symbols: {len(result.batch.symbols)}[/dim]"
        )
        if result.strategy.meta.market.value == "a_share" and config.data.adapter == "yfinance":
            console.print(
                "[dim]Hint: A-share strategies usually work better with "
                "`--adapter akshare` than the default yfinance adapter.[/dim]"
            )
        console.print(
            "[dim]Hint: try a longer date range, `--sampling strategy_scoped`, "
            "or evolve the strategy to relax strict thresholds.[/dim]"
        )

    if result.report_path:
        console.print(f"\n📄 Report saved to: {result.report_path}")


def _print_data_quality_summary(data_quality: object) -> None:
    """Render run-level data-quality findings for CLI users."""
    if data_quality is None:
        return
    findings = getattr(data_quality, "findings", [])
    if not findings:
        return

    risk = getattr(data_quality, "risk_level", "low")
    style = "red" if risk == "high" else "yellow"
    table = Table(title="Data Quality Gate")
    table.add_column("Severity", style=style)
    table.add_column("Category", style="cyan")
    table.add_column("Finding")
    for finding in findings[:6]:
        evidence = getattr(finding, "evidence", [])
        message = getattr(finding, "message", "")
        if evidence:
            message = f"{message} ({'; '.join(str(item) for item in evidence[:2])})"
        table.add_row(
            str(getattr(finding, "severity", "")),
            str(getattr(finding, "category", "")),
            message,
        )
    console.print(table)

    if getattr(data_quality, "blocks_strategy_iteration", False):
        console.print(
            "[yellow]Data quality gate would pause strategy mutation until provider/proxy "
            "coverage is diagnosed.[/yellow]"
        )


def _split_optimize_spaces(
    raw_spaces: list[str] | None,
) -> tuple[list[str] | None, list[str] | None, bool]:
    """Split CLI optimize spaces between exit-grid and DSL-param optimizers."""
    if raw_spaces is None:
        return None, None, True

    exit_spaces: list[str] = []
    param_spaces: list[str] = []
    run_param = False
    param_all = False
    exit_all = False

    for space in raw_spaces:
        key = space.strip().lower().replace("-", "_")
        if key == "all":
            exit_all = True
            param_all = True
            run_param = True
            continue
        param_space = _cli_param_space(key)
        if param_space is not None:
            run_param = True
            if param_space == "all":
                param_all = True
            elif param_space not in param_spaces:
                param_spaces.append(param_space)
            continue
        exit_spaces.append(space)

    return (
        None if exit_all else exit_spaces,
        None if param_all else param_spaces,
        run_param,
    )


def _cli_param_space(key: str) -> str | None:
    aliases = {
        "param": "all",
        "params": "all",
        "parameter": "all",
        "parameters": "all",
        "tunable": "all",
        "tunables": "all",
        "entry": "entry",
        "buy": "entry",
        "buy_signal": "entry",
        "buytrigger": "entry",
        "trigger": "entry",
        "triggers": "entry",
        "guard": "entry",
        "guards": "entry",
        "filter": "entry",
        "filters": "entry",
        "indicator": "indicator",
        "period": "indicator",
        "window": "indicator",
        "lookback": "indicator",
    }
    return aliases.get(key)


def optimize_command(
    strategy_id: str = typer.Argument(..., help="Strategy ID to optimize"),
    samples: int = typer.Option(60, "--samples", "-n", help="Max stock samples"),
    start_date: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    report_dir: str = typer.Option("reports", "--output", "-o", help="Report output directory"),
    adapter: str | None = typer.Option(None, "--adapter", help="Data adapter override"),
    sampling: str | None = typer.Option(None, "--sampling", help="Sampling method"),
    fill_policy: str | None = typer.Option(
        None,
        "--fill-policy",
        help="Intrabar stop/take-profit conflict policy: conservative/optimistic/close_first",
    ),
    spaces: str | None = typer.Option(
        None,
        "--spaces",
        help="Comma-separated spaces: entry,params,indicator,exit,stoploss,takeprofit,holding,all",
    ),
    max_candidates: int = typer.Option(
        128,
        "--max-candidates",
        help="Maximum candidates to evaluate",
    ),
    objective: str = typer.Option(
        "confidence",
        "--objective",
        help=(
            "Ranking objective: "
            "confidence/win_rate/avg_return/drawdown/quality/profit_quality/"
            "robust_profit_quality"
        ),
    ),
    min_win_rate: float | None = typer.Option(
        None,
        "--min-win-rate",
        help="Minimum win rate gate, e.g. 0.50",
    ),
    min_avg_return: float | None = typer.Option(
        None,
        "--min-avg-return",
        help="Minimum average return gate, e.g. 0.0 to reject negative expectancy",
    ),
    min_total_return: float | None = typer.Option(
        None,
        "--min-total-return",
        help="Minimum compounded total return gate, e.g. 0.10",
    ),
    min_profit_loss_ratio: float | None = typer.Option(
        None,
        "--min-profit-loss-ratio",
        help="Minimum profit/loss ratio gate, e.g. 1.0",
    ),
    max_drawdown: float | None = typer.Option(
        None,
        "--max-drawdown",
        help="Maximum drawdown gate, e.g. 0.30",
    ),
    min_signals: int | None = typer.Option(
        None,
        "--min-signals",
        help="Minimum signal-count gate for qualified candidates",
    ),
    reject_overfit: bool = typer.Option(
        False,
        "--reject-overfit",
        help="Reject candidates flagged by train/val/test or sensitivity overfit checks",
    ),
    max_train_val_gap: float | None = typer.Option(
        None,
        "--max-train-val-gap",
        help="Maximum train-validation win-rate gap for full-evaluated candidates",
    ),
    max_val_test_gap: float | None = typer.Option(
        None,
        "--max-val-test-gap",
        help="Maximum validation-test win-rate gap for full-evaluated candidates",
    ),
    max_walk_forward_gap: float | None = typer.Option(
        None,
        "--max-walk-forward-gap",
        help="Maximum mean walk-forward train-test gap for full-evaluated candidates",
    ),
    min_walk_forward_pass_rate: float | None = typer.Option(
        None,
        "--min-walk-forward-pass-rate",
        help="Minimum walk-forward fold pass rate for full-evaluated candidates",
    ),
    max_values_per_param: int = typer.Option(
        5,
        "--max-values-per-param",
        help="Maximum tested values per tunable parameter",
    ),
    param_max_changes: int = typer.Option(
        1,
        "--param-max-changes",
        help="Maximum params.tunable changes per parameter candidate",
    ),
    evaluation_mode: str = typer.Option(
        "fast",
        "--evaluation-mode",
        help="Candidate evaluation mode: fast/full",
    ),
    full_eval_top: int = typer.Option(
        5,
        "--full-eval-top",
        help="With fast mode, fully re-evaluate the top N candidates",
    ),
    parallel_workers: int = typer.Option(
        1,
        "--parallel-workers",
        help="Parallel candidate-evaluation workers for optimization search",
    ),
    joint_top: int = typer.Option(
        0,
        "--joint-top",
        help="Refine top N parameter candidates with exit/risk search",
    ),
    joint_candidates_per_seed: int = typer.Option(
        64,
        "--joint-candidates-per-seed",
        help="Maximum exit/risk candidates per parameter seed in joint search",
    ),
    save_best: bool = typer.Option(
        False,
        "--save-best",
        help="Save the best candidate strategy into the strategy store",
    ),
) -> None:
    """Optimize executable entry parameters plus exit/risk rules."""
    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    if fill_policy is not None:
        overrides.setdefault("backtest", {})["fill_policy"] = fill_policy
    config = _get_config(overrides or None)

    from alphaevo.models.enums import SamplingMethod

    sampling_method: SamplingMethod | None = None
    if sampling:
        try:
            sampling_method = SamplingMethod(sampling)
        except ValueError:
            console.print(f"[red]Unknown sampling method: {sampling}[/red]")
            raise typer.Exit(1) from None

    dr: tuple[date, date] | None = None
    if start_date:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else date.today()
        if s > e:
            console.print("[red]Error: --start date must be before --end date[/red]")
            raise typer.Exit(1)
        dr = (s, e)
    elif end_date:
        console.print("[red]Error: --end requires --start[/red]")
        raise typer.Exit(1)

    raw_spaces: list[str] | None = None
    if spaces:
        raw_spaces = [part.strip() for part in spaces.split(",") if part.strip()]
    exit_spaces, param_spaces, run_param_optimizer = _split_optimize_spaces(raw_spaces)
    run_exit_optimizer = exit_spaces is None or bool(exit_spaces)
    objective_value = cast("OptimizationObjective", objective)
    evaluation_mode_value = cast("OptimizationEvaluationMode", evaluation_mode)
    gates_configured = any(
        value is not None
        for value in (
            min_win_rate,
            min_avg_return,
            min_total_return,
            min_profit_loss_ratio,
            max_drawdown,
            min_signals,
            reject_overfit,
            max_train_val_gap,
            max_val_test_gap,
            max_walk_forward_gap,
            min_walk_forward_pass_rate,
        )
    )

    if not run_exit_optimizer and not run_param_optimizer:
        console.print("[red]No optimization spaces selected[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"🧪 Optimizing strategy parameters for [cyan]{strategy_id}[/cyan]",
            style="bold green",
        )
    )

    from alphaevo.optimizer import (
        ExitOptimizer,
        JointOptimizer,
        ParamOptimizer,
        export_best_param_strategy,
        export_best_strategy,
        render_exit_optimization_report,
        render_param_optimization_report,
    )
    from alphaevo.orchestrator.pipeline import RunPipeline

    pipeline = RunPipeline(config)
    pipeline.ensure_builtin_strategies()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Preparing baseline data...", total=None)

        def on_progress(msg: str) -> None:
            progress.update(task, description=msg)

        run_kwargs: dict = {
            "max_symbols": samples,
            "date_range": dr,
            "report_dir": None,
            "on_progress": on_progress,
        }
        if sampling_method is not None:
            run_kwargs["sampling_method"] = sampling_method

        try:
            baseline = asyncio.run(pipeline.run(strategy_id, **run_kwargs))
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from None
        except Exception as e:
            logger.exception("Optimization baseline failed")
            console.print(f"[red]Optimization baseline failed: {e}[/red]")
            raise typer.Exit(1) from None

        exit_result = None
        joint_result = None
        if run_exit_optimizer:
            progress.update(task, description="Searching exit/risk candidates...")
            exit_optimizer = ExitOptimizer(
                slippage=config.backtest.slippage,
                commission=config.backtest.commission,
                min_data_days=config.backtest.min_data_days,
                fill_policy=config.backtest.fill_policy,
                backtest_config=config.backtest,
            )
            try:
                exit_result = exit_optimizer.optimize(
                    baseline.strategy,
                    baseline._data or {},
                    baseline.batch,
                    contexts=baseline._contexts,
                    spaces=exit_spaces,
                    max_candidates=max_candidates,
                    objective=objective_value,
                    evaluation_mode=evaluation_mode_value,
                    full_eval_top_n=full_eval_top,
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
                )
            except ValueError as e:
                console.print(f"[red]Exit optimization failed: {e}[/red]")
                raise typer.Exit(1) from None

        param_result = None
        if run_param_optimizer:
            progress.update(task, description="Searching tunable parameter candidates...")
            param_optimizer = ParamOptimizer(
                slippage=config.backtest.slippage,
                commission=config.backtest.commission,
                min_data_days=config.backtest.min_data_days,
                fill_policy=config.backtest.fill_policy,
                backtest_config=config.backtest,
            )
            try:
                param_result = param_optimizer.optimize(
                    baseline.strategy,
                    baseline._data or {},
                    baseline.batch,
                    contexts=baseline._contexts,
                    spaces=param_spaces,
                    max_candidates=max_candidates,
                    max_values_per_param=max_values_per_param,
                    max_changes=param_max_changes,
                    objective=objective_value,
                    evaluation_mode=evaluation_mode_value,
                    full_eval_top_n=full_eval_top,
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
                )
            except ValueError as e:
                console.print(f"[red]Parameter optimization failed: {e}[/red]")
                raise typer.Exit(1) from None

        if joint_top > 0 and run_exit_optimizer and param_result is not None:
            progress.update(task, description="Searching joint entry+exit candidates...")
            seed_candidates = _select_joint_seed_candidates(
                param_result.candidates,
                limit=joint_top,
                min_avg_return=min_avg_return,
                min_total_return=min_total_return,
                min_profit_loss_ratio=min_profit_loss_ratio,
                min_signals=min_signals,
            )
            seed_strategies = [candidate.strategy for candidate in seed_candidates]
            joint_optimizer = JointOptimizer(
                slippage=config.backtest.slippage,
                commission=config.backtest.commission,
                min_data_days=config.backtest.min_data_days,
                fill_policy=config.backtest.fill_policy,
                backtest_config=config.backtest,
            )
            try:
                joint_result = joint_optimizer.optimize(
                    strategy_id,
                    seed_strategies,
                    baseline._data or {},
                    baseline.batch,
                    contexts=baseline._contexts,
                    spaces=exit_spaces,
                    max_candidates_per_seed=joint_candidates_per_seed,
                    objective=objective_value,
                    evaluation_mode=evaluation_mode_value,
                    full_eval_top_n=full_eval_top,
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
                )
            except ValueError as e:
                console.print(f"[red]Joint optimization failed: {e}[/red]")
                raise typer.Exit(1) from None

    if exit_result is not None:
        table = Table(title=f"🏁 Exit Optimization: {strategy_id}")
        table.add_column("Rank", justify="right")
        table.add_column("Candidate", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg Return", justify="right")
        table.add_column("Avg Win", justify="right")
        table.add_column("Avg Loss", justify="right")
        table.add_column("Total Return", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Signals", justify="right")
        table.add_column("Gate", justify="center")
        table.add_column("Eval", justify="center")
        table.add_column("Changes")
        for idx, exit_candidate in enumerate(exit_result.candidates[:10], start=1):
            ev = exit_candidate.evaluation.overall
            table.add_row(
                str(idx),
                exit_candidate.candidate_id,
                f"{exit_candidate.evaluation.confidence_score:.1%}",
                f"{ev.win_rate:.1%}",
                f"{ev.avg_return:.2%}",
                f"{ev.avg_win_return:.2%}",
                f"{ev.avg_loss_return:.2%}",
                f"{ev.total_return:.2%}",
                f"{ev.profit_loss_ratio:.2f}",
                f"{ev.max_drawdown:.1%}",
                str(ev.signal_count),
                "PASS" if exit_candidate.passed_gate else "FAIL",
                exit_candidate.evaluation_mode,
                "; ".join(exit_candidate.changes),
            )
        console.print(table)
        if gates_configured and exit_result.qualified_count == 0:
            console.print(
                "[yellow]No exit/risk candidate passed the configured quality gates.[/yellow]"
            )

    if param_result is not None:
        table = Table(title=f"🎚 Parameter Optimization: {strategy_id}")
        table.add_column("Rank", justify="right")
        table.add_column("Candidate", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg Return", justify="right")
        table.add_column("Avg Win", justify="right")
        table.add_column("Avg Loss", justify="right")
        table.add_column("Total Return", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Signals", justify="right")
        table.add_column("Gate", justify="center")
        table.add_column("Eval", justify="center")
        table.add_column("Changes")
        for idx, param_candidate in enumerate(param_result.candidates[:10], start=1):
            ev = param_candidate.evaluation.overall
            table.add_row(
                str(idx),
                param_candidate.candidate_id,
                f"{param_candidate.evaluation.confidence_score:.1%}",
                f"{ev.win_rate:.1%}",
                f"{ev.avg_return:.2%}",
                f"{ev.avg_win_return:.2%}",
                f"{ev.avg_loss_return:.2%}",
                f"{ev.total_return:.2%}",
                f"{ev.profit_loss_ratio:.2f}",
                f"{ev.max_drawdown:.1%}",
                str(ev.signal_count),
                "PASS" if param_candidate.passed_gate else "FAIL",
                param_candidate.evaluation_mode,
                "; ".join(param_candidate.changes),
            )
        console.print(table)
        if gates_configured and param_result.qualified_count == 0:
            console.print(
                "[yellow]No tunable-parameter candidate passed the configured quality gates.[/yellow]"
            )

    if joint_result is not None:
        table = Table(title=f"🔁 Joint Entry+Exit Optimization: {strategy_id}")
        table.add_column("Rank", justify="right")
        table.add_column("Candidate", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg Return", justify="right")
        table.add_column("Avg Win", justify="right")
        table.add_column("Avg Loss", justify="right")
        table.add_column("Total Return", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Signals", justify="right")
        table.add_column("Gate", justify="center")
        table.add_column("Eval", justify="center")
        table.add_column("Changes")
        for idx, joint_candidate in enumerate(joint_result.candidates[:10], start=1):
            ev = joint_candidate.evaluation.overall
            table.add_row(
                str(idx),
                joint_candidate.candidate_id,
                f"{joint_candidate.evaluation.confidence_score:.1%}",
                f"{ev.win_rate:.1%}",
                f"{ev.avg_return:.2%}",
                f"{ev.avg_win_return:.2%}",
                f"{ev.avg_loss_return:.2%}",
                f"{ev.total_return:.2%}",
                f"{ev.profit_loss_ratio:.2f}",
                f"{ev.max_drawdown:.1%}",
                str(ev.signal_count),
                "PASS" if joint_candidate.passed_gate else "FAIL",
                joint_candidate.evaluation_mode,
                "; ".join(joint_candidate.changes),
            )
        console.print(table)
        if gates_configured and joint_result.qualified_count == 0:
            console.print(
                "[yellow]No joint entry+exit candidate passed the configured quality gates.[/yellow]"
            )

    out_path = Path(report_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if exit_result is not None:
        report_path = out_path / f"{strategy_id}_exit_optimization.md"
        report_path.write_text(render_exit_optimization_report(exit_result), encoding="utf-8")
        console.print(f"[green]✓[/green] Exit report saved to {report_path}")

        best_path = export_best_strategy(exit_result, out_path)
        if best_path is not None:
            console.print(f"[green]✓[/green] Best exit strategy YAML: {best_path}")

    if param_result is not None:
        report_path = out_path / f"{strategy_id}_param_optimization.md"
        report_path.write_text(render_param_optimization_report(param_result), encoding="utf-8")
        console.print(f"[green]✓[/green] Parameter report saved to {report_path}")

        best_path = export_best_param_strategy(param_result, out_path)
        if best_path is not None:
            console.print(f"[green]✓[/green] Best parameter strategy YAML: {best_path}")

    if joint_result is not None:
        report_path = out_path / f"{strategy_id}_joint_optimization.md"
        report_path.write_text(render_exit_optimization_report(joint_result), encoding="utf-8")
        console.print(f"[green]✓[/green] Joint report saved to {report_path}")

        best_path = export_best_strategy(joint_result, out_path)
        if best_path is not None:
            console.print(f"[green]✓[/green] Best joint strategy YAML: {best_path}")

    best_candidates = [
        candidate
        for candidate in (
            exit_result.best_candidate if exit_result is not None else None,
            param_result.best_candidate if param_result is not None else None,
            joint_result.best_candidate if joint_result is not None else None,
        )
        if candidate is not None
    ]
    best = (
        max(
            best_candidates,
            key=lambda candidate: _optimization_candidate_sort_key(candidate, objective_value),
        )
        if best_candidates
        else None
    )
    all_candidates = [
        candidate
        for result in (exit_result, param_result, joint_result)
        if result is not None
        for candidate in result.candidates
    ]
    from alphaevo.optimizer.summary import select_high_win_return_candidate

    showcase = select_high_win_return_candidate(all_candidates)
    if best is not None:
        _print_best_optimization_candidate(best, title="Best Ranked Strategy Candidate")
    if showcase is not None and (best is None or showcase.candidate_id != best.candidate_id):
        _print_best_optimization_candidate(
            showcase,
            title="Best High-Win/High-Return Candidate",
        )
    if save_best and best is not None and best.passed_gate:
        pipeline.store.save(best.strategy)
        console.print(f"[green]✓[/green] Saved best strategy: {best.candidate_id}")
    elif best is not None:
        if save_best:
            console.print(
                "[yellow]No qualified candidate passed the configured gates; nothing saved.[/yellow]"
            )


def _print_best_optimization_candidate(candidate: Any, *, title: str) -> None:
    from alphaevo.optimizer.summary import format_best_candidate_console

    border_style = "green" if candidate.passed_gate else "yellow"
    if not candidate.passed_gate:
        title += " (failed gates)"
    console.print(
        Panel(
            format_best_candidate_console(candidate),
            title=title,
            border_style=border_style,
        )
    )


def _optimization_candidate_sort_key(
    candidate: _OptimizationCandidateLike,
    objective: OptimizationObjective,
) -> tuple[float, float, float, float, float, float, float, float, float, int, float]:
    from alphaevo.optimizer.scoring import candidate_sort_key

    evaluation = candidate.evaluation
    return candidate_sort_key(
        evaluation,
        passed_gate=candidate.passed_gate,
        objective=objective,
    )


def _optimization_objective_value(
    candidate: _OptimizationCandidateLike,
    objective: OptimizationObjective,
) -> float:
    from alphaevo.optimizer.scoring import objective_value

    return objective_value(candidate.evaluation, objective)


def _select_joint_seed_candidates(
    candidates: Sequence[_OptimizationCandidateLike],
    *,
    limit: int,
    min_avg_return: float | None,
    min_total_return: float | None,
    min_profit_loss_ratio: float | None,
    min_signals: int | None,
) -> list[_OptimizationCandidateLike]:
    """Pick diverse parameter seeds for the joint entry+exit pass."""
    if limit <= 0:
        return []
    selected: list[_OptimizationCandidateLike] = []
    seen: set[str] = set()

    def eligible(candidate: _OptimizationCandidateLike) -> bool:
        ev = candidate.evaluation.overall
        if min_signals is not None and ev.signal_count < min_signals:
            return False
        if min_avg_return is not None and ev.avg_return < min_avg_return * 0.5:
            return False
        if min_total_return is not None and ev.total_return < min_total_return * 0.5:
            return False
        return not (
            min_profit_loss_ratio is not None
            and ev.profit_loss_ratio < max(0.8, min_profit_loss_ratio * 0.8)
        )

    def add(candidate: _OptimizationCandidateLike) -> None:
        if len(selected) >= limit or candidate.candidate_id in seen:
            return
        if not eligible(candidate):
            return
        selected.append(candidate)
        seen.add(candidate.candidate_id)

    ranked = list(candidates)
    for candidate in ranked[:1]:
        add(candidate)
    for key in (
        lambda candidate: candidate.evaluation.overall.win_rate,
        lambda candidate: candidate.evaluation.overall.avg_return,
        lambda candidate: candidate.evaluation.overall.total_return,
        lambda candidate: candidate.evaluation.overall.profit_loss_ratio,
    ):
        for candidate in sorted(ranked, key=key, reverse=True):
            before = len(selected)
            add(candidate)
            if len(selected) > before:
                break
    for candidate in ranked:
        add(candidate)
        if len(selected) >= limit:
            break
    if not selected:
        for candidate in ranked[:limit]:
            if candidate.candidate_id in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.candidate_id)
    return selected


def evolve_command(
    strategy_id: str = typer.Argument(..., help="Strategy ID to evolve"),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Evolution rounds"),
    method: str = typer.Option("hybrid", "--method", "-m", help="llm/param_search/hybrid"),
    samples: int = typer.Option(60, "--samples", "-n", help="Max stock samples per round"),
    start_date: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        help="Data adapter override; core: yfinance/akshare, optional bridge: dsa",
    ),
    model: str | None = typer.Option(None, "--model", help="LLM model override"),
    reflect_model: str | None = typer.Option(
        None, "--reflect-model", help="Reflection model override"
    ),
    sampling: str | None = typer.Option(None, "--sampling", help="Sampling method"),
    output_dir: str = typer.Option(
        "reports", "--output", "-o", help="Output directory for artifacts"
    ),
    wf_folds: int | None = typer.Option(None, "--wf-folds", help="Walk-forward folds"),
    wf_train_pct: float | None = typer.Option(None, "--wf-train-pct", help="Walk-forward train %"),
    wf_pass_gap: float | None = typer.Option(None, "--wf-pass-gap", help="Walk-forward pass gap"),
    fill_policy: str | None = typer.Option(
        None,
        "--fill-policy",
        help="Intrabar stop/take-profit conflict policy: conservative/optimistic/close_first",
    ),
    stress_window_days: int | None = typer.Option(
        None, "--stress-window-days", help="Stress window days"
    ),
    stress_window_top_k: int | None = typer.Option(
        None, "--stress-window-top-k", help="Stress window top-k"
    ),
) -> None:
    """Self-evolve a strategy through multi-round improvement."""
    from alphaevo.models.enums import EvolutionMethod, SamplingMethod
    from alphaevo.orchestrator.evolution import EvolutionPipeline

    try:
        evo_method = EvolutionMethod(method)
    except ValueError:
        console.print(f"[red]Unknown method: {method}. Use: llm, param_search, hybrid[/red]")
        raise typer.Exit(1) from None

    sampling_method: SamplingMethod | None = None
    if sampling:
        try:
            sampling_method = SamplingMethod(sampling)
        except ValueError:
            console.print(f"[red]Unknown sampling method: {sampling}[/red]")
            raise typer.Exit(1) from None

    dr: tuple[date, date] | None = None
    if start_date:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else date.today()
        if s > e:
            console.print("[red]Error: --start date must be before --end date[/red]")
            raise typer.Exit(1)
        dr = (s, e)
    elif end_date:
        console.print("[red]Error: --end requires --start[/red]")
        raise typer.Exit(1)

    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    llm_overrides: dict = {}
    if model:
        llm_overrides["model"] = model
    if reflect_model:
        llm_overrides["reflect_model"] = reflect_model
    if llm_overrides:
        overrides["llm"] = llm_overrides
    bt_overrides: dict = {}
    if wf_folds is not None:
        bt_overrides["walk_forward_folds"] = wf_folds
    if wf_train_pct is not None:
        bt_overrides["walk_forward_train_pct"] = wf_train_pct
    if wf_pass_gap is not None:
        bt_overrides["walk_forward_pass_gap"] = wf_pass_gap
    if fill_policy is not None:
        bt_overrides["fill_policy"] = fill_policy
    if stress_window_days is not None:
        bt_overrides["stress_window_days"] = stress_window_days
    if stress_window_top_k is not None:
        bt_overrides["stress_window_top_k"] = stress_window_top_k
    if bt_overrides:
        overrides["backtest"] = bt_overrides

    config = _get_config(overrides or None)

    console.print(
        Panel(
            f"🧬 Evolving [cyan]{strategy_id}[/cyan] for {rounds} rounds\n"
            f"Method: {evo_method.value} | Samples: {samples}",
            style="bold magenta",
        )
    )

    pipeline = EvolutionPipeline(config)
    log_lines: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting evolution...", total=None)

        def on_progress(msg: str) -> None:
            progress.update(task, description=msg)
            log_lines.append(msg)

        evolve_kwargs: dict = {
            "rounds": rounds,
            "method": evo_method,
            "max_symbols": samples,
            "date_range": dr,
            "on_progress": on_progress,
        }
        if sampling_method is not None:
            evolve_kwargs["sampling_method"] = sampling_method

        try:
            result = pipeline.evolve(strategy_id, **evolve_kwargs)
        except Exception as e:
            logger.exception("Evolution failed")
            console.print(f"[red]Evolution failed: {e}[/red]")
            console.print("[dim]Set ALPHAEVO_LOG_LEVEL=DEBUG for full traceback[/dim]")
            raise typer.Exit(1) from None

    # Summary table
    if result.rounds:
        table = Table(title=f"🧬 Evolution Results: {strategy_id}")
        table.add_column("Round", justify="right", style="bold")
        table.add_column("Strategy", style="cyan")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg Ret", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Signals", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Status")

        for r in result.rounds:
            ev = r.evaluation
            status = "[green]✓ improved[/green]" if r.improved else "[dim]—[/dim]"
            table.add_row(
                str(r.round_num),
                r.strategy.meta.id,
                f"{ev.overall.win_rate:.1%}",
                f"{ev.overall.avg_return:.2%}",
                f"{ev.overall.profit_loss_ratio:.2f}",
                f"{ev.overall.max_drawdown:.1%}",
                str(ev.overall.signal_count),
                f"{ev.confidence_score:.1%}",
                status,
            )
        console.print(table)

        # Evolution score progress chart
        round_scores = [(r.strategy.meta.id, r.evaluation.confidence_score) for r in result.rounds]
        from alphaevo.evaluator.reporter import Reporter

        evo_chart = Reporter.plot_evolution_scores(
            round_scores,
            title=f"🧬 Evolution Progress — {strategy_id}",
        )
        console.print(Text.from_ansi(f"\n{evo_chart}"))

        # Equity curve for champion
        champion_round = next(
            (r for r in result.rounds if r.strategy.meta.id == result.champion_id),
            result.rounds[-1],
        )
        if champion_round.evaluation.overall.signal_count > 0:
            bt = getattr(champion_round, "backtest_result", None)
            if bt is not None and bt.signals:
                eq_chart = Reporter.plot_equity_curve(
                    bt.signals,
                    title=f"📈 Champion Equity — {result.champion_id}",
                )
                console.print(Text.from_ansi(f"\n{eq_chart}"))
    else:
        console.print("[yellow]No successful evolution rounds were recorded.[/yellow]")

    # Per-round changes detail
    has_changes = any(r.reflection and r.reflection.proposed_changes for r in result.rounds)
    if has_changes:
        console.print("\n[bold]📝 Changes Applied Per Round[/bold]")
        for r in result.rounds:
            if r.reflection and r.reflection.proposed_changes:
                console.print(f"  [cyan]Round {r.round_num}[/cyan]:")
                for ch in r.reflection.proposed_changes:
                    console.print(
                        _format_applied_change_line(
                            ch.change_type.value,
                            ch.target,
                            ch.from_value,
                            ch.to_value,
                        )
                    )
                    if ch.reason:
                        console.print(f"      [dim]{ch.reason}[/dim]")

    # Anti-overfit warnings
    if result.rounds:
        last = result.rounds[-1].evaluation
        af = last.anti_overfit
        if af.train_val_gap > 0.10:
            console.print(
                f"\n[yellow]⚠ Overfit warning: train-val gap = "
                f"{af.train_val_gap:.1%} (threshold: 10%)[/yellow]"
            )
        if af.val_test_gap > 0.08:
            console.print(
                f"[yellow]⚠ Overfit warning: val-test gap = "
                f"{af.val_test_gap:.1%} (threshold: 8%)[/yellow]"
            )

    if evo_method != EvolutionMethod.PARAM_SEARCH:
        telemetry_rows = []
        for round_result in result.rounds:
            telemetry = getattr(getattr(round_result, "reflection", None), "llm_telemetry", None)
            if telemetry is None:
                continue
            calls = getattr(telemetry, "calls", []) or []
            timeout_failures = sum(
                1
                for call in calls
                if not getattr(call, "success", True)
                and any(
                    token in str(getattr(call, "error", "")).lower()
                    for token in ("timeout", "timed out")
                )
            )
            telemetry_rows.append(
                (
                    str(round_result.round_num),
                    getattr(telemetry, "path", ""),
                    str(len(calls)),
                    f"{getattr(telemetry, 'total_duration_ms', 0)} ms",
                    str(timeout_failures),
                )
            )

        if telemetry_rows:
            telemetry_table = Table(title="🧠 LLM Runtime Telemetry")
            telemetry_table.add_column("Round", justify="right", style="bold")
            telemetry_table.add_column("Path", style="cyan")
            telemetry_table.add_column("Calls", justify="right")
            telemetry_table.add_column("Total", justify="right")
            telemetry_table.add_column("Timeouts", justify="right")
            for row in telemetry_rows:
                telemetry_table.add_row(*row)
            console.print(telemetry_table)

    if result.rounds:
        console.print(
            f"\n🏆 Champion: [cyan]{result.champion_id}[/cyan] "
            f"(score: {result.champion_score:.1%}, "
            f"improvement: {result.improvement:+.1%})"
        )
        if result.champion_id:
            console.print(
                f"[dim]View champion DSL: alphaevo strategy show {result.champion_id}[/dim]"
            )
    if result.early_stopped:
        console.print(f"[yellow]⚠ Early stopped: {result.stop_reason}[/yellow]")

    # Write output artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    from alphaevo.evaluator.reporter import Reporter

    evo_md = Reporter.evolution_report(result)
    (out_path / f"{strategy_id}_evolution.md").write_text(evo_md, encoding="utf-8")
    research_md = Reporter.research_report(result)
    (out_path / f"{strategy_id}_research_report.md").write_text(research_md, encoding="utf-8")
    if evo_method != EvolutionMethod.PARAM_SEARCH:
        llm_md = Reporter.llm_evidence_report(result)
        (out_path / f"{strategy_id}_llm_evidence.md").write_text(llm_md, encoding="utf-8")

    if hasattr(pipeline, "research_log"):
        log_md = pipeline.research_log.to_markdown()
        (out_path / f"{strategy_id}_research_log.md").write_text(log_md, encoding="utf-8")

    strategy_dir, champion_yaml, written_count = _write_strategy_yaml_artifacts(
        result,
        strategy_id=strategy_id,
        out_path=out_path,
    )

    # Export trajectory data for training (JSONL + ShareGPT + preference pairs)
    _trajectory_files: list[str] = []
    trajectory = getattr(result, "trajectory", None)
    if trajectory is not None and trajectory.steps:
        from alphaevo.research_log.trajectory import (
            export_jsonl,
            export_preference_pairs,
            export_sharegpt,
        )

        traj_dir = out_path / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        try:
            jsonl_path = export_jsonl(trajectory, traj_dir / f"{strategy_id}_trajectory.jsonl")
            _trajectory_files.append(str(jsonl_path))
        except Exception as e:
            logger.debug("JSONL export failed: %s", e)
        try:
            sharegpt_path = export_sharegpt(trajectory, traj_dir / f"{strategy_id}_sharegpt.json")
            _trajectory_files.append(str(sharegpt_path))
        except Exception as e:
            logger.debug("ShareGPT export failed: %s", e)
        try:
            pref_path = export_preference_pairs(
                trajectory, traj_dir / f"{strategy_id}_preference.json"
            )
            _trajectory_files.append(str(pref_path))
        except Exception as e:
            logger.debug("Preference pairs export failed: %s", e)

    console.print(f"\n📄 Artifacts saved to: {out_path}")
    if strategy_dir is not None and written_count > 0:
        console.print(f"[dim]Strategy YAMLs: {strategy_dir}[/dim]")
    else:
        console.print("[dim]No strategy YAML artifacts were written.[/dim]")
    if champion_yaml is not None:
        console.print(f"[dim]Champion YAML: {champion_yaml}[/dim]")
    if _trajectory_files:
        console.print(
            f"[dim]Trajectory data: {len(_trajectory_files)} files in {out_path / 'trajectory'}[/dim]"
        )
    if result.early_stopped and not result.rounds:
        raise typer.Exit(1)


def evolve_islands_command(
    strategy_id: str = typer.Argument(..., help="Base strategy ID to evolve"),
    islands: int = typer.Option(3, "--islands", "-n", help="Number of parallel islands"),
    generations: int = typer.Option(3, "--generations", "-g", help="Number of generations"),
    rounds_per_gen: int = typer.Option(
        2, "--rounds-per-gen", help="Evolution rounds per island per generation"
    ),
    samples: int = typer.Option(60, "--samples", "-s", help="Max stocks per backtest"),
    start_date: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    adapter: str | None = typer.Option(None, "--adapter", help="Data adapter override"),
    model: str | None = typer.Option(None, "--model", help="LLM model override"),
) -> None:
    """Multi-island parallel evolution for diverse strategy exploration."""
    from alphaevo.orchestrator.islands import IslandEvolution

    dr: tuple[date, date] | None = None
    if start_date:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else date.today()
        dr = (s, e)

    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    if model:
        overrides.setdefault("llm", {})["model"] = model

    config = _get_config(overrides or None)

    console.print(
        Panel(
            f"🏝️ Island Evolution: [cyan]{strategy_id}[/cyan]\n"
            f"Islands: {islands} | Generations: {generations} | "
            f"Rounds/gen: {rounds_per_gen}",
            style="bold magenta",
        )
    )

    island_evo = IslandEvolution(
        config,
        n_islands=islands,
        rounds_per_generation=rounds_per_gen,
        generations=generations,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting island evolution...", total=None)

        try:
            result = island_evo.evolve(
                [strategy_id],
                max_symbols=samples,
                date_range=dr,
                on_progress=lambda msg: progress.update(task, description=msg),
            )
        except Exception as e:
            logger.exception("Island evolution failed")
            console.print(f"[red]Island evolution failed: {e}[/red]")
            console.print("[dim]Set ALPHAEVO_LOG_LEVEL=DEBUG for full traceback[/dim]")
            raise typer.Exit(1) from None

    # Summary table
    table = Table(title="🏝️ Island Evolution Results")
    table.add_column("Island", justify="right", style="bold")
    table.add_column("Champion", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Generation")

    for island in result.islands:
        table.add_row(
            str(island.island_id),
            island.champion_id or "—",
            f"{island.champion_score:.1%}",
            str(island.generation),
        )
    console.print(table)

    console.print(
        f"\n🏆 Global champion: [cyan]{result.global_champion_id}[/cyan] "
        f"(score: {result.global_champion_score:.1%})"
    )
    console.print(
        f"📊 Total evaluations: {result.total_evaluations} | "
        f"Migrations: {result.migrations} | "
        f"Diversity: {result.diversity_score:.2f}"
    )


def evolve_curriculum_command(
    strategy_id: str = typer.Argument(..., help="Strategy ID to train progressively"),
    adapter: str | None = typer.Option(None, "--adapter", help="Data adapter override"),
    model: str | None = typer.Option(None, "--model", help="LLM model override"),
) -> None:
    """Progressive difficulty evolution (easy -> medium -> hard -> reality)."""
    from alphaevo.orchestrator.curriculum import CurriculumEvolution

    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    if model:
        overrides.setdefault("llm", {})["model"] = model

    config = _get_config(overrides or None)

    console.print(
        Panel(
            f"📚 Curriculum Evolution: [cyan]{strategy_id}[/cyan]\n"
            "Stages: Easy → Medium → Hard → Reality",
            style="bold magenta",
        )
    )

    curriculum = CurriculumEvolution(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting curriculum evolution...", total=None)

        try:
            result = curriculum.evolve(
                strategy_id,
                on_progress=lambda msg: progress.update(task, description=msg),
            )
        except Exception as e:
            logger.exception("Curriculum evolution failed")
            console.print(f"[red]Curriculum evolution failed: {e}[/red]")
            console.print("[dim]Set ALPHAEVO_LOG_LEVEL=DEBUG for full traceback[/dim]")
            raise typer.Exit(1) from None

    # Stage results
    table = Table(title="📚 Curriculum Results")
    table.add_column("Stage", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Status")

    all_stages = ["easy", "medium", "hard", "reality"]
    for stage in all_stages:
        if stage in result.stage_scores:
            score = result.stage_scores[stage]
            status = "[green]✅ Graduated[/green]"
            table.add_row(stage.upper(), f"{score:.1%}", status)
        elif stage in result.stages_completed:
            table.add_row(stage.upper(), "—", "[red]❌ Failed[/red]")
        else:
            table.add_row(stage.upper(), "—", "[dim]⏭️ Skipped[/dim]")
    console.print(table)

    if result.graduated:
        console.print(
            f"\n🎓 Graduated all stages! Champion: [cyan]{result.champion_id}[/cyan] "
            f"(score: {result.champion_score:.1%})"
        )
    else:
        console.print(f"\n📊 Completed {len(result.stages_completed)}/4 stages")
        if result.champion_id:
            console.print(
                f"  Best so far: [cyan]{result.champion_id}[/cyan] "
                f"(score: {result.champion_score:.1%})"
            )
