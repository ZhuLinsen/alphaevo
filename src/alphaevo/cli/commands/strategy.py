"""Strategy CLI sub-commands."""

from __future__ import annotations

import asyncio
import difflib
from datetime import date
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from alphaevo.cli._helpers import _get_config, _get_store

strategy_app = typer.Typer(help="Strategy management commands")
console = Console()


@strategy_app.command("create")
def strategy_create(
    market: str = typer.Option("a_share", "--market", "-m", help="Market: a_share/us/hk"),
) -> None:
    """Interactive strategy creation from natural language (requires alphaevo[llm])."""
    console.print(Panel("🧬 Strategy Creator", style="bold cyan"))

    description = typer.prompt(
        "Describe your strategy idea",
        default="",
    )
    if not description:
        console.print("[yellow]No description provided.[/yellow]")
        raise typer.Exit(0)

    config = _get_config()
    try:
        from alphaevo.core.llm import LLMClient
        from alphaevo.strategy.generator import StrategyGenerator

        llm = LLMClient.from_config(config)
        gen = StrategyGenerator(llm)
    except Exception as e:
        console.print(f"[red]LLM setup failed: {e}[/red]")
        console.print("[dim]Install with: pip install alphaevo\\[llm][/dim]")
        raise typer.Exit(1) from None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Generating strategy...", total=None)
        try:
            strategy = gen.generate(description, market=market)
        except ValueError as e:
            console.print(f"[red]Generation failed: {e}[/red]")
            raise typer.Exit(1) from None
        except Exception as e:
            message = str(e).strip().splitlines()[0] if str(e).strip() else e.__class__.__name__
            console.print(f"[red]Generation failed: {message}[/red]")
            raise typer.Exit(1) from None

    store = _get_store(config)
    store.save(strategy)
    console.print(f"[green]✓[/green] Created: {strategy.meta.id} ({strategy.meta.name})")
    console.print(f"[dim]View with: alphaevo strategy show {strategy.meta.id}[/dim]")


@strategy_app.command("draft")
def strategy_draft(
    description: str = typer.Argument(..., help="Natural-language strategy idea"),
    market: str = typer.Option("a_share", "--market", "-m", help="Market: a_share/us/hk"),
    strategy_id: str | None = typer.Option(None, "--id", help="Strategy ID to use"),
    save: bool = typer.Option(False, "--save", help="Save the draft into the strategy store"),
    output: str | None = typer.Option(None, "--output", "-o", help="Write YAML to a file"),
) -> None:
    """Draft an executable long-only strategy DSL without calling an LLM."""
    from alphaevo.strategy import StrategyDraftBuilder, StrategySerializer

    try:
        strategy = StrategyDraftBuilder().from_text(
            description,
            market=market,
            strategy_id=strategy_id,
        )
    except ValueError as e:
        console.print(f"[red]Draft failed: {e}[/red]")
        raise typer.Exit(1) from None

    serializer = StrategySerializer()
    yaml_str = serializer.to_yaml(strategy)
    if output is not None:
        output_path = Path(output)
        serializer.to_file(strategy, output_path)
        console.print(f"[green]✓[/green] Wrote draft YAML: {output_path}")
    if save:
        store = _get_store(_get_config())
        store.save(strategy)
        console.print(f"[green]✓[/green] Saved draft: {strategy.meta.id}")

    console.print(Panel(Text(yaml_str), title=f"Draft: {strategy.meta.id}", style="cyan"))
    _print_strategy_next_steps(strategy.meta.id, saved=save, output=output)


@strategy_app.command("research")
def strategy_research(
    description: str = typer.Argument(..., help="Natural-language strategy idea"),
    market: str = typer.Option("a_share", "--market", "-m", help="Market: a_share/us/hk"),
    strategy_id: str | None = typer.Option(None, "--id", help="Strategy ID to use"),
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
    optimize_exits: bool = typer.Option(
        True,
        "--optimize-exits/--no-optimize-exits",
        help="Run exit/stop/take-profit optimization after the baseline backtest",
    ),
    optimize_params: bool = typer.Option(
        True,
        "--optimize-params/--no-optimize-params",
        help="Run params.tunable entry/indicator optimization after the baseline backtest",
    ),
    max_candidates: int = typer.Option(
        64,
        "--max-candidates",
        help="Maximum candidates per optimizer",
    ),
    save_best: bool = typer.Option(
        False,
        "--save-best",
        help="Save the best optimized candidate strategy",
    ),
) -> None:
    """Draft, save, backtest, and optionally optimize a plain-language strategy."""
    from alphaevo.strategy import StrategyDraftBuilder, StrategySerializer

    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    if fill_policy is not None:
        overrides.setdefault("backtest", {})["fill_policy"] = fill_policy
    config = _get_config(overrides or None)
    report_path = Path(report_dir)

    try:
        strategy = StrategyDraftBuilder().from_text(
            description,
            market=market,
            strategy_id=strategy_id,
        )
        date_range = _parse_date_range(start_date, end_date)
        sampling_method = _parse_sampling_method(sampling)
    except ValueError as e:
        console.print(f"[red]Research setup failed: {e}[/red]")
        raise typer.Exit(1) from None

    store = _get_store(config)
    store.save(strategy)
    console.print(f"[green]✓[/green] Drafted and saved strategy: {strategy.meta.id}")

    strategy_dir = report_path / f"{strategy.meta.id}_research"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    draft_path = strategy_dir / f"{strategy.meta.id}.yaml"
    StrategySerializer().to_file(strategy, draft_path)
    console.print(f"[green]✓[/green] Draft YAML: {draft_path}")

    from alphaevo.orchestrator.pipeline import RunPipeline

    pipeline = RunPipeline(config)
    pipeline.ensure_builtin_strategies()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Backtesting drafted strategy...", total=None)

        def on_progress(msg: str) -> None:
            progress.update(task, description=msg)

        run_kwargs: dict = {
            "max_symbols": samples,
            "date_range": date_range,
            "report_dir": strategy_dir,
            "on_progress": on_progress,
        }
        if sampling_method is not None:
            run_kwargs["sampling_method"] = sampling_method

        try:
            baseline = asyncio.run(pipeline.run(strategy.meta.id, **run_kwargs))
        except ValueError as e:
            console.print(f"[red]Backtest failed: {e}[/red]")
            raise typer.Exit(1) from None
        except Exception as e:
            console.print(f"[red]Backtest failed: {e}[/red]")
            raise typer.Exit(1) from None

        exit_optimization = None
        if optimize_exits:
            progress.update(task, description="Optimizing exit/risk candidates...")
            exit_optimization = _run_exit_optimization(
                baseline,
                config,
                max_candidates=max_candidates,
            )
        param_optimization = None
        if optimize_params:
            progress.update(task, description="Optimizing tunable parameter candidates...")
            param_optimization = _run_param_optimization(
                baseline,
                config,
                max_candidates=max_candidates,
            )

    _print_research_summary(strategy.meta.id, baseline)
    if baseline.report_path:
        console.print(f"[green]✓[/green] Baseline report: {baseline.report_path}")

    if exit_optimization is not None:
        from alphaevo.optimizer import export_best_strategy, render_exit_optimization_report

        _print_exit_optimization_summary(exit_optimization)
        opt_report = strategy_dir / f"{strategy.meta.id}_exit_optimization.md"
        opt_report.write_text(render_exit_optimization_report(exit_optimization), encoding="utf-8")
        console.print(f"[green]✓[/green] Exit optimization report: {opt_report}")
        best_path = export_best_strategy(exit_optimization, strategy_dir)
        if best_path is not None:
            console.print(f"[green]✓[/green] Best optimized YAML: {best_path}")

    if param_optimization is not None:
        from alphaevo.optimizer import export_best_param_strategy, render_param_optimization_report

        _print_param_optimization_summary(param_optimization)
        param_report = strategy_dir / f"{strategy.meta.id}_param_optimization.md"
        param_report.write_text(
            render_param_optimization_report(param_optimization), encoding="utf-8"
        )
        console.print(f"[green]✓[/green] Parameter optimization report: {param_report}")
        best_path = export_best_param_strategy(param_optimization, strategy_dir)
        if best_path is not None:
            console.print(f"[green]✓[/green] Best parameter YAML: {best_path}")

    best_optimization = _best_optimization_result(exit_optimization, param_optimization)
    best_candidate = getattr(best_optimization, "best_candidate", None)
    if save_best and best_candidate is not None:
        store.save(best_candidate.strategy)
        console.print(f"[green]✓[/green] Saved best strategy: {best_candidate.candidate_id}")

    from alphaevo.evaluator.advice import build_research_advice, render_research_advice

    advice = build_research_advice(
        strategy,
        baseline.evaluation,
        optimization=best_optimization,
        min_signal_count=config.evolution.min_signal_count,
    )
    _print_advice_summary(advice)
    advice_path = strategy_dir / f"{strategy.meta.id}_research_advice.md"
    advice_path.write_text(render_research_advice(advice), encoding="utf-8")
    console.print(f"[green]✓[/green] Research advice: {advice_path}")

    console.print(f'[dim]Revise with: alphaevo strategy revise {strategy.meta.id} "<改法>"[/dim]')


@strategy_app.command("revise")
def strategy_revise(
    strategy_id: str = typer.Argument(..., help="Existing strategy ID"),
    instruction: str = typer.Argument(..., help="Revision instruction"),
    new_id: str | None = typer.Option(None, "--id", help="New strategy ID to use"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save the revised strategy"),
    output: str | None = typer.Option(None, "--output", "-o", help="Write YAML to a file"),
) -> None:
    """Revise a stored strategy with bounded rule-based changes."""
    from alphaevo.strategy import StrategyDraftBuilder, StrategySerializer

    store = _get_store()
    strategy = store.get(strategy_id)
    if strategy is None:
        console.print(f"[red]Strategy not found: {strategy_id}[/red]")
        raise typer.Exit(1)

    try:
        revised = StrategyDraftBuilder().revise(strategy, instruction, strategy_id=new_id)
    except ValueError as e:
        console.print(f"[red]Revision failed: {e}[/red]")
        raise typer.Exit(1) from None

    serializer = StrategySerializer()
    yaml_str = serializer.to_yaml(revised)
    if output is not None:
        output_path = Path(output)
        serializer.to_file(revised, output_path)
        console.print(f"[green]✓[/green] Wrote revised YAML: {output_path}")
    if save:
        store.save(revised)
        console.print(f"[green]✓[/green] Saved revision: {revised.meta.id}")

    console.print(Panel(Text(yaml_str), title=f"Revision: {revised.meta.id}", style="cyan"))
    _print_strategy_next_steps(revised.meta.id, saved=save, output=output)


@strategy_app.command("improve")
def strategy_improve(
    strategy_id: str = typer.Argument(..., help="Existing strategy ID"),
    instruction: str = typer.Argument(..., help="Improvement instruction"),
    new_id: str | None = typer.Option(None, "--id", help="New strategy ID to use"),
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
    optimize_exits: bool = typer.Option(
        False,
        "--optimize-exits/--no-optimize-exits",
        help="Run exit/stop/take-profit optimization after the revised backtest",
    ),
    optimize_params: bool = typer.Option(
        False,
        "--optimize-params/--no-optimize-params",
        help="Run params.tunable entry/indicator optimization after the revised backtest",
    ),
    max_candidates: int = typer.Option(
        64,
        "--max-candidates",
        help="Maximum candidates per optimizer",
    ),
    save_best: bool = typer.Option(
        False,
        "--save-best",
        help="Save the best optimized candidate strategy",
    ),
) -> None:
    """Revise an existing strategy, backtest the revision, and write advice."""
    from alphaevo.strategy import StrategyDraftBuilder, StrategySerializer

    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    if fill_policy is not None:
        overrides.setdefault("backtest", {})["fill_policy"] = fill_policy
    config = _get_config(overrides or None)

    try:
        date_range = _parse_date_range(start_date, end_date)
        sampling_method = _parse_sampling_method(sampling)
    except ValueError as e:
        console.print(f"[red]Improve setup failed: {e}[/red]")
        raise typer.Exit(1) from None

    store = _get_store(config)
    base = store.get(strategy_id)
    if base is None:
        console.print(f"[red]Strategy not found: {strategy_id}[/red]")
        raise typer.Exit(1)

    prior_best_score = _best_stored_score(store, strategy_id)
    try:
        revised = StrategyDraftBuilder().revise(base, instruction, strategy_id=new_id)
    except ValueError as e:
        console.print(f"[red]Improve failed: {e}[/red]")
        raise typer.Exit(1) from None

    store.save(revised)
    console.print(f"[green]✓[/green] Saved revised strategy: {revised.meta.id}")

    improve_dir = Path(report_dir) / f"{revised.meta.id}_improve"
    improve_dir.mkdir(parents=True, exist_ok=True)
    revised_yaml = improve_dir / f"{revised.meta.id}.yaml"
    StrategySerializer().to_file(revised, revised_yaml)
    console.print(f"[green]✓[/green] Revised YAML: {revised_yaml}")

    from alphaevo.orchestrator.pipeline import RunPipeline

    pipeline = RunPipeline(config)
    pipeline.ensure_builtin_strategies()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Backtesting revised strategy...", total=None)

        def on_progress(msg: str) -> None:
            progress.update(task, description=msg)

        run_kwargs: dict = {
            "max_symbols": samples,
            "date_range": date_range,
            "report_dir": improve_dir,
            "on_progress": on_progress,
        }
        if sampling_method is not None:
            run_kwargs["sampling_method"] = sampling_method

        try:
            result = asyncio.run(pipeline.run(revised.meta.id, **run_kwargs))
        except ValueError as e:
            console.print(f"[red]Backtest failed: {e}[/red]")
            raise typer.Exit(1) from None
        except Exception as e:
            console.print(f"[red]Backtest failed: {e}[/red]")
            raise typer.Exit(1) from None

        exit_optimization = None
        if optimize_exits:
            progress.update(task, description="Optimizing exit/risk candidates...")
            exit_optimization = _run_exit_optimization(
                result,
                config,
                max_candidates=max_candidates,
            )
        param_optimization = None
        if optimize_params:
            progress.update(task, description="Optimizing tunable parameter candidates...")
            param_optimization = _run_param_optimization(
                result,
                config,
                max_candidates=max_candidates,
            )

    _print_research_summary(revised.meta.id, result)
    _print_improve_delta(prior_best_score, result.evaluation.confidence_score)
    if result.report_path:
        console.print(f"[green]✓[/green] Revised report: {result.report_path}")

    if exit_optimization is not None:
        from alphaevo.optimizer import export_best_strategy, render_exit_optimization_report

        _print_exit_optimization_summary(exit_optimization)
        opt_report = improve_dir / f"{revised.meta.id}_exit_optimization.md"
        opt_report.write_text(render_exit_optimization_report(exit_optimization), encoding="utf-8")
        console.print(f"[green]✓[/green] Exit optimization report: {opt_report}")
        best_path = export_best_strategy(exit_optimization, improve_dir)
        if best_path is not None:
            console.print(f"[green]✓[/green] Best optimized YAML: {best_path}")

    if param_optimization is not None:
        from alphaevo.optimizer import export_best_param_strategy, render_param_optimization_report

        _print_param_optimization_summary(param_optimization)
        param_report = improve_dir / f"{revised.meta.id}_param_optimization.md"
        param_report.write_text(
            render_param_optimization_report(param_optimization), encoding="utf-8"
        )
        console.print(f"[green]✓[/green] Parameter optimization report: {param_report}")
        best_path = export_best_param_strategy(param_optimization, improve_dir)
        if best_path is not None:
            console.print(f"[green]✓[/green] Best parameter YAML: {best_path}")

    best_optimization = _best_optimization_result(exit_optimization, param_optimization)
    best_candidate = getattr(best_optimization, "best_candidate", None)
    if save_best and best_candidate is not None:
        store.save(best_candidate.strategy)
        console.print(f"[green]✓[/green] Saved best strategy: {best_candidate.candidate_id}")

    from alphaevo.evaluator.advice import build_research_advice, render_research_advice

    advice = build_research_advice(
        revised,
        result.evaluation,
        optimization=best_optimization,
        min_signal_count=config.evolution.min_signal_count,
    )
    _print_advice_summary(advice)
    advice_path = improve_dir / f"{revised.meta.id}_research_advice.md"
    advice_path.write_text(render_research_advice(advice), encoding="utf-8")
    console.print(f"[green]✓[/green] Research advice: {advice_path}")


def _print_strategy_next_steps(
    strategy_id: str,
    *,
    saved: bool,
    output: str | None,
) -> None:
    if saved:
        console.print(f"[dim]Backtest with: alphaevo run {strategy_id}[/dim]")
        console.print(f"[dim]Optimize strategy with: alphaevo optimize {strategy_id}[/dim]")
    elif output is not None:
        console.print(f"[dim]Import with: alphaevo strategy import {output}[/dim]")
    else:
        console.print(
            "[dim]Use --save to backtest this draft directly, or --output to write YAML.[/dim]"
        )


def _parse_date_range(
    start_date: str | None,
    end_date: str | None,
) -> tuple[date, date] | None:
    if start_date:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date) if end_date else date.today()
        if start > end:
            raise ValueError("--start date must be before --end date")
        return (start, end)
    if end_date:
        raise ValueError("--end requires --start")
    return None


def _parse_sampling_method(sampling: str | None) -> Any | None:
    if not sampling:
        return None
    from alphaevo.models.enums import SamplingMethod

    try:
        return SamplingMethod(sampling)
    except ValueError as exc:
        raise ValueError(f"Unknown sampling method: {sampling}") from exc


def _run_exit_optimization(
    baseline: Any,
    config: Any,
    *,
    max_candidates: int,
) -> Any:
    from alphaevo.optimizer import ExitOptimizer

    optimizer = ExitOptimizer(
        slippage=config.backtest.slippage,
        commission=config.backtest.commission,
        min_data_days=config.backtest.min_data_days,
        fill_policy=config.backtest.fill_policy,
        backtest_config=config.backtest,
    )
    return optimizer.optimize(
        baseline.strategy,
        baseline._data or {},
        baseline.batch,
        contexts=baseline._contexts,
        max_candidates=max_candidates,
    )


def _run_param_optimization(
    baseline: Any,
    config: Any,
    *,
    max_candidates: int,
) -> Any:
    from alphaevo.optimizer import ParamOptimizer

    optimizer = ParamOptimizer(
        slippage=config.backtest.slippage,
        commission=config.backtest.commission,
        min_data_days=config.backtest.min_data_days,
        fill_policy=config.backtest.fill_policy,
        backtest_config=config.backtest,
    )
    return optimizer.optimize(
        baseline.strategy,
        baseline._data or {},
        baseline.batch,
        contexts=baseline._contexts,
        max_candidates=max_candidates,
    )


def _best_optimization_result(*optimizations: Any) -> Any:
    best_result = None
    best_score: float | None = None
    for optimization in optimizations:
        candidate = getattr(optimization, "best_candidate", None)
        evaluation = getattr(candidate, "evaluation", None)
        score = getattr(evaluation, "confidence_score", None)
        if not isinstance(score, (int, float)):
            continue
        numeric_score = float(score)
        if best_score is None or numeric_score > best_score:
            best_score = numeric_score
            best_result = optimization
    return best_result


def _best_stored_score(store: Any, strategy_id: str) -> float | None:
    evaluations = store.get_evaluations(strategy_id)
    if not evaluations:
        return None
    return max(float(ev.confidence_score) for ev in evaluations)


def _print_research_summary(strategy_id: str, result: Any) -> None:
    ev = result.evaluation
    table = Table(title=f"🔬 Research Summary: {strategy_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Win Rate", f"{ev.overall.win_rate:.1%}")
    table.add_row("Avg Return", f"{ev.overall.avg_return:.2%}")
    table.add_row("P/L Ratio", f"{ev.overall.profit_loss_ratio:.2f}")
    table.add_row("Max Drawdown", f"{ev.overall.max_drawdown:.1%}")
    table.add_row("Signals", str(ev.overall.signal_count))
    table.add_row("Confidence", f"[bold green]{ev.confidence_score:.1%}[/bold green]")
    console.print(table)


def _print_exit_optimization_summary(result: Any) -> None:
    table = Table(title="🏁 Exit Optimization Top Candidates")
    table.add_column("Rank", justify="right")
    table.add_column("Candidate", style="cyan")
    table.add_column("Confidence", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Return", justify="right")
    table.add_column("Changes")
    for idx, candidate in enumerate(result.candidates[:5], start=1):
        ev = candidate.evaluation.overall
        table.add_row(
            str(idx),
            candidate.candidate_id,
            f"{candidate.evaluation.confidence_score:.1%}",
            f"{ev.win_rate:.1%}",
            f"{ev.avg_return:.2%}",
            "; ".join(candidate.changes),
        )
    console.print(table)


def _print_param_optimization_summary(result: Any) -> None:
    table = Table(title="🎚 Parameter Optimization Top Candidates")
    table.add_column("Rank", justify="right")
    table.add_column("Candidate", style="cyan")
    table.add_column("Confidence", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Return", justify="right")
    table.add_column("Changes")
    for idx, candidate in enumerate(result.candidates[:5], start=1):
        ev = candidate.evaluation.overall
        table.add_row(
            str(idx),
            candidate.candidate_id,
            f"{candidate.evaluation.confidence_score:.1%}",
            f"{ev.win_rate:.1%}",
            f"{ev.avg_return:.2%}",
            "; ".join(candidate.changes),
        )
    console.print(table)


def _print_improve_delta(previous_score: float | None, new_score: float) -> None:
    if previous_score is None:
        return
    delta = new_score - previous_score
    style = "green" if delta >= 0 else "yellow"
    console.print(
        f"[{style}]Score delta vs stored parent best: "
        f"{previous_score:.1%} -> {new_score:.1%} ({delta:+.1%})[/{style}]"
    )


def _print_advice_summary(advice: Any) -> None:
    panel_lines = [f"[bold]{advice.status}[/bold]", advice.summary]
    if advice.recommendations:
        panel_lines.append("")
        for rec in advice.recommendations[:3]:
            line = f"- {rec.priority.upper()}: {rec.action}"
            if rec.command:
                line += f"\n  [dim]{rec.command}[/dim]"
            panel_lines.append(line)
    console.print(Panel("\n".join(panel_lines), title="Research Advice", style="cyan"))


@strategy_app.command("list")
def strategy_list() -> None:
    """List all strategies in the store."""
    store = _get_store()
    strategies = store.list_all()
    if not strategies:
        console.print("[yellow]No strategies found. Import some with:[/yellow]")
        console.print("  alphaevo strategy import <file.yaml>")
        console.print("  alphaevo demo  [dim]# loads builtin strategies[/dim]")
        return

    table = Table(title="📋 Strategies")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Ver", justify="right")
    table.add_column("Category")
    table.add_column("Market")
    table.add_column("Tags")
    for s in strategies:
        name = s.meta.name
        if s.meta.experimental:
            name = f"{name} ⚠️"
        table.add_row(
            s.meta.id,
            name,
            str(s.meta.version),
            s.meta.category.value if s.meta.category else "—",
            s.meta.market.value if s.meta.market else "—",
            ", ".join(s.meta.tags) if s.meta.tags else "—",
        )
    console.print(table)


@strategy_app.command("show")
def strategy_show(strategy_id: str) -> None:
    """Show strategy details."""
    store = _get_store()
    strategy = store.get(strategy_id)
    if strategy is None:
        console.print(f"[red]Strategy not found: {strategy_id}[/red]")
        raise typer.Exit(1)

    from alphaevo.strategy.dsl.serializer import StrategySerializer

    tags = ", ".join(strategy.meta.tags) if strategy.meta.tags else "—"
    parent = strategy.meta.parent_id or "—"
    experimental_note = ""
    if strategy.meta.experimental:
        experimental_note = "\n[bold yellow]⚠️  EXPERIMENTAL — uses proxy indicators, not real data feeds[/bold yellow]"
    console.print(
        Panel(
            f"[bold]{strategy.meta.name}[/bold]\n"
            f"ID: {strategy.meta.id} | Version: {strategy.meta.version}\n"
            f"Category: {strategy.meta.category.value} | Market: {strategy.meta.market.value}\n"
            f"Tags: {tags} | Parent: {parent}"
            f"{experimental_note}\n\n"
            f"[dim]{strategy.description}[/dim]",
            title="📋 Strategy Details",
            style="cyan",
        )
    )
    yaml_str = StrategySerializer().to_yaml(strategy)
    console.print(Panel(Text(yaml_str), title="YAML DSL", style="dim"))


@strategy_app.command("diff")
def strategy_diff(
    id1: str = typer.Argument(..., help="Base strategy ID"),
    id2: str = typer.Argument(..., help="Target strategy ID"),
) -> None:
    """Show structural and YAML diff between two strategies."""
    store = _get_store()
    s1 = store.get(id1)
    s2 = store.get(id2)
    if not s1 or not s2:
        missing = [i for i, s in [(id1, s1), (id2, s2)] if s is None]
        console.print(f"[red]Strategy not found: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    from alphaevo.cli.commands.analysis import _format_change_line, _strategy_diff_rows
    from alphaevo.strategy.dsl.serializer import StrategySerializer

    console.print(Panel(f"{id1} -> {id2}", title="🧩 Strategy Diff", style="cyan"))

    diff_rows = _strategy_diff_rows(s1, s2)
    if diff_rows:
        console.print("[bold]Changed Fields[/bold]")
        for path, v1, v2 in diff_rows:
            console.print(_format_change_line(path, v1, v2))
    else:
        console.print("[dim]No structural strategy differences beyond version metadata.[/dim]")

    serializer = StrategySerializer()
    yaml1 = serializer.to_yaml(s1).splitlines()
    yaml2 = serializer.to_yaml(s2).splitlines()
    diff_lines = list(difflib.unified_diff(yaml1, yaml2, fromfile=id1, tofile=id2, lineterm=""))
    if diff_lines:
        console.print("\n[bold]📄 YAML Diff[/bold]")
        console.print(Text("\n".join(diff_lines)))


@strategy_app.command("import")
def strategy_import(
    file: str = typer.Argument(..., help="Path to strategy YAML file"),
) -> None:
    """Import a strategy from YAML file."""
    path = Path(file)
    if not path.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    store = _get_store()
    try:
        strategy = store.import_from_file(path)
        console.print(f"[green]✓[/green] Imported: {strategy.meta.id} ({strategy.meta.name})")
    except Exception as e:
        console.print(f"[red]Import failed: {e}[/red]")
        raise typer.Exit(1) from None


@strategy_app.command("validate")
def strategy_validate(
    file: str = typer.Argument(..., help="Path to strategy YAML file"),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors"),
    bias_check: bool = typer.Option(
        False,
        "--bias-check",
        help="Also run heuristic lookahead/repainting checks",
    ),
) -> None:
    """Validate a strategy YAML file."""
    from alphaevo.strategy.dsl.parser import StrategyParseError, StrategyParser

    path = Path(file)
    if not path.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    parser = StrategyParser()
    try:
        strategy = parser.parse_file(path)
        diagnostics = parser.assert_valid(strategy, strict=strict)
        console.print(f"[green]✓[/green] Valid: {strategy.meta.id} ({strategy.meta.name})")
        for w in diagnostics.warnings:
            console.print(f"  [yellow]⚠ {w}[/yellow]")
        if bias_check:
            from alphaevo.strategy.bias import analyze_strategy_bias

            report = analyze_strategy_bias(strategy)
            console.print(f"[bold]Bias risk:[/bold] {report.risk_level}")
            for finding in report.findings:
                style = {
                    "error": "red",
                    "warning": "yellow",
                    "info": "blue",
                }[finding.severity]
                console.print(
                    f"  [{style}]{finding.severity.upper()} {finding.category}"
                    f" at {finding.location}: {finding.message}[/{style}]"
                )
            if report.has_errors:
                raise typer.Exit(1)
    except StrategyParseError as e:
        console.print(f"[red]✗ Invalid: {e}[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]✗ Invalid: {e}[/red]")
        raise typer.Exit(1) from None
