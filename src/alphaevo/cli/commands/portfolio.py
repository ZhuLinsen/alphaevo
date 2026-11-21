"""Portfolio allocation CLI command — causal regime-adaptive multi-strategy budgeting."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alphaevo.cli._helpers import _get_config

portfolio_app = typer.Typer(
    help="Multi-strategy regime-adaptive portfolio allocation & risk budgeting"
)
console = Console()


@portfolio_app.command("allocate")
def portfolio_allocate(
    strategy_ids: Annotated[
        list[str],
        typer.Argument(help="Strategy IDs or YAML paths to combine in a risk-budgeted portfolio"),
    ],
    live: bool = typer.Option(False, "--live", help="Fetch live market data instead of snapshot"),
    bundle: str | None = typer.Option(
        None, "--bundle", "-b", help="Evaluate against a verified immutable DataBundle ID"
    ),
    bundle_root: str = typer.Option(
        "reports/bundles", "--bundle-root", help="Root directory storing immutable bundles"
    ),
    target_vol: float = typer.Option(
        0.18, "--target-vol", help="Annualized portfolio volatility target"
    ),
    max_weight: float = typer.Option(
        0.60, "--max-weight", help="Maximum single-strategy weight cap"
    ),
    regime_haircut: float = typer.Option(
        0.35, "--regime-haircut", help="Multiplier applied when causal regime mismatches preference"
    ),
    dd_threshold: float = typer.Option(
        0.12, "--dd-threshold", help="Portfolio drawdown threshold triggering 50% deleveraging"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Optional path to write Markdown allocation report"
    ),
) -> None:
    """Compute causal regime-adaptive risk weights across multiple strategies."""
    from alphaevo.alpha_factory.factory import load_registered_factors
    from alphaevo.backtest.allocator import RegimeAdaptiveAllocator, RiskBudgetConfig
    from alphaevo.cli.commands.rsi import _load_market_data, _resolve_strategy

    config = _get_config()
    if str(config.db_path) != ":memory:" and Path(config.db_path).expanduser().is_file():
        load_registered_factors(config.db_path, status="active")

    strategies = []
    for ref in strategy_ids:
        try:
            strategies.append(_resolve_strategy(ref, config))
        except ValueError as exc:
            console.print(f"[red]Strategy resolution failed for '{ref}': {exc}[/red]")
            raise typer.Exit(1) from None

    if bundle is not None:
        from alphaevo.provenance.bundle import DataBundleStore

        try:
            market_data, _manifest = DataBundleStore(bundle_root).load_bundle(bundle, verify=True)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Data bundle error: {exc}[/red]")
            raise typer.Exit(1) from None
    else:
        market_data = _load_market_data(live=live)

    alloc_cfg = RiskBudgetConfig(
        target_annual_vol=target_vol,
        max_strategy_weight=max_weight,
        regime_mismatch_haircut=regime_haircut,
        drawdown_deleveraging_threshold=dd_threshold,
    )
    allocator = RegimeAdaptiveAllocator(alloc_cfg)
    report = allocator.allocate(strategies, market_data, backtest_config=config.backtest)

    console.print(
        Panel(
            f"[bold]Window:[/bold] {report.date_range[0]} to {report.date_range[1]} "
            f"({report.trading_days} days)\n"
            f"[bold]Combined Return:[/bold] [green]{report.combined_total_return:.2%}[/green] "
            f"(Equal-Weight: {report.equal_weight_total_return:.2%})\n"
            f"[bold]Combined Max DD:[/bold] {report.combined_max_drawdown:.1%} "
            f"(Equal-Weight: {report.equal_weight_max_drawdown:.1%})\n"
            f"[bold]Combined Sharpe:[/bold] {report.combined_sharpe_ratio:.2f} "
            f"(Equal-Weight: {report.equal_weight_sharpe_ratio:.2f})",
            title="⚖️ Regime-Adaptive Multi-Strategy Portfolio",
            border_style="cyan",
        )
    )

    table = Table(title="Strategy Risk Budget Summary")
    table.add_column("Strategy", style="cyan")
    table.add_column("Pref Regimes")
    table.add_column("Signals", justify="right")
    table.add_column("Mean Wt", justify="right")
    table.add_column("Max Wt", justify="right")
    table.add_column("Standalone Ret", justify="right")
    table.add_column("Standalone DD", justify="right")
    table.add_column("Contribution", justify="right", style="bold")
    for s in report.strategies:
        table.add_row(
            s.strategy_id,
            ",".join(s.preferred_regimes) if s.preferred_regimes else "any",
            str(s.signal_count),
            f"{s.mean_weight:.1%}",
            f"{s.max_weight:.1%}",
            f"{s.standalone_total_return:.2%}",
            f"{s.standalone_max_drawdown:.1%}",
            f"{s.return_contribution:+.2%}",
        )
    console.print(table)

    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(allocator.render_markdown(report), encoding="utf-8")
        console.print(f"[dim]Report written to {out_path}[/dim]")


__all__ = ["portfolio_app"]
