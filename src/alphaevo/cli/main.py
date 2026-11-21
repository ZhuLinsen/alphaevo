"""AlphaEvo CLI — Self-Evolving Stock Agent command line interface.

Thin assembler that registers sub-command modules.
"""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

# ── App ────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="alphaevo",
    help="🧬 AlphaEvo — Open-Source RSI (Recursive Self-Improvement) Engine for Stock Research",
    no_args_is_help=True,
)
console = Console()

# ── Shared helpers (kept importable for backward compat with tests) ───

from alphaevo.cli._helpers import _get_config, _get_store  # noqa: E402

# ── Sub-command registration ──────────────────────────────────────────
from alphaevo.cli.commands.analysis import (  # noqa: E402
    compare_command,
    leaderboard_command,
    tree_command,
)
from alphaevo.cli.commands.config import config_app  # noqa: E402
from alphaevo.cli.commands.data import data_app  # noqa: E402
from alphaevo.cli.commands.evolution import (  # noqa: E402
    evolve_command,
    evolve_curriculum_command,
    evolve_islands_command,
    optimize_command,
    run_command,
)
from alphaevo.cli.commands.factor import factor_app  # noqa: E402
from alphaevo.cli.commands.portfolio import portfolio_app  # noqa: E402
from alphaevo.cli.commands.rsi import rsi_app, shadow_app, truth_app  # noqa: E402
from alphaevo.cli.commands.runs import runs_app  # noqa: E402
from alphaevo.cli.commands.strategy import strategy_app  # noqa: E402

app.add_typer(strategy_app, name="strategy")
app.add_typer(factor_app, name="factor")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(rsi_app, name="rsi")
app.add_typer(rsi_app, name="autoresearch")
app.add_typer(shadow_app, name="shadow")
app.add_typer(truth_app, name="truth")
app.add_typer(runs_app, name="runs")
app.add_typer(portfolio_app, name="portfolio")

# Top-level commands delegated to command modules
app.command("run")(run_command)
app.command("optimize")(optimize_command)
app.command("evolve")(evolve_command)
app.command("evolve-islands")(evolve_islands_command)
app.command("evolve-curriculum")(evolve_curriculum_command)
app.command("leaderboard")(leaderboard_command)
app.command("compare")(compare_command)
app.command("tree")(tree_command)

# ── Init ───────────────────────────────────────────────────────────────

_CORE_DATA_ADAPTERS = ["auto", "tencent", "yfinance", "akshare"]
_OPTIONAL_DATA_BRIDGES = ["dsa"]
_KNOWN_DATA_ADAPTERS = [*_CORE_DATA_ADAPTERS, *_OPTIONAL_DATA_BRIDGES]
_LLM_MODELS = [
    "gemini/gemini-3.8-flash",
    "deepseek/deepseek-v4-flash",
    "openai/gpt-5-mini",
    "anthropic/claude-sonnet-5.5",
]


@app.command()
def init() -> None:
    """Interactive first-time setup — configure data source, LLM, and directories."""
    from alphaevo.core.config import ConfigManager

    console.print(Panel("🧬 AlphaEvo Setup", style="bold cyan"))

    mgr = ConfigManager()
    overrides: dict[str, Any] = {}

    # 1. Data adapter
    console.print("\n[bold]1. Data source[/bold]")
    console.print("  Core adapters: auto (default), tencent (A-share K-line), yfinance, akshare")
    console.print(
        "  Optional bridge: dsa [dim](daily_stock_analysis enhancement, requires ALPHAEVO_DSA_PATH)[/dim]"
    )
    adapter = typer.prompt("  Choose data adapter", default="auto")
    if adapter not in _KNOWN_DATA_ADAPTERS:
        console.print(f"[yellow]Warning: '{adapter}' is not a known adapter, using anyway[/yellow]")
    elif adapter in _OPTIONAL_DATA_BRIDGES:
        console.print(
            "[dim]Using dsa as an optional external bridge, not a core built-in data source.[/dim]"
        )
    overrides.setdefault("data", {})["adapter"] = adapter

    # 2. LLM model (optional)
    console.print(
        "\n[bold]2. LLM model[/bold]"
        " [dim](for strategy generation/evolution, skip if not needed)[/dim]"
    )
    console.print("  Suggestions: " + ", ".join(_LLM_MODELS))
    model = typer.prompt("  LLM model (enter to skip)", default="")
    if model:
        overrides.setdefault("llm", {})["model"] = model

    # 3. Save
    config = mgr.load(cli_overrides=overrides)
    mgr.ensure_dirs(config)
    mgr.save_user_config(config)

    # 4. Initialize database
    from alphaevo.strategy.store import StrategyStore

    StrategyStore(config.db_path)

    console.print(f"\n[green]✓[/green] Config saved to {mgr.USER_CONFIG_FILE}")
    console.print(f"[green]✓[/green] Database at {config.db_path}")
    console.print(f"[green]✓[/green] Cache dir at {config.data.cache_dir}")
    console.print("\n[dim]Run 'alphaevo showcase' to try it out![/dim]")


# ── Version ────────────────────────────────────────────────────────────


@app.command()
def version() -> None:
    """Show AlphaEvo version."""
    from alphaevo import __version__

    console.print(f"AlphaEvo v{__version__}")


# ── Demo ───────────────────────────────────────────────────────────────


@app.command()
def demo(
    real: bool = typer.Option(False, "--real", help="Use real market data (requires network)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use synthetic data"),
    market: str = typer.Option("us", "--market", help="Market for real data: us or cn"),
) -> None:
    """Run the quick demo — real snapshot by default, synthetic with `--synthetic`."""
    if real:
        from alphaevo.cli.demo import run_real_demo

        run_real_demo(console, market=market)
    elif synthetic:
        from alphaevo.cli.demo import run_demo

        run_demo(console)
    else:
        from alphaevo.cli.demo import run_showcase

        run_showcase(console)


@app.command()
def showcase(
    live: bool = typer.Option(False, "--live", help="Try live yfinance before snapshot fallback"),
    write_docs: bool = typer.Option(
        False,
        "--write-docs",
        help="Also update docs/reports/showcase_rsi_reversion_real_snapshot.md",
    ),
    output: str = typer.Option("reports/showcase", "--output", "-o", help="Report directory"),
) -> None:
    """Run the star-facing real-data showcase and generate a report."""
    from alphaevo.cli.demo import run_showcase

    run_showcase(console, live=live, write_docs=write_docs, output_dir=output)


# ── Backward-compat re-exports (used by tests) ────────────────────────

from alphaevo.cli.commands.factor import (  # noqa: E402, F811
    _build_forward_returns,
    _load_factor_history,
)

__all__ = [
    "app",
    "_get_config",
    "_get_store",
    "_load_factor_history",
    "_build_forward_returns",
]


if __name__ == "__main__":
    app()
