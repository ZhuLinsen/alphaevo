"""Data management CLI commands — immutable OHLCV data bundles."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

data_app = typer.Typer(help="Manage immutable research data bundles")
bundle_app = typer.Typer(help="Create, inspect, and verify immutable OHLCV data bundles")
data_app.add_typer(bundle_app, name="bundle")
console = Console()


@bundle_app.command("create")
def bundle_create(
    name: str = typer.Option("us_tech", "--name", "-n", help="Logical bundle name prefix"),
    live: bool = typer.Option(False, "--live", help="Fetch live market data instead of snapshot"),
    symbols: str | None = typer.Option(
        None, "--symbols", "-s", help="Comma-separated symbols to fetch when using --live"
    ),
    days: int = typer.Option(365, "--days", help="Lookback calendar days when using --live"),
    market: str = typer.Option("us", "--market", help="Market tag (us | a_share | hk)"),
    bundle_root: str = typer.Option(
        "reports/bundles", "--bundle-root", help="Destination directory for immutable bundles"
    ),
    description: str = typer.Option("", "--description", "-d", help="Optional bundle description"),
) -> None:
    """Create a hash-verified, immutable OHLCV data bundle for reproducible runs."""
    import asyncio

    from alphaevo.provenance.bundle import DataBundleStore

    adapter_label = "snapshot"
    if not live:
        from alphaevo.cli.demo import load_showcase_snapshot

        market_data, _snap = load_showcase_snapshot()
        if symbols:
            keep = {s.strip().upper() for s in symbols.split(",") if s.strip()}
            market_data = {k: v for k, v in market_data.items() if k.upper() in keep}
    else:
        from alphaevo.cli._helpers import _get_config
        from alphaevo.data.adapter import DataManager, get_adapter_chain
        from alphaevo.data.cache import DataCache

        config = _get_config()
        adapter_label = config.data.adapter
        mgr = DataManager(
            adapters=get_adapter_chain(config),
            cache=DataCache(config.data.cache_dir),
        )
        end_d = date.today()
        start_d = end_d - timedelta(days=max(30, days))
        target_syms = (
            [s.strip() for s in symbols.split(",") if s.strip()]
            if symbols
            else ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]
        )

        async def _fetch_all() -> dict[str, Any]:
            sem = asyncio.Semaphore(max(1, config.data.max_concurrent_fetches))

            async def _one(sym: str) -> tuple[str, Any]:
                async with sem:
                    try:
                        df = await mgr.get_history(sym, start_d, end_d)
                        return sym, df
                    except Exception:
                        return sym, None

            pairs = await asyncio.gather(*[_one(s) for s in target_syms])
            return {
                sym: df for sym, df in pairs if df is not None and not getattr(df, "empty", True)
            }

        market_data = asyncio.run(_fetch_all())

    if not market_data:
        console.print("[red]Error: no non-empty market data available to bundle.[/red]")
        raise typer.Exit(1)

    manifest = DataBundleStore(bundle_root).create_bundle(
        market_data,
        name=name,
        market=market,
        adapter=adapter_label,
        description=description,
    )
    console.print(
        Panel(
            f"[bold]Bundle ID:[/bold] [cyan]{manifest.bundle_id}[/cyan]\n"
            f"[bold]Symbols:[/bold] {manifest.symbol_count} | "
            f"[bold]Total Rows:[/bold] {manifest.total_rows}\n"
            f"[bold]Date Range:[/bold] {manifest.date_range[0]} to {manifest.date_range[1]}\n"
            f"[bold]Hash:[/bold] {manifest.bundle_hash}\n"
            f"[bold]Location:[/bold] {Path(bundle_root) / manifest.bundle_id}",
            title="📦 Immutable Data Bundle Created",
            border_style="green",
        )
    )


@bundle_app.command("list")
def bundle_list(
    bundle_root: str = typer.Option(
        "reports/bundles", "--bundle-root", help="Root directory storing immutable bundles"
    ),
) -> None:
    """List immutable data bundles sorted newest first."""
    from alphaevo.provenance.bundle import DataBundleStore

    bundles = DataBundleStore(bundle_root).list_bundles()
    if not bundles:
        console.print(f"[yellow]No data bundles found under {bundle_root}.[/yellow]")
        return

    table = Table(title="📦 Immutable Data Bundles")
    table.add_column("Bundle ID", style="cyan", no_wrap=True)
    table.add_column("Market")
    table.add_column("Adapter")
    table.add_column("Symbols", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Window")
    table.add_column("Hash")
    for b in bundles:
        table.add_row(
            b.bundle_id,
            b.market,
            b.adapter,
            str(b.symbol_count),
            str(b.total_rows),
            f"{b.date_range[0]}..{b.date_range[1]}",
            b.bundle_hash[:19],
        )
    console.print(table)


@bundle_app.command("show")
def bundle_show(
    bundle_id: str = typer.Argument(..., help="Data bundle ID"),
    bundle_root: str = typer.Option(
        "reports/bundles", "--bundle-root", help="Root directory storing immutable bundles"
    ),
) -> None:
    """Inspect manifest and per-symbol checksums for one data bundle."""
    from alphaevo.provenance.bundle import DataBundleStore

    manifest = DataBundleStore(bundle_root).get_manifest(bundle_id)
    if manifest is None:
        console.print(f"[red]Data bundle not found: {bundle_id}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Bundle ID:[/bold] {manifest.bundle_id}\n"
            f"[bold]Market/Adapter:[/bold] {manifest.market} / {manifest.adapter}\n"
            f"[bold]Window:[/bold] {manifest.date_range[0]} to {manifest.date_range[1]}\n"
            f"[bold]Aggregate Hash:[/bold] {manifest.bundle_hash}",
            title=f"Bundle — {manifest.bundle_id}",
            border_style="cyan",
        )
    )
    table = Table(title="Per-Symbol Snapshot Checksums")
    table.add_column("Symbol", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("SHA-256")
    for sym in manifest.symbols:
        table.add_row(
            sym.symbol,
            str(sym.rows),
            sym.start_date,
            sym.end_date,
            sym.ohlcv_sha256[:16],
        )
    console.print(table)


@bundle_app.command("verify")
def bundle_verify(
    bundle_id: str = typer.Argument(..., help="Data bundle ID to verify"),
    bundle_root: str = typer.Option(
        "reports/bundles", "--bundle-root", help="Root directory storing immutable bundles"
    ),
) -> None:
    """Verify SHA-256 checksums of every CSV file in an immutable data bundle."""
    from alphaevo.provenance.bundle import DataBundleStore

    verdict = DataBundleStore(bundle_root).verify_bundle(bundle_id)
    if verdict.verified:
        console.print(
            f"[green]✓ Verified cleanly:[/green] {verdict.bundle_id} "
            f"({verdict.checked_symbols} symbols, {verdict.actual_hash[:23]})"
        )
        return

    console.print(f"[red]✗ Verification failed for {verdict.bundle_id}:[/red]")
    for err in verdict.errors:
        console.print(f"  [red]• {err}[/red]")
    raise typer.Exit(1)


__all__ = ["bundle_app", "data_app"]
