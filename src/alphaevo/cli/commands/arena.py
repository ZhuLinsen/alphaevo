"""CLI commands for Live Self-Evolving Arena, static HTML sites & notifications."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from alphaevo.cli._helpers import _get_config

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.models.strategy import Strategy

arena_app = typer.Typer(help="Live Self-Evolving Strategy Arena & interactive HTML dashboards")
console = Console()


def _load_arena_strategies(
    strategy_refs: str | None,
    config_obj: object,
) -> list[Strategy]:
    from alphaevo.cli.commands.rsi import _resolve_strategy
    from alphaevo.cli.demo import _find_builtin_strategy_dir
    from alphaevo.core.config import AppConfig
    from alphaevo.strategy.dsl.parser import StrategyParser

    cfg = config_obj if isinstance(config_obj, AppConfig) else _get_config()
    if strategy_refs:
        refs = [s.strip() for s in strategy_refs.split(",") if s.strip()]
        return [_resolve_strategy(ref, cfg) for ref in refs]

    defaults = [
        "rsi_reversion",
        "ma_crossover",
        "mean_reversion_oversold",
        "volatility_compression_breakout",
    ]
    builtin_dir = _find_builtin_strategy_dir()
    parser = StrategyParser()
    loaded: list[Strategy] = []
    if builtin_dir is not None:
        for stem in defaults:
            yaml_path = builtin_dir / f"{stem}.yaml"
            if yaml_path.is_file():
                loaded.append(parser.parse_file(yaml_path))
    if not loaded and builtin_dir is not None:
        loaded = parser.parse_directory(builtin_dir)[:4]
    if not loaded:
        raise ValueError("No strategies found for Arena evaluation")
    return loaded


@arena_app.command("run")
def arena_run(
    strategies: str | None = typer.Option(
        None,
        "--strategies",
        "-s",
        help="Comma-separated strategy IDs or YAML paths (defaults to 4 core OHLCV templates)",
    ),
    rounds: int = typer.Option(2, "--rounds", "-r", help="RSI evolution rounds per strategy"),
    live: bool = typer.Option(
        False, "--live", help="Fetch live market data (falls back to snapshot if unavailable)"
    ),
    market: str = typer.Option("us", "--market", "-m", help="Market preset for live mode: us|cn"),
    bundle: str | None = typer.Option(
        None, "--bundle", "-b", help="Optional immutable DataBundle ID"
    ),
    bundle_root: str = typer.Option(
        "reports/bundles", "--bundle-root", help="Root directory for immutable bundles"
    ),
    shadow_dir: str | None = typer.Option(
        None,
        "--shadow-dir",
        help="Optional persistent Shadow Ledger directory for replay commitments",
    ),
    output_dir: str = typer.Option(
        "reports/arena", "--output-dir", "-o", help="Output directory for index.html & JSON"
    ),
    notify: bool = typer.Option(
        False, "--notify", help="Dispatch digest to configured notification channels"
    ),
    dry_run_notify: bool = typer.Option(
        False, "--dry-run-notify", help="Preview notification delivery without network POSTs"
    ),
) -> None:
    """Run multi-strategy RSI evolution, shadow verification & generate an interactive HTML Arena."""
    from alphaevo.arena import ArenaRunner, render_arena_html

    config = _get_config()
    try:
        strat_list = _load_arena_strategies(strategies, config)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None

    market_label = "US Tech Snapshot"
    data_source = "bundled_snapshot"
    market_data: dict[str, pd.DataFrame]

    if bundle:
        from alphaevo.provenance.bundle import DataBundleStore

        market_data, b_manifest = DataBundleStore(bundle_root).load_bundle(bundle, verify=True)
        market_label = f"Bundle {b_manifest.name} ({b_manifest.market})"
        data_source = f"bundle:{b_manifest.bundle_id}"
    elif live:
        from alphaevo.cli.commands.rsi import _load_market_data

        syms = (
            ["600519", "000858", "601318", "000333", "600036"]
            if market.lower() in {"cn", "a_share"}
            else ["AAPL", "MSFT", "NVDA", "AMD", "META"]
        )
        try:
            fetched = _load_market_data(live=True, symbols=syms, days=260)
            if fetched:
                market_data = fetched
                market_label = f"Live {market.upper()} Basket"
                data_source = f"live:{config.data.adapter}"
            else:
                raise RuntimeError("Empty live market response")
        except Exception as exc:
            console.print(f"[yellow]Live fetch fell back to bundled snapshot ({exc})[/yellow]")
            market_data = _load_market_data(live=False)
    else:
        from alphaevo.cli.demo import load_showcase_snapshot

        market_data, _ = load_showcase_snapshot()

    runner = ArenaRunner(backtest_config=config.backtest, db_path=config.db_path)
    snapshot = runner.run(
        strat_list,
        market_data,
        rounds=rounds,
        shadow_dir=Path(shadow_dir) if shadow_dir else None,
        market_label=market_label,
        data_source=data_source,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    snap_file = out_path / "arena_snapshot.json"
    html_file = out_path / "index.html"
    digest_file = out_path / "digest.md"

    snap_file.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    html_file.write_text(render_arena_html(snapshot), encoding="utf-8")
    digest_md = ArenaRunner.render_digest_markdown(snapshot)
    digest_file.write_text(digest_md, encoding="utf-8")

    table = Table(title=f"🏟️ AlphaEvo Live Self-Evolving Arena — {snapshot.market_label}")
    table.add_column("Strategy", style="cyan")
    table.add_column("Champion")
    table.add_column("Score (Base -> Champ)", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("AvgRet", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Null", justify="center")
    table.add_column("Shadow")
    for e in snapshot.entries:
        table.add_row(
            e.strategy_id,
            e.champion_id,
            f"{e.baseline_score:.1%} -> {e.champion_score:.1%}",
            f"{e.win_rate:.1%}",
            f"{e.avg_return:+.2%}",
            f"{e.max_drawdown:.1%}",
            "PASS" if e.null_passed else "FAIL",
            e.shadow_stage,
        )
    console.print(table)
    console.print(
        f"[green]✓ Arena artifacts saved:[/green] {html_file}, {snap_file}, {digest_file}"
    )

    if notify or dry_run_notify:
        from alphaevo.notify import NotificationDispatcher, NotificationMessage

        dispatcher = NotificationDispatcher(config.notify)
        channels = dispatcher.configured_channels()
        if not channels:
            console.print("[yellow]No notification channels configured.[/yellow]")
        else:
            msg = NotificationMessage(
                title=f"AlphaEvo Arena ({snapshot.market_label})",
                summary=(
                    f"{snapshot.total_strategies} strategies · "
                    f"{snapshot.total_promotions} promotions · {snapshot.total_vetoes} vetoes"
                ),
                markdown_body=digest_md,
                severity="success" if snapshot.total_promotions > 0 else "info",
                tags=["alphaevo", "arena"],
            )
            results = dispatcher.send(msg, dry_run=dry_run_notify)
            for res in results:
                tag = (
                    "DRY-RUN OK"
                    if res.dry_run and res.success
                    else ("OK" if res.success else f"ERR: {res.error}")
                )
                console.print(f"  Notify [{res.channel}]: {tag}")


@arena_app.command("build-site")
def arena_build_site(
    snapshot_file: str = typer.Option(
        "reports/arena/arena_snapshot.json",
        "--snapshot",
        "-s",
        help="Path to arena_snapshot.json",
    ),
    output_dir: str = typer.Option(
        "site", "--output-dir", "-o", help="Directory to write index.html"
    ),
) -> None:
    """Build static single-file Arena HTML site from an existing arena_snapshot.json."""
    from alphaevo.arena import ArenaSnapshot, render_arena_html

    snap_path = Path(snapshot_file)
    out_path = Path(output_dir)
    if not snap_path.is_file():
        console.print(f"[red]Snapshot not found: {snap_path}[/red]")
        raise typer.Exit(1)

    snapshot = ArenaSnapshot.model_validate_json(snap_path.read_text(encoding="utf-8"))
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "index.html").write_text(render_arena_html(snapshot), encoding="utf-8")
    if snap_path.resolve() != (out_path / "arena_snapshot.json").resolve():
        (out_path / "arena_snapshot.json").write_text(
            snapshot.model_dump_json(indent=2), encoding="utf-8"
        )
    console.print(f"[green]✓ Built static Arena site at[/green] {out_path / 'index.html'}")


@arena_app.command("notify-test")
def arena_notify_test(
    send: bool = typer.Option(
        False, "--send", help="Actually send HTTP requests (defaults to dry-run)"
    ),
) -> None:
    """Validate configured Feishu/WeCom/DingTalk/Telegram/Discord/Slack/Webhook channels."""
    from alphaevo.notify import NotificationDispatcher, NotificationMessage

    config = _get_config()
    dispatcher = NotificationDispatcher(config.notify)
    channels = dispatcher.configured_channels()
    if not channels:
        console.print(
            "[yellow]No channels configured. Set ALPHAEVO_FEISHU_WEBHOOK, ALPHAEVO_TELEGRAM_BOT_TOKEN, etc.[/yellow]"
        )
        return

    msg = NotificationMessage(
        title="AlphaEvo Notification Smoke Check",
        summary="Channel payload configuration check passed.",
        markdown_body="- Triggered via `alphaevo arena notify-test`",
        severity="info",
    )
    for res in dispatcher.send(msg, dry_run=not send):
        state = "PASS" if res.success else f"FAIL ({res.error})"
        mode = "dry-run" if res.dry_run else "live"
        console.print(f"  [{mode}] {res.channel}: {state}")


__all__ = ["arena_app"]
