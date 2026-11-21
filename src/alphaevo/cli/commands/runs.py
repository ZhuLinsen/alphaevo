"""Run provenance CLI commands — list, show, compare immutable research runs."""

from __future__ import annotations

from typing import cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alphaevo.cli._helpers import _get_config

runs_app = typer.Typer(help="Inspect and compare immutable research run artifacts")
console = Console()


@runs_app.command("list")
def runs_list(
    kind: str | None = typer.Option(
        None, "--kind", "-k", help="Filter by kind: run|evolve|optimize|rsi"
    ),
    strategy: str | None = typer.Option(None, "--strategy", "-s", help="Filter by strategy ID"),
    protocol: str | None = typer.Option(
        None,
        "--protocol",
        "-p",
        help="Filter by protocol: fast_research_v1|full_research_v1|official_v1",
    ),
    official_only: bool = typer.Option(
        False, "--official-only", help="Show only official_v1 verified runs"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum rows to display"),
    artifact_root: str = typer.Option(
        "reports", "--artifact-root", "-a", help="Run artifacts root directory"
    ),
) -> None:
    """List immutable research runs indexed in SQLite."""
    from alphaevo.provenance.contracts import ProtocolName, RunKind
    from alphaevo.provenance.store import RunArtifactStore

    config = _get_config()
    store = RunArtifactStore(artifact_root=artifact_root, db_path=config.db_path)
    entries = store.list_runs(
        kind=cast("RunKind | None", kind),
        strategy_id=strategy,
        protocol=cast("ProtocolName | None", protocol),
        official_only=official_only,
        limit=limit,
    )
    if not entries:
        console.print("[yellow]No matching research run artifacts found.[/yellow]")
        return

    table = Table(title="🧾 Research Runs")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Kind")
    table.add_column("Strategy")
    table.add_column("Protocol")
    table.add_column("Dirty", justify="center")
    table.add_column("Repro")
    table.add_column("Signals", justify="right")
    table.add_column("Score", justify="right", style="bold green")
    for entry in entries:
        table.add_row(
            entry.run_id,
            entry.kind,
            entry.strategy_id,
            entry.effective_protocol,
            "yes" if entry.code_dirty else "no",
            entry.data_reproducibility,
            str(entry.signal_count),
            f"{entry.score:.1%}",
        )
    console.print(table)


@runs_app.command("show")
def runs_show(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    artifact_root: str = typer.Option(
        "reports", "--artifact-root", "-a", help="Run artifacts root directory"
    ),
) -> None:
    """Inspect full manifest provenance for a single run."""
    from alphaevo.provenance.store import RunArtifactStore

    config = _get_config()
    store = RunArtifactStore(artifact_root=artifact_root, db_path=config.db_path)
    manifest = store.get_manifest(run_id)
    if manifest is None:
        console.print(f"[red]Run manifest not found: {run_id}[/red]")
        raise typer.Exit(1)

    lines = [
        f"[bold]Run ID:[/bold] {manifest.run_id}",
        f"[bold]Kind:[/bold] {manifest.kind} | [bold]Status:[/bold] {manifest.status}",
        f"[bold]Strategy:[/bold] {manifest.strategy.id} (v{manifest.strategy.version}, {manifest.strategy.snapshot_hash[:19]})",
        f"[bold]Protocol:[/bold] {manifest.effective_protocol} (requested={manifest.requested_protocol})",
        f"[bold]Code:[/bold] {manifest.code_version.source}:{manifest.code_version.commit} dirty={manifest.code_version.dirty}",
        f"[bold]Data:[/bold] adapter={manifest.data.adapter} repro={manifest.data.reproducibility} mutability={manifest.data.data_mutability}",
        f"[bold]Metrics:[/bold] score={manifest.summary.score:.1%} win={manifest.summary.win_rate:.1%} avg={manifest.summary.avg_return:.2%} dd={manifest.summary.max_drawdown:.1%} sharpe={manifest.summary.sharpe_ratio:.2f} signals={manifest.summary.signal_count}",
        f"[bold]Artifact Dir:[/bold] {manifest.artifacts.artifact_dir}",
    ]
    if manifest.protocol_downgrade_reason:
        lines.append(
            f"[bold yellow]Downgrade Reason:[/bold yellow] {manifest.protocol_downgrade_reason}"
        )
    console.print(
        Panel("\n".join(lines), title=f"Manifest — {manifest.run_id}", border_style="cyan")
    )


@runs_app.command("compare")
def runs_compare(
    run_id_1: str = typer.Argument(..., help="Baseline run ID"),
    run_id_2: str = typer.Argument(..., help="Candidate run ID"),
    artifact_root: str = typer.Option(
        "reports", "--artifact-root", "-a", help="Run artifacts root directory"
    ),
) -> None:
    """High-signal comparison between two immutable research runs."""
    from alphaevo.provenance.store import RunArtifactStore

    config = _get_config()
    store = RunArtifactStore(artifact_root=artifact_root, db_path=config.db_path)
    try:
        comp = store.compare_runs(run_id_1, run_id_2)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    comp_label = "[green]True[/green]" if comp.comparable else "[red]False[/red]"
    console.print(
        Panel(
            f"[bold]Comparable:[/bold] {comp_label}\n"
            f"[bold]Protocols:[/bold] {comp.protocol_1} vs {comp.protocol_2}\n"
            f"[bold]Hashes Changed:[/bold] strategy={comp.strategy_hash_changed}, "
            f"config={comp.config_hash_changed}, data={comp.data_fingerprint_changed}",
            title=f"Run Compare: {run_id_1} -> {run_id_2}",
            border_style="cyan" if comp.comparable else "yellow",
        )
    )
    if not comp.comparable:
        console.print(
            "[yellow]Warning: Protocols differ (comparable: false); diagnostic diff only.[/yellow]"
        )
        for reason in comp.incomparability_reasons:
            console.print(f"  [yellow]• {reason}[/yellow]")

    table = Table(title="Metric Deltas (run2 - run1)")
    table.add_column("Metric", style="cyan")
    table.add_column("Delta", justify="right")
    table.add_row("Score", f"{comp.score_delta:+.2%}")
    table.add_row("Win Rate", f"{comp.win_rate_delta:+.2%}")
    table.add_row("Avg Return", f"{comp.avg_return_delta:+.2%}")
    table.add_row("Max Drawdown", f"{comp.max_drawdown_delta:+.2%}")
    table.add_row("Sharpe Ratio", f"{comp.sharpe_ratio_delta:+.2f}")
    table.add_row("Signals", f"{comp.signal_count_delta:+d}")
    console.print(table)

    if comp.changed_strategy_paths:
        console.print("\n[bold]Changed Strategy DSL Paths[/bold]")
        for path in comp.changed_strategy_paths[:20]:
            console.print(f"  • [cyan]{path}[/cyan]")


@runs_app.command("rerun-official")
def runs_rerun_official(
    target: str = typer.Argument(
        ..., help="Existing run_id (frozen snapshot) or strategy_id / YAML path"
    ),
    bundle_id: str = typer.Option(..., "--bundle", "-b", help="Verified DataBundle ID"),
    artifact_root: str = typer.Option(
        "reports", "--artifact-root", "-a", help="Run artifacts root directory"
    ),
    bundle_root: str = typer.Option(
        "reports/bundles", "--bundle-root", help="Root directory storing immutable bundles"
    ),
) -> None:
    """Re-evaluate a frozen snapshot or strategy under ``official_v1`` using a verified bundle."""
    from alphaevo.cli.commands.rsi import _resolve_strategy
    from alphaevo.provenance.bundle import DataBundleStore, build_bundle_summary_dict
    from alphaevo.provenance.store import RunArtifactStore
    from alphaevo.truth.kernel import EvaluationKernel

    config = _get_config()
    art_store = RunArtifactStore(artifact_root=artifact_root, db_path=config.db_path)

    strategy = art_store.load_strategy_snapshot(target)
    if strategy is None:
        try:
            strategy = _resolve_strategy(target, config)
        except ValueError as exc:
            console.print(f"[red]Target run or strategy not found: {exc}[/red]")
            raise typer.Exit(1) from None

    try:
        market_data, b_manifest = DataBundleStore(bundle_root).load_bundle(bundle_id, verify=True)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Data bundle error: {exc}[/red]")
        raise typer.Exit(1) from None

    kernel = EvaluationKernel(
        backtest_config=config.backtest,
        db_path=config.db_path,
        min_data_days=min(15, config.backtest.min_data_days),
    )
    outcome = kernel.evaluate(
        strategy,
        market_data,
        mode="full",
        record_trial=True,
        notes=f"official_rerun:{b_manifest.bundle_id}",
    )
    manifest = art_store.save_run(
        kind="run",
        strategy=strategy,
        evaluation=outcome.evaluation,
        market_data=market_data,
        trades=outcome.backtest_result.signals,
        config=config,
        adapter_name=f"bundle:{b_manifest.adapter}",
        requested_protocol="official_v1",
        data_reproducibility="replayable",
        data_mutability="immutable_bundle",
        data_bundle_id=b_manifest.bundle_id,
        extra_config={"bundle": build_bundle_summary_dict(b_manifest)},
        min_signals_official=min(15, config.evolution.min_signal_count),
    )

    color = "green" if manifest.effective_protocol == "official_v1" else "yellow"
    console.print(
        Panel(
            f"[bold]Run ID:[/bold] [cyan]{manifest.run_id}[/cyan]\n"
            f"[bold]Strategy:[/bold] {manifest.strategy.id} | "
            f"[bold]Bundle:[/bold] {b_manifest.bundle_id}\n"
            f"[bold]Effective Protocol:[/bold] [{color}]{manifest.effective_protocol}[/{color}] "
            f"(requested={manifest.requested_protocol})\n"
            f"[bold]Metrics:[/bold] score={manifest.summary.score:.1%} "
            f"win={manifest.summary.win_rate:.1%} avg={manifest.summary.avg_return:.2%} "
            f"dd={manifest.summary.max_drawdown:.1%} signals={manifest.summary.signal_count}",
            title="🏅 Official Bundle Rerun",
            border_style=color,
        )
    )
    if manifest.protocol_downgrade_reason:
        console.print(f"[yellow]Downgrade reason: {manifest.protocol_downgrade_reason}[/yellow]")


__all__ = ["runs_app"]
