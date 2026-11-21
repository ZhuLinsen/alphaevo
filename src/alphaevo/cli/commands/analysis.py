"""Analysis CLI commands — leaderboard, compare, tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from alphaevo.cli._helpers import _get_config, _get_store

if TYPE_CHECKING:
    from alphaevo.models.strategy import Strategy
    from alphaevo.strategy.store import StrategyStore

console = Console()

_IGNORED_DIFF_PATHS = {
    "meta.id",
    "meta.name",
    "meta.version",
    "meta.parent_id",
    "meta.created_at",
    "meta.status",
}


def _flatten_strategy(obj: object, *, prefix: str = "") -> dict[str, str]:
    """Flatten nested strategy data into path -> display value."""
    flattened: dict[str, str] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_strategy(value, prefix=next_prefix))
        return flattened

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            flattened.update(_flatten_strategy(value, prefix=next_prefix))
        return flattened

    flattened[prefix] = str(obj)
    return flattened


def _strategy_diff_rows(s1: Strategy, s2: Strategy) -> list[tuple[str, str, str]]:
    """Return changed strategy fields as rows for a compare table."""
    from alphaevo.strategy.dsl.serializer import StrategySerializer

    serializer = StrategySerializer()
    d1 = _flatten_strategy(serializer.to_dict(s1))
    d2 = _flatten_strategy(serializer.to_dict(s2))

    rows: list[tuple[str, str, str]] = []
    for path in sorted(set(d1) | set(d2)):
        if path in _IGNORED_DIFF_PATHS:
            continue
        v1 = d1.get(path, "—")
        v2 = d2.get(path, "—")
        if v1 != v2:
            rows.append((path, v1, v2))
    return rows


def _format_change_line(path: str, before: str, after: str) -> Text:
    """Render a diff line without Rich treating selectors as markup."""
    line = Text("  • ")
    line.append(path, style="cyan")
    line.append(": ")
    line.append(before)
    line.append(" -> ")
    line.append(after)
    return line


def _best_score(store: StrategyStore, strategy_id: str) -> float | None:
    """Return the best recorded confidence score for a strategy."""
    evaluations = store.get_evaluations(strategy_id)
    if not evaluations:
        return None
    return max((float(ev.confidence_score) for ev in evaluations), default=None)


def _tree_label(
    store: StrategyStore,
    strategy: Strategy,
    *,
    champion_id: str | None,
    parent: Strategy | None,
) -> str:
    """Build one human-readable tree label."""
    score = _best_score(store, strategy.meta.id)
    parts = [f"{strategy.meta.id}"]
    if score is not None:
        parts.append(f"score={score:.1%}")
    else:
        parts.append("score=—")
    if champion_id == strategy.meta.id:
        parts.append("champion")
    if parent is not None:
        diff_rows = _strategy_diff_rows(parent, strategy)
        if diff_rows:
            examples = ", ".join(path for path, _, _ in diff_rows[:2])
            parts.append(f"changes={len(diff_rows)} [{examples}]")
    return " | ".join(parts)


def _render_tree(
    store: StrategyStore,
    strategy: Strategy,
    *,
    strategies_by_id: dict[str, Strategy],
    children_map: dict[str | None, list[Strategy]],
    champion_id: str | None,
    prefix: str = "",
    is_last: bool = True,
    is_root: bool = False,
) -> list[str]:
    """Render one strategy subtree as ASCII lines."""
    parent = strategies_by_id.get(strategy.meta.parent_id) if strategy.meta.parent_id else None
    connector = "" if is_root else ("└── " if is_last else "├── ")
    lines = [
        f"{prefix}{connector}{_tree_label(store, strategy, champion_id=champion_id, parent=parent)}"
    ]

    children = sorted(children_map.get(strategy.meta.id, []), key=lambda item: item.meta.version)
    child_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
    for idx, child in enumerate(children):
        child_is_last = idx == len(children) - 1
        lines.extend(
            _render_tree(
                store,
                child,
                strategies_by_id=strategies_by_id,
                children_map=children_map,
                champion_id=champion_id,
                prefix=child_prefix,
                is_last=child_is_last,
                is_root=False,
            )
        )
    return lines


def leaderboard_command(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of strategies to show"),
    scope: str = typer.Option(
        "local",
        "--scope",
        help="Leaderboard scope: local or official",
    ),
    artifact_root: str = typer.Option(
        "reports",
        "--artifact-root",
        help="Run artifacts root directory",
    ),
) -> None:
    """Show strategy leaderboard ranked by confidence score."""
    from alphaevo.leaderboard.service import LeaderboardScope, LeaderboardService

    normalized_scope = scope.strip().lower()
    if normalized_scope not in {"local", "official"}:
        console.print(f"[red]Invalid scope '{scope}': choose 'local' or 'official'[/red]")
        raise typer.Exit(1)

    scope_val: LeaderboardScope = "official" if normalized_scope == "official" else "local"
    config = _get_config()
    service = LeaderboardService(db_path=config.db_path, artifact_root=artifact_root)
    entries = service.get_entries(
        scope=scope_val,
        limit=limit,
        min_signal_count=config.evolution.min_signal_count,
    )

    if not entries:
        if normalized_scope == "official":
            console.print(
                "[yellow]No official_v1 runs exist yet (local runs are never promoted automatically).[/yellow]"
            )
            return
        store: StrategyStore = _get_store()
        all_strategies = store.list_all()
        total_evaluations = sum(len(store.get_evaluations(s.meta.id)) for s in all_strategies)
        if total_evaluations:
            console.print("[yellow]No leaderboard-eligible evaluations yet.[/yellow]")
            console.print(
                f"[dim]Entries need at least {config.evolution.min_signal_count} signals. "
                "Saved demo runs and tiny experiments stay off the leaderboard until they have enough data.[/dim]"
            )
        else:
            console.print("[yellow]No evaluations yet. Run some strategies first![/yellow]")
        return

    title = (
        "🏅 Official Strategy Leaderboard (official_v1)"
        if normalized_scope == "official"
        else "🏆 Strategy Leaderboard (local)"
    )
    table = Table(title=title)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Strategy", style="cyan")
    table.add_column("Protocol")
    table.add_column("Dirty", justify="center")
    table.add_column("Repro")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Return", justify="right")
    table.add_column("P/L Ratio", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Score", justify="right", style="bold green")

    for i, entry in enumerate(entries, 1):
        table.add_row(
            str(i),
            entry.strategy_id,
            entry.protocol,
            "yes" if entry.code_dirty else "no",
            entry.data_reproducibility,
            f"{entry.win_rate:.1%}",
            f"{entry.avg_return:.2%}",
            f"{entry.profit_loss_ratio:.2f}",
            f"{entry.max_drawdown:.1%}",
            str(entry.signal_count),
            f"{entry.confidence_score:.2f}",
        )
    console.print(table)


def compare_command(
    id1: str = typer.Argument(..., help="First strategy ID"),
    id2: str = typer.Argument(..., help="Second strategy ID"),
) -> None:
    """Compare two strategies side by side."""
    store: StrategyStore = _get_store()
    s1 = store.get(id1)
    s2 = store.get(id2)
    if not s1 or not s2:
        missing = [i for i, s in [(id1, s1), (id2, s2)] if s is None]
        console.print(f"[red]Strategy not found: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    evals1 = store.get_evaluations(id1)
    evals2 = store.get_evaluations(id2)

    table = Table(title=f"📊 {id1} vs {id2}")
    table.add_column("Metric", style="cyan")
    table.add_column(id1, justify="right")
    table.add_column(id2, justify="right")

    if evals1 and evals2:
        e1, e2 = evals1[0], evals2[0]  # newest first
        table.add_row("Win Rate", f"{e1.overall.win_rate:.1%}", f"{e2.overall.win_rate:.1%}")
        table.add_row("Avg Return", f"{e1.overall.avg_return:.2%}", f"{e2.overall.avg_return:.2%}")
        pl1 = f"{e1.overall.profit_loss_ratio:.2f}"
        pl2 = f"{e2.overall.profit_loss_ratio:.2f}"
        table.add_row("P/L Ratio", pl1, pl2)
        table.add_row("Max DD", f"{e1.overall.max_drawdown:.1%}", f"{e2.overall.max_drawdown:.1%}")
        table.add_row("Sharpe", f"{e1.overall.sharpe_ratio:.2f}", f"{e2.overall.sharpe_ratio:.2f}")
        table.add_row("Signals", str(e1.overall.signal_count), str(e2.overall.signal_count))
        table.add_row("Score", f"{e1.confidence_score:.2f}", f"{e2.confidence_score:.2f}")
    else:
        table.add_row(
            "Status",
            "✅ evaluated" if evals1 else "❌ not evaluated",
            "✅ evaluated" if evals2 else "❌ not evaluated",
        )
    console.print(table)

    diff_rows = _strategy_diff_rows(s1, s2)
    if diff_rows:
        console.print("\n[bold]🧩 Strategy Differences[/bold]")
        for path, v1, v2 in diff_rows:
            console.print(_format_change_line(path, v1, v2))
    else:
        console.print("\n[dim]No structural strategy differences beyond version metadata.[/dim]")


def tree_command(
    strategy_id: str | None = typer.Argument(None, help="Strategy family ID"),
    all_trees: bool = typer.Option(False, "--all", help="Show all trees"),
) -> None:
    """Visualize strategy evolution tree."""
    store: StrategyStore = _get_store()

    if all_trees:
        strategies = store.list_all()
    elif strategy_id:
        family = strategy_id.rsplit("_v", 1)[0]
        strategies = store.list_by_family(family)
    else:
        console.print("[red]Provide a strategy ID or use --all[/red]")
        raise typer.Exit(1)

    if not strategies:
        console.print("[yellow]No strategies found[/yellow]")
        return

    console.print("[bold]🌳 Evolution Tree[/bold]\n")

    if all_trees:
        families: dict[str, list[Strategy]] = {}
        for strategy in strategies:
            family_id = strategy.meta.family_id or strategy.meta.id.rsplit("_v", 1)[0]
            families.setdefault(family_id, []).append(strategy)
        family_groups = sorted(families.items(), key=lambda item: item[0])
    else:
        family_id = strategies[0].meta.family_id or strategies[0].meta.id.rsplit("_v", 1)[0]
        family_groups = [(family_id, strategies)]

    for family_idx, (family_id, family_strategies) in enumerate(family_groups):
        family_strategies = sorted(family_strategies, key=lambda item: item.meta.version)
        strategies_by_id = {strategy.meta.id: strategy for strategy in family_strategies}
        children_map: dict[str | None, list[Strategy]] = {}
        for strategy in family_strategies:
            parent_id = strategy.meta.parent_id
            if parent_id not in strategies_by_id:
                parent_id = None
            children_map.setdefault(parent_id, []).append(strategy)

        score_map = {
            strategy.meta.id: _best_score(store, strategy.meta.id) for strategy in family_strategies
        }
        scored = {
            strategy_id: score for strategy_id, score in score_map.items() if score is not None
        }
        champion_id = max(scored, key=lambda strategy_id: scored[strategy_id]) if scored else None

        console.print(f"[cyan]{family_id}[/cyan]")
        roots = sorted(children_map.get(None, []), key=lambda item: item.meta.version)
        for root_idx, root in enumerate(roots):
            root_is_last = root_idx == len(roots) - 1
            for line in _render_tree(
                store,
                root,
                strategies_by_id=strategies_by_id,
                children_map=children_map,
                champion_id=champion_id,
                prefix="",
                is_last=root_is_last,
                is_root=True,
            ):
                console.print(f"  {line}")
        if family_idx != len(family_groups) - 1:
            console.print()
