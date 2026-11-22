"""CLI commands for Dual-Clock Stock RSI, Shadow Ledger, and L0 Truth Kernel."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alphaevo.cli._helpers import _get_config

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.core.config import AppConfig
    from alphaevo.models.strategy import Strategy
    from alphaevo.rsi.council import HeterogeneousCouncil

rsi_app = typer.Typer(
    help="Dual-Clock Stock RSI — evolve strategies (L1), factors (L2), and harnesses (L3)"
)
shadow_app = typer.Typer(
    help="Bitemporal Forward Shadow Ledger — hash-chained commitments, backfill & promotion gates"
)
truth_app = typer.Typer(
    help="L0 Anti-Delusion Truth Kernel — trial ledger, DSR & runtime PIT probing"
)
console = Console()


def _resolve_strategy(strategy_ref: str, config: AppConfig) -> Strategy:
    """Resolve strategy from YAML path, builtin filename/id, or SQLite StrategyStore."""
    from alphaevo.cli.demo import _find_builtin_strategy_dir
    from alphaevo.strategy.dsl.parser import StrategyParser
    from alphaevo.strategy.store import StrategyStore

    parser = StrategyParser()
    candidate_path = Path(strategy_ref)
    if candidate_path.is_file():
        return parser.parse_file(candidate_path)

    store = StrategyStore(config.db_path)
    stored = store.get(strategy_ref)
    if stored is not None:
        return stored

    builtin_dir = _find_builtin_strategy_dir()
    if builtin_dir is not None:
        direct_yaml = builtin_dir / f"{strategy_ref}.yaml"
        if direct_yaml.is_file():
            return parser.parse_file(direct_yaml)
        stem = strategy_ref.rsplit("_v", 1)[0]
        stem_yaml = builtin_dir / f"{stem}.yaml"
        if stem_yaml.is_file():
            return parser.parse_file(stem_yaml)
        for strat in parser.parse_directory(builtin_dir):
            if strat.meta.id == strategy_ref or strat.meta.family_id == strategy_ref:
                return strat

    raise ValueError(f"Strategy not found: {strategy_ref}")


def _load_market_data(
    *,
    live: bool = False,
    symbols: list[str] | None = None,
    days: int = 240,
) -> dict[str, pd.DataFrame]:
    """Load real-snapshot stock data (default zero-network) or live market data."""
    if not live:
        from alphaevo.cli.demo import load_showcase_snapshot

        data, _ = load_showcase_snapshot()
        return data

    from alphaevo.core.config import ConfigManager
    from alphaevo.data.adapter import DataManager, get_adapter_chain
    from alphaevo.data.cache import DataCache

    config = ConfigManager().load()
    chain = get_adapter_chain(config)
    mgr = DataManager(adapters=chain, cache=DataCache(config.data.cache_dir))
    end_d = date.today()
    start_d = date.fromordinal(end_d.toordinal() - days)
    target_syms = symbols or ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]

    async def _fetch_one(sym: str) -> tuple[str, pd.DataFrame | None]:
        try:
            df = await mgr.get_history(sym, start_d, end_d)
            if not df.empty:
                return sym, df
        except Exception:
            return sym, None
        return sym, None

    async def _fetch() -> dict[str, pd.DataFrame]:
        pairs = await asyncio.gather(*[_fetch_one(sym) for sym in target_syms])
        return {sym: df for sym, df in pairs if df is not None and not df.empty}

    return asyncio.run(_fetch())


def _build_council(
    *,
    method: str,
    proposer_model: str | None,
    prosecutor_model: str | None,
) -> HeterogeneousCouncil:
    """Build Adversarial Heterogeneous Council respecting method and model families."""
    from alphaevo.rsi.council import (
        HeterogeneousCouncil,
        HeterogeneousCouncilConfig,
        ModelSpec,
    )

    if method == "heuristic" and not proposer_model and not prosecutor_model:
        return HeterogeneousCouncil()

    def _make_spec(name: str | None) -> ModelSpec:
        if not name:
            return ModelSpec(model="heuristic", backend="deterministic")
        lower = name.lower()
        if lower.startswith("agy:"):
            return ModelSpec(model=name.split(":", 1)[1], backend="cli", cli_tool="agy")
        if lower.startswith("codex:"):
            return ModelSpec(model=name.split(":", 1)[1], backend="cli", cli_tool="codex")
        return ModelSpec(model=name, backend="litellm")

    p_spec = _make_spec(proposer_model or "heuristic")
    q_spec = _make_spec(prosecutor_model or "heuristic")
    cfg = HeterogeneousCouncilConfig(proposer=p_spec, builder=p_spec, prosecutor=q_spec)
    return HeterogeneousCouncil(config=cfg)


# ── alphaevo rsi ... ────────────────────────────────────────────────────


@rsi_app.command("run")
def rsi_run(
    strategy_id: str = typer.Argument("rsi_reversion_v1", help="Seed strategy ID or YAML path"),
    program: Annotated[
        Path | None,
        typer.Option(
            "--program",
            "-p",
            help="Path to Karpathy-style program.md RSI charter",
        ),
    ] = None,
    rounds: int | None = typer.Option(None, "--rounds", "-r", help="Max RSI evolution rounds"),
    levels: str = typer.Option(
        "L1,L2",
        "--levels",
        help="Comma-separated allowed RSI levels (L1,L2)",
    ),
    method: str = typer.Option("heuristic", "--method", help="Proposal mode: heuristic | llm"),
    council_proposer: str | None = typer.Option(
        None,
        "--council-proposer",
        help="Proposer model (e.g. openai/gpt-6.1-sol or codex:gpt-6-astra)",
    ),
    council_prosecutor: str | None = typer.Option(
        None,
        "--council-prosecutor",
        help="Prosecutor model from a different family (e.g. anthropic/claude-sonnet-5.5 or agy:claude-opus-5-5-high)",
    ),
    live: bool = typer.Option(False, "--live", help="Fetch live market data instead of snapshot"),
    holdout_pct: float = typer.Option(
        0.0,
        "--holdout-pct",
        help="Chronological locked out-of-sample holdout fraction (e.g. 0.20)",
    ),
    portfolio_gate: bool = typer.Option(
        False,
        "--portfolio-gate",
        help="Enforce Layer-2 portfolio drawdown/return hard veto gate",
    ),
    enroll_shadow: bool = typer.Option(
        False,
        "--enroll-shadow",
        help="Preregister promoted champion into the Forward Shadow Ledger",
    ),
    save_store: bool = typer.Option(
        True,
        "--save-store/--no-save-store",
        help="Persist promoted champion into StrategyStore",
    ),
    output_dir: str = typer.Option(
        "reports/rsi",
        "--output",
        "-o",
        help="Directory for results.tsv, Markdown report, and champion YAML",
    ),
) -> None:
    """Run the L0-grounded Stock RSI evolution ratchet loop."""
    from alphaevo.models.enums import RSILevel
    from alphaevo.rsi.engine import StockRSIEngine
    from alphaevo.rsi.program import RSICharter, parse_rsi_program

    config = _get_config()
    try:
        seed_strategy = _resolve_strategy(strategy_id, config)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None

    if program is not None:
        resolved_program = program
        if not resolved_program.is_file():
            from alphaevo.cli.demo import _resolve_bundled_path

            bundled_prog = _resolve_bundled_path(program.name)
            if bundled_prog is not None:
                resolved_program = bundled_prog
        charter = parse_rsi_program(resolved_program)
        charter.seed_strategy_id = seed_strategy.meta.id
    else:
        parsed_levels = [
            RSILevel(lvl.strip().upper())
            for lvl in levels.split(",")
            if lvl.strip().upper() in {"L1", "L2"}
        ] or [RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR]
        charter = RSICharter(
            name=f"rsi_stock_{seed_strategy.meta.id}",
            seed_strategy_id=seed_strategy.meta.id,
            allowed_levels=parsed_levels,
            max_rounds=rounds or 5,
            null_permutations=min(config.rsi.null_permutations, 60),
        )

    if rounds is not None:
        charter.max_rounds = rounds
    if holdout_pct > 0.0:
        charter.locked_holdout_pct = holdout_pct
    if portfolio_gate:
        charter.require_portfolio_gate = True

    try:
        council = _build_council(
            method=method,
            proposer_model=council_proposer or config.rsi.proposer_model,
            prosecutor_model=council_prosecutor or config.rsi.prosecutor_model,
        )
    except Exception as exc:
        console.print(f"[red]Council configuration error: {exc}[/red]")
        raise typer.Exit(1) from None

    market_data = _load_market_data(live=live)
    console.print(
        Panel(
            f"[bold cyan]AlphaEvo Dual-Clock Stock RSI[/bold cyan]\n"
            f"Charter: {charter.name} | Seed: {seed_strategy.meta.id} | "
            f"Levels: {','.join(lvl.value for lvl in charter.allowed_levels)} | Rounds: {charter.max_rounds}",
            style="green",
        )
    )

    from alphaevo.forward.shadow_ledger import ShadowLedger

    engine = StockRSIEngine(
        db_path=config.db_path,
        council=council,
        backtest_config=config.backtest,
    )
    shadow_ledger = ShadowLedger(config.rsi.shadow_ledger_dir) if enroll_shadow else None
    result = engine.run(
        seed_strategy,
        market_data,
        charter=charter,
        output_dir=Path(output_dir),
        persist_champion=save_store,
        shadow_ledger=shadow_ledger,
        auto_enroll_shadow=enroll_shadow,
    )

    table = Table(title=f"RSI Ratchet Ledger — {result.charter_name}")
    table.add_column("Rd", justify="right", style="cyan")
    table.add_column("Strategy ID")
    table.add_column("Lvl", justify="center")
    table.add_column("Signals", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("AvgRet", justify="right")
    table.add_column("DSR", justify="right")
    table.add_column("Null", justify="center")
    table.add_column("Fitness", justify="right")
    table.add_column("Status", style="bold")
    table.add_column("Change")

    for r in result.rows:
        st_color = {
            "baseline": "blue",
            "keep": "green",
            "stepping_stone": "cyan",
            "revert": "yellow",
            "vetoed": "red",
        }.get(r.status, "white")
        table.add_row(
            str(r.round_num),
            r.strategy_id,
            r.level.value,
            str(r.signals),
            f"{r.win_rate:.1%}",
            f"{r.avg_return:.2%}",
            f"{r.dsr_probability:.1%}",
            "PASS" if r.null_passed else "FAIL",
            f"{r.fitness_score:.1%}",
            f"[{st_color}]{r.status.upper()}[/{st_color}]",
            r.description[:44],
        )

    console.print(table)
    console.print(
        f"\n[bold]Champion:[/bold] {result.champion_id} | "
        f"[bold]L0 Kernel Intact:[/bold] {result.kernel_intact} | "
        f"[bold]Total Lineage Trials:[/bold] {result.total_trials_recorded} | "
        f"[bold]QD Niches:[/bold] {result.archive_summary.total_cells_occupied}"
    )
    if result.holdout_evaluation is not None:
        ho = result.holdout_evaluation.overall
        console.print(
            f"[bold]Locked Holdout:[/bold] signals={ho.signal_count} | "
            f"win={ho.win_rate:.1%} | avg={ho.avg_return:.2%} | max_dd={ho.max_drawdown:.1%}"
        )
    if result.persisted_factors:
        console.print(f"[dim]Persisted L2 factors: {', '.join(result.persisted_factors)}[/dim]")
    if result.enrolled_in_shadow:
        console.print(
            f"[green]✓ Enrolled champion in Forward Shadow Ledger: {result.champion_id}[/green]"
        )
    if result.run_id:
        console.print(f"[dim]Run provenance ID: {result.run_id}[/dim]")
    if result.tsv_path:
        console.print(f"[dim]Artifacts written to {result.tsv_path} and {result.report_path}[/dim]")
    console.print(
        "[dim]Research & educational use only. Forward Shadow Ledger validation required before any live use.[/dim]"
    )


@rsi_app.command("null-audit")
def rsi_null_audit(
    strategy_id: str = typer.Argument("rsi_reversion_v1", help="Seed strategy ID or YAML path"),
    worlds: int = typer.Option(3, "--worlds", "-w", help="Number of shuffled-return null worlds"),
    rounds: int = typer.Option(3, "--rounds", "-r", help="RSI rounds per null world"),
) -> None:
    """Tier-2 L0 Anti-Delusion check: verify 0 promotions occur on shuffled returns."""
    from alphaevo.models.enums import RSILevel
    from alphaevo.rsi.engine import StockRSIEngine
    from alphaevo.rsi.program import RSICharter
    from alphaevo.truth.null_control import NullSignalController

    config = _get_config()
    seed_strategy = _resolve_strategy(strategy_id, config)
    market_data = _load_market_data(live=False)

    def _run_world(strat: Strategy, null_market: dict[str, pd.DataFrame]) -> int:
        engine = StockRSIEngine(db_path=":memory:")
        charter = RSICharter(
            name="null_audit",
            seed_strategy_id=strat.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=rounds,
            require_null_control=True,
            null_permutations=40,
        )
        res = engine.run(strat, null_market, charter=charter)
        return res.promotions

    audit = NullSignalController().verify_zero_promotion_under_shuffled_returns(
        seed_strategy,
        market_data,
        _run_world,
        n_worlds=worlds,
    )

    table = Table(title="L0 Anti-Delusion Tier-2 Null Audit")
    table.add_column("World", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("False Promotions", justify="right")
    table.add_column("Notes")
    for w in audit.worlds:
        table.add_row(str(w.world_idx), str(w.seed), str(w.promotions), w.notes)
    console.print(table)
    status = "[green]PASSED (0 false promotions)[/green]" if audit.passed else "[red]FAILED[/red]"
    console.print(f"\nAudit Verdict: {status}")


@rsi_app.command("mechanism-ab")
def rsi_mechanism_ab(
    strategy_id: str = typer.Argument("rsi_reversion_v1", help="Seed strategy ID or YAML path"),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Budget per harness variant"),
    cutoffs: str | None = typer.Option(
        None,
        "--cutoffs",
        help="Comma-separated train cutoff ratios for multi-cutoff L3 evaluation (e.g. 0.55,0.70)",
    ),
    portfolio_gate: bool = typer.Option(
        False,
        "--portfolio-gate",
        help="Enable Layer-2 portfolio hard veto gate in the guarded harness",
    ),
    output_file: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional Markdown report path",
    ),
) -> None:
    """L3 Meta-RSI paired AB experiment comparing harness calibration on unseen forward bars."""
    from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator

    config = _get_config()
    seed_strategy = _resolve_strategy(strategy_id, config)
    market_data = _load_market_data(live=False)
    cutoff_list: list[float] | None = None
    if cutoffs:
        cutoff_list = [float(part.strip()) for part in cutoffs.split(",") if part.strip()]

    naive = HarnessVariant(
        variant_id="naive_no_null",
        name="Naive Harness (No Null Gate / No Scars)",
        require_null_control=False,
        enable_pinned_scars=False,
    )
    guarded = HarnessVariant(
        variant_id="l0_guarded_rsi",
        name="AlphaEvo Anti-Delusion Harness (Null Control + Pinned Scars)",
        require_null_control=True,
        enable_pinned_scars=True,
        require_portfolio_gate=portfolio_gate,
    )
    evaluator = MechanismEvaluator()
    exp = evaluator.evaluate_ab(
        seed_strategy,
        market_data,
        baseline_variant=naive,
        candidate_variant=guarded,
        max_rounds=rounds,
        cutoff_ratios=cutoff_list,
    )

    table = Table(title="L3 Meta-RSI Paired Mechanism AB Experiment (Forward Calibration)")
    table.add_column("Harness Variant")
    table.add_column("Champion")
    table.add_column("Promos", justify="right")
    table.add_column("Train AvgRet", justify="right")
    table.add_column("Fwd AvgRet", justify="right")
    table.add_column("Fwd Sharpe", justify="right")
    table.add_column("Decay Ratio", justify="right")
    table.add_column("False Promo", justify="center")
    table.add_column("Calibration", justify="right", style="bold")
    for run in (exp.baseline_run, exp.candidate_run):
        decay_str = f"{run.decay_ratio:.2f}x" if run.decay_ratio is not None else "n/a"
        table.add_row(
            run.variant_name,
            run.champion_id,
            str(run.promotions_count),
            f"{run.train_avg_return:.2%}",
            f"{run.forward_avg_return:.2%}",
            f"{run.forward_sharpe:.2f}",
            decay_str,
            "YES" if run.false_promotion else "NO",
            f"{run.calibration_score:.4f}",
        )
    console.print(table)
    console.print(f"[bold]Winner:[/bold] {exp.winner_variant_id} — {exp.rationale}")
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(evaluator.render_markdown(exp), encoding="utf-8")
        console.print(f"[dim]Report written to {out_path}[/dim]")


@rsi_app.command("benchmark")
def rsi_benchmark(
    strategies: str | None = typer.Option(
        None,
        "--strategies",
        "-s",
        help="Comma-separated strategy IDs/YAML paths (defaults to 4 core OHLCV templates)",
    ),
    rounds: int = typer.Option(2, "--rounds", "-r", help="Evolution rounds per ablation arm"),
    holdout_pct: float = typer.Option(
        0.20, "--holdout-pct", help="Chronological locked OOS holdout fraction"
    ),
    live: bool = typer.Option(False, "--live", help="Fetch live market data"),
    null_audit: bool = typer.Option(
        True,
        "--null-audit/--no-null-audit",
        help="Verify 0 false promotions under shuffled returns",
    ),
    output_dir: str = typer.Option(
        "reports/benchmark",
        "--output",
        "-o",
        help="Output directory for benchmark_report.md & .json",
    ),
) -> None:
    """Run controlled Baseline vs L1-only vs L1+L2+Crossover ablation benchmark."""
    from alphaevo.cli.demo import _find_builtin_strategy_dir
    from alphaevo.rsi.benchmark import render_benchmark_markdown, run_evolution_benchmark
    from alphaevo.strategy.dsl.parser import StrategyParser

    config = _get_config()
    strat_list: list[Strategy] = []
    if strategies:
        for ref in [part.strip() for part in strategies.split(",") if part.strip()]:
            strat_list.append(_resolve_strategy(ref, config))
    else:
        bdir = _find_builtin_strategy_dir()
        parser = StrategyParser()
        for stem in (
            "rsi_reversion",
            "ma_crossover",
            "mean_reversion_oversold",
            "volatility_compression_breakout",
        ):
            if bdir is not None and (bdir / f"{stem}.yaml").is_file():
                strat_list.append(parser.parse_file(bdir / f"{stem}.yaml"))

    if not strat_list:
        console.print("[red]Error: no strategies resolved for benchmark[/red]")
        raise typer.Exit(1)

    market_data = _load_market_data(live=live)
    report = run_evolution_benchmark(
        strat_list,
        market_data,
        rounds=rounds,
        locked_holdout_pct=holdout_pct,
        run_null_audit=null_audit,
        data_source=f"live:{config.data.adapter}" if live else "bundled_snapshot",
        backtest_config=config.backtest,
    )

    table = Table(title="AlphaEvo Self-Evolution Ablation Benchmark")
    table.add_column("Strategy", style="cyan")
    table.add_column("Baseline Fit (Sig)", justify="right")
    table.add_column("L1-Only Fit (Holdout)", justify="right")
    table.add_column("L1+L2+XOver Fit (Holdout)", justify="right", style="bold")
    table.add_column("Niches", justify="right")
    for row in report.rows:
        l1_ho = (
            f"{row.l1_only.holdout_avg_return:+.2%}"
            if row.l1_only.holdout_avg_return is not None
            else "n/a"
        )
        f_ho = (
            f"{row.l1_l2_crossover.holdout_avg_return:+.2%}"
            if row.l1_l2_crossover.holdout_avg_return is not None
            else "n/a"
        )
        table.add_row(
            row.strategy_id,
            f"{row.baseline.fitness_score:.3f} ({row.baseline.signals})",
            f"{row.l1_only.fitness_score:.3f} ({l1_ho})",
            f"{row.l1_l2_crossover.fitness_score:.3f} ({f_ho})",
            str(row.l1_l2_crossover.map_elites_cells),
        )
    console.print(table)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    md_file = out_path / "benchmark_report.md"
    json_file = out_path / "benchmark_report.json"
    md_file.write_text(render_benchmark_markdown(report), encoding="utf-8")
    json_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(
        f"[green]✓ Benchmark saved:[/green] {md_file}, {json_file} | "
        f"Mean Fitness: {report.mean_baseline_fitness:.3f} -> {report.mean_l1_fitness:.3f} -> {report.mean_full_fitness:.3f}"
    )


# ── alphaevo shadow ... ─────────────────────────────────────────────────


@shadow_app.command("register")
def shadow_register(
    strategy_id: str = typer.Argument(
        ..., help="Strategy ID or YAML path to register in Shadow Ledger"
    ),
) -> None:
    """Preregister a strategy candidate into the Forward Shadow Ledger."""
    from alphaevo.alpha_factory.factory import load_registered_factors
    from alphaevo.forward.shadow_ledger import ShadowLedger
    from alphaevo.truth.kernel import EvaluationKernel

    config = _get_config()
    if str(config.db_path) != ":memory:" and Path(config.db_path).expanduser().is_file():
        load_registered_factors(config.db_path, status="active")
    strat = _resolve_strategy(strategy_id, config)
    market_data = _load_market_data(live=False)
    kernel = EvaluationKernel(backtest_config=config.backtest, min_data_days=15)
    outcome = kernel.evaluate(strat, market_data, mode="fast", record_trial=False)
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    spec = ledger.register_candidate(strat, outcome.evaluation)
    console.print(
        f"[green]✓ Registered[/green] {spec.strategy_id} (stage={spec.stage.value}, hash={spec.strategy_hash[:12]})"
    )


@shadow_app.command("emit")
def shadow_emit(
    strategy_id: str = typer.Argument(..., help="Registered strategy ID or YAML path"),
    as_of: str | None = typer.Option(None, "--as-of", help="Point-in-time date YYYY-MM-DD"),
    live: bool = typer.Option(False, "--live", help="Fetch live market data"),
    replay: bool | None = typer.Option(
        None,
        "--replay/--no-replay",
        help="Mark backdated commit as replay (defaults to True when --as-of is given without --no-replay)",
    ),
) -> None:
    """Emit point-in-time forward setups strictly truncated to ``--as-of``."""
    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.alpha_factory.factory import load_registered_factors
    from alphaevo.forward.shadow_ledger import ShadowLedger

    config = _get_config()
    if str(config.db_path) != ":memory:" and Path(config.db_path).expanduser().is_file():
        load_registered_factors(config.db_path, status="active")
    strat = _resolve_strategy(strategy_id, config)
    market_data = _load_market_data(live=live)
    max_avail = max(
        pd.to_datetime(df["date"]).dt.date.max()
        for df in market_data.values()
        if not df.empty and "date" in df
    )
    if replay is False and not live:
        raise typer.BadParameter("Live non-replay commitments require --live market data.")
    eval_date = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else max_avail
    is_replay = replay if replay is not None else (not live or as_of is not None)
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    commit = ledger.commit_signals(
        strat,
        market_data,
        as_of=eval_date,
        is_replay=is_replay,
        enforce_wall_clock=not is_replay,
    )
    console.print(
        f"[green]✓ Committed[/green] {commit.commit_id} | as_of={commit.as_of} | "
        f"raw={commit.total_raw_candidates} | capped={commit.sector_capped_count} | "
        f"hash={commit.record_hash[:12]}"
    )


@shadow_app.command("backfill")
def shadow_backfill(
    as_of: str | None = typer.Option(None, "--as-of", help="Backfill cutoff date YYYY-MM-DD"),
    live: bool = typer.Option(False, "--live", help="Fetch live market data"),
) -> None:
    """Backfill T+5/10/20 forward outcomes up to ``--as-of`` (immature signals stay excluded)."""
    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.forward.shadow_ledger import ShadowLedger

    config = _get_config()
    market_data = _load_market_data(live=live)
    max_avail = max(
        pd.to_datetime(df["date"]).dt.date.max()
        for df in market_data.values()
        if not df.empty and "date" in df
    )
    eval_date = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else max_avail
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    bench_df = (
        ShadowLedger.build_equal_weight_benchmark(market_data) if len(market_data) >= 2 else None
    )
    events = ledger.backfill_outcomes(
        market_data,
        evaluated_as_of=eval_date,
        benchmark_df=bench_df,
    )
    mature_10 = sum(1 for ev in events if 10 in ev.horizons and ev.horizons[10].mature)
    console.print(
        f"[green]✓ Backfilled[/green] {len(events)} signals as of {eval_date} "
        f"(T+10 mature: {mature_10}, immature: {len(events) - mature_10})"
    )


@shadow_app.command("status")
def shadow_status(
    strategy_id: str | None = typer.Argument(None, help="Optional strategy filter"),
    horizon: int = typer.Option(10, "--horizon", help="Horizon days: 5, 10, or 20"),
) -> None:
    """Display Forward Shadow Ledger decay ratios, N_eff, and stage status."""
    from alphaevo.forward.shadow_ledger import ShadowLedger

    config = _get_config()
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    candidates = ledger.list_candidates()
    if strategy_id:
        candidates = [c for c in candidates if c.strategy_id == strategy_id]

    table = Table(title=f"Forward Shadow Ledger Status (T+{horizon})")
    table.add_column("Strategy ID")
    table.add_column("Stage")
    table.add_column("Days", justify="right")
    table.add_column("Mature", justify="right")
    table.add_column("N_eff", justify="right")
    table.add_column("Fwd Return", justify="right")
    table.add_column("Decay Ratio", justify="right")
    table.add_column("Max DD", justify="right")

    for cand in candidates:
        rep = ledger.build_decay_report(cand.strategy_id, horizon_days=horizon)
        decay_txt = f"{rep.decay_ratio:.2f}x" if rep.decay_ratio is not None else "n/a"
        table.add_row(
            cand.strategy_id,
            cand.stage.value,
            str(rep.mature_trading_days),
            str(rep.mature_signals),
            f"{rep.effective_signals:.1f}",
            f"{rep.forward_avg_post_cost_return:.2%}",
            decay_txt,
            f"{rep.max_forward_drawdown:.1%}",
        )
    console.print(table)


@shadow_app.command("promote")
def shadow_promote(
    strategy_id: str = typer.Argument(..., help="Registered strategy ID to evaluate for promotion"),
    horizon: int = typer.Option(10, "--horizon", help="Evaluation horizon: 5, 10, or 20"),
    require_benchmark: bool = typer.Option(
        False,
        "--require-benchmark/--no-require-benchmark",
        help="Require >=80% benchmark coverage for excess-return promotion gate",
    ),
) -> None:
    """Evaluate asymmetric promotion/halt gates for a Forward Shadow candidate."""
    from alphaevo.forward.shadow_ledger import ShadowLedger

    config = _get_config()
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    decision = ledger.evaluate_promotion(
        strategy_id,
        horizon_days=horizon,
        min_trading_days=config.rsi.min_shadow_trading_days,
        min_mature_signals=config.rsi.min_shadow_signals,
        min_effective_signals=float(config.rsi.min_effective_signals),
        require_benchmark=require_benchmark,
    )
    console.print(
        f"[bold]{decision.strategy_id}[/bold]: {decision.previous_stage.value} -> "
        f"[bold cyan]{decision.new_stage.value}[/bold cyan] ({decision.action.upper()})"
    )
    if decision.blocking_reasons:
        for reason in decision.blocking_reasons:
            console.print(f"  [yellow]• Blocked: {reason}[/yellow]")


@shadow_app.command("verify")
def shadow_verify() -> None:
    """Verify cryptographic SHA-256 hash chain integrity of commits.jsonl."""
    from alphaevo.forward.shadow_ledger import ShadowLedger

    config = _get_config()
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    ok, errs = ledger.verify_chain()
    if ok:
        commits = ledger.load_commits()
        console.print(f"[green]✓ Hash chain verified cleanly ({len(commits)} commits)[/green]")
    else:
        for err in errs:
            console.print(f"[red]✗ {err}[/red]")
        raise typer.Exit(1)


@shadow_app.command("simulate")
def shadow_simulate(
    strategy_id: str = typer.Argument(
        "rsi_reversion_v1", help="Strategy ID or YAML path to simulate in Shadow Ledger"
    ),
    days: int = typer.Option(45, "--days", help="Number of trailing trading days to simulate"),
    step_days: int = typer.Option(1, "--step-days", help="Simulate every N trading days"),
    horizon: int = typer.Option(10, "--horizon", help="Outcome horizon: 5, 10, or 20"),
    min_days: int = typer.Option(10, "--min-days", help="Min mature trading days to promote"),
    min_signals: int = typer.Option(10, "--min-signals", help="Min mature signals to promote"),
    min_eff: float = typer.Option(5.0, "--min-eff", help="Min cluster-adjusted N_eff to promote"),
    live: bool = typer.Option(False, "--live", help="Fetch live market data"),
    output_file: str | None = typer.Option(
        None, "--output", "-o", help="Optional Markdown report path"
    ),
) -> None:
    """Simulate chronological day-by-day bitemporal shadow commitments & promotion gates."""
    import tempfile

    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.alpha_factory.factory import load_registered_factors
    from alphaevo.forward.shadow_ledger import ShadowLedger

    config = _get_config()
    if str(config.db_path) != ":memory:" and Path(config.db_path).expanduser().is_file():
        load_registered_factors(config.db_path, status="active")
    try:
        strat = _resolve_strategy(strategy_id, config)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None

    market_data = _load_market_data(live=live)
    all_dates = sorted(
        {
            d
            for df in market_data.values()
            if not df.empty and "date" in df.columns
            for d in pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().tolist()
        }
    )
    if not all_dates:
        console.print("[red]No market dates available for shadow simulation.[/red]")
        raise typer.Exit(1)

    start_d = all_dates[max(0, len(all_dates) - max(5, days))]
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = ShadowLedger(tmpdir)
        report = ledger.simulate_rolling(
            strat,
            market_data,
            start_date=start_d,
            end_date=all_dates[-1],
            step_days=step_days,
            horizon_days=horizon,
            min_trading_days=min_days,
            min_mature_signals=min_signals,
            min_effective_signals=min_eff,
            is_replay=not live,
        )

    decay = report.final_decay_report
    decay_txt = f"{decay.decay_ratio:.2f}x" if decay.decay_ratio is not None else "n/a"
    table = Table(title=f"Rolling Forward Shadow Simulation — {strat.meta.id}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Window", f"{report.start_date} -> {report.end_date}")
    table.add_row("Stage Transition", f"{report.initial_stage.value} -> {report.final_stage.value}")
    table.add_row("Steps / Commits", f"{report.total_steps} / {report.total_commits}")
    table.add_row("Committed Signals", str(report.total_committed_signals))
    table.add_row(f"Mature Signals (T+{report.horizon_days})", str(decay.mature_signals))
    table.add_row("Effective Sample N_eff", f"{decay.effective_signals:.1f}")
    table.add_row("Forward Avg Return", f"{decay.forward_avg_post_cost_return:+.2%}")
    table.add_row("Decay Ratio vs Backtest", decay_txt)
    table.add_row("Max Forward Drawdown", f"{decay.max_forward_drawdown:.1%}")
    table.add_row("SHA-256 Chain Integrity", "VERIFIED" if report.chain_verified else "FAILED")
    console.print(table)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(ShadowLedger.render_simulation_markdown(report), encoding="utf-8")
        console.print(f"[dim]Report written to {out_path}[/dim]")


# ── alphaevo truth ... ──────────────────────────────────────────────────


@truth_app.command("trials")
def truth_trials(
    family: str | None = typer.Option(None, "--family", "-f", help="Filter by trial_family_id"),
    limit: int = typer.Option(25, "--limit", "-n", help="Max rows to display"),
) -> None:
    """Inspect L0 TrialLedger accounting and Deflated Sharpe probabilities."""
    from alphaevo.truth.ledger import TrialLedger

    config = _get_config()
    ledger = TrialLedger(config.db_path)
    records = ledger.list_trials(trial_family_id=family, limit=limit)

    table = Table(title="L0 Truth Kernel — TrialLedger")
    table.add_column("Family")
    table.add_column("Strategy")
    table.add_column("Lvl")
    table.add_column("Signals", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("DSR Prob", justify="right")
    table.add_column("Null")
    table.add_column("Keep")

    for rec in records:
        null_txt = (
            "PASS"
            if rec.passed_null_control is True
            else ("FAIL" if rec.passed_null_control is False else "—")
        )
        table.add_row(
            rec.trial_family_id,
            rec.strategy_id,
            rec.rsi_level.value,
            str(rec.signal_count),
            f"{rec.sharpe_ratio:.2f}",
            f"{rec.deflated_sharpe_prob:.1%}",
            null_txt,
            "yes" if rec.accepted else "no",
        )
    console.print(table)


@truth_app.command("pit-probe")
def truth_pit_probe(
    factor_file: Annotated[
        Path,
        typer.Argument(help="Python file defining compute(df, idx) -> float"),
    ],
) -> None:
    """Run AST whitelist + runtime future-noise/truncation PIT probe on factor code."""
    from alphaevo.truth.pit_probe import probe_point_in_time_invariance

    if not factor_file.is_file():
        console.print(f"[red]File not found: {factor_file}[/red]")
        raise typer.Exit(1)
    code = factor_file.read_text(encoding="utf-8")
    data = _load_market_data(live=False)
    sample_df = next(iter(data.values()))
    res = probe_point_in_time_invariance(code, sample_df)
    if res.passed:
        console.print(
            f"[green]✓ PIT invariance PASSED[/green] (checked indices={res.checked_indices})"
        )
        return
    console.print("[red]✗ PIT invariance FAILED[/red]")
    if res.error:
        console.print(f"  Error: {res.error}")
    for v in res.violations:
        console.print(
            f"  Violation [{v.probe_type}] at idx={v.probe_index}: "
            f"clean={v.clean_value} vs perturbed={v.perturbed_value} (diff={v.abs_diff:.4g})"
        )
    raise typer.Exit(1)


__all__ = ["rsi_app", "shadow_app", "truth_app"]
