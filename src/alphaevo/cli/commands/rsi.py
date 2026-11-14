"""CLI commands for Dual-Clock Stock RSI, Shadow Ledger, and L0 Truth Kernel."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import pandas as pd  # type: ignore[import-untyped]
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.cli._helpers import _get_config
from alphaevo.cli.demo import _find_builtin_strategy_dir, load_showcase_snapshot
from alphaevo.core.config import ConfigManager
from alphaevo.data.adapter import DataManager, get_adapter_chain
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.forward.shadow_ledger import ShadowLedger
from alphaevo.models.enums import RSILevel, SamplingMethod
from alphaevo.models.execution import SampleBatch
from alphaevo.models.strategy import Strategy
from alphaevo.rsi.council import (
    HeterogeneousCouncil,
    HeterogeneousCouncilConfig,
    ModelSpec,
)
from alphaevo.rsi.engine import StockRSIEngine
from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator
from alphaevo.rsi.program import RSICharter, parse_rsi_program
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.store import StrategyStore
from alphaevo.truth.ledger import TrialLedger
from alphaevo.truth.null_control import NullSignalController
from alphaevo.truth.pit_probe import probe_point_in_time_invariance

if TYPE_CHECKING:
    from alphaevo.core.config import AppConfig

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
        data, _ = load_showcase_snapshot()
        return data

    config = ConfigManager().load()
    chain = get_adapter_chain(config)
    mgr = DataManager(adapters=chain)
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
        help="Proposer model (e.g. openai/gpt-4o or codex:gpt-6-astra)",
    ),
    council_prosecutor: str | None = typer.Option(
        None,
        "--council-prosecutor",
        help="Prosecutor model from a different family (e.g. anthropic/claude-3-5-sonnet or agy:claude-opus-5-5-high)",
    ),
    live: bool = typer.Option(False, "--live", help="Fetch live market data instead of snapshot"),
    output_dir: str = typer.Option(
        "reports/rsi",
        "--output",
        "-o",
        help="Directory for results.tsv, Markdown report, and champion YAML",
    ),
) -> None:
    """Run the L0-grounded Stock RSI evolution ratchet loop."""
    config = _get_config()
    try:
        seed_strategy = _resolve_strategy(strategy_id, config)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None

    if program is not None:
        charter = parse_rsi_program(program)
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

    engine = StockRSIEngine(db_path=config.db_path, council=council)
    result = engine.run(seed_strategy, market_data, charter=charter, output_dir=Path(output_dir))

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
) -> None:
    """L3 Meta-RSI paired AB experiment comparing harness calibration on unseen forward bars."""
    config = _get_config()
    seed_strategy = _resolve_strategy(strategy_id, config)
    market_data = _load_market_data(live=False)

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
    )
    exp = MechanismEvaluator().evaluate_ab(
        seed_strategy,
        market_data,
        baseline_variant=naive,
        candidate_variant=guarded,
        max_rounds=rounds,
    )

    table = Table(title="L3 Meta-RSI Paired Mechanism AB Experiment (Forward Calibration)")
    table.add_column("Harness Variant")
    table.add_column("Champion")
    table.add_column("Train AvgRet", justify="right")
    table.add_column("Fwd AvgRet", justify="right")
    table.add_column("Decay Ratio", justify="right")
    table.add_column("False Promo", justify="center")
    table.add_column("Calibration", justify="right", style="bold")
    for run in (exp.baseline_run, exp.candidate_run):
        decay_str = f"{run.decay_ratio:.2f}x" if run.decay_ratio is not None else "n/a"
        table.add_row(
            run.variant_name,
            run.champion_id,
            f"{run.train_avg_return:.2%}",
            f"{run.forward_avg_return:.2%}",
            decay_str,
            "YES" if run.false_promotion else "NO",
            f"{run.calibration_score:.4f}",
        )
    console.print(table)
    console.print(f"[bold]Winner:[/bold] {exp.winner_variant_id} — {exp.rationale}")


# ── alphaevo shadow ... ─────────────────────────────────────────────────


@shadow_app.command("register")
def shadow_register(
    strategy_id: str = typer.Argument(
        ..., help="Strategy ID or YAML path to register in Shadow Ledger"
    ),
) -> None:
    """Preregister a strategy candidate into the Forward Shadow Ledger."""
    config = _get_config()
    strat = _resolve_strategy(strategy_id, config)
    market_data = _load_market_data(live=False)
    ds = [
        pd.to_datetime(df["date"]).dt.date
        for df in market_data.values()
        if not df.empty and "date" in df
    ]
    start_d = min(s.min() for s in ds)
    end_d = max(s.max() for s in ds)
    batch = SampleBatch(
        batch_id="shadow_reg",
        strategy_id=strat.meta.id,
        symbols=sorted(market_data.keys()),
        date_range=(start_d, end_d),
        sampling_method=SamplingMethod.REPRESENTATIVE,
    )
    bt = BacktestEngine(min_data_days=15).run(strat, market_data, batch)
    ev = Evaluator().evaluate_fast(bt, strat)
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    spec = ledger.register_candidate(strat, ev)
    console.print(
        f"[green]✓ Registered[/green] {spec.strategy_id} (stage={spec.stage.value}, hash={spec.strategy_hash[:12]})"
    )


@shadow_app.command("emit")
def shadow_emit(
    strategy_id: str = typer.Argument(..., help="Registered strategy ID or YAML path"),
    as_of: str | None = typer.Option(None, "--as-of", help="Point-in-time date YYYY-MM-DD"),
    live: bool = typer.Option(False, "--live", help="Fetch live market data"),
) -> None:
    """Emit point-in-time forward setups strictly truncated to ``--as-of``."""
    config = _get_config()
    strat = _resolve_strategy(strategy_id, config)
    market_data = _load_market_data(live=live)
    max_avail = max(
        pd.to_datetime(df["date"]).dt.date.max()
        for df in market_data.values()
        if not df.empty and "date" in df
    )
    eval_date = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else max_avail
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    commit = ledger.commit_signals(strat, market_data, as_of=eval_date)
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
    config = _get_config()
    market_data = _load_market_data(live=live)
    max_avail = max(
        pd.to_datetime(df["date"]).dt.date.max()
        for df in market_data.values()
        if not df.empty and "date" in df
    )
    eval_date = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else max_avail
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    events = ledger.backfill_outcomes(market_data, evaluated_as_of=eval_date)
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
) -> None:
    """Evaluate asymmetric promotion/halt gates for a Forward Shadow candidate."""
    config = _get_config()
    ledger = ShadowLedger(config.rsi.shadow_ledger_dir)
    decision = ledger.evaluate_promotion(
        strategy_id,
        horizon_days=horizon,
        min_trading_days=config.rsi.min_shadow_trading_days,
        min_mature_signals=config.rsi.min_shadow_signals,
        min_effective_signals=float(config.rsi.min_effective_signals),
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


# ── alphaevo truth ... ──────────────────────────────────────────────────


@truth_app.command("trials")
def truth_trials(
    family: str | None = typer.Option(None, "--family", "-f", help="Filter by trial_family_id"),
    limit: int = typer.Option(25, "--limit", "-n", help="Max rows to display"),
) -> None:
    """Inspect L0 TrialLedger accounting and Deflated Sharpe probabilities."""
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
