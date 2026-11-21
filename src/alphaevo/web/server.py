"""Zero-dependency stdlib HTTP server & JSON service layer for ``alphaevo web``."""

from __future__ import annotations

import json
import logging
import queue
import tempfile
import threading
from collections.abc import Callable
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

from alphaevo import __version__
from alphaevo.core.config import AppConfig
from alphaevo.leaderboard.service import LeaderboardService
from alphaevo.provenance.store import RunArtifactStore
from alphaevo.web.contracts import build_strategy_card, default_web_manifest
from alphaevo.web.ui import render_web_studio_html

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.models.strategy import Strategy

logger = logging.getLogger(__name__)
_MAX_BODY_BYTES = 2 * 1024 * 1024


class AlphaEvoWebService:
    """Typed service backend powering both the HTTP Web Studio and MCP tools."""

    def __init__(
        self,
        config: AppConfig,
        *,
        report_dir: Path | str = "reports",
        live_data: bool = False,
    ) -> None:
        self.config = config
        self.report_dir = Path(report_dir)
        self.live_data = live_data

    def health(self) -> dict[str, Any]:
        """Return API service health and web capability manifest."""
        return {
            "status": "ok",
            "version": __version__,
            "manifest": default_web_manifest().model_dump(mode="json"),
        }

    def list_strategies(self) -> dict[str, Any]:
        """List stored and built-in strategies as web cards."""
        from alphaevo.cli.demo import _find_builtin_strategy_dir
        from alphaevo.strategy.dsl.parser import StrategyParser
        from alphaevo.strategy.store import StrategyStore

        seen: dict[str, Strategy] = {}
        try:
            store = StrategyStore(self.config.db_path)
            for s in store.list_all():
                seen[s.meta.id] = s
        except Exception:
            pass

        bdir = _find_builtin_strategy_dir()
        if bdir is not None:
            parser = StrategyParser()
            for s in parser.parse_directory(bdir):
                seen.setdefault(s.meta.id, s)

        cards = [build_strategy_card(s).model_dump(mode="json") for s in seen.values()]
        cards.sort(key=lambda item: str(item.get("strategy_id", "")))
        return {"count": len(cards), "strategies": cards}

    def get_strategy_yaml(self, strategy_id: str) -> dict[str, Any]:
        """Resolve a strategy and return its YAML DSL text."""
        from alphaevo.cli.commands.rsi import _resolve_strategy
        from alphaevo.strategy.dsl.serializer import StrategySerializer

        try:
            strat = _resolve_strategy(strategy_id, self.config)
        except ValueError as exc:
            raise FileNotFoundError(str(exc)) from exc
        return {
            "strategy_id": strat.meta.id,
            "name": strat.meta.name,
            "description": strat.description,
            "yaml": StrategySerializer().to_yaml(strat),
        }

    def draft_strategy(
        self,
        idea: str,
        *,
        market: str = "a_share",
        strategy_id: str | None = None,
        save: bool = False,
    ) -> dict[str, Any]:
        """Compile a plain-language idea into an executable strategy YAML."""
        from alphaevo.strategy import StrategyDraftBuilder, StrategySerializer
        from alphaevo.strategy.store import StrategyStore

        if not idea.strip():
            raise ValueError("Strategy idea text cannot be empty")
        strategy = StrategyDraftBuilder().from_text(
            idea,
            market=market,
            strategy_id=strategy_id,
        )
        if save:
            StrategyStore(self.config.db_path).save(strategy)
        return {
            "strategy_id": strategy.meta.id,
            "name": strategy.meta.name,
            "market": strategy.meta.market.value,
            "saved": save,
            "yaml": StrategySerializer().to_yaml(strategy),
        }

    def save_strategy(self, *, strategy_yaml: str) -> dict[str, Any]:
        """Validate and persist a YAML DSL strategy into SQLite StrategyStore."""
        from alphaevo.strategy.dsl.parser import StrategyParser
        from alphaevo.strategy.store import StrategyStore

        if not strategy_yaml.strip():
            raise ValueError("strategy_yaml cannot be empty")
        strategy = StrategyParser().parse_yaml(strategy_yaml)
        StrategyStore(self.config.db_path).save(strategy)
        return {
            "strategy_id": strategy.meta.id,
            "name": strategy.meta.name,
            "version": strategy.meta.version,
            "market": strategy.meta.market.value,
            "saved": True,
        }

    def run_strategy(
        self,
        *,
        strategy_id: str | None = None,
        strategy_yaml: str | None = None,
        live: bool | None = None,
        protocol: str = "full_research_v1",
    ) -> dict[str, Any]:
        """Evaluate a strategy through EvaluationKernel and persist run artifacts."""
        from alphaevo.arena.html_renderer import render_equity_svg, render_trade_kline_svg
        from alphaevo.arena.runner import ArenaRunner
        from alphaevo.research_committee import ResearchCommittee
        from alphaevo.strategy.dsl.serializer import StrategySerializer
        from alphaevo.truth.kernel import EvaluationKernel

        strategy = self._resolve_input_strategy(
            strategy_id=strategy_id, strategy_yaml=strategy_yaml
        )
        market_data, source_lbl, repro, mutability = self._load_market_data(live=live)

        kernel = EvaluationKernel(
            backtest_config=self.config.backtest,
            db_path=self.config.db_path,
            min_data_days=min(15, self.config.backtest.min_data_days),
        )
        outcome = kernel.evaluate(
            strategy,
            market_data,
            mode="full",
            record_trial=True,
            run_null_control=True,
            null_permutations=25,
            notes="web_run",
        )
        committee = ResearchCommittee().review(
            strategy,
            outcome.evaluation,
            data_source=source_lbl,
            symbols=sorted(market_data.keys()),
        )
        eq_points = ArenaRunner.compute_equity_curve(outcome.backtest_result.signals, market_data)
        eq_svg = render_equity_svg(eq_points)
        replays = ArenaRunner.build_trade_replays(
            outcome.backtest_result.signals, market_data, max_trades=15
        )
        replay_items = [
            {**rep.model_dump(mode="json"), "kline_svg": render_trade_kline_svg(rep)}
            for rep in replays
        ]

        req_proto: Literal["fast_research_v1", "full_research_v1", "official_v1"] = (
            protocol  # type: ignore[assignment]
            if protocol in {"fast_research_v1", "full_research_v1", "official_v1"}
            else "full_research_v1"
        )
        art_store = RunArtifactStore(artifact_root=self.report_dir, db_path=self.config.db_path)
        manifest = art_store.save_run(
            kind="run",
            strategy=strategy,
            evaluation=outcome.evaluation,
            market_data=market_data,
            trades=outcome.backtest_result.signals,
            config=self.config,
            adapter_name=source_lbl,
            requested_protocol=req_proto,
            data_reproducibility=repro,
            data_mutability=mutability,
            min_signals_official=min(15, self.config.evolution.min_signal_count),
        )
        ov = outcome.evaluation.overall
        return {
            "run_id": manifest.run_id,
            "requested_protocol": manifest.requested_protocol,
            "effective_protocol": manifest.effective_protocol,
            "strategy_id": strategy.meta.id,
            "strategy_yaml": StrategySerializer().to_yaml(strategy),
            "evaluation": {
                "confidence_score": outcome.evaluation.confidence_score,
                "win_rate": ov.win_rate,
                "avg_return": ov.avg_return,
                "total_return": ov.total_return,
                "max_drawdown": ov.max_drawdown,
                "sharpe_ratio": ov.sharpe_ratio,
                "signal_count": ov.signal_count,
                "deflated_sharpe_prob": outcome.evaluation.deflated_sharpe_prob,
                "null_control_passed": outcome.evaluation.null_control_passed,
            },
            "equity_curve": [p.model_dump(mode="json") for p in eq_points],
            "equity_svg": eq_svg,
            "committee_verdicts": [v.model_dump(mode="json") for v in committee.verdicts],
            "trades": [t.model_dump(mode="json") for t in outcome.backtest_result.signals[:100]],
            "trade_replays": replay_items,
        }

    @staticmethod
    def _build_council(
        proposer_model: str | None,
        prosecutor_model: str | None,
        api_key: str | None,
    ) -> Any:
        """Build optional HeterogeneousCouncil from session parameters."""
        from alphaevo.rsi.council import (
            HeterogeneousCouncil,
            HeterogeneousCouncilConfig,
            ModelSpec,
        )

        p_raw = (proposer_model or "").strip()
        q_raw = (prosecutor_model or "").strip()
        if (not p_raw or p_raw == "heuristic") and (not q_raw or q_raw == "heuristic"):
            return None

        p_eff = p_raw or "heuristic"
        q_eff = q_raw or "heuristic"
        p_spec = ModelSpec(
            model=p_eff,
            backend="deterministic" if p_eff.lower().startswith("heuristic") else "litellm",
        )
        q_spec = ModelSpec(
            model=q_eff,
            backend="deterministic" if q_eff.lower().startswith("heuristic") else "litellm",
        )
        cfg = HeterogeneousCouncilConfig(proposer=p_spec, prosecutor=q_spec)
        clean_key = (api_key or "").strip() or None
        return HeterogeneousCouncil(config=cfg, api_key=clean_key)

    def diagnose_strategy(
        self,
        *,
        strategy_id: str | None = None,
        strategy_yaml: str | None = None,
        live: bool | None = None,
    ) -> dict[str, Any]:
        """Measure per-condition structural pass rates without consuming L0 trials."""
        from alphaevo.rsi.diagnostics import analyze_entry_bottlenecks

        strategy = self._resolve_input_strategy(
            strategy_id=strategy_id, strategy_yaml=strategy_yaml
        )
        market_data, _src, _repro, _mut = self._load_market_data(live=live)
        report = analyze_entry_bottlenecks(strategy, market_data)
        return report.model_dump(mode="json")

    def run_rsi(
        self,
        *,
        strategy_id: str | None = None,
        strategy_yaml: str | None = None,
        rounds: int = 2,
        holdout_pct: float = 0.0,
        portfolio_gate: bool = False,
        live: bool | None = None,
        enable_l2: bool = True,
        proposer_model: str | None = None,
        prosecutor_model: str | None = None,
        api_key: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run StockRSIEngine evolution ratchet and return rounds, debates, replays & diff."""
        import difflib

        from alphaevo.arena.html_renderer import (
            render_equity_svg,
            render_evolution_tree_svg,
            render_map_elites_svg,
            render_trade_kline_svg,
        )
        from alphaevo.arena.runner import ArenaRunner
        from alphaevo.models.enums import RSILevel
        from alphaevo.rsi.diagnostics import analyze_entry_bottlenecks
        from alphaevo.rsi.engine import StockRSIEngine
        from alphaevo.rsi.program import RSICharter
        from alphaevo.rsi.trajectory import build_rsi_trajectory
        from alphaevo.strategy.dsl.serializer import StrategySerializer
        from alphaevo.truth.kernel import EvaluationKernel

        council = self._build_council(proposer_model, prosecutor_model, api_key)
        strategy = self._resolve_input_strategy(
            strategy_id=strategy_id, strategy_yaml=strategy_yaml
        )
        market_data, _src, _repro, _mut = self._load_market_data(live=live)
        bottleneck = analyze_entry_bottlenecks(strategy, market_data)

        allowed_levels = (
            [RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR] if enable_l2 else [RSILevel.L1_STRATEGY]
        )
        charter = RSICharter(
            name=f"web_rsi_{strategy.meta.id}",
            seed_strategy_id=strategy.meta.id,
            allowed_levels=allowed_levels,
            max_rounds=max(1, min(6, rounds)),
            max_candidates_per_round=3,
            max_trials_budget=max(10, rounds * 10 + 4),
            min_signals=min(15, self.config.backtest.min_data_days),
            require_null_control=True,
            null_permutations=25,
            locked_holdout_pct=holdout_pct,
            require_portfolio_gate=portfolio_gate,
        )
        l2_llm = (
            proposer_model.strip()
            if proposer_model
            and proposer_model.strip()
            and not proposer_model.strip().lower().startswith("heuristic")
            else None
        )
        engine = StockRSIEngine(
            db_path=self.config.db_path,
            backtest_config=self.config.backtest,
            council=council,
            enable_crossover=True,
            l2_llm_model=l2_llm,
            l2_api_key=(api_key or "").strip() or None,
        )
        res = engine.run(
            strategy,
            market_data,
            charter=charter,
            persist_champion=False,
            on_event=on_event,
        )
        champ = res.champion_strategy
        outcome = EvaluationKernel(
            backtest_config=self.config.backtest,
            db_path=self.config.db_path,
            min_data_days=min(15, self.config.backtest.min_data_days),
        ).evaluate(champ, market_data, mode="fast", record_trial=False, run_null_control=False)

        eq_points = ArenaRunner.compute_equity_curve(outcome.backtest_result.signals, market_data)
        replays = ArenaRunner.build_trade_replays(
            outcome.backtest_result.signals, market_data, max_trades=12
        )
        replay_items = [
            {**rep.model_dump(mode="json"), "kline_svg": render_trade_kline_svg(rep)}
            for rep in replays
        ]

        serializer = StrategySerializer()
        base_yaml = serializer.to_yaml(strategy)
        champ_yaml = serializer.to_yaml(champ)
        yaml_diff = list(
            difflib.unified_diff(
                base_yaml.splitlines(),
                champ_yaml.splitlines(),
                fromfile=f"{strategy.meta.id}.yaml",
                tofile=f"{champ.meta.id}.yaml",
                lineterm="",
            )
        )
        rows_json = [row.model_dump(mode="json") for row in res.rows]
        tree_nodes = [
            {
                "round_num": r["round_num"],
                "strategy_id": r["strategy_id"],
                "parent_id": r.get("parent_id"),
                "secondary_parent_id": r.get("secondary_parent_id"),
                "status": r["status"],
                "score": r["fitness_score"],
                "dsr_probability": r["dsr_probability"],
            }
            for r in rows_json
        ]
        map_cells = [
            {
                "regime_bucket": cell.niche.regime_bucket,
                "style_bucket": cell.niche.style_bucket,
                "horizon_bucket": cell.niche.horizon_bucket,
                "strategy_id": cell.strategy_id,
                "fitness_score": round(float(cell.fitness_score), 4),
            }
            for cell in res.archive_summary.cells
        ]
        traj_export = build_rsi_trajectory(res)
        return {
            "seed_strategy_id": strategy.meta.id,
            "champion_id": res.champion_id,
            "champion_promoted": res.champion_promoted,
            "champion_score": res.champion_evaluation.confidence_score,
            "promotions": res.promotions,
            "kept_rounds": res.kept_rounds,
            "total_trials_recorded": res.total_trials_recorded,
            "kernel_intact": res.kernel_intact,
            "rows": rows_json,
            "deliberations": [d.model_dump(mode="json") for d in res.deliberations],
            "veto_breakdown": dict(res.veto_breakdown),
            "persisted_factors": list(res.persisted_factors),
            "crossovers_tried": res.crossovers_tried,
            "crossovers_kept": res.crossovers_kept,
            "map_elites_cells": map_cells,
            "map_elites_svg": render_map_elites_svg(map_cells),
            "bottleneck_report": bottleneck.model_dump(mode="json"),
            "dpo_pairs": [p.model_dump(mode="json") for p in traj_export.dpo_pairs],
            "evolution_tree_svg": render_evolution_tree_svg(tree_nodes),
            "trade_replays": replay_items,
            "champion_yaml": champ_yaml,
            "yaml_diff": yaml_diff,
            "equity_curve": [p.model_dump(mode="json") for p in eq_points],
            "equity_svg": render_equity_svg(eq_points),
        }

    def run_mechanism_ab(
        self,
        *,
        strategy_id: str | None = None,
        strategy_yaml: str | None = None,
        rounds: int = 2,
        cutoffs: list[float] | None = None,
        live: bool | None = None,
    ) -> dict[str, Any]:
        """Run paired chronological L3 Meta-RSI harness comparison (naive vs strict L0+Scars+L2)."""
        from alphaevo.models.enums import RSILevel
        from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator

        strategy = self._resolve_input_strategy(
            strategy_id=strategy_id, strategy_yaml=strategy_yaml
        )
        market_data, _src, _repro, _mut = self._load_market_data(live=live)
        baseline_var = HarnessVariant(
            variant_id="naive_off",
            name="Unconstrained (No Null Control / No Scars)",
            require_null_control=False,
            enable_pinned_scars=False,
            complexity_limit=8,
            min_signal_gate=5,
            allowed_levels=[RSILevel.L1_STRATEGY],
        )
        candidate_var = HarnessVariant(
            variant_id="l0_scars_l2",
            name="Full L0 Null + Pinned Scars + L2 Co-STEER",
            require_null_control=True,
            enable_pinned_scars=True,
            complexity_limit=8,
            min_signal_gate=15,
            allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
        )
        rep = MechanismEvaluator().evaluate_ab(
            strategy,
            market_data,
            baseline_variant=baseline_var,
            candidate_variant=candidate_var,
            max_rounds=max(1, min(4, rounds)),
            cutoff_ratios=cutoffs or [0.55, 0.70],
        )
        return {
            "result": rep.model_dump(mode="json"),
            "markdown": MechanismEvaluator.render_markdown(rep),
        }

    def run_benchmark(
        self,
        *,
        strategy_ids: list[str] | None = None,
        rounds: int = 2,
        holdout_pct: float = 0.20,
        live: bool | None = None,
        run_null_audit: bool = True,
    ) -> dict[str, Any]:
        """Run multi-strategy Baseline vs L1 vs L1+L2+Crossover ablation benchmark."""
        from alphaevo.cli.commands.arena import _load_arena_strategies
        from alphaevo.rsi.benchmark import render_benchmark_markdown, run_evolution_benchmark

        refs = ",".join(strategy_ids) if strategy_ids else None
        strategies = _load_arena_strategies(refs, self.config)
        market_data, src_lbl, _repro, _mut = self._load_market_data(live=live)
        rep = run_evolution_benchmark(
            strategies,
            market_data,
            rounds=max(1, min(4, rounds)),
            locked_holdout_pct=max(0.0, min(0.4, holdout_pct)),
            run_null_audit=run_null_audit,
            null_worlds=1,
            data_source=src_lbl,
            backtest_config=self.config.backtest,
        )
        return {
            "result": rep.model_dump(mode="json"),
            "markdown": render_benchmark_markdown(rep),
        }

    def list_runs(self, limit: int = 30) -> dict[str, Any]:
        """List indexed immutable runs from SQLite artifact index."""
        store = RunArtifactStore(artifact_root=self.report_dir, db_path=self.config.db_path)
        entries = store.list_runs(limit=limit)
        return {
            "count": len(entries),
            "runs": [e.model_dump(mode="json") for e in entries],
        }

    def leaderboard(
        self,
        *,
        scope: str = "local",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return scoped local or official leaderboard entries."""
        eff_scope: Literal["local", "official"] = "official" if scope == "official" else "local"
        svc = LeaderboardService(artifact_root=self.report_dir, db_path=self.config.db_path)
        entries = svc.get_entries(scope=eff_scope, limit=limit)
        return {
            "scope": eff_scope,
            "count": len(entries),
            "entries": [e.model_dump(mode="json") for e in entries],
        }

    def shadow_status(self, horizon: int = 10) -> dict[str, Any]:
        """Inspect Forward Shadow Ledger candidates, decay ratios & hash chain."""
        from alphaevo.forward.shadow_ledger import ShadowLedger

        ledger = ShadowLedger(self.config.rsi.shadow_ledger_dir)
        chain_ok, chain_errors = ledger.verify_chain()
        cands = ledger.list_candidates()
        items = [
            ledger.build_decay_report(c.strategy_id, horizon_days=horizon).model_dump(mode="json")
            for c in cands
        ]
        return {
            "chain_verified": chain_ok,
            "chain_errors": chain_errors,
            "count": len(items),
            "candidates": items,
        }

    def shadow_simulate(
        self,
        *,
        strategy_id: str | None = None,
        strategy_yaml: str | None = None,
        days: int = 35,
        horizon: int = 5,
        live: bool | None = None,
    ) -> dict[str, Any]:
        """Run rolling chronological bitemporal simulation in an isolated ledger."""
        import pandas as pd  # type: ignore[import-untyped]

        from alphaevo.forward.shadow_ledger import ShadowLedger

        strategy = self._resolve_input_strategy(
            strategy_id=strategy_id, strategy_yaml=strategy_yaml
        )
        market_data, _src, _repro, _mut = self._load_market_data(live=live)
        all_dates = sorted(
            {
                d
                for df in market_data.values()
                if not df.empty and "date" in df.columns
                for d in pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().tolist()
                if isinstance(d, date)
            }
        )
        if not all_dates:
            raise ValueError("No market dates available for shadow simulation")
        start_d = all_dates[max(0, len(all_dates) - max(5, days))]

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ShadowLedger(tmpdir)
            rep = ledger.simulate_rolling(
                strategy,
                market_data,
                start_date=start_d,
                end_date=all_dates[-1],
                step_days=1,
                horizon_days=horizon,
                min_trading_days=5,
                min_mature_signals=5,
                min_effective_signals=3.0,
                is_replay=not bool(live),
            )
        return {
            "report": rep.model_dump(mode="json"),
            "markdown": ShadowLedger.render_simulation_markdown(rep),
        }

    def list_factors(self) -> dict[str, Any]:
        """List factors stored in FactorStore."""
        from alphaevo.alpha_factory.factor_store import FactorStore

        store = FactorStore(self.config.db_path)
        try:
            records = store.list_all()
            return {
                "count": len(records),
                "factors": [r.model_dump(mode="json") for r in records],
            }
        finally:
            store.close()

    def audit_factors(self, *, auto_retire: bool = False) -> dict[str, Any]:
        """Audit persisted dynamic factors on OOS IC/IR decay, PIT invariance, and redundancy."""
        from alphaevo.alpha_factory.governance import FactorGovernanceEngine

        market_data, _src, _repro, _mut = self._load_market_data(live=False)
        engine = FactorGovernanceEngine(db_path=self.config.db_path)
        report = engine.audit_factors(market_data, auto_retire=auto_retire)
        return {
            "report": report.model_dump(mode="json"),
            "markdown": FactorGovernanceEngine.render_markdown(report),
        }

    def allocate_portfolio(
        self,
        *,
        strategy_ids: list[str],
        live: bool | None = None,
    ) -> dict[str, Any]:
        """Compute strictly causal regime-adaptive portfolio risk weights across strategies."""
        from alphaevo.backtest.allocator import RegimeAdaptiveAllocator
        from alphaevo.cli.commands.rsi import _resolve_strategy

        if not strategy_ids:
            raise ValueError("Provide at least one strategy_id")
        strategies = [_resolve_strategy(sid, self.config) for sid in strategy_ids]
        market_data, _src, _repro, _mut = self._load_market_data(live=live)
        allocator = RegimeAdaptiveAllocator()
        report = allocator.allocate(
            strategies,
            market_data,
            backtest_config=self.config.backtest,
        )
        return {
            "report": report.model_dump(mode="json"),
            "markdown": RegimeAdaptiveAllocator.render_markdown(report),
        }

    def _resolve_input_strategy(
        self,
        *,
        strategy_id: str | None = None,
        strategy_yaml: str | None = None,
    ) -> Strategy:
        from alphaevo.cli.commands.rsi import _resolve_strategy
        from alphaevo.strategy.dsl.parser import StrategyParser

        if strategy_yaml and strategy_yaml.strip():
            return StrategyParser().parse_yaml(strategy_yaml)
        if strategy_id and strategy_id.strip():
            try:
                return _resolve_strategy(strategy_id.strip(), self.config)
            except ValueError as exc:
                raise FileNotFoundError(str(exc)) from exc
        raise ValueError("Provide either strategy_id or strategy_yaml")

    def _load_market_data(
        self,
        *,
        live: bool | None = None,
    ) -> tuple[
        dict[str, pd.DataFrame],
        str,
        Literal["replayable", "refetch_dependent", "non_reproducible"],
        Literal["immutable_bundle", "mutable_provider"],
    ]:
        from alphaevo.cli.commands.rsi import _load_market_data
        from alphaevo.cli.demo import load_showcase_snapshot

        use_live = self.live_data if live is None else live
        if use_live:
            try:
                data = _load_market_data(live=True)
                if data:
                    return (
                        data,
                        f"live:{self.config.data.adapter}",
                        "refetch_dependent",
                        "mutable_provider",
                    )
            except Exception:
                pass
        data, _manifest = load_showcase_snapshot()
        return data, "bundled_snapshot", "replayable", "immutable_bundle"


def create_web_server(
    service: AlphaEvoWebService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a localhost-first ThreadingHTTPServer serving the AlphaEvo Web Studio."""
    studio_html = render_web_studio_html().encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug("web: " + format, *args)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> dict[str, Any]:
            length_raw = self.headers.get("Content-Length", "0")
            try:
                length = int(length_raw)
            except ValueError:
                raise ValueError("Invalid Content-Length header") from None
            if length < 0 or length > _MAX_BODY_BYTES:
                raise ValueError("Request body too large or invalid")
            if length == 0:
                return {}
            body = self.rfile.read(length).decode("utf-8")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise ValueError("JSON request body must be an object")
            return parsed

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            try:
                if path == "/" or path == "/index.html":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(studio_html)))
                    self.end_headers()
                    self.wfile.write(studio_html)
                    return
                if path == "/api/health":
                    self._send_json(HTTPStatus.OK, service.health())
                    return
                if path == "/api/strategies":
                    self._send_json(HTTPStatus.OK, service.list_strategies())
                    return
                if path.startswith("/api/strategies/"):
                    sid = unquote(path[len("/api/strategies/") :])
                    self._send_json(HTTPStatus.OK, service.get_strategy_yaml(sid))
                    return
                if path == "/api/runs":
                    limit = int(query.get("limit", ["30"])[0])
                    self._send_json(HTTPStatus.OK, service.list_runs(limit=limit))
                    return
                if path == "/api/leaderboard":
                    scope = query.get("scope", ["local"])[0]
                    limit = int(query.get("limit", ["20"])[0])
                    self._send_json(HTTPStatus.OK, service.leaderboard(scope=scope, limit=limit))
                    return
                if path == "/api/shadow":
                    horizon = int(query.get("horizon", ["10"])[0])
                    self._send_json(HTTPStatus.OK, service.shadow_status(horizon=horizon))
                    return
                if path == "/api/factors":
                    self._send_json(HTTPStatus.OK, service.list_factors())
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown GET endpoint: {path}"})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def _send_sse(self, event_name: str, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False, default=str)
            frame = f"event: {event_name}\ndata: {data}\n\n".encode()
            self.wfile.write(frame)
            self.wfile.flush()

        def _stream_rsi(self, body: dict[str, Any]) -> None:
            event_q: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

            def _worker() -> None:
                try:

                    def _on_event(evt: dict[str, Any]) -> None:
                        etype = str(evt.get("type", "progress"))
                        event_q.put((etype, evt))

                    res = service.run_rsi(
                        strategy_id=body.get("strategy_id"),
                        strategy_yaml=body.get("strategy_yaml"),
                        rounds=int(body.get("rounds", 2)),
                        holdout_pct=float(body.get("holdout_pct", 0.0)),
                        portfolio_gate=bool(body.get("portfolio_gate", False)),
                        live=body.get("live"),
                        enable_l2=bool(body.get("enable_l2", True)),
                        proposer_model=body.get("proposer_model"),
                        prosecutor_model=body.get("prosecutor_model"),
                        api_key=body.get("api_key"),
                        on_event=_on_event,
                    )
                    event_q.put(("result", res))
                except Exception as exc:
                    event_q.put(("error", {"error": str(exc)}))
                finally:
                    event_q.put(None)

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()
            try:
                self._send_sse("start", {"status": "running"})
                while True:
                    item = event_q.get()
                    if item is None:
                        break
                    ev_name, ev_payload = item
                    self._send_sse(ev_name, ev_payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                worker.join(timeout=5.0)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = self._read_json_body()
                if path == "/api/strategies/draft":
                    self._send_json(
                        HTTPStatus.OK,
                        service.draft_strategy(
                            str(body.get("idea", "")),
                            market=str(body.get("market", "a_share")),
                            strategy_id=body.get("strategy_id"),
                            save=bool(body.get("save", False)),
                        ),
                    )
                    return
                if path == "/api/strategies/diagnose":
                    self._send_json(
                        HTTPStatus.OK,
                        service.diagnose_strategy(
                            strategy_id=body.get("strategy_id"),
                            strategy_yaml=body.get("strategy_yaml"),
                            live=body.get("live"),
                        ),
                    )
                    return
                if path == "/api/strategies/save":
                    self._send_json(
                        HTTPStatus.OK,
                        service.save_strategy(strategy_yaml=str(body.get("strategy_yaml", ""))),
                    )
                    return
                if path == "/api/runs":
                    self._send_json(
                        HTTPStatus.OK,
                        service.run_strategy(
                            strategy_id=body.get("strategy_id"),
                            strategy_yaml=body.get("strategy_yaml"),
                            live=body.get("live"),
                            protocol=str(body.get("protocol", "full_research_v1")),
                        ),
                    )
                    return
                if path == "/api/rsi/run":
                    self._send_json(
                        HTTPStatus.OK,
                        service.run_rsi(
                            strategy_id=body.get("strategy_id"),
                            strategy_yaml=body.get("strategy_yaml"),
                            rounds=int(body.get("rounds", 2)),
                            holdout_pct=float(body.get("holdout_pct", 0.0)),
                            portfolio_gate=bool(body.get("portfolio_gate", False)),
                            live=body.get("live"),
                            enable_l2=bool(body.get("enable_l2", True)),
                            proposer_model=body.get("proposer_model"),
                            prosecutor_model=body.get("prosecutor_model"),
                            api_key=body.get("api_key"),
                        ),
                    )
                    return
                if path == "/api/rsi/stream":
                    self._stream_rsi(body)
                    return
                if path == "/api/rsi/mechanism-ab":
                    raw_cutoffs = body.get("cutoffs")
                    parsed_cutoffs = (
                        [float(x) for x in raw_cutoffs]
                        if isinstance(raw_cutoffs, list) and raw_cutoffs
                        else None
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        service.run_mechanism_ab(
                            strategy_id=body.get("strategy_id"),
                            strategy_yaml=body.get("strategy_yaml"),
                            rounds=int(body.get("rounds", 2)),
                            cutoffs=parsed_cutoffs,
                            live=body.get("live"),
                        ),
                    )
                    return
                if path == "/api/rsi/benchmark":
                    raw_sids = body.get("strategy_ids")
                    sids_list = (
                        [str(s) for s in raw_sids]
                        if isinstance(raw_sids, list) and raw_sids
                        else None
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        service.run_benchmark(
                            strategy_ids=sids_list,
                            rounds=int(body.get("rounds", 2)),
                            holdout_pct=float(body.get("holdout_pct", 0.20)),
                            live=body.get("live"),
                            run_null_audit=bool(body.get("run_null_audit", True)),
                        ),
                    )
                    return
                if path == "/api/shadow/simulate":
                    self._send_json(
                        HTTPStatus.OK,
                        service.shadow_simulate(
                            strategy_id=body.get("strategy_id"),
                            strategy_yaml=body.get("strategy_yaml"),
                            days=int(body.get("days", 35)),
                            horizon=int(body.get("horizon", 5)),
                            live=body.get("live"),
                        ),
                    )
                    return
                if path == "/api/factors/audit":
                    self._send_json(
                        HTTPStatus.OK,
                        service.audit_factors(auto_retire=bool(body.get("auto_retire", False))),
                    )
                    return
                if path == "/api/portfolio/allocate":
                    sids = [str(s) for s in (body.get("strategy_ids") or [])]
                    self._send_json(
                        HTTPStatus.OK,
                        service.allocate_portfolio(strategy_ids=sids, live=body.get("live")),
                    )
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown POST endpoint: {path}"})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), _Handler)


__all__ = ["AlphaEvoWebService", "create_web_server"]
