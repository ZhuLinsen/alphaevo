"""Phase 8 tests: semantic crossover, bottleneck-aware hybrid L1/L2, ablation benchmark & studio save."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from alphaevo.alpha_factory.factor_store import FactorStore
from alphaevo.alpha_factory.synthesizer import FactorHypothesis
from alphaevo.arena import ArenaRunner, render_evolution_tree_svg
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.cli.demo import load_showcase_snapshot
from alphaevo.cli.main import app
from alphaevo.core.config import AppConfig, BacktestConfig
from alphaevo.mcp.server import AlphaEvoMCPServer
from alphaevo.models.enums import ChangeType, ExitReason, RSILevel, SignalDirection
from alphaevo.models.execution import EvaluationReport, OverallMetrics, TradeSignal
from alphaevo.rsi import (
    BlindSpotCluster,
    BlindSpotMiner,
    ExecutableScar,
    HeterogeneousCouncil,
    HeterogeneousCouncilConfig,
    ModelSpec,
    RSICharter,
    SemanticCrossoverBreeder,
    StockRSIEngine,
    analyze_entry_bottlenecks,
    render_benchmark_markdown,
    run_evolution_benchmark,
)
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.truth.null_control import NullSignalController
from alphaevo.web.server import AlphaEvoWebService, create_web_server


def test_semantic_crossover_breeder_operators_caps_and_critic_safety() -> None:
    parser = StrategyParser()
    rsi_strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    vcb_strat = parser.parse_file(Path("strategies/builtin/volatility_compression_breakout.yaml"))

    breeder = SemanticCrossoverBreeder(complexity_limit=8)
    assert breeder.breed(rsi_strat, rsi_strat) == []

    # Primary=rsi_reversion (pct/rr), Secondary=volatility_compression_breakout (trailing/guards)
    children = breeder.breed(rsi_strat, vcb_strat, max_children=3)
    ops = {c.operator for c in children}
    assert "exit_graft" in ops
    assert "filter_graft" in ops
    assert "trigger_blend" in ops

    # Ensure complexity limit strictly caps offspring
    capped = SemanticCrossoverBreeder(complexity_limit=3).breed(
        vcb_strat, rsi_strat, max_children=3
    )
    for cand in capped:
        total_conds = (
            len(cand.child_strategy.entry.triggers)
            + len(cand.child_strategy.entry.conditions)
            + len(cand.child_strategy.entry.guards)
            + len(cand.child_strategy.entry.filters)
            + len(cand.child_strategy.exit.triggers)
        )
        assert total_conds <= 3

    # Ensure summary_changes pass SelfCritic without MutationError (regression guard)
    council = HeterogeneousCouncil()
    eval_rep = EvaluationReport(
        strategy_id=rsi_strat.meta.id,
        overall=OverallMetrics(signal_count=25, win_rate=0.52, avg_return=0.015),
    )
    for cand in children:
        if cand.operator == "exit_graft":
            aud = council.audit_changes(rsi_strat, eval_rep, cand.summary_changes)
            assert aud.approved is True, aud.objections


def test_hybrid_diagnostic_council_receives_bottleneck_and_falls_back_on_bad_json(
    monkeypatch: Any,
) -> None:
    from alphaevo.core.llm import LLMClient

    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    bn = analyze_entry_bottlenecks(strat, market_data)
    eval_rep = EvaluationReport(
        strategy_id=strat.meta.id,
        overall=OverallMetrics(signal_count=0, win_rate=0.0, avg_return=0.0),
    )
    bs = BlindSpotCluster(
        cluster_id="zero_signal_lockout",
        severity="high",
        evidence="signal_count=0",
        suggested_action="Relax entry.logic to or",
    )
    scar = ExecutableScar(
        scar_id="scar_test_1",
        strategy_family=strat.meta.family_id,
        failure_mechanism="Blocked zero-signal RSI tightening",
        forbidden_change_type=ChangeType.TIGHTEN_FILTER,
        target_pattern="rsi_14",
    )

    cfg = HeterogeneousCouncilConfig(
        proposer=ModelSpec(model="openai/gpt-6-astra", backend="litellm"),
        prosecutor=ModelSpec(model="anthropic/claude-sonnet-5.5", backend="litellm"),
    )
    council = HeterogeneousCouncil(config=cfg, scars=[scar], api_key="secret-test-key-never-stored")

    seen_prompts: list[str] = []

    def _fake_chat_json(self: Any, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        del self, kwargs
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        seen_prompts.append(prompt_text)
        if "risk prosecutor" in prompt_text.lower():
            return {"veto": False, "reason": "Clear"}
        return {
            "rationale": "Unlock AND stack guided by X-Ray",
            "changes": [
                {
                    "change_type": "change_logic",
                    "target": "entry.logic",
                    "from_value": "and",
                    "to_value": "or",
                    "reason": "Fix AND lockout",
                }
            ],
        }

    monkeypatch.setattr(LLMClient, "chat_json", _fake_chat_json)
    delibs = council.generate_candidates(
        strat,
        eval_rep,
        mission_notes="Respect zero false promotions",
        max_candidates=2,
        blind_spot=bs,
        bottleneck=bn,
    )
    assert delibs
    assert any("Bottleneck X-Ray:" in p and "Active Pinned Scars" in p for p in seen_prompts)
    assert any("Respect zero false promotions" in p for p in seen_prompts)
    assert delibs[0].proposal.rationale == "Unlock AND stack guided by X-Ray"
    # Secret must never appear inside Deliberation output or prompts
    dumped = json.dumps([d.model_dump(mode="json") for d in delibs])
    assert "secret-test-key-never-stored" not in dumped

    # Malformed/empty LLM JSON falls back cleanly to deterministic heuristic proposals
    def _broken_chat_json(
        self: Any, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        del self, messages, kwargs
        raise ValueError("malformed json")

    monkeypatch.setattr(LLMClient, "chat_json", _broken_chat_json)
    fallback_delibs = council.generate_candidates(
        strat,
        eval_rep,
        max_candidates=2,
        blind_spot=bs,
        bottleneck=bn,
    )
    assert len(fallback_delibs) >= 1


def test_co_steer_hybrid_llm_factor_discovery_rejects_lookahead_and_accepts_pit_safe(
    tmp_path: Path,
) -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    sample_df = next(iter(market_data.values()))
    strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    strat.entry.logic = "or"

    chop_eval = EvaluationReport(
        strategy_id=strat.meta.id,
        overall=OverallMetrics(signal_count=45, win_rate=0.44, avg_return=0.002),
        failure_cases=[
            TradeSignal(
                symbol="AAPL",
                signal_date=sample_df["date"].iloc[15],
                direction=SignalDirection.LONG,
                entry_price=100.0,
                exit_price=95.0,
                exit_reason=ExitReason.STOP_LOSS,
                return_pct=-0.05,
            )
            for _ in range(3)
        ],
    )

    # 1. Mock synthesizer returning sneaky future leakage -> MUST be rejected by PIT probe!
    class _SneakyLookaheadSynth:
        def generate(self, **kwargs: Any) -> list[FactorHypothesis]:
            del kwargs
            return [
                FactorHypothesis(
                    name="sneaky_future_close_leak",
                    description="Uses future bar close[-1] dynamically",
                    rationale="cheat",
                    code=(
                        "def compute(df, idx):\n"
                        "    return float(df['close'].iloc[-1]) - float(df['close'].iloc[idx])\n"
                    ),
                    expected_direction="positive",
                )
            ]

    db_path = tmp_path / "l2_factors.db"
    store = FactorStore(db_path)
    try:
        miner_leak = BlindSpotMiner(factor_store=store, synthesizer=_SneakyLookaheadSynth())
        dec_fallback = miner_leak.steer(
            strat,
            chop_eval,
            sample_df,
            allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
            stagnant_l1_rounds=2,
        )
        assert "sneaky_future_close_leak" in miner_leak._failed_llm_factors
        assert dec_fallback.chosen_level == RSILevel.L2_FACTOR
        assert dec_fallback.synthesized_via == "offline_template"
        assert dec_fallback.synthesized_factor_name != "sneaky_future_close_leak"
        assert dec_fallback.proposed_changes[0].target == "entry.filters"

        # 2. Mock synthesizer returning a valid PIT-safe factor -> passes AST + PIT + causal calibration
        factor_name = "test_range_body_conviction_5d"

        class _SafeSynth:
            def generate(self, **kwargs: Any) -> list[FactorHypothesis]:
                del kwargs
                return [
                    FactorHypothesis(
                        name=factor_name,
                        description="5-day causal close-vs-low conviction",
                        rationale="Filter choppy knife catches",
                        code=(
                            "import numpy as np\n"
                            "def compute(df, idx):\n"
                            "    if idx < 5:\n"
                            "        return 0.0\n"
                            "    sub = df.iloc[idx - 4 : idx + 1]\n"
                            "    span = np.maximum(1e-6, sub['high'].astype(float) - sub['low'].astype(float))\n"
                            "    return float(((sub['close'].astype(float) - sub['low'].astype(float)) / span).mean())\n"
                        ),
                        expected_direction="positive",
                    )
                ]

        miner_safe = BlindSpotMiner(factor_store=store, synthesizer=_SafeSynth())
        dec_llm = miner_safe.steer(
            strat,
            chop_eval,
            sample_df,
            allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
            stagnant_l1_rounds=2,
        )
        assert dec_llm.chosen_level == RSILevel.L2_FACTOR
        assert dec_llm.synthesized_via == "llm"
        assert dec_llm.synthesized_factor_name == factor_name
        assert dec_llm.pit_verified is True
        assert dec_llm.persisted_to_store is True
        assert dec_llm.proposed_changes[0].target == "entry.filters"
        assert store.get(factor_name) is not None
    finally:
        IndicatorRegistry.unregister_dynamic("test_range_body_conviction_5d")
        store.close()


def test_rsi_engine_crossover_and_svg_edge() -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))

    engine = StockRSIEngine(
        db_path=":memory:",
        backtest_config=BacktestConfig(min_data_days=15),
        enable_crossover=True,
    )
    charter = RSICharter(
        name="xover_test",
        seed_strategy_id=strat.meta.id,
        allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
        max_rounds=3,
        max_candidates_per_round=2,
        max_trials_budget=30,
        min_signals=15,
        null_permutations=20,
    )
    res = engine.run(strat, market_data, charter=charter, persist_champion=False)
    assert res.kernel_intact is True
    assert res.crossovers_tried >= 0

    svg = render_evolution_tree_svg(
        [
            {"round_num": 0, "strategy_id": "s_v1", "status": "baseline", "score": 0.1},
            {
                "round_num": 1,
                "strategy_id": "s_v2",
                "parent_id": "s_v1",
                "status": "stepping_stone",
                "score": 0.25,
            },
            {
                "round_num": 2,
                "strategy_id": "s_v3",
                "parent_id": "s_v2",
                "secondary_parent_id": "s_v1",
                "status": "keep",
                "score": 0.42,
            },
        ]
    )
    assert "#a855f7" in svg
    assert "🧬" in svg


def test_tier2_null_audit_zero_false_promotions_with_l2_and_crossover() -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))

    def _run_world(seed_strat: Any, null_market: dict[str, Any]) -> int:
        engine = StockRSIEngine(
            db_path=":memory:",
            backtest_config=BacktestConfig(min_data_days=15),
            enable_crossover=True,
        )
        charter = RSICharter(
            name="tier2_phase8_null_audit",
            seed_strategy_id=seed_strat.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
            max_rounds=2,
            max_candidates_per_round=2,
            min_signals=15,
            require_null_control=True,
            null_permutations=25,
        )
        return engine.run(
            seed_strat, null_market, charter=charter, persist_champion=False
        ).promotions

    audit = NullSignalController().verify_zero_promotion_under_shuffled_returns(
        strat,
        market_data,
        _run_world,
        n_worlds=2,
    )
    assert audit.passed is True
    assert audit.false_promotions == 0


def test_ablation_benchmark_and_cli_surface(tmp_path: Path) -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    strategies = [
        parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml")),
        parser.parse_file(Path("strategies/builtin/volatility_compression_breakout.yaml")),
    ]
    report = run_evolution_benchmark(
        strategies,
        market_data,
        rounds=2,
        locked_holdout_pct=0.20,
        run_null_audit=True,
        null_worlds=1,
    )
    assert len(report.rows) == 2
    assert report.kernel_intact is True
    assert report.null_audit_passed is True
    assert report.null_false_promotions == 0
    md = render_benchmark_markdown(report)
    assert "Ablation Benchmark" in md
    assert "rsi_reversion_v1" in md

    runner = CliRunner()
    assert runner.invoke(app, ["rsi", "benchmark", "--help"]).exit_code == 0
    r_arena = runner.invoke(app, ["arena", "run", "--help"], env={"NO_COLOR": "1", "TERM": "dumb"})
    assert r_arena.exit_code == 0
    clean_arena_out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", r_arena.output)
    assert "--crossover" in clean_arena_out
    assert "--proposer-model" in clean_arena_out


def test_web_save_champion_benchmark_and_mcp_tool(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.db_path = tmp_path / "studio_phase8.db"
    cfg.rsi.shadow_ledger_dir = str(tmp_path / "shadow")
    svc = AlphaEvoWebService(cfg, report_dir=tmp_path / "reports")
    srv = create_web_server(svc, host="127.0.0.1", port=0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/", timeout=10) as resp:
            html_body = resp.read().decode("utf-8")
        assert "btn-save-champion" in html_body
        assert "btn-benchmark" in html_body
        assert "benchmark-box" in html_body

        with urllib.request.urlopen(f"{base}/api/health", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        paths = {r["path"] for r in health["manifest"]["routes"]}
        assert "/api/strategies/save" in paths
        assert "/api/rsi/benchmark" in paths

        parser = StrategyParser()
        strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
        strat.meta.id = "custom_champ_saved_v9"
        strat.meta.version = 9
        from alphaevo.strategy.dsl.serializer import StrategySerializer

        save_req = urllib.request.Request(
            f"{base}/api/strategies/save",
            data=json.dumps({"strategy_yaml": StrategySerializer().to_yaml(strat)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(save_req, timeout=10) as resp:
            saved = json.loads(resp.read().decode("utf-8"))
        assert saved["saved"] is True
        assert saved["strategy_id"] == "custom_champ_saved_v9"

        with urllib.request.urlopen(f"{base}/api/strategies", timeout=10) as resp:
            listing = json.loads(resp.read().decode("utf-8"))
        assert any(s["strategy_id"] == "custom_champ_saved_v9" for s in listing["strategies"])

        bench_req = urllib.request.Request(
            f"{base}/api/rsi/benchmark",
            data=json.dumps(
                {
                    "strategy_ids": ["rsi_reversion_v1"],
                    "rounds": 1,
                    "holdout_pct": 0.20,
                    "run_null_audit": False,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(bench_req, timeout=30) as resp:
            bench_out = json.loads(resp.read().decode("utf-8"))
        assert len(bench_out["result"]["rows"]) == 1
        assert "Ablation Benchmark" in bench_out["markdown"]
    finally:
        srv.shutdown()
        srv.server_close()

    mcp = AlphaEvoMCPServer(cfg, report_dir=tmp_path / "mcp_reports")
    listed = mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert listed is not None
    tool_names = [tool["name"] for tool in listed["result"]["tools"]]
    assert len(tool_names) == 12
    assert "alphaevo_benchmark_strategies" in tool_names

    called = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "alphaevo_benchmark_strategies",
                "arguments": {
                    "strategy_ids": ["rsi_reversion_v1"],
                    "rounds": 1,
                    "holdout_pct": 0.20,
                    "run_null_audit": False,
                },
            },
        }
    )
    assert called is not None
    assert called["result"]["isError"] is False


def test_arena_crossover_digest_and_cold_subprocess_import() -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))

    snap = ArenaRunner().run(
        [strat],
        market_data,
        rounds=2,
        enable_l2_factors=True,
        enable_crossover=True,
    )
    digest = ArenaRunner.render_digest_markdown(snap)
    assert "Crossovers Kept:" in digest

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from alphaevo.rsi import SemanticCrossoverBreeder, run_evolution_benchmark; print('COLD OK')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "COLD OK" in proc.stdout
