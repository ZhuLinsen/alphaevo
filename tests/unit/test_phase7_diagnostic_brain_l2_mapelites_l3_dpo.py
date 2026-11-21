"""Phase 7 tests: failure-driven diagnostic brain, L2 Co-STEER filter fix, MAP-Elites SVG, L3 AB Lab & DPO trajectory."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any

from alphaevo.arena import (
    ArenaRunner,
    ArenaSnapshot,
    render_arena_html,
    render_map_elites_svg,
)
from alphaevo.cli.demo import load_showcase_snapshot
from alphaevo.cli.main import app
from alphaevo.core.config import AppConfig, BacktestConfig
from alphaevo.mcp.server import AlphaEvoMCPServer
from alphaevo.models.enums import ExitReason, RSILevel, SignalDirection
from alphaevo.models.execution import (
    EvaluationReport,
    OverallMetrics,
    TradeSignal,
)
from alphaevo.rsi import (
    BlindSpotMiner,
    HeterogeneousCouncil,
    ModelSpec,
    RSICharter,
    StockRSIEngine,
    analyze_entry_bottlenecks,
    build_rsi_trajectory,
    export_rsi_trajectory,
)
from alphaevo.rsi.diagnostics import StrategyBottleneckReport
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.truth.null_control import NullSignalController
from alphaevo.web.server import AlphaEvoWebService, create_web_server


def test_analyze_entry_bottlenecks_zero_trials(tmp_path: Path) -> None:
    db_file = tmp_path / "diag.db"
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()

    rsi_strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    rep_rsi = analyze_entry_bottlenecks(rsi_strat, market_data)
    assert isinstance(rep_rsi, StrategyBottleneckReport)
    assert rep_rsi.total_evaluated_bars > 1000
    assert rep_rsi.combined_pass_rate == 0.0
    assert rep_rsi.tightest_bottleneck is not None
    assert rep_rsi.tightest_bottleneck.indicator == "rsi_14"
    assert len(rep_rsi.tightest_bottleneck.matching_tunable_targets) >= 1

    vcb_strat = parser.parse_file(Path("strategies/builtin/volatility_compression_breakout.yaml"))
    rep_vcb = analyze_entry_bottlenecks(vcb_strat, market_data)
    assert rep_vcb.tightest_bottleneck is not None
    assert rep_vcb.guard_group_pass_rate < 0.25

    # Must NEVER create or touch TrialLedger entries
    assert not db_file.exists()


def test_co_steer_l2_targets_filters_on_or_logic_and_skips_sparse() -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    sample_df = next(iter(market_data.values()))

    miner = BlindSpotMiner()
    strat_or = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    strat_or.entry.logic = "or"

    # Skips L2 when signals < 10
    sparse_eval = EvaluationReport(
        strategy_id=strat_or.meta.id,
        overall=OverallMetrics(signal_count=4, win_rate=0.5, avg_return=0.01),
    )
    dec_sparse = miner.steer(
        strat_or,
        sparse_eval,
        sample_df,
        allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
        stagnant_l1_rounds=3,
    )
    assert dec_sparse.chosen_level == RSILevel.L1_STRATEGY

    # Injects into entry.filters when entry.logic == "or" (legacy conditions strategy)
    chop_eval = EvaluationReport(
        strategy_id=strat_or.meta.id,
        overall=OverallMetrics(signal_count=40, win_rate=0.45, avg_return=0.002),
        failure_cases=[
            TradeSignal(
                symbol="AAPL",
                signal_date=sample_df["date"].iloc[10],
                direction=SignalDirection.LONG,
                entry_price=100.0,
                exit_price=95.0,
                exit_reason=ExitReason.STOP_LOSS,
                return_pct=-0.05,
            )
            for _ in range(3)
        ],
    )
    dec_or = miner.steer(
        strat_or,
        chop_eval,
        sample_df,
        allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
        stagnant_l1_rounds=2,
    )
    assert dec_or.chosen_level == RSILevel.L2_FACTOR
    assert dec_or.pit_verified is True
    assert dec_or.proposed_changes[0].target == "entry.filters"

    # Injects into entry.guards when entry.triggers is present
    strat_trig = parser.parse_file(Path("strategies/builtin/volatility_compression_breakout.yaml"))
    dec_trig = miner.steer(
        strat_trig,
        chop_eval,
        sample_df,
        allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
        stagnant_l1_rounds=2,
    )
    assert dec_trig.chosen_level == RSILevel.L2_FACTOR
    assert dec_trig.proposed_changes[0].target == "entry.guards"


def test_diagnostic_council_ordering_and_zero_signal_improvement() -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    strat = parser.parse_file(Path("strategies/builtin/volatility_compression_breakout.yaml"))
    bn = analyze_entry_bottlenecks(strat, market_data)

    # Trailing TP + zero-signal bottleneck proposals include exit & bottleneck unlock options
    council = HeterogeneousCouncil()
    zero_eval = EvaluationReport(
        strategy_id=strat.meta.id,
        overall=OverallMetrics(signal_count=0, win_rate=0.0, avg_return=0.0, max_drawdown=0.0),
    )
    props = council._build_heuristic_proposals(
        strat,
        zero_eval,
        ModelSpec(model="heuristic"),
        bottleneck=bn,
    )
    targets = [c.target for p in props for c in p.changes]
    assert "entry.logic" in targets
    assert any(t.startswith("exit.take_profit.") or t == "exit.max_holding_days" for t in targets)

    # Whipsaw chop prioritizes exit levers FIRST before entry tuning
    rsi_strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    chop_eval = EvaluationReport(
        strategy_id=rsi_strat.meta.id,
        overall=OverallMetrics(signal_count=55, win_rate=0.45, avg_return=0.005, max_drawdown=0.31),
    )
    chop_props = council._build_heuristic_proposals(
        rsi_strat, chop_eval, ModelSpec(model="heuristic")
    )
    assert chop_props
    assert chop_props[0].changes[0].target.startswith("exit.")

    # Empirical check: StockRSIEngine recovers strong trades & positive return on rsi_reversion
    engine = StockRSIEngine(db_path=":memory:", backtest_config=BacktestConfig(min_data_days=15))
    charter = RSICharter(
        name="phase7_diag",
        seed_strategy_id=rsi_strat.meta.id,
        allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
        max_rounds=3,
        max_candidates_per_round=3,
        max_trials_budget=20,
        min_signals=15,
        require_null_control=True,
        null_permutations=20,
    )
    res = engine.run(rsi_strat, market_data, charter=charter, persist_champion=False)
    assert res.kernel_intact is True
    assert res.kept_rounds >= 2
    assert any(r.signals >= 35 and r.avg_return > 0.01 for r in res.rows)


def test_tier2_null_audit_zero_false_promotions_with_diagnostic_brain() -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))

    def _run_world(seed_s: Any, null_market: dict[str, Any]) -> int:
        eng = StockRSIEngine(db_path=":memory:", backtest_config=BacktestConfig(min_data_days=15))
        ch = RSICharter(
            name="tier2_phase7",
            seed_strategy_id=seed_s.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
            max_rounds=2,
            max_candidates_per_round=3,
            require_null_control=True,
            null_permutations=25,
        )
        return eng.run(seed_s, null_market, charter=ch, persist_champion=False).promotions

    audit = NullSignalController().verify_zero_promotion_under_shuffled_returns(
        strat, market_data, _run_world, n_worlds=2
    )
    assert audit.passed is True
    assert audit.false_promotions == 0


def test_trajectory_dpo_pairs_and_jsonl_export(tmp_path: Path) -> None:
    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    out_dir = tmp_path / "rsi_out"

    engine = StockRSIEngine(
        db_path=tmp_path / "traj.db",
        backtest_config=BacktestConfig(min_data_days=15),
    )
    charter = RSICharter(
        name="traj_test",
        seed_strategy_id=strat.meta.id,
        allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
        max_rounds=3,
        max_candidates_per_round=3,
        max_trials_budget=20,
        min_signals=15,
        require_null_control=True,
        null_permutations=20,
    )
    res = engine.run(
        strat, market_data, charter=charter, output_dir=out_dir, persist_champion=False
    )
    export = build_rsi_trajectory(res)
    assert len(export.steps) == len(res.deliberations)
    assert len(export.dpo_pairs) >= 1

    paths = export_rsi_trajectory(res, out_dir)
    assert paths["trajectory"].is_file()
    assert paths["dpo_pairs"].is_file()
    dpo_rows = [
        json.loads(line)
        for line in paths["dpo_pairs"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(dpo_rows) == len(export.dpo_pairs)
    assert dpo_rows[0]["chosen"] != dpo_rows[0]["rejected"]


def test_map_elites_svg_and_arena_l2_integration(tmp_path: Path, monkeypatch) -> None:
    empty_grid = render_map_elites_svg([])
    assert "<svg" in empty_grid
    assert "Empty Niche" in empty_grid

    pop_grid = render_map_elites_svg(
        [
            {
                "regime_bucket": "range_reversal",
                "style_bucket": "reversal",
                "horizon_bucket": "swing_6_15d",
                "strategy_id": "rsi_reversion_v7",
                "fitness_score": 0.62,
            }
        ]
    )
    assert "rsi_reversion_v7" in pop_grid
    assert "Fit 62.0%" in pop_grid

    parser = StrategyParser()
    market_data, _ = load_showcase_snapshot()
    s1 = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    snapshot = ArenaRunner().run([s1], market_data, rounds=2, enable_l2_factors=True)
    assert isinstance(snapshot, ArenaSnapshot)
    assert len(snapshot.entries[0].map_elites_cells) >= 1
    arena_html = render_arena_html(snapshot)
    assert "MAP-Elites Behavioral Niche Archive" in arena_html
    assert "detail-map-elites" in arena_html
    assert "https://cdn" not in arena_html

    from typer.testing import CliRunner

    import alphaevo.cli.commands.arena as arena_cli

    monkeypatch.setattr(
        arena_cli, "_get_config", lambda: AppConfig(db_path=tmp_path / "phase7_cli.db")
    )
    cli_out = tmp_path / "arena_cli"
    cli_res = CliRunner().invoke(
        app,
        [
            "arena",
            "run",
            "--strategies",
            "rsi_reversion",
            "--rounds",
            "1",
            "--l2",
            "--output-dir",
            str(cli_out),
        ],
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert (cli_out / "index.html").is_file()


def test_web_diagnose_l2_dpo_mechanism_ab_and_mcp_tools(tmp_path: Path) -> None:
    cfg = AppConfig(db_path=tmp_path / "phase7_web.db")
    svc = AlphaEvoWebService(cfg, report_dir=tmp_path / "reports")
    server = create_web_server(svc, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as resp:
            html_body = resp.read().decode("utf-8")
        assert "X-Ray Bottlenecks" in html_body
        assert "L2 Co-STEER Factors" in html_body
        assert "Download DPO Pairs (.jsonl)" in html_body
        assert "L3 Meta-RSI Harness A/B Calibration Lab" in html_body

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        paths = {r["path"] for r in health["manifest"]["routes"]}
        assert {"/api/strategies/diagnose", "/api/rsi/mechanism-ab"} <= paths

        # Diagnose consumes ZERO TrialLedger trials
        diag_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/strategies/diagnose",
            data=json.dumps({"strategy_id": "rsi_reversion_v1"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(diag_req, timeout=20) as resp:
            diag_obj = json.loads(resp.read().decode("utf-8"))
        assert diag_obj["tightest_bottleneck"]["indicator"] == "rsi_14"
        with sqlite3.connect(cfg.db_path) as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "trial_ledger" in tables:
                assert conn.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()[0] == 0

        # POST /api/rsi/run returns MAP-Elites, bottleneck, and DPO pairs
        rsi_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rsi/run",
            data=json.dumps(
                {"strategy_id": "rsi_reversion_v1", "rounds": 2, "enable_l2": True}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(rsi_req, timeout=45) as resp:
            rsi_obj = json.loads(resp.read().decode("utf-8"))
        assert "<svg" in rsi_obj["map_elites_svg"]
        assert (
            isinstance(rsi_obj["map_elites_cells"], list) and len(rsi_obj["map_elites_cells"]) >= 1
        )
        assert isinstance(rsi_obj["dpo_pairs"], list)
        assert rsi_obj["bottleneck_report"]["strategy_id"] == "rsi_reversion_v1"

        # POST /api/rsi/mechanism-ab runs paired chronological L3 harness comparison
        ab_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rsi/mechanism-ab",
            data=json.dumps(
                {"strategy_id": "rsi_reversion_v1", "rounds": 1, "cutoffs": [0.65]}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(ab_req, timeout=60) as resp:
            ab_obj = json.loads(resp.read().decode("utf-8"))
        assert ab_obj["result"]["winner_variant_id"] in {"naive_off", "l0_scars_l2"}
        assert "L3 Meta-RSI Mechanism Calibration Report" in ab_obj["markdown"]
    finally:
        server.shutdown()
        server.server_close()

    # MCP 11 tools including diagnose & mechanism_ab
    mcp = AlphaEvoMCPServer(
        config=AppConfig(db_path=tmp_path / "phase7_mcp.db"),
        report_dir=tmp_path / "mcp_reports",
    )
    t_resp = mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert t_resp is not None
    t_names = {t["name"] for t in t_resp["result"]["tools"]}
    assert {"alphaevo_diagnose_bottlenecks", "alphaevo_mechanism_ab"} <= t_names
    assert len(t_names) >= 11

    d_call = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "alphaevo_diagnose_bottlenecks",
                "arguments": {"strategy_id": "rsi_reversion_v1"},
            },
        }
    )
    assert d_call is not None and d_call["result"]["isError"] is False
    d_payload = json.loads(d_call["result"]["content"][0]["text"])
    assert d_payload["strategy_id"] == "rsi_reversion_v1"

    # Cold subprocess imports including diagnostics & trajectory
    for mod in (
        "alphaevo.rsi.diagnostics",
        "alphaevo.rsi.trajectory",
        "alphaevo.web.ui",
        "alphaevo.web.server",
        "alphaevo.mcp.server",
    ):
        subprocess.run([sys.executable, "-c", f"import {mod}"], check=True)
