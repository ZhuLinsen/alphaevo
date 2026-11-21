"""Phase 6 tests: Evolution Observatory, Council Debate capture, Trade K-Line Replay, SSE stream & Pages/Docker."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from alphaevo.arena import (
    ArenaRunner,
    ArenaSnapshot,
    ArenaStrategyEntry,
    TradeReplayView,
    render_arena_html,
    render_evolution_tree_svg,
    render_run_html,
    render_trade_kline_svg,
)
from alphaevo.cli.demo import load_showcase_snapshot
from alphaevo.cli.main import app
from alphaevo.core.config import AppConfig, BacktestConfig
from alphaevo.models.enums import ExitReason, RSILevel, SignalDirection
from alphaevo.models.execution import (
    AntiFitMetrics,
    EvaluationReport,
    OverallMetrics,
    TradeSignal,
)
from alphaevo.rsi import (
    DeliberationRecord,
    HeterogeneousCouncil,
    HeterogeneousCouncilConfig,
    ModelSpec,
    RSICharter,
    StockRSIEngine,
)
from alphaevo.rsi.engine import _diagnose_gate_failures
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.web.server import AlphaEvoWebService, create_web_server


def test_rsi_engine_captures_deliberations_vetoes_and_events() -> None:
    parser = StrategyParser()
    seed_strat = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    market_data, _ = load_showcase_snapshot()

    events: list[dict[str, Any]] = []

    def _recording_cb(evt: dict[str, Any]) -> None:
        events.append(evt)

    engine = StockRSIEngine(
        db_path=":memory:",
        backtest_config=BacktestConfig(min_data_days=15),
    )
    charter = RSICharter(
        name="phase6_test",
        seed_strategy_id=seed_strat.meta.id,
        allowed_levels=[RSILevel.L1_STRATEGY],
        max_rounds=2,
        max_candidates_per_round=2,
        max_trials_budget=16,
        min_signals=15,
        require_null_control=True,
        null_permutations=20,
    )
    res = engine.run(
        seed_strat,
        market_data,
        charter=charter,
        persist_champion=False,
        on_event=_recording_cb,
    )
    assert res.kernel_intact is True
    assert len(res.deliberations) >= 1
    assert all(isinstance(d, DeliberationRecord) for d in res.deliberations)
    assert sum(res.veto_breakdown.values()) >= 0
    assert any(r.parent_id is not None for r in res.rows if r.round_num > 0)

    ev_types = [e["type"] for e in events]
    assert ev_types[0] == "baseline"
    assert ev_types[-1] == "complete"
    assert "round_complete" in ev_types
    assert "council_deliberation" in ev_types

    # Exception resilience: failing callback must not abort RSI evolution
    def _boom_cb(_evt: dict[str, Any]) -> None:
        raise RuntimeError("callback failure ignored")

    res_safe = engine.run(
        seed_strat,
        market_data,
        charter=charter,
        persist_champion=False,
        on_event=_boom_cb,
    )
    assert res_safe.kernel_intact is True
    assert len(res_safe.rows) >= 2


def test_diagnose_gate_failures_and_council_privacy() -> None:
    cfg = RSICharter(
        seed_strategy_id="s1",
        min_signals=20,
        min_dsr_prob=0.5,
        max_drawdown=0.25,
        require_null_control=True,
        require_portfolio_gate=True,
    )
    ev = EvaluationReport(
        strategy_id="cand_v2",
        overall=OverallMetrics(
            win_rate=0.4,
            avg_return=-0.01,
            max_drawdown=0.32,
            signal_count=8,
        ),
        anti_overfit=AntiFitMetrics(train_win_rate=0.8, val_win_rate=0.5, train_val_gap=0.3),
        null_control_passed=False,
    )
    outcome: Any = SimpleNamespace(portfolio_verdict=SimpleNamespace(passed=False))
    reasons = _diagnose_gate_failures(
        0.2,
        0.5,
        ev,
        outcome,
        cfg,
        dsr_prob=0.1,
        rerun_null_failed=True,
    )
    assert set(reasons) == {
        "no_fitness_gain",
        "negative_expectancy",
        "max_drawdown",
        "overfit_gate",
        "min_signals",
        "dsr_gate",
        "null_control",
        "portfolio_gate",
        "strategy_rerun_null",
    }

    secret_val = "sk-transient-never-leaked-999"
    council = HeterogeneousCouncil(api_key=secret_val)
    assert secret_val not in json.dumps(council.config.model_dump(mode="json"))

    # Same non-deterministic family must raise ValueError
    try:
        HeterogeneousCouncilConfig(
            proposer=ModelSpec(model="openai/gpt-5", backend="litellm"),
            prosecutor=ModelSpec(model="openai/gpt-4.1", backend="litellm"),
        )
        raise AssertionError("Expected ValueError for same-family council")
    except ValueError as exc:
        assert "proposer_family != prosecutor_family" in str(exc)


def test_trade_replays_kline_tree_and_observatory_html() -> None:
    dates = pd.bdate_range("2026-01-02", periods=30).date
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i * 0.4 for i in range(30)],
            "high": [101.2 + i * 0.4 for i in range(30)],
            "low": [99.1 + i * 0.4 for i in range(30)],
            "close": [100.5 + i * 0.4 for i in range(30)],
        }
    )
    market_data = {"AAPL": df}
    signals = [
        TradeSignal(
            symbol="AAPL",
            signal_date=dates[10],
            direction=SignalDirection.LONG,
            entry_price=104.0,
            exit_price=108.5,
            exit_date=dates[16],
            exit_reason=ExitReason.TAKE_PROFIT,
            return_pct=0.0432,
            holding_days=6,
        ),
        TradeSignal(
            symbol="AAPL",
            signal_date=dates[18],
            direction=SignalDirection.LONG,
            entry_price=107.2,
            exit_price=103.5,
            exit_date=dates[22],
            exit_reason=ExitReason.STOP_LOSS,
            return_pct=-0.0345,
            holding_days=4,
        ),
    ]
    replays = ArenaRunner.build_trade_replays(signals, market_data, max_trades=12)
    assert len(replays) == 2
    assert len(replays[0].bars) >= 5

    kline_svg = render_trade_kline_svg(replays[0])
    assert "<svg" in kline_svg
    assert "AAPL" in kline_svg
    assert "BUY" in kline_svg
    assert "EXIT" in kline_svg

    empty_kline = render_trade_kline_svg(
        TradeReplayView(symbol="MSFT", signal_date="2026-01-02", entry_price=200.0, bars=[])
    )
    assert "insufficient OHLC bars" in empty_kline

    tree_svg = render_evolution_tree_svg(
        [
            {"round_num": 0, "strategy_id": "base_v1", "status": "baseline", "score": 0.1},
            {
                "round_num": 1,
                "strategy_id": "base_v2",
                "parent_id": "base_v1",
                "status": "keep",
                "score": 0.55,
                "dsr_prob": 0.8,
            },
            {
                "round_num": 2,
                "strategy_id": "base_v3",
                "parent_id": "base_v2",
                "status": "revert",
                "score": 0.25,
                "dsr_prob": 0.1,
            },
        ]
    )
    assert "<svg" in tree_svg
    assert "KEEP" in tree_svg
    assert "REVERT" in tree_svg

    empty_tree = render_evolution_tree_svg([])
    assert "No evolution rounds recorded yet" in empty_tree

    snap = ArenaSnapshot(
        market_label="Synthetic Test",
        data_source="unit_test",
        date_range=(dates[0].isoformat(), dates[-1].isoformat()),
        symbols=["AAPL"],
        total_strategies=1,
        total_promotions=1,
        total_vetoes=2,
        total_deliberations=2,
        total_prosecutor_vetoes=1,
        veto_breakdown={"null_control": 1, "prosecutor_veto": 1},
        entries=[
            ArenaStrategyEntry(
                strategy_id="base_v1",
                champion_id="base_v2",
                name="Test",
                category="trend",
                market="us",
                baseline_score=0.1,
                champion_score=0.55,
                trade_replays=replays,
                veto_breakdown={"null_control": 1},
                deliberations=[
                    {
                        "round_num": 1,
                        "parent_strategy_id": "base_v1",
                        "candidate_strategy_id": "base_v2",
                        "proposer_family": "deterministic",
                        "rationale": "Unlock entry signal flow",
                        "proposed_changes": ["entry.logic: and -> or"],
                        "prosecutor_family": "deterministic",
                        "prosecutor_approved": True,
                        "prosecutor_objections": [],
                        "l0_outcome": "keep",
                        "veto_reasons": [],
                        "signals": 25,
                        "dsr_probability": 0.85,
                    }
                ],
            )
        ],
    )
    arena_html = render_arena_html(snap)
    assert "Overfit Graveyard" in arena_html
    assert "Trade K-Line Replay Inspector" in arena_html
    assert "Adversarial Council Debate" in arena_html
    assert "https://cdn" not in arena_html

    digest_md = ArenaRunner.render_digest_markdown(snap)
    assert "Overfit Graveyard" in digest_md
    assert "null_control" in digest_md

    run_html = render_run_html(
        run_id="r1",
        manifest={"effective_protocol": "full_research_v1", "strategy": {"id": "base_v1"}},
        evaluation={"confidence_score": 0.55, "overall": {"signal_count": 2}},
        strategy_yaml="meta:\n  id: base_v1\n",
        trades=[
            {
                "symbol": "AAPL",
                "signal_date": dates[10].isoformat(),
                "exit_date": dates[16].isoformat(),
                "entry_price": 104.0,
                "exit_price": 108.5,
                "return_pct": 0.0432,
                "exit_reason": "take_profit",
            }
        ],
        trade_replays=replays,
    )
    assert "Trade K-Line Replay Inspector" in run_html
    assert "selectRunReplay" in run_html


def test_web_sse_stream_replays_and_validation(tmp_path: Path) -> None:
    cfg = AppConfig(db_path=tmp_path / "phase6_web.db")
    svc = AlphaEvoWebService(cfg, report_dir=tmp_path / "reports")
    server = create_web_server(svc, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as resp:
            html_txt = resp.read().decode("utf-8")
        assert "Stream Stock RSI Evolution" in html_txt
        assert "Adversarial Council" in html_txt
        assert "Trade K-Line Replay Inspector" in html_txt

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        assert health["version"] == "0.2.0"
        route_paths = {r["path"] for r in health["manifest"]["routes"]}
        assert "/api/rsi/stream" in route_paths
        assert health["manifest"]["capabilities"]["sse_progress"] is True

        # POST /api/runs includes trade_replays with kline_svg
        run_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/runs",
            data=json.dumps({"strategy_id": "ma_crossover_v1"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(run_req, timeout=30) as resp:
            run_obj = json.loads(resp.read().decode("utf-8"))
        assert isinstance(run_obj.get("trade_replays"), list)
        if run_obj["trade_replays"]:
            assert "<svg" in run_obj["trade_replays"][0]["kline_svg"]

        # Same-family council returns HTTP 400 on /api/rsi/run
        bad_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rsi/run",
            data=json.dumps(
                {
                    "strategy_id": "rsi_reversion_v1",
                    "rounds": 1,
                    "proposer_model": "openai/gpt-5",
                    "prosecutor_model": "openai/gpt-4o",
                    "api_key": "sk-test-secret",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(bad_req, timeout=15)
            raise AssertionError("Expected HTTPError 400 for same-family council")
        except urllib.error.HTTPError as err:
            assert err.code == 400

        # POST /api/rsi/stream delivers SSE events and final result without leaking api_key
        stream_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rsi/stream",
            data=json.dumps(
                {
                    "strategy_id": "rsi_reversion_v1",
                    "rounds": 2,
                    "proposer_model": "heuristic",
                    "prosecutor_model": "heuristic",
                    "api_key": "sk-should-never-appear-in-sse",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(stream_req, timeout=45) as resp:
            assert resp.headers.get("Content-Type", "").startswith("text/event-stream")
            raw_sse = resp.read().decode("utf-8")

        assert "sk-should-never-appear-in-sse" not in raw_sse
        parsed_events: list[tuple[str, dict[str, Any]]] = []
        for blk in raw_sse.strip().split("\n\n"):
            ev_name = ""
            data_str = ""
            for line in blk.splitlines():
                if line.startswith("event:"):
                    ev_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_str += line.split(":", 1)[1].strip()
            if ev_name and data_str:
                parsed_events.append((ev_name, json.loads(data_str)))

        names = [n for n, _ in parsed_events]
        assert names[0] == "start"
        assert "baseline" in names
        assert "round_complete" in names
        assert names[-1] == "result"
        final_res = parsed_events[-1][1]
        assert final_res["kernel_intact"] is True
        assert "<svg" in final_res["evolution_tree_svg"]
        assert isinstance(final_res["deliberations"], list)
        assert isinstance(final_res["veto_breakdown"], dict)
    finally:
        server.shutdown()
        server.server_close()


def test_arena_cli_dockerfile_and_pages_workflow(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    import alphaevo.cli.commands.arena as arena_cli

    monkeypatch.setattr(
        arena_cli, "_get_config", lambda: AppConfig(db_path=tmp_path / "phase6_arena.db")
    )
    out_dir = tmp_path / "arena_site"
    res = CliRunner().invoke(
        app,
        [
            "arena",
            "run",
            "--strategies",
            "rsi_reversion,ma_crossover",
            "--rounds",
            "2",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    snap = ArenaSnapshot.model_validate_json((out_dir / "arena_snapshot.json").read_text("utf-8"))
    assert snap.total_deliberations >= 1

    dockerfile_txt = Path("Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 8765" in dockerfile_txt
    assert 'CMD ["web", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]' in dockerfile_txt
    assert "SKILL.md" in dockerfile_txt

    wf = yaml.safe_load(Path(".github/workflows/arena-daily.yml").read_text(encoding="utf-8"))
    assert "deploy-pages" in wf["jobs"]
    assert wf["permissions"]["pages"] == "write"
    assert wf["permissions"]["id-token"] == "write"

    # Also ensure date range typing smoke check remains clean
    assert isinstance(date.today(), date)


def test_cold_process_imports_have_no_circular_import() -> None:
    for mod in (
        "alphaevo.web.ui",
        "alphaevo.web.server",
        "alphaevo.mcp.server",
        "alphaevo.arena",
        "alphaevo.evaluator",
    ):
        subprocess.run([sys.executable, "-c", f"import {mod}"], check=True)
