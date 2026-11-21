"""Phase 5 tests: multi-channel notifications, Live Arena HTML, zero-dep Web Studio, and MCP server."""

from __future__ import annotations

import io
import json
import threading
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from alphaevo.arena import ArenaRunner, render_arena_html, render_equity_svg, render_run_html
from alphaevo.cli.commands.arena import arena_app  # noqa: F401
from alphaevo.cli.demo import load_showcase_snapshot
from alphaevo.cli.main import app
from alphaevo.core.config import AppConfig, ConfigManager, NotifyConfig
from alphaevo.mcp.server import AlphaEvoMCPServer
from alphaevo.notify import NotificationDispatcher, NotificationMessage
from alphaevo.provenance.store import build_safe_config_snapshot
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.web.server import AlphaEvoWebService, create_web_server


def test_notifier_payloads_send_and_secret_scrubbing(monkeypatch) -> None:
    monkeypatch.setenv(
        "ALPHAEVO_FEISHU_WEBHOOK", "https://open.feishu.cn/open-apis/bot/v2/hook/secret1"
    )
    monkeypatch.setenv(
        "ALPHAEVO_WECOM_WEBHOOK", "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=secret2"
    )
    monkeypatch.setenv(
        "ALPHAEVO_DINGTALK_WEBHOOK", "https://oapi.dingtalk.com/robot/send?access_token=secret3"
    )
    monkeypatch.setenv("ALPHAEVO_TELEGRAM_BOT_TOKEN", "123456:secret-tg-token")
    monkeypatch.setenv("ALPHAEVO_TELEGRAM_CHAT_ID", "-100999888")
    monkeypatch.setenv("ALPHAEVO_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/1/secret4")
    monkeypatch.setenv("ALPHAEVO_SLACK_WEBHOOK", "https://hooks.slack.com/services/T/B/secret5")
    monkeypatch.setenv("ALPHAEVO_WEBHOOK_URL", "https://example.com/alphaevo-hook")

    cfg = ConfigManager().load()
    safe_snap = build_safe_config_snapshot(cfg)
    safe_json = json.dumps(safe_snap)
    assert "secret-tg-token" not in safe_json
    assert "secret1" not in safe_json
    assert "notify" not in safe_snap

    dispatcher = NotificationDispatcher(NotifyConfig.model_validate(cfg.notify.model_dump()))
    channels = dispatcher.configured_channels()
    assert channels == [
        "feishu",
        "wecom",
        "dingtalk",
        "telegram",
        "discord",
        "slack",
        "webhook",
    ]

    msg = NotificationMessage(
        title="Arena Daily Update",
        summary="2 strategies evolved cleanly",
        markdown_body="- `rsi_reversion_v1`: win 52%",
        severity="success",
        url="https://example.org/arena/",
        tags=["alphaevo", "daily"],
    )

    dry_results = dispatcher.send(msg, dry_run=True)
    assert len(dry_results) == 7
    assert all(r.success and r.dry_run for r in dry_results)

    captured: list[tuple[str, dict[str, Any]]] = []

    class _FakeResponse:
        status = 200

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 10.0) -> _FakeResponse:
        body = json.loads(req.data.decode("utf-8")) if req.data else {}
        captured.append((req.full_url, body))
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    live_results = dispatcher.send(msg, dry_run=False)
    assert len(live_results) == 7
    assert all(r.success and not r.dry_run and r.status_code == 200 for r in live_results)
    assert len(captured) == 7
    assert any(
        "feishu.cn" in url and body.get("msg_type") == "interactive" for url, body in captured
    )
    assert any(
        "api.telegram.org/bot123456:secret-tg-token/sendMessage" in url for url, _ in captured
    )
    assert any("discord.com" in url and "embeds" in body for url, body in captured)


def test_arena_runner_html_export_and_cli(tmp_path: Path, monkeypatch) -> None:
    parser = StrategyParser()
    s1 = parser.parse_file(Path("strategies/builtin/rsi_reversion.yaml"))
    s2 = parser.parse_file(Path("strategies/builtin/ma_crossover.yaml"))
    market_data, _ = load_showcase_snapshot()

    runner = ArenaRunner()
    snapshot = runner.run(
        [s1, s2],
        market_data,
        rounds=1,
        market_label="Unit Snapshot",
        data_source="unit_test",
    )
    assert snapshot.total_strategies == 2
    assert snapshot.chain_verified is True
    assert snapshot.date_range[0] < snapshot.date_range[1]
    assert len(snapshot.allocation_weights) == 2

    html_doc = render_arena_html(snapshot)
    assert "AlphaEvo Live Self-Evolving Strategy Arena" in html_doc
    assert "https://cdn" not in html_doc
    assert "<svg" in html_doc
    assert "not investment advice" in html_doc

    empty_svg = render_equity_svg([])
    assert "<svg" in empty_svg

    run_html = render_run_html(
        run_id="20261121T000000Z_rsi_reversion_v1_abc12345",
        manifest={"effective_protocol": "full_research_v1", "strategy": {"id": "rsi_reversion_v1"}},
        evaluation={
            "confidence_score": 0.55,
            "overall": {
                "win_rate": 0.5,
                "avg_return": 0.01,
                "sharpe_ratio": 1.1,
                "max_drawdown": 0.12,
                "signal_count": 10,
            },
        },
        strategy_yaml="meta:\n  id: rsi_reversion_v1\n",
        trades=[
            {
                "symbol": "AAPL",
                "signal_date": "2025-05-01",
                "exit_date": "2025-05-10",
                "entry_price": 100.0,
                "exit_price": 103.0,
                "return_pct": 0.03,
                "exit_reason": "take_profit",
            }
        ],
    )
    assert "Equity Curve" in run_html
    assert "AAPL" in run_html

    from typer.testing import CliRunner

    import alphaevo.cli.commands.arena as arena_cli

    monkeypatch.setattr(arena_cli, "_get_config", lambda: AppConfig(db_path=tmp_path / "cli.db"))
    arena_out = tmp_path / "arena_out"
    res = CliRunner().invoke(
        app,
        [
            "arena",
            "run",
            "--strategies",
            "rsi_reversion,ma_crossover",
            "--rounds",
            "1",
            "--output-dir",
            str(arena_out),
            "--dry-run-notify",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (arena_out / "index.html").is_file()
    assert (arena_out / "arena_snapshot.json").is_file()
    assert (arena_out / "digest.md").is_file()

    site_out = tmp_path / "site_out"
    build_res = CliRunner().invoke(
        app,
        [
            "arena",
            "build-site",
            "--snapshot",
            str(arena_out / "arena_snapshot.json"),
            "--output-dir",
            str(site_out),
        ],
    )
    assert build_res.exit_code == 0, build_res.output
    assert (site_out / "index.html").is_file()


def test_web_service_and_real_http_endpoints(tmp_path: Path) -> None:
    cfg = AppConfig(db_path=tmp_path / "web.db")
    report_dir = tmp_path / "reports"
    svc = AlphaEvoWebService(cfg, report_dir=report_dir)
    server = create_web_server(svc, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def _req(path: str, payload: dict[str, Any] | None = None) -> tuple[int, str]:
        url = f"http://127.0.0.1:{port}{path}"
        if payload is None:
            r = urllib.request.Request(url, method="GET")
        else:
            r = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        with urllib.request.urlopen(r, timeout=30) as resp:
            return int(resp.status), resp.read().decode("utf-8")

    try:
        st, html_body = _req("/")
        assert st == 200
        assert "AlphaEvo Research Studio" in html_body

        st, health_txt = _req("/api/health")
        assert st == 200
        assert json.loads(health_txt)["status"] == "ok"

        st, strat_txt = _req("/api/strategies")
        assert st == 200
        assert json.loads(strat_txt)["count"] >= 4

        st, draft_txt = _req(
            "/api/strategies/draft",
            {"idea": "RSI below 32 above MA20 with 5% stop loss", "market": "us"},
        )
        assert st == 200
        draft_obj = json.loads(draft_txt)
        assert "rsi_14" in draft_obj["yaml"]

        st, run_txt = _req("/api/runs", {"strategy_id": "rsi_reversion_v1"})
        assert st == 200
        run_obj = json.loads(run_txt)
        assert run_obj["run_id"]
        assert "<svg" in run_obj["equity_svg"]

        st, runs_txt = _req("/api/runs?limit=5")
        assert st == 200
        assert json.loads(runs_txt)["count"] >= 1

        # Verify `alphaevo runs html <run_id>` against the run created via HTTP
        from typer.testing import CliRunner

        import alphaevo.cli.commands.runs as runs_cli

        out_html = tmp_path / "exported_run.html"
        original_get_cfg = runs_cli._get_config
        runs_cli._get_config = lambda: cfg  # type: ignore[assignment]
        try:
            cli_html = CliRunner().invoke(
                app,
                [
                    "runs",
                    "html",
                    run_obj["run_id"],
                    "--artifact-root",
                    str(report_dir),
                    "--output",
                    str(out_html),
                ],
            )
            assert cli_html.exit_code == 0, cli_html.output
            assert out_html.is_file()
        finally:
            runs_cli._get_config = original_get_cfg  # type: ignore[assignment]

        st, rsi_txt = _req("/api/rsi/run", {"strategy_id": "rsi_reversion_v1", "rounds": 1})
        assert st == 200
        rsi_obj = json.loads(rsi_txt)
        assert rsi_obj["kernel_intact"] is True
        assert len(rsi_obj["rows"]) >= 1

        st, sim_txt = _req(
            "/api/shadow/simulate",
            {"strategy_id": "ma_crossover_v1", "days": 25, "horizon": 5},
        )
        assert st == 200
        assert json.loads(sim_txt)["report"]["chain_verified"] is True

        st, alloc_txt = _req(
            "/api/portfolio/allocate",
            {"strategy_ids": ["rsi_reversion_v1", "ma_crossover_v1"]},
        )
        assert st == 200
        assert len(json.loads(alloc_txt)["report"]["strategies"]) == 2
    finally:
        server.shutdown()
        server.server_close()


def test_mcp_server_jsonrpc_and_stdio_framing(tmp_path: Path) -> None:
    mcp = AlphaEvoMCPServer(
        config=AppConfig(db_path=tmp_path / "mcp.db"),
        report_dir=tmp_path / "reports",
    )

    init_resp = mcp.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "alphaevo"

    assert mcp.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    tools_resp = mcp.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert {
        "alphaevo_list_strategies",
        "alphaevo_draft_strategy",
        "alphaevo_evaluate_strategy",
        "alphaevo_evolve_rsi",
        "alphaevo_null_audit",
        "alphaevo_shadow_simulate",
        "alphaevo_audit_factors",
        "alphaevo_allocate_portfolio",
        "alphaevo_list_runs",
    } <= tool_names

    res_read = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "alphaevo://program"},
        }
    )
    assert res_read is not None
    assert (
        "program.md" in res_read["result"]["contents"][0]["text"]
        or "RSI" in res_read["result"]["contents"][0]["text"]
    )

    draft_call = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "alphaevo_draft_strategy",
                "arguments": {"idea": "RSI below 30 above MA20", "market": "us"},
            },
        }
    )
    assert draft_call is not None
    assert draft_call["result"]["isError"] is False
    draft_payload = json.loads(draft_call["result"]["content"][0]["text"])
    assert "yaml" in draft_payload

    null_call = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "alphaevo_null_audit",
                "arguments": {"strategy_id": "rsi_reversion_v1", "worlds": 1, "rounds": 1},
            },
        }
    )
    assert null_call is not None
    null_payload = json.loads(null_call["result"]["content"][0]["text"])
    assert null_payload["passed"] is True

    # Test newline framing AND Content-Length framing in serve_stdio
    msg1 = json.dumps({"jsonrpc": "2.0", "id": 10, "method": "ping"}).encode("utf-8") + b"\n"
    msg2_body = json.dumps({"jsonrpc": "2.0", "id": 11, "method": "tools/list"}).encode("utf-8")
    msg2 = f"Content-Length: {len(msg2_body)}\r\n\r\n".encode("ascii") + msg2_body
    in_buf = io.BytesIO(msg1 + msg2)
    out_buf = io.BytesIO()
    mcp.serve_stdio(stdin=in_buf, stdout=out_buf)
    raw_out = out_buf.getvalue().decode("utf-8")
    assert '"id": 10' in raw_out
    assert "Content-Length:" in raw_out
    assert '"id": 11' in raw_out


def test_skill_markdown_and_workflows_valid() -> None:
    skill_text = Path("SKILL.md").read_text(encoding="utf-8")
    assert skill_text.startswith("---\n")
    frontmatter = skill_text.split("---\n", 2)[1]
    meta = yaml.safe_load(frontmatter)
    assert meta["name"] == "alphaevo"
    assert "mcp" in skill_text

    arena_wf = yaml.safe_load(Path(".github/workflows/arena-daily.yml").read_text(encoding="utf-8"))
    assert "evolve-arena" in arena_wf["jobs"]

    release_wf = yaml.safe_load(Path(".github/workflows/release.yml").read_text(encoding="utf-8"))
    assert "build-and-verify" in release_wf["jobs"]
