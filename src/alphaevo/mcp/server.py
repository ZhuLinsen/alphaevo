"""Stdlib JSON-RPC 2.0 MCP STDIO server exposing AlphaEvo research, RSI, and governance tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, BinaryIO

from alphaevo import __version__
from alphaevo.core.config import AppConfig
from alphaevo.web.server import AlphaEvoWebService

_PROTOCOL_VERSION = "2024-11-05"


def _program_resource_text() -> str:
    candidates = [
        Path.cwd() / "examples" / "rsi_stock_program.md",
        Path(__file__).resolve().parents[3] / "examples" / "rsi_stock_program.md",
    ]
    for p in candidates:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    try:
        from alphaevo.cli.demo import _resolve_bundled_path

        bundled = _resolve_bundled_path("rsi_stock_program.md")
        if bundled is not None and bundled.is_file():
            return bundled.read_text(encoding="utf-8")
    except Exception:
        pass
    return "# AlphaEvo Stock RSI Charter\nHistorical backtests only veto; forward shadow ledgers promote.\n"


_TOOLS: list[dict[str, Any]] = [
    {
        "name": "alphaevo_list_strategies",
        "description": "List available built-in and stored AlphaEvo stock strategies (YAML DSL).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "alphaevo_draft_strategy",
        "description": "Compile a plain-language trading idea into an executable AlphaEvo YAML DSL strategy (no LLM required).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "idea": {"type": "string", "description": "Natural language entry/exit rules"},
                "market": {"type": "string", "enum": ["a_share", "us", "hk"], "default": "us"},
                "strategy_id": {"type": "string"},
                "save": {"type": "boolean", "default": False},
            },
            "required": ["idea"],
        },
    },
    {
        "name": "alphaevo_evaluate_strategy",
        "description": "Evaluate a strategy through EvaluationKernel (daily MTM, DSR, null control & ResearchCommittee) and save immutable artifacts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
                "strategy_yaml": {"type": "string"},
                "live": {"type": "boolean", "default": False},
                "protocol": {
                    "type": "string",
                    "enum": ["fast_research_v1", "full_research_v1", "official_v1"],
                    "default": "full_research_v1",
                },
            },
        },
    },
    {
        "name": "alphaevo_evolve_rsi",
        "description": "Run L0-grounded Stock RSI keep/revert ratchet evolution with DSR, null control & optional locked holdout/portfolio veto.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
                "strategy_yaml": {"type": "string"},
                "rounds": {"type": "integer", "minimum": 1, "maximum": 6, "default": 2},
                "holdout_pct": {"type": "number", "minimum": 0.0, "maximum": 0.4, "default": 0.0},
                "portfolio_gate": {"type": "boolean", "default": False},
                "live": {"type": "boolean", "default": False},
                "proposer_model": {
                    "type": "string",
                    "description": "Optional LiteLLM proposer model (or 'heuristic')",
                },
                "prosecutor_model": {
                    "type": "string",
                    "description": "Optional adversarial LiteLLM prosecutor model (must differ in family from proposer when both are LLMs)",
                },
            },
        },
    },
    {
        "name": "alphaevo_null_audit",
        "description": "Run Tier-2 Anti-Delusion null audit verifying 0 false promotions occur across shuffled-return null worlds.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string", "default": "rsi_reversion_v1"},
                "worlds": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
                "rounds": {"type": "integer", "minimum": 1, "maximum": 4, "default": 2},
            },
        },
    },
    {
        "name": "alphaevo_shadow_simulate",
        "description": "Simulate day-by-day bitemporal forward shadow commitments, T+H outcome backfills, SHA-256 verification & promotion/halt transitions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
                "strategy_yaml": {"type": "string"},
                "days": {"type": "integer", "minimum": 10, "maximum": 120, "default": 35},
                "horizon": {"type": "integer", "enum": [5, 10, 20], "default": 5},
                "live": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "alphaevo_audit_factors",
        "description": "Audit FactorStore dynamic alpha factors for OOS IC/IR decay, runtime PIT invariance, redundancy & strategy lineage.",
        "inputSchema": {
            "type": "object",
            "properties": {"auto_retire": {"type": "boolean", "default": False}},
        },
    },
    {
        "name": "alphaevo_allocate_portfolio",
        "description": "Compute strictly causal regime-adaptive multi-strategy risk budget weights (inverse-vol, regime haircut, correlation penalty & drawdown deleveraging).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "live": {"type": "boolean", "default": False},
            },
            "required": ["strategy_ids"],
        },
    },
    {
        "name": "alphaevo_list_runs",
        "description": "List indexed immutable research run artifacts saved in SQLite.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
            },
        },
    },
]

_RESOURCES: list[dict[str, str]] = [
    {
        "uri": "alphaevo://program",
        "name": "Stock RSI Research Charter (program.md)",
        "description": "Default research charter separating fast offline veto search from slow forward promotion.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "alphaevo://strategies",
        "name": "Available Strategy Catalog",
        "description": "JSON catalog of stored and built-in YAML strategies.",
        "mimeType": "application/json",
    },
]


class AlphaEvoMCPServer:
    """JSON-RPC 2.0 MCP Server exposing AlphaEvo workflows over STDIO."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        report_dir: Path | str = "reports",
    ) -> None:
        self.config = config or AppConfig()
        self.service = AlphaEvoWebService(self.config, report_dir=Path(report_dir))

    def handle_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC 2.0 request or notification."""
        method = payload.get("method")
        msg_id = payload.get("id")
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        # Notifications without id require no response
        if msg_id is None:
            return None

        try:
            if method == "initialize":
                return self._ok(
                    msg_id,
                    {
                        "protocolVersion": _PROTOCOL_VERSION,
                        "capabilities": {
                            "tools": {"listChanged": False},
                            "resources": {"subscribe": False, "listChanged": False},
                        },
                        "serverInfo": {"name": "alphaevo", "version": __version__},
                    },
                )
            if method == "ping":
                return self._ok(msg_id, {})
            if method == "tools/list":
                return self._ok(msg_id, {"tools": _TOOLS})
            if method == "tools/call":
                name = str(params.get("name", ""))
                args = params.get("arguments") or {}
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                result = self._call_tool(name, args)
                return self._ok(
                    msg_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    result, ensure_ascii=False, indent=2, default=str
                                ),
                            }
                        ],
                        "isError": False,
                    },
                )
            if method == "resources/list":
                return self._ok(msg_id, {"resources": _RESOURCES})
            if method == "resources/read":
                uri = str(params.get("uri", ""))
                if uri == "alphaevo://program":
                    return self._ok(
                        msg_id,
                        {
                            "contents": [
                                {
                                    "uri": uri,
                                    "mimeType": "text/markdown",
                                    "text": _program_resource_text(),
                                }
                            ]
                        },
                    )
                if uri == "alphaevo://strategies":
                    return self._ok(
                        msg_id,
                        {
                            "contents": [
                                {
                                    "uri": uri,
                                    "mimeType": "application/json",
                                    "text": json.dumps(
                                        self.service.list_strategies(),
                                        ensure_ascii=False,
                                        indent=2,
                                        default=str,
                                    ),
                                }
                            ]
                        },
                    )
                return self._err(msg_id, -32602, f"Unknown resource URI: {uri}")

            return self._err(msg_id, -32601, f"Method not found: {method}")
        except Exception as exc:
            if method == "tools/call":
                return self._ok(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                )
            return self._err(msg_id, -32000, str(exc))

    def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "alphaevo_list_strategies":
            return self.service.list_strategies()
        if name == "alphaevo_draft_strategy":
            return self.service.draft_strategy(
                str(args.get("idea", "")),
                market=str(args.get("market", "us")),
                strategy_id=args.get("strategy_id"),
                save=bool(args.get("save", False)),
            )
        if name == "alphaevo_evaluate_strategy":
            return self.service.run_strategy(
                strategy_id=args.get("strategy_id"),
                strategy_yaml=args.get("strategy_yaml"),
                live=args.get("live"),
                protocol=str(args.get("protocol", "full_research_v1")),
            )
        if name == "alphaevo_evolve_rsi":
            return self.service.run_rsi(
                strategy_id=args.get("strategy_id"),
                strategy_yaml=args.get("strategy_yaml"),
                rounds=int(args.get("rounds", 2)),
                holdout_pct=float(args.get("holdout_pct", 0.0)),
                portfolio_gate=bool(args.get("portfolio_gate", False)),
                live=args.get("live"),
                proposer_model=args.get("proposer_model"),
                prosecutor_model=args.get("prosecutor_model"),
            )
        if name == "alphaevo_null_audit":
            return self._run_null_audit(
                strategy_id=str(args.get("strategy_id", "rsi_reversion_v1")),
                worlds=int(args.get("worlds", 2)),
                rounds=int(args.get("rounds", 2)),
            )
        if name == "alphaevo_shadow_simulate":
            return self.service.shadow_simulate(
                strategy_id=args.get("strategy_id"),
                strategy_yaml=args.get("strategy_yaml"),
                days=int(args.get("days", 35)),
                horizon=int(args.get("horizon", 5)),
                live=args.get("live"),
            )
        if name == "alphaevo_audit_factors":
            return self.service.audit_factors(auto_retire=bool(args.get("auto_retire", False)))
        if name == "alphaevo_allocate_portfolio":
            sids = [str(s) for s in (args.get("strategy_ids") or [])]
            return self.service.allocate_portfolio(strategy_ids=sids, live=args.get("live"))
        if name == "alphaevo_list_runs":
            return self.service.list_runs(limit=int(args.get("limit", 20)))
        raise ValueError(f"Unknown MCP tool: {name}")

    def _run_null_audit(
        self,
        *,
        strategy_id: str,
        worlds: int,
        rounds: int,
    ) -> dict[str, Any]:
        from alphaevo.models.enums import RSILevel
        from alphaevo.models.strategy import Strategy
        from alphaevo.rsi.engine import StockRSIEngine
        from alphaevo.rsi.program import RSICharter
        from alphaevo.truth.null_control import NullSignalController

        strat = self.service._resolve_input_strategy(strategy_id=strategy_id)
        market_data, _src, _repro, _mut = self.service._load_market_data(live=False)

        def _run_world(seed_strat: Strategy, null_market: dict[str, Any]) -> int:
            engine = StockRSIEngine(db_path=":memory:", backtest_config=self.config.backtest)
            charter = RSICharter(
                name="mcp_null_audit",
                seed_strategy_id=seed_strat.meta.id,
                allowed_levels=[RSILevel.L1_STRATEGY],
                max_rounds=max(1, rounds),
                require_null_control=True,
                null_permutations=30,
            )
            res = engine.run(seed_strat, null_market, charter=charter)
            return res.promotions

        audit = NullSignalController().verify_zero_promotion_under_shuffled_returns(
            strat,
            market_data,
            _run_world,
            n_worlds=max(1, worlds),
        )
        return audit.model_dump(mode="json")

    def serve_stdio(
        self,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
    ) -> None:
        """Serve JSON-RPC 2.0 MCP messages over STDIO (supports newline and Content-Length)."""
        in_stream = stdin if stdin is not None else sys.stdin.buffer
        out_stream = stdout if stdout is not None else sys.stdout.buffer

        while True:
            msg, use_lsp = self._read_frame(in_stream)
            if msg is None:
                break
            reply = self.handle_message(msg)
            if reply is not None:
                self._write_frame(out_stream, reply, use_lsp=use_lsp)

    @staticmethod
    def _read_frame(stream: BinaryIO) -> tuple[dict[str, Any] | None, bool]:
        while True:
            raw_line = stream.readline()
            if not raw_line:
                return None, False
            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if text.lower().startswith("content-length:"):
                content_length = int(text.split(":", 1)[1].strip())
                while True:
                    hdr = stream.readline()
                    if not hdr or hdr in {b"\r\n", b"\n"}:
                        break
                body = stream.read(content_length)
                if not body:
                    return None, True
                parsed = json.loads(body.decode("utf-8"))
                return (parsed if isinstance(parsed, dict) else None), True
            parsed = json.loads(text)
            return (parsed if isinstance(parsed, dict) else None), False

    @staticmethod
    def _write_frame(
        stream: BinaryIO,
        payload: dict[str, Any],
        *,
        use_lsp: bool = False,
    ) -> None:
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        if use_lsp:
            header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
            stream.write(header + raw)
        else:
            stream.write(raw + b"\n")
        stream.flush()

    @staticmethod
    def _ok(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


__all__ = ["AlphaEvoMCPServer"]
