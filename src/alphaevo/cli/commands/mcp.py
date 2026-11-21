"""CLI entrypoint for AlphaEvo Model Context Protocol (MCP) STDIO server."""

from __future__ import annotations

from pathlib import Path

import typer

from alphaevo.cli._helpers import _get_config


def mcp_command(
    report_dir: str = typer.Option(
        "reports", "--report-dir", "-o", help="Root directory for persisted run artifacts"
    ),
) -> None:
    """Start AlphaEvo MCP (Model Context Protocol) JSON-RPC 2.0 STDIO server."""
    from alphaevo.mcp.server import AlphaEvoMCPServer

    config = _get_config()
    server = AlphaEvoMCPServer(config=config, report_dir=Path(report_dir))
    server.serve_stdio()


__all__ = ["mcp_command"]
