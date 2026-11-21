"""CLI entrypoint for launching the zero-dependency local Web Research Studio."""

from __future__ import annotations

import contextlib
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from alphaevo.cli._helpers import _get_config

console = Console()


def web_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host address"),
    port: int = typer.Option(8765, "--port", "-p", help="HTTP port"),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not open default web browser automatically"
    ),
    report_dir: str = typer.Option(
        "reports", "--report-dir", "-o", help="Root directory for run artifacts"
    ),
    live: bool = typer.Option(
        False, "--live", help="Default to live market fetches instead of bundled snapshot"
    ),
) -> None:
    """Launch the zero-dependency AlphaEvo Web Research Studio in your browser."""
    from alphaevo.web.server import AlphaEvoWebService, create_web_server

    config = _get_config()
    service = AlphaEvoWebService(config, report_dir=Path(report_dir), live_data=live)
    server = create_web_server(service, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}/"
    console.print(
        Panel(
            f"[bold cyan]AlphaEvo Web Research Studio[/bold cyan]\n"
            f"Listening on [link={url}]{url}[/link]\n"
            "[dim]Press Ctrl+C to stop · Research & educational software only.[/dim]",
            border_style="green",
        )
    )
    if not no_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping AlphaEvo Web Research Studio...[/yellow]")
    finally:
        server.server_close()


__all__ = ["web_command"]
