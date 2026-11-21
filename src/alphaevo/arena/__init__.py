"""Live Self-Evolving Strategy Arena & Interactive HTML Dashboard Renderers."""

from alphaevo.arena.html_renderer import render_arena_html, render_equity_svg, render_run_html
from alphaevo.arena.models import (
    ArenaSnapshot,
    ArenaStrategyEntry,
    EquityPoint,
    EvolutionNodeView,
)
from alphaevo.arena.runner import ArenaRunner

__all__ = [
    "ArenaRunner",
    "ArenaSnapshot",
    "ArenaStrategyEntry",
    "EquityPoint",
    "EvolutionNodeView",
    "render_arena_html",
    "render_equity_svg",
    "render_run_html",
]
