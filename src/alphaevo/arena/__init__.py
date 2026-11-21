"""Live Self-Evolving Strategy Arena & Interactive HTML Dashboard Renderers."""

from alphaevo.arena.html_renderer import (
    render_arena_html,
    render_equity_svg,
    render_evolution_tree_svg,
    render_map_elites_svg,
    render_run_html,
    render_trade_kline_svg,
)
from alphaevo.arena.models import (
    ArenaSnapshot,
    ArenaStrategyEntry,
    EquityPoint,
    EvolutionNodeView,
    TradeReplayView,
)
from alphaevo.arena.runner import ArenaRunner

__all__ = [
    "ArenaRunner",
    "ArenaSnapshot",
    "ArenaStrategyEntry",
    "EquityPoint",
    "EvolutionNodeView",
    "TradeReplayView",
    "render_arena_html",
    "render_equity_svg",
    "render_evolution_tree_svg",
    "render_map_elites_svg",
    "render_run_html",
    "render_trade_kline_svg",
]
