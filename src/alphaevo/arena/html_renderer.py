"""Self-contained zero-CDN HTML/SVG renderer for Arena Observatories & Run Reports."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Any

from alphaevo.arena.models import (
    ArenaSnapshot,
    EquityPoint,
    EvolutionNodeView,
    TradeReplayView,
)


def _safe_json_for_script(obj: Any) -> str:
    """Serialize *obj* safely for inline <script> embedding."""
    raw = json.dumps(obj, ensure_ascii=False, default=str)
    return raw.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def render_equity_svg(
    points: Sequence[EquityPoint],
    *,
    width: int = 760,
    height: int = 220,
) -> str:
    """Render an inline SVG dual-line equity curve vs equal-weight benchmark."""
    if len(points) < 2:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
            'aria-label="Equity Curve Empty">'
            f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            'fill="#64748b" font-size="13">Not enough executed trades to plot daily MTM equity curve</text>'
            "</svg>"
        )

    pad_l, pad_r, pad_t, pad_b = 54, 18, 22, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    strat_vals = [float(p.strategy_equity) for p in points]
    bench_vals = [float(p.benchmark_equity) for p in points]
    all_vals = strat_vals + bench_vals
    min_v = min(all_vals)
    max_v = max(all_vals)
    if max_v - min_v < 1e-6:
        max_v += 0.05
        min_v = max(0.01, min_v - 0.05)
    pad_span = (max_v - min_v) * 0.1
    lo_v = min_v - pad_span
    hi_v = max_v + pad_span
    span = max(1e-6, hi_v - lo_v)

    n = len(points)

    def _xy(idx: int, val: float) -> tuple[float, float]:
        x = pad_l + (idx / max(1, n - 1)) * plot_w
        y = pad_t + plot_h - ((val - lo_v) / span) * plot_h
        return round(x, 2), round(y, 2)

    strat_pts = [_xy(i, v) for i, v in enumerate(strat_vals)]
    bench_pts = [_xy(i, v) for i, v in enumerate(bench_vals)]

    strat_poly = " ".join(f"{x},{y}" for x, y in strat_pts)
    bench_poly = " ".join(f"{x},{y}" for x, y in bench_pts)
    area_pts = (
        f"{strat_pts[0][0]},{pad_t + plot_h} "
        + strat_poly
        + f" {strat_pts[-1][0]},{pad_t + plot_h}"
    )

    grid_lines: list[str] = []
    for step in range(4):
        frac = step / 3.0
        yv = round(pad_t + plot_h * (1.0 - frac), 2)
        val_lbl = lo_v + span * frac
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{yv}" x2="{width - pad_r}" y2="{yv}" '
            'stroke="#1e293b" stroke-dasharray="3,3" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{yv + 4}" text-anchor="end" '
            f'fill="#64748b" font-size="10">{val_lbl:.2f}x</text>'
        )

    start_d = html.escape(points[0].date)
    end_d = html.escape(points[-1].date)
    final_strat = strat_vals[-1]
    final_bench = bench_vals[-1]

    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="Equity Curve">'
        "<defs>"
        '<linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#10b981" stop-opacity="0.32"/>'
        '<stop offset="100%" stop-color="#10b981" stop-opacity="0.02"/>'
        "</linearGradient>"
        "</defs>"
        f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>'
        + "".join(grid_lines)
        + f'<polygon points="{area_pts}" fill="url(#eqGrad)"/>'
        f'<polyline fill="none" stroke="#64748b" stroke-width="1.6" stroke-dasharray="5,4" points="{bench_poly}"/>'
        f'<polyline fill="none" stroke="#10b981" stroke-width="2.4" points="{strat_poly}"/>'
        f'<circle cx="{strat_pts[-1][0]}" cy="{strat_pts[-1][1]}" r="4" fill="#10b981"/>'
        f'<text x="{pad_l}" y="{height - 8}" fill="#94a3b8" font-size="10">{start_d}</text>'
        f'<text x="{width - pad_r}" y="{height - 8}" text-anchor="end" fill="#94a3b8" font-size="10">{end_d}</text>'
        f'<text x="{pad_l + 6}" y="15" fill="#10b981" font-size="11" font-weight="600">Strategy {final_strat:.2f}x</text>'
        f'<text x="{pad_l + 130}" y="15" fill="#94a3b8" font-size="11">Benchmark {final_bench:.2f}x</text>'
        "</svg>"
    )


def render_trade_kline_svg(
    trade: TradeReplayView | dict[str, Any],
    *,
    width: int = 640,
    height: int = 210,
) -> str:
    """Render an inline SVG OHLC candlestick replay with entry/exit markers."""
    bars = _field(trade, "bars", []) or []
    symbol = str(_field(trade, "symbol", "TICKER"))
    signal_date = str(_field(trade, "signal_date", ""))
    exit_date = str(_field(trade, "exit_date", "") or signal_date)
    entry_px = float(_field(trade, "entry_price", 0.0) or 0.0)
    exit_px_raw = _field(trade, "exit_price", None)
    exit_px = float(exit_px_raw) if exit_px_raw is not None else entry_px
    ret_pct = float(_field(trade, "return_pct", 0.0) or 0.0)
    holding_days = int(_field(trade, "holding_days", 0) or 0)
    exit_reason = str(_field(trade, "exit_reason", "") or "close")

    if len(bars) < 2:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
            'aria-label="Trade K-Line Empty">'
            f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'fill="#64748b" font-size="13">{html.escape(symbol)}: insufficient OHLC bars for candlestick replay</text>'
            "</svg>"
        )

    pad_l, pad_r, pad_t, pad_b = 54, 18, 26, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    lows = [float(b.get("low", b.get("close", 0.0))) for b in bars]
    highs = [float(b.get("high", b.get("close", 0.0))) for b in bars]
    if entry_px > 0:
        lows.append(entry_px)
        highs.append(entry_px)
    if exit_px > 0:
        lows.append(exit_px)
        highs.append(exit_px)

    lo_p = min(lows)
    hi_p = max(highs)
    if hi_p - lo_p < 1e-6:
        hi_p += 1.0
        lo_p = max(0.01, lo_p - 1.0)
    pad_p = (hi_p - lo_p) * 0.1
    lo_v = lo_p - pad_p
    hi_v = hi_p + pad_p
    span = max(1e-6, hi_v - lo_v)

    n = len(bars)
    step_w = plot_w / max(1, n)
    candle_w = max(3.0, min(14.0, step_w * 0.56))

    def _cx(idx: int) -> float:
        return round(pad_l + (idx + 0.5) * step_w, 2)

    def _py(price: float) -> float:
        return round(pad_t + plot_h - ((price - lo_v) / span) * plot_h, 2)

    entry_idx = 0
    exit_idx = n - 1
    for i, b in enumerate(bars):
        bd = str(b.get("date", ""))
        if bd <= signal_date:
            entry_idx = i
        if bd <= exit_date:
            exit_idx = i
    if exit_idx < entry_idx:
        exit_idx = entry_idx

    band_x1 = round(_cx(entry_idx) - step_w * 0.45, 2)
    band_x2 = round(_cx(exit_idx) + step_w * 0.45, 2)
    band_w = max(4.0, round(band_x2 - band_x1, 2))
    band_color = "rgba(16,185,129,0.09)" if ret_pct >= 0 else "rgba(239,68,68,0.09)"

    grid_svg: list[str] = []
    for step in range(4):
        frac = step / 3.0
        yv = round(pad_t + plot_h * (1.0 - frac), 2)
        p_lbl = lo_v + span * frac
        grid_svg.append(
            f'<line x1="{pad_l}" y1="{yv}" x2="{width - pad_r}" y2="{yv}" '
            'stroke="#1e293b" stroke-dasharray="3,3" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{yv + 4}" text-anchor="end" '
            f'fill="#64748b" font-size="10">{p_lbl:.2f}</text>'
        )

    candles_svg: list[str] = []
    for i, b in enumerate(bars):
        o = float(b.get("open", b.get("close", 0.0)))
        h = float(b.get("high", max(o, float(b.get("close", 0.0)))))
        lo = float(b.get("low", min(o, float(b.get("close", 0.0)))))
        c = float(b.get("close", o))
        cx = _cx(i)
        is_up = c >= o
        col = "#10b981" if is_up else "#ef4444"
        y_hi, y_lo = _py(h), _py(lo)
        y_top = _py(max(o, c))
        y_bot = _py(min(o, c))
        body_h = max(1.5, round(y_bot - y_top, 2))
        candles_svg.append(
            f'<line x1="{cx}" y1="{y_hi}" x2="{cx}" y2="{y_lo}" stroke="{col}" stroke-width="1.3"/>'
            f'<rect x="{round(cx - candle_w / 2, 2)}" y="{y_top}" width="{round(candle_w, 2)}" '
            f'height="{body_h}" rx="1" fill="{col}"/>'
        )

    en_x, en_y = _cx(entry_idx), _py(entry_px)
    ex_x, ex_y = _cx(exit_idx), _py(exit_px)
    ex_col = "#10b981" if ret_pct >= 0 else "#ef4444"
    ret_col = "#10b981" if ret_pct >= 0 else "#ef4444"
    first_d = html.escape(str(bars[0].get("date", "")))
    last_d = html.escape(str(bars[-1].get("date", "")))

    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
        f'aria-label="Trade K-Line Replay {html.escape(symbol)}">'
        f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>'
        f'<rect x="{band_x1}" y="{pad_t}" width="{band_w}" height="{plot_h}" fill="{band_color}"/>'
        + "".join(grid_svg)
        + "".join(candles_svg)
        + f'<line x1="{pad_l}" y1="{en_y}" x2="{width - pad_r}" y2="{en_y}" '
        'stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="4,3"/>'
        f'<line x1="{pad_l}" y1="{ex_y}" x2="{width - pad_r}" y2="{ex_y}" '
        f'stroke="{ex_col}" stroke-width="1.2" stroke-dasharray="4,3"/>'
        f'<circle cx="{en_x}" cy="{en_y}" r="4.5" fill="#38bdf8" stroke="#0f172a" stroke-width="1.5"/>'
        f'<text x="{en_x}" y="{max(pad_t + 10, en_y - 7)}" text-anchor="middle" '
        'fill="#38bdf8" font-size="9.5" font-weight="700">BUY</text>'
        f'<circle cx="{ex_x}" cy="{ex_y}" r="4.5" fill="{ex_col}" stroke="#0f172a" stroke-width="1.5"/>'
        f'<text x="{ex_x}" y="{min(pad_t + plot_h - 4, ex_y + 13)}" text-anchor="middle" '
        f'fill="{ex_col}" font-size="9.5" font-weight="700">EXIT ({html.escape(exit_reason)})</text>'
        f'<text x="{pad_l}" y="{height - 8}" fill="#94a3b8" font-size="10">{first_d}</text>'
        f'<text x="{width - pad_r}" y="{height - 8}" text-anchor="end" fill="#94a3b8" font-size="10">{last_d}</text>'
        f'<text x="{pad_l + 4}" y="16" fill="#e2e8f0" font-size="11.5" font-weight="700">'
        f"{html.escape(symbol)} · {entry_px:.2f} → {exit_px:.2f} · {holding_days}d</text>"
        f'<text x="{width - pad_r}" y="16" text-anchor="end" fill="{ret_col}" font-size="11.5" font-weight="700">'
        f"{ret_pct:+.2%}</text>"
        "</svg>"
    )


def render_evolution_tree_svg(
    nodes: Sequence[EvolutionNodeView | dict[str, Any]],
    *,
    width: int = 680,
    height: int = 190,
) -> str:
    """Render an inline SVG DAG of parent -> child mutation rounds and gate statuses."""
    if not nodes:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
            'aria-label="Evolution Lineage Empty">'
            f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            'fill="#64748b" font-size="13">No evolution rounds recorded yet</text>'
            "</svg>"
        )

    n = len(nodes)
    pad_x = 24
    usable_w = width - pad_x * 2
    slot_w = usable_w / max(1, n)
    box_w = min(132.0, max(82.0, slot_w - 14.0))
    box_h = 68.0

    coords: list[tuple[float, float]] = []
    pos_by_id: dict[str, tuple[float, float]] = {}
    for idx, node in enumerate(nodes):
        cx = round(pad_x + (idx + 0.5) * slot_w, 2)
        status = str(_field(node, "status", "baseline"))
        y_offset = 0.0
        if status in {"revert", "vetoed"}:
            y_offset = 24.0
        elif status == "keep":
            y_offset = -18.0
        cy = round(height / 2.0 + y_offset, 2)
        coords.append((cx, cy))
        sid = str(_field(node, "strategy_id", ""))
        if sid and sid not in pos_by_id:
            pos_by_id[sid] = (cx, cy)

    palette = {
        "baseline": ("#38bdf8", "#0c2233"),
        "keep": ("#10b981", "#07291f"),
        "stepping_stone": ("#f59e0b", "#2a1d06"),
        "revert": ("#ef4444", "#2b0d12"),
        "vetoed": ("#ef4444", "#2b0d12"),
    }

    edges_svg: list[str] = []
    for idx, node in enumerate(nodes):
        if idx == 0:
            continue
        p_id = _field(node, "parent_id", None)
        p_xy = pos_by_id.get(str(p_id), coords[idx - 1]) if p_id else coords[idx - 1]
        c_xy = coords[idx]
        status = str(_field(node, "status", "revert"))
        stroke, _fill = palette.get(status, ("#64748b", "#182235"))
        dash = ' stroke-dasharray="4,3"' if status in {"revert", "vetoed"} else ""
        mid_x = round((p_xy[0] + c_xy[0]) / 2.0, 2)
        edges_svg.append(
            f'<path d="M {p_xy[0]} {p_xy[1]} C {mid_x} {p_xy[1]}, {mid_x} {c_xy[1]}, {c_xy[0]} {c_xy[1]}" '
            f'fill="none" stroke="{stroke}" stroke-width="2" opacity="0.75"{dash}/>'
        )

    nodes_svg: list[str] = []
    for idx, node in enumerate(nodes):
        cx, cy = coords[idx]
        x0 = round(cx - box_w / 2.0, 2)
        y0 = round(cy - box_h / 2.0, 2)
        status = str(_field(node, "status", "baseline"))
        rnd = int(_field(node, "round_num", idx))
        sid = str(_field(node, "strategy_id", f"v{idx}"))
        short_sid = sid if len(sid) <= 15 else ("..." + sid[-12:])
        score = float(_field(node, "score", 0.0) or 0.0)
        dsr = float(_field(node, "dsr_prob", _field(node, "dsr_probability", 0.0)) or 0.0)
        stroke, fill = palette.get(status, ("#64748b", "#182235"))
        nodes_svg.append(
            f'<rect x="{x0}" y="{y0}" width="{round(box_w, 2)}" height="{box_h}" '
            f'rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
            f'<text x="{round(x0 + 8, 2)}" y="{round(y0 + 17, 2)}" fill="{stroke}" '
            f'font-size="10" font-weight="700">R{rnd} · {html.escape(status.upper())}</text>'
            f'<text x="{round(x0 + 8, 2)}" y="{round(y0 + 35, 2)}" fill="#e2e8f0" '
            f'font-size="10.5" font-weight="600">{html.escape(short_sid)}</text>'
            f'<text x="{round(x0 + 8, 2)}" y="{round(y0 + 53, 2)}" fill="#94a3b8" '
            f'font-size="9.5">Fit {score * 100:.1f}% · DSR {dsr * 100:.0f}%</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="Evolution Lineage Tree">'
        f'<rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>'
        + "".join(edges_svg)
        + "".join(nodes_svg)
        + "</svg>"
    )


_BASE_CSS = """
:root {
  --bg: #0b0f17;
  --surface: #121826;
  --surface-2: #182235;
  --border: #243047;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --green: #10b981;
  --cyan: #38bdf8;
  --amber: #f59e0b;
  --red: #ef4444;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  line-height: 1.5;
}
.container { max-width: 1240px; margin: 0 auto; padding: 24px 20px 48px; }
.header {
  display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
  gap: 16px; border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 20px;
}
.brand h1 { margin: 0; font-size: 24px; letter-spacing: -0.02em; }
.brand p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 600; border: 1px solid var(--border);
  background: var(--surface-2); color: var(--cyan);
}
.badge.pass { color: var(--green); border-color: rgba(16,185,129,0.4); }
.badge.fail { color: var(--red); border-color: rgba(239,68,68,0.4); }
.badge.watch { color: var(--amber); border-color: rgba(245,158,11,0.4); }
.kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
.kpi {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
}
.kpi .label { font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.04em; }
.kpi .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 18px; margin-bottom: 18px;
}
.card h2 { margin: 0 0 12px; font-size: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 10px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; font-size: 12px; }
tr.clickable { cursor: pointer; transition: background 0.15s; }
tr.clickable:hover, tr.clickable.active { background: var(--surface-2); }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
.chart-svg { width: 100%; height: auto; display: block; border-radius: 12px; }
.node-list { display: flex; flex-direction: column; gap: 8px; max-height: 360px; overflow-y: auto; }
.node {
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
}
.node-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; }
.node-meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
.debate-vs {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px; margin-top: 8px;
}
.debate-side {
  background: #0e1523; border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px;
}
.trade-pill {
  width: 100%; text-align: left; background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 9px; padding: 8px 10px; color: var(--text); cursor: pointer; font-size: 12px;
  display: flex; justify-content: space-between; align-items: center;
}
.trade-pill:hover, .trade-pill.active { border-color: var(--cyan); background: #1d2a42; }
pre.code {
  margin: 0; background: #090d14; border: 1px solid var(--border); border-radius: 10px;
  padding: 12px; overflow-x: auto; font-size: 12px; color: #cbd5e1; max-height: 320px;
}
.diff-add { color: #34d399; }
.diff-del { color: #f87171; }
.diff-hdr { color: #38bdf8; font-weight: 600; }
.bar-track { background: #0f172a; border-radius: 999px; height: 10px; overflow: hidden; margin-top: 4px; }
.bar-fill { background: linear-gradient(90deg, var(--cyan), var(--green)); height: 100%; }
.bar-fill.veto { background: linear-gradient(90deg, var(--amber), var(--red)); }
footer { margin-top: 28px; color: var(--muted); font-size: 12px; text-align: center; }
"""

_VETO_LABELS: dict[str, str] = {
    "prosecutor_veto": "Prosecutor / Pinned Scars",
    "null_control": "Shuffled-Return Null Floor",
    "strategy_rerun_null": "Full-Rerun Null Worlds",
    "dsr_gate": "Deflated Sharpe Ratio (DSR)",
    "overfit_gate": "Train/Val/Test & Sensitivity Overfit",
    "min_signals": "Insufficient Trade Sample (< Min)",
    "max_drawdown": "Max Drawdown Breach",
    "negative_expectancy": "Non-Positive Expectancy",
    "portfolio_gate": "Layer-2 Portfolio Veto",
    "no_fitness_gain": "No Net Fitness Improvement",
    "mutation_failed": "Invalid Mutation Candidate",
}


def render_arena_html(snapshot: ArenaSnapshot) -> str:
    """Render interactive standalone single-file HTML for the Strategy Evolution Observatory."""
    entries_payload: list[dict[str, Any]] = []
    for entry in snapshot.entries:
        d = entry.model_dump(mode="json")
        d["equity_svg"] = render_equity_svg(entry.equity_curve)
        d["tree_svg"] = render_evolution_tree_svg(entry.evolution_nodes)
        replay_items: list[dict[str, Any]] = []
        for tr in entry.trade_replays:
            tr_d = tr.model_dump(mode="json")
            tr_d["kline_svg"] = render_trade_kline_svg(tr)
            replay_items.append(tr_d)
        d["trade_replays"] = replay_items
        entries_payload.append(d)

    payload_json = _safe_json_for_script(entries_payload)
    first_svg = entries_payload[0]["equity_svg"] if entries_payload else render_equity_svg([])
    first_tree = (
        entries_payload[0]["tree_svg"] if entries_payload else render_evolution_tree_svg([])
    )
    market_lbl = html.escape(snapshot.market_label)
    data_src = html.escape(snapshot.data_source)
    dr_text = f"{html.escape(snapshot.date_range[0])} → {html.escape(snapshot.date_range[1])}"

    rows_html: list[str] = []
    for idx, entry in enumerate(snapshot.entries):
        cls = "clickable active" if idx == 0 else "clickable"
        null_cls = "pass" if entry.null_passed else "fail"
        null_lbl = "PASS" if entry.null_passed else "FAIL"
        delta_col = "color:var(--green)" if entry.score_delta >= 0 else "color:var(--red)"
        rows_html.append(
            f'<tr class="{cls}" data-idx="{idx}" onclick="selectEntry({idx})">'
            f"<td><strong>{html.escape(entry.strategy_id)}</strong><br>"
            f'<span style="color:var(--muted);font-size:11px">{html.escape(entry.champion_id)}</span></td>'
            f"<td>{html.escape(entry.category)}</td>"
            f"<td>{entry.baseline_score:.1%} → <strong>{entry.champion_score:.1%}</strong> "
            f'<span style="{delta_col}">({entry.score_delta:+.1%})</span></td>'
            f"<td>{entry.win_rate:.1%}</td>"
            f"<td>{entry.avg_return:+.2%}</td>"
            f"<td>{entry.sharpe_ratio:.2f}</td>"
            f"<td>{entry.max_drawdown:.1%}</td>"
            f'<td><span class="badge {null_cls}">{null_lbl}</span></td>'
            f'<td><span class="badge">{html.escape(entry.shadow_stage)}</span></td>'
            "</tr>"
        )

    alloc_bars: list[str] = []
    for sid, weight in snapshot.allocation_weights.items():
        pct = max(0.0, min(100.0, weight * 100.0))
        alloc_bars.append(
            f'<div style="margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;font-size:12px">'
            f"<span>{html.escape(sid)}</span><span>{pct:.1f}%</span></div>"
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
            "</div>"
        )
    if not alloc_bars:
        alloc_bars.append(
            '<div style="color:var(--muted);font-size:12px">Need >= 2 strategies for causal risk allocation.</div>'
        )

    veto_bars: list[str] = []
    total_veto_events = sum(snapshot.veto_breakdown.values())
    if snapshot.veto_breakdown and total_veto_events > 0:
        for reason, cnt in sorted(snapshot.veto_breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = max(4.0, min(100.0, (cnt / total_veto_events) * 100.0))
            lbl = _VETO_LABELS.get(reason, reason)
            veto_bars.append(
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px">'
                f"<span>{html.escape(lbl)} (<code>{html.escape(reason)}</code>)</span>"
                f"<span><strong>{cnt}</strong> ({cnt / total_veto_events:.0%})</span></div>"
                f'<div class="bar-track"><div class="bar-fill veto" style="width:{pct:.1f}%"></div></div>'
                "</div>"
            )
    else:
        veto_bars.append(
            '<div style="color:var(--muted);font-size:12px">No overfit or prosecutor vetoes triggered in this run.</div>'
        )

    chain_cls = "pass" if snapshot.chain_verified else "fail"
    chain_lbl = "SHA-256 VERIFIED" if snapshot.chain_verified else "CHAIN FAILED"
    pf_sharpe = snapshot.allocation_metrics.get("portfolio_sharpe", 0.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AlphaEvo Strategy Evolution Observatory &amp; Overfit Graveyard — {market_lbl}</title>
<style>{_BASE_CSS}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="brand">
      <h1>🧬 AlphaEvo Strategy Evolution Observatory &amp; Overfit Graveyard</h1>
      <p>AlphaEvo Live Self-Evolving Strategy Arena (Not Stock Picking) · Market: <strong>{market_lbl}</strong> · Source: <code>{data_src}</code> · Window: {dr_text}</p>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <span class="badge {chain_cls}">{chain_lbl}</span>
      <span class="badge">Backtests Veto · Shadow Ledgers Promote</span>
    </div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">Strategies Evolving</div><div class="value">{snapshot.total_strategies}</div></div>
    <div class="kpi"><div class="label">L0 Validated Promotions</div><div class="value" style="color:var(--green)">{snapshot.total_promotions}</div></div>
    <div class="kpi"><div class="label">Overfit Graveyard Kills</div><div class="value" style="color:var(--red)">{snapshot.total_vetoes}</div></div>
    <div class="kpi"><div class="label">Council Deliberations</div><div class="value" style="color:var(--cyan)">{snapshot.total_deliberations}</div></div>
    <div class="kpi"><div class="label">Causal Portfolio Sharpe</div><div class="value" style="color:var(--amber)">{pf_sharpe:.2f}</div></div>
  </div>

  <div class="card">
    <h2>Strategy Evolution Leaderboard <span style="color:var(--muted);font-size:12px;font-weight:400">(Click a row to inspect lineage tree, council debate, K-line trade replay &amp; YAML diff)</span></h2>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr><th>Strategy / Champion</th><th>Category</th><th>Score (Base → Champ)</th><th>Win%</th><th>AvgRet</th><th>Sharpe</th><th>MaxDD</th><th>Null Gate</th><th>Shadow</th></tr>
        </thead>
        <tbody>{"".join(rows_html)}</tbody>
      </table>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Evolution Lineage Tree (v1 → Champion)</h2>
      <div id="detail-tree">{first_tree}</div>
    </div>
    <div class="card">
      <h2 id="detail-title">Equity Curve vs Equal-Weight Benchmark</h2>
      <div id="detail-chart">{first_svg}</div>
      <div id="detail-verdicts" style="margin-top:12px;display:flex;flex-wrap:wrap;gap:6px"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Adversarial Council Debate (Proposer vs. Prosecutor)</h2>
      <div id="detail-deliberations" class="node-list"></div>
    </div>
    <div class="card">
      <h2>RSI Keep / Revert / Veto Ratchet</h2>
      <div id="detail-nodes" class="node-list"></div>
    </div>
  </div>

  <div class="card">
    <h2>Trade K-Line Replay Inspector <span style="color:var(--muted);font-size:12px;font-weight:400">(Executed champion backtest entries, exits &amp; holding windows)</span></h2>
    <div class="grid-2">
      <div id="detail-replay-list" class="node-list" style="max-height:240px"></div>
      <div id="detail-kline"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Overfit Graveyard &amp; Risk Budget</h2>
      {"".join(veto_bars)}
      <div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border)">
        <div style="font-size:12px;font-weight:600;margin-bottom:8px">Causal Regime-Adaptive Allocation</div>
        {"".join(alloc_bars)}
      </div>
      <div style="margin-top:12px;font-size:12px;color:var(--muted)">
        Alpha Factory Governance: audited={snapshot.factor_governance.get("total_audited", 0)},
        healthy={snapshot.factor_governance.get("healthy", 0)},
        decayed={snapshot.factor_governance.get("decayed", 0)},
        pit_unsafe={snapshot.factor_governance.get("pit_unsafe", 0)}
      </div>
    </div>
    <div class="card">
      <h2>Champion YAML &amp; Colorized Diff</h2>
      <pre id="detail-diff" class="code"></pre>
    </div>
  </div>

  <footer>
    Generated at {html.escape(snapshot.generated_at.isoformat())} · AlphaEvo Open-Source Strategy Self-Evolution Observatory · Research &amp; educational software only, not investment advice.
  </footer>
</div>

<script>
const ENTRIES = {payload_json};
let CURRENT_ENTRY = null;
function esc(s) {{
  return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}
function renderColoredDiff(lines, fallbackYaml) {{
  if (!lines || !lines.length) return esc(fallbackYaml || "No changes recorded.");
  return lines.map((line) => {{
    if (line.startsWith("+") && !line.startsWith("+++")) return `<span class="diff-add">${{esc(line)}}</span>`;
    if (line.startsWith("-") && !line.startsWith("---")) return `<span class="diff-del">${{esc(line)}}</span>`;
    if (line.startsWith("@@") || line.startsWith("+++") || line.startsWith("---")) return `<span class="diff-hdr">${{esc(line)}}</span>`;
    return esc(line);
  }}).join("\\n");
}}
function selectReplay(rIdx) {{
  const replays = (CURRENT_ENTRY && CURRENT_ENTRY.trade_replays) || [];
  const rep = replays[rIdx];
  document.querySelectorAll(".trade-pill").forEach((btn, idx) => {{
    btn.classList.toggle("active", idx === rIdx);
  }});
  document.getElementById("detail-kline").innerHTML = rep && rep.kline_svg
    ? rep.kline_svg
    : '<div style="color:var(--muted);font-size:13px;padding:12px">No executed trades available for candlestick replay.</div>';
}}
function selectEntry(idx) {{
  const e = ENTRIES[idx];
  if (!e) return;
  CURRENT_ENTRY = e;
  document.querySelectorAll("tr.clickable").forEach((tr) => {{
    tr.classList.toggle("active", Number(tr.dataset.idx) === idx);
  }});
  document.getElementById("detail-tree").innerHTML = e.tree_svg || "";
  document.getElementById("detail-title").textContent = `${{e.champion_id}} — Daily MTM Equity vs Benchmark`;
  document.getElementById("detail-chart").innerHTML = e.equity_svg || "";
  const verdictsEl = document.getElementById("detail-verdicts");
  verdictsEl.innerHTML = (e.committee_verdicts || []).map((v) => {{
    const cls = v.status === "pass" ? "pass" : (v.status === "fail" ? "fail" : "watch");
    return `<span class="badge ${{cls}}" title="${{esc(v.summary)}}">${{esc(v.analyst)}}: ${{esc(v.status).toUpperCase()}}</span>`;
  }}).join("");

  const delibEl = document.getElementById("detail-deliberations");
  const delibs = e.deliberations || [];
  delibEl.innerHTML = delibs.length ? delibs.map((d) => {{
    const cls = d.l0_outcome === "keep" ? "pass" : (d.l0_outcome === "stepping_stone" ? "watch" : "fail");
    const objs = (d.prosecutor_objections || []).length ? d.prosecutor_objections.map(esc).join("; ") : "Approved by Prosecutor";
    const vetoes = (d.veto_reasons || []).length ? ` · Gates failed: ${{d.veto_reasons.map(esc).join(", ")}}` : "";
    return `<div class="node">
      <div class="node-top">
        <span>Round ${{d.round_num}} · ${{esc(d.candidate_strategy_id || d.parent_strategy_id)}}</span>
        <span class="badge ${{cls}}">${{esc(d.l0_outcome).toUpperCase()}}</span>
      </div>
      <div class="debate-vs">
        <div class="debate-side"><strong>Proposer (${{esc(d.proposer_family)}})</strong><br>${{esc(d.rationale)}}<br><span style="color:var(--cyan)">${{(d.proposed_changes || []).map(esc).join(" | ")}}</span></div>
        <div class="debate-side"><strong>Prosecutor (${{esc(d.prosecutor_family)}})</strong><br>${{objs}}<br><span style="color:var(--muted)">Signals ${{d.signals ?? "-"}} · DSR ${{d.dsr_probability != null ? (d.dsr_probability*100).toFixed(0)+"%" : "-"}}${{vetoes}}</span></div>
      </div>
    </div>`;
  }}).join("") : '<div style="color:var(--muted);font-size:12px">No adversarial deliberations recorded.</div>';

  const nodesEl = document.getElementById("detail-nodes");
  nodesEl.innerHTML = (e.evolution_nodes || []).map((n) => {{
    const cls = n.status === "keep" ? "pass" : ((n.status === "vetoed" || n.status === "revert") ? "fail" : "watch");
    const vr = (n.veto_reasons || []).length ? ` · Failed: ${{n.veto_reasons.map(esc).join(", ")}}` : "";
    return `<div class="node">
      <div class="node-top">
        <span>Round ${{n.round_num}} · ${{esc(n.strategy_id)}}</span>
        <span class="badge ${{cls}}">${{esc(n.status).toUpperCase()}}</span>
      </div>
      <div class="node-meta">Score ${{Number(n.score * 100).toFixed(1)}}% · Signals ${{n.signals}} · Win ${{Number(n.win_rate * 100).toFixed(1)}}% · DSR ${{Number(n.dsr_prob * 100).toFixed(1)}}% · Null ${{n.null_passed ? "PASS" : "FAIL"}}${{vr}}</div>
      <div class="node-meta">${{esc(n.description)}}</div>
    </div>`;
  }}).join("");

  const repListEl = document.getElementById("detail-replay-list");
  const replays = e.trade_replays || [];
  if (replays.length) {{
    repListEl.innerHTML = replays.map((tr, rIdx) => {{
      const col = tr.return_pct >= 0 ? "var(--green)" : "var(--red)";
      return `<button type="button" class="trade-pill${{rIdx === 0 ? " active" : ""}}" onclick="selectReplay(${{rIdx}})">
        <span><strong>${{esc(tr.symbol)}}</strong> · ${{esc(tr.signal_date)}} → ${{esc(tr.exit_date || "-")}} (${{esc(tr.exit_reason)}})</span>
        <span style="color:${{col}};font-weight:700">${{(Number(tr.return_pct)*100).toFixed(2)}}%</span>
      </button>`;
    }}).join("");
    selectReplay(0);
  }} else {{
    repListEl.innerHTML = '<div style="color:var(--muted);font-size:12px">No executed trades for this strategy.</div>';
    selectReplay(-1);
  }}

  document.getElementById("detail-diff").innerHTML = renderColoredDiff(e.yaml_diff, e.champion_yaml);
}}
if (ENTRIES.length > 0) selectEntry(0);
</script>
</body>
</html>
"""


def render_run_html(
    *,
    run_id: str,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    strategy_yaml: str,
    trades: list[dict[str, Any]],
    equity_points: Sequence[EquityPoint] | None = None,
    trade_replays: Sequence[TradeReplayView | dict[str, Any]] | None = None,
) -> str:
    """Render a standalone single-file interactive HTML report for one run artifact."""
    if equity_points is None:
        eq = 1.0
        peak = 1.0
        computed: list[EquityPoint] = []
        for tr in trades:
            ret = float(tr.get("return_pct", 0.0) or 0.0)
            d = str(tr.get("exit_date") or tr.get("signal_date") or "")
            if not d:
                continue
            eq *= max(0.01, 1.0 + ret)
            peak = max(peak, eq)
            computed.append(
                EquityPoint(
                    date=d,
                    strategy_equity=round(eq, 4),
                    benchmark_equity=1.0,
                    drawdown=round((peak - eq) / peak if peak > 0 else 0.0, 4),
                )
            )
        equity_points = computed

    ov = evaluation.get("overall", {})
    strat_info = manifest.get("strategy", {})
    protocol = str(manifest.get("effective_protocol", "full_research_v1"))
    proto_cls = "pass" if protocol == "official_v1" else "watch"
    svg_html = render_equity_svg(equity_points)

    trade_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(t.get('symbol', '')))}</td>"
        f"<td>{html.escape(str(t.get('signal_date', '')))}</td>"
        f"<td>{html.escape(str(t.get('exit_date', '')))}</td>"
        f"<td>{float(t.get('entry_price', 0.0)):.2f}</td>"
        f"<td>{float(t.get('exit_price', 0.0) or 0.0):.2f}</td>"
        f"<td>{float(t.get('return_pct', 0.0)):+.2%}</td>"
        f"<td>{html.escape(str(t.get('exit_reason', '')))}</td>"
        "</tr>"
        for t in trades[:100]
    )

    replay_payload: list[dict[str, Any]] = []
    for item in trade_replays or []:
        rep_d: dict[str, Any] = (
            item.model_dump(mode="json") if isinstance(item, TradeReplayView) else dict(item)
        )
        rep_d["kline_svg"] = render_trade_kline_svg(rep_d)
        replay_payload.append(rep_d)

    replay_section = ""
    if replay_payload:
        buttons_html = "".join(
            f'<button type="button" class="trade-pill{" active" if idx == 0 else ""}" '
            f'onclick="selectRunReplay({idx})">'
            f"<span><strong>{html.escape(str(r.get('symbol', '')))}</strong> · "
            f"{html.escape(str(r.get('signal_date', '')))} → {html.escape(str(r.get('exit_date', '') or '-'))} "
            f"({html.escape(str(r.get('exit_reason', '')))})</span>"
            f'<span style="color:{"var(--green)" if float(r.get("return_pct", 0.0)) >= 0 else "var(--red)"};font-weight:700">'
            f"{float(r.get('return_pct', 0.0)):+.2%}</span></button>"
            for idx, r in enumerate(replay_payload)
        )
        replay_json = _safe_json_for_script(replay_payload)
        replay_section = f"""
  <div class="card">
    <h2>Trade K-Line Replay Inspector</h2>
    <div class="grid-2">
      <div class="node-list" style="max-height:240px">{buttons_html}</div>
      <div id="run-kline-box">{replay_payload[0]["kline_svg"]}</div>
    </div>
  </div>
  <script>
  const RUN_REPLAYS = {replay_json};
  function selectRunReplay(idx) {{
    document.querySelectorAll(".trade-pill").forEach((b, i) => b.classList.toggle("active", i === idx));
    const item = RUN_REPLAYS[idx];
    if (item && item.kline_svg) document.getElementById("run-kline-box").innerHTML = item.kline_svg;
  }}
  </script>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AlphaEvo Run Report — {html.escape(run_id)}</title>
<style>{_BASE_CSS}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="brand">
      <h1>🧬 AlphaEvo Run Report — {html.escape(str(strat_info.get("id", run_id)))}</h1>
      <p>Run ID: <code>{html.escape(run_id)}</code> · Metrics Basis: <code>{html.escape(str(ov.get("metrics_basis", "daily_mtm")))}</code></p>
    </div>
    <span class="badge {proto_cls}">{html.escape(protocol)}</span>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">Confidence Score</div><div class="value" style="color:var(--green)">{float(evaluation.get("confidence_score", 0.0)):.1%}</div></div>
    <div class="kpi"><div class="label">Win Rate</div><div class="value">{float(ov.get("win_rate", 0.0)):.1%}</div></div>
    <div class="kpi"><div class="label">Avg Return</div><div class="value">{float(ov.get("avg_return", 0.0)):+.2%}</div></div>
    <div class="kpi"><div class="label">Sharpe Ratio</div><div class="value" style="color:var(--cyan)">{float(ov.get("sharpe_ratio", 0.0)):.2f}</div></div>
    <div class="kpi"><div class="label">Max Drawdown</div><div class="value" style="color:var(--amber)">{float(ov.get("max_drawdown", 0.0)):.1%}</div></div>
    <div class="kpi"><div class="label">Signals</div><div class="value">{int(ov.get("signal_count", 0))}</div></div>
  </div>

  <div class="card">
    <h2>Equity Curve</h2>
    {svg_html}
  </div>

  {replay_section}

  <div class="grid-2">
    <div class="card">
      <h2>Strategy DSL Snapshot</h2>
      <pre class="code">{html.escape(strategy_yaml)}</pre>
    </div>
    <div class="card">
      <h2>Executed Trades (showing up to 100 of {len(trades)})</h2>
      <div style="overflow:auto;max-height:320px">
        <table>
          <thead><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry Px</th><th>Exit Px</th><th>Return</th><th>Reason</th></tr></thead>
          <tbody>{trade_rows or '<tr><td colspan="7" style="color:var(--muted)">No executed trades recorded.</td></tr>'}</tbody>
        </table>
      </div>
    </div>
  </div>

  <footer>AlphaEvo Open-Source Strategy Evolution Engine · Research &amp; educational output only, not investment advice.</footer>
</div>
</body>
</html>
"""


__all__ = [
    "render_arena_html",
    "render_equity_svg",
    "render_evolution_tree_svg",
    "render_run_html",
    "render_trade_kline_svg",
]
