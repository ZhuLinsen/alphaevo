"""Self-contained zero-CDN HTML/SVG renderer for Arena Dashboards & Single Run Reports."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Any

from alphaevo.arena.models import ArenaSnapshot, EquityPoint


def _safe_json_for_script(obj: Any) -> str:
    """Serialize *obj* safely for inline <script> embedding."""
    raw = json.dumps(obj, ensure_ascii=False, default=str)
    return raw.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render_equity_svg(
    points: Sequence[EquityPoint],
    *,
    width: int = 760,
    height: int = 220,
) -> str:
    """Render an inline SVG dual-line equity curve + drawdown ribbon."""
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
.container { max-width: 1220px; margin: 0 auto; padding: 24px 20px 48px; }
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
.node-list { display: flex; flex-direction: column; gap: 8px; }
.node {
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
}
.node-top { display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: 600; }
.node-meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
pre.code {
  margin: 0; background: #090d14; border: 1px solid var(--border); border-radius: 10px;
  padding: 12px; overflow-x: auto; font-size: 12px; color: #cbd5e1; max-height: 320px;
}
.bar-track { background: #0f172a; border-radius: 999px; height: 10px; overflow: hidden; margin-top: 4px; }
.bar-fill { background: linear-gradient(90deg, var(--cyan), var(--green)); height: 100%; }
footer { margin-top: 28px; color: var(--muted); font-size: 12px; text-align: center; }
"""


def render_arena_html(snapshot: ArenaSnapshot) -> str:
    """Render interactive standalone single-file HTML for the Live Strategy Arena."""
    entries_payload: list[dict[str, Any]] = []
    for entry in snapshot.entries:
        d = entry.model_dump(mode="json")
        d["equity_svg"] = render_equity_svg(entry.equity_curve)
        entries_payload.append(d)

    payload_json = _safe_json_for_script(entries_payload)
    first_svg = entries_payload[0]["equity_svg"] if entries_payload else render_equity_svg([])
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

    chain_cls = "pass" if snapshot.chain_verified else "fail"
    chain_lbl = "SHA-256 VERIFIED" if snapshot.chain_verified else "CHAIN FAILED"
    pf_sharpe = snapshot.allocation_metrics.get("portfolio_sharpe", 0.0)
    pf_ret = snapshot.allocation_metrics.get("portfolio_total_return", 0.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AlphaEvo Live Self-Evolving Arena — {market_lbl}</title>
<style>{_BASE_CSS}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="brand">
      <h1>🧬 AlphaEvo Live Self-Evolving Strategy Arena</h1>
      <p>Market: <strong>{market_lbl}</strong> · Source: <code>{data_src}</code> · Window: {dr_text} · Basket: {len(snapshot.symbols)} symbols</p>
    </div>
    <div>
      <span class="badge {chain_cls}">{chain_lbl}</span>
      <span class="badge">Backtests Veto · Shadow Ledgers Promote</span>
    </div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">Strategies Evolving</div><div class="value">{snapshot.total_strategies}</div></div>
    <div class="kpi"><div class="label">Validated Promotions</div><div class="value" style="color:var(--green)">{snapshot.total_promotions}</div></div>
    <div class="kpi"><div class="label">Overfit / Null Vetoes</div><div class="value" style="color:var(--amber)">{snapshot.total_vetoes}</div></div>
    <div class="kpi"><div class="label">Causal Portfolio Return</div><div class="value">{pf_ret:+.1%}</div></div>
    <div class="kpi"><div class="label">Causal Portfolio Sharpe</div><div class="value" style="color:var(--cyan)">{pf_sharpe:.2f}</div></div>
  </div>

  <div class="card">
    <h2>Strategy Evolution Leaderboard <span style="color:var(--muted);font-size:12px;font-weight:400">(Click a row to inspect curves, ratchet rounds & YAML diff)</span></h2>
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
      <h2 id="detail-title">Equity Curve vs Equal-Weight Benchmark</h2>
      <div id="detail-chart">{first_svg}</div>
      <div id="detail-verdicts" style="margin-top:12px;display:flex;flex-wrap:wrap;gap:6px"></div>
    </div>
    <div class="card">
      <h2>RSI Keep / Revert / Veto Ratchet</h2>
      <div id="detail-nodes" class="node-list"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Causal Regime-Adaptive Risk Budget</h2>
      {"".join(alloc_bars)}
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border);font-size:12px;color:var(--muted)">
        Alpha Factory Governance: audited={snapshot.factor_governance.get("total_audited", 0)},
        healthy={snapshot.factor_governance.get("healthy", 0)},
        decayed={snapshot.factor_governance.get("decayed", 0)},
        pit_unsafe={snapshot.factor_governance.get("pit_unsafe", 0)}
      </div>
    </div>
    <div class="card">
      <h2>Champion YAML & Unified Diff</h2>
      <pre id="detail-diff" class="code"></pre>
    </div>
  </div>

  <footer>
    Generated at {html.escape(snapshot.generated_at.isoformat())} · AlphaEvo Open-Source Research Engine · Simulated backtests and replay shadow runs are for research & educational purposes only, not investment advice.
  </footer>
</div>

<script>
const ENTRIES = {payload_json};
function esc(s) {{
  return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}
function selectEntry(idx) {{
  const e = ENTRIES[idx];
  if (!e) return;
  document.querySelectorAll("tr.clickable").forEach((tr) => {{
    tr.classList.toggle("active", Number(tr.dataset.idx) === idx);
  }});
  document.getElementById("detail-title").textContent = `${{e.champion_id}} — Daily MTM Equity vs Benchmark`;
  document.getElementById("detail-chart").innerHTML = e.equity_svg || "";
  const verdictsEl = document.getElementById("detail-verdicts");
  verdictsEl.innerHTML = (e.committee_verdicts || []).map((v) => {{
    const cls = v.status === "pass" ? "pass" : (v.status === "fail" ? "fail" : "watch");
    return `<span class="badge ${{cls}}" title="${{esc(v.summary)}}">${{esc(v.analyst)}}: ${{esc(v.status).toUpperCase()}}</span>`;
  }}).join("");
  const nodesEl = document.getElementById("detail-nodes");
  nodesEl.innerHTML = (e.evolution_nodes || []).map((n) => {{
    const cls = n.status === "keep" ? "pass" : (n.status === "vetoed" ? "fail" : "watch");
    return `<div class="node">
      <div class="node-top">
        <span>Round ${{n.round_num}} · ${{esc(n.strategy_id)}}</span>
        <span class="badge ${{cls}}">${{esc(n.status).toUpperCase()}}</span>
      </div>
      <div class="node-meta">Score ${{Number(n.score * 100).toFixed(1)}}% · Signals ${{n.signals}} · Win ${{Number(n.win_rate * 100).toFixed(1)}}% · DSR ${{Number(n.dsr_prob * 100).toFixed(1)}}% · Null ${{n.null_passed ? "PASS" : "FAIL"}}</div>
      <div class="node-meta">${{esc(n.description)}}</div>
    </div>`;
  }}).join("");
  const diffText = (e.yaml_diff && e.yaml_diff.length) ? e.yaml_diff.join("\\n") : (e.champion_yaml || "");
  document.getElementById("detail-diff").textContent = diffText;
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

  <footer>AlphaEvo Open-Source Research Engine · Research & educational output only, not investment advice.</footer>
</div>
</body>
</html>
"""


__all__ = ["render_arena_html", "render_equity_svg", "render_run_html"]
