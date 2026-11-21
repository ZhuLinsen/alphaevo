"""Self-contained zero-CDN browser UI for the AlphaEvo Web Research Studio."""

from __future__ import annotations


def render_web_studio_html() -> str:
    """Return single-file interactive SPA HTML served by ``alphaevo web``."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AlphaEvo Research Studio — Self-Evolving Stock Strategy Workbench</title>
<style>
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
.topbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 14px 24px;
  display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
}
.brand { font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
.tabs { display: flex; gap: 8px; flex-wrap: wrap; }
.tab {
  background: transparent; border: 1px solid var(--border); color: var(--muted);
  padding: 7px 14px; border-radius: 999px; cursor: pointer; font-size: 13px; font-weight: 600;
}
.tab.active { background: var(--surface-2); color: var(--cyan); border-color: rgba(56,189,248,0.5); }
.wrap { max-width: 1260px; margin: 0 auto; padding: 20px; }
.panel { display: none; }
.panel.active { display: block; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 18px; margin-bottom: 16px;
}
.card h2 { margin: 0 0 12px; font-size: 16px; }
label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
input, select, textarea {
  width: 100%; background: #090d14; color: var(--text); border: 1px solid var(--border);
  border-radius: 10px; padding: 9px 12px; font-size: 13px;
}
textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; min-height: 220px; resize: vertical; }
.row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 10px; }
button.btn {
  background: linear-gradient(135deg, #0ea5e9, #10b981); color: #041016;
  border: 0; border-radius: 10px; padding: 9px 16px; font-weight: 700; font-size: 13px;
  cursor: pointer;
}
button.btn.secondary {
  background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
}
button.btn:disabled { opacity: 0.55; cursor: wait; }
.chip {
  background: var(--surface-2); border: 1px solid var(--border); color: var(--cyan);
  border-radius: 999px; padding: 4px 10px; font-size: 11px; cursor: pointer;
}
.kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 12px 0;
}
.kpi { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.kpi .l { font-size: 11px; color: var(--muted); }
.kpi .v { font-size: 19px; font-weight: 700; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-size: 12px; }
.badge {
  display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600;
  border: 1px solid var(--border); background: var(--surface-2); color: var(--cyan);
}
.badge.pass { color: var(--green); border-color: rgba(16,185,129,0.45); }
.badge.fail { color: var(--red); border-color: rgba(239,68,68,0.45); }
.badge.watch { color: var(--amber); border-color: rgba(245,158,11,0.45); }
pre {
  background: #090d14; border: 1px solid var(--border); border-radius: 10px;
  padding: 12px; overflow: auto; font-size: 12px; color: #cbd5e1; max-height: 280px; margin: 0;
}
.diff-add { color: #34d399; }
.diff-del { color: #f87171; }
.diff-hdr { color: #38bdf8; font-weight: 600; }
.chart-svg { width: 100%; height: auto; display: block; border-radius: 12px; }
.status-msg { font-size: 12px; color: var(--muted); margin-left: 8px; }
.node-list { display: flex; flex-direction: column; gap: 8px; max-height: 300px; overflow-y: auto; }
.node {
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
}
.node-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; }
.node-meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
.debate-vs { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 8px; margin-top: 8px; }
.debate-side { background: #0e1523; border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; }
.trade-pill {
  width: 100%; text-align: left; background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 9px; padding: 8px 10px; color: var(--text); cursor: pointer; font-size: 12px;
  display: flex; justify-content: space-between; align-items: center;
}
.trade-pill:hover, .trade-pill.active { border-color: var(--cyan); background: #1d2a42; }
.bar-track { background: #0f172a; border-radius: 999px; height: 9px; overflow: hidden; margin-top: 4px; }
.bar-fill.veto { background: linear-gradient(90deg, var(--amber), var(--red)); height: 100%; }
.live-feed {
  margin-top: 10px; background: #090d14; border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
  max-height: 140px; overflow-y: auto; color: #94a3b8;
}
details.council-cfg {
  margin-top: 10px; background: #0e1523; border: 1px solid var(--border); border-radius: 10px; padding: 8px 12px;
}
details.council-cfg summary { cursor: pointer; font-size: 12px; font-weight: 600; color: var(--cyan); }
footer { text-align: center; color: var(--muted); font-size: 12px; padding: 16px 0 32px; }
</style>
</head>
<body>
<div class="topbar">
  <div class="brand">🧬 AlphaEvo Research Studio <span class="badge">Strategy Evolution Lab + L0 Truth</span></div>
  <div class="tabs">
    <button class="tab active" onclick="showTab('lab', this)">Strategy &amp; RSI Lab</button>
    <button class="tab" onclick="showTab('runs', this); loadRuns()">Leaderboard &amp; Runs</button>
    <button class="tab" onclick="showTab('shadow', this); loadShadow()">Forward Shadow Ledger</button>
    <button class="tab" onclick="showTab('gov', this); loadFactors()">Factors &amp; Risk Budget</button>
  </div>
</div>

<div class="wrap">
  <!-- Tab 1: Strategy & RSI Lab -->
  <div id="tab-lab" class="panel active">
    <div class="card">
      <h2>1. Draft Strategy from Plain Language (Zero LLM Required)</h2>
      <div class="row" style="margin-top:0">
        <input id="idea-input" placeholder="e.g. RSI oversold below 30 above MA20 with 4% stop loss and 10d max hold" />
        <select id="idea-market" style="width:140px">
          <option value="us">US Market</option>
          <option value="a_share">A-Share</option>
          <option value="hk">HK</option>
        </select>
        <button class="btn secondary" onclick="draftIdea()">Draft YAML</button>
      </div>
      <div class="row">
        <span style="font-size:12px;color:var(--muted)">Examples:</span>
        <span class="chip" onclick="useIdea('RSI oversold below 32 above MA60, 1.5x volume rebound, 5% stop loss')">US RSI Reversal</span>
        <span class="chip" onclick="useIdea('20日突破新高且波动率收缩，跌破10日均线止盈，6%止损')">20日突破收缩</span>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>2. Select or Edit Strategy DSL (YAML)</h2>
        <div class="row" style="margin-top:0;margin-bottom:10px">
          <select id="strategy-select" onchange="loadSelectedYaml()" style="flex:1"></select>
          <input id="rsi-rounds" type="number" min="1" max="5" value="2" style="width:90px" title="RSI Rounds" />
        </div>
        <textarea id="yaml-editor" spellcheck="false"></textarea>
        <details class="council-cfg">
          <summary>Adversarial Council (Optional Session LLM — Never Saved to Disk)</summary>
          <div class="grid" style="margin-top:10px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px">
            <div>
              <label for="proposer-model">Proposer Model</label>
              <input id="proposer-model" value="heuristic" placeholder="heuristic or gemini/gemini-3.8-flash" />
            </div>
            <div>
              <label for="prosecutor-model">Prosecutor Model</label>
              <input id="prosecutor-model" value="heuristic" placeholder="heuristic or anthropic/claude-sonnet-5.5" />
            </div>
            <div>
              <label for="session-api-key">Session API Key (Memory Only)</label>
              <input id="session-api-key" type="password" autocomplete="off" placeholder="Optional API key for LiteLLM calls" />
            </div>
          </div>
        </details>
        <div class="row">
          <button class="btn" id="btn-run" onclick="runBacktest()">Run EvaluationKernel</button>
          <button class="btn" id="btn-rsi" onclick="runRSI()">Stream Stock RSI Evolution</button>
          <span id="lab-status" class="status-msg"></span>
        </div>
        <div id="live-feed" class="live-feed">Live SSE stream idle. Click "Stream Stock RSI Evolution" to watch round-by-round ratchets &amp; vetoes.</div>
      </div>

      <div class="card">
        <h2>3. Evaluation, Lineage &amp; Equity Curve</h2>
        <div id="lab-kpis" class="kpis"></div>
        <div id="lab-tree" style="margin-top:10px"></div>
        <div id="lab-chart" style="margin-top:10px"></div>
        <div id="lab-verdicts" style="margin-top:10px;display:flex;flex-wrap:wrap;gap:6px"></div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Adversarial Council Debate (Proposer vs. Prosecutor)</h2>
        <div id="council-debate-box" class="node-list">
          <div style="color:var(--muted);font-size:13px">Deliberations will appear here live during Stock RSI Evolution.</div>
        </div>
      </div>
      <div class="card">
        <h2>Overfit Graveyard &amp; RSI Ratchet Ledger</h2>
        <div id="lab-veto-bars" style="margin-bottom:12px"></div>
        <div id="rsi-rounds-box" style="overflow-x:auto">
          <div style="color:var(--muted);font-size:13px">Run Stock RSI Evolution to inspect candidate rounds, DSR probabilities, and Null-Control gates.</div>
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <h2 style="margin:0">Trade K-Line Replay Inspector</h2>
          <span style="font-size:12px;color:var(--muted)">Executed backtest trades</span>
        </div>
        <div id="trade-replay-box">
          <div style="color:var(--muted);font-size:13px">Run EvaluationKernel or Stock RSI Evolution to inspect entry/exit candlesticks.</div>
        </div>
      </div>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <h2 style="margin:0">Champion YAML &amp; Diff</h2>
          <button class="btn secondary" id="btn-load-champ" onclick="loadChampionYaml()" disabled>Load Champion into Editor</button>
        </div>
        <pre id="lab-diff">No evolution diff yet.</pre>
      </div>
    </div>
  </div>

  <!-- Tab 2: Leaderboard & Runs -->
  <div id="tab-runs" class="panel">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <h2 style="margin:0">Scoped Leaderboard &amp; Immutable Runs</h2>
        <div class="row" style="margin-top:0">
          <select id="lb-scope" onchange="loadRuns()" style="width:150px">
            <option value="local">local scope</option>
            <option value="official">official scope</option>
          </select>
          <button class="btn secondary" onclick="loadRuns()">Refresh</button>
        </div>
      </div>
      <div id="runs-table" style="overflow-x:auto">Loading...</div>
    </div>
  </div>

  <!-- Tab 3: Shadow Ledger -->
  <div id="tab-shadow" class="panel">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:10px">
        <h2 style="margin:0">Forward Shadow Ledger <span id="chain-badge" class="badge">Checking...</span></h2>
        <div class="row" style="margin-top:0">
          <select id="shadow-strat" style="width:220px"></select>
          <button class="btn" onclick="simulateShadow()">Simulate Rolling Bitemporal Run</button>
        </div>
      </div>
      <div id="shadow-table" style="overflow-x:auto">Loading...</div>
      <pre id="shadow-sim-out" style="margin-top:12px;display:none"></pre>
    </div>
  </div>

  <!-- Tab 4: Factor Governance & Portfolio Risk Budget -->
  <div id="tab-gov" class="panel">
    <div class="grid">
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <h2 style="margin:0">Alpha Factory Factor Governance</h2>
          <button class="btn secondary" onclick="auditFactors(false)">Run OOS + PIT Audit</button>
        </div>
        <div id="factors-box" style="overflow-x:auto">Loading...</div>
      </div>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <h2 style="margin:0">Causal Regime-Adaptive Risk Budget</h2>
          <button class="btn" onclick="allocatePortfolio()">Allocate Core Strategies</button>
        </div>
        <div id="alloc-box" style="color:var(--muted);font-size:13px">Click "Allocate Core Strategies" to run causal inverse-vol, regime haircut &amp; correlation penalty allocation.</div>
      </div>
    </div>
  </div>
</div>

<footer>AlphaEvo Open-Source Strategy Self-Evolution Studio · Research &amp; educational workspace only — not investment advice.</footer>

<script>
let LAST_CHAMPION_YAML = "";
let LAST_REPLAYS = [];

async function api(url, method="GET", body=null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== null) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
function esc(s) {
  return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function showTab(name, el) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  if (el) el.classList.add("active");
  document.getElementById(`tab-${name}`).classList.add("active");
}
function useIdea(txt) {
  document.getElementById("idea-input").value = txt;
}
function appendFeed(line) {
  const feed = document.getElementById("live-feed");
  const div = document.createElement("div");
  div.textContent = line;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}
function renderColoredDiff(lines, fallbackYaml) {
  if (!lines || !lines.length) return esc(fallbackYaml || "No changes recorded.");
  return lines.map((line) => {
    if (line.startsWith("+") && !line.startsWith("+++")) return `<span class="diff-add">${esc(line)}</span>`;
    if (line.startsWith("-") && !line.startsWith("---")) return `<span class="diff-del">${esc(line)}</span>`;
    if (line.startsWith("@@") || line.startsWith("+++") || line.startsWith("---")) return `<span class="diff-hdr">${esc(line)}</span>`;
    return esc(line);
  }).join("\\n");
}
function renderVetoBars(vetoMap) {
  const box = document.getElementById("lab-veto-bars");
  const entries = Object.entries(vetoMap || {}).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((acc, [, v]) => acc + v, 0);
  if (!entries.length || total <= 0) {
    box.innerHTML = "";
    return;
  }
  box.innerHTML = entries.map(([k, v]) => {
    const pct = Math.max(5, Math.min(100, (v / total) * 100));
    return `<div style="margin-bottom:6px"><div style="display:flex;justify-content:space-between;font-size:12px"><span><code>${esc(k)}</code></span><span><strong>${v}</strong></span></div><div class="bar-track"><div class="bar-fill veto" style="width:${pct.toFixed(1)}%"></div></div></div>`;
  }).join("");
}
function selectStudioReplay(idx) {
  const rep = LAST_REPLAYS[idx];
  document.querySelectorAll("#trade-replay-box .trade-pill").forEach((b, i) => b.classList.toggle("active", i === idx));
  const chartBox = document.getElementById("studio-kline-slot");
  if (chartBox && rep) chartBox.innerHTML = rep.kline_svg || "";
}
function renderTradeReplays(replays) {
  LAST_REPLAYS = replays || [];
  const box = document.getElementById("trade-replay-box");
  if (!LAST_REPLAYS.length) {
    box.innerHTML = `<div style="color:var(--muted);font-size:13px">No executed trades recorded for K-line replay.</div>`;
    return;
  }
  const pills = LAST_REPLAYS.map((tr, i) => {
    const col = tr.return_pct >= 0 ? "var(--green)" : "var(--red)";
    return `<button type="button" class="trade-pill${i === 0 ? " active" : ""}" onclick="selectStudioReplay(${i})"><span><strong>${esc(tr.symbol)}</strong> · ${esc(tr.signal_date)} → ${esc(tr.exit_date || "-")} (${esc(tr.exit_reason)})</span><span style="color:${col};font-weight:700">${(Number(tr.return_pct)*100).toFixed(2)}%</span></button>`;
  }).join("");
  box.innerHTML = `<div class="node-list" style="max-height:145px;margin-bottom:10px">${pills}</div><div id="studio-kline-slot">${LAST_REPLAYS[0].kline_svg || ""}</div>`;
}
function renderDebates(deliberations) {
  const box = document.getElementById("council-debate-box");
  const items = deliberations || [];
  if (!items.length) {
    box.innerHTML = `<div style="color:var(--muted);font-size:13px">No council deliberations recorded in this session.</div>`;
    return;
  }
  box.innerHTML = items.map((d) => {
    const cls = d.l0_outcome === "keep" ? "pass" : (d.l0_outcome === "stepping_stone" ? "watch" : "fail");
    const objs = (d.prosecutor_objections || []).length ? d.prosecutor_objections.map(esc).join("; ") : "Approved by Prosecutor";
    const vetoes = (d.veto_reasons || []).length ? ` · Failed: ${d.veto_reasons.map(esc).join(", ")}` : "";
    return `<div class="node"><div class="node-top"><span>Round ${d.round_num} · ${esc(d.candidate_strategy_id || d.parent_strategy_id)}</span><span class="badge ${cls}">${esc(d.l0_outcome).toUpperCase()}</span></div><div class="debate-vs"><div class="debate-side"><strong>Proposer (${esc(d.proposer_family)})</strong><br>${esc(d.rationale)}<br><span style="color:var(--cyan)">${(d.proposed_changes || []).map(esc).join(" | ")}</span></div><div class="debate-side"><strong>Prosecutor (${esc(d.prosecutor_family)})</strong><br>${objs}<br><span style="color:var(--muted)">Signals ${d.signals ?? "-"} · DSR ${d.dsr_probability != null ? (d.dsr_probability*100).toFixed(0)+"%" : "-"}${vetoes}</span></div></div></div>`;
  }).join("");
}
function loadChampionYaml() {
  if (!LAST_CHAMPION_YAML) return;
  document.getElementById("yaml-editor").value = LAST_CHAMPION_YAML;
  document.getElementById("lab-status").textContent = "Loaded champion YAML into editor.";
}
async function initStudio() {
  try {
    const data = await api("/api/strategies");
    const sel = document.getElementById("strategy-select");
    const shSel = document.getElementById("shadow-strat");
    sel.innerHTML = ""; shSel.innerHTML = "";
    (data.strategies || []).forEach((s) => {
      const opt1 = document.createElement("option");
      opt1.value = s.strategy_id; opt1.textContent = `${s.strategy_id} (${s.category})`;
      sel.appendChild(opt1);
      const opt2 = opt1.cloneNode(true);
      shSel.appendChild(opt2);
    });
    if ((data.strategies || []).length > 0) await loadSelectedYaml();
  } catch (e) {
    document.getElementById("lab-status").textContent = e.message;
  }
}
async function loadSelectedYaml() {
  const sid = document.getElementById("strategy-select").value;
  if (!sid) return;
  const d = await api(`/api/strategies/${encodeURIComponent(sid)}`);
  document.getElementById("yaml-editor").value = d.yaml || "";
}
async function draftIdea() {
  const idea = document.getElementById("idea-input").value.trim();
  if (!idea) return;
  const market = document.getElementById("idea-market").value;
  document.getElementById("lab-status").textContent = "Drafting DSL...";
  try {
    const res = await api("/api/strategies/draft", "POST", { idea, market, save: false });
    document.getElementById("yaml-editor").value = res.yaml || "";
    document.getElementById("lab-status").textContent = `Drafted ${res.strategy_id}`;
  } catch (e) {
    document.getElementById("lab-status").textContent = `Error: ${e.message}`;
  }
}
function renderKpis(items) {
  document.getElementById("lab-kpis").innerHTML = items.map(([l, v]) =>
    `<div class="kpi"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div></div>`
  ).join("");
}
async function runBacktest() {
  const yaml = document.getElementById("yaml-editor").value;
  const status = document.getElementById("lab-status");
  status.textContent = "Running EvaluationKernel...";
  try {
    const res = await api("/api/runs", "POST", { strategy_yaml: yaml });
    const ev = res.evaluation || {};
    renderKpis([
      ["Score", `${(ev.confidence_score * 100).toFixed(1)}%`],
      ["Win Rate", `${(ev.win_rate * 100).toFixed(1)}%`],
      ["Avg Return", `${(ev.avg_return * 100).toFixed(2)}%`],
      ["Max Drawdown", `${(ev.max_drawdown * 100).toFixed(1)}%`],
      ["Signals", ev.signal_count],
      ["Protocol", res.effective_protocol],
    ]);
    document.getElementById("lab-tree").innerHTML = "";
    document.getElementById("lab-chart").innerHTML = res.equity_svg || "";
    document.getElementById("lab-verdicts").innerHTML = (res.committee_verdicts || []).map(v => {
      const c = v.status === "pass" ? "pass" : (v.status === "fail" ? "fail" : "watch");
      return `<span class="badge ${c}" title="${esc(v.summary)}">${esc(v.analyst)}: ${esc(v.status).toUpperCase()}</span>`;
    }).join("");
    renderTradeReplays(res.trade_replays || []);
    status.textContent = `Saved run ${res.run_id}`;
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
}
function applyRsiResult(res) {
  renderKpis([
    ["Champion", res.champion_id],
    ["Champ Score", `${(res.champion_score * 100).toFixed(1)}%`],
    ["Promotions", res.promotions],
    ["Trials Recorded", res.total_trials_recorded],
    ["Kernel Intact", res.kernel_intact ? "YES" : "NO"],
  ]);
  document.getElementById("lab-tree").innerHTML = res.evolution_tree_svg || "";
  document.getElementById("lab-chart").innerHTML = res.equity_svg || "";
  document.getElementById("lab-verdicts").innerHTML = "";
  renderVetoBars(res.veto_breakdown || {});
  renderDebates(res.deliberations || []);
  renderTradeReplays(res.trade_replays || []);
  const rowsHtml = (res.rows || []).map(r => {
    const cls = r.status === "keep" ? "pass" : ((r.status === "vetoed" || r.status === "revert") ? "fail" : "watch");
    return `<tr><td>${r.round_num}</td><td>${esc(r.strategy_id)}</td><td><span class="badge ${cls}">${esc(r.status).toUpperCase()}</span></td><td>${r.signals}</td><td>${(r.win_rate*100).toFixed(1)}%</td><td>${(r.dsr_probability*100).toFixed(1)}%</td><td>${r.null_passed ? "PASS" : "FAIL"}</td><td>${esc(r.description)}</td></tr>`;
  }).join("");
  document.getElementById("rsi-rounds-box").innerHTML = `<table><thead><tr><th>Rd</th><th>ID</th><th>Status</th><th>Signals</th><th>Win%</th><th>DSR</th><th>Null</th><th>Change</th></tr></thead><tbody>${rowsHtml}</tbody></table>`;
  document.getElementById("lab-diff").innerHTML = renderColoredDiff(res.yaml_diff, res.champion_yaml);
  LAST_CHAMPION_YAML = res.champion_yaml || "";
  document.getElementById("btn-load-champ").disabled = !LAST_CHAMPION_YAML;
}
async function runRSI() {
  const yaml = document.getElementById("yaml-editor").value;
  const rounds = Number(document.getElementById("rsi-rounds").value || 2);
  const proposer_model = document.getElementById("proposer-model").value.trim() || null;
  const prosecutor_model = document.getElementById("prosecutor-model").value.trim() || null;
  const api_key = document.getElementById("session-api-key").value || null;
  const status = document.getElementById("lab-status");
  const btn = document.getElementById("btn-rsi");
  const feed = document.getElementById("live-feed");

  feed.innerHTML = "";
  appendFeed("→ Connecting to /api/rsi/stream...");
  status.textContent = "Streaming Stock RSI Evolution...";
  btn.disabled = true;

  try {
    const res = await fetch("/api/rsi/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        strategy_yaml: yaml,
        rounds,
        proposer_model,
        prosecutor_model,
        api_key,
      }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalRes = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\\n\\n");
      buffer = blocks.pop() || "";
      for (const blk of blocks) {
        let evName = "message";
        let dataStr = "";
        for (const line of blk.split("\\n")) {
          if (line.startsWith("event:")) evName = line.slice(6).trim();
          else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
        }
        if (!dataStr) continue;
        const payload = JSON.parse(dataStr);
        if (evName === "error") throw new Error(payload.error || "RSI stream failed");
        if (evName === "baseline" && payload.row) {
          appendFeed(`[R0] Baseline ${payload.row.strategy_id}: Fit ${(payload.row.fitness_score * 100).toFixed(1)}% · Signals ${payload.row.signals}`);
        } else if (evName === "council_deliberation" && payload.deliberation) {
          const d = payload.deliberation;
          appendFeed(`[R${d.round_num}] ${d.proposer_family} vs ${d.prosecutor_family} -> ${String(d.l0_outcome).toUpperCase()} (${(d.veto_reasons || []).join(", ") || "clear"})`);
        } else if (evName === "round_complete" && payload.row) {
          const r = payload.row;
          appendFeed(`[R${r.round_num}] Round Verdict: ${String(r.status).toUpperCase()} · ${r.strategy_id} · Fit ${(r.fitness_score * 100).toFixed(1)}%`);
        } else if (evName === "result") {
          finalRes = payload;
        }
      }
    }
    if (finalRes) {
      applyRsiResult(finalRes);
      status.textContent = `RSI complete (${finalRes.champion_id})`;
    } else {
      throw new Error("Stream ended without final result");
    }
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    appendFeed(`✖ Error: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}
async function loadRuns() {
  const scope = document.getElementById("lb-scope").value;
  const [lb, runs] = await Promise.all([api(`/api/leaderboard?scope=${scope}`), api("/api/runs")]);
  const rows = (lb.entries || []).length ? lb.entries : (runs.runs || []);
  document.getElementById("runs-table").innerHTML = rows.length ? `<table><thead><tr><th>Strategy</th><th>Protocol</th><th>Score</th><th>Win%</th><th>AvgRet</th><th>MaxDD</th><th>Signals</th><th>Dirty</th></tr></thead><tbody>` +
    rows.map(r => {
      const proto = r.effective_protocol ?? r.protocol ?? "full_research_v1";
      const sc = r.score ?? r.confidence_score ?? 0;
      const rid = r.run_id ?? r.source ?? "";
      return `<tr><td><code>${esc(r.strategy_id)}</code><br><span style="font-size:11px;color:var(--muted)">${esc(rid)}</span></td><td><span class="badge">${esc(proto)}</span></td><td>${(sc*100).toFixed(1)}%</td><td>${(r.win_rate*100).toFixed(1)}%</td><td>${(r.avg_return*100).toFixed(2)}%</td><td>${(r.max_drawdown*100).toFixed(1)}%</td><td>${r.signal_count}</td><td>${r.code_dirty ? "yes" : "no"}</td></tr>`;
    }).join("") +
    `</tbody></table>` : `<div style="color:var(--muted)">No runs indexed yet in this workspace. Run a strategy in Strategy Lab first!</div>`;
}
async function loadShadow() {
  const d = await api("/api/shadow");
  const badge = document.getElementById("chain-badge");
  badge.textContent = d.chain_verified ? "SHA-256 VERIFIED" : "CHAIN ERROR";
  badge.className = `badge ${d.chain_verified ? "pass" : "fail"}`;
  const cands = d.candidates || [];
  document.getElementById("shadow-table").innerHTML = cands.length ? `<table><thead><tr><th>Strategy</th><th>Stage</th><th>Mature</th><th>N_eff</th><th>Fwd Return</th><th>Decay</th><th>MaxDD</th></tr></thead><tbody>` +
    cands.map(c => `<tr><td>${esc(c.strategy_id)}</td><td><span class="badge">${esc(c.stage)}</span></td><td>${c.mature_signals}</td><td>${c.effective_signals}</td><td>${(c.forward_avg_post_cost_return*100).toFixed(2)}%</td><td>${c.decay_ratio ?? "n/a"}</td><td>${(c.max_forward_drawdown*100).toFixed(1)}%</td></tr>`).join("") +
    `</tbody></table>` : `<div style="color:var(--muted)">No preregistered candidates yet. Use "Simulate Rolling Bitemporal Run" above to test forward commitments.</div>`;
}
async function simulateShadow() {
  const sid = document.getElementById("shadow-strat").value;
  const pre = document.getElementById("shadow-sim-out");
  pre.style.display = "block"; pre.textContent = "Running chronological bitemporal shadow simulation...";
  const res = await api("/api/shadow/simulate", "POST", { strategy_id: sid, days: 35, horizon: 5 });
  pre.textContent = res.markdown || JSON.stringify(res, null, 2);
}
async function loadFactors() {
  const d = await api("/api/factors");
  const fs = d.factors || [];
  document.getElementById("factors-box").innerHTML = fs.length ? `<table><thead><tr><th>Name</th><th>IR</th><th>IC Mean</th><th>Status</th></tr></thead><tbody>` +
    fs.map(f => `<tr><td>${esc(f.name)}</td><td>${Number(f.ir).toFixed(2)}</td><td>${Number(f.ic_mean).toFixed(3)}</td><td>${esc(f.status)}</td></tr>`).join("") +
    `</tbody></table>` : `<div style="color:var(--muted)">No dynamic factors registered yet in FactorStore.</div>`;
}
async function auditFactors(autoRetire) {
  const res = await api("/api/factors/audit", "POST", { auto_retire: autoRetire });
  document.getElementById("factors-box").innerHTML = `<pre>${esc(res.markdown || "No factors audited.")}</pre>`;
}
async function allocatePortfolio() {
  const box = document.getElementById("alloc-box");
  box.textContent = "Computing strictly causal inverse-vol & correlation-penalized risk budget...";
  const sel = Array.from(document.getElementById("strategy-select").options).slice(0, 3).map(o => o.value);
  const res = await api("/api/portfolio/allocate", "POST", { strategy_ids: sel });
  box.innerHTML = `<pre>${esc(res.markdown || JSON.stringify(res, null, 2))}</pre>`;
}
initStudio();
</script>
</body>
</html>
"""


__all__ = ["render_web_studio_html"]
