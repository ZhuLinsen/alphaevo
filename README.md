<div align="center">

<img src="sources/alphaevo_terminal_v1_agent_evolution.svg" alt="AlphaEvo Logo" width="680">

# 🧬 AlphaEvo

**An Open-Source Self-Evolving Equity Strategy Research Studio, Live Arena & Anti-Delusion RSI Engine**

*Go from plain-language trading idea -> YAML DSL -> self-evolving AI loop -> falsification gates -> shareable HTML & bitemporal shadow verification*

[![GitHub stars](https://img.shields.io/github/stars/ZhuLinsen/alphaevo?style=social)](https://github.com/ZhuLinsen/alphaevo/stargazers)
[![CI](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![MCP Ready](https://img.shields.io/badge/MCP-STDIO%20Ready-10b981.svg)](SKILL.md)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[**Studio & Arena**](#-visual-research-studio--live-arena) · [**Why AlphaEvo**](#-why-alphaevo) · [**4 Ways to Start**](#-4-zero-friction-ways-to-start) · [**MCP & Skill**](#-3-connect-to-claude-code-openclaw-cursor-codex-or-antigravity-native-mcp--skillmd) · [**CLI**](#-cli-commands)

[English](README.md) | [中文](docs/README_CN.md)

</div>

---

## ✨ Why AlphaEvo

Most open-source AI finance projects stop at **one-shot LLM stock chatter**, **opaque Python code generation**, or **overfitted backtest curves**.

**AlphaEvo treats trading strategies as inspectable, falsifiable research assets** stored as human-readable **YAML DSLs**:

1. **Draft & Evolve Transparently**: Turn plain English/Chinese ideas into readable YAML rules (`triggers`, `guards`, `exit.triggers`), mutate one lever at a time, and inspect exact YAML diffs across generations.
2. **Enforce Anti-Delusion Falsification**: A zero-bypass `EvaluationKernel` couples **O(N) 28.9× vectorized daily MTM backtesting**, `TrialLedger` **Deflated Sharpe Ratio (DSR)**, cross-sectional **shuffled-return Null Controls**, **Locked Holdout Vault**, and **runtime Point-in-Time (PIT) leakage probes**.
3. **Separate Search from Promotion**: Historical backtests can *only veto*; promotion requires chronological commitments in a **SHA-256 hash-chained Forward Shadow Ledger** (`T+5` / `T+10` / `T+20`).
4. **Ship Ready-to-Use Interfaces**: Zero-dep local **Web Studio** (`alphaevo web`), self-contained single-file **Live Arena HTML dashboards** (`alphaevo arena run`), **Fork-and-Run daily GitHub Actions + 7-channel push alerts**, and a native **STDIO MCP server** (`alphaevo mcp` + `SKILL.md`).

<p align="center" id="visual-research-studio--live-arena">
  <img src="sources/alphaevo_studio_preview.svg" alt="AlphaEvo Web Research Studio & Live Self-Evolving Arena" width="820">
</p>

## ⚡ 4 Zero-Friction Ways to Start

### 1. Local Web Research Studio & Live SSE Evolution (Zero Extra Dependencies)
```bash
git clone https://github.com/ZhuLinsen/alphaevo.git && cd alphaevo
pip install -e .
alphaevo web
# Or run via Docker in one command:
# docker compose up --build
```
Open `http://127.0.0.1:8765` to draft YAML strategies from plain language, stream round-by-round Stock RSI evolution over SSE (`/api/rsi/stream`), inspect **Proposer vs. Prosecutor debate cards**, view **Evolution Lineage Trees** & **Overfit Graveyard** veto bars, replay executed trades on **OHLC K-line charts**, simulate the SHA-256 Shadow Ledger, audit factors, and allocate risk budgets.

### 2. Evolution Observatory, Overfit Graveyard & HTML Reports (30s, No Key Required)
```bash
alphaevo showcase                       # 30s bundled snapshot evolution (report.md + interactive K-line report.html)
alphaevo arena run --rounds 2           # Strategy Evolution Observatory -> reports/arena/index.html + digest.md
```

### 3. Connect to Claude Code, OpenClaw, Cursor, Codex, or Antigravity (Native MCP + `SKILL.md`)
```json
{
  "mcpServers": {
    "alphaevo": { "command": "alphaevo", "args": ["mcp"] }
  }
}
```
Exposes 9 research tools (`alphaevo_draft_strategy`, `alphaevo_evaluate_strategy`, `alphaevo_evolve_rsi`, `alphaevo_null_audit`, `alphaevo_shadow_simulate`, `alphaevo_audit_factors`, `alphaevo_allocate_portfolio`, etc.) plus `alphaevo://program` and `alphaevo://strategies` resources. See [SKILL.md](SKILL.md).

### 4. Fork & Deploy GitHub Pages Evolution Observatory + Push Alerts (Free Cloud via Actions)
1. Fork this repository, enable **GitHub Actions**, and optionally enable **GitHub Pages -> GitHub Actions** (`Settings -> Pages`) to publish `https://<owner>.github.io/<repo>/`.
2. Optionally add Secrets (`ALPHAEVO_FEISHU_WEBHOOK`, `ALPHAEVO_WECOM_WEBHOOK`, `ALPHAEVO_DINGTALK_WEBHOOK`, `ALPHAEVO_TELEGRAM_BOT_TOKEN` + `ALPHAEVO_TELEGRAM_CHAT_ID`, `ALPHAEVO_DISCORD_WEBHOOK`, `ALPHAEVO_SLACK_WEBHOOK`, `ALPHAEVO_WEBHOOK_URL`, `ALPHAEVO_API_KEY`).
3. [arena-daily.yml](.github/workflows/arena-daily.yml) automatically evolves strategies on weekdays, builds the **Evolution Observatory & Overfit Graveyard** single-file dashboard, deploys to GitHub Pages (when enabled), posts `digest.md` to Step Summary, and notifies your channels.

## 🧭 Why AlphaEvo vs. Alternatives

| Dimension | Traditional Backtesters (`Backtrader` / `VectorBT`) | Multi-Agent Trading (`TradingAgents` / `ai-hedge-fund`) | Code Auto-Research (`RD-Agent` / `autoresearch`) | **AlphaEvo** |
|-----------|----------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------|--------------|
| **Unit of Evolution** | Manual Python scripts | Ephemeral chat/agent votes | Arbitrary Python code patches | **Readable YAML DSL + L2 AST/PIT-safe factors** |
| **Multiple-Testing & Null Control** | DIY | None by default | Single holdout metric | **`EvaluationKernel` + `TrialLedger` DSR + Shuffled Null Worlds + Locked Holdout** |
| **Forward Verification** | Manual paper script | Replay / episodic calls | Offline split only | **SHA-256 Bitemporal Shadow Ledger (`T+5/10/20`, `N_eff`, `HALTED` trips)** |
| **Out-of-Box UI & Agent Hub** | Library only | Terminal / per-project scripts | Server / heavy setup | **Stdlib `alphaevo web`, single-file HTML Arena/Run reports, `alphaevo mcp` + `SKILL.md`** |
| **Portfolio & Factor Governance** | External libraries | Basic sizing | Manual factor cleanup | **Causal `RegimeAdaptiveAllocator` + OOS/PIT `FactorGovernanceEngine`** |

---

### Real-World Benchmark Snapshot

To keep self-evolution honest on noisy equity data, AlphaEvo separates **fast offline hypothesis search** from **slow forward promotion**: historical backtests can only veto weak candidates, while a bitemporal forward shadow ledger tracks out-of-sample commitments (`T+5` / `T+10` / `T+20`) before any promotion.

### Reproducible no-key showcase

Bundled frozen yfinance snapshot, US tech basket, 2025-02-11 to 2026-04-10 (evaluated under chronological equal-weight **daily MTM**, gap-aware open SL/TP fills, and **measured parameter sensitivity**):

| Strategy | Signals | Win Rate | Avg Return | Max DD | Score |
|----------|---------|----------|------------|--------|-------|
| `rsi_reversion_v1` baseline | 0 | 0.0% | 0.00% | 0.0% | 6.6% |
| `rsi_reversion_v7` champion | 37 | 48.6% | 2.94% | 26.7% | 65.9% |

What happened: the deterministic research committee flagged an over-confirmed entry stack, unlocked enough signals (`22.1%`), widened the stop (Sharpe/expectancy bridge), extended holding, raised the payoff target, switched support context to MA60, then tightened volume confirmation. Each change was accepted only after retest. The champion keeps train-val and val-test gaps at 7.1% / 7.1% on this snapshot. See [showcase_rsi_reversion_real_snapshot.md](docs/reports/showcase_rsi_reversion_real_snapshot.md).

### Live-data LLM evolution path

Live yfinance data, configured external LLM provider, `alphaevo evolve rsi_reversion_v1 --method llm`, April 10, 2026:

| Strategy | Signals | Win Rate | Avg Return | Score |
|----------|---------|----------|------------|-------|
| `rsi_reversion_v1` baseline | 0 | 0.0% | 0.00% | 8.1% |
| `rsi_reversion_v3` LLM champion | 498 | 52.6% | 1.22% | 56.3% |

Different data windows and protocols should not be ranked directly; the snapshot showcase is the stable no-key first run, while this path demonstrates the LLM research loop on live market data.

## 🧠 Core Capabilities & Maturity Status

- **Vectorized Quant Engine & Execution Realism**: O(N) precomputed indicators (`IndicatorRegistry.precompute`), O(N) causal regime detection, date-aligned benchmark context, gap-aware open stop/take-profit fills, frozen entry ATR, `next_open` entry-bar A-share limit/ST/suspension enforcement, and dual-layer evaluation (`daily_mtm` primary basket DD/Sharpe + `trade_sequence_*` diagnostics).
- **Recursive self-improvement (`alphaevo rsi`)**: `program.md`-steered L1 strategy mutations, L2 Co-STEER factor synthesis, `(regime × style × horizon)` MAP-Elites archives, persistent executable reflection scars, and heterogeneous proposer/prosecutor reviews (CLI + LiteLLM) with `results.tsv` provenance.
- **L0 truth kernel & forward shadow ledger (`truth` / `shadow`)**: fail-closed SQLite `TrialLedger` with Deflated Sharpe Ratio (DSR), synchronous cross-sectional block-shuffled null controls & strategy-rerun null worlds, restricted-namespace subprocess/runtime point-in-time (PIT) probes, and SHA-256 hash-chained forward commitments with monotonic `as_of`, replay-gate isolation, per-stage `CANARY -> ACTIVE` dwell, and `HALTED` protection.
- **Traceable research workflow**: deterministic or LLM research committee verdicts, evolution trees, and SFT/DPO trajectory exports for every iteration.

| Capability Area | Status | Notes |
|-----------------|--------|-------|
| YAML Strategy DSL v0.5, O(N) Backtest & Dual-Layer MTM Evaluator | **Implemented** | Bundled assets in wheel; cross-platform cache & lazy CLI startup (~0.1s) |
| Unified `EvaluationKernel` & Locked Holdout Vault | **Implemented** | Chronological holdout isolation, Layer-2 portfolio veto gate, Optimizer `TrialLedger` tracking & CI all-indicator PIT invariance tests |
| L0 Truth Kernel (`TrialLedger`, DSR, Joint Null Controls, PIT Probe) | **Implemented** | Fail-closed persistence, runtime PIT truncation on custom factors & CI Tier-2 0-false-promotion audit |
| Dual-Clock RSI Ratchet & Forward Shadow Ledger | **Implemented** | `FactorStore`-persisted Co-STEER L2 factors, unified `ExperienceStore` metrics, SHA-256 hash chain & optional auto-enrollment |
| Immutable Run Provenance & Scoped Leaderboard | **Implemented** | `RunArtifactStore` manifests, secret-free config snapshots, `DataBundleStore` SHA-256 bundles, `runs rerun-official` & `leaderboard --scope local\|official` |
| Multi-Strategy Regime-Adaptive Allocator | **Implemented** | Causal inverse-vol, regime haircut, pairwise correlation penalty & drawdown deleveraging (`alphaevo portfolio allocate`) |
| Alpha Factory Lifecycle Governance | **Implemented** | OOS IC/IR decay auditing, PIT re-probing, pairwise redundancy pruning & strategy lineage (`alphaevo factor audit/prune`) |
| Heterogeneous CLI / LiteLLM Multi-Model Council | **Implemented** | Deterministic general heuristic fallback when API keys unavailable |
| Event/News & Sector Context Indicators (L2/L3) | **Experimental** | Degrades to price/volume proxy or static snapshot with explicit warnings |
| Web UI Dashboard (`alphaevo web`) & Live Arena | **Implemented** | Zero-dep stdlib HTTP Research Studio, inline SVG/HTML Arena & single-file run reports + 7-channel webhook notifier |
| Native AI Agent Integration (`alphaevo mcp` & `SKILL.md`) | **Implemented** | Stdlib JSON-RPC 2.0 STDIO MCP server + agent charter resources for Claude Code, OpenClaw, Cursor & Codex |

## 🚀 Quick Start

### Try the real-data showcase in 30 seconds (no API key needed)

```bash
git clone https://github.com/ZhuLinsen/alphaevo.git
cd alphaevo
pip install -e .
alphaevo showcase
```

<p align="center">
  <img src="docs/demo.gif" alt="AlphaEvo Demo" width="720">
</p>

This runs a real historical-data showcase from a bundled yfinance snapshot:

```
Showcase Chain: baseline + up to 6 validated mutations

Round 1 │ rsi_reversion_v1 │ Signals: 0  │ Score: 6.6%
  Committee: entry stack is too strict
  Mutation: entry.logic and -> or

Round 2 │ rsi_reversion_v2 │ Signals: 79 │ Score: 22.1%
  Mutation: exit.stop_loss.value 0.05 -> 0.08

Round 7 │ rsi_reversion_v7 │ Signals: 37 │ Score: 65.9% 🏆
```

### Choose the right path

| Goal | Command | Data | LLM |
|------|---------|------|-----|
| Real-data showcase (30s first run) | `alphaevo showcase` | Bundled yfinance snapshot | No |
| Synthetic dev smoke test | `alphaevo demo --synthetic` | Synthetic | No |
| Draft & research a plain-language idea | `alphaevo strategy research "<idea>"` | Real data | No |
| Run a fuller real-data backtest | `alphaevo run ma_crossover_v1` | Live yfinance | No |
| Test breakout/volatility-compression template | `alphaevo run volatility_compression_breakout_v1` | Live yfinance | No |
| Search parameters with anti-overfit gates | `alphaevo optimize <id> --objective robust_profit_quality --reject-overfit` | Real data | No |
| Flagship LLM research-agent evolution | `alphaevo evolve <id> --method llm --output reports/` | Real data | Yes |
| Run Stock RSI ratchet loop (`program.md`) | `alphaevo rsi run rsi_reversion_v1 --program examples/rsi_stock_program.md` | Snapshot or `--live` | Optional |
| Verify zero false promotions under shuffled noise | `alphaevo rsi null-audit rsi_reversion_v1` | Snapshot or `--live` | No |
| Track bitemporal forward commitments | `alphaevo shadow register <id> && alphaevo shadow status` | Snapshot or live | No |

Real-data commands need a data adapter extra: `pip install -e ".[data-yfinance]"` (US/global), `pip install -e ".[data-akshare]"` (A-share), or `pip install -e ".[data-full]"`. For US strategies, optional `ADANOS_API_KEY` enriches L3 event indicators with sentiment context (falling back to price/volume proxies when unset).

`alphaevo showcase --live` tries live yfinance before falling back to the bundled snapshot. For `optimize` quality/overfit gates, inspect all options with `alphaevo optimize --help` (keep `--full-eval-top` high enough under `--evaluation-mode fast` when using `--reject-overfit`).

### Current release scope

- **Flagship LLM proof path**: `rsi_reversion_v1` and `ma_crossover_v1` on real data.
- **Most reliable strategy families right now**: trend + reversal strategies whose core signals come from OHLCV / benchmark context.
- **Return-oriented trend template**: `volatility_compression_breakout_v1` uses explicit prior-high breakout triggers, range-position guards, volatility-compression filters, and trailing take profit.
- **Experimental families**: event + rotation strategies still rely partly on proxy context for news / sector-flow semantics, so treat them as research previews rather than the main launch proof.

### Full setup (with LLM evolution)

```bash
# From the cloned repo root, add LLM + default real-data support
pip install -e ".[llm,data-yfinance]"

# Set your LLM API key
export ALPHAEVO_API_KEY=your_api_key
export ALPHAEVO_LLM_MODEL=gemini/gemini-3.8-flash  # or openai/gpt-6.1-sol, anthropic/claude-sonnet-5.5, deepseek/deepseek-v4-flash

# Run the flagship LLM research path
alphaevo evolve rsi_reversion_v1 --method llm --rounds 3 --output reports/rsi_evolve/

# Or run the L0-grounded Stock RSI loop with a research charter
alphaevo rsi run rsi_reversion_v1 --program examples/rsi_stock_program.md --output reports/rsi/
```

For built-in A-share workflows, install `pip install -e ".[llm,data-full]"` and run `alphaevo run trend_pullback_rebound_v1 --adapter akshare`.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI (Typer + Rich)                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Orchestrator & Stock RSI Loop                │
│   charter → sample → backtest → evaluate (L0 Truth)      │
│         → reflect / Co-STEER → evolve → shadow ledger    │
└──┬────────┬────────┬────────┬────────┬────────┬─────────┘
   │        │        │        │        │        │
┌──▼──┐ ┌──▼──┐ ┌───▼──┐ ┌──▼──┐ ┌───▼──┐ ┌──▼────────┐
│Data │ │Stgy │ │Sample│ │Back │ │Truth/│ │Reflection │
│Layer│ │Layer│ │Layer │ │test │ │Eval  │ │& RSI Engine│
└─────┘ └─────┘ └──────┘ └─────┘ └──────┘ └───────────┘
```

### Core Architecture Layers

| Layer | Purpose |
|-------|---------|
| **Data Layer** | Multi-source market data (`yfinance`, `akshare`, or `daily_stock_analysis` plugin) |
| **Strategy Layer** | Dual-representation: human-readable + executable YAML DSL (`SetupSpec`, `PortfolioProfile`) |
| **Sampler Layer** | Smart sampling by market regime, style, and strategy scope |
| **Backtest Engine** | Signal-level simulation with slippage, fees, and point-in-time truncation |
| **Evaluator & L0 Truth (`truth`)** | Multi-dimensional metrics + `TrialLedger`, Deflated Sharpe Ratio, null-signal controls, runtime PIT leak probe |
| **Reflection & RSI (`reflection` / `rsi`)** | LLM & deterministic committee failure attribution, Co-STEER L1/L2 routing, MAP-Elites archive, heterogeneous council, `results.tsv` ratchet |
| **Forward Shadow Ledger (`forward`)** | Bitemporal (`as_of` / `available_at`) SHA-256 hash-chained commitments, cluster-adjusted $N_{\text{eff}}$, and slow-promote / instant-halt gates |

See [Dual-Clock Stock RSI Architecture](docs/rsi_stock_architecture.md) and [examples/rsi_stock_program.md](examples/rsi_stock_program.md) for details.

## 🧬 How Self-Evolution Works

```
Round 1: rsi_reversion_v1 — live yfinance data, LLM-only evolve
         → 0 signals
         → Confidence: 8.1%  ❌ non-functional

         LLM diagnosis: "Entry conditions are contradictory and too strict."
         Changes: entry.logic and→or, RSI 30→35

Round 2: rsi_reversion_v2 — strategy becomes tradable
         → 522 signals
         → Win rate: 52.7%, Avg return: +0.96%
         → Confidence: 39.2%

         LLM diagnosis: "The OR logic over-corrected and now admits noisy entries."
         Changes: RSI 35→32, volume_ratio 1.3→1.15, stop_loss pct→atr

Round 3: rsi_reversion_v3 — champion
         → 498 signals
         → Win rate: 52.6%, Avg return: +1.22%
         → Confidence: 56.3%  (+48.2pp from start)
```

> These are **real results** from an April 10, 2026 run using live yfinance data and a configured external LLM provider with `--method llm`. Not synthetic, not heuristic fallback.

## 📋 Strategy DSL Example

Strategies are stored as human-readable YAML, so the LLM can explain and mutate them without hiding the logic in code.

```yaml
meta:
  id: trend_pullback_rebound_v3
  name: 强趋势回踩放量反包
  version: 3
  category: trend

entry:
  logic: and
  triggers:
    - indicator: relative_strength_20d
      op: ">"
      value: 0.12
  guards:
    - indicator: ma5_above_ma10
      op: "=="
      value: true

exit:
  triggers:
    - indicator: close_below_ma10
      op: "=="
      value: true
  stop_loss:
    type: atr
    atr_period: 21
    multiplier: 2.0
  take_profit:
    type: rr
    value: 2.0

params:
  tunable:
    - target: entry.triggers[indicator=relative_strength_20d].value
      range: [0.05, 0.20]
      step: 0.01
```

See [technical design](docs/technical_design.md) for the full DSL.

## 🔬 Real-World Validation (April 10, 2026)

We validate the system on **live yfinance data** with real `--method llm` runs, and we intentionally show both success and honest failure.

| Case | Start | Best Outcome | Why It Matters |
|------|-------|--------------|----------------|
| `rsi_reversion_v1` | `8.1%` | `56.3%` champion | LLM turned a zero-signal strategy into a tradable one in 3 rounds |
| `ma_crossover_v1` | `24.2%` | `24.2%` champion | LLM proposed changes, but anti-overfit rejected weak generalization |
| `sector_rotation_leader_v1` smoke | `11.3%` | `12.3%` tested | Candidate improved in-sample, but `train_val_gap=18.9%` blocked promotion |

This is part of the product story, not a footnote: AlphaEvo should be trusted more when it stops honestly than when it invents a prettier curve.

Real factor discovery is also live-tested: `alphaevo factor discover AAPL` proposed `3` factors, passed `3` through sandboxing, validated `2`, and registered `2` ([Factor Discovery Walkthrough](docs/cookbook/07_factor_discovery_walkthrough.md)).

Each real run can export `<strategy_id>_research_report.md`, `<strategy_id>_llm_evidence.md`, `<strategy_id>_research_log.md`, and `trajectory/*.jsonl|json`.

### Reproducible Benchmark Path

For a fixed-input benchmark suite, use:

```bash
python scripts/experiments/run_repro_benchmark.py \
  --adapter yfinance \
  --method llm \
  --rounds 3 \
  --output results/repro-benchmark/

python scripts/validate_real_data.py \
  --adapter yfinance \
  --days 365 \
  --output results/real-validation/
```

See the detailed write-up: [April 10, 2026 real LLM validation](docs/reports/2026-04-10-real-llm-validation.md).

Current limits:
- live external LLM providers can still introduce latency spikes and timeout-driven fallback
- some strategies, like `ma_crossover_v1`, show that a valid LLM diagnosis does not automatically survive anti-overfit checks
- discovered factors are research artifacts first and should still be reviewed before any production-style use

## 🧠 Trajectory Data Flywheel

Every real evolution run can export more than scores:

- `trajectory.jsonl` captures each round as `(state -> diagnosis -> hypothesis -> change -> outcome)`
- `sharegpt.jsonl` reformats the same run into SFT-style conversations
- `preference.jsonl` stores improved vs. non-improved steps for preference learning

See: [Trajectory Data Flywheel](docs/trajectory_data_flywheel.md)

## 🛡️ Anti-Overfitting

AlphaEvo takes overfitting seriously:

- **Time Separation**: Train / Validation / Test periods strictly separated
- **Walk-Forward**: Rolling 12-month train → 1-month test windows
- **Complexity & Stability Checks**: Complexity penalty, minimum signal count (`< 30` discounted), multi-year/sector consistency, ±10% parameter perturbation sensitivity
- **Multiple-Testing Control (`TrialLedger` + DSR)**: Append-only trial counting prevents silent p-hacking and penalizes inflated historical Sharpe ratios
- **Null-Signal & Runtime PIT Probes**: Random-entry / shuffled-return null floors plus isolated subprocess future-noise perturbation checks block forward-looking factor leaks
- **Forward Shadow Promotion Gate**: Historical backtests only veto; promotion requires bitemporal forward commitments with cluster-adjusted effective sample size ($N_{\text{eff}}$) and instant `HALTED` trips on drawdown/expectancy breaches

## 🔌 Compatibility

### Standalone Mode
Works out of the box with `yfinance` or `akshare` for data.

### Plugin Mode (daily_stock_analysis)
Seamlessly integrates with daily_stock_analysis for multi-source data with automatic fallback.

```python
from alphaevo.data.adapters.dsa import DSAAdapter
data_manager = DataManager([DSAAdapter(dsa_path="/path/to/dsa")])
```

## 📊 CLI Commands

```bash
alphaevo web                       # Zero-dependency local Web Research Studio (http://127.0.0.1:8765)
alphaevo arena run --rounds 2     # Multi-strategy Live Arena -> single-file interactive index.html + notify
alphaevo arena notify-test        # Dry-run or test Feishu/WeCom/DingTalk/Telegram/Discord/Slack alerts
alphaevo mcp                      # STDIO MCP JSON-RPC 2.0 server for AI coding/research agents
alphaevo showcase                 # Real-data 30s snapshot showcase (report.md + report.html, no API key)
alphaevo demo --real              # Real data demo without API key
alphaevo data bundle create       # Freeze & SHA-256 hash market data into an immutable bundle
alphaevo run <id> --bundle <bid>  # Run full loop on an immutable verified bundle
alphaevo runs list                # Inspect indexed run/evolve/optimize/rsi artifacts
alphaevo runs html <run_id>       # Export any immutable run as standalone single-file HTML
alphaevo runs rerun-official <id> --bundle <bid>  # Verify candidate under official_v1
alphaevo runs compare <r1> <r2>   # Compare immutable manifests & strategy YAML diffs
alphaevo portfolio allocate <s1> <s2>  # Causal regime-adaptive multi-strategy risk budget
alphaevo evolve <id> --method llm --rounds 3 --output reports/  # LLM-first evolution
alphaevo rsi run <id> --holdout-pct 0.20 --portfolio-gate --enroll-shadow
alphaevo rsi null-audit <id>      # Verify 0 false promotions under shuffled returns
alphaevo rsi mechanism-ab <id> --cutoffs 0.55,0.70  # L3 multi-cutoff kernel mechanism audit
alphaevo shadow simulate <id>     # Rolling bitemporal forward shadow ledger simulation
alphaevo shadow status            # Forward shadow ledger & promotion stages
alphaevo truth trials             # Inspect append-only TrialLedger & DSR records
alphaevo factor audit             # Audit OOS IC/IR decay, PIT invariance & redundancy
alphaevo factor prune --dry-run   # Retire decayed, redundant, or PIT-unsafe factors
alphaevo factor discover <symbol> # LLM-driven factor discovery
alphaevo leaderboard --scope local  # Strategy rankings (use --scope official for verified runs)
alphaevo tree <id>                # Evolution tree visualization
```

For the full command surface, run `alphaevo --help`.

## 📚 Research Inspirations

- **FunSearch (Nature 2024)** — island-style parallel search and branch competition
- **OPRO (DeepMind 2023)** — optimizer-style prompt refinement with trajectory history
- **Voyager (2023)** — reusable skill/pattern libraries and long-horizon memory
- **autoresearch (Karpathy)** — `program.md`-steered hypothesis charters, keep/revert ratchets, and verifiable `results.tsv` ledgers
- **RD-Agent / Co-STEER** — co-steered strategy-level vs. factor-level blind-spot routing and empirical scar reuse

AlphaEvo adapts these ideas to falsifiable quantitative equity strategy research rather than general coding or benchmark optimization.

## 🗺️ Roadmap

- [x] Phase 1: Strategy Research Loop (MVP) — backtest engine, indicators, evaluator
- [x] Phase 2: Self-Evolution Pipeline — LLM reflection, mutation, multi-round improvement
- [x] Phase 3: CLI & Orchestration — full command suite, strategy store, leaderboard
- [x] Phase 4: Open-Source Polish — CI/CD, docs, English templates, CHANGELOG
- [x] Phase 5: Recursive Stock Self-Improvement & Shadow Ledger — `TrialLedger` + DSR, null controls, runtime PIT probe, MAP-Elites, heterogeneous council, bitemporal forward tracking
- [x] Phase 6: Market Regime Adaptive Gating & Canonical Data Bundles — causal multi-strategy `RegimeAdaptiveAllocator`, risk budgets, SHA-256 `DataBundleStore`, unified `EvaluationKernel`, and `runs rerun-official`
- [x] Phase 7: Web Studio, Evolution Observatory & Agent Ecosystem — zero-dep `alphaevo web` with live SSE `/api/rsi/stream`, Adversarial Council Debate cards, Evolution Lineage Tree & Trade K-Line Replay SVGs, Overfit Graveyard `alphaevo arena`, `alphaevo mcp` + `SKILL.md`, 7-channel push alerts, Docker Web Studio & GitHub Pages deployment

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Especially looking for:**
- New strategy templates
- Data source adapters
- Evaluation metrics
- UI/visualization improvements

## ☕ Support & Contact

If this project helps your research, consider giving it a ⭐ and sharing it!

<table>
  <tr>
    <td align="center" width="220">
      <a href="https://www.xiaohongshu.com/user/profile/61594417000000000201fa68" target="_blank">
        <img src="./sources/xiaohongshu.png" width="160" alt="小红书"><br>
        <b>Xiaohongshu 📱</b>
      </a><br>
      <sub>Follow for quant strategy research updates</sub>
    </td>
    <td valign="top" style="padding-left: 24px">
      <b>📬 Contact & Collaboration</b><br><br>
      🐛 &nbsp;<a href="https://github.com/ZhuLinsen/alphaevo/issues">Submit an Issue</a> — Bug reports / feature requests<br>
      📧 &nbsp;<a href="mailto:zhuls345@gmail.com">zhuls345@gmail.com</a> — Business inquiries<br>
      🔗 &nbsp;<a href="https://github.com/ZhuLinsen/daily_stock_analysis">daily_stock_analysis</a> — Sister project, AI-powered daily stock analysis
    </td>
  </tr>
</table>

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It does not constitute investment advice. The authors are not responsible for any financial losses incurred from using this software. Always do your own research and consult qualified financial advisors before making investment decisions.

Past strategy performance does not guarantee future results. All backtesting results are simulated and may not reflect real market conditions.

## 📄 License

Apache-2.0 License — see [LICENSE](LICENSE) for details.

If you use or build upon this project, a credit with a link back to this repository is appreciated.

---

<div align="center">


</div>
