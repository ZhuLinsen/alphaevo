<div align="center">

<img src="sources/alphaevo_terminal_v1_agent_evolution.svg" alt="AlphaEvo Logo" width="660">

# 🧬 AlphaEvo

**`RSI (Evo) + Stock` — The Open-Source Recursive Self-Improvement Engine for Equity Research**

*Evolve Stock Strategies (L1), Point-in-Time Factors (L2), and Research Harnesses (L3) under Anti-Delusion Null Controls & Forward Shadow Ledgers*

[![GitHub stars](https://img.shields.io/github/stars/ZhuLinsen/alphaevo?style=social)](https://github.com/ZhuLinsen/alphaevo/stargazers)
[![CI](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://github.com/ZhuLinsen/alphaevo)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[English](README.md) | [中文](docs/README_CN.md) | [RSI + Stock Spec](docs/rsi_stock_architecture.md) | [Full DSL & Design](docs/technical_design.md)

</div>

---

## ⚡ Why RSI (Evo) + Stock

> **Historical backtests only VETO; forward out-of-sample Shadow Ledgers PROMOTE.**

Naive `autoresearch`-style loops collapse on equities due to low signal-to-noise ratios, same-day cross-sectional clustering ($N_{\text{eff}} \ll N$), multiple-testing Sharpe inflation, and subtle future data leaks inside synthesized indicators. **AlphaEvo (`RSI (Evo) + Stock`)** turns recursive self-improvement into a falsifiable equity workflow:

| Dimension | Generic Code/ML RSI (`autoresearch`) | Typical LLM Stock Agents | **AlphaEvo (`RSI (Evo) + Stock`)** |
|---|---|---|---|
| **Search Loop** | Single-script keep/revert ratchet | One-shot prompt / static screener | **3-Level RSI** (L1 Strategy + L2 Factor + L3 Harness) + **MAP-Elites** |
| **Multiple-Testing Defense** | Single validation split | Rarely tracks failed retries | Append-only **`TrialLedger`** + **Deflated Sharpe Ratio (DSR)** |
| **Zero-Edge Sanity Gate** | None | None | **Two-Tier Null Control** (`block_shuffle` / `time_shift` / `random_entry`, **0** false promotions) |
| **Lookahead Leakage Audit** | N/A | Prompt-level hope | **Runtime PIT Future-Noise & Truncation Probe** + strict `<= as_of` slicing |
| **Promotion Governance** | Direct git commit | No forward accountability | **SHA-256 Hash-Chained Shadow Ledger**, $N_{\text{eff}}$ gating, instant `HALTED` redline |

---

## 🚀 30-Second Quick Start (No API Key Needed)

```bash
git clone https://github.com/ZhuLinsen/alphaevo.git && cd alphaevo
pip install -e .
alphaevo showcase                                      # Frozen real yfinance snapshot evolution
alphaevo rsi run --program examples/rsi_stock_program.md  # L0-guarded RSI ratchet + results.tsv
```

<p align="center">
  <img src="docs/demo.gif" alt="AlphaEvo Demo" width="700">
</p>

### Pick Your Path

| Goal | Command | Data / LLM |
|---|---|---|
| **30s real-snapshot showcase** | `alphaevo showcase` | Bundled snapshot / Zero-key |
| **RSI (Evo) + Stock ratchet (`program.md`)** | `alphaevo rsi run --program examples/rsi_stock_program.md` | Snapshot or `--live` / Optional |
| **Tier-2 shuffled-return audit** | `alphaevo rsi null-audit --worlds 3 --rounds 3` | Shuffled market nulls / No |
| **Forward Shadow Ledger lifecycle** | `alphaevo shadow register\|emit\|backfill\|status\|verify` | Strict PIT OHLCV / No |
| **Multi-model Adversarial Council** | `alphaevo rsi run --method llm --council-proposer codex:... --council-prosecutor agy:...` | Any / Distinct families (`proposer != prosecutor`) |
| **US / A-share live LLM & param evolve** | `pip install -e ".[llm,data-full]"` → `alphaevo evolve rsi_reversion_v1 --method llm` | Live `yfinance` / `akshare` / Yes |

*For US workflows with social sentiment context, set `ADANOS_API_KEY` optionally; when unset, AlphaEvo uses deterministic price/volume proxies.*

---

## 🏗️ Dual-Clock Architecture & Stock DSL

```
FAST CLOCK (Offline Hypothesis Search — L1 / L2 / L3 RSI)
  program.md (RSICharter) ─▶ BlindSpotMiner (L1 Strategy vs L2 Factor + Runtime PIT Probe)
           ▲                          │
           │ Keep / Revert            ▼
  MAP-Elites Archive ◀── HeterogeneousCouncil (Proposer family != Prosecutor family + Pinned Scars)
  (Regime×Style×Hold)                 │
                                      ▼
                    L0 Immutable Truth Kernel (`alphaevo/truth`):
                    TrialLedger · Deflated Sharpe (DSR) · NullSignalController
                                      │ Survivors only
                                      ▼
SLOW CLOCK (Out-of-Sample Forward Accountability — `alphaevo/forward`)
  CandidateSpec ─▶ commit_signals(as_of <= t) ─▶ backfill T+5/10/20 post-cost ─▶ N_eff + Asymmetric Gate
                    [SHA-256 Chain + Hard Cap]                                 (DRAFT→SHADOW→CANARY→ACTIVE|HALTED)
```

Strategies live in a declarative YAML DSL with stock-native `SetupSpec` and tunable ranges ([full spec](docs/rsi_stock_architecture.md) · [technical design](docs/technical_design.md)):

```yaml
meta:
  id: trend_pullback_rsi_v1
  market: us
  category: reversal
setup:
  trigger_bar: "rsi_14 < 30 and volume_ratio_1d_5d >= 1.5"
  invalidation_rule: "close < ma60 * 0.96"
  target_rr: 3.0
  min_rr: 2.0
entry:
  logic: or
  conditions:
    - { indicator: rsi_14, op: "<", value: 30 }
    - { indicator: close_to_ma60_pct, op: "<=", value: 0.06 }
exit:
  stop_loss: { type: pct, value: 0.08 }
  take_profit: { type: rr, value: 4.0 }
  max_holding_days: 14
params:
  tunable:
    - { target: entry.conditions[indicator=rsi_14].value, range: [22, 36], step: 2 }
```

---

## 🔬 Real-Data Validation & Honest Stops

We report verified wins alongside cases where anti-overfit and L0 null controls refuse promotion:

| Benchmark / Check | Protocol & Dataset | Observed Outcome | Reference |
|---|---|---|---|
| **No-Key Snapshot Showcase** | `rsi_reversion_v1 -> v7`, frozen US tech basket (2025-02 to 2026-04) | `0 -> 37` signals, `48.6%` win rate, `+2.94%` avg ret, `23.6%` max DD, `8.1% -> 68.7%` | [Snapshot report](docs/reports/showcase_rsi_reversion_real_snapshot.md) |
| **Live LLM Evolution** | `rsi_reversion_v1 -> v3`, live `yfinance` + `--method llm` | `0 -> 498` signals, `52.6%` win rate, `+1.22%` avg ret, `8.1% -> 56.3%` | [LLM report](docs/reports/2026-04-10-real-llm-validation.md) |
| **Overfit Rejection** | `ma_crossover_v1` & `sector_rotation_leader_v1` | LLM changes rejected (`train_val_gap=18.9%` / weak OOS stability) | [LLM report](docs/reports/2026-04-10-real-llm-validation.md) |
| **L0 Tier-2 Null Audit** | `alphaevo rsi null-audit` across block-shuffled null markets | **0 false promotions** under pure synthetic/shuffled noise | [RSI + Stock spec](docs/rsi_stock_architecture.md) |
| **L2 Factor Sandbox** | `alphaevo factor discover AAPL` | `3` proposed → `3` sandboxed → `2` validated & registered | [Factor walkthrough](docs/cookbook/07_factor_discovery_walkthrough.md) |

---

## 📊 CLI & Ecosystem

```bash
# Dual-Clock RSI (Evo) + Stock
alphaevo rsi run --program examples/rsi_stock_program.md   # L1/L2 keep/revert + results.tsv & report
alphaevo rsi null-audit --worlds 3                         # Zero-promotion check on shuffled markets
alphaevo rsi mechanism-ab --rounds 3                       # L3 Meta-RSI paired harness AB test
alphaevo shadow register <id> && alphaevo shadow emit <id> # SHA-256 hash-chained PIT commitment
alphaevo shadow backfill && alphaevo shadow status         # T+5/10/20 decay & cluster-adjusted N_eff
alphaevo truth trials && alphaevo truth pit-probe f.py     # L0 TrialLedger DSR & future-noise PIT audit

# Classic Strategy, Optimization & Evolution
alphaevo run <id> && alphaevo evolve <id> --method llm     # Full backtest or LLM-guided evolution
alphaevo optimize <id> --objective robust_profit_quality   # Joint entry/exit search with overfit gates
alphaevo factor discover <symbol> && alphaevo leaderboard  # L2 factor synthesis & strategy rankings
```

Run `alphaevo --help` or `alphaevo <cmd> --help` for the full option surface.

- **Multi-Source Data & DSA Plugin**: Standalone `yfinance` (US/HK) and `akshare` (A-share), or plug in [daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) via `DSAAdapter`.
- **Trajectory Data Flywheel**: Every run can export `trajectory.jsonl`, `sharegpt.jsonl` (SFT), and `preference.jsonl` (DPO) — see [Trajectory Data Flywheel](docs/trajectory_data_flywheel.md).
- **Multi-Layer Anti-Overfit**: Strict train/val/test separation, rolling walk-forward, CPCV, DSR trial accounting, and parameter sensitivity checks.

---

## 🗺️ Roadmap

- [x] **Core Loop & DSL**: YAML DSL + `SetupSpec`, signal backtester, multi-dimensional evaluator & maturity gates
- [x] **Dual-Clock RSI (Evo) + Stock**: `TrialLedger` + DSR, Null-Signal Controls, Runtime PIT Probe, Co-STEER L2, Pinned Scars, Heterogeneous Council, and Bitemporal Shadow Ledger
- [ ] **Portfolio Multi-Strategy Allocation**: Joint cross-strategy equity backtesting & dynamic capital budgeting
- [ ] **Web Evolution Lab UI**: Interactive strategy lineage tree, shadow-ledger monitor, and MAP-Elites heatmap dashboard

---

## 🤝 Contributing, Disclaimer & License

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under **Apache-2.0** ([LICENSE](LICENSE)).

> **Disclaimer**: Research and educational tooling only — **not investment advice**. Simulated backtests and forward shadow ledgers do not guarantee future performance.

<table>
  <tr>
    <td align="center" width="180">
      <a href="https://www.xiaohongshu.com/user/profile/61594417000000000201fa68" target="_blank">
        <img src="sources/xiaohongshu.png" width="130" alt="Xiaohongshu"><br>
        <b>Xiaohongshu 📱</b>
      </a>
    </td>
    <td valign="top" style="padding-left: 20px">
      <b>📬 Contact</b><br><br>
      🐛 <a href="https://github.com/ZhuLinsen/alphaevo/issues">GitHub Issues</a> — Bug reports &amp; feature requests<br>
      📧 <a href="mailto:zhuls345@gmail.com">zhuls345@gmail.com</a> — Collaboration<br>
      🔗 <a href="https://github.com/ZhuLinsen/daily_stock_analysis">daily_stock_analysis</a> — Sister AI stock analysis project
    </td>
  </tr>
</table>
