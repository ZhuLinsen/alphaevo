---
name: alphaevo
description: Autonomous recursive self-improvement (RSI) engine for equity trading strategies. Use when asked to draft, backtest, evolve, stress-test, null-audit, shadow-verify, or risk-allocate stock trading strategies (US, A-share, HK) with anti-delusion guarantees.
---

# AlphaEvo Agent Skill

AlphaEvo is a **stock trading strategy self-evolution, backtesting, and anti-overfitting research engine** — not a full-market daily stock screener or trade-recommendation bot.

It separates **fast offline hypothesis search** (which can only *veto*) from **slow bitemporal forward shadow ledgers** (which govern *promotion*). Trading strategies live as inspectable YAML DSL files rather than opaque code blobs.

## Connect via MCP (Recommended)

Add AlphaEvo's zero-dependency STDIO MCP server to Claude Code, Cursor, OpenClaw, Codex, or Antigravity:

```json
{
  "mcpServers": {
    "alphaevo": {
      "command": "alphaevo",
      "args": ["mcp"]
    }
  }
}
```

Available MCP tools:
- `alphaevo_list_strategies` — list stored & built-in strategies
- `alphaevo_draft_strategy` — compile natural language ideas into executable YAML DSL (zero LLM needed)
- `alphaevo_diagnose_bottlenecks` — X-ray per-condition pass rates without consuming L0 trials
- `alphaevo_evaluate_strategy` — evaluate through `EvaluationKernel` (`daily_mtm`, DSR, null control, ResearchCommittee) and persist immutable artifacts
- `alphaevo_evolve_rsi` — run Karpathy `program.md`-steered L1/L2 keep/revert ratchets (with Co-STEER PIT-safe factor synthesis)
- `alphaevo_mechanism_ab` — compare L3 Meta-RSI harnesses (Naive vs L0+Pinned Scars+L2) across chronological OOS cutoffs
- `alphaevo_null_audit` — Tier-2 audit verifying 0 false promotions under shuffled returns
- `alphaevo_shadow_simulate` — run rolling bitemporal forward shadow commitments & SHA-256 chain checks
- `alphaevo_audit_factors` — audit dynamic Alpha Factory factors on OOS IC/IR decay & runtime PIT safety
- `alphaevo_allocate_portfolio` — compute causal inverse-vol, regime-gated, correlation-penalized portfolio weights
- `alphaevo_list_runs` — inspect indexed immutable runs

## Core CLI Workflows

```bash
# Launch interactive local Web Research Studio
alphaevo web

# Run multi-strategy Live Arena + generate shareable single-file HTML dashboard
alphaevo arena run --rounds 2 --output-dir reports/arena

# Draft and evolve a new idea from scratch
alphaevo strategy draft "RSI below 30 above MA60 with 5% stop loss" --market us --save --id my_rsi_v1
alphaevo rsi run my_rsi_v1 --program examples/rsi_stock_program.md --holdout-pct 0.20 --enroll-shadow
alphaevo rsi null-audit my_rsi_v1
alphaevo shadow simulate my_rsi_v1
```

## Non-Negotiable Integrity Rules

1. Never promote a strategy whose `is_overfit=True`, `null_control_passed=False`, or `signal_count < min_signals`.
2. Never rank `fast_research_v1` / `full_research_v1` runs alongside `official_v1` runs (`alphaevo leaderboard --scope official`).
3. Always attach the research-only disclaimer: AlphaEvo is educational & research software, not investment advice.
