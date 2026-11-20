# AlphaEvo Real-Data Showcase: RSI Reversion

> Research tooling only. Not investment advice.

## Summary

| Item | Value |
|------|-------|
| Run ID | `20261120T151813Z_rsi_reversion_1181fd2d` |
| Generated At | `2026-11-20T15:18:13+00:00` |
| Data Source | bundled frozen yfinance snapshot |
| Snapshot | `us_tech_showcase_2025_2026` |
| Date Range | 2025-02-11 to 2026-04-10 |
| Symbols | AAPL, AMD, MSFT, NVDA, TSLA |
| Baseline | `rsi_reversion_v1` score 6.6% |
| Champion | `rsi_reversion_v7` score 65.9% |

## Evolution Results

| Round | Strategy | Change | Signals | Win Rate | Avg Return | Max DD | Score |
|-------|----------|--------|---------|----------|------------|--------|-------|
| 1 | `rsi_reversion_v1` | baseline | 0 | 0.0% | 0.00% | 0.0% | 6.6% |
| 2 | `rsi_reversion_v2` | `entry.logic`: `and` -> `or` | 79 | 40.5% | 0.35% | 37.0% | 22.1% |
| 3 | `rsi_reversion_v3` | `exit.stop_loss.value`: `0.05` -> `0.08` | 65 | 52.3% | 1.17% | 34.7% | 20.9% |
| 4 | `rsi_reversion_v4` | `exit.max_holding_days`: `10` -> `14` | 57 | 47.4% | 1.93% | 28.2% | 38.0% |
| 5 | `rsi_reversion_v5` | `exit.take_profit.value`: `2.0` -> `6.0` | 56 | 48.2% | 2.74% | 27.4% | 44.9% |
| 6 | `rsi_reversion_v6` | `entry.conditions[indicator=close_to_ma20_pct].indicator`: `close_to_ma20_pct` -> `close_to_ma60_pct` | 38 | 47.4% | 2.59% | 27.5% | 51.8% |
| 7 | `rsi_reversion_v7` | `entry.conditions[indicator=volume_ratio_1d_5d].value`: `1.3` -> `2.0` | 37 | 48.6% | 2.94% | 26.7% | 65.9% |

## Champion Diagnostics

| Metric | Value |
|--------|-------|
| Signals | 37 |
| Win Rate | 48.6% |
| Average Return | 2.94% |
| Sequential Total Return | 31.78% |
| Profit/Loss Ratio | 2.46 |
| Max Drawdown | 26.7% |
| Train-Val Gap | 7.1% |
| Val-Test Gap | 7.1% |
| Yearly Consistency | 97.1% |
| Overfit Flag | no |

The champion remains a showcase result, not an official benchmark: it uses a fixed five-symbol snapshot and should be revalidated on broader universes before any serious research claim.

## Research Committee

| Round | Analyst | Status | Verdict |
|-------|---------|--------|---------|
| 1 | Technical Analyst | fail | Entry stack is too strict for a reliable research sample. |
| 1 | Risk Analyst | pass | Drawdown and loss clustering are within the research tolerance band. |
| 1 | Overfit Critic | fail | Sample is too small to trust improvement claims. |
| 1 | Data Quality Auditor | pass | Data source and symbol basket are explicit enough for a showcase. |
| 1 | Mutation Planner | pass | Plan starts by unlocking enough signals for a valid retest. |
| 2 | Technical Analyst | pass | Technical signal has enough trades and positive payoff shape. |
| 2 | Risk Analyst | watch | Risk is usable for research but still needs exit discipline. |
| 2 | Overfit Critic | fail | Anti-overfit checks reject promotion. |
| 2 | Data Quality Auditor | pass | Data source and symbol basket are explicit enough for a showcase. |
| 2 | Mutation Planner | pass | Plan changes one controlled lever at a time. |
| 3 | Technical Analyst | pass | Technical signal has enough trades and positive payoff shape. |
| 3 | Risk Analyst | watch | Risk is usable for research but still needs exit discipline. |
| 3 | Overfit Critic | fail | Anti-overfit checks reject promotion. |
| 3 | Data Quality Auditor | pass | Data source and symbol basket are explicit enough for a showcase. |
| 3 | Mutation Planner | pass | Plan changes one controlled lever at a time. |
| 4 | Technical Analyst | pass | Technical signal has enough trades and positive payoff shape. |
| 4 | Risk Analyst | pass | Drawdown and loss clustering are within the research tolerance band. |
| 4 | Overfit Critic | fail | Anti-overfit checks reject promotion. |
| 4 | Data Quality Auditor | pass | Data source and symbol basket are explicit enough for a showcase. |
| 4 | Mutation Planner | pass | Plan changes one controlled lever at a time. |
| 5 | Technical Analyst | pass | Technical signal has enough trades and positive payoff shape. |
| 5 | Risk Analyst | pass | Drawdown and loss clustering are within the research tolerance band. |
| 5 | Overfit Critic | fail | Anti-overfit checks reject promotion. |
| 5 | Data Quality Auditor | pass | Data source and symbol basket are explicit enough for a showcase. |
| 5 | Mutation Planner | pass | Plan changes one controlled lever at a time. |
| 6 | Technical Analyst | pass | Technical signal has enough trades and positive payoff shape. |
| 6 | Risk Analyst | pass | Drawdown and loss clustering are within the research tolerance band. |
| 6 | Overfit Critic | fail | Anti-overfit checks reject promotion. |
| 6 | Data Quality Auditor | pass | Data source and symbol basket are explicit enough for a showcase. |
| 6 | Mutation Planner | pass | Plan changes one controlled lever at a time. |
| 7 | Technical Analyst | pass | Technical signal has enough trades and positive payoff shape. |
| 7 | Risk Analyst | pass | Drawdown and loss clustering are within the research tolerance band. |
| 7 | Overfit Critic | watch | Sample is acceptable for showcase, but still thin. |
| 7 | Data Quality Auditor | pass | Data source and symbol basket are explicit enough for a showcase. |
| 7 | Mutation Planner | watch | No mutation plan is available; evaluate current version only. |

## Mutation Evidence

### Round 2: `rsi_reversion_v2`
- `entry.logic` changed from `and` to `or`.
  Rationale: Baseline fired too few signals; test OR logic to unlock oversold reversal candidates without adding complexity.

### Round 3: `rsi_reversion_v3`
- `exit.stop_loss.value` changed from `0.05` to `0.08`.
  Rationale: Losses were being cut inside normal volatility; test a wider stop and require the retest to improve.

### Round 4: `rsi_reversion_v4`
- `exit.max_holding_days` changed from `10` to `14`.
  Rationale: Mean reversion needed more time to complete; test a longer holding window while watching drawdown.

### Round 5: `rsi_reversion_v5`
- `exit.take_profit.value` changed from `2.0` to `6.0`.
  Rationale: Validated high-volume reversals needed a larger payoff target; test a wider reward multiple instead of taking profits too early.

### Round 6: `rsi_reversion_v6`
- `entry.conditions[indicator=close_to_ma20_pct].indicator` changed from `close_to_ma20_pct` to `close_to_ma60_pct`.
  Rationale: A longer support anchor reduced whipsaw drawdown on the snapshot; test MA60 proximity as the reversal support context.

### Round 7: `rsi_reversion_v7`
- `entry.conditions[indicator=volume_ratio_1d_5d].value` changed from `1.3` to `2.0`.
  Rationale: After OR logic unlocked signals, require stronger volume confirmation to filter weaker rebounds.

## Run Provenance

| Field | Value |
|-------|-------|
| Strategy Hash | `37ed9aac9f73` |
| Data Fingerprint | `1181fd2d09f0` |
| Data Reproducibility | `replayable_snapshot` |
| Adapter | `yfinance` |
| Config | `showcase_default_v1` |

Live reruns can differ because public providers may revise historical data.
