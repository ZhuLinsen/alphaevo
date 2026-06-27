# Issue #6 — OHLCV-only trend-pullback research note

## Scope

Issue #6 asked for an OHLCV-only branch of the `trend_pullback_rebound` family that removes proxy-heavy event/news filters and compares the candidate against the current v11/v12 A-share baselines.

This note is research output only, not investment advice and not an official benchmark.

## Candidate produced

- YAML: `strategies/research/trend_pullback_rebound_ohlcv_only_v13.yaml`
- Base: `trend_pullback_rebound_trigger_guard_v12_ma20_slope`
- Structural change: removed `negative_news_score` and `st_flag` filters; did not add any event/news/proxy indicators.
- Remaining executable entry indicators:
  - trigger: `volume_ratio_1d_3d > 1.2`
  - guards: `relative_strength_43d > 0.06`, `ma5_above_ma10 == true`, `close_to_ma10_pct <= 0.015`, `close_above_ma20 == true`, `ma20_slope > 0.0`
- Exit: unchanged trailing profile from v12 (`stop_loss=2.56%`, `take_profit=trailing 10%/5%`, `max_holding_days=10`).

## Representative run protocol

- Adapter: `yfinance`
- Market: A-share
- Sampling: `strategy_scoped`
- Max symbols: 63
- Date range: 2024-01-01 → 2026-06-21
- Compared strategies:
  - `trend_pullback_rebound_trigger_guard_v11`
  - `trend_pullback_rebound_trigger_guard_v12_ma20_slope`
  - `trend_pullback_rebound_ohlcv_only_v13`

Generated local reports were written under ignored local artifacts:

- `reports/issue6_ohlcv_only_comparison/trend_pullback_rebound_trigger_guard_v11_report.md`
- `reports/issue6_ohlcv_only_comparison/trend_pullback_rebound_trigger_guard_v12_ma20_slope_report.md`
- `reports/issue6_ohlcv_only_comparison/trend_pullback_rebound_ohlcv_only_v13_report.md`

## Result summary

| Strategy | Event/news entry indicators | Signals | Avg Return | P/L | Max DD | Total Return | Confidence | WF Gap | WF Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v11 baseline | `negative_news_score` | 47 | 1.97% | 2.85 | 7.93% | 129.63% | 62.47% | 30.17% | 33.33% |
| v12 ma20_slope | `negative_news_score` | 45 | 2.06% | 2.96 | 6.54% | 130.67% | 47.41% | 30.36% | 50.00% |
| v13 OHLCV-only | none | 45 | 2.06% | 2.96 | 6.54% | 130.67% | 49.41% | 30.36% | 50.00% |

## Gate check

| Gate from issue #6 | Outcome |
|---|---|
| Minimum signals around 45-48 | Pass: v13 fired 45 signals. |
| Avg return at/above current baseline | Pass vs v11 and v12: 2.06% >= 1.97% and equal to v12. |
| P/L ratio at/above current baseline | Pass vs v11 and v12: 2.96 >= 2.85 and equal to v12. |
| Max drawdown no worse than current baseline | Pass vs v11 and v12: 6.54% <= 7.93% and equal to v12. |
| No more proxy/event indicators | Pass: v13 has no event/news indicators in the executable entry stack. |

## Verdict

`trend_pullback_rebound_ohlcv_only_v13` is a risk-adjusted OHLCV-only alternative, not a new confidence champion.

Why:

- It exactly preserved the v12 risk-adjusted metrics in this representative run while removing the proxy-heavy event/news filter.
- It improved the v12 confidence score slightly because the active entry stack no longer depends on event/news indicators.
- It still trails v11 on confidence score and trails buy-and-hold on benchmark total return, so it should not be promoted as a champion.
- Walk-forward gaps remain high (30.36%), so the next useful work is robustness/generalization, not more event/news threshold tuning.

## Research implication

The previous `negative_news_score` filter was not adding measurable value in this sample once the `ma20_slope > 0.0` guard was present. Removing it avoids proxy-dominant event/news dependence without weakening the observed v12 metrics.

Recommended next step: keep v13 as the event/news-free comparison branch for future optimization, but require walk-forward robustness improvement before champion promotion.
