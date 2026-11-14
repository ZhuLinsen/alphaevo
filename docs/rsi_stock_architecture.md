# AlphaEvo Dual-Clock RSI (Evo) + Stock Architecture

> **Core Invariant**: *Historical backtests only VETO; Forward out-of-sample Shadow Ledgers PROMOTE.*

## 1. Why Stock RSI Differs From Generic `autoresearch`

In code or ML training (`karpathy/autoresearch`), a single scalar validation metric changes smoothly and deterministically after a patch. In equity research, naive single-clock self-improvement breaks due to four structural hazards:

1. **Non-stationarity & high Sharpe standard error**: 5-day equity Sharpe variance is huge; multiple testing turns noise into apparent breakout curves.
2. **Multiple-testing / Goodhart collapse**: Trying 50 mutations in silence inflates max historical Sharpe even under zero true alpha.
3. **Hidden lookahead in synthesized factors**: LLM-written `compute(df, idx)` code can call global statistics (`df["close"].mean()`, `df.iloc[-1]`) that evade simple AST pattern matching.
4. **Cross-sectional signal clustering**: Firing 15 signals on semiconductor stocks on the same sell-off day is closer to 2–3 independent bets ($N_{\text{eff}}$), not 15 i.i.d. observations.

## 2. Dual-Clock Architecture

```
  FAST CLOCK (Offline Hypothesis Search — L1 / L2 / L3 RSI)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ program.md (RSICharter) ──▶ BlindSpotMiner (L1 Strategy vs L2 Factor│
  │         ▲                            │                              │
  │         │ Keep/Revert                ▼                              │
  │  MAP-Elites Archive ◀── HeterogeneousCouncil (Proposer != Prosecutor│
  │  (Regime x Style x Horizon)          │                              │
  │         │                            ▼                              │
  │         └──────── L0 Immutable Truth Kernel (`alphaevo/truth`) ─────│
  │                   • TrialLedger + Deflated Sharpe Ratio (DSR)       │
  │                   • NullSignalController (block_shuffle/shift/rand) │
  │                   • Runtime PIT Future-Noise & Truncation Probe     │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │ Veto-surviving candidates only
                                     ▼
  SLOW CLOCK (Out-of-Sample Forward Accountability — `alphaevo/forward`)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ CandidateSpec (SHADOW) ──▶ commit_signals(as_of <= today)           │
  │                              │ sector hard_cap + SHA-256 chain      │
  │                              ▼                                      │
  │                        backfill_outcomes(T+5, T+10, T+20)           │
  │                              │ tradable post-cost return + N_eff    │
  │                              ▼                                      │
  │  PromotionStage: DRAFT -> VALIDATED -> SHADOW -> CANARY -> ACTIVE   │
  │                  (Slow Promote: >=30d, N_eff, LB90 > 0)             │
  │                  | Instant HALTED on drawdown/expectancy breach     │
  └─────────────────────────────────────────────────────────────────────┘
```

## 3. Package Map

- `src/alphaevo/truth/` — L0 Truth Kernel:
  - `ledger.py`: append-only SQLite `TrialLedger`, `compute_deflated_sharpe`, `compute_kernel_fingerprint`.
  - `null_control.py`: `NullSignalController` (Tier-1 candidate null p95 gate; Tier-2 `verify_zero_promotion_under_shuffled_returns`).
  - `pit_probe.py`: `probe_point_in_time_invariance` executing factor code in isolated subprocesses against corrupted future bars and truncated prefixes.
- `src/alphaevo/forward/` — Forward Shadow Ledger:
  - `contracts.py`: `BitemporalEnvelope` (`as_of` vs `available_at`), `SetupInstance`, hash-chained `ForecastCommit`, `OutcomeEvent`, `ForwardDecayReport`, `PromotionDecision`.
  - `shadow_ledger.py`: PIT truncation, sector `hard_cap` + market-temperature capacity reduction, SHA-256 chain verification, cluster-adjusted $N_{\text{eff}}$, asymmetric promotion/halt gates.
- `src/alphaevo/rsi/` — L1 + L2 + L3 Self-Improvement Engine:
  - `program.py`: parses `examples/rsi_stock_program.md` charters.
  - `qd_archive.py`: `MAPElitesArchive` maintaining diversity across `(regime_bucket, style_bucket, horizon_bucket)`.
  - `co_steer.py`: `BlindSpotMiner` (L1 vs L2 routing) + `ExecutableScar` & `ScarRegistry` wired into `SelfCritic`.
  - `council.py`: `HeterogeneousCouncil` enforcing `proposer_family != prosecutor_family` across Anthropic/OpenAI/Google/DeepSeek or deterministic zero-key fallback.
  - `mechanism.py`: L3 `MechanismEvaluator` comparing research harness configurations strictly on out-of-sample forward calibration & false-promotion rate.
  - `engine.py`: `StockRSIEngine` producing Karpathy-style `results.tsv` + Markdown proof reports.

---

*Research & educational software only. Simulated backtests and shadow ledgers do not constitute investment advice.*
