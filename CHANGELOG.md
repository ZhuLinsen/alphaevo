# Changelog

All notable changes to AlphaEvo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Dual-Clock Stock RSI Loop (`alphaevo rsi`)**: `program.md`-driven hypothesis loop
  with heterogeneous Proposer/Prosecutor/Adjudicator council (`rsi/council.py`),
  Keep/Revert ratchet, and Research Maturity + Data Quality gates.
- **Immutable L0 Truth Kernel (`alphaevo truth`)**: SHA-256 hash-locked verification
  manifest covering evaluator, backtest, data-quality, and forward-ledger files.
- **Bitemporal Forward Shadow Ledger (`alphaevo shadow`)**: append-only paper-signal
  seal (`decision_ts` / `knowledge_ts` / `code_commit`) and forward OOS settlement gate.
- **AlphaFactory → Evolution Integration**: Factor discovery pipeline now triggers
  automatically during evolution when conventional reflection yields no improvements.
  LLM generates candidate indicator code → sandboxed execution → statistical
  validation → dynamic registration → strategy injection.
- **Structural Mutation (CHANGE_LOGIC)**: Evolution can now switch entry logic
  between AND/OR. Heuristic fallback triggers when many AND conditions produce
  too few signals.
- **Factor Discovery Mutation (DISCOVER_FACTOR)**: New ChangeType allows the
  evolution loop to inject LLM-discovered factors as strategy conditions.
- **Portfolio-Level Backtesting**: `PortfolioBacktester` simulates execution with
  $100K initial capital, max 5 concurrent positions, 20% position sizing.
  Reports portfolio drawdown, Sharpe ratio, and capital utilization.
- **LLM vs Param_Search Baseline**: After LLM/hybrid evolution, the pipeline
  automatically runs a param_search-only baseline for comparison.
  Shows incremental value of LLM reflection.
- **Experimental Strategy Marking**: Strategies using proxy indicators
  (event_driven_breakout, sector_rotation_leader) now have `experimental: true`
  in their YAML meta. CLI displays ⚠️ warning in `strategy list` and `strategy show`.
- **Demo Synthetic Data Disclaimer**: Demo now clearly states results use synthetic
  data and suggests `alphaevo demo --real` for real market data.
- **Strategy Drafting CLI**: `alphaevo strategy draft` and `alphaevo strategy revise`
  convert plain-language ideas into executable strategy YAML and support iterative edits.
- **One-Shot Research CLI**: `alphaevo strategy research` drafts a plain-language
  strategy, saves it, backtests it, and can run entry-parameter plus exit/risk
  optimization in one command.
- **Research Advice Reports**: one-shot research now writes deterministic next-step
  guidance based on sample size, payoff profile, overfit diagnostics, and exit diagnostics.
- **Strategy Improve CLI**: `alphaevo strategy improve` applies a bounded
  natural-language revision to an existing strategy, backtests it, can run
  parameter/exit optimization, and writes advice.
- **Explicit Entry/Exit Semantics**: DSL adds `entry.triggers`, `entry.guards`, and
  `exit.triggers` so buy signals, hard filters, and sell signals are independently backtestable.
- **Exit/Risk Optimizer**: `alphaevo optimize` searches sell triggers, stop loss,
  take profit, and max holding rules, with optional best-candidate YAML export.
- **Parameter Optimizer**: `alphaevo optimize --spaces entry,params,indicator`
  now searches executable `params.tunable` entry thresholds and indicator windows
  on the same sampled market data.
- **Optimization Gates**: `alphaevo optimize` supports `--objective win_rate`,
  `--min-win-rate`, `--min-avg-return`, `--min-profit-loss-ratio`,
  `--max-drawdown`, `--min-signals`, and `--param-max-changes` so candidates
  can be judged against explicit qualification thresholds such as 50%+ win rate
  without negative expectancy or weak payoff quality.
- **Balanced Quality Objective**: `alphaevo optimize --objective quality`
  ranks candidates by a blended score across win rate, average return,
  profit/loss ratio, drawdown, and signal count.
- **Return-Focused Objective**: `alphaevo optimize --objective profit_quality`
  ranks candidates by average return, total return, payoff ratio, win rate,
  and drawdown for cases where positive but thin returns are not enough.
- **Robust Return Objective**: `alphaevo optimize --objective robust_profit_quality`
  blends return quality with anti-overfit and walk-forward stability scores.
- **Total Return Gate**: `alphaevo optimize` supports `--min-total-return`
  and optimization reports now show total return alongside average return.
- **Fast Optimization Evaluation**: candidate search now supports
  `--evaluation-mode fast` plus `--full-eval-top` so large grids can screen on
  trade metrics first and fully re-evaluate only the leading candidates.
- **Parallel Optimization Search**: `alphaevo optimize --parallel-workers`
  evaluates parameter and exit/risk candidates concurrently for larger
  return-quality and joint-search runs.
- **Robust Optimization Gates**: `alphaevo optimize` can reject candidates
  flagged by anti-overfit checks or by train/validation/test and walk-forward
  performance gaps via `--reject-overfit`, `--max-train-val-gap`,
  `--max-val-test-gap`, `--max-walk-forward-gap`, and
  `--min-walk-forward-pass-rate`.
- **Best Strategy Candidate Summary**: optimization output and reports now show
  the best high-win/high-return candidate as a readable strategy summary,
  including metrics, rule changes, entry/exit rules, and any gate-failure
  reasons.
- **Joint Entry+Exit Search**: `alphaevo optimize --joint-top` now refines the
  leading entry/parameter candidates with exit/risk optimization so buy-side
  filters and sell-side rules can be evaluated together. Joint seeds are
  diversified across win rate, return, total return, and payoff ratio.
- **Trailing Take-Profit Search**: exit optimization now tests trailing
  take-profit candidates alongside fixed reward/risk targets.
- **Breakout/Compression Strategy Template**: Added `breakout_high_Nd` and
  `price_position_Nd` OHLCV indicators plus the
  `volatility_compression_breakout_v1` built-in strategy with explicit breakout
  triggers, volatility-compression guards, and trailing profit capture.
- **Breakout-Aware Strategy Drafting**: `alphaevo strategy draft` now maps
  plain-language breakout/new-high ideas to executable breakout triggers,
  range-position guards, and trailing take-profit rules.
- **Exit Diagnostics**: Optimization reports now summarize MFE/MAE, giveback,
  potentially early exits, late exits, effective stops, and truncated take profits.
- **Backtest Fill Policy**: Added configurable same-candle stop-loss/take-profit
  conflict handling via `backtest.fill_policy` / `--fill-policy`.
- **Daily Stock Analysis Adapter**: Added the `dsa` data adapter path for reusing
  daily_stock_analysis market data through AlphaEvo's adapter boundary.

### Changed
- Upgraded default `llm.model` from `gemini/gemini-2.0-flash` to
  `gemini/gemini-3.8-flash` (plus modern CLI/docs examples for OpenAI GPT-6/5,
  Anthropic Claude 5.5, and DeepSeek V4), and updated `LLMClient` with automatic
  temperature normalization (`1.0`) and `litellm.drop_params = True` for Gemini 3/4
  and reasoning-first models (`gpt-5`/`gpt-6`/`o1`/`o3`/`o4`/`deepseek-v4-pro`).
- Default `alphaevo showcase` now runs a six-step validated mutation chain on
  the bundled yfinance snapshot, improving the reproducible no-key RSI showcase
  from 8.1% to a 68.7% champion while lowering max drawdown to 23.6%.
- `ChangeType` enum extended with `CHANGE_LOGIC` and `DISCOVER_FACTOR`
- `StrategyMeta` model gains `experimental: bool = False` field
- `EvolutionResult` gains `baseline_param_search_score: float | None` field
- Serializer omits `experimental: false` from YAML output to avoid clutter
- LLM reflection prompts now include `change_logic` and `discover_factor` as
  valid change types
- Backtesting now treats `entry.triggers` as the preferred buy signal group and
  falls back to legacy `entry.conditions` for existing strategy YAML.

## [0.1.0] — 2026-03-31

### Added

#### Real-World Validation
- **Real evolve run** with live yfinance data + DeepSeek LLM: `ma_crossover_v1` → `v3` over 3 rounds
  - Confidence: 29.2% → 46.8% (+17.6pp)
  - Win rate: 41.7% → 55.4%, avg return: -0.39% → +1.41%
- **Validation reports** in `reports/real_validation_20260331/`
- **LLM connectivity smoke test** verified against real DeepSeek endpoint
- **Evolution experience store** — persists lessons learned from each evolution round for cross-round and cross-strategy learning
- **Round history context** — LLM reflection now includes previous rounds' changes and outcomes to avoid repeating failed approaches

#### Core Pipeline
- **Strategy DSL v0.3** — YAML-based strategy definition with entry/exit conditions, universe filters, tunable parameters
- **Strategy Parser & Serializer** — bidirectional YAML ↔ Pydantic model conversion
- **Indicator Registry** — 12 MVP indicators computed from OHLCV data, with graceful degradation for unavailable indicators
- **Condition Evaluator** — evaluate entry/exit conditions with support for `and`/`or` logic
- **Backtest Engine** — event-driven backtesting with stop-loss (pct/atr/pct_from_low/composite), take-profit (rr/pct/trailing), and A-share market rules (T+1, limit up/down)
- **Multi-dimensional Evaluator** — win rate, P/L ratio, Sharpe, drawdown, confidence score with anti-overfit penalties
- **Run Pipeline** — end-to-end orchestration: sample → fetch data → backtest → evaluate → report

#### LLM Evolution (requires `alphaevo[llm]`)
- **LLMClient** — lazy litellm wrapper with retry, JSON extraction from markdown fences, separate reflection model support
- **ReflectionAnalyzer** — LLM-driven failure analysis with smart heuristic fallback (handles 5 scenarios: low signals, low win rate, poor P/L, high drawdown, fine-tuning)
- **StrategyMutator** — deterministic strategy mutation with safety guardrails (max 3 changes/round, complexity limit of 8 conditions)
- **StrategyGenerator** — natural language → Strategy creation with auto-retry on parse errors
- **EvolutionPipeline** — multi-round self-improvement loop with 3 methods (llm, param_search, hybrid), early stopping on stagnation/overfitting

#### Data Layer
- **YFinance Adapter** — async data fetching for US/HK/A-share markets with symbol conversion
- **DataManager** — multi-adapter support with fallback chain

#### Strategy Management
- **StrategyStore** — SQLite-based CRUD with evaluation persistence, family queries, leaderboard
- **6 built-in strategy templates** — trend, reversal, event, rotation (A-share) + RSI reversion, MA crossover (US)

#### CLI (`alphaevo` command)
- `alphaevo init` — interactive first-time setup
- `alphaevo strategy create/list/show/import/validate` — full strategy management
- `alphaevo run <id>` — run research loop with Rich progress display
- `alphaevo evolve <id>` — multi-round strategy evolution
- `alphaevo leaderboard` — ranked strategy table
- `alphaevo compare <id1> <id2>` — side-by-side comparison
- `alphaevo tree <id>` — evolution tree visualization
- `alphaevo config show/set` — configuration management
- `alphaevo demo` — **self-evolution showcase** with synthetic data (no API key/network needed)
- `alphaevo version` — version display

#### Infrastructure
- **AppConfig** — unified configuration with priority chain (CLI > env > project > user > defaults)
- **Adaptive Sampler** — stratified stock sampling by market cap
- **Reporter** — JSON + Markdown report generation, multi-strategy comparison tables
- **256 unit tests** with full coverage of all modules
- **GitHub Actions CI** — Python 3.10/3.11/3.12 matrix
- **CONTRIBUTING.md** + **docs/README_CN.md** documentation

### DSL Changelog
- **v0.1** — Initial: meta/description/universe/entry/exit/params
- **v0.2** — `entry.logic` (and/or), indicator-name tunable keys, `market_rules`
- **v0.3** — `entry.execution` (timing + slippage), `meta.preferred_regime`, composite conditions standardized to `StrategyCondition`

---

> ⚠️ **Disclaimer**: This project is for educational and research purposes only. It does not constitute investment advice. Past strategy performance does not guarantee future results.
