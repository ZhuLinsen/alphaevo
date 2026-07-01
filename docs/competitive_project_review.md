# Competitive Project Review

This note records the GitHub projects reviewed while tightening AlphaEvo's
strategy optimization and self-evolution workflow. Star counts and descriptions
were checked through the GitHub API on 2026-06-30.

AlphaEvo should not try to become a copy of any single framework. The useful
lesson is the operating discipline shared by mature quant projects: explicit
baselines, robust validation, data provenance, reproducible optimization
objectives, and visible promotion gates.

## Projects Reviewed

| Project | Focus | Observed Practice | AlphaEvo Takeaway |
|---------|-------|-------------------|-------------------|
| [microsoft/qlib](https://github.com/microsoft/qlib) | AI-oriented quant research platform | Separates research workflow, data layer, model workflow, and production-oriented validation. | Keep data quality and validation protocol as first-class gates before strategy promotion. |
| [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Financial reinforcement learning | Treats trading as an environment/task protocol with repeated train/test evaluation. | Keep evolution decisions tied to explicit research protocol, not only latest score. |
| [TradeMaster-NTU/TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster) | RL-powered quantitative trading platform | Emphasizes task setup, agents, datasets, and evaluation modules. | Surface whether a strategy run is ready for further optimization or still missing evidence. |
| [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | Fast vectorized backtesting | Encourages running many ideas and comparing parameter spaces quickly. | Reports should make candidate/baseline readiness visible without hiding behind a single score. |
| [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | Trading bot and hyperopt workflow | Uses explicit optimization objectives, constraints, and operational guardrails. | Keep robust objectives and gate reasons visible before accepting optimized candidates. |
| [mementum/backtrader](https://github.com/mementum/backtrader) | Backtesting engine | Mature strategy/backtest separation with benchmark-style evaluation patterns. | Keep executable DSL separate from evaluation/reporting concerns. |
| [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | Multi-agent investment analysis | Makes analyst roles explicit for explainability. | Continue exposing research committee and data-quality auditor outputs in reports. |
| [OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB) | Financial data platform for analysts, quants, and AI agents | Prioritizes broad data access and agent-facing data workflows. | Data provenance and provider coverage must be visible before news/event strategy mutation. |
| [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) | Portfolio analytics and tear sheets | Focuses on translating returns into risk/analytics reports. | Reports should not only list checks; they should recommend the next research action. |
| [quantopian/pyfolio](https://github.com/quantopian/pyfolio) | Portfolio and risk analytics | Popularized tear-sheet style risk diagnostics for strategies. | Maturity reporting should elevate risk/protocol interpretation above raw metrics. |
| [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | Algorithmic trading engine | Production-oriented engine with explicit algorithm lifecycle and validation concerns. | AlphaEvo should keep promotion/validation gates explicit before a strategy leaves research mode. |
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | Python quant trading framework | Practical trading framework with operational workflow emphasis. | AlphaEvo reports should guide the operator toward the next concrete command. |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | Deterministic event-driven trading engine | Emphasizes deterministic architecture and production-grade workflow discipline. | Keep deterministic next-action heuristics separate from LLM reflection. |

## Implemented In This PR

- Added a reusable research maturity checklist in `alphaevo.evaluator.maturity`.
- Rendered the checklist in Markdown reports so every run shows promotion
  readiness instead of only performance metrics.
- The checklist covers sample evidence, baseline protocol, robustness protocol,
  data quality, strategy complexity, and optimization readiness.
- Added a structured recommended next action for the maturity report, mapping
  failed/watch gates to concrete commands such as expanding samples, repairing
  data coverage, simplifying the DSL, running walk-forward validation, or
  launching robust optimization.
- Exposed the same maturity status, gate list, and recommended next action in
  web-facing evaluation summaries so future dashboards can drive the same
  research loop as Markdown reports.
- The maturity checks complement the existing data-quality gate: proxy-dominant
  event/news context now blocks promotion readiness and strategy optimization
  readiness.

## Follow-Up Candidates

- Add Pareto-frontier candidate reporting for `optimize`, inspired by fast
  parameter-space exploration tools.
- Persist maturity checklist artifacts into run manifests once the full
  artifact/provenance roadmap lands.
- Add official/local leaderboard filters for maturity status so weak-protocol
  local runs cannot appear as showcase winners.
