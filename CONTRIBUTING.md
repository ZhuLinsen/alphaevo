# Contributing to AlphaEvo

Thank you for your interest in AlphaEvo! This guide will help you get started.

## Prerequisites

- **Python >= 3.10**
- **Git**
- A virtual environment tool (`venv`, `conda`, etc.)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ZhuLinsen/alphaevo.git
cd alphaevo

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify everything works
python -m pytest tests/ -q
```

## Project Structure

```
src/alphaevo/
├── alpha_factory/       # LLM-assisted factor discovery + sandbox
├── backtest/            # Indicator registry, condition evaluator, engine
├── cli/                 # Typer + Rich CLI
├── core/                # Config, LLM client
├── data/                # Data adapters (auto, tencent, yfinance, akshare, dsa)
├── evaluator/           # Multi-dimensional metrics + maturity checks
├── forward/             # Bitemporal forward shadow ledger
├── leaderboard/         # Strategy ranking
├── models/              # Pydantic v2 data models
├── optimizer/           # Entry, parameter, exit/risk optimizers
├── orchestrator/        # End-to-end research & evolution pipelines
├── reflection/          # Failure analysis, strategy mutation, experience/scars
├── research_committee/  # Multi-gate adversarial research review
├── research_log/        # Structured research audit logs
├── rsi/                 # Dual-clock RSI engine, charter parser, heterogeneous council
├── sampler/             # Market-aware sampling
├── strategy/            # Strategy DSL parser, serializer, generator
├── truth/               # Immutable L0 SHA-256 Truth Kernel
├── utils/               # Shared utilities
└── web/                 # Web-ready Pydantic view contracts
examples/                # Showcase data & RSI charter templates (rsi_stock_program.md)
strategies/
└── builtin/             # Built-in strategy YAML templates
tests/
├── unit/                # Unit tests
└── integration/         # Integration tests
```

## Coding Standards

For the full set of rules, see [`AGENTS.md`](AGENTS.md). Key points:

- **Pydantic v2 models** for all public interfaces — no bare `dict`.
- **No hardcoded** secrets, paths, model names, or ports. Use `AppConfig` or environment variables.
- **UTC datetimes** — use `datetime.now(timezone.utc)` for all time-related defaults.
- **Type hints** on all public functions and methods.
- **Respect directory boundaries** — don't put backtest logic in `strategy/`, etc.
- **`__init__.py`** with type exports required for every new module.
- Strategies must have both a **human-readable description** and an **executable DSL**.

## Git & Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add RSI indicator to registry
fix: correct walk-forward window overlap
docs: update CLI command reference
refactor: extract condition evaluator from engine
test: add backtest engine edge cases
chore: bump pydantic to 2.6
```

## Testing

All new features **must** have corresponding tests.

- Unit tests go in `tests/unit/`.
- Integration tests go in `tests/integration/`.

```bash
# Run all tests
python -m pytest tests/ -q

# Run only unit tests
python -m pytest tests/unit/ -q

# Run a specific test file
python -m pytest tests/unit/test_parser.py -v
```

## Linting

```bash
# Check for errors
ruff check src/ tests/ scripts/

# Check formatting
ruff format --check src/

# Type checking
python -m mypy src/alphaevo/ --ignore-missing-imports

# Auto-fix where possible
ruff check src/ tests/ scripts/ --fix
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. Make your changes, following the coding standards above.
3. Add or update tests as needed.
4. Run the full test suite, linter, formatter, and type checker to make sure nothing is broken:
   ```bash
   python -m pytest tests/ -q && ruff check src/ tests/ scripts/ && ruff format --check src/ && python -m mypy src/alphaevo/ --ignore-missing-imports
   ```
5. Submit a PR with the following in the description:
   - **What changed** — summary of the changes.
   - **Why** — motivation or linked issue.
   - **Test results** — paste test output or describe how you verified.
   - **Risk assessment** — anything that might break, edge cases, or known limitations.

### PR Checklist

- [ ] Tests pass (`python -m pytest tests/ -q`)
- [ ] Lint & format pass (`ruff check src/ tests/ scripts/ && ruff format --check src/`)
- [ ] Type check passes (`python -m mypy src/alphaevo/ --ignore-missing-imports`)
- [ ] New public APIs use Pydantic v2 models
- [ ] No hardcoded secrets or paths
- [ ] Documentation updated if applicable

## Contributing Strategies

Strategy YAML files live in `strategies/builtin/`. To contribute a new strategy:

1. Create a YAML file following the [Strategy DSL spec](AGENTS.md#7-策略-dsl-规范-v03).
2. Include complete `meta`, `description`, `universe`, `entry`, `exit`, and `params` sections.
3. Make sure it parses cleanly:
   ```python
   from alphaevo.strategy.dsl.parser import StrategyParser
   strategy = StrategyParser.parse_file("strategies/builtin/your_strategy.yaml")
   print(strategy)
   ```
4. Add a brief comment in the PR explaining the strategy's thesis and target market conditions.

### Strategy Quality Guidelines

- Entry conditions should be specific and testable against OHLCV data.
- Avoid excessive conditions (> 8 total) — complexity is penalized.
- Include tunable parameters with reasonable ranges.
- Add `meta.preferred_regime` if the strategy targets specific market environments.

## Reporting Issues

When filing an issue, please include:

- AlphaEvo version (`alphaevo version`)
- Python version (`python --version`)
- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs or error messages

## Code of Conduct

We are committed to providing a welcoming and respectful environment for everyone.

- Be kind and constructive in discussions and code reviews.
- Respect differing viewpoints and experiences.
- Focus on what is best for the project and community.
- Harassment, trolling, and personal attacks will not be tolerated.

## Questions?

- Open a [GitHub Discussion](https://github.com/ZhuLinsen/alphaevo/discussions) for general questions.
- Open an [Issue](https://github.com/ZhuLinsen/alphaevo/issues) for bugs or feature requests.

---

Thank you for helping make AlphaEvo better! 🧬
