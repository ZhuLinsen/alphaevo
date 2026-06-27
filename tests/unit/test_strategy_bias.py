"""Tests for strategy lookahead/repainting bias checks."""

from pathlib import Path

from typer.testing import CliRunner

from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.cli.main import app
from alphaevo.strategy.bias import analyze_strategy_bias
from alphaevo.strategy.dsl.parser import StrategyParser

runner = CliRunner()


def _parse_strategy(yaml_text: str):
    return StrategyParser().parse_yaml(yaml_text)


def test_bias_checker_flags_future_looking_indicator() -> None:
    IndicatorRegistry.register_dynamic("future_return_5d", lambda df, idx, ctx=None: 0.0)
    try:
        strategy = _parse_strategy(
            """
meta:
  id: biased_v1
  name: Biased
  version: 1
  category: trend
  market: a_share
description: test
entry:
  triggers:
    - indicator: future_return_5d
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
"""
        )
        report = analyze_strategy_bias(strategy)
    finally:
        IndicatorRegistry.unregister_dynamic("future_return_5d")

    assert report.lookahead_checked is True
    assert report.risk_level == "high"
    assert report.has_errors is True
    assert report.findings[0].category == "future_indicator"
    assert report.findings[0].location == "entry.triggers[0].indicator"


def test_bias_checker_warns_on_same_bar_close_and_notes_event_alignment() -> None:
    strategy = _parse_strategy(
        """
meta:
  id: event_close_v1
  name: Event Close
  version: 1
  category: event
  market: a_share
description: test
entry:
  execution:
    timing: close
  triggers:
    - indicator: negative_news_score
      op: "<"
      value: 0.4
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
"""
    )

    report = analyze_strategy_bias(strategy)

    assert report.risk_level == "medium"
    categories = {finding.category for finding in report.findings}
    assert "same_bar_execution" in categories
    assert "event_effective_date" in categories


def test_strategy_validate_bias_check_exits_nonzero_for_future_indicator(tmp_path: Path) -> None:
    IndicatorRegistry.register_dynamic("future_return_5d", lambda df, idx, ctx=None: 0.0)
    try:
        strategy_file = tmp_path / "biased.yaml"
        strategy_file.write_text(
            """
meta:
  id: biased_cli_v1
  name: Biased CLI
  version: 1
  category: trend
  market: a_share
description: test
entry:
  conditions:
    - indicator: future_return_5d
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["strategy", "validate", str(strategy_file), "--bias-check"])
    finally:
        IndicatorRegistry.unregister_dynamic("future_return_5d")

    assert result.exit_code == 1
    assert "Bias risk:" in result.stdout
    assert "future_indicator" in result.stdout


def test_strategy_validate_bias_check_keeps_clean_builtin_valid() -> None:
    strategy_file = Path("strategies/builtin/ma_crossover.yaml")

    result = runner.invoke(app, ["strategy", "validate", str(strategy_file), "--bias-check"])

    assert result.exit_code == 0
    assert "Bias risk:" in result.stdout
    assert "low" in result.stdout
