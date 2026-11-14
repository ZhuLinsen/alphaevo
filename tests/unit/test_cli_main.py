"""Tests for CLI command plumbing in cli.main."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from alphaevo.alpha_factory import FactorRecord, FactorStore
from alphaevo.cli.main import app
from alphaevo.core.config import AppConfig, DataConfig
from alphaevo.models.enums import SamplingMethod
from alphaevo.models.execution import EvaluationReport, OverallMetrics
from alphaevo.strategy.draft import StrategyDraftBuilder
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.store import StrategyStore

runner = CliRunner()
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data=DataConfig(
            adapter="yfinance",
            cache_dir=tmp_path / "cache",
        ),
        db_path=tmp_path / "alphaevo.db",
        strategies_dir=tmp_path / "strategies",
    )


def _make_evaluation(strategy_id: str = "test_strat_v1") -> EvaluationReport:
    return EvaluationReport(
        strategy_id=strategy_id,
        overall=OverallMetrics(
            win_rate=0.6,
            avg_return=0.02,
            profit_loss_ratio=1.8,
            max_drawdown=0.1,
            sharpe_ratio=1.2,
            signal_count=42,
            avg_holding_days=5.0,
            total_return=0.18,
        ),
        confidence_score=0.55,
    )


class TestRunCommand:
    def test_run_applies_adapter_override_and_single_start_date(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        captured: list[dict | None] = []

        def fake_get_config(overrides: dict | None = None) -> AppConfig:
            captured.append(overrides)
            return config

        fake_result = SimpleNamespace(
            evaluation=_make_evaluation(),
            report_path=tmp_path / "reports" / "test_strat_v1_report.md",
            batch=SimpleNamespace(
                sampling_method=SamplingMethod.STRATEGY_SCOPED,
                market_regimes=[],
            ),
            backtest_result=SimpleNamespace(signals=[]),
            strategy=SimpleNamespace(
                meta=SimpleNamespace(market=SimpleNamespace(value="us")),
            ),
        )

        with (
            patch("alphaevo.cli.commands.evolution._get_config", side_effect=fake_get_config),
            patch("alphaevo.orchestrator.pipeline.RunPipeline") as MockPipeline,
        ):
            pipeline = MockPipeline.return_value
            pipeline.ensure_builtin_strategies.return_value = 0
            pipeline.run = AsyncMock(return_value=fake_result)

            result = runner.invoke(
                app,
                [
                    "run",
                    "test_strat_v1",
                    "--start",
                    "2025-01-01",
                    "--adapter",
                    "akshare",
                    "--sampling",
                    "strategy_scoped",
                    "--wf-folds",
                    "4",
                    "--wf-train-pct",
                    "0.75",
                    "--wf-pass-gap",
                    "0.08",
                    "--stress-window-days",
                    "15",
                    "--stress-window-top-k",
                    "2",
                    "--output",
                    str(tmp_path / "reports"),
                ],
            )

        assert result.exit_code == 0
        assert captured == [
            {
                "data": {"adapter": "akshare"},
                "backtest": {
                    "walk_forward_folds": 4,
                    "walk_forward_train_pct": 0.75,
                    "walk_forward_pass_gap": 0.08,
                    "stress_window_days": 15,
                    "stress_window_top_k": 2,
                },
            }
        ]
        run_kwargs = pipeline.run.await_args.kwargs
        assert run_kwargs["date_range"][0] == date(2025, 1, 1)
        assert run_kwargs["date_range"][1] >= date(2025, 1, 1)
        assert run_kwargs["sampling_method"].value == "strategy_scoped"

    def test_run_rejects_invalid_date_order(self) -> None:
        result = runner.invoke(
            app,
            [
                "run",
                "test_strat_v1",
                "--start",
                "2025-03-01",
                "--end",
                "2025-01-01",
            ],
        )

        assert result.exit_code == 1
        assert "--start date must be before --end date" in result.stdout


class TestShowcaseCommand:
    def test_showcase_delegates_to_runner(self) -> None:
        with patch("alphaevo.cli.demo.run_showcase") as mock_run_showcase:
            result = runner.invoke(app, ["showcase", "--output", "tmp_showcase", "--write-docs"])

        assert result.exit_code == 0
        mock_run_showcase.assert_called_once()
        assert mock_run_showcase.call_args.kwargs["output_dir"] == "tmp_showcase"
        assert mock_run_showcase.call_args.kwargs["write_docs"] is True

    def test_demo_defaults_to_showcase_snapshot(self) -> None:
        with patch("alphaevo.cli.demo.run_showcase") as mock_run_showcase:
            result = runner.invoke(app, ["demo"])

        assert result.exit_code == 0
        mock_run_showcase.assert_called_once()


class TestEvolveCommand:
    def test_evolve_applies_runtime_overrides(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        captured: list[dict | None] = []

        def fake_get_config(overrides: dict | None = None) -> AppConfig:
            captured.append(overrides)
            return config

        round_result = SimpleNamespace(
            round_num=1,
            strategy=SimpleNamespace(meta=SimpleNamespace(id="test_strat_v1")),
            evaluation=_make_evaluation(),
            reflection=None,
            improved=False,
        )
        evolve_result = SimpleNamespace(
            original_strategy_id="test_strat_v1",
            rounds=[round_result],
            champion_id="test_strat_v1",
            champion_score=0.55,
            improvement=0.0,
            total_rounds=1,
            velocity=[],
            efficiency=0.0,
            early_stopped=False,
            stop_reason="",
        )

        with (
            patch("alphaevo.cli.commands.evolution._get_config", side_effect=fake_get_config),
            patch("alphaevo.orchestrator.evolution.EvolutionPipeline") as MockPipeline,
            patch(
                "alphaevo.evaluator.reporter.Reporter.evolution_report",
                return_value="# Evolution",
            ),
            patch(
                "alphaevo.evaluator.reporter.Reporter.research_report",
                return_value="# Research Report",
            ),
            patch(
                "alphaevo.evaluator.reporter.Reporter.llm_evidence_report",
                return_value="# LLM Evidence",
            ),
        ):
            pipeline = MockPipeline.return_value
            pipeline.evolve.return_value = evolve_result
            pipeline.research_log.to_markdown.return_value = "# Research Log"

            result = runner.invoke(
                app,
                [
                    "evolve",
                    "test_strat_v1",
                    "--adapter",
                    "akshare",
                    "--model",
                    "openai/gpt-5-mini",
                    "--reflect-model",
                    "openai/gpt-6.1-sol",
                    "--sampling",
                    "regime_based",
                    "--wf-folds",
                    "5",
                    "--wf-train-pct",
                    "0.8",
                    "--wf-pass-gap",
                    "0.06",
                    "--stress-window-days",
                    "18",
                    "--stress-window-top-k",
                    "4",
                    "--output",
                    str(tmp_path / "reports"),
                ],
            )

        assert result.exit_code == 0
        assert captured == [
            {
                "data": {"adapter": "akshare"},
                "llm": {
                    "model": "openai/gpt-5-mini",
                    "reflect_model": "openai/gpt-6.1-sol",
                },
                "backtest": {
                    "walk_forward_folds": 5,
                    "walk_forward_train_pct": 0.8,
                    "walk_forward_pass_gap": 0.06,
                    "stress_window_days": 18,
                    "stress_window_top_k": 4,
                },
            }
        ]
        assert pipeline.evolve.called
        assert pipeline.evolve.call_args.kwargs["sampling_method"].value == "regime_based"

    def test_evolve_writes_output_files(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        round_result = SimpleNamespace(
            round_num=1,
            strategy=SimpleNamespace(meta=SimpleNamespace(id="test_strat_v1")),
            evaluation=_make_evaluation(),
            reflection=None,
            improved=False,
        )
        evolve_result = SimpleNamespace(
            original_strategy_id="test_strat_v1",
            rounds=[round_result],
            champion_id="test_strat_v1",
            champion_score=0.55,
            improvement=0.0,
            total_rounds=1,
            velocity=[],
            efficiency=0.0,
            early_stopped=False,
            stop_reason="",
        )

        def fake_to_file(*args) -> None:
            path = args[-1]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("meta:\n  id: test_strat_v1\n", encoding="utf-8")

        with (
            patch("alphaevo.cli.commands.evolution._get_config", return_value=config),
            patch("alphaevo.orchestrator.evolution.EvolutionPipeline") as MockPipeline,
            patch(
                "alphaevo.evaluator.reporter.Reporter.evolution_report",
                return_value="# Evolution",
            ),
            patch(
                "alphaevo.evaluator.reporter.Reporter.research_report",
                return_value="# Research Report",
            ),
            patch(
                "alphaevo.evaluator.reporter.Reporter.llm_evidence_report",
                return_value="# LLM Evidence",
            ),
            patch(
                "alphaevo.strategy.dsl.serializer.StrategySerializer.to_file",
                side_effect=fake_to_file,
            ),
        ):
            pipeline = MockPipeline.return_value
            pipeline.evolve.return_value = evolve_result
            pipeline.research_log.to_markdown.return_value = "# Research Log"

            result = runner.invoke(
                app,
                [
                    "evolve",
                    "test_strat_v1",
                    "--output",
                    str(tmp_path / "artifacts"),
                ],
            )

        assert result.exit_code == 0
        assert (tmp_path / "artifacts" / "test_strat_v1_evolution.md").read_text(
            encoding="utf-8"
        ) == "# Evolution"
        assert (tmp_path / "artifacts" / "test_strat_v1_research_report.md").read_text(
            encoding="utf-8"
        ) == "# Research Report"
        assert (tmp_path / "artifacts" / "test_strat_v1_llm_evidence.md").read_text(
            encoding="utf-8"
        ) == "# LLM Evidence"
        assert (tmp_path / "artifacts" / "test_strat_v1_research_log.md").read_text(
            encoding="utf-8"
        ) == "# Research Log"
        assert (tmp_path / "artifacts" / "test_strat_v1_champion.yaml").exists()
        assert (tmp_path / "artifacts" / "test_strat_v1_strategies" / "test_strat_v1.yaml").exists()

    def test_evolve_fails_cleanly_when_no_rounds_recorded(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        evolve_result = SimpleNamespace(
            original_strategy_id="test_strat_v1",
            rounds=[],
            champion_id="test_strat_v1",
            champion_score=0.0,
            improvement=0.0,
            total_rounds=0,
            velocity=[],
            efficiency=0.0,
            early_stopped=True,
            stop_reason="Backtest failed: cannot convert NaN to integer ratio",
        )

        with (
            patch("alphaevo.cli.commands.evolution._get_config", return_value=config),
            patch("alphaevo.orchestrator.evolution.EvolutionPipeline") as MockPipeline,
            patch(
                "alphaevo.evaluator.reporter.Reporter.evolution_report",
                return_value="# Evolution",
            ),
            patch(
                "alphaevo.evaluator.reporter.Reporter.research_report",
                return_value="# Research Report",
            ),
            patch(
                "alphaevo.evaluator.reporter.Reporter.llm_evidence_report",
                return_value="# LLM Evidence",
            ),
        ):
            pipeline = MockPipeline.return_value
            pipeline.evolve.return_value = evolve_result
            pipeline.research_log.to_markdown.return_value = "# Research Log"

            result = runner.invoke(
                app,
                [
                    "evolve",
                    "test_strat_v1",
                    "--output",
                    str(tmp_path / "artifacts"),
                ],
            )

        assert result.exit_code == 1
        assert "No successful evolution rounds were recorded" in result.stdout
        assert "Early stopped" in result.stdout
        assert "No strategy YAML artifacts were written" in result.stdout
        assert not (tmp_path / "artifacts" / "test_strat_v1_champion.yaml").exists()


class TestCompareCommand:
    def test_compare_shows_strategy_differences(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        store = StrategyStore(config.db_path)
        parser = StrategyParser()

        s1 = parser.parse_file(Path("strategies/builtin/trend_pullback_rebound.yaml"))
        s2 = s1.model_copy(deep=True)
        s2.meta.id = "trend_pullback_rebound_v3"
        s2.meta.version = 3
        s2.meta.parent_id = s1.meta.id
        s2.entry.conditions[0].value = 0.05
        s2.exit.stop_loss.value = 0.032

        store.save(s1)
        store.save(s2)
        store.save_evaluation(_make_evaluation(s1.meta.id))
        store.save_evaluation(_make_evaluation(s2.meta.id))

        with patch("alphaevo.cli.commands.analysis._get_store", return_value=store):
            result = runner.invoke(app, ["compare", s1.meta.id, s2.meta.id])

        assert result.exit_code == 0
        assert "Strategy Differences" in result.stdout
        assert "entry.conditions[0].value" in result.stdout
        assert "0.08" in result.stdout
        assert "0.05" in result.stdout
        assert "exit.stop_loss.value" in result.stdout
        assert "0.04" in result.stdout
        assert "0.032" in result.stdout


class TestTreeCommand:
    def test_tree_renders_ascii_hierarchy_with_champion_and_changes(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        store = StrategyStore(config.db_path)
        parser = StrategyParser()

        root = parser.parse_file(Path("strategies/builtin/trend_pullback_rebound.yaml"))

        child_a = root.model_copy(deep=True)
        child_a.meta.id = "trend_pullback_rebound_v2"
        child_a.meta.version = 2
        child_a.meta.parent_id = root.meta.id
        child_a.entry.conditions[0].value = 0.05

        child_b = root.model_copy(deep=True)
        child_b.meta.id = "trend_pullback_rebound_v3"
        child_b.meta.version = 3
        child_b.meta.parent_id = root.meta.id
        child_b.exit.stop_loss.value = 0.032

        store.save(root)
        store.save(child_a)
        store.save(child_b)

        store.save_evaluation(_make_evaluation(root.meta.id))
        store.save_evaluation(_make_evaluation(child_a.meta.id))
        store.save_evaluation(
            _make_evaluation(child_b.meta.id).model_copy(update={"confidence_score": 0.72})
        )

        with patch("alphaevo.cli.commands.analysis._get_store", return_value=store):
            result = runner.invoke(app, ["tree", root.meta.id])

        assert result.exit_code == 0
        assert "Evolution Tree" in result.stdout
        assert "trend_pullback_rebound" in result.stdout
        assert "trend_pullback_rebound_v3 | score=72.0% | champion" in result.stdout
        assert "changes=" in result.stdout
        assert "├──" in result.stdout or "└──" in result.stdout


class TestStrategyDiffCommand:
    def test_strategy_diff_shows_structural_and_yaml_diff(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        store = StrategyStore(config.db_path)
        parser = StrategyParser()

        s1 = parser.parse_file(Path("strategies/builtin/trend_pullback_rebound.yaml"))
        s2 = s1.model_copy(deep=True)
        s2.meta.id = "trend_pullback_rebound_v3"
        s2.meta.version = 3
        s2.meta.parent_id = s1.meta.id
        s2.entry.conditions[0].value = 0.05
        s2.exit.stop_loss.value = 0.032

        store.save(s1)
        store.save(s2)

        with patch("alphaevo.cli.commands.strategy._get_store", return_value=store):
            result = runner.invoke(app, ["strategy", "diff", s1.meta.id, s2.meta.id])

        assert result.exit_code == 0
        assert "Strategy Diff" in result.stdout
        assert "Changed Fields" in result.stdout
        assert "entry.conditions[0].value" in result.stdout
        assert "0.08 -> 0.05" in result.stdout
        assert "exit.stop_loss.value" in result.stdout
        assert "0.04 -> 0.032" in result.stdout
        assert "--- trend_pullback_rebound_v1" in result.stdout
        assert "+++ trend_pullback_rebound_v3" in result.stdout


class TestStrategyShowCommand:
    def test_strategy_show_preserves_tunable_selectors(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        store = StrategyStore(config.db_path)
        strategy = StrategyParser().parse_file(
            Path("strategies/builtin/trend_pullback_rebound.yaml")
        )
        store.save(strategy)

        with patch("alphaevo.cli.commands.strategy._get_store", return_value=store):
            result = runner.invoke(app, ["strategy", "show", strategy.meta.id])

        assert result.exit_code == 0
        assert "entry.conditions[indicator=relative_strength_20d].value" in result.stdout
        assert "entry.conditions[indicator=relative_strength_20d].indicator" in result.stdout

    def test_strategy_create_handles_llm_runtime_errors(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)

        with (
            patch("alphaevo.cli.commands.strategy._get_config", return_value=config),
            patch("alphaevo.core.llm.LLMClient.from_config", return_value=MagicMock()),
            patch(
                "alphaevo.strategy.generator.StrategyGenerator.generate",
                side_effect=RuntimeError("Temporary failure in name resolution"),
            ),
        ):
            result = runner.invoke(
                app,
                ["strategy", "create", "--market", "us"],
                input="Simple bullish momentum strategy\n",
            )

        assert result.exit_code == 1
        assert "Generation failed: Temporary failure in name resolution" in result.stdout


class TestStrategyResearchCommand:
    def test_strategy_research_drafts_saves_and_runs(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        fake_result = SimpleNamespace(
            evaluation=_make_evaluation("rsi_10_3_5_v1"),
            report_path=tmp_path / "reports" / "rsi_10_3_5_v1_report.md",
            batch=SimpleNamespace(),
            backtest_result=SimpleNamespace(signals=[]),
            strategy=SimpleNamespace(),
            _data={},
            _contexts={},
        )

        with (
            patch("alphaevo.cli.commands.strategy._get_config", return_value=config),
            patch("alphaevo.orchestrator.pipeline.RunPipeline") as MockPipeline,
        ):
            pipeline = MockPipeline.return_value
            pipeline.ensure_builtin_strategies.return_value = 0
            pipeline.run = AsyncMock(return_value=fake_result)

            result = runner.invoke(
                app,
                [
                    "strategy",
                    "research",
                    "RSI超跌反转，跌破10日线卖出，止损3%，持有5天",
                    "--market",
                    "us",
                    "--samples",
                    "7",
                    "--no-optimize-exits",
                    "--no-optimize-params",
                    "--output",
                    str(tmp_path / "research"),
                ],
            )

        assert result.exit_code == 0
        assert "Drafted and saved strategy" in result.stdout
        assert "Research Summary" in result.stdout
        assert pipeline.run.await_args.args[0] == "rsi_10_3_5_v1"
        assert pipeline.run.await_args.kwargs["max_symbols"] == 7

        saved = StrategyStore(config.db_path).get("rsi_10_3_5_v1")
        assert saved is not None
        assert saved.entry.triggers
        assert [condition.indicator for condition in saved.exit.triggers] == ["close_below_ma10"]
        research_dir = tmp_path / "research" / "rsi_10_3_5_v1_research"
        assert (research_dir / "rsi_10_3_5_v1.yaml").exists()
        assert (research_dir / "rsi_10_3_5_v1_research_advice.md").exists()

    def test_strategy_research_help(self) -> None:
        result = runner.invoke(app, ["strategy", "research", "--help"])
        help_text = ANSI_RE.sub("", result.stdout)

        assert result.exit_code == 0
        assert "Draft, save, backtest" in help_text
        assert "optimize-exits" in help_text
        assert "optimize-params" in help_text


class TestStrategyImproveCommand:
    def test_strategy_improve_revises_saves_and_runs(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        store = StrategyStore(config.db_path)
        base = StrategyDraftBuilder().from_text(
            "RSI超跌反转，跌破10日线卖出，止损3%，持有5天",
            market="us",
            strategy_id="rsi_custom_v1",
        )
        store.save(base)
        store.save_evaluation(_make_evaluation(base.meta.id))

        fake_result = SimpleNamespace(
            evaluation=_make_evaluation("rsi_custom_v2").model_copy(
                update={"confidence_score": 0.62}
            ),
            report_path=tmp_path / "reports" / "rsi_custom_v2_report.md",
            batch=SimpleNamespace(),
            backtest_result=SimpleNamespace(signals=[]),
            strategy=SimpleNamespace(),
            _data={},
            _contexts={},
        )

        with (
            patch("alphaevo.cli.commands.strategy._get_config", return_value=config),
            patch("alphaevo.orchestrator.pipeline.RunPipeline") as MockPipeline,
        ):
            pipeline = MockPipeline.return_value
            pipeline.ensure_builtin_strategies.return_value = 0
            pipeline.run = AsyncMock(return_value=fake_result)

            result = runner.invoke(
                app,
                [
                    "strategy",
                    "improve",
                    "rsi_custom_v1",
                    "减少交易次数，降低回撤，右侧确认",
                    "--samples",
                    "9",
                    "--output",
                    str(tmp_path / "improve"),
                ],
            )

        assert result.exit_code == 0
        assert "Saved revised strategy: rsi_custom_v2" in result.stdout
        assert "Score delta vs stored parent best" in result.stdout
        assert pipeline.run.await_args.args[0] == "rsi_custom_v2"
        assert pipeline.run.await_args.kwargs["max_symbols"] == 9

        saved = StrategyStore(config.db_path).get("rsi_custom_v2")
        assert saved is not None
        assert saved.meta.parent_id == "rsi_custom_v1"
        assert saved.entry.execution is not None
        assert saved.entry.execution.timing == "breakout_high"
        improve_dir = tmp_path / "improve" / "rsi_custom_v2_improve"
        assert (improve_dir / "rsi_custom_v2.yaml").exists()
        assert (improve_dir / "rsi_custom_v2_research_advice.md").exists()

    def test_strategy_improve_missing_strategy(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)

        with patch("alphaevo.cli.commands.strategy._get_config", return_value=config):
            result = runner.invoke(
                app,
                ["strategy", "improve", "missing_v1", "降低回撤"],
            )

        assert result.exit_code == 1
        assert "Strategy not found" in result.stdout


class TestInitCommand:
    def test_init_initializes_database(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)

        with (
            patch("alphaevo.core.config.ConfigManager.load", return_value=config),
            patch("alphaevo.core.config.ConfigManager.ensure_dirs") as mock_ensure_dirs,
            patch("alphaevo.core.config.ConfigManager.save_user_config") as mock_save,
            patch("alphaevo.cli.main.typer.prompt", side_effect=["yfinance", ""]),
        ):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert config.db_path.exists()
        mock_ensure_dirs.assert_called_once_with(config)
        mock_save.assert_called_once_with(config)


class TestStrategyValidateCommand:
    def test_validate_fails_on_semantic_errors(self, tmp_path: Path) -> None:
        strategy_file = tmp_path / "bad_strategy.yaml"
        strategy_file.write_text(
            """
meta:
  id: bad_v1
  name: Bad
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: no_such_indicator
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["strategy", "validate", str(strategy_file)])

        assert result.exit_code == 1
        assert "Unknown indicator: no_such_indicator" in result.stdout

    def test_validate_strict_fails_on_warnings(self, tmp_path: Path) -> None:
        strategy_file = tmp_path / "warn_strategy.yaml"
        strategy_file.write_text(
            """
meta:
  id: warn_v1
  name: Warn
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
  max_holding_days: 120
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["strategy", "validate", str(strategy_file), "--strict"])

        assert result.exit_code == 1
        assert "unusually long" in result.stdout


class TestFactorCommands:
    def test_factor_list_show_and_retire(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        store = FactorStore(config.db_path)
        try:
            store.save(
                FactorRecord(
                    name="alpha_momentum_test",
                    description="Momentum factor",
                    rationale="Trend continuation",
                    code="def compute(df, idx): return float(df['close'].iloc[idx])",
                    ic_mean=0.031,
                    ir=0.62,
                    monthly_win_rate=0.67,
                )
            )
        finally:
            store.close()

        with patch("alphaevo.cli.commands.factor._get_config", return_value=config):
            list_result = runner.invoke(app, ["factor", "list"])
            show_result = runner.invoke(app, ["factor", "show", "alpha_momentum_test"])
            retire_result = runner.invoke(app, ["factor", "retire", "alpha_momentum_test"])

        assert list_result.exit_code == 0
        assert "alpha_momentum_test" in list_result.stdout
        assert show_result.exit_code == 0
        assert "Momentum factor" in show_result.stdout
        assert retire_result.exit_code == 0

        store = FactorStore(config.db_path)
        try:
            record = store.get("alpha_momentum_test")
            assert record is not None
            assert record.status == "retired"
        finally:
            store.close()

    def test_factor_discover_writes_report(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=60, freq="D"),
                "open": [100.0 + i for i in range(60)],
                "high": [101.0 + i for i in range(60)],
                "low": [99.0 + i for i in range(60)],
                "close": [100.5 + i for i in range(60)],
                "volume": [1_000_000] * 60,
            }
        )
        fake_result = SimpleNamespace(
            proposed=[SimpleNamespace(name="alpha_volume_spike")],
            sandbox_passed=["alpha_volume_spike"],
            sandbox_failed=[],
            validation_passed=["alpha_volume_spike"],
            validation_failed=[],
            registered=["alpha_volume_spike"],
        )

        with (
            patch("alphaevo.cli.commands.factor._get_config", return_value=config),
            patch(
                "alphaevo.cli.commands.factor._load_factor_history", new=AsyncMock(return_value=df)
            ),
            patch("alphaevo.core.llm.LLMClient.from_config", return_value=MagicMock()),
            patch("alphaevo.alpha_factory.load_registered_factors"),
            patch("alphaevo.alpha_factory.AlphaFactory") as MockFactory,
        ):
            factory = MockFactory.return_value
            factory.discover = AsyncMock(return_value=fake_result)

            result = runner.invoke(
                app,
                [
                    "factor",
                    "discover",
                    "AAPL",
                    "--context",
                    "find false breakout filters",
                    "--output",
                    str(tmp_path / "factor_report.md"),
                ],
            )

        assert result.exit_code == 0
        assert (tmp_path / "factor_report.md").exists()
        report_text = (tmp_path / "factor_report.md").read_text(encoding="utf-8")
        assert "alpha_volume_spike" in report_text
        assert "## Summary" in report_text
        assert "find false breakout filters" in report_text
