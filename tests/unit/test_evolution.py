"""Tests for EvolutionPipeline — multi-round evolution with mocked components."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from alphaevo.core.config import AppConfig
from alphaevo.data.quality import build_data_quality_report
from alphaevo.models.enums import ChangeType, EvolutionMethod, MarketRegime
from alphaevo.models.execution import (
    AntiFitMetrics,
    BacktestResult,
    CandidateExperiment,
    EvaluationReport,
    EventContextMetrics,
    OverallMetrics,
    ReflectionResult,
    RegimeMetrics,
    ResearchHypothesis,
    SampleBatch,
    StrategyChange,
)
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    TunableParam,
    UniverseConfig,
)
from alphaevo.orchestrator.evolution import (
    EvolutionPipeline,
    EvolutionResult,
    _ScreenedMutationCandidate,
)
from alphaevo.orchestrator.pipeline import RunResult


def _make_strategy(strategy_id: str = "test_v1", version: int = 1) -> Strategy:
    return Strategy(
        meta=StrategyMeta(id=strategy_id, name="Test", version=version),
        description="Test",
        universe=UniverseConfig(market=["us"]),
        entry=StrategyEntry(
            conditions=[
                StrategyCondition(indicator="rsi_14", op="<", value=30),
            ],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
        params=StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=rsi_14].value",
                    range=[20.0, 40.0],
                    step=5.0,
                ),
            ]
        ),
    )


def _make_evaluation(
    strategy_id: str = "test_v1",
    confidence: float = 0.4,
    win_rate: float = 0.45,
    overfit: bool = False,
    signal_count: int = 50,
) -> EvaluationReport:
    return EvaluationReport(
        evaluation_id=f"eval-{strategy_id}",
        strategy_id=strategy_id,
        overall=OverallMetrics(
            win_rate=win_rate,
            avg_return=0.01,
            profit_loss_ratio=1.5,
            max_drawdown=0.10,
            sharpe_ratio=0.8,
            signal_count=signal_count,
        ),
        confidence_score=confidence,
        anti_overfit=AntiFitMetrics(
            train_val_gap=0.20 if overfit else 0.05,
        ),
    )


def _make_run_result(
    strategy_id: str = "test_v1",
    confidence: float = 0.4,
    version: int = 1,
    overfit: bool = False,
    signal_count: int = 50,
    insufficient_signals: bool = False,
) -> RunResult:
    strat = _make_strategy(strategy_id, version=version)
    ev = _make_evaluation(
        strategy_id,
        confidence,
        overfit=overfit,
        signal_count=signal_count,
    )
    batch = SampleBatch(
        batch_id="batch-test",
        strategy_id=strategy_id,
        symbols=["AAPL", "GOOG"],
        date_range=(date(2024, 1, 1), date(2024, 12, 31)),
        signal_count_target=30,
        signal_count_reached=signal_count,
        insufficient_signals=insufficient_signals,
    )
    bt = BacktestResult(
        strategy_id=strategy_id,
        batch_id="batch-test",
        signals=[],
    )
    return RunResult(
        strategy=strat,
        batch=batch,
        backtest_result=bt,
        evaluation=ev,
    )


def _make_custom_run_result(
    strategy: Strategy,
    evaluation: EvaluationReport,
) -> RunResult:
    batch = SampleBatch(
        batch_id="batch-test",
        strategy_id=strategy.meta.id,
        symbols=["AAPL", "GOOG"],
        date_range=(date(2024, 1, 1), date(2024, 12, 31)),
        signal_count_target=30,
        signal_count_reached=evaluation.overall.signal_count,
        insufficient_signals=False,
    )
    bt = BacktestResult(
        strategy_id=strategy.meta.id,
        batch_id="batch-test",
        signals=[],
    )
    return RunResult(
        strategy=strategy,
        batch=batch,
        backtest_result=bt,
        evaluation=evaluation,
    )


@pytest.fixture
def config():
    return AppConfig()


class TestEvolutionPipeline:
    def test_single_round_no_improvement(self, config):
        """Single round should work without LLM."""
        pipeline = EvolutionPipeline(config)

        async def mock_run(*args, **kwargs):
            return _make_run_result(confidence=0.4)

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=1,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        assert result.original_strategy_id == "test_v1"
        assert len(result.rounds) == 1
        assert result.champion_score == 0.4

    def test_improvement_across_rounds(self, config):
        """Score improves across rounds."""
        pipeline = EvolutionPipeline(config)

        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            score = 0.3 + call_count[0] * 0.1
            sid = f"test_v{call_count[0]}"
            return _make_run_result(sid, confidence=score, version=call_count[0])

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=3,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        assert len(result.rounds) == 3
        assert result.champion_score > 0.4
        assert result.improvement > 0

    def test_early_stop_no_improvement(self, config):
        """Stop after 2 rounds without improvement."""
        pipeline = EvolutionPipeline(config)

        async def mock_run(*args, **kwargs):
            return _make_run_result(confidence=0.3)

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=7,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        assert result.early_stopped
        assert "no improvement" in result.stop_reason.lower()
        assert len(result.rounds) < 7

    def test_early_stop_overfit(self, config):
        """Stop when overfitting detected."""
        pipeline = EvolutionPipeline(config)

        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            score = 0.3 + call_count[0] * 0.1
            return _make_run_result(
                f"test_v{call_count[0]}",
                confidence=score,
                version=call_count[0],
                overfit=(call_count[0] >= 2),
            )

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=5,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        assert result.early_stopped
        assert "overfit" in result.stop_reason.lower()

    def test_overfit_baseline_still_sets_champion_score(self, config):
        """Round-1 baseline should remain the champion even if flagged as overfit."""
        pipeline = EvolutionPipeline(config)

        async def mock_run(*args, **kwargs):
            return _make_run_result(confidence=0.24, overfit=True)

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=1,
            method=EvolutionMethod.HYBRID,
        )

        assert result.champion_id == "test_v1"
        assert result.champion_score == 0.24

    def test_param_search_reflection(self, config):
        """Param search generates changes from tunable params."""
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.35)

        result = pipeline._param_search_reflection(strategy, evaluation)

        assert result is not None
        assert len(result.proposed_changes) >= 1
        assert result.proposed_changes[0].change_type in (
            ChangeType.TIGHTEN_FILTER,
            ChangeType.LOOSEN_FILTER,
        )

    def test_early_stop_when_signals_remain_insufficient(self, config):
        """Sparse-signal runs should stop before reflection/mutation continues."""
        config.evolution.min_signal_count = 30
        pipeline = EvolutionPipeline(config)

        async def mock_run(*args, **kwargs):
            return _make_run_result(
                confidence=0.18,
                signal_count=9,
                insufficient_signals=True,
            )

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=3,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        assert result.early_stopped is True
        assert "Insufficient signals" in result.stop_reason
        assert len(result.rounds) == 1
        assert result.rounds[0].batch is not None
        assert result.rounds[0].batch.insufficient_signals is True

    def test_data_quality_gate_blocks_proxy_dominant_strategy_iteration(self, config):
        """Proxy-dominant event context should stop mutation before threshold tuning."""
        pipeline = EvolutionPipeline(config)

        async def mock_run(*args, **kwargs):
            run_result = _make_run_result(confidence=0.32, signal_count=60)
            run_result.evaluation.event_context = EventContextMetrics(
                total_symbols=10,
                provider_symbols=1,
                proxy_symbols=9,
                provider_coverage=0.10,
                proxy_only_coverage=0.90,
                relevant_indicators=["negative_news_score"],
            )
            run_result.data_quality = build_data_quality_report(
                event_context=run_result.evaluation.event_context,
                data_source_health=[],
            )
            return run_result

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=3,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        assert result.early_stopped is True
        assert "Data quality gate" in result.stop_reason
        assert len(result.rounds) == 1
        assert result.rounds[0].hypothesis_status == "data_quality_blocked"

    def test_param_search_can_adjust_ma_period(self, config):
        """Param search can tune MA periods like ma60 -> ma55."""
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        strategy.entry.conditions = [
            StrategyCondition(indicator="close_above_ma60", op="==", value=True),
        ]
        strategy.params.tunable = [
            TunableParam(
                target="entry.conditions[indicator=close_above_ma60].indicator",
                range=[20.0, 120.0],
                step=5.0,
            )
        ]
        evaluation = _make_evaluation(win_rate=0.35)

        result = pipeline._param_search_reflection(strategy, evaluation)

        assert result is not None
        assert result.proposed_changes[0].target == (
            "entry.conditions[indicator=close_above_ma60].indicator"
        )
        assert result.proposed_changes[0].to_value == 55.0

    def test_param_search_can_adjust_window_indicator_period(self, config):
        """Param search can tune non-MA indicator windows like 5d -> 4d."""
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        strategy.entry.conditions = [
            StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.5),
        ]
        strategy.params.tunable = [
            TunableParam(
                target="entry.conditions[indicator=volume_ratio_1d_5d].indicator",
                range=[3.0, 20.0],
                step=1.0,
            )
        ]

        tight_result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.35))
        assert tight_result is not None
        assert tight_result.proposed_changes[0].to_value == 4

        loose_result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.65))
        assert loose_result is not None
        assert loose_result.proposed_changes[0].to_value == 6

    def test_param_search_can_adjust_atr_stop_loss_period(self, config):
        """Param search can tune ATR stop-loss lookback and mark it as an exit change."""
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        strategy.exit.stop_loss = StopLossConfig(type="atr", multiplier=2.0)
        strategy.params.tunable = [
            TunableParam(
                target="exit.stop_loss.atr_period",
                range=[7.0, 30.0],
                step=1.0,
            )
        ]

        result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.35))

        assert result is not None
        assert result.proposed_changes[0].change_type == ChangeType.ADJUST_EXIT
        assert result.proposed_changes[0].target == "exit.stop_loss.atr_period"
        assert result.proposed_changes[0].to_value == 13

    def test_param_search_can_adjust_macd_signal_period(self, config):
        """Param search can tune MACD signal lookback via .indicator.signal."""
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        strategy.entry.conditions = [
            StrategyCondition(indicator="macd_histogram", op=">", value=0),
        ]
        strategy.params.tunable = [
            TunableParam(
                target="entry.conditions[indicator=macd_histogram].indicator.signal",
                range=[5.0, 15.0],
                step=1.0,
            )
        ]

        tight_result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.35))
        assert tight_result is not None
        assert tight_result.proposed_changes[0].target.endswith(".indicator.signal")
        assert tight_result.proposed_changes[0].to_value == 8

        loose_result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.65))
        assert loose_result is not None
        assert loose_result.proposed_changes[0].to_value == 10

    def test_param_search_can_adjust_bollinger_std(self, config):
        """Param search can tune Bollinger std multiplier as a float."""
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        strategy.entry.conditions = [
            StrategyCondition(indicator="bollinger_band_width", op="<", value=0.2),
        ]
        strategy.params.tunable = [
            TunableParam(
                target="entry.conditions[indicator=bollinger_band_width].indicator.std",
                range=[1.0, 3.0],
                step=0.5,
            )
        ]

        tight_result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.35))
        assert tight_result is not None
        assert tight_result.proposed_changes[0].to_value == 2.5

        loose_result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.65))
        assert loose_result is not None
        assert loose_result.proposed_changes[0].to_value == 1.5

    def test_param_search_can_adjust_dual_ma_fast_and_slow(self, config):
        """Param search can tune fast/slow periods for crossover indicators."""
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        strategy.entry.conditions = [
            StrategyCondition(indicator="ma5_ge_ma10_or_crossing", op="==", value=True),
        ]
        strategy.params.tunable = [
            TunableParam(
                target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast",
                range=[3.0, 8.0],
                step=1.0,
            ),
            TunableParam(
                target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow",
                range=[9.0, 20.0],
                step=1.0,
            ),
        ]

        tight_result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.35))
        assert tight_result is not None
        assert tight_result.proposed_changes[0].target.endswith(".indicator.fast")
        assert tight_result.proposed_changes[0].to_value == 6
        assert tight_result.proposed_changes[1].target.endswith(".indicator.slow")
        assert tight_result.proposed_changes[1].to_value == 9

        loose_result = pipeline._param_search_reflection(strategy, _make_evaluation(win_rate=0.65))
        assert loose_result is not None
        assert loose_result.proposed_changes[0].to_value == 4
        assert loose_result.proposed_changes[1].to_value == 11

    def test_param_search_no_tunables(self, config):
        """No tunable params → no reflection."""
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        strategy.params.tunable = []
        evaluation = _make_evaluation()

        result = pipeline._param_search_reflection(strategy, evaluation)
        assert result is None

    def test_progress_callback(self, config):
        """Progress messages are reported."""
        pipeline = EvolutionPipeline(config)
        messages = []

        async def mock_run(*args, **kwargs):
            return _make_run_result(confidence=0.4)

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        pipeline.evolve(
            "test_v1",
            rounds=1,
            method=EvolutionMethod.PARAM_SEARCH,
            on_progress=lambda msg: messages.append(msg),
        )

        assert len(messages) > 0
        assert any("Round 1" in m for m in messages)

    def test_overfit_hard_gate_prevents_champion(self, config):
        """An overfit strategy should not become champion, even with higher score."""
        pipeline = EvolutionPipeline(config)

        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            # Round 1: score=0.3, clean
            # Round 2: score=0.5 but overfit (train_val_gap=0.20)
            if call_count[0] == 1:
                return _make_run_result("test_v1", confidence=0.3, version=1, overfit=False)
            return _make_run_result("test_v2", confidence=0.5, version=2, overfit=True)

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=2,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        # Champion should remain round 1 (overfit round 2 is rejected)
        assert result.champion_score == 0.3
        assert result.early_stopped  # because is_overfit triggers early stop
        assert "overfit" in result.stop_reason.lower()

    def test_overfit_round1_still_evolves(self, config):
        """If the original strategy is overfit, round 1 should NOT early-stop.

        The whole point of the overfit playbook is to fix overfit strategies,
        so round 1 must be allowed to reflect+mutate even when is_overfit=True.
        """
        pipeline = EvolutionPipeline(config)

        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            # Round 1: overfit original; Round 2: fixed version (not overfit)
            if call_count[0] == 1:
                return _make_run_result("test_v1", confidence=0.3, version=1, overfit=True)
            return _make_run_result("test_v2", confidence=0.5, version=2, overfit=False)

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=3,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        # Evolution should proceed past round 1 despite overfit original
        assert call_count[0] >= 2
        # Round 2 (non-overfit, higher score) should become champion
        assert result.champion_score >= 0.5

    def test_baseline_param_search_saves_strategy(self, config):
        """Baseline comparison should save ps_strategy with a non-colliding ID."""
        pipeline = EvolutionPipeline(config)

        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            sid = args[0] if args else f"test_v{call_count[0]}"
            return _make_run_result(str(sid), confidence=0.4 + call_count[0] * 0.05)

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()

        saved_strategies: dict[str, Strategy] = {}
        original_save = pipeline.store.save

        def track_save(strategy):
            saved_strategies[strategy.meta.id] = strategy
            return original_save(strategy)

        pipeline.store.save = track_save

        # Run a single-round LLM evolution (which triggers baseline comparison)
        pipeline.evolve(
            "test_v1",
            rounds=1,
            method=EvolutionMethod.LLM,
        )

        # The baseline strategy should use a distinct _ps_baseline suffix
        # so it never collides with real evolved versions (e.g. test_v2).
        ps_ids = [sid for sid in saved_strategies if "ps_baseline" in sid]
        assert ps_ids, "Baseline param_search strategy was not saved"
        assert ps_ids[0] == "test_ps_baseline"

    def test_baseline_does_not_overwrite_evolved_strategy(self, config):
        """Regression: baseline mutation must NOT overwrite a real round-2 strategy."""
        pipeline = EvolutionPipeline(config)

        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            # Round 1: original, Round 2: improved
            score = 0.3 + call_count[0] * 0.1
            sid = f"test_v{call_count[0]}"
            return _make_run_result(sid, confidence=score, version=call_count[0])

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()

        saved_strategies: dict[str, Strategy] = {}
        original_save = pipeline.store.save

        def track_save(strategy):
            saved_strategies[strategy.meta.id] = strategy
            return original_save(strategy)

        pipeline.store.save = track_save

        # Run 2 rounds with HYBRID (triggers baseline comparison at end)
        pipeline.evolve(
            "test_v1",
            rounds=2,
            method=EvolutionMethod.HYBRID,
        )

        # test_v2 should be the real evolved strategy, NOT the baseline
        if "test_v2" in saved_strategies:
            v2 = saved_strategies["test_v2"]
            # The baseline would have been saved as test_ps_baseline, not test_v2
            assert v2.meta.parent_id == "test_v1"
        # The baseline should have a distinct ID
        ps_ids = [sid for sid in saved_strategies if "ps_baseline" in sid]
        # Baseline may or may not fire (depends on param_search reflection
        # producing changes), but if it does, it must not be test_v2
        for ps_id in ps_ids:
            assert ps_id != "test_v2"


class TestEvolutionResult:
    def test_improvement_empty(self):
        result = EvolutionResult(original_strategy_id="x")
        assert result.improvement == 0.0

    def test_improvement_calculated(self):
        result = EvolutionResult(
            original_strategy_id="x",
            champion_score=0.6,
        )
        # Add a fake round with first eval
        from alphaevo.orchestrator.evolution import EvolutionRound

        result.rounds.append(
            EvolutionRound(
                round_num=1,
                strategy=_make_strategy(),
                evaluation=_make_evaluation(confidence=0.4),
            )
        )
        assert result.improvement == pytest.approx(0.2)


class TestEvolutionHistoryAndExperience:
    """Tests for experience recording."""

    def test_experience_prompt_text_empty_when_no_records(self, config):
        from alphaevo.reflection.experience import ExperienceStore

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        text = pipeline._experience_store.format_for_prompt(family_id="test", limit=10)
        assert text == ""

    def test_record_pending_experience_persists(self, config):
        from alphaevo.models.enums import ChangeType
        from alphaevo.models.execution import StrategyChange
        from alphaevo.reflection.experience import ExperienceQuery, ExperienceStore

        # Use an in-memory store
        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        change = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=30,
            to_value=25,
            reason="Tighter RSI",
        )
        pipeline._record_pending_experience(
            [(change, 0.30, "test_v2", 1, "", "param_search", "")],
            family_id="test",
            score_after=0.45,
            worked=True,
        )

        records = pipeline._experience_store.query(ExperienceQuery())
        assert len(records) == 1
        assert records[0].worked is True
        assert records[0].score_before == pytest.approx(0.30)
        assert records[0].score_after == pytest.approx(0.45)
        assert records[0].strategy_family == "test"

    def test_record_pending_experience_no_improvement(self, config):
        from alphaevo.models.enums import ChangeType
        from alphaevo.models.execution import StrategyChange
        from alphaevo.reflection.experience import ExperienceQuery, ExperienceStore

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        change = StrategyChange(
            change_type=ChangeType.LOOSEN_FILTER,
            target="exit.stop_loss.value",
            from_value=0.04,
            to_value=0.06,
            reason="Widen stop",
        )
        pipeline._record_pending_experience(
            [(change, 0.40, "test_v2", 2, "", "param_search", "")],
            family_id="test",
            score_after=0.38,
            worked=False,
        )

        records = pipeline._experience_store.query(ExperienceQuery())
        assert len(records) == 1
        assert records[0].worked is False
        assert records[0].score_delta == pytest.approx(-0.02)
        assert "NOT improve" in records[0].lesson

    def test_record_pending_experience_auto_worked_by_delta(self, config):
        from alphaevo.models.enums import ChangeType
        from alphaevo.models.execution import StrategyChange
        from alphaevo.reflection.experience import ExperienceQuery, ExperienceStore

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        change = StrategyChange(
            change_type=ChangeType.ADJUST_EXIT,
            target="exit.stop_loss.value",
            from_value=0.04,
            to_value=0.03,
            reason="Tighten stop",
        )
        pipeline._record_pending_experience(
            [(change, 0.40, "test_v2", 2, "", "param_search", "")],
            family_id="test",
            score_after=0.42,
            worked=None,
        )

        records = pipeline._experience_store.query(ExperienceQuery())
        assert len(records) == 1
        assert records[0].worked is True
        assert records[0].score_delta == pytest.approx(0.02)

    def test_filter_failed_repeated_changes(self, config):
        from alphaevo.models.enums import ChangeType
        from alphaevo.models.execution import StrategyChange
        from alphaevo.reflection.experience import ExperienceStore

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        failed_change = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=30,
            to_value=25,
            reason="Prior failed attempt",
        )
        pipeline._record_pending_experience(
            [(failed_change, 0.40, "demo_strategy_v2", 2, "", "llm", "")],
            family_id="demo_strategy",
            score_after=0.35,
            worked=False,
        )
        pipeline._record_pending_experience(
            [(failed_change, 0.35, "demo_strategy_v3", 3, "", "llm", "")],
            family_id="demo_strategy",
            score_after=0.34,
            worked=False,
        )

        candidates = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=25,
                reason="Try same failed change again",
            ),
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.value",
                from_value=0.04,
                to_value=0.03,
                reason="New idea",
            ),
        ]

        filtered = pipeline._filter_failed_repeated_changes("demo_strategy", candidates)
        assert len(filtered) == 1
        assert filtered[0].target == "exit.stop_loss.value"

    def test_filter_failed_repeated_changes_requires_threshold(self, config):
        """Single failed attempt should not blacklist a change permanently."""
        from alphaevo.models.enums import ChangeType
        from alphaevo.models.execution import StrategyChange
        from alphaevo.reflection.experience import ExperienceStore

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        once_failed = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=30,
            to_value=25,
            reason="One failed attempt",
        )
        pipeline._record_pending_experience(
            [(once_failed, 0.40, "demo_strategy_v2", 2, "", "llm", "")],
            family_id="demo_strategy",
            score_after=0.39,
            worked=False,
        )

        candidates = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=25,
                reason="Retry once-failed change",
            )
        ]

        filtered = pipeline._filter_failed_repeated_changes("demo_strategy", candidates)
        assert len(filtered) == 1

    def test_record_pending_experience_splits_delta_for_multi_change(self, config):
        from alphaevo.models.enums import ChangeType
        from alphaevo.models.execution import StrategyChange
        from alphaevo.reflection.experience import ExperienceQuery, ExperienceStore

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        c1 = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=30,
            to_value=25,
            reason="Change 1",
        )
        c2 = StrategyChange(
            change_type=ChangeType.ADJUST_EXIT,
            target="exit.stop_loss.value",
            from_value=0.04,
            to_value=0.03,
            reason="Change 2",
        )

        pipeline._record_pending_experience(
            [(c1, 0.40, "test_v2", 1, "", "llm", ""), (c2, 0.40, "test_v2", 1, "", "llm", "")],
            family_id="test",
            score_after=0.50,
            worked=None,
        )

        records = pipeline._experience_store.query(ExperienceQuery(limit=100))
        assert len(records) == 2
        assert all(r.score_delta == pytest.approx(0.05) for r in records)

    def test_experience_recorded_during_evolution(self, config):
        """Multi-round evolution records experience to the store."""
        from alphaevo.reflection.experience import ExperienceQuery, ExperienceStore

        pipeline = EvolutionPipeline(config)
        # In-memory store so we can inspect it
        pipeline._experience_store = ExperienceStore(":memory:")

        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            score = 0.3 + call_count[0] * 0.1
            sid = f"test_v{call_count[0]}"
            return _make_run_result(sid, confidence=score, version=call_count[0])

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=3,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        # Experience should be recorded for the param_search changes
        records = pipeline._experience_store.query(ExperienceQuery(limit=100))
        # At least round 1's changes should be recorded after round 2 runs
        assert len(records) >= 1
        assert result.champion_score > 0.3

    def test_experience_prompt_can_exclude_test_records(self, config):
        from alphaevo.models.enums import ChangeType
        from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")
        pipeline._experience_store.record_batch(
            [
                ExperienceRecord(
                    strategy_family="demo",
                    strategy_id="test_v1",
                    round_num=1,
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    to_value=25,
                    worked=True,
                    lesson="test lesson",
                    source="test",
                ),
                ExperienceRecord(
                    strategy_family="demo",
                    strategy_id="live_v2",
                    round_num=2,
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.stop_loss.type",
                    to_value="atr",
                    worked=True,
                    lesson="live lesson",
                    source="llm",
                ),
            ]
        )

        text = pipeline._experience_store.format_for_prompt(
            family_id="demo",
            limit=10,
            exclude_test_sources=True,
        )

        assert "live lesson" in text
        assert "test lesson" not in text


class TestStagnationRecovery:
    """Test that evolution attempts recovery before early-stopping."""

    def test_recovery_reverts_to_champion(self, config):
        """On stagnation, should revert to champion and try one more round."""
        from alphaevo.reflection.experience import ExperienceStore

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        round_ids = []
        call_count = [0]

        async def mock_run(sid, **kwargs):
            call_count[0] += 1
            round_ids.append(sid)
            # Round 1: 0.5, Round 2: 0.4, Round 3: 0.3 (stagnation triggers)
            # After recovery: reverts to champion (round 1's id), Round 4: 0.45
            scores = {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.45, 5: 0.35}
            score = scores.get(call_count[0], 0.3)
            return _make_run_result(sid, confidence=score, version=call_count[0])

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=5,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        # Should have attempted recovery (ran more than 3 rounds)
        assert call_count[0] >= 4
        # Champion should still be the best score from round 1
        assert result.champion_score >= 0.5


class TestCandidateFallback:
    """Tests for candidate fallback on mutation failure."""

    def test_mutation_fallback_to_next_candidate(self, config):
        """When top candidate mutation fails, pipeline should try next candidate."""
        from alphaevo.models.execution import (
            CandidateExperiment,
            ReflectionResult,
            ResearchHypothesis,
            StrategyChange,
        )
        from alphaevo.reflection.mutator import MutationError

        pipeline = EvolutionPipeline(config)

        strategy = _make_strategy()

        # Create 2 candidates — first will fail, second will succeed
        bad_change = StrategyChange(
            change_type=ChangeType.ADD_CONDITION,
            target="entry.conditions",
            to_value={"indicator": "nonexistent_xyz", "op": ">", "value": 1},
            reason="will fail",
        )
        good_change = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=30,
            to_value=25,
            reason="will succeed",
        )
        reflection = ReflectionResult(
            strategy_id="test_v1",
            evaluation_id="eval-001",
            proposed_changes=[bad_change],
            candidates=[
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(problem="p1", hypothesis="h1"),
                    proposed_changes=[bad_change],
                    priority_score=0.9,
                ),
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(problem="p2", hypothesis="h2"),
                    proposed_changes=[good_change],
                    priority_score=0.7,
                ),
            ],
        )

        call_count = [0]
        original_mutate = pipeline._mutator.mutate

        def mock_mutate(strat, changes, atomic=False):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MutationError("Unknown indicator")
            return original_mutate(strat, changes, atomic=atomic)

        pipeline._mutator.mutate = mock_mutate
        pipeline.store.save = MagicMock()

        mutated = False
        candidates_to_try = list(reflection.candidates) if reflection.candidates else [None]

        for candidate in candidates_to_try:
            if candidate is not None:
                changes_to_apply = candidate.proposed_changes
            else:
                changes_to_apply = reflection.proposed_changes
            if not changes_to_apply:
                continue
            try:
                new_strategy = pipeline._mutator.mutate(strategy, changes_to_apply)
                pipeline.store.save(new_strategy)
                reflection.proposed_changes = changes_to_apply
                mutated = True
                break
            except MutationError:
                continue

        assert mutated
        assert call_count[0] == 2
        assert reflection.proposed_changes == [good_change]

    def test_all_candidates_fail_stops_evolution(self, config):
        """If all candidates fail mutation, pipeline should stop."""
        from alphaevo.models.execution import (
            CandidateExperiment,
            ReflectionResult,
            ResearchHypothesis,
            StrategyChange,
        )
        from alphaevo.reflection.mutator import MutationError

        pipeline = EvolutionPipeline(config)
        pipeline.store.save = MagicMock()

        strategy = _make_strategy()

        bad1 = StrategyChange(
            change_type=ChangeType.ADD_CONDITION,
            target="entry.conditions",
            to_value={"indicator": "fake_a", "op": ">", "value": 1},
            reason="fail",
        )
        bad2 = StrategyChange(
            change_type=ChangeType.ADD_CONDITION,
            target="entry.conditions",
            to_value={"indicator": "fake_b", "op": ">", "value": 1},
            reason="fail too",
        )
        reflection = ReflectionResult(
            strategy_id="test_v1",
            evaluation_id="eval-001",
            proposed_changes=[bad1],
            candidates=[
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(problem="p1", hypothesis="h1"),
                    proposed_changes=[bad1],
                    priority_score=0.9,
                ),
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(problem="p2", hypothesis="h2"),
                    proposed_changes=[bad2],
                    priority_score=0.7,
                ),
            ],
        )

        # Make all mutations fail
        pipeline._mutator.mutate = MagicMock(side_effect=MutationError("bad"))

        mutated = False
        candidates_to_try = list(reflection.candidates)
        for candidate in candidates_to_try:
            try:
                pipeline._mutator.mutate(strategy, candidate.proposed_changes)
                mutated = True
                break
            except MutationError:
                continue

        assert not mutated

    def test_candidate_screening_prefers_best_scored_mutation(self, config):
        """When screening is available, choose the candidate that scores best now."""
        pipeline = EvolutionPipeline(config)
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        first_round = _make_run_result(confidence=0.40, signal_count=55)
        second_round = _make_run_result(strategy_id="test_v2", confidence=0.55, version=2)

        async def mock_run(sid, **kwargs):
            if sid == "test_v1":
                return first_round
            return second_round

        reflection = ReflectionResult(
            strategy_id="test_v1",
            evaluation_id="eval-test_v1",
            proposed_changes=[
                StrategyChange(
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value=30,
                    to_value=25,
                    reason="candidate one",
                )
            ],
            candidates=[
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(problem="p1", hypothesis="h1"),
                    proposed_changes=[
                        StrategyChange(
                            change_type=ChangeType.TIGHTEN_FILTER,
                            target="entry.conditions[indicator=rsi_14].value",
                            from_value=30,
                            to_value=25,
                            reason="candidate one",
                        )
                    ],
                    priority_score=0.9,
                ),
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(problem="p2", hypothesis="h2"),
                    proposed_changes=[
                        StrategyChange(
                            change_type=ChangeType.LOOSEN_FILTER,
                            target="entry.conditions[indicator=rsi_14].value",
                            from_value=30,
                            to_value=35,
                            reason="candidate two",
                        )
                    ],
                    priority_score=0.7,
                ),
            ],
        )

        def _screen(run_result, mutated_strategy, changes, candidate_option):
            to_value = changes[0].to_value
            if to_value == 25:
                evaluation = _make_evaluation(
                    strategy_id=mutated_strategy.meta.id,
                    confidence=0.42,
                    signal_count=50,
                )
            else:
                evaluation = _make_evaluation(
                    strategy_id=mutated_strategy.meta.id,
                    confidence=0.55,
                    signal_count=68,
                )
            return _ScreenedMutationCandidate(
                strategy=mutated_strategy,
                changes=changes,
                candidate=candidate_option,
                evaluation=evaluation,
                score_delta=evaluation.confidence_score - run_result.evaluation.confidence_score,
                signal_delta=evaluation.overall.signal_count
                - run_result.evaluation.overall.signal_count,
            )

        pipeline._run_pipeline.run = mock_run
        pipeline._do_reflection = MagicMock(return_value=reflection)
        pipeline._param_search_reflection = MagicMock(return_value=None)
        pipeline._critic.rank_candidates = MagicMock(side_effect=lambda *args: args[2])
        pipeline._critic.critique = MagicMock(
            side_effect=lambda strategy, evaluation, refl: SimpleNamespace(
                approved=list(refl.proposed_changes),
                rejected=[],
                warnings=[],
            )
        )
        pipeline._can_screen_mutation_candidates = MagicMock(return_value=True)
        pipeline._screen_mutation_candidate = MagicMock(side_effect=_screen)

        pipeline.evolve("test_v1", rounds=2, method=EvolutionMethod.HYBRID)

        saved_strategy = pipeline.store.save.call_args_list[0].args[0]
        assert saved_strategy.entry.conditions[0].value == 35

    def test_candidate_screening_penalizes_signal_collapse(self, config):
        """Prefer a slightly weaker score if it preserves far more evidence."""
        pipeline = EvolutionPipeline(config)
        base_run = _make_run_result(confidence=0.40, signal_count=100)

        collapsed = _ScreenedMutationCandidate(
            strategy=_make_strategy("test_v2", version=2),
            changes=[],
            evaluation=_make_evaluation("test_v2", confidence=0.56, signal_count=45),
            score_delta=0.16,
            signal_delta=-55,
        )
        stable = _ScreenedMutationCandidate(
            strategy=_make_strategy("test_v3", version=3),
            changes=[],
            evaluation=_make_evaluation("test_v3", confidence=0.53, signal_count=82),
            score_delta=0.13,
            signal_delta=-18,
        )

        chosen = pipeline._select_screened_candidate(base_run, [collapsed, stable])

        assert chosen.strategy.meta.id == "test_v3"


class TestHeuristicIndicatorSuggestion:
    """Test _heuristic_indicator_suggestion on EvolutionPipeline."""

    @pytest.fixture
    def pipeline(self, config):
        return EvolutionPipeline(config)

    def test_low_win_rate_suggests_filter(self, pipeline):
        strategy = _make_strategy()
        ev = EvaluationReport(
            evaluation_id="eval-1",
            strategy_id="test_v1",
            overall=OverallMetrics(
                signal_count=50,
                win_rate=0.35,
                avg_return=0.01,
                profit_loss_ratio=1.8,
                max_drawdown=0.10,
                sharpe_ratio=0.5,
            ),
            confidence_score=0.20,
        )
        changes = pipeline._heuristic_indicator_suggestion(strategy, ev)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.ADD_CONDITION

    def test_few_signals_returns_empty(self, pipeline):
        strategy = _make_strategy()
        ev = EvaluationReport(
            evaluation_id="eval-1",
            strategy_id="test_v1",
            overall=OverallMetrics(
                signal_count=5,
                win_rate=0.35,
                avg_return=0.01,
                profit_loss_ratio=1.8,
                max_drawdown=0.10,
                sharpe_ratio=0.5,
            ),
            confidence_score=0.10,
        )
        changes = pipeline._heuristic_indicator_suggestion(strategy, ev)
        assert len(changes) == 0


class TestHypothesisAwareEvolution:
    def test_reflection_can_escalate_parameter_issue_to_thesis(self, config):
        pipeline = EvolutionPipeline(config)
        base_strategy = _make_strategy()
        first_eval = _make_evaluation(
            strategy_id="test_v1",
            confidence=0.34,
            win_rate=0.42,
            signal_count=70,
            overfit=True,
        )
        assessment = base_strategy.assess_market_hypothesis(first_eval)
        assert assessment.status == "parameter_misaligned"

        reflection = ReflectionResult(
            strategy_id="test_v1",
            evaluation_id="eval-test_v1",
            diagnosis=(
                "The core hypothesis is not supported by the data and the strategy fails in its "
                "intended regime."
            ),
        )

        refined = pipeline._refine_hypothesis_assessment(assessment, reflection)

        assert refined.status == "thesis_misaligned"
        assert "core hypothesis" in refined.rationale.lower()

    def test_thesis_misaligned_promotes_structural_mutation(self, config):
        pipeline = EvolutionPipeline(config)
        base_strategy = _make_strategy()
        base_strategy.meta.preferred_regime = ["trending_up"]
        base_strategy.entry.logic = "and"
        base_strategy.entry.conditions = [
            StrategyCondition(indicator="rsi_14", op="<", value=30),
            StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.5),
            StrategyCondition(indicator="close_above_ma20", op="==", value=True),
        ]

        first_eval = _make_evaluation(
            strategy_id="test_v1",
            confidence=0.34,
            win_rate=0.40,
            signal_count=50,
        )
        first_eval.by_regime = [
            RegimeMetrics(
                regime=MarketRegime.TRENDING_UP,
                win_rate=0.35,
                avg_return=-0.02,
                signal_count=40,
            )
        ]

        async def mock_run(sid, **kwargs):
            if sid == "test_v1":
                return _make_custom_run_result(base_strategy, first_eval)
            return _make_run_result(sid, confidence=0.36, version=2, signal_count=55)

        reflection = ReflectionResult(
            strategy_id="test_v1",
            evaluation_id="eval-test_v1",
            proposed_changes=[
                StrategyChange(
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value=30,
                    to_value=25,
                    reason="parameter-only baseline suggestion",
                )
            ],
            candidates=[
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(
                        problem="low quality entries",
                        hypothesis="tighten RSI",
                    ),
                    proposed_changes=[
                        StrategyChange(
                            change_type=ChangeType.TIGHTEN_FILTER,
                            target="entry.conditions[indicator=rsi_14].value",
                            from_value=30,
                            to_value=25,
                            reason="parameter-only baseline suggestion",
                        )
                    ],
                    priority_score=0.9,
                )
            ],
        )

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline._do_reflection = MagicMock(return_value=reflection)
        pipeline._param_search_reflection = MagicMock(return_value=None)
        pipeline._critic.rank_candidates = MagicMock(side_effect=lambda *args: args[2])
        pipeline._critic.critique = MagicMock(
            side_effect=lambda strategy, evaluation, refl: SimpleNamespace(
                approved=list(refl.proposed_changes),
                rejected=[],
                warnings=[],
            )
        )
        pipeline.store.save = MagicMock()

        pipeline.evolve("test_v1", rounds=2, method=EvolutionMethod.HYBRID)

        saved_strategy = pipeline.store.save.call_args_list[0].args[0]
        assert saved_strategy.entry.logic == "or"
        assert len(saved_strategy.entry.conditions) <= 3

    def test_execution_misaligned_adds_exit_recovery_changes(self, config):
        pipeline = EvolutionPipeline(config)
        base_strategy = _make_strategy()

        first_eval = _make_evaluation(
            strategy_id="test_v1",
            confidence=0.38,
            win_rate=0.58,
            signal_count=60,
        )
        first_eval.overall.avg_return = -0.01
        second_eval = _make_evaluation(
            strategy_id="test_v2",
            confidence=0.40,
            win_rate=0.56,
            signal_count=60,
        )

        async def mock_run(sid, **kwargs):
            if sid == "test_v1":
                return _make_custom_run_result(base_strategy, first_eval)
            return _make_custom_run_result(
                _make_strategy(sid, version=2),
                second_eval,
            )

        reflection = ReflectionResult(
            strategy_id="test_v1",
            evaluation_id="eval-test_v1",
            proposed_changes=[
                StrategyChange(
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value=30,
                    to_value=28,
                    reason="baseline suggestion",
                )
            ],
        )

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline._do_reflection = MagicMock(return_value=reflection)
        pipeline._param_search_reflection = MagicMock(return_value=None)
        pipeline._critic.rank_candidates = MagicMock(side_effect=lambda *args: args[2])
        pipeline._critic.critique = MagicMock(
            side_effect=lambda strategy, evaluation, refl: SimpleNamespace(
                approved=list(refl.proposed_changes),
                rejected=[],
                warnings=[],
            )
        )
        pipeline.store.save = MagicMock()

        pipeline.evolve("test_v1", rounds=2, method=EvolutionMethod.HYBRID)

        saved_strategy = pipeline.store.save.call_args_list[0].args[0]
        assert saved_strategy.exit.stop_loss.type == "atr"

    def test_good_metrics_returns_empty(self, config):
        pipeline = EvolutionPipeline(config)
        strategy = _make_strategy()
        ev = EvaluationReport(
            evaluation_id="eval-1",
            strategy_id="test_v1",
            overall=OverallMetrics(
                signal_count=80,
                win_rate=0.65,
                avg_return=0.03,
                profit_loss_ratio=2.5,
                max_drawdown=0.10,
                sharpe_ratio=1.5,
            ),
            confidence_score=0.60,
        )
        changes = pipeline._heuristic_indicator_suggestion(strategy, ev)
        assert len(changes) == 0


class TestChampionPromotionGuards:
    def test_negative_trade_payoff_cannot_become_champion(self, config):
        pipeline = EvolutionPipeline(config)

        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_run_result("test_v1", confidence=0.30, version=1, signal_count=80)
            result = _make_run_result("test_v2", confidence=0.45, version=2, signal_count=80)
            result.evaluation.overall.avg_return = -0.01
            result.evaluation.overall.total_return = -0.08
            return result

        pipeline._run_pipeline.run = mock_run
        pipeline._run_pipeline.ensure_builtin_strategies = MagicMock()
        pipeline.store.save = MagicMock()

        result = pipeline.evolve(
            "test_v1",
            rounds=2,
            method=EvolutionMethod.PARAM_SEARCH,
        )

        assert result.champion_id == "test_v1"
        assert result.champion_score == pytest.approx(0.30)
