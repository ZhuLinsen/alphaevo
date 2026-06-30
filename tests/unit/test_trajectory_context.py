"""Tests for trajectory export and tiered context builder."""

import json
import tempfile
from pathlib import Path  # noqa: E402 (moved up)

from alphaevo.data.adapter import DataSourceHealth
from alphaevo.data.quality import build_data_quality_report
from alphaevo.models.enums import (
    MarketType,
    StrategyCategory,
)
from alphaevo.models.execution import (
    EvaluationReport,
    EventContextMetrics,
    OverallMetrics,
)
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)
from alphaevo.research_log.context import ContextBuilder, ResearchContext
from alphaevo.research_log.trajectory import (
    EvolutionTrajectory,
    TrajectoryCollector,
    TrajectoryStep,
    export_jsonl,
    export_preference_pairs,
    export_sharegpt,
)


def _strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id="ctx_test_v1",
            name="Context Test",
            version=1,
            market=MarketType.A_SHARE,
            category=StrategyCategory.TREND,
        ),
        description="Test",
        entry=StrategyEntry(
            conditions=[
                StrategyCondition(indicator="rsi_14", op="<", value=30),
                StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.5),
            ],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
        ),
    )


def _evaluation(
    signal_count: int = 50,
    win_rate: float = 0.55,
    max_drawdown: float = 0.12,
    pl_ratio: float = 1.8,
) -> EvaluationReport:
    return EvaluationReport(
        evaluation_id="eval-ctx",
        strategy_id="ctx_test_v1",
        overall=OverallMetrics(
            signal_count=signal_count,
            win_rate=win_rate,
            avg_return=0.02,
            profit_loss_ratio=pl_ratio,
            max_drawdown=max_drawdown,
            sharpe_ratio=1.2,
        ),
        confidence_score=0.45,
    )


# ── Trajectory Tests ────────────────────────────────────────────────


class TestTrajectoryCollector:
    def test_collect_and_finalize(self):
        collector = TrajectoryCollector("traj_001", "test_family")
        collector.set_initial_score(0.30)
        collector.set_metadata("method", "hybrid")

        step1 = TrajectoryStep(
            round_num=1,
            strategy_id="test_v1",
            score_before=0.30,
            win_rate_before=0.45,
            signal_count_before=40,
            diagnosis="Low win rate due to noisy entries",
            hypothesis="Add RSI filter",
            changes=[{"change_type": "add_condition", "target": "entry.conditions"}],
            improved=True,
        )
        step2 = TrajectoryStep(
            round_num=2,
            strategy_id="test_v2",
            score_before=0.40,
            win_rate_before=0.55,
            signal_count_before=35,
            diagnosis="Win rate improved but P/L ratio still low",
            hypothesis="Widen take profit",
            changes=[{"change_type": "adjust_exit", "target": "exit.take_profit"}],
            improved=False,
        )

        collector.record_step(step1)
        collector.record_step(step2)

        traj = collector.finalize()
        assert traj.trajectory_id == "traj_001"
        assert traj.strategy_family == "test_family"
        assert traj.initial_score == 0.30
        assert traj.total_rounds == 2
        assert len(traj.steps) == 2
        assert traj.success_rate == 0.5

    def test_empty_trajectory(self):
        collector = TrajectoryCollector("empty", "test")
        traj = collector.finalize()
        assert traj.total_rounds == 0
        assert traj.success_rate == 0.0


class TestTrajectoryExport:
    def _sample_trajectory(self) -> EvolutionTrajectory:
        return EvolutionTrajectory(
            trajectory_id="test_traj",
            strategy_family="test",
            initial_score=0.30,
            final_score=0.45,
            total_rounds=2,
            steps=[
                TrajectoryStep(
                    round_num=1,
                    strategy_id="v1",
                    score_before=0.30,
                    score_after=0.40,
                    score_delta=0.10,
                    diagnosis="Low win rate",
                    hypothesis="Add filter",
                    changes=[{"type": "add_condition"}],
                    improved=True,
                ),
                TrajectoryStep(
                    round_num=2,
                    strategy_id="v2",
                    score_before=0.40,
                    score_after=0.38,
                    score_delta=-0.02,
                    diagnosis="Overfit",
                    hypothesis="Simplify",
                    changes=[{"type": "remove_condition"}],
                    improved=False,
                ),
            ],
        )

    def test_export_jsonl(self):
        traj = self._sample_trajectory()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            result = export_jsonl(traj, path)
            assert result.exists()
            lines = result.read_text().strip().split("\n")
            assert len(lines) == 2
            data = json.loads(lines[0])
            assert data["round"] == 1
            assert data["outcome"]["improved"] is True

    def test_export_sharegpt(self):
        traj = self._sample_trajectory()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sharegpt.jsonl"
            result = export_sharegpt(traj, path)
            assert result.exists()
            lines = result.read_text().strip().split("\n")
            assert len(lines) == 2
            conv = json.loads(lines[0])
            assert len(conv["conversations"]) == 2
            assert conv["conversations"][0]["from"] == "human"

    def test_export_preference_pairs(self):
        traj = self._sample_trajectory()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.jsonl"
            result = export_preference_pairs(traj, path)
            assert result.exists()
            lines = result.read_text().strip().split("\n")
            assert len(lines) >= 1  # At least one pair
            pair = json.loads(lines[0])
            assert "chosen" in pair
            assert "rejected" in pair


# ── Context Builder Tests ────────────────────────────────────────────


class TestContextBuilder:
    def test_build_summary(self):
        builder = ContextBuilder()
        ctx = builder.build(_strategy(), _evaluation())
        assert not ctx.tier1_summary.is_empty
        assert "ctx_test_v1" in ctx.tier1_summary.content
        assert "55.0%" in ctx.tier1_summary.content

    def test_classify_problems(self):
        builder = ContextBuilder()
        # Low signal count
        problems = builder._classify_problems(_evaluation(signal_count=10))
        assert "low_signals" in problems

        # Low win rate
        problems = builder._classify_problems(_evaluation(win_rate=0.30))
        assert "low_win_rate" in problems

        # High drawdown
        problems = builder._classify_problems(_evaluation(max_drawdown=0.25))
        assert "high_drawdown" in problems

        # Good metrics — no problems
        problems = builder._classify_problems(
            _evaluation(signal_count=100, win_rate=0.60, max_drawdown=0.08)
        )
        assert len(problems) == 0

    def test_classify_data_quality_when_event_context_is_proxy_heavy(self):
        builder = ContextBuilder()
        evaluation = _evaluation(signal_count=100, win_rate=0.60, max_drawdown=0.08)
        evaluation.event_context = EventContextMetrics(
            total_symbols=20,
            provider_symbols=2,
            proxy_symbols=18,
            provider_coverage=0.10,
            proxy_only_coverage=0.90,
            relevant_indicators=["negative_news_score"],
        )

        problems = builder._classify_problems(evaluation)

        assert "data_quality" in problems

    def test_classify_data_quality_from_source_health(self):
        builder = ContextBuilder()
        evaluation = _evaluation(signal_count=100, win_rate=0.60, max_drawdown=0.08)
        data_quality = build_data_quality_report(
            data_source_health=[
                DataSourceHealth(
                    name="tencent",
                    priority=0,
                    success_count=3,
                    failure_count=1,
                    last_error="timeout",
                )
            ]
        )

        problems = builder._classify_problems(
            evaluation,
            data_quality_report=data_quality,
        )

        assert "data_quality" in problems

    def test_data_quality_context_is_injected(self):
        builder = ContextBuilder()
        evaluation = _evaluation(signal_count=100, win_rate=0.60, max_drawdown=0.08)
        data_quality = build_data_quality_report(
            data_source_health=[
                DataSourceHealth(
                    name="tencent",
                    priority=0,
                    success_count=3,
                    failure_count=1,
                    last_error="timeout",
                )
            ]
        )

        ctx = builder.build(
            _strategy(),
            evaluation,
            data_quality_report=data_quality,
        )

        prompt = ctx.to_prompt()
        assert "Data Quality Gate" in prompt
        assert "source_recovered_errors" in prompt

    def test_to_prompt_without_detail(self):
        builder = ContextBuilder()
        ctx = builder.build(_strategy(), _evaluation())
        prompt = ctx.to_prompt(include_detail=False)
        assert "Current State" in prompt
        # Tier 3 should NOT be included
        assert "Detailed History" not in prompt

    def test_to_prompt_with_detail(self):
        builder = ContextBuilder()
        ctx = builder.build(_strategy(), _evaluation())
        # Even with include_detail, tier3 is empty without rounds
        prompt = ctx.to_prompt(include_detail=True)
        assert "Current State" in prompt

    def test_truncate(self):
        from alphaevo.research_log.context import ContextTier

        tier = ContextTier(name="test", content="x" * 5000, max_chars=100)
        truncated = tier.truncate()
        assert len(truncated) <= 100
        assert "truncated" in truncated

    def test_with_playbook_store(self):
        from alphaevo.reflection.playbook import PlaybookStore

        pb_store = PlaybookStore(db_path=":memory:")
        builder = ContextBuilder(playbook_store=pb_store)
        ctx = builder.build(
            _strategy(),
            _evaluation(signal_count=10, win_rate=0.35),
        )
        # Should have playbook-relevant content in tier2
        # (playbooks are injected via the main pipeline, not builder directly)
        assert not ctx.tier1_summary.is_empty


class TestResearchContext:
    def test_empty_context(self):
        ctx = ResearchContext()
        prompt = ctx.to_prompt()
        assert prompt == ""

    def test_assembled_prompt(self):
        ctx = ResearchContext()
        ctx.tier1_summary.content = "Summary here"
        ctx.tier2_retrieval.content = "Relevant lessons"
        ctx.tier3_detail.content = "Full history"

        prompt = ctx.to_prompt(include_detail=False)
        assert "Summary here" in prompt
        assert "Relevant lessons" in prompt
        assert "Full history" not in prompt

        prompt_full = ctx.to_prompt(include_detail=True)
        assert "Full history" in prompt_full


class TestContextBuilderOverfitClassification:
    """Verify that anti_overfit (not anti_fit) is correctly checked."""

    def test_overfit_classified_when_train_val_gap_high(self):
        from alphaevo.models.execution import AntiFitMetrics

        ev = _evaluation()
        ev.anti_overfit = AntiFitMetrics(train_val_gap=0.15, val_test_gap=0.05)
        builder = ContextBuilder()
        problems = builder._classify_problems(ev)
        assert "overfit" in problems

    def test_no_overfit_when_anti_overfit_default(self):
        ev = _evaluation()
        builder = ContextBuilder()
        problems = builder._classify_problems(ev)
        assert "overfit" not in problems


class TestTrajectoryBackfill:
    """Verify score_after, win_rate_after, signal_count_after, improved are correct."""

    def test_finalize_after_backfill_has_correct_final_score(self):
        """When backfill happens before finalize, final_score should reflect best."""
        collector = TrajectoryCollector("bf_test", "family")
        collector.set_initial_score(0.30)

        s1 = TrajectoryStep(
            round_num=1, strategy_id="v1", score_before=0.30,
            win_rate_before=0.40, signal_count_before=30,
        )
        s2 = TrajectoryStep(
            round_num=2, strategy_id="v2", score_before=0.45,
            win_rate_before=0.55, signal_count_before=40,
        )
        collector.record_step(s1)
        collector.record_step(s2)

        # Simulate backfill (as evolution.py now does before finalize)
        steps = collector._steps
        best_score = 0.50
        for i, step in enumerate(steps):
            if i + 1 < len(steps):
                nxt = steps[i + 1]
                step.score_after = nxt.score_before
                step.win_rate_after = nxt.win_rate_before
                step.signal_count_after = nxt.signal_count_before
            else:
                step.score_after = best_score
                step.win_rate_after = step.win_rate_before
                step.signal_count_after = step.signal_count_before
            step.score_delta = step.score_after - step.score_before
            step.improved = step.score_delta > 0

        traj = collector.finalize()

        # final_score should pick up the backfilled last step's score_after
        assert traj.final_score == best_score

        # Step 1 outcomes should match step 2's inputs
        assert traj.steps[0].score_after == 0.45
        assert traj.steps[0].win_rate_after == 0.55
        assert traj.steps[0].signal_count_after == 40
        assert traj.steps[0].improved is True  # 0.45 > 0.30

        # Step 2 (last) outcomes should be best_score
        assert traj.steps[1].score_after == best_score
        assert traj.steps[1].improved is True  # 0.50 > 0.45

    def test_improved_flag_reflects_delta_not_champion(self):
        """improved should be True only when score_delta > 0."""
        collector = TrajectoryCollector("imp_test", "family")
        collector.set_initial_score(0.50)

        # Round 1: score went up
        s1 = TrajectoryStep(
            round_num=1, strategy_id="v1", score_before=0.50,
            win_rate_before=0.55, signal_count_before=50,
        )
        # Round 2: score went down
        s2 = TrajectoryStep(
            round_num=2, strategy_id="v2", score_before=0.60,
            win_rate_before=0.60, signal_count_before=45,
        )
        collector.record_step(s1)
        collector.record_step(s2)

        steps = collector._steps
        # Backfill: s1.after = s2.before (0.60), s2.after = best (0.55, regression)
        steps[0].score_after = steps[1].score_before  # 0.60
        steps[0].score_delta = 0.10
        steps[0].improved = steps[0].score_delta > 0  # True

        steps[1].score_after = 0.55  # Regression
        steps[1].score_delta = 0.55 - 0.60  # -0.05
        steps[1].improved = steps[1].score_delta > 0  # False

        traj = collector.finalize()
        assert traj.steps[0].improved is True
        assert traj.steps[1].improved is False


# ── Trajectory field population tests ───────────────────────────────


class TestTrajectoryFieldPopulation:
    """Verify that critic_verdict and playbook_used survive roundtrip."""

    def test_critic_verdict_persists_in_export(self):
        """critic_verdict should appear in JSONL export."""
        traj = EvolutionTrajectory(
            trajectory_id="test_fields",
            strategy_family="fields_family",
            initial_score=0.30,
            final_score=0.45,
            total_rounds=1,
            steps=[
                TrajectoryStep(
                    round_num=1,
                    strategy_id="v1",
                    score_before=0.30,
                    score_after=0.45,
                    score_delta=0.15,
                    diagnosis="Low win rate",
                    hypothesis="Add RSI filter",
                    changes=[{"type": "tighten_filter"}],
                    improved=True,
                    critic_verdict="approved=2, rejected=1 [loosen_filter(exit.stop_loss.value): conflicts with low drawdown]",
                    playbook_used="pb_low_win_rate, pb_overfit_recovery",
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            result = export_jsonl(traj, path)
            data = json.loads(result.read_text().strip())
            assert data["action"]["critic_verdict"] == traj.steps[0].critic_verdict
            assert data["action"]["playbook_used"] == traj.steps[0].playbook_used

    def test_critic_verdict_in_sharegpt(self):
        """critic_verdict should appear in ShareGPT assistant text."""
        traj = EvolutionTrajectory(
            trajectory_id="sharegpt_fields",
            strategy_family="fields",
            initial_score=0.30,
            final_score=0.40,
            total_rounds=1,
            steps=[
                TrajectoryStep(
                    round_num=1,
                    strategy_id="v1",
                    score_before=0.30,
                    score_after=0.40,
                    score_delta=0.10,
                    diagnosis="Noisy entries",
                    hypothesis="Tighten filter",
                    changes=[{"type": "tighten_filter"}],
                    improved=True,
                    critic_verdict="approved=1, rejected=0",
                    playbook_used="pb_low_signals",
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sharegpt.jsonl"
            result = export_sharegpt(traj, path)
            content = result.read_text().strip()
            conv = json.loads(content)
            # The assistant message should contain the critic verdict and playbook
            assistant_text = conv["conversations"][1]["value"]
            assert "critic_verdict" in assistant_text or "approved=" in assistant_text

    def test_empty_fields_default_to_empty_string(self):
        """Fields should default to empty string when not populated."""
        step = TrajectoryStep(
            round_num=1,
            strategy_id="v1",
            score_before=0.30,
        )
        assert step.critic_verdict == ""
        assert step.playbook_used == ""


class TestContextBuilderPlaybookTracking:
    """Ensure ResearchContext.playbooks_used is populated."""

    def test_no_playbooks_when_no_store(self):
        builder = ContextBuilder()
        ctx = builder.build(_strategy(), _evaluation())
        assert ctx.playbooks_used == []

    def test_playbooks_used_is_list(self):
        """ResearchContext.playbooks_used should always be a list."""
        ctx = ResearchContext()
        assert isinstance(ctx.playbooks_used, list)
        assert len(ctx.playbooks_used) == 0
