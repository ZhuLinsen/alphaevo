"""Unit tests for L1/L2/L3 Stock RSI Engine, MAP-Elites, Scars & Heterogeneous Council."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from alphaevo.cli.demo import load_showcase_snapshot
from alphaevo.models.enums import ChangeType, EvidenceLevel, RSILevel
from alphaevo.models.execution import (
    EvaluationReport,
    OverallMetrics,
    ReflectionResult,
    StrategyChange,
)
from alphaevo.reflection.critic import SelfCritic
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore
from alphaevo.rsi.co_steer import ExecutableScar
from alphaevo.rsi.council import HeterogeneousCouncilConfig, ModelSpec, infer_model_family
from alphaevo.rsi.engine import StockRSIEngine
from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator
from alphaevo.rsi.program import parse_rsi_program
from alphaevo.rsi.qd_archive import MAPElitesArchive
from alphaevo.strategy.dsl.parser import StrategyParser


def _builtin_rsi_strategy():
    path = Path(__file__).resolve().parents[2] / "strategies" / "builtin" / "rsi_reversion.yaml"
    return StrategyParser().parse_file(path)


def test_parse_rsi_program_example() -> None:
    prog_path = Path(__file__).resolve().parents[2] / "examples" / "rsi_stock_program.md"
    charter = parse_rsi_program(prog_path)
    assert charter.name == "stock_rsi_reversion_charter"
    assert charter.seed_strategy_id == "rsi_reversion_v1"
    assert RSILevel.L1_STRATEGY in charter.allowed_levels
    assert RSILevel.L2_FACTOR in charter.allowed_levels
    assert charter.require_null_control is True


def test_map_elites_preserves_distinct_niches() -> None:
    archive = MAPElitesArchive()
    base = _builtin_rsi_strategy()
    ev = EvaluationReport(
        strategy_id=base.meta.id, overall=OverallMetrics(win_rate=0.55, signal_count=40)
    )

    short_hold = base.model_copy(deep=True)
    short_hold.meta.id = "short_cell"
    short_hold.exit.max_holding_days = 4

    swing_hold = base.model_copy(deep=True)
    swing_hold.meta.id = "swing_cell"
    swing_hold.exit.max_holding_days = 10

    swing_better = base.model_copy(deep=True)
    swing_better.meta.id = "swing_cell_better"
    swing_better.exit.max_holding_days = 10

    assert archive.insert(short_hold, ev, 0.45) is True
    assert archive.insert(swing_hold, ev, 0.50) is True
    assert archive.insert(swing_better, ev, 0.42) is False
    assert archive.insert(swing_better, ev, 0.62) is True

    summary = archive.coverage_summary()
    assert summary.total_cells_occupied == 2
    assert summary.best_strategy_id == "swing_cell_better"


def test_executable_scar_blocks_critic_and_experience_persists_evidence_level() -> None:
    strat = _builtin_rsi_strategy()
    scar = ExecutableScar(
        scar_id="scar_test_1",
        strategy_family=strat.meta.family_id,
        failure_mechanism="Narrowing RSI below 20 causes zero-signal lockout",
        forbidden_change_type=ChangeType.TIGHTEN_FILTER,
        target_pattern="rsi_14",
        op="min_value",
        threshold=22.0,
    )
    critic = SelfCritic(scars=[scar])
    ev = EvaluationReport(strategy_id=strat.meta.id, overall=OverallMetrics(signal_count=20))
    bad_change = StrategyChange(
        change_type=ChangeType.TIGHTEN_FILTER,
        target="entry.triggers[indicator=rsi_14].value",
        from_value=30.0,
        to_value=15.0,
    )
    verdict = critic.critique(
        strat,
        ev,
        ReflectionResult(
            strategy_id=strat.meta.id, evaluation_id="e1", proposed_changes=[bad_change]
        ),
    )
    assert len(verdict.rejected) == 1
    assert "scar_test_1" in verdict.rejected[0][1]

    store = ExperienceStore(":memory:")
    store.record(
        ExperienceRecord(
            strategy_family="rsi_reversion",
            strategy_id="rsi_reversion_v2",
            round_num=1,
            change_type=ChangeType.CHANGE_LOGIC,
            target="entry.logic",
            from_value="and",
            to_value="or",
            score_before=0.08,
            score_after=0.55,
            score_delta=0.47,
            worked=True,
            evidence_level=EvidenceLevel.NULL_PASSED,
        )
    )
    rows = store.get_family_lessons("rsi_reversion")
    assert len(rows) == 1
    assert rows[0].evidence_level == EvidenceLevel.NULL_PASSED


def test_heterogeneous_council_enforces_adversarial_family_separation() -> None:
    assert infer_model_family("gpt-6-astra") == "openai"
    assert infer_model_family("claude-opus-5-5-high") == "anthropic"
    assert infer_model_family("gemini-4-carbon") == "google"

    # Same non-deterministic family must raise ValidationError
    with pytest.raises(ValidationError, match="proposer_family != prosecutor_family"):
        HeterogeneousCouncilConfig(
            proposer=ModelSpec(model="openai/gpt-6-astra", backend="cli", cli_tool="codex"),
            prosecutor=ModelSpec(model="openai/gpt-6.1-sol", backend="litellm"),
        )

    # Distinct families pass
    valid_cfg = HeterogeneousCouncilConfig(
        proposer=ModelSpec(model="gpt-6-astra", backend="cli", cli_tool="codex"),
        builder=ModelSpec(model="gpt-6.1-sol", backend="litellm"),
        prosecutor=ModelSpec(model="claude-opus-5-5-high", backend="cli", cli_tool="agy"),
    )
    assert valid_cfg.proposer.family == "openai"
    assert valid_cfg.prosecutor.family == "anthropic"


def test_stock_rsi_engine_end_to_end_and_mechanism_ab(tmp_path: Path) -> None:
    data, _ = load_showcase_snapshot()
    strat = _builtin_rsi_strategy()
    out_dir = tmp_path / "rsi_out"

    charter = parse_rsi_program(
        Path(__file__).resolve().parents[2] / "examples" / "rsi_stock_program.md"
    )
    charter.max_rounds = 3
    charter.max_candidates_per_round = 3
    charter.null_permutations = 30

    engine = StockRSIEngine(db_path=tmp_path / "rsi.db")
    res = engine.run(strat, data, charter=charter, output_dir=out_dir)

    assert res.kernel_intact is True
    assert res.total_trials_recorded >= 2
    assert res.tsv_path is not None and res.tsv_path.is_file()
    assert res.report_path is not None and res.report_path.is_file()
    assert res.champion_yaml_path is not None and res.champion_yaml_path.is_file()
    assert res.champion_evaluation.overall.signal_count > 0
    assert res.champion_evaluation.overall.avg_return > 0.0

    mech = MechanismEvaluator().evaluate_ab(
        strat,
        {k: data[k] for k in list(data)[:3]},
        baseline_variant=HarnessVariant(
            variant_id="naive",
            name="Naive",
            require_null_control=False,
            enable_pinned_scars=False,
        ),
        candidate_variant=HarnessVariant(
            variant_id="guarded",
            name="Guarded",
            require_null_control=True,
            enable_pinned_scars=True,
        ),
        max_rounds=2,
    )
    assert mech.winner_variant_id in {"naive", "guarded"}
