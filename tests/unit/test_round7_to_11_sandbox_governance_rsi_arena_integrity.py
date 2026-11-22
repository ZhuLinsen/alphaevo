"""Regression tests for Rounds 7-11 hardening across sandbox, governance, RSI, council, and arena."""

from __future__ import annotations

import math
import subprocess
import threading
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
from alphaevo.alpha_factory.factory import _build_indicator_fn
from alphaevo.alpha_factory.governance import (
    FactorGovernanceConfig,
    FactorGovernanceEngine,
    _split_panel,
)
from alphaevo.alpha_factory.sandbox import (
    FactorSandbox,
    _execute_in_worker_namespace,
    build_restricted_namespace,
)
from alphaevo.arena import ArenaRunner
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.models.enums import ChangeType, MarketType, RSILevel, StrategyCategory
from alphaevo.models.execution import (
    AntiFitMetrics,
    EvaluationReport,
    OverallMetrics,
    StrategyChange,
)
from alphaevo.models.strategy import (
    ExecutionConfig,
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    TunableParam,
)
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore
from alphaevo.rsi.co_steer import _OFFLINE_FACTOR_TEMPLATES, BlindSpotMiner
from alphaevo.rsi.council import (
    CouncilProposal,
    HeterogeneousCouncil,
    HeterogeneousCouncilConfig,
    ModelSpec,
    ProsecutorAudit,
)
from alphaevo.rsi.crossover import SemanticCrossoverBreeder
from alphaevo.rsi.diagnostics import ConditionPassRate, StrategyBottleneckReport
from alphaevo.rsi.engine import StockRSIEngine
from alphaevo.rsi.mechanism import _temporal_split_with_warmup
from alphaevo.rsi.program import RSICharter
from alphaevo.strategy.store import StrategyStore


def _make_ohlcv(
    n_bars: int = 80,
    *,
    start: date = date(2025, 1, 2),
    seed: int = 42,
    drift: float = 0.001,
    osc_amp: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [start + timedelta(days=i) for i in range(n_bars)]
    closes: list[float] = []
    px = 100.0
    for i in range(n_bars):
        px = px * (
            1.0 + drift + osc_amp * math.sin(i * math.pi / 4.0) + float(rng.normal(0.0, 0.004))
        )
        closes.append(max(5.0, px))
    arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": arr * 0.998,
            "high": arr * 1.01,
            "low": arr * 0.99,
            "close": arr,
            "volume": np.full(n_bars, 1_500_000.0),
        }
    )


def _make_strategy(
    strat_id: str = "r7_11_base_v1",
    *,
    logic: str = "and",
    triggers: list[StrategyCondition] | None = None,
    conditions: list[StrategyCondition] | None = None,
    guards: list[StrategyCondition] | None = None,
    filters: list[StrategyCondition] | None = None,
    sl: StopLossConfig | None = None,
    tp: TakeProfitConfig | None = None,
    exit_triggers: list[StrategyCondition] | None = None,
    tunable: list[TunableParam] | None = None,
) -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id=strat_id,
            name=f"Strat {strat_id}",
            market=MarketType.US,
            category=StrategyCategory.TREND,
        ),
        description="Synthetic strategy for Rounds 7-11 hardening tests.",
        entry=StrategyEntry(
            logic="or" if logic == "or" else "and",
            triggers=triggers or [],
            conditions=conditions
            if conditions is not None
            else [StrategyCondition(indicator="rsi_14", op="<=", value=60.0)],
            guards=guards or [],
            filters=filters or [],
            execution=ExecutionConfig(timing="next_open", slippage=0.001),
        ),
        exit=StrategyExit(
            triggers=exit_triggers or [],
            stop_loss=sl or StopLossConfig(type="pct", value=0.05),
            take_profit=tp or TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=8,
        ),
        params=StrategyParams(tunable=tunable or []),
    )


# ---------------------------------------------------------------------------
# Round 7: FactorSandbox AST, Restricted Builtins, & Spawn Worker Isolation
# ---------------------------------------------------------------------------


def test_round7_sandbox_ast_hardening_and_worker_cache() -> None:
    ns_builtins = build_restricted_namespace()["__builtins__"]
    for blocked_builtin in ("type", "hasattr", "id"):
        assert blocked_builtin not in ns_builtins

    with FactorSandbox(timeout_seconds=10) as sb:
        dangerous_snippets = [
            "from pandas.io.pickle import read_pickle\ndef compute(df, idx): return 0.0",
            "from . import helper\ndef compute(df, idx): return 0.0",
            "from math import *\ndef compute(df, idx): return 0.0",
            "def compute(df, idx):\n    df.to_csv('/tmp/leak.csv')\n    return 0.0",
            "def compute(df, idx):\n    return float(len(df.query('close > 0')))",
            "def compute(df, idx):\n    g = (x for x in [1])\n    return float(bool(g.gi_frame))",
            "def compute(df, idx):\n    return float(bool(type(df)))",
            "def compute(df, idx):\n    return float(bool(hasattr(df, 'close')))",
            "def compute(df, idx):\n    return float(id(df))",
            "def compute(df, idx):\n    return float(bool(''.__eq__('a')))",
        ]
        for snippet in dangerous_snippets:
            ok, msg = sb.preflight_code(snippet)
            assert not ok, f"Expected rejection for snippet:\n{snippet}"
            assert msg

        bad_sig = "def compute(df):\n    return 1.0"
        ok_sig, _ = sb.preflight_code(bad_sig)
        assert not ok_sig

        # Verify _build_indicator_fn executes strictly inside isolated worker with PIT slicing + LRU cache
        lookfuture_code = (
            "def compute(df, idx):\n"
            "    return float(len(df) - 1 - idx + float(df['close'].iloc[idx]))\n"
        )
        df = _make_ohlcv(30, seed=11)
        fn = _build_indicator_fn(lookfuture_code, sb)
        with patch.object(sb, "execute", wraps=sb.execute) as spy_exec:
            v1 = fn(df, 10)
            v2 = fn(df, 12)
            assert spy_exec.call_count == 1  # second call hits series LRU cache
            assert spy_exec.call_args.kwargs.get("enforce_pit_slice") is True
            assert v1 == pytest.approx(float(df["close"].iloc[10]))
            assert v2 == pytest.approx(float(df["close"].iloc[12]))

            # Mutating df in-place invalidates blake2b cache key cleanly
            df.iloc[12, df.columns.get_loc("close")] = 777.5
            v2_mut = fn(df, 12)
            assert spy_exec.call_count == 2
            assert v2_mut == pytest.approx(777.5)


# ---------------------------------------------------------------------------
# Round 8: FactorGovernanceEngine Purged/Warmup Split & Negative L2 Direction
# ---------------------------------------------------------------------------


def test_round8_governance_split_warmup_no_leak_and_negative_direction(tmp_path: Path) -> None:
    # Build staggered dates panel (A starts earlier, B starts later) so calendar alignment is exercised
    base_start = date(2025, 1, 2)
    market_data: dict[str, pd.DataFrame] = {}
    for sym_idx, sym in enumerate(("GOV_A", "GOV_B", "GOV_C")):
        n_bars = 120
        dates = [base_start + timedelta(days=i + sym_idx * 3) for i in range(n_bars)]
        # Synthetic cyclic compression -> larger compression precedes lower forward returns (inverse relation)
        closes = [
            100.0 * (1.0 + 0.04 * math.sin((i + sym_idx) * math.pi / 4.0)) for i in range(n_bars)
        ]
        arr = np.asarray(closes, dtype=float)
        spread = np.array([0.04 if (i % 4 == 0) else 0.005 for i in range(n_bars)], dtype=float)
        market_data[sym] = pd.DataFrame(
            {
                "date": dates,
                "open": arr,
                "high": arr * (1.0 + spread),
                "low": arr * (1.0 - spread),
                "close": arr,
                "volume": np.full(n_bars, 1_000_000.0 + sym_idx * 5_000.0),
            }
        )

    split = _split_panel(market_data, train_ratio=0.60, forward_horizon=5)
    assert "_fwd_ret" not in split.train_ohlcv.columns
    assert "_fwd_ret" not in split.oos_eval_ohlcv.columns
    train_dates = set(pd.to_datetime(split.train_ohlcv["date"]).dt.date)
    oos_only_dates = set(pd.to_datetime(split.oos_eval_ohlcv.loc[split.oos_mask, "date"]).dt.date)
    assert train_dates.isdisjoint(oos_only_dates)
    # OOS warmup prefix exists before masked OOS window
    assert (~split.oos_mask).sum() > 0

    # Verify offline negative-direction L2 templates & LLM direction preservation
    tpl_dirs = {t.name: t.expected_direction for t in _OFFLINE_FACTOR_TEMPLATES}
    assert tpl_dirs["intraday_range_compression_10d"] == "negative"
    assert tpl_dirs["downside_vol_ratio_10d"] == "negative"
    assert tpl_dirs["atr_expansion_ratio_14d"] == "negative"

    fake_synth = SimpleNamespace(
        generate=lambda **kwargs: [
            SimpleNamespace(
                name="inverse_range_compression_test",
                code=(
                    "def compute(df, idx):\n"
                    "    if idx < 1:\n"
                    "        return 0.0\n"
                    "    return float((df['high'].iloc[idx] - df['low'].iloc[idx]) / max(1e-6, df['close'].iloc[idx]))\n"
                ),
                description="High spread precedes reversion",
                expected_direction="negative",
            )
        ]
    )
    strat = _make_strategy("co_dir_strat_v1")
    weak_eval = EvaluationReport(
        strategy_id=strat.meta.id,
        overall=OverallMetrics(win_rate=0.42, avg_return=-0.01, signal_count=20),
    )
    fname = "inverse_range_compression_test"
    try:
        with BlindSpotMiner(synthesizer=fake_synth) as miner:
            decision = miner.steer(
                strat,
                weak_eval,
                market_data["GOV_A"],
                allowed_levels=[RSILevel.L2_FACTOR],
                stagnant_l1_rounds=3,
                persist=False,
            )
        assert decision.synthesized_factor_name == fname
        assert decision.persisted_to_store is False
        assert decision.synthesized_factor_record is not None
        assert decision.synthesized_factor_record.expected_direction == "negative"
    finally:
        IndicatorRegistry.unregister_dynamic(fname)

    # Confirm governance reads no _fwd_ret leak column from df
    db_path = tmp_path / "gov_r8.db"
    with FactorStore(db_path) as fs:
        fs.save(
            FactorRecord(
                name="fwd_ret_sniffer",
                description="Must observe zero _fwd_ret column",
                rationale="Leak test",
                code=(
                    "def compute(df, idx):\n    return 1.0 if '_fwd_ret' in df.columns else 0.0\n"
                ),
                expected_direction="positive",
            )
        )
    with FactorGovernanceEngine(
        db_path=db_path,
        config=FactorGovernanceConfig(
            train_ratio=0.60,
            forward_horizon=5,
            min_oos_ic=0.02,
            min_oos_ir=0.10,
            require_pit_invariance=False,
        ),
    ) as gov:
        rep = gov.audit_factors(market_data, strategies=[strat], auto_retire=False)
    by_name = {e.name: e for e in rep.entries}
    assert by_name["fwd_ret_sniffer"].health == "decayed"
    assert by_name["fwd_ret_sniffer"].oos_ic == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Round 9: StockRSIEngine Fail-Closed Holdout, Non-Starvation, Deferred L2 & SQLite
# ---------------------------------------------------------------------------


def test_round9_rsi_holdout_fail_closed_and_non_starving_promotion(tmp_path: Path) -> None:
    short_market = {"SHORT": _make_ohlcv(20, seed=1)}
    strat = _make_strategy("holdout_fail_v1")
    with StockRSIEngine(db_path=":memory:", enable_crossover=False) as engine:
        charter = RSICharter(
            name="strict_holdout",
            seed_strategy_id=strat.meta.id,
            locked_holdout_pct=0.25,
            max_rounds=1,
            null_permutations=10,
        )
        with pytest.raises(ValueError):
            engine.run(strat, short_market, charter=charter)

    # Non-starvation: higher-fitness Candidate A violates overfit gate; lower-fitness B passes all gates
    market_data = {"M1": _make_ohlcv(90, seed=7), "M2": _make_ohlcv(90, seed=13)}
    base = _make_strategy(
        "starve_seed_v1",
        conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=55.0)],
    )
    high_fail_change = [
        StrategyChange(
            change_type=ChangeType.LOOSEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=55.0,
            to_value=70.0,
            reason="Candidate A high raw fitness but overfit",
        )
    ]
    pass_change = [
        StrategyChange(
            change_type=ChangeType.LOOSEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=55.0,
            to_value=62.0,
            reason="Candidate B passes all L0 gates",
        )
    ]

    with StockRSIEngine(db_path=":memory:", enable_crossover=False) as engine:
        charter = RSICharter(
            name="non_starvation",
            seed_strategy_id=base.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=1,
            max_candidates_per_round=2,
            min_signals=5,
            max_drawdown=0.45,
            min_dsr_prob=0.0,
            require_null_control=False,
        )

        def _fake_generate(parent, ev, **kwargs):
            del parent, ev, kwargs
            return [
                SimpleNamespace(
                    proposal=CouncilProposal(
                        proposer_model="m1",
                        proposer_family="f1",
                        rationale="High overfit cand A",
                        changes=high_fail_change,
                    ),
                    audit=ProsecutorAudit(
                        prosecutor_model="p1",
                        prosecutor_family="f2",
                        approved=True,
                    ),
                ),
                SimpleNamespace(
                    proposal=CouncilProposal(
                        proposer_model="m1",
                        proposer_family="f1",
                        rationale="Clean passing cand B",
                        changes=pass_change,
                    ),
                    audit=ProsecutorAudit(
                        prosecutor_model="p1",
                        prosecutor_family="f2",
                        approved=True,
                    ),
                ),
            ]

        def _fake_eval(self_arg, *, strategy, **kwargs):
            del self_arg, kwargs
            thresh = float(strategy.entry.conditions[0].value)
            if strategy.meta.id == base.meta.id:
                ov = OverallMetrics(
                    win_rate=0.45, avg_return=0.002, max_drawdown=0.08, signal_count=20
                )
                ao = AntiFitMetrics()
            elif thresh >= 69.0:
                # Higher raw fitness (win_rate=0.65, avg_return=0.04) but fails overfit gate!
                ov = OverallMetrics(
                    win_rate=0.65, avg_return=0.04, max_drawdown=0.05, signal_count=25
                )
                ao = AntiFitMetrics(train_val_gap=0.25)
            else:
                # Slightly lower fitness than A, higher than baseline, cleanly passes all gates!
                ov = OverallMetrics(
                    win_rate=0.58, avg_return=0.02, max_drawdown=0.06, signal_count=22
                )
                ao = AntiFitMetrics()
            ev = EvaluationReport(
                strategy_id=strategy.meta.id,
                overall=ov,
                anti_overfit=ao,
                confidence_score=ov.win_rate,
                deflated_sharpe_prob=0.80,
                null_control_passed=True,
            )
            return SimpleNamespace(
                evaluation=ev,
                null_control=None,
                portfolio_verdict=None,
                trial_record=None,
                strategy_rerun_null=None,
            )

        with (
            patch.object(engine._council, "generate_candidates", side_effect=_fake_generate),
            patch.object(
                StockRSIEngine, "_eval_with_kernel", autospec=True, side_effect=_fake_eval
            ),
        ):
            res = engine.run(base, market_data, charter=charter, persist_champion=False)

    assert res.promotions == 1
    assert res.champion_strategy.entry.conditions[0].value == pytest.approx(62.0)


def test_round9_deferred_l2_unregister_on_revert_and_sqlite_concurrency(tmp_path: Path) -> None:
    db_path = tmp_path / "l2_defer.db"
    market_data = {"M1": _make_ohlcv(70, seed=3)}
    strat = _make_strategy("l2_defer_v1")
    transient_factor = "transient_reverted_l2_test"

    def _fake_steer(parent, ev, sample_df, **kwargs):
        del parent, ev, sample_df, kwargs
        code = "def compute(df, idx):\n    return 1.0\n"
        IndicatorRegistry.register_dynamic(transient_factor, lambda df, idx, ctx=None: 1.0)
        return SimpleNamespace(
            chosen_level=RSILevel.L2_FACTOR,
            cluster=SimpleNamespace(bottleneck_hint=""),
            synthesized_factor_name=transient_factor,
            synthesized_via="offline_template",
            pit_verified=True,
            persisted_to_store=False,
            synthesized_factor_record=FactorRecord(
                name=transient_factor,
                description="Transient factor",
                rationale="Should unregister and skip store on revert",
                code=code,
                expected_direction="positive",
            ),
            proposed_changes=[
                StrategyChange(
                    change_type=ChangeType.ADD_CONDITION,
                    target="entry.filters",
                    from_value=None,
                    to_value={"indicator": transient_factor, "op": ">=", "value": 0.5},
                    reason="Add transient L2 filter",
                )
            ],
        )

    try:
        with StockRSIEngine(db_path=db_path, enable_crossover=False) as engine:
            charter = RSICharter(
                name="l2_revert",
                seed_strategy_id=strat.meta.id,
                allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
                max_rounds=1,
                max_candidates_per_round=1,
                min_signals=500,  # force revert
                require_null_control=False,
            )
            with (
                patch.object(engine._miner, "steer", side_effect=_fake_steer),
                patch.object(
                    engine._council,
                    "generate_candidates",
                    return_value=[
                        SimpleNamespace(
                            proposal=CouncilProposal(
                                proposer_model="m1",
                                proposer_family="f1",
                                rationale="Use transient L2",
                                changes=[
                                    StrategyChange(
                                        change_type=ChangeType.ADD_CONDITION,
                                        target="entry.filters",
                                        from_value=None,
                                        to_value={
                                            "indicator": transient_factor,
                                            "op": ">=",
                                            "value": 0.5,
                                        },
                                        reason="Test transient revert",
                                    )
                                ],
                            ),
                            audit=ProsecutorAudit(
                                prosecutor_model="p1",
                                prosecutor_family="f2",
                                approved=True,
                            ),
                        )
                    ],
                ),
            ):
                res = engine.run(strat, market_data, charter=charter, persist_champion=False)

        assert res.promotions == 0
        assert res.persisted_factors == []
        assert not IndicatorRegistry.is_registered(transient_factor)
        with FactorStore(db_path) as check_fs:
            assert check_fs.get(transient_factor) is None
    finally:
        IndicatorRegistry.unregister_dynamic(transient_factor)

    # SQLite thread safety across :memory: stores & context manager cleanup on file stores
    with (
        StrategyStore(":memory:") as ms,
        ExperienceStore(":memory:") as me,
        FactorStore(":memory:") as mf,
    ):
        errors: list[Exception] = []

        def _worker(tid: int) -> None:
            try:
                s = _make_strategy(f"thread_strat_{tid}")
                ms.save(s)
                assert ms.get(s.meta.id) is not None
                me.record(
                    ExperienceRecord(
                        strategy_family="f",
                        strategy_id=s.meta.id,
                        round_num=tid,
                        change_type=ChangeType.LOOSEN_FILTER,
                        target="entry.conditions[0].value",
                        to_value=60.0,
                    )
                )
                mf.save(
                    FactorRecord(
                        name=f"tf_{tid}",
                        description="t",
                        rationale="t",
                        code="def compute(df, idx): return 0.0",
                    )
                )
                assert mf.get(f"tf_{tid}") is not None
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# Round 10: Council Committee & Fail-Closed Prosecutors, Crossover, Mechanism
# ---------------------------------------------------------------------------


def test_round10_council_crossover_and_mechanism_cutoff() -> None:
    strat = _make_strategy("council_r10_v1")
    overfit_eval = EvaluationReport(
        strategy_id=strat.meta.id,
        overall=OverallMetrics(win_rate=0.45, avg_return=0.001, signal_count=20),
        anti_overfit=AntiFitMetrics(train_val_gap=0.25),
    )
    council = HeterogeneousCouncil()
    add_change = [
        StrategyChange(
            change_type=ChangeType.ADD_CONDITION,
            target="entry.filters",
            from_value=None,
            to_value={"indicator": "close_above_ma20", "op": "==", "value": True},
            reason="Add condition while overfit",
        )
    ]
    with patch.object(
        council._committee,
        "review",
        return_value=SimpleNamespace(overall_status="fail"),
    ):
        aud = council._run_prosecutor(strat, overfit_eval, add_change)
    assert aud.approved is False
    assert any("committee: overall_status=fail" in o for o in aud.objections)

    # Heuristic relaxation for <= with negative and zero thresholds
    strat_neg = _make_strategy(
        "neg_thresh_v1",
        guards=[
            StrategyCondition(indicator="neg_ind", op="<=", value=-0.05),
            StrategyCondition(indicator="zero_ind", op="<", value=0.0),
        ],
    )
    bn = StrategyBottleneckReport(
        strategy_id=strat_neg.meta.id,
        entry_logic=strat_neg.entry.logic,
        total_evaluated_bars=100,
        conditions=[
            ConditionPassRate(
                group="guards",
                indicator="neg_ind",
                op="<=",
                value=-0.05,
                pass_rate=0.04,
            ),
            ConditionPassRate(
                group="guards",
                indicator="zero_ind",
                op="<",
                value=0.0,
                pass_rate=0.03,
            ),
        ],
    )
    props = council._build_heuristic_proposals(
        strat_neg,
        EvaluationReport(
            strategy_id=strat_neg.meta.id,
            overall=OverallMetrics(win_rate=0.4, signal_count=2),
        ),
        ModelSpec(model="heuristic"),
        bottleneck=bn,
    )
    relaxed_targets = {c.target: c.to_value for p in props for c in p.changes}
    assert relaxed_targets["entry.guards[indicator=neg_ind].value"] == pytest.approx(-0.0275)
    assert relaxed_targets["entry.guards[indicator=zero_ind].value"] == pytest.approx(0.1)

    # Fail-closed CLI prosecutor on error & malformed output
    cli_council = HeterogeneousCouncil(
        config=HeterogeneousCouncilConfig(
            proposer=ModelSpec(model="openai/gpt-6-astra"),
            prosecutor=ModelSpec(
                model="anthropic/claude-opus-5.5", backend="cli", cli_tool="codex"
            ),
        )
    )
    with patch("alphaevo.rsi.council.subprocess.run") as mock_sub:
        mock_sub.return_value = subprocess.CompletedProcess(
            args=["codex"], returncode=1, stdout="", stderr="boom"
        )
        veto_err = cli_council._try_cli_prosecutor(
            strat, overfit_eval, add_change, cli_council.config.prosecutor
        )
        assert veto_err and "fail-closed" in veto_err[0]
        mock_sub.return_value = subprocess.CompletedProcess(
            args=["codex"], returncode=0, stdout="not json", stderr=""
        )
        veto_bad = cli_council._try_cli_prosecutor(
            strat, overfit_eval, add_change, cli_council.config.prosecutor
        )
        assert veto_bad and "fail-closed" in veto_bad[0]

    # Fail-closed LiteLLM prosecutor when api_key is configured, offline fallback when None
    llm_council = HeterogeneousCouncil(
        config=HeterogeneousCouncilConfig(
            proposer=ModelSpec(model="openai/gpt-6-astra"),
            prosecutor=ModelSpec(model="anthropic/claude-opus-5.5", backend="litellm"),
        ),
        api_key="sk-configured-test",
    )
    from alphaevo.core.llm import LLMClient

    with patch.object(LLMClient, "chat_json", side_effect=RuntimeError("timeout")):
        llm_veto = llm_council._try_litellm_prosecutor(
            strat, overfit_eval, add_change, llm_council.config.prosecutor
        )
        assert llm_veto and "fail-closed" in llm_veto[0]

    offline_council = HeterogeneousCouncil(
        config=HeterogeneousCouncilConfig(
            proposer=ModelSpec(model="openai/gpt-6-astra"),
            prosecutor=ModelSpec(model="anthropic/claude-opus-5.5", backend="litellm"),
        ),
        api_key=None,
    )
    with patch("alphaevo.core.config.ConfigManager.get_llm_api_key", return_value=None):
        assert (
            offline_council._try_litellm_prosecutor(
                strat, overfit_eval, add_change, offline_council.config.prosecutor
            )
            == []
        )

    # Fail-closed CLI prosecutor when cli_tool is unconfigured/unsupported
    bad_cli_council = HeterogeneousCouncil(
        config=HeterogeneousCouncilConfig(
            proposer=ModelSpec(model="openai/gpt-6-astra"),
            prosecutor=ModelSpec(model="anthropic/claude-opus-5.5", backend="cli", cli_tool=None),
        )
    )
    bad_cli_veto = bad_cli_council._try_cli_prosecutor(
        strat, overfit_eval, add_change, bad_cli_council.config.prosecutor
    )
    assert bad_cli_veto and "misconfigured" in bad_cli_veto[0]

    # Crossover exit_graft: summarizes ALL exit fields, passes SelfCritic, prunes dangling tunables;
    # trigger_blend preserves multi-AND conditions in entry.guards instead of loosening them under OR.
    p_strat = _make_strategy(
        "p_xover",
        logic="and",
        conditions=[
            StrategyCondition(indicator="rsi_14", op="<=", value=40.0),
            StrategyCondition(indicator="close_above_ma20", op="==", value=True),
        ],
        tp=TakeProfitConfig(type="target_ma", target="ma60"),
        tunable=[
            TunableParam(target="exit.take_profit.target", range=(20, 80), step=5),
            TunableParam(target="exit.stop_loss.value", range=(0.02, 0.08), step=0.01),
            TunableParam(target="exit.max_holding_days", range=(5, 20), step=5),
        ],
    )
    s_strat = _make_strategy(
        "s_xover",
        conditions=[StrategyCondition(indicator="momentum_10d", op=">", value=0.02)],
        sl=StopLossConfig(type="atr", multiplier=2.0, atr_period=21),
        tp=TakeProfitConfig(type="pct", value=0.12),
        exit_triggers=[StrategyCondition(indicator="close_below_ma10", op="==", value=True)],
    )
    children = SemanticCrossoverBreeder().breed(p_strat, s_strat, max_children=3)
    by_op = {c.operator: c for c in children}
    assert "trigger_blend" in by_op
    blend_child = by_op["trigger_blend"].child_strategy
    assert blend_child.entry.logic == "or"
    assert blend_child.entry.conditions == []
    assert [c.indicator for c in blend_child.entry.triggers] == ["rsi_14", "momentum_10d"]
    assert [c.indicator for c in blend_child.entry.guards] == ["close_above_ma20"]

    exit_child = by_op["exit_graft"].child_strategy
    assert all(p.target != "exit.take_profit.target" for p in exit_child.params.tunable)
    assert all(p.target != "exit.stop_loss.value" for p in exit_child.params.tunable)
    assert any(p.target == "exit.max_holding_days" for p in exit_child.params.tunable)
    summary_targets = {c.target for c in by_op["exit_graft"].summary_changes}
    assert {
        "exit.stop_loss.atr_period",
        "exit.take_profit.target",
        "exit.triggers",
    } <= summary_targets
    exit_aud = council.audit_changes(
        p_strat,
        EvaluationReport(
            strategy_id=p_strat.meta.id,
            overall=OverallMetrics(win_rate=0.52, avg_return=0.015, signal_count=25),
        ),
        by_op["exit_graft"].summary_changes,
    )
    assert exit_aud.approved is True, exit_aud.objections

    # Mechanism unified calendar cutoff across staggered symbols even when an undated symbol is present
    mech_data = {
        "EARLY": _make_ohlcv(80, start=date(2025, 1, 1), seed=1),
        "LATE": _make_ohlcv(80, start=date(2025, 1, 10), seed=2),
        "UNDATED": _make_ohlcv(80, start=date(2025, 1, 1), seed=3).drop(columns=["date"]),
    }
    _tr, _fwd, cutoffs = _temporal_split_with_warmup(mech_data, train_ratio=0.65, warmup_bars=10)
    assert cutoffs["EARLY"] == cutoffs["LATE"]
    assert "UNDATED" in cutoffs


# ---------------------------------------------------------------------------
# Round 11: ArenaRunner Strict Out-of-Sample Evolution vs Unseen Shadow Replay
# ---------------------------------------------------------------------------


def test_round11_arena_evolves_strictly_before_shadow_cutoff() -> None:
    strat = _make_strategy("arena_oos_v1")
    clean_market = {
        "US_1": _make_ohlcv(80, start=date(2025, 1, 2), seed=10, drift=0.002),
        "US_2": _make_ohlcv(80, start=date(2025, 1, 2), seed=20, drift=0.002),
    }
    perturbed_market = {k: v.copy() for k, v in clean_market.items()}
    # Severely corrupt trailing 30% (>= bar 56): future bars after shadow cutoff must NOT alter champion or metrics
    for _sym, df in perturbed_market.items():
        df.loc[58:, ["open", "high", "low", "close"]] = 1.0

    snap_clean = ArenaRunner().run(
        [strat], clean_market, rounds=1, enable_l2_factors=False, enable_crossover=False, seed=42
    )
    snap_perturbed = ArenaRunner().run(
        [strat],
        perturbed_market,
        rounds=1,
        enable_l2_factors=False,
        enable_crossover=False,
        seed=42,
    )

    e_clean = snap_clean.entries[0]
    e_pert = snap_perturbed.entries[0]
    assert e_clean.champion_yaml == e_pert.champion_yaml
    assert e_clean.baseline_score == pytest.approx(e_pert.baseline_score)
    assert e_clean.champion_score == pytest.approx(e_pert.champion_score)
    assert e_clean.equity_curve == e_pert.equity_curve
    assert e_clean.shadow_mature_signals >= 1
    assert e_clean.shadow_chain_verified is True

    # Short (<25 bar) panel cannot separate unseen shadow replay from evolution:
    # shadow replay must be skipped and reported truthfully as unverified.
    tiny_market = {"US_SHORT": _make_ohlcv(20, start=date(2025, 1, 2), seed=9)}
    snap_tiny = ArenaRunner().run(
        [strat], tiny_market, rounds=1, enable_l2_factors=False, enable_crossover=False, seed=42
    )
    assert snap_tiny.entries[0].shadow_chain_verified is False
    assert snap_tiny.chain_verified is False


def test_round7_to_11_codex_hardening_invariants(tmp_path: Path) -> None:
    import multiprocessing

    from alphaevo.alpha_factory.sandbox import _sandbox_worker_main
    from alphaevo.backtest.engine import BacktestResult
    from alphaevo.models.enums import SignalDirection
    from alphaevo.models.execution import TradeSignal
    from alphaevo.truth.ledger import compute_deflated_sharpe

    # 1. Worker proxied modules + AST IO attributes + worker sys.addaudithook runtime guard
    ns = build_restricted_namespace()
    with pytest.raises(AttributeError):
        _ = ns["np"].loadtxt
    with pytest.raises(AttributeError):
        _ = ns["pd"].read_csv

    with FactorSandbox(timeout_seconds=10) as sb:
        for snippet in (
            "import numpy as np\ndef compute(df, idx): return float(len(np.loadtxt('x')))",
            "def compute(df, idx): return float(df['close'].to_numpy().base is not None)",
            "def compute(df, idx): return float(bool(df._mgr))",
            "import pandas as pd\ndef compute(df, idx): return float(bool(pd.read_json('{}')))",
        ):
            ok, _ = sb.preflight_code(snippet)
            assert not ok, f"Expected AST rejection for {snippet}"

    # Verify spawned worker audit hook raises PermissionError even if code bypasses AST check
    spawn_ctx = multiprocessing.get_context("spawn")
    p_conn, c_conn = spawn_ctx.Pipe(duplex=True)
    proc = spawn_ctx.Process(target=_sandbox_worker_main, args=(c_conn, 256), daemon=True)
    proc.start()
    c_conn.close()
    try:
        df_small = _make_ohlcv(12, seed=5)
        blocked_out = tmp_path / "blocked.csv"
        p_conn.send(
            (
                f"def compute(df, idx):\n    df.to_csv({str(blocked_out)!r})\n    return 1.0",
                df_small,
                False,
            )
        )
        assert p_conn.poll(5.0)
        status, payload = p_conn.recv()
        assert status == "err"
        assert "PermissionError" in str(payload)
        assert not blocked_out.exists()

        # Deep PIT copy isolates underlying array buffer to [: idx + 1]
        values = _execute_in_worker_namespace(
            "def compute(df, idx):\n    return float(len(df['close'].to_numpy().tolist()))\n",
            df_small,
            enforce_pit_slice=True,
        )
        assert values == [float(i + 1) for i in range(len(df_small))]
        assert df_small.iloc[:4].copy(deep=True)["close"].to_numpy().shape == (4,)
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            p_conn.send(None)
        p_conn.close()
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)

    # 2. Governance keeps healthy negative-direction factor using true signed IR (not negated std)
    base_start = date(2025, 1, 2)
    gov_market: dict[str, pd.DataFrame] = {}
    for sym_idx, sym in enumerate(("NEG1", "NEG2", "NEG3")):
        n_bars = 140
        dates = [base_start + timedelta(days=i) for i in range(n_bars)]
        arr = np.array(
            [
                100.0 * (1.0 + 0.04 * math.sin((i + sym_idx * 1.7) * math.pi / 5.0))
                for i in range(n_bars)
            ],
            dtype=float,
        )
        gov_market[sym] = pd.DataFrame(
            {
                "date": dates,
                "open": arr * 0.999,
                "high": arr * 1.005,
                "low": arr * 0.995,
                "close": arr,
                "volume": np.full(n_bars, 1_000_000.0),
            }
        )
    neg_db = tmp_path / "neg_gov.db"
    with FactorStore(neg_db) as fs:
        fs.save(
            FactorRecord(
                name="inverse_mean_reversion_healthy",
                description="High above 5d mean predicts negative 5d forward return",
                rationale="Negative expected_direction IC & IR test",
                code=(
                    "def compute(df, idx):\n"
                    "    if idx < 4:\n"
                    "        return 0.0\n"
                    "    w = df['close'].iloc[idx - 4 : idx + 1]\n"
                    "    return float(df['close'].iloc[idx] / w.mean() - 1.0)\n"
                ),
                expected_direction="negative",
            )
        )
    with FactorGovernanceEngine(
        db_path=neg_db,
        config=FactorGovernanceConfig(
            train_ratio=0.55,
            forward_horizon=5,
            min_oos_ic=0.05,
            min_oos_ir=0.20,
            require_pit_invariance=True,
        ),
    ) as gov_eng:
        neg_rep = gov_eng.audit_factors(
            gov_market, strategies=[_make_strategy("ref_neg")], auto_retire=True
        )
    neg_entry = neg_rep.entries[0]
    assert neg_entry.health == "healthy", neg_entry.reasons
    assert neg_entry.oos_ic < -0.05
    assert neg_entry.oos_ir < -0.20
    assert neg_entry.action_taken == "kept"

    # 3. Co-STEER rejects LLM factor names colliding with built-in indicators
    colliding_synth = SimpleNamespace(
        generate=lambda **kwargs: [
            SimpleNamespace(
                name="rsi_14",
                code="def compute(df, idx):\n    return 50.0\n",
                description="Built-in collision attempt",
                expected_direction="positive",
            )
        ]
    )
    df_sample = gov_market["NEG1"]
    strat_no_rsi = _make_strategy(
        "collision_guard_v1",
        conditions=[StrategyCondition(indicator="close_above_ma20", op="==", value=True)],
    )
    with BlindSpotMiner(synthesizer=colliding_synth) as miner:
        col_dec = miner.steer(
            strat_no_rsi,
            EvaluationReport(
                strategy_id=strat_no_rsi.meta.id,
                overall=OverallMetrics(win_rate=0.40, avg_return=-0.01, signal_count=20),
            ),
            df_sample,
            allowed_levels=[RSILevel.L2_FACTOR],
            stagnant_l1_rounds=3,
        )
    assert col_dec.synthesized_via != "llm"
    assert "rsi_14" not in IndicatorRegistry.dynamic_names()

    # 4. Round-end DSR refresh prevents early candidate promotion under stale trial count
    dsr_market = {"D1": _make_ohlcv(90, seed=19), "D2": _make_ohlcv(90, seed=29)}
    base_dsr = _make_strategy(
        "dsr_round_end_v1",
        conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=55.0)],
    )
    with StockRSIEngine(db_path=":memory:", enable_crossover=False) as dsr_engine:
        charter_dsr = RSICharter(
            name="dsr_refresh",
            seed_strategy_id=base_dsr.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=1,
            max_candidates_per_round=2,
            min_signals=5,
            max_drawdown=0.5,
            min_dsr_prob=0.70,
            require_null_control=False,
        )
        ch1 = [
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=55.0,
                to_value=60.0,
                reason="Candidate 1 early DSR",
            )
        ]
        ch2 = [
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=55.0,
                to_value=65.0,
                reason="Candidate 2 bumps family trials",
            )
        ]

        def _fake_gen_dsr(parent, ev, **kwargs):
            del parent, ev, kwargs
            return [
                SimpleNamespace(
                    proposal=CouncilProposal(
                        proposer_model="m1", proposer_family="f1", rationale="C1", changes=ch1
                    ),
                    audit=ProsecutorAudit(
                        prosecutor_model="p1", prosecutor_family="f2", approved=True
                    ),
                ),
                SimpleNamespace(
                    proposal=CouncilProposal(
                        proposer_model="m1", proposer_family="f1", rationale="C2", changes=ch2
                    ),
                    audit=ProsecutorAudit(
                        prosecutor_model="p1", prosecutor_family="f2", approved=True
                    ),
                ),
            ]

        trials_seen = {"n": 0}

        def _fake_eval_dsr(self_arg, *, strategy, **kwargs):
            del self_arg, kwargs
            if strategy.meta.id == base_dsr.meta.id:
                trials_seen["n"] = 1
                ov = OverallMetrics(
                    win_rate=0.45, avg_return=0.001, max_drawdown=0.05, signal_count=20
                )
                dsr_p = 0.40
                f_trials = 1
            elif abs(float(strategy.entry.conditions[0].value) - 60.0) < 1e-6:
                trials_seen["n"] = 2
                ov = OverallMetrics(
                    win_rate=0.60, avg_return=0.02, max_drawdown=0.05, signal_count=20
                )
                dsr_p = 0.85  # passes at n_trials=2, fails when round-end trials becomes 3!
                f_trials = 2
            else:
                trials_seen["n"] = 3
                ov = OverallMetrics(
                    win_rate=0.50, avg_return=0.005, max_drawdown=0.05, signal_count=20
                )
                dsr_p = 0.30
                f_trials = 3
            sigs = [
                TradeSignal(
                    symbol="D1",
                    signal_date=date(2025, 1, 2) + timedelta(days=i),
                    direction=SignalDirection.LONG,
                    entry_price=100.0,
                    exit_price=101.0,
                    return_pct=0.01,
                    holding_days=1,
                )
                for i in range(20)
            ]
            return SimpleNamespace(
                evaluation=EvaluationReport(
                    strategy_id=strategy.meta.id,
                    overall=ov,
                    anti_overfit=AntiFitMetrics(),
                    confidence_score=ov.win_rate,
                    family_trial_count=f_trials,
                    deflated_sharpe_prob=dsr_p,
                    null_control_passed=True,
                ),
                backtest_result=BacktestResult(
                    strategy_id=strategy.meta.id,
                    batch_id="b_dsr",
                    signals=sigs,
                ),
                null_control=None,
                portfolio_verdict=None,
                trial_record=SimpleNamespace(trial_id=f"tr_{f_trials}"),
                strategy_rerun_null=None,
            )

        with (
            patch.object(dsr_engine._council, "generate_candidates", side_effect=_fake_gen_dsr),
            patch.object(
                StockRSIEngine, "_eval_with_kernel", autospec=True, side_effect=_fake_eval_dsr
            ),
            patch.object(
                dsr_engine._trial_ledger,
                "count_family_trials",
                side_effect=lambda _fid: trials_seen["n"],
            ),
            patch.object(
                dsr_engine._trial_ledger,
                "compute_dsr_for_trials",
                side_effect=lambda **kw: (
                    compute_deflated_sharpe(
                        1.0,
                        max(1, int(kw["n_trials"])),
                        20,
                        min_dsr_prob=0.70,
                    )
                    if int(kw["n_trials"]) < 3
                    else SimpleNamespace(dsr_probability=0.45)
                ),
            ),
        ):
            dsr_res = dsr_engine.run(
                base_dsr, dsr_market, charter=charter_dsr, persist_champion=False
            )
        assert dsr_res.promotions == 0
        c1_delibs = [d for d in dsr_res.deliberations if d.dsr_probability == pytest.approx(0.45)]
        assert c1_delibs and "dsr_gate" in c1_delibs[0].veto_reasons

    # 5. FactorStore save failure aborts BEFORE marking trial accepted
    fail_save_db = tmp_path / "fail_save.db"
    fail_factor = "l2_save_fail_test_factor"
    try:
        with StockRSIEngine(db_path=fail_save_db, enable_crossover=False) as fs_engine:
            rec = FactorRecord(
                name=fail_factor,
                description="fails persistence",
                rationale="atomic order test",
                code="def compute(df, idx):\n    return 1.0\n",
                expected_direction="positive",
            )
            IndicatorRegistry.register_dynamic(fail_factor, lambda df, idx, ctx=None: 1.0)
            l2_ch = [
                StrategyChange(
                    change_type=ChangeType.ADD_CONDITION,
                    target="entry.filters",
                    from_value=None,
                    to_value={"indicator": fail_factor, "op": ">=", "value": 0.5},
                    reason="L2 keep candidate",
                )
            ]
            with (
                patch.object(
                    fs_engine._miner,
                    "steer",
                    return_value=SimpleNamespace(
                        chosen_level=RSILevel.L2_FACTOR,
                        cluster=SimpleNamespace(bottleneck_hint=""),
                        synthesized_factor_name=fail_factor,
                        synthesized_via="offline_template",
                        pit_verified=True,
                        persisted_to_store=False,
                        synthesized_factor_record=rec,
                        proposed_changes=l2_ch,
                    ),
                ),
                patch.object(
                    fs_engine._council,
                    "generate_candidates",
                    return_value=[
                        SimpleNamespace(
                            proposal=CouncilProposal(
                                proposer_model="m1",
                                proposer_family="f1",
                                rationale="Use L2",
                                changes=l2_ch,
                            ),
                            audit=ProsecutorAudit(
                                prosecutor_model="p1",
                                prosecutor_family="f2",
                                approved=True,
                            ),
                        )
                    ],
                ),
                patch.object(
                    fs_engine._factor_store,
                    "save_if_absent",
                    side_effect=RuntimeError("simulated sqlite failure"),
                ),
                patch.object(fs_engine._trial_ledger, "mark_trial_accepted") as spy_accept,
            ):
                charter_fail = RSICharter(
                    name="fail_save",
                    seed_strategy_id=base_dsr.meta.id,
                    allowed_levels=[RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR],
                    max_rounds=1,
                    max_candidates_per_round=1,
                    min_signals=1,
                    max_drawdown=0.50,
                    min_dsr_prob=0.0,
                    require_null_control=False,
                )

                def _fake_eval_save_fail(self_arg, *, strategy, **kwargs):
                    del self_arg, kwargs
                    is_seed = strategy.meta.id == base_dsr.meta.id
                    wr = 0.45 if is_seed else 0.70
                    ret = 0.002 if is_seed else 0.04
                    return SimpleNamespace(
                        evaluation=EvaluationReport(
                            strategy_id=strategy.meta.id,
                            overall=OverallMetrics(
                                win_rate=wr, avg_return=ret, max_drawdown=0.05, signal_count=20
                            ),
                            anti_overfit=AntiFitMetrics(),
                            confidence_score=wr,
                            deflated_sharpe_prob=0.90,
                            null_control_passed=True,
                        ),
                        null_control=None,
                        portfolio_verdict=None,
                        trial_record=SimpleNamespace(trial_id="t_fail_save"),
                        strategy_rerun_null=None,
                    )

                with patch.object(
                    StockRSIEngine,
                    "_eval_with_kernel",
                    autospec=True,
                    side_effect=_fake_eval_save_fail,
                ):
                    fs_res = fs_engine.run(
                        base_dsr, dsr_market, charter=charter_fail, persist_champion=False
                    )
                assert fs_res.promotions == 0
                assert spy_accept.call_count == 0
                assert any("factor_store_save_failed" in r.veto_reasons for r in fs_res.rows)
    finally:
        IndicatorRegistry.unregister_dynamic(fail_factor)


def test_round7_to_11_codex_second_pass_hardening_invariants(tmp_path: Path) -> None:
    import os

    from alphaevo.alpha_factory.factory import _df_cache_key
    from alphaevo.backtest.engine import BacktestResult
    from alphaevo.core.config import BacktestConfig
    from alphaevo.models.enums import SignalDirection
    from alphaevo.models.execution import TradeSignal
    from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator

    # 1. FactorSandbox.probe_pit runs in hardened spawn worker & FactorGovernanceEngine delegates to it
    df_pit = _make_ohlcv(25, seed=31)
    with FactorSandbox(timeout_seconds=10) as sb:
        causal_code = (
            "def compute(df, idx):\n"
            "    if idx < 2:\n"
            "        return 0.0\n"
            "    return float(df['close'].iloc[idx] - df['close'].iloc[idx - 1])\n"
        )
        assert sb.probe_pit(causal_code, df_pit).passed is True
        leak_code = (
            "def compute(df, idx):\n"
            "    return float(df['close'].iloc[-1] - df['close'].iloc[idx])\n"
        )
        assert sb.probe_pit(leak_code, df_pit).passed is False

        # Bypass preflight only to prove worker audit hook & _SafeModuleProxy enforce runtime limits in probe_pit
        blocked_csv = tmp_path / "pit_blocked.csv"
        with patch.object(sb, "preflight_code", return_value=(True, "")):
            io_pit = sb.probe_pit(
                f"def compute(df, idx):\n    df.to_csv({str(blocked_csv)!r})\n    return 1.0\n",
                df_pit,
            )
            np_pit = sb.probe_pit(
                "import numpy as np\ndef compute(df, idx):\n    return float(len(np.loadtxt('x')))\n",
                df_pit,
            )
        assert io_pit.passed is False
        assert "PermissionError" in (io_pit.error or "")
        assert not blocked_csv.exists()
        assert np_pit.passed is False
        assert "AttributeError" in (np_pit.error or "")

    gov_db = tmp_path / "gov_sandbox_pit.db"
    with FactorStore(gov_db) as gfs:
        gfs.save(
            FactorRecord(
                name="gov_sandbox_delegate_factor",
                description="Causal momentum check",
                rationale="Delegates PIT probe to self._sandbox.probe_pit",
                code=(
                    "def compute(df, idx):\n"
                    "    if idx < 3:\n"
                    "        return 0.0\n"
                    "    return float(df['close'].iloc[idx] / df['close'].iloc[idx - 2] - 1.0)\n"
                ),
                expected_direction="positive",
            )
        )
    gov_market = {
        "GA": _make_ohlcv(90, seed=1),
        "GB": _make_ohlcv(90, seed=2),
    }
    with (
        FactorGovernanceEngine(
            db_path=gov_db,
            config=FactorGovernanceConfig(
                train_ratio=0.55,
                forward_horizon=5,
                require_pit_invariance=True,
            ),
        ) as gov_eng,
        patch.object(
            gov_eng._sandbox, "probe_pit", wraps=gov_eng._sandbox.probe_pit
        ) as spy_sandbox_pit,
        patch(
            "alphaevo.truth.pit_probe.probe_point_in_time_invariance",
            side_effect=AssertionError("unhardened pit_probe worker must not be called"),
        ),
    ):
        gov_rep = gov_eng.audit_factors(
            gov_market, strategies=[_make_strategy("gov_ref_s1")], auto_retire=False
        )
    assert spy_sandbox_pit.call_count == 1
    assert gov_rep.entries and gov_rep.entries[0].health != "pit_unsafe"

    # 2. _df_cache_key distinguishes identical-value int64 vs uint64 frames
    df_i64 = pd.DataFrame({"close": pd.Series([5] * 6, dtype=np.int64)})
    df_u64 = pd.DataFrame({"close": pd.Series([5] * 6, dtype=np.uint64)})
    assert _df_cache_key(df_i64) != _df_cache_key(df_u64)
    with FactorSandbox(timeout_seconds=10) as sb_dtype:
        fn_dtype = _build_indicator_fn(
            "def compute(df, idx):\n    return float((df['close'] - 10).iloc[idx])\n",
            sb_dtype,
        )
        assert getattr(fn_dtype, "_alphaevo_factor_code_hash", None) is not None
        with patch.object(sb_dtype, "execute", wraps=sb_dtype.execute) as spy_dtype_exec:
            val_i = fn_dtype(df_i64, 2)
            val_u = fn_dtype(df_u64, 2)
        assert spy_dtype_exec.call_count == 2
        assert val_i == pytest.approx(-5.0)
        assert val_u != val_i

    # 3. Co-STEER skips offline template whose name collides with an existing different dynamic callable
    collide_name = "intraday_range_compression_10d"
    collision_db = tmp_path / "offline_collide.db"
    IndicatorRegistry.register_dynamic(collide_name, lambda df, idx, ctx=None: 999.0)
    try:
        with (
            FactorStore(collision_db) as fs_col,
            BlindSpotMiner(factor_store=fs_col) as miner_col,
        ):
            dec_col = miner_col.steer(
                _make_strategy(
                    "offline_col_v1",
                    conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=60.0)],
                ),
                EvaluationReport(
                    strategy_id="offline_col_v1",
                    overall=OverallMetrics(win_rate=0.40, avg_return=-0.01, signal_count=20),
                ),
                df_pit,
                allowed_levels=[RSILevel.L2_FACTOR],
                stagnant_l1_rounds=3,
            )
            assert dec_col.synthesized_factor_name != collide_name
            assert fs_col.get(collide_name) is None
        if dec_col.synthesized_factor_name:
            IndicatorRegistry.unregister_dynamic(dec_col.synthesized_factor_name)
    finally:
        IndicatorRegistry.unregister_dynamic(collide_name)

    # 4. Unconditional round-end DSR refresh updates the LAST candidate AND MAP-Elites stores refreshed fitness
    last_base = _make_strategy(
        "last_cand_dsr_v1",
        conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=55.0)],
    )
    dsr_mkt = {"D1": _make_ohlcv(90, seed=51), "D2": _make_ohlcv(90, seed=52)}
    with StockRSIEngine(db_path=":memory:", enable_crossover=False) as dsr_eng2:
        dsr_charter = RSICharter(
            name="last_dsr_and_map_elites",
            seed_strategy_id=last_base.meta.id,
            objective="deflated_sharpe",
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=1,
            max_candidates_per_round=2,
            min_signals=5,
            max_drawdown=0.50,
            min_dsr_prob=0.70,
            require_null_control=False,
        )
        ch_a = [
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=55.0,
                to_value=60.0,
                reason="Candidate A",
            )
        ]
        ch_b = [
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=55.0,
                to_value=65.0,
                reason="Last Candidate B",
            )
        ]

        def _fake_gen2(parent, ev, **kwargs):
            del parent, ev, kwargs
            return [
                SimpleNamespace(
                    proposal=CouncilProposal(
                        proposer_model="m1", proposer_family="f1", rationale="A", changes=ch_a
                    ),
                    audit=ProsecutorAudit(
                        prosecutor_model="p1", prosecutor_family="f2", approved=True
                    ),
                ),
                SimpleNamespace(
                    proposal=CouncilProposal(
                        proposer_model="m1", proposer_family="f1", rationale="B", changes=ch_b
                    ),
                    audit=ProsecutorAudit(
                        prosecutor_model="p1", prosecutor_family="f2", approved=True
                    ),
                ),
            ]

        def _fake_eval2(self_arg, *, strategy, **kwargs):
            del self_arg, kwargs
            if strategy.meta.id == last_base.meta.id:
                dsr_p, f_tr = 0.50, 1
            elif abs(float(strategy.entry.conditions[0].value) - 60.0) < 1e-6:
                dsr_p, f_tr = 0.65, 2
            else:
                dsr_p, f_tr = 0.90, 3  # Last candidate already has final count, pre-refresh high!
            ov = OverallMetrics(
                win_rate=0.55,
                avg_return=0.01,
                sharpe_ratio=1.2,
                max_drawdown=0.05,
                signal_count=20,
            )
            sigs = [
                TradeSignal(
                    symbol="D1",
                    signal_date=date(2025, 1, 2) + timedelta(days=i),
                    direction=SignalDirection.LONG,
                    entry_price=100.0,
                    exit_price=101.0,
                    return_pct=0.01,
                    holding_days=1,
                )
                for i in range(20)
            ]
            return SimpleNamespace(
                evaluation=EvaluationReport(
                    strategy_id=strategy.meta.id,
                    overall=ov,
                    anti_overfit=AntiFitMetrics(),
                    confidence_score=0.55,
                    family_trial_count=f_tr,
                    deflated_sharpe_prob=dsr_p,
                    null_control_passed=True,
                ),
                backtest_result=BacktestResult(
                    strategy_id=strategy.meta.id,
                    batch_id="b_last",
                    signals=sigs,
                ),
                null_control=None,
                portfolio_verdict=None,
                trial_record=SimpleNamespace(trial_id=f"tr2_{f_tr}"),
                strategy_rerun_null=None,
            )

        with (
            patch.object(dsr_eng2._council, "generate_candidates", side_effect=_fake_gen2),
            patch.object(
                StockRSIEngine, "_eval_with_kernel", autospec=True, side_effect=_fake_eval2
            ),
            patch.object(dsr_eng2._trial_ledger, "count_family_trials", return_value=3),
            patch.object(
                dsr_eng2._trial_ledger,
                "compute_dsr_for_trials",
                return_value=SimpleNamespace(dsr_probability=0.20),
            ) as spy_dsr,
        ):
            res2 = dsr_eng2.run(last_base, dsr_mkt, charter=dsr_charter, persist_champion=False)
        assert spy_dsr.call_count == 2
        assert res2.promotions == 0
        # Seed enters archive with fitness 0.504; refreshed candidates fall to 0.354 and must NOT beat seed
        assert res2.archive_summary is not None
        assert res2.archive_summary.best_strategy_id == last_base.meta.id
        assert res2.archive_summary.best_fitness == pytest.approx(0.504, abs=1e-4)

    # 5. MechanismEvaluator assigns synthetic dates & strict fwd_cutoffs to undated frames (zero warmup leak)
    undated = pd.DataFrame(
        {
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.0] * 20 + [70.0] * 5 + [100.0] * 35,
            "volume": [1_000_000.0] * 60,
        }
    )
    tr_u, fwd_u, cut_u = _temporal_split_with_warmup(
        {"SYNTH": undated}, train_ratio=0.65, warmup_bars=30
    )
    assert "SYNTH" in cut_u
    assert "date" in tr_u["SYNTH"].columns and "date" in fwd_u["SYNTH"].columns
    oversold_strat = _make_strategy(
        "undated_mech_v1",
        conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=40.0)],
    )
    mech = MechanismEvaluator()
    with patch.object(
        StockRSIEngine,
        "run",
        return_value=SimpleNamespace(
            champion_strategy=oversold_strat,
            champion_evaluation=EvaluationReport(
                strategy_id=oversold_strat.meta.id,
                overall=OverallMetrics(win_rate=0.6, avg_return=0.02, signal_count=5),
            ),
            promotions=0,
            total_trials_recorded=1,
        ),
    ):
        paired = mech._run_single_variant(
            seed_strategy=oversold_strat,
            train_data=tr_u,
            fwd_data=fwd_u,
            fwd_cutoffs=cut_u,
            variant=HarnessVariant(variant_id="v1", name="Base", require_null_control=False),
            max_rounds=1,
            seed=42,
        )
    assert paired.forward_signals == 0

    # 6. ArenaRunner with min_data_days=10 and short panel skips shadow replay instead of OOS baseline fallback
    short_arena_market = {
        "SHORT_A": _make_ohlcv(22, start=date(2025, 1, 2), seed=61),
        "SHORT_B": _make_ohlcv(22, start=date(2025, 1, 2), seed=62),
    }
    snap_short = ArenaRunner(backtest_config=BacktestConfig(min_data_days=10)).run(
        [_make_strategy("arena_short_v1")],
        short_arena_market,
        rounds=1,
        enable_l2_factors=False,
        enable_crossover=False,
        seed=42,
    )
    assert snap_short.entries[0].shadow_chain_verified is False
    assert snap_short.chain_verified is False

    # 7. Crossover _summarize_exit_changes preserves None resets on TP trailing & ATR multiplier
    p_trail = _make_strategy(
        "p_trail",
        sl=StopLossConfig(type="atr", multiplier=2.5, atr_period=14),
        tp=TakeProfitConfig(type="trailing", trigger_pct=0.08, trail_pct=0.03),
    )
    s_trail = _make_strategy(
        "s_trail",
        sl=StopLossConfig(type="pct", value=0.04),
        tp=TakeProfitConfig(type="trailing"),
    )
    reset_changes = SemanticCrossoverBreeder._summarize_exit_changes(p_trail, s_trail)
    by_target = {c.target: c for c in reset_changes}
    assert by_target["exit.stop_loss.multiplier"].to_value is None
    assert by_target["exit.take_profit.trigger_pct"].to_value is None
    assert by_target["exit.take_profit.trail_pct"].to_value is None

    # 8. Council _try_litellm_prosecutor fails closed when ALPHAEVO_API_KEY env var is set and ConfigManager.load fails
    env_council = HeterogeneousCouncil(
        config=HeterogeneousCouncilConfig(
            proposer=ModelSpec(model="openai/gpt-6-astra"),
            prosecutor=ModelSpec(model="anthropic/claude-opus-5.5", backend="litellm"),
        ),
        api_key=None,
    )
    old_key = os.environ.get("ALPHAEVO_API_KEY")
    os.environ["ALPHAEVO_API_KEY"] = "sk-env-fail-closed-test"
    try:
        with patch("alphaevo.core.config.ConfigManager.load", side_effect=RuntimeError("boom")):
            env_veto = env_council._try_litellm_prosecutor(
                p_trail,
                EvaluationReport(
                    strategy_id=p_trail.meta.id,
                    overall=OverallMetrics(win_rate=0.5, signal_count=10),
                ),
                [],
                env_council.config.prosecutor,
            )
    finally:
        if old_key is None:
            os.environ.pop("ALPHAEVO_API_KEY", None)
        else:
            os.environ["ALPHAEVO_API_KEY"] = old_key
    assert env_veto and "fail-closed" in env_veto[0]


def test_round7_to_11_codex_third_pass_edge_cases(tmp_path: Path) -> None:
    from alphaevo.alpha_factory.factory import _df_cache_key
    from alphaevo.reflection.critic import SelfCritic
    from alphaevo.reflection.mutator import StrategyMutator
    from alphaevo.rsi.co_steer import _OFFLINE_FACTOR_TEMPLATES
    from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator

    # 1. Mechanism per-cutoff execution: warmup entries before dated cutoff never block post-cutoff entries
    # Dated symbol dips during warmup (2025-03-26..28, while its cutoff is 2025-04-01) so position is closed or skipped,
    # and dips AGAIN on 2025-04-01..02; mixed with an earlier undated symbol whose synthetic cutoff is 2020.
    n_dated = 45
    dated_start = date(2025, 2, 18)
    dated_dates = [dated_start + timedelta(days=i) for i in range(n_dated)]
    # Make dip on indices 36..41 (Mar 26 .. Mar 31) and again on indices 42..44 (Apr 1 .. Apr 3)
    dated_closes = [100.0] * n_dated
    for i in range(36, 42):
        dated_closes[i] = 75.0
    dated_df = pd.DataFrame(
        {
            "date": dated_dates,
            "open": dated_closes,
            "high": [c * 1.01 for c in dated_closes],
            "low": [c * 0.99 for c in dated_closes],
            "close": dated_closes,
            "volume": [1_500_000.0] * n_dated,
        }
    )
    undated_df = pd.DataFrame(
        {
            "open": [100.0] * 45,
            "high": [101.0] * 45,
            "low": [99.0] * 45,
            "close": [100.0] * 45,
            "volume": [1_500_000.0] * 45,
        }
    )
    tr_mix, fwd_mix, cut_mix = _temporal_split_with_warmup(
        {"DATED": dated_df, "UNDATED": undated_df}, train_ratio=0.70, warmup_bars=25
    )
    assert "DATED" in cut_mix and "UNDATED" in cut_mix
    # Override DATED cutoff to index 41 (2025-03-31) after warmup dip (36..40) that would otherwise stay open
    # under min(cutoffs) (2020) and block entry on 2025-04-01 with max_holding_days=10!
    cut_override = {"DATED": dated_dates[41], "UNDATED": cut_mix["UNDATED"]}
    trig_strat = _make_strategy(
        "mech_multi_cut_v1",
        conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=45.0)],
        tp=TakeProfitConfig(type="pct", value=0.50),
        sl=StopLossConfig(type="pct", value=0.50),
    )
    trig_strat.exit.max_holding_days = 15
    mech_ev = MechanismEvaluator()
    with patch.object(
        StockRSIEngine,
        "run",
        return_value=SimpleNamespace(
            champion_strategy=trig_strat,
            champion_evaluation=EvaluationReport(
                strategy_id=trig_strat.meta.id,
                overall=OverallMetrics(win_rate=0.5, avg_return=0.01, signal_count=5),
            ),
            promotions=0,
            total_trials_recorded=1,
        ),
    ):
        paired_mix = mech_ev._run_single_variant(
            seed_strategy=trig_strat,
            train_data=tr_mix,
            fwd_data={"DATED": dated_df, "UNDATED": fwd_mix["UNDATED"]},
            fwd_cutoffs=cut_override,
            variant=HarnessVariant(variant_id="mix", name="Mix", require_null_control=False),
            max_rounds=1,
            seed=42,
        )
    assert paired_mix.forward_signals >= 1

    # 2. Co-STEER blocks reactivation of retired offline templates, unloaded LLM name reuse, and active metric wipes
    tpl = next(t for t in _OFFLINE_FACTOR_TEMPLATES if t.name == "trend_alignment_slope_20d")
    retired_db = tmp_path / "retired_co_steer.db"
    sample_df = _make_ohlcv(25, seed=77)
    try:
        with FactorStore(retired_db) as rfs:
            rfs.save(
                FactorRecord(
                    name=tpl.name,
                    description="Previously retired offline template",
                    rationale="Must stay retired",
                    code=tpl.code,
                    expected_direction=tpl.expected_direction,
                    status="retired",
                )
            )
            rfs.save(
                FactorRecord(
                    name="custom_retired_llm_factor",
                    description="Retired unloaded LLM factor",
                    rationale="Must not be overwritten",
                    code="def compute(df, idx):\n    return 0.0\n",
                    expected_direction="positive",
                    status="retired",
                )
            )
            ret_synth = SimpleNamespace(
                generate=lambda **kwargs: [
                    SimpleNamespace(
                        name="custom_retired_llm_factor",
                        code="def compute(df, idx):\n    return 1.0\n",
                        description="Overwrite retired factor",
                        expected_direction="positive",
                    )
                ]
            )
            with BlindSpotMiner(factor_store=rfs, synthesizer=ret_synth) as r_miner:
                dec_ret = r_miner.steer(
                    _make_strategy(
                        "ret_strat_v1",
                        conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=60.0)],
                    ),
                    EvaluationReport(
                        strategy_id="ret_strat_v1",
                        overall=OverallMetrics(win_rate=0.40, avg_return=-0.01, signal_count=20),
                    ),
                    sample_df,
                    allowed_levels=[RSILevel.L2_FACTOR],
                    stagnant_l1_rounds=3,
                    persist=True,
                )
            assert dec_ret.synthesized_factor_name != tpl.name
            assert dec_ret.synthesized_factor_name != "custom_retired_llm_factor"
            assert rfs.get(tpl.name).status == "retired"  # type: ignore[union-attr]
            assert (
                rfs.get("custom_retired_llm_factor").code  # type: ignore[union-attr]
                == "def compute(df, idx):\n    return 0.0\n"
            )
        if dec_ret.synthesized_factor_name:
            IndicatorRegistry.unregister_dynamic(dec_ret.synthesized_factor_name)
    finally:
        IndicatorRegistry.unregister_dynamic(tpl.name)

    # Active offline factor keeps existing governance metrics/usage_count on reuse
    active_db = tmp_path / "active_preserve.db"
    try:
        with FactorStore(active_db) as afs:
            afs.save(
                FactorRecord(
                    name=tpl.name,
                    description=tpl.description,
                    rationale="Tracked active factor",
                    code=tpl.code,
                    expected_direction=tpl.expected_direction,
                    ic_mean=0.14,
                    ic_std=0.05,
                    ir=2.8,
                    usage_count=9,
                    status="active",
                )
            )
            with BlindSpotMiner(factor_store=afs) as a_miner:
                dec_act = a_miner.steer(
                    _make_strategy(
                        "act_strat_v1",
                        conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=60.0)],
                    ),
                    EvaluationReport(
                        strategy_id="act_strat_v1",
                        overall=OverallMetrics(win_rate=0.40, avg_return=-0.01, signal_count=20),
                    ),
                    sample_df,
                    allowed_levels=[RSILevel.L2_FACTOR],
                    stagnant_l1_rounds=3,
                    persist=True,
                )
            assert dec_act.synthesized_factor_name == tpl.name
            stored_after = afs.get(tpl.name)
            assert stored_after is not None
            assert stored_after.ic_mean == pytest.approx(0.14)
            assert stored_after.usage_count == 9
    finally:
        IndicatorRegistry.unregister_dynamic(tpl.name)

    # 3. _df_cache_key distinguishes identical-value CategoricalDtype frames with reversed categories
    df_cat1 = pd.DataFrame(
        {"tag": pd.Categorical(["a", "b", "a", "b"], categories=["a", "b"], ordered=True)}
    )
    df_cat2 = pd.DataFrame(
        {"tag": pd.Categorical(["a", "b", "a", "b"], categories=["b", "a"], ordered=True)}
    )
    assert _df_cache_key(df_cat1) != _df_cache_key(df_cat2)
    with FactorSandbox(timeout_seconds=10) as sb_cat:
        fn_cat = _build_indicator_fn(
            "def compute(df, idx):\n    return float(df['tag'].cat.codes.iloc[idx])\n",
            sb_cat,
        )
        assert fn_cat(df_cat1, 0) == pytest.approx(0.0)
        assert fn_cat(df_cat2, 0) == pytest.approx(1.0)

    # 4. Exit-graft preserves overlapping fields on type transitions so SelfCritic/Mutator replay identically
    p_pl = _make_strategy("p_pl_eq", sl=StopLossConfig(type="pct", value=0.04))
    s_pl = _make_strategy("s_pl_eq", sl=StopLossConfig(type="price_level", value=0.04))
    summary = SemanticCrossoverBreeder._summarize_exit_changes(p_pl, s_pl)
    targets = [c.target for c in summary]
    assert targets == ["exit.stop_loss.type", "exit.stop_loss.value"]
    replayed = StrategyMutator(max_changes=10).mutate(p_pl, summary, atomic=True)
    assert replayed.exit.stop_loss.type == "price_level"
    assert replayed.exit.stop_loss.value == pytest.approx(0.04)
    verdict = SelfCritic().critique(
        p_pl,
        EvaluationReport(
            strategy_id=p_pl.meta.id,
            overall=OverallMetrics(win_rate=0.52, avg_return=0.015, signal_count=25),
        ),
        SimpleNamespace(proposed_changes=summary),  # type: ignore[arg-type]
    )
    assert len(verdict.rejected) == 0


def test_round7_to_11_codex_fourth_pass_invariants(tmp_path: Path) -> None:
    from alphaevo.alpha_factory.factory import _df_cache_key
    from alphaevo.reflection.critic import SelfCritic
    from alphaevo.reflection.mutator import StrategyMutator
    from alphaevo.rsi.co_steer import _OFFLINE_FACTOR_TEMPLATES
    from alphaevo.rsi.engine import _acquire_dynamic_lease, _release_dynamic_lease_entry
    from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator
    from alphaevo.truth.kernel import EvaluationKernel
    from alphaevo.truth.null_control import NullSignalController

    sample_df = _make_ohlcv(25, seed=91)
    tpl = next(t for t in _OFFLINE_FACTOR_TEMPLATES if t.name == "trend_alignment_slope_20d")

    # 1. Atomic FactorStore.save_if_absent blocks competing overwrites/reactivations
    atomic_db = tmp_path / "atomic_fs.db"
    with FactorStore(atomic_db) as afs:
        retired_rec = FactorRecord(
            name="atomic_race_factor",
            description="Originally retired",
            rationale="Do not reactivate",
            code="def compute(df, idx):\n    return 0.0\n",
            expected_direction="positive",
            status="retired",
        )
        stored1, ins1 = afs.save_if_absent(retired_rec)
        assert ins1 is True
        assert stored1.status == "retired"

        competing_rec = FactorRecord(
            name="atomic_race_factor",
            description="Competing overwrite",
            rationale="Should lose race cleanly",
            code="def compute(df, idx):\n    return 1.0\n",
            expected_direction="positive",
            status="active",
        )
        stored2, ins2 = afs.save_if_absent(competing_rec)
        assert ins2 is False
        assert stored2.status == "retired"
        assert stored2.code == "def compute(df, idx):\n    return 0.0\n"

        # Simulate BlindSpotMiner.steer racing with another writer on save_if_absent -> rollback to L1
        with (
            patch.object(afs, "get", return_value=None),
            patch.object(afs, "save_if_absent", return_value=(retired_rec, False)),
            BlindSpotMiner(factor_store=afs) as race_miner,
        ):
            try:
                dec_race = race_miner.steer(
                    _make_strategy(
                        "race_l1_fallback_v1",
                        conditions=[StrategyCondition(indicator="rsi_14", op="<=", value=60.0)],
                    ),
                    EvaluationReport(
                        strategy_id="race_l1_fallback_v1",
                        overall=OverallMetrics(win_rate=0.40, avg_return=-0.01, signal_count=20),
                    ),
                    sample_df,
                    allowed_levels=[RSILevel.L2_FACTOR],
                    stagnant_l1_rounds=3,
                    persist=True,
                )
            finally:
                IndicatorRegistry.unregister_dynamic(tpl.name)
        assert dec_race.chosen_level == RSILevel.L1_STRATEGY
        assert dec_race.proposed_changes == []
        assert not IndicatorRegistry.is_registered(tpl.name)

    # 2. Lease refcounting protects shared transient dynamic factors across concurrent RSI runs
    lease_db = tmp_path / "lease_fs.db"
    shared_factor = "concurrent_lease_test_factor"
    dummy_strat = _make_strategy("lease_dummy_v1")
    with FactorStore(lease_db) as lfs:
        IndicatorRegistry.register_dynamic(shared_factor, lambda df, idx, ctx=None: 1.0)
        try:
            rec_lease = FactorRecord(
                name=shared_factor,
                description="Concurrent transient",
                rationale="Lease test",
                code="def compute(df, idx):\n    return 1.0\n",
                expected_direction="positive",
            )
            l1 = _acquire_dynamic_lease(
                shared_factor,
                in_pre_registered=False,
                factor_record=rec_lease,
                factor_store=lfs,
            )
            l2 = _acquire_dynamic_lease(
                shared_factor,
                in_pre_registered=True,
                factor_record=rec_lease,
                factor_store=lfs,
            )
            assert l1 == shared_factor and l2 == shared_factor
            _release_dynamic_lease_entry(l1, persisted=False, protected_strategies=(dummy_strat,))
            # Run 2 is still active -> factor MUST remain registered!
            assert IndicatorRegistry.is_registered(shared_factor) is True
            _release_dynamic_lease_entry(l2, persisted=False, protected_strategies=(dummy_strat,))
            # Last lease released -> transient factor cleanly unregistered!
            assert IndicatorRegistry.is_registered(shared_factor) is False
        finally:
            IndicatorRegistry.unregister_dynamic(shared_factor)

    # 3. Mechanism forward null test receives strictly OOS-only bars (no pre-cutoff warmup rows)
    mech_df = _make_ohlcv(70, start=date(2025, 1, 2), seed=88)
    tr_n, fwd_n, cut_n = _temporal_split_with_warmup(
        {"NULL_CHK": mech_df}, train_ratio=0.65, warmup_bars=25
    )
    strat_n = _make_strategy("mech_null_oos_v1")
    captured_null_dfs: list[dict[str, pd.DataFrame]] = []
    orig_null = NullSignalController.evaluate_candidate_against_nulls

    def _spy_null(self_arg, signals, market_data, **kwargs):
        captured_null_dfs.append({k: v.copy() for k, v in market_data.items()})
        return orig_null(self_arg, signals, market_data, **kwargs)

    with (
        patch.object(
            StockRSIEngine,
            "run",
            return_value=SimpleNamespace(
                champion_strategy=strat_n,
                champion_evaluation=EvaluationReport(
                    strategy_id=strat_n.meta.id,
                    overall=OverallMetrics(win_rate=0.6, avg_return=0.02, signal_count=10),
                ),
                promotions=1,
                total_trials_recorded=1,
            ),
        ),
        patch.object(EvaluationKernel, "evaluate") as mock_ke,
        patch.object(
            NullSignalController,
            "evaluate_candidate_against_nulls",
            autospec=True,
            side_effect=_spy_null,
        ),
    ):
        from alphaevo.models.enums import SignalDirection
        from alphaevo.models.execution import BacktestResult, TradeSignal

        fut_sigs = [
            TradeSignal(
                symbol="NULL_CHK",
                signal_date=cut_n["NULL_CHK"] + timedelta(days=i),
                direction=SignalDirection.LONG,
                entry_price=100.0,
                exit_price=102.0,
                return_pct=0.02,
                holding_days=1,
            )
            for i in range(6)
        ]
        mock_ke.return_value = SimpleNamespace(
            backtest_result=BacktestResult(
                strategy_id=strat_n.meta.id,
                batch_id="b_null",
                signals=fut_sigs,
                total_signals=6,
                executed_signals=6,
                skipped_signals=0,
                date_range=(cut_n["NULL_CHK"], cut_n["NULL_CHK"] + timedelta(days=5)),
            ),
            evaluation=EvaluationReport(
                strategy_id=strat_n.meta.id,
                overall=OverallMetrics(win_rate=1.0, avg_return=0.02, signal_count=6),
            ),
            portfolio_verdict=None,
        )
        MechanismEvaluator()._run_single_variant(
            seed_strategy=strat_n,
            train_data=tr_n,
            fwd_data=fwd_n,
            fwd_cutoffs=cut_n,
            variant=HarnessVariant(variant_id="n", name="Null", require_null_control=False),
            max_rounds=1,
            seed=42,
        )
    assert len(captured_null_dfs) == 1
    null_dates = pd.to_datetime(captured_null_dfs[0]["NULL_CHK"]["date"]).dt.date
    assert (null_dates >= cut_n["NULL_CHK"]).all()

    # 4. Governance never retires active factors when disjoint calendars yield empty train/OOS split
    disjoint_data = {
        str(y): pd.DataFrame(
            {
                "date": pd.date_range(f"{y}-01-01", periods=40),
                "open": range(1, 41),
                "high": [x + 1 for x in range(1, 41)],
                "low": list(range(1, 41)),
                "close": range(1, 41),
                "volume": [100_000.0] * 40,
            }
        )
        for y in (2023, 2025)
    }
    disjoint_db = tmp_path / "disjoint_gov.db"
    with FactorStore(disjoint_db) as dfs:
        dfs.save(
            FactorRecord(
                name="disjoint_kept_factor",
                description="Must not auto-retire on empty panel split",
                rationale="Disjoint calendar safety",
                code="def compute(df, idx):\n    return 1.0\n",
                expected_direction="positive",
                status="active",
            )
        )
    with FactorGovernanceEngine(
        db_path=disjoint_db,
        config=FactorGovernanceConfig(train_ratio=0.5, require_pit_invariance=False),
    ) as d_gov:
        d_rep = d_gov.audit_factors(disjoint_data, strategies=[], auto_retire=True)
    assert d_rep.retired_count == 0
    assert d_rep.entries[0].action_taken == "kept"
    with FactorStore(disjoint_db) as chk_fs:
        assert chk_fs.get("disjoint_kept_factor").status == "active"  # type: ignore[union-attr]

    # 5. CategoricalDtype int64 vs uint64 categories distinguish column & CategoricalIndex cache keys
    df_ci64 = pd.DataFrame(
        {"c": pd.Categorical([1, 2, 1, 2], categories=pd.Index([1, 2], dtype=np.int64))},
        index=pd.CategoricalIndex([1, 2, 1, 2], categories=pd.Index([1, 2], dtype=np.int64)),
    )
    df_cu64 = pd.DataFrame(
        {"c": pd.Categorical([1, 2, 1, 2], categories=pd.Index([1, 2], dtype=np.uint64))},
        index=pd.CategoricalIndex([1, 2, 1, 2], categories=pd.Index([1, 2], dtype=np.uint64)),
    )
    assert _df_cache_key(df_ci64) != _df_cache_key(df_cu64)
    with FactorSandbox(timeout_seconds=10) as sb_cdtype:
        fn_cdtype = _build_indicator_fn(
            "def compute(df, idx):\n    return 1.0 if df['c'].cat.categories.dtype.kind == 'u' else 0.0\n",
            sb_cdtype,
        )
        assert fn_cdtype(df_ci64, 0) == pytest.approx(0.0)
        assert fn_cdtype(df_cu64, 0) == pytest.approx(1.0)

    # 6. Composite->pct exit_graft preserves None conditions & prunes dangling exit tunables identically on replay
    p_comp = _make_strategy(
        "p_comp_r",
        sl=StopLossConfig(
            type="composite",
            conditions=[StrategyCondition(indicator="close_below_ma10", op="==", value=True)],
        ),
        tunable=[
            TunableParam(target="exit.stop_loss.value", range=(0.02, 0.08), step=0.01),
            TunableParam(target="exit.max_holding_days", range=(5, 20), step=5),
        ],
    )
    s_pct = _make_strategy("s_pct_r", sl=StopLossConfig(type="pct", value=0.04))
    comp_cands = SemanticCrossoverBreeder().breed(p_comp, s_pct, max_children=3)
    comp_graft = next(c for c in comp_cands if c.operator == "exit_graft")
    replayed_comp = StrategyMutator(max_changes=20).mutate(
        p_comp, comp_graft.summary_changes, atomic=True
    )
    assert replayed_comp.exit == comp_graft.child_strategy.exit
    assert replayed_comp.params.tunable == comp_graft.child_strategy.params.tunable

    # Also ATR exit_graft drops exit.stop_loss.value tunable identically on replay
    p_atr = _make_strategy(
        "p_atr_r",
        sl=StopLossConfig(type="pct", value=0.04),
        tunable=[TunableParam(target="exit.stop_loss.value", range=(0.02, 0.08), step=0.01)],
    )
    s_atr = _make_strategy(
        "s_atr_r",
        conditions=[StrategyCondition(indicator="momentum_10d", op=">", value=0.02)],
        sl=StopLossConfig(type="atr", multiplier=2.0, atr_period=14),
    )
    atr_graft = next(
        c
        for c in SemanticCrossoverBreeder().breed(p_atr, s_atr, max_children=3)
        if c.operator == "exit_graft"
    )
    replayed_atr = StrategyMutator(max_changes=20).mutate(
        p_atr, atr_graft.summary_changes, atomic=True
    )
    assert replayed_atr.exit == atr_graft.child_strategy.exit
    assert replayed_atr.params.tunable == atr_graft.child_strategy.params.tunable

    # 7. Legacy multi-AND trigger_blend replay reproduces triggers + guards + entry.logic + tunables identically
    p_leg = _make_strategy(
        "p_leg_r",
        logic="and",
        conditions=[
            StrategyCondition(indicator="rsi_14", op="<=", value=40.0),
            StrategyCondition(indicator="close_above_ma20", op="==", value=True),
        ],
        tunable=[
            TunableParam(target="entry.conditions[indicator=rsi_14].value", range=(25, 50), step=5),
            TunableParam(
                target="entry.conditions[indicator=close_above_ma20].value",
                range=(0, 1),
                step=1,
            ),
        ],
    )
    s_leg = _make_strategy(
        "s_leg_r",
        conditions=[StrategyCondition(indicator="momentum_10d", op=">", value=0.02)],
    )
    leg_blend = next(
        c
        for c in SemanticCrossoverBreeder().breed(p_leg, s_leg, max_children=3)
        if c.operator == "trigger_blend"
    )
    replayed_blend = StrategyMutator(max_changes=10).mutate(
        p_leg, leg_blend.summary_changes, atomic=True
    )
    assert replayed_blend.entry == leg_blend.child_strategy.entry
    assert replayed_blend.params.tunable == leg_blend.child_strategy.params.tunable
    verdict_blend = SelfCritic().critique(
        p_leg,
        EvaluationReport(
            strategy_id=p_leg.meta.id,
            overall=OverallMetrics(win_rate=0.30, avg_return=0.01, signal_count=25),
        ),
        SimpleNamespace(proposed_changes=leg_blend.summary_changes),  # type: ignore[arg-type]
    )
    assert len(verdict_blend.rejected) == 0


def test_round7_to_11_codex_fifth_pass_invariants(tmp_path: Path) -> None:
    from alphaevo.alpha_factory.factory import _df_cache_key
    from alphaevo.models.enums import SignalDirection
    from alphaevo.models.execution import BacktestResult, TradeSignal
    from alphaevo.reflection.mutator import StrategyMutator
    from alphaevo.rsi.mechanism import HarnessVariant, MechanismEvaluator
    from alphaevo.strategy.tunable import resolve_tunable_target
    from alphaevo.truth.kernel import EvaluationKernel

    sample_df = _make_ohlcv(25, seed=97)

    # 1. Atomic co_steer registration + exception-safe lease cleanup in StockRSIEngine.run
    from alphaevo.rsi.co_steer import _DYNAMIC_LEASES, _TRANSIENT_FACTORS

    synth_code = "def compute(df, idx):\n    return float(df['close'].iloc[idx] * 0.01)\n"
    atomic_factor = "trend_alignment_slope_20d"

    lease_db = tmp_path / "atomic_lease.db"
    strat_l = _make_strategy("atomic_l_v1")
    weak_ev = EvaluationReport(
        strategy_id=strat_l.meta.id,
        overall=OverallMetrics(win_rate=0.40, avg_return=-0.01, signal_count=20),
    )
    try:
        with FactorStore(lease_db) as lfs:
            m1 = BlindSpotMiner(factor_store=lfs)
            m2 = BlindSpotMiner(factor_store=lfs)
            with m1, m2:
                # Run 1 publishes and atomically acquires lease #1
                d1 = m1.steer(
                    strat_l,
                    weak_ev,
                    sample_df,
                    allowed_levels=[RSILevel.L2_FACTOR],
                    stagnant_l1_rounds=3,
                    persist=False,
                    acquire_lease=True,
                )
                assert d1.synthesized_factor_name == atomic_factor
                assert d1.lease_acquired is True
                assert atomic_factor in _TRANSIENT_FACTORS
                assert _DYNAMIC_LEASES.get(atomic_factor) == 1

                # Run 2 enters AFTER IndicatorRegistry sees atomic_factor -> MUST still acquire lease #2
                assert IndicatorRegistry.is_registered(atomic_factor)
                d2 = m2.steer(
                    strat_l,
                    weak_ev,
                    sample_df,
                    allowed_levels=[RSILevel.L2_FACTOR],
                    stagnant_l1_rounds=3,
                    persist=False,
                    acquire_lease=True,
                )
                assert d2.synthesized_factor_name == atomic_factor
                assert d2.lease_acquired is True
                assert _DYNAMIC_LEASES.get(atomic_factor) == 2

                # Releasing Run 1 keeps atomic_factor alive for Run 2
                from alphaevo.rsi.co_steer import release_dynamic_lease

                release_dynamic_lease(
                    atomic_factor, persisted=False, protected_strategies=(strat_l,)
                )
                assert IndicatorRegistry.is_registered(atomic_factor) is True
                release_dynamic_lease(
                    atomic_factor, persisted=False, protected_strategies=(strat_l,)
                )
                assert IndicatorRegistry.is_registered(atomic_factor) is False

        # Exception inside StockRSIEngine.run after steer releases transient lease cleanly
        exc_factor = "exception_cleanup_lease_factor"
        with StockRSIEngine(db_path=":memory:", enable_crossover=False) as exc_eng:

            def _boom_steer(*args, **kwargs):
                del args, kwargs
                IndicatorRegistry.register_dynamic(exc_factor, lambda df, idx, ctx=None: 1.0)
                return SimpleNamespace(
                    chosen_level=RSILevel.L2_FACTOR,
                    cluster=SimpleNamespace(bottleneck_hint=""),
                    synthesized_factor_name=exc_factor,
                    synthesized_via="offline_template",
                    pit_verified=True,
                    persisted_to_store=False,
                    lease_acquired=False,
                    synthesized_factor_record=FactorRecord(
                        name=exc_factor,
                        description="exc",
                        rationale="exc",
                        code=synth_code,
                        expected_direction="positive",
                    ),
                    proposed_changes=[],
                )

            with (
                patch.object(exc_eng._miner, "steer", side_effect=_boom_steer),
                patch.object(
                    exc_eng._council, "generate_candidates", side_effect=RuntimeError("boom")
                ),
                pytest.raises(RuntimeError, match="boom"),
            ):
                exc_eng.run(
                    strat_l,
                    {"M1": _make_ohlcv(50, seed=5)},
                    charter=RSICharter(
                        name="exc_lease",
                        seed_strategy_id=strat_l.meta.id,
                        max_rounds=1,
                        null_permutations=10,
                    ),
                    persist_champion=False,
                )
        assert not IndicatorRegistry.is_registered(exc_factor)
        assert exc_factor not in _DYNAMIC_LEASES
    finally:
        IndicatorRegistry.unregister_dynamic(atomic_factor)

    # 2. Governance never retires active factor when OOS length <= forward_horizon (0 valid labels)
    staggered_short = {
        "SHORT_A": _make_ohlcv(30, start=date(2025, 1, 1), seed=1),
        "SHORT_B": _make_ohlcv(30, start=date(2025, 1, 5), seed=2),
    }
    hz_db = tmp_path / "gov_horizon_nan.db"
    with FactorStore(hz_db) as hfs:
        hfs.save(
            FactorRecord(
                name="zero_oos_label_kept_factor",
                description="Must survive when OOS window <= forward_horizon",
                rationale="No bogus 0-label retirement",
                code="def compute(df, idx):\n    return 1.0\n",
                expected_direction="positive",
                status="active",
            )
        )
    with FactorGovernanceEngine(
        db_path=hz_db,
        config=FactorGovernanceConfig(
            train_ratio=0.5, forward_horizon=20, require_pit_invariance=False
        ),
    ) as hgov:
        hrep = hgov.audit_factors(staggered_short, strategies=[], auto_retire=True)
    assert hrep.retired_count == 0
    assert hrep.entries[0].action_taken == "kept"

    # 3. _df_cache_key distinguishes object-dtype [1, 2] vs ["1", "2"]
    df_obj_int = pd.DataFrame({"v": pd.Series([1, 2, 1, 2], dtype=object)})
    df_obj_str = pd.DataFrame({"v": pd.Series(["1", "2", "1", "2"], dtype=object)})
    assert _df_cache_key(df_obj_int) != _df_cache_key(df_obj_str)
    with FactorSandbox(timeout_seconds=10) as sb_obj:
        fn_obj = _build_indicator_fn(
            "def compute(df, idx):\n    return 1.0 if isinstance(df['v'].iloc[idx], str) else 0.0\n",
            sb_obj,
        )
        assert fn_obj(df_obj_int, 0) == pytest.approx(0.0)
        assert fn_obj(df_obj_str, 0) == pytest.approx(1.0)

    # 4. Mechanism forward null test refuses centered fallback when OOS length < avg_hold + 3
    short_oos_df = _make_ohlcv(40, start=date(2025, 1, 1), seed=19)
    tr_s = {"S1": short_oos_df.iloc[:28].copy().reset_index(drop=True)}
    fwd_s = {"S1": short_oos_df.iloc[28:].copy().reset_index(drop=True)}  # 12 bars
    cut_s = {"S1": short_oos_df["date"].iloc[28]}
    ident_sigs = [
        TradeSignal(
            symbol="S1",
            signal_date=short_oos_df["date"].iloc[28 + i],
            direction=SignalDirection.LONG,
            entry_price=100.0,
            exit_price=104.0,
            return_pct=0.04,
            holding_days=10,
        )
        for i in range(5)
    ]
    strat_s = _make_strategy("mech_deg_null_v1")
    with (
        patch.object(
            StockRSIEngine,
            "run",
            return_value=SimpleNamespace(
                champion_strategy=strat_s,
                champion_evaluation=EvaluationReport(
                    strategy_id=strat_s.meta.id,
                    overall=OverallMetrics(win_rate=0.6, avg_return=0.02, signal_count=10),
                ),
                promotions=1,
                total_trials_recorded=1,
            ),
        ),
        patch.object(
            EvaluationKernel,
            "evaluate",
            return_value=SimpleNamespace(
                backtest_result=BacktestResult(
                    strategy_id=strat_s.meta.id,
                    batch_id="b_deg",
                    signals=ident_sigs,
                    total_signals=5,
                    executed_signals=5,
                    skipped_signals=0,
                    date_range=(short_oos_df["date"].iloc[28], short_oos_df["date"].iloc[39]),
                ),
                evaluation=EvaluationReport(
                    strategy_id=strat_s.meta.id,
                    overall=OverallMetrics(win_rate=1.0, avg_return=0.04, signal_count=5),
                ),
                portfolio_verdict=None,
            ),
        ),
    ):
        paired_deg = MechanismEvaluator()._run_single_variant(
            seed_strategy=strat_s,
            train_data=tr_s,
            fwd_data=fwd_s,
            fwd_cutoffs=cut_s,
            variant=HarnessVariant(variant_id="deg", name="Deg", require_null_control=False),
            max_rounds=1,
            seed=42,
        )
    assert paired_deg.false_promotion is True

    # 5. Legacy multi-AND trigger_blend remaps positional entry.conditions[0/1] accurately with guard offset
    p_pos = _make_strategy(
        "p_pos_blend",
        logic="and",
        conditions=[
            StrategyCondition(indicator="rsi_14", op="<=", value=35.0),
            StrategyCondition(indicator="close_above_ma20", op="==", value=True),
        ],
        guards=[StrategyCondition(indicator="ma5_above_ma10", op="==", value=True)],
        tunable=[
            TunableParam(target="entry.conditions[0].value", range=(20, 45), step=5),
            TunableParam(target="entry.conditions[1].value", range=(0, 1), step=1),
        ],
    )
    s_pos = _make_strategy(
        "s_pos_blend",
        conditions=[StrategyCondition(indicator="momentum_10d", op=">", value=0.02)],
    )
    pos_blend = next(
        c
        for c in SemanticCrossoverBreeder().breed(p_pos, s_pos, max_children=3)
        if c.operator == "trigger_blend"
    )
    rep_pos = StrategyMutator(max_changes=10).mutate(p_pos, pos_blend.summary_changes, atomic=True)
    expected_tgts = ["entry.triggers[0].value", "entry.guards[1].value"]
    assert [p.target for p in pos_blend.child_strategy.params.tunable] == expected_tgts
    assert [p.target for p in rep_pos.params.tunable] == expected_tgts
    assert resolve_tunable_target(rep_pos, "entry.triggers[0].value") == 35.0
    assert resolve_tunable_target(rep_pos, "entry.guards[1].value") is True

    # 6. Exit-graft None vs [] conditions replay faithfully
    p_none_c = _make_strategy(
        "p_none_c", sl=StopLossConfig(type="pct", value=0.04, conditions=None)
    )
    s_empty_c = _make_strategy(
        "s_empty_c",
        conditions=[StrategyCondition(indicator="momentum_10d", op=">", value=0.02)],
        sl=StopLossConfig(type="pct", value=0.04, conditions=[]),
    )
    empty_graft = next(
        c
        for c in SemanticCrossoverBreeder().breed(p_none_c, s_empty_c, max_children=3)
        if c.operator == "exit_graft"
    )
    assert any(c.target == "exit.stop_loss.conditions" for c in empty_graft.summary_changes)
    rep_empty_c = StrategyMutator(max_changes=10).mutate(
        p_none_c, empty_graft.summary_changes, atomic=True
    )
    assert rep_empty_c.exit.stop_loss.conditions == []
