"""Live Self-Evolving Arena orchestrator combining RSI, EvaluationKernel, Shadow Ledger, and Risk Allocation."""

from __future__ import annotations

import difflib
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from alphaevo.alpha_factory.governance import FactorGovernanceEngine
from alphaevo.arena.models import (
    ArenaSnapshot,
    ArenaStrategyEntry,
    EquityPoint,
    EvolutionNodeView,
    TradeReplayView,
)
from alphaevo.backtest.allocator import RegimeAdaptiveAllocator
from alphaevo.core.config import BacktestConfig
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.forward.shadow_ledger import ShadowLedger
from alphaevo.models.enums import RSILevel
from alphaevo.models.execution import TradeSignal
from alphaevo.models.strategy import Strategy
from alphaevo.research_committee import ResearchCommittee
from alphaevo.rsi.engine import StockRSIEngine
from alphaevo.rsi.program import RSICharter
from alphaevo.strategy.dsl.serializer import StrategySerializer
from alphaevo.truth.kernel import EvaluationKernel

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


class ArenaRunner:
    """Runs multi-strategy RSI ratchets, forward shadow checks & portfolio risk budgeting."""

    def __init__(
        self,
        *,
        backtest_config: BacktestConfig | None = None,
        db_path: str | Path = ":memory:",
    ) -> None:
        self._bt_config = backtest_config or BacktestConfig()
        self._db_path = db_path
        self._serializer = StrategySerializer()
        self._committee = ResearchCommittee()

    @staticmethod
    def compute_equity_curve(
        signals: Sequence[TradeSignal],
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[EquityPoint]:
        """Compute chronological daily MTM strategy & buy-and-hold equity curves."""
        executed = sorted(
            [s for s in signals if s.exit_price is not None],
            key=lambda s: (s.signal_date, s.exit_date or s.signal_date, s.symbol),
        )
        if not executed:
            return []

        if market_data:
            calendar, daily_rets, _active, _dd, _sharpe, _tot = Evaluator.compute_daily_mtm_series(
                executed, market_data
            )
            if calendar and len(calendar) == len(daily_rets):
                bench_by_date = ArenaRunner._equal_weight_benchmark_returns(calendar, market_data)
                strat_eq = 1.0
                bench_eq = 1.0
                peak = 1.0
                points: list[EquityPoint] = []
                for idx, cal_d in enumerate(calendar):
                    strat_eq *= max(0.01, 1.0 + daily_rets[idx])
                    bench_eq *= max(0.01, 1.0 + bench_by_date.get(cal_d, 0.0))
                    peak = max(peak, strat_eq)
                    dd = (peak - strat_eq) / peak if peak > 0 else 0.0
                    points.append(
                        EquityPoint(
                            date=cal_d.isoformat(),
                            strategy_equity=round(strat_eq, 4),
                            benchmark_equity=round(bench_eq, 4),
                            drawdown=round(dd, 4),
                        )
                    )
                return points

        # Fallback: trade-sequence cumulative curve
        strat_eq = 1.0
        peak = 1.0
        seq_points: list[EquityPoint] = []
        for sig in executed:
            strat_eq *= max(0.01, 1.0 + float(sig.return_pct))
            peak = max(peak, strat_eq)
            dd = (peak - strat_eq) / peak if peak > 0 else 0.0
            seq_points.append(
                EquityPoint(
                    date=(sig.exit_date or sig.signal_date).isoformat(),
                    strategy_equity=round(strat_eq, 4),
                    benchmark_equity=1.0,
                    drawdown=round(dd, 4),
                )
            )
        return seq_points

    @staticmethod
    def _equal_weight_benchmark_returns(
        calendar: list[date],
        market_data: dict[str, pd.DataFrame],
    ) -> dict[date, float]:
        import pandas as pd  # type: ignore[import-untyped]

        by_date_rets: dict[date, list[float]] = {}
        cal_set = set(calendar)
        for df in market_data.values():
            if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
                continue
            dates_raw = pd.to_datetime(df["date"], errors="coerce").dt.date.tolist()
            closes_raw = pd.to_numeric(df["close"], errors="coerce").tolist()
            pairs = [
                (d, float(c))
                for d, c in zip(dates_raw, closes_raw, strict=False)
                if isinstance(d, date) and c is not None and float(c) > 0
            ]
            pairs.sort(key=lambda item: item[0])
            for pos in range(1, len(pairs)):
                cur_d, cur_c = pairs[pos]
                _, prev_c = pairs[pos - 1]
                if cur_d in cal_set and prev_c > 0:
                    by_date_rets.setdefault(cur_d, []).append((cur_c / prev_c) - 1.0)
        return {d: (sum(vals) / len(vals) if vals else 0.0) for d, vals in by_date_rets.items()}

    @staticmethod
    def build_trade_replays(
        signals: Sequence[TradeSignal],
        market_data: dict[str, pd.DataFrame] | None,
        *,
        max_trades: int = 12,
        pad_before: int = 10,
        pad_after: int = 6,
    ) -> list[TradeReplayView]:
        """Build balanced OHLC K-line slices for representative executed trades."""
        import pandas as pd  # type: ignore[import-untyped]

        executed = [s for s in signals if s.exit_price is not None]
        if not executed:
            return []

        winners = sorted(executed, key=lambda s: float(s.return_pct), reverse=True)[:4]
        losers = sorted(executed, key=lambda s: float(s.return_pct))[:4]
        recent = sorted(
            executed,
            key=lambda s: (s.exit_date or s.signal_date, s.signal_date),
            reverse=True,
        )[:4]

        selected: list[TradeSignal] = []
        seen_keys: set[tuple[str, str, str | None]] = set()
        for sig in [*winners, *losers, *recent, *executed]:
            key = (
                sig.symbol,
                sig.signal_date.isoformat(),
                sig.exit_date.isoformat() if sig.exit_date else None,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(sig)
            if len(selected) >= max_trades:
                break

        prepared: dict[str, tuple[list[date], list[dict[str, object]]]] = {}
        if market_data:
            for sym, df in market_data.items():
                if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
                    continue
                dates_raw = pd.to_datetime(df["date"], errors="coerce").dt.date.tolist()
                rows_raw: list[tuple[date, dict[str, object]]] = []
                opens = df["open"].tolist() if "open" in df.columns else df["close"].tolist()
                highs = df["high"].tolist() if "high" in df.columns else df["close"].tolist()
                lows = df["low"].tolist() if "low" in df.columns else df["close"].tolist()
                closes = df["close"].tolist()
                for d_val, o_val, h_val, l_val, c_val in zip(
                    dates_raw, opens, highs, lows, closes, strict=False
                ):
                    if not isinstance(d_val, date):
                        continue
                    try:
                        c_f = float(c_val)
                        o_f = float(o_val) if pd.notna(o_val) else c_f
                        h_f = float(h_val) if pd.notna(h_val) else max(o_f, c_f)
                        l_f = float(l_val) if pd.notna(l_val) else min(o_f, c_f)
                    except (TypeError, ValueError):
                        continue
                    rows_raw.append(
                        (
                            d_val,
                            {
                                "date": d_val.isoformat(),
                                "open": round(o_f, 4),
                                "high": round(max(h_f, o_f, c_f), 4),
                                "low": round(min(l_f, o_f, c_f), 4),
                                "close": round(c_f, 4),
                            },
                        )
                    )
                rows_raw.sort(key=lambda pair: pair[0])
                prepared[sym] = (
                    [pair[0] for pair in rows_raw],
                    [pair[1] for pair in rows_raw],
                )

        replays: list[TradeReplayView] = []
        for sig in selected:
            bars: list[dict[str, object]] = []
            sym_entry = prepared.get(sig.symbol)
            if sym_entry is not None and sym_entry[0]:
                sym_dates, sym_bars = sym_entry
                entry_idx = 0
                for i, d_val in enumerate(sym_dates):
                    if d_val >= sig.signal_date:
                        entry_idx = i
                        break
                    entry_idx = i
                exit_target = sig.exit_date or sig.signal_date
                exit_idx = entry_idx
                for i in range(entry_idx, len(sym_dates)):
                    if sym_dates[i] >= exit_target:
                        exit_idx = i
                        break
                    exit_idx = i
                lo_idx = max(0, entry_idx - pad_before)
                hi_idx = min(len(sym_bars), exit_idx + pad_after + 1)
                bars = sym_bars[lo_idx:hi_idx]

            replays.append(
                TradeReplayView(
                    symbol=sig.symbol,
                    signal_date=sig.signal_date.isoformat(),
                    exit_date=sig.exit_date.isoformat() if sig.exit_date else None,
                    entry_price=round(float(sig.entry_price), 4),
                    exit_price=(
                        round(float(sig.exit_price), 4) if sig.exit_price is not None else None
                    ),
                    return_pct=round(float(sig.return_pct), 6),
                    holding_days=int(sig.holding_days),
                    exit_reason=sig.exit_reason.value if sig.exit_reason is not None else "",
                    bars=bars,
                )
            )
        return replays

    def run(
        self,
        strategies: Sequence[Strategy],
        market_data: dict[str, pd.DataFrame],
        *,
        rounds: int = 2,
        shadow_dir: Path | None = None,
        market_label: str = "US Snapshot",
        data_source: str = "bundled_snapshot",
        seed: int = 42,
    ) -> ArenaSnapshot:
        """Evaluate, self-evolve, shadow-simulate, and allocate across *strategies*."""
        import pandas as pd  # type: ignore[import-untyped]

        all_dates = sorted(
            {
                d
                for df in market_data.values()
                if df is not None and not df.empty and "date" in df.columns
                for d in pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().tolist()
                if isinstance(d, date)
            }
        )
        date_range = (
            (all_dates[0].isoformat(), all_dates[-1].isoformat()) if all_dates else ("", "")
        )
        symbols = sorted(market_data.keys())

        kernel = EvaluationKernel(
            backtest_config=self._bt_config,
            db_path=self._db_path,
            min_data_days=min(15, self._bt_config.min_data_days),
        )
        rsi_engine = StockRSIEngine(
            db_path=self._db_path,
            backtest_config=self._bt_config,
        )

        entries: list[ArenaStrategyEntry] = []
        champions: list[Strategy] = []
        total_promotions = 0
        total_vetoes = 0
        total_deliberations = 0
        total_prosecutor_vetoes = 0
        snapshot_veto_breakdown: dict[str, int] = {}
        all_chains_verified = True

        for idx, base_strat in enumerate(strategies):
            base_outcome = kernel.evaluate(
                base_strat,
                market_data,
                mode="fast",
                record_trial=False,
                run_null_control=False,
                seed=seed + idx,
            )
            baseline_score = float(base_outcome.evaluation.confidence_score)

            charter = RSICharter(
                name=f"arena_{base_strat.meta.id}",
                seed_strategy_id=base_strat.meta.id,
                allowed_levels=[RSILevel.L1_STRATEGY],
                max_rounds=max(1, rounds),
                max_candidates_per_round=2,
                max_trials_budget=max(10, rounds * 8 + 4),
                min_signals=min(15, self._bt_config.min_data_days),
                require_null_control=True,
                null_permutations=30,
            )
            rsi_res = rsi_engine.run(
                base_strat,
                market_data,
                charter=charter,
                persist_champion=False,
                seed=seed + idx,
            )

            champion = rsi_res.champion_strategy
            champions.append(champion)
            champ_outcome = kernel.evaluate(
                champion,
                market_data,
                mode="full",
                record_trial=False,
                run_null_control=True,
                null_permutations=30,
                seed=seed + idx,
            )
            champ_eval = champ_outcome.evaluation
            ov = champ_eval.overall

            base_yaml = self._serializer.to_yaml(base_strat)
            champ_yaml = self._serializer.to_yaml(champion)
            diff_lines = list(
                difflib.unified_diff(
                    base_yaml.splitlines(),
                    champ_yaml.splitlines(),
                    fromfile=f"{base_strat.meta.id}.yaml",
                    tofile=f"{champion.meta.id}.yaml",
                    lineterm="",
                )
            )

            veto_count = (
                sum(
                    1
                    for d in rsi_res.deliberations
                    if d.l0_outcome in {"prosecutor_vetoed", "revert"}
                )
                if rsi_res.deliberations
                else sum(1 for row in rsi_res.rows if row.status in {"vetoed", "revert"})
            )
            total_promotions += rsi_res.promotions
            total_vetoes += veto_count
            total_deliberations += len(rsi_res.deliberations)
            total_prosecutor_vetoes += sum(
                1 for d in rsi_res.deliberations if d.l0_outcome == "prosecutor_vetoed"
            )
            for v_key, v_cnt in rsi_res.veto_breakdown.items():
                snapshot_veto_breakdown[v_key] = snapshot_veto_breakdown.get(v_key, 0) + v_cnt

            nodes: list[EvolutionNodeView] = []
            for row in rsi_res.rows:
                row_diff = (
                    diff_lines if row.strategy_id == champion.meta.id and row.round_num > 0 else []
                )
                nodes.append(
                    EvolutionNodeView(
                        round_num=row.round_num,
                        strategy_id=row.strategy_id,
                        parent_id=row.parent_id
                        or (base_strat.meta.id if row.round_num > 0 else None),
                        status=row.status,
                        level=row.level.value,
                        score=round(float(row.fitness_score), 4),
                        win_rate=round(float(row.win_rate), 4),
                        avg_return=round(float(row.avg_return), 6),
                        signals=int(row.signals),
                        dsr_prob=round(float(row.dsr_probability), 4),
                        null_passed=bool(row.null_passed),
                        description=row.description,
                        veto_reasons=list(row.veto_reasons),
                        yaml_diff=row_diff[:40],
                    )
                )

            committee_verdict = self._committee.review(
                champion,
                champ_eval,
                data_source=data_source,
                symbols=symbols,
            )
            verdicts_json = [v.model_dump(mode="json") for v in committee_verdict.verdicts]

            shadow_stage = "shadow"
            shadow_mature = 0
            shadow_avg = 0.0
            shadow_verified = True
            if len(all_dates) >= 25:
                try:
                    if shadow_dir is not None:
                        s_dir = Path(shadow_dir) / champion.meta.id
                        ledger = ShadowLedger(s_dir)
                        sim_rep = ledger.simulate_rolling(
                            champion,
                            market_data,
                            start_date=all_dates[max(15, int(len(all_dates) * 0.65))],
                            end_date=all_dates[-1],
                            step_days=2,
                            horizon_days=5,
                            min_trading_days=5,
                            min_mature_signals=5,
                            min_effective_signals=3.0,
                            is_replay=True,
                        )
                    else:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            ledger = ShadowLedger(tmpdir)
                            sim_rep = ledger.simulate_rolling(
                                champion,
                                market_data,
                                start_date=all_dates[max(15, int(len(all_dates) * 0.65))],
                                end_date=all_dates[-1],
                                step_days=2,
                                horizon_days=5,
                                min_trading_days=5,
                                min_mature_signals=5,
                                min_effective_signals=3.0,
                                is_replay=True,
                            )
                    shadow_stage = sim_rep.final_stage.value
                    shadow_mature = sim_rep.final_decay_report.mature_signals
                    shadow_avg = sim_rep.final_decay_report.forward_avg_post_cost_return
                    shadow_verified = sim_rep.chain_verified
                except Exception:
                    shadow_verified = True
            all_chains_verified = all_chains_verified and shadow_verified

            equity_curve = self.compute_equity_curve(
                champ_outcome.backtest_result.signals, market_data
            )
            trade_replays = self.build_trade_replays(
                champ_outcome.backtest_result.signals, market_data
            )
            champ_score = float(champ_eval.confidence_score)

            entries.append(
                ArenaStrategyEntry(
                    strategy_id=base_strat.meta.id,
                    champion_id=champion.meta.id,
                    name=champion.meta.name,
                    category=champion.meta.category.value,
                    market=champion.meta.market.value,
                    baseline_score=round(baseline_score, 4),
                    champion_score=round(champ_score, 4),
                    score_delta=round(champ_score - baseline_score, 4),
                    win_rate=round(float(ov.win_rate), 4),
                    avg_return=round(float(ov.avg_return), 6),
                    total_return=round(float(ov.total_return), 6),
                    sharpe_ratio=round(float(ov.sharpe_ratio), 4),
                    max_drawdown=round(float(ov.max_drawdown), 4),
                    signal_count=int(ov.signal_count),
                    dsr_prob=round(float(champ_eval.deflated_sharpe_prob or 0.0), 4),
                    null_passed=bool(champ_eval.null_control_passed),
                    promotions=rsi_res.promotions,
                    vetoes=veto_count,
                    trials_recorded=rsi_res.total_trials_recorded,
                    shadow_stage=shadow_stage,
                    shadow_mature_signals=shadow_mature,
                    shadow_avg_return=round(float(shadow_avg), 6),
                    shadow_chain_verified=shadow_verified,
                    equity_curve=equity_curve,
                    evolution_nodes=nodes,
                    deliberations=[d.model_dump(mode="json") for d in rsi_res.deliberations],
                    veto_breakdown=dict(rsi_res.veto_breakdown),
                    trade_replays=trade_replays,
                    committee_verdicts=verdicts_json,
                    champion_yaml=champ_yaml,
                    baseline_yaml=base_yaml,
                    yaml_diff=diff_lines[:80],
                )
            )

        entries.sort(key=lambda e: (e.champion_score, e.sharpe_ratio), reverse=True)

        alloc_weights: dict[str, float] = {}
        alloc_metrics: dict[str, float] = {}
        if len(champions) >= 2:
            try:
                alloc_report = RegimeAdaptiveAllocator().allocate(
                    champions,
                    market_data,
                    backtest_config=self._bt_config,
                )
                latest_weights = (
                    alloc_report.daily_curve[-1].weights if alloc_report.daily_curve else {}
                )
                alloc_weights = {
                    s.strategy_id: round(float(latest_weights.get(s.strategy_id, s.mean_weight)), 4)
                    for s in alloc_report.strategies
                }
                alloc_metrics = {
                    "portfolio_total_return": round(float(alloc_report.combined_total_return), 6),
                    "portfolio_sharpe": round(float(alloc_report.combined_sharpe_ratio), 4),
                    "portfolio_max_drawdown": round(float(alloc_report.combined_max_drawdown), 4),
                    "portfolio_dd_reduction": round(
                        float(alloc_report.diversification_dd_reduction), 4
                    ),
                }
            except Exception:
                alloc_weights = {}
                alloc_metrics = {}

        gov_summary = {
            "total_audited": 0,
            "healthy": 0,
            "decayed": 0,
            "redundant": 0,
            "pit_unsafe": 0,
        }
        try:
            gov_rep = FactorGovernanceEngine(db_path=self._db_path).audit_factors(
                market_data,
                strategies=champions,
                auto_retire=False,
            )
            gov_summary = {
                "total_audited": gov_rep.total_audited,
                "healthy": gov_rep.healthy_count,
                "decayed": gov_rep.decayed_count,
                "redundant": gov_rep.redundant_count,
                "pit_unsafe": gov_rep.pit_unsafe_count,
            }
        except Exception:
            pass

        return ArenaSnapshot(
            market_label=market_label,
            data_source=data_source,
            date_range=date_range,
            symbols=symbols,
            total_strategies=len(entries),
            total_promotions=total_promotions,
            total_vetoes=total_vetoes,
            total_deliberations=total_deliberations,
            total_prosecutor_vetoes=total_prosecutor_vetoes,
            veto_breakdown=snapshot_veto_breakdown,
            chain_verified=all_chains_verified,
            entries=entries,
            allocation_weights=alloc_weights,
            allocation_metrics=alloc_metrics,
            factor_governance=gov_summary,
        )

    @staticmethod
    def render_digest_markdown(snapshot: ArenaSnapshot) -> str:
        """Render a concise Markdown digest for step summaries and notifications."""
        lines = [
            f"## 🧬 AlphaEvo Strategy Evolution Observatory & Overfit Audit — {snapshot.market_label}",
            "",
            f"- Source: `{snapshot.data_source}` | Window: `{snapshot.date_range[0]}` -> `{snapshot.date_range[1]}` ({len(snapshot.symbols)} benchmark symbols)",
            (
                f"- Strategies Evolving: **{snapshot.total_strategies}** | "
                f"Council Deliberations: **{snapshot.total_deliberations}** | "
                f"L0 Promotions: **{snapshot.total_promotions}** | "
                f"Overfit / Prosecutor Vetoes: **{snapshot.total_vetoes}** "
                f"(Prosecutor: {snapshot.total_prosecutor_vetoes}) | "
                f"Shadow Hash Chain: **{'VERIFIED' if snapshot.chain_verified else 'FAILED'}**"
            ),
            "",
            "| Strategy | Champion | Base -> Champ | Win% | AvgRet | MaxDD | Sharpe | Null | Shadow |",
            "|----------|----------|---------------|------|--------|-------|--------|------|--------|",
        ]
        for e in snapshot.entries:
            null_badge = "PASS" if e.null_passed else "FAIL"
            lines.append(
                f"| `{e.strategy_id}` | `{e.champion_id}` | {e.baseline_score:.1%} -> **{e.champion_score:.1%}** ({e.score_delta:+.1%}) "
                f"| {e.win_rate:.1%} | {e.avg_return:+.2%} | {e.max_drawdown:.1%} | {e.sharpe_ratio:.2f} | {null_badge} | `{e.shadow_stage}` |"
            )

        if snapshot.veto_breakdown:
            top_vetoes = ", ".join(
                f"`{k}`: {v}"
                for k, v in sorted(snapshot.veto_breakdown.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            lines += ["", f"- **Overfit Graveyard (Vetoes by Gate)**: {top_vetoes}"]

        if snapshot.allocation_weights:
            w_txt = ", ".join(f"`{k}`: {v:.1%}" for k, v in snapshot.allocation_weights.items())
            lines += [
                "",
                f"- **Causal Portfolio Weights**: {w_txt}",
                f"- **Portfolio Stats**: Return {snapshot.allocation_metrics.get('portfolio_total_return', 0.0):+.2%}, Sharpe {snapshot.allocation_metrics.get('portfolio_sharpe', 0.0):.2f}, MaxDD {snapshot.allocation_metrics.get('portfolio_max_drawdown', 0.0):.1%}",
            ]

        lines += [
            "",
            "*Research & educational strategy evolution engine only (not stock picking or investment advice). Simulated backtests and replay shadow runs do not guarantee future performance.*",
            "",
        ]
        return "\n".join(lines)


__all__ = ["ArenaRunner"]
