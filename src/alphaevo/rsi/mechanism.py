"""L3 Meta-RSI Mechanism Experiment: evaluates research harnesses on forward calibration."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from statistics import mean

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from alphaevo.evaluator.metrics import Evaluator
from alphaevo.models.enums import RSILevel
from alphaevo.models.execution import BacktestResult, TradeSignal
from alphaevo.models.strategy import Strategy
from alphaevo.rsi.program import RSICharter
from alphaevo.truth.kernel import EvaluationKernel, PortfolioGateConfig
from alphaevo.truth.null_control import NullSignalController


class HarnessVariant(BaseModel):
    """Configuration of the research/verification harness tested in L3 Meta-RSI."""

    variant_id: str
    name: str
    require_null_control: bool = True
    enable_pinned_scars: bool = True
    complexity_limit: int = Field(default=6, ge=2, le=12)
    min_signal_gate: int = Field(default=15, ge=5)
    allowed_levels: list[RSILevel] = Field(default_factory=lambda: [RSILevel.L1_STRATEGY])
    require_portfolio_gate: bool = False
    require_strategy_rerun_null: bool = False
    strategy_rerun_null_worlds: int = Field(default=2, ge=1, le=10)


class PairedMechanismRun(BaseModel):
    """Out-of-sample forward calibration metrics for one harness variant."""

    variant_id: str
    variant_name: str
    champion_id: str
    train_signals: int
    train_avg_return: float
    forward_signals: int
    forward_avg_return: float
    forward_win_rate: float
    decay_ratio: float | None = None
    false_promotion: bool = False
    calibration_score: float = 0.0
    promotions_count: int = 0
    trials_recorded: int = 0
    forward_sharpe: float = 0.0
    forward_max_drawdown: float = 0.0
    forward_confidence_score: float = 0.0
    portfolio_gate_passed: bool = True


class MechanismFoldComparison(BaseModel):
    """Single temporal cutoff paired comparison in L3 Meta-RSI."""

    cutoff_ratio: float
    baseline_run: PairedMechanismRun
    candidate_run: PairedMechanismRun
    winner_variant_id: str


class MechanismExperimentResult(BaseModel):
    """Paired AB/BA L3 harness comparison outcome."""

    split_cutoff_ratio: float
    baseline_run: PairedMechanismRun
    candidate_run: PairedMechanismRun
    winner_variant_id: str
    rationale: str
    fold_comparisons: list[MechanismFoldComparison] = Field(default_factory=list)
    baseline_false_promotion_rate: float = 0.0
    candidate_false_promotion_rate: float = 0.0
    mean_calibration_delta: float = 0.0


def _temporal_split_market(
    market_data: dict[str, pd.DataFrame],
    train_ratio: float = 0.65,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Chronologically split each symbol DataFrame into train and forward slices."""
    train_data: dict[str, pd.DataFrame] = {}
    fwd_data: dict[str, pd.DataFrame] = {}
    for sym, df in market_data.items():
        if df.empty or len(df) < 35:
            continue
        ordered = (
            df.sort_values("date").reset_index(drop=True)
            if "date" in df
            else df.reset_index(drop=True)
        )
        split_idx = max(20, int(len(ordered) * train_ratio))
        # Include brief indicator warmup overlap ONLY before evaluation; simple clean slice if >= 25 forward bars
        if len(ordered) - split_idx < 20:
            split_idx = max(20, len(ordered) - 20)
        train_data[sym] = ordered.iloc[:split_idx].copy().reset_index(drop=True)
        fwd_data[sym] = ordered.iloc[split_idx:].copy().reset_index(drop=True)
    return train_data, fwd_data


def _temporal_split_with_warmup(
    market_data: dict[str, pd.DataFrame],
    train_ratio: float = 0.65,
    warmup_bars: int = 60,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, date]]:
    """Split each symbol into train and warmup-padded forward slices with a unified calendar cutoff."""
    train_data: dict[str, pd.DataFrame] = {}
    fwd_data: dict[str, pd.DataFrame] = {}
    fwd_cutoffs: dict[str, date] = {}
    ordered_by_sym: dict[str, pd.DataFrame] = {}
    for sym, df in market_data.items():
        if df.empty or len(df) < 35:
            continue
        ordered = (
            df.sort_values("date").reset_index(drop=True)
            if "date" in df
            else df.reset_index(drop=True)
        )
        ordered_by_sym[sym] = ordered

    dated_syms = {s for s, df in ordered_by_sym.items() if "date" in df.columns}
    cutoff_date: date | None = None
    min_side = 20
    if dated_syms:
        panel_dates = sorted(
            {
                d
                for sym in dated_syms
                for d in pd.to_datetime(ordered_by_sym[sym]["date"], errors="coerce")
                .dt.date.dropna()
                .tolist()
            }
        )
        if len(panel_dates) >= 25:
            min_side = 20 if len(panel_dates) >= 40 else 10
            cut_pos = max(
                min_side,
                min(len(panel_dates) - min_side, int(len(panel_dates) * train_ratio)),
            )
            cutoff_date = panel_dates[cut_pos]

    for sym, ordered in ordered_by_sym.items():
        if cutoff_date is not None and sym in dated_syms:
            sym_dates = pd.to_datetime(ordered["date"], errors="coerce").dt.date
            split_idx = int((sym_dates < cutoff_date).sum())
            if split_idx < min_side or len(ordered) - split_idx < min_side:
                continue
            train_data[sym] = ordered.iloc[:split_idx].copy().reset_index(drop=True)
            fwd_start = max(0, split_idx - warmup_bars)
            fwd_data[sym] = ordered.iloc[fwd_start:].copy().reset_index(drop=True)
            fwd_cutoffs[sym] = cutoff_date
        else:
            split_idx = max(20, int(len(ordered) * train_ratio))
            if len(ordered) - split_idx < 20:
                split_idx = max(20, len(ordered) - 20)
            work = ordered.copy()
            has_valid_dates = False
            parsed_split_date: date | None = None
            if "date" in work.columns:
                parsed_dates = pd.to_datetime(work["date"], errors="coerce").dt.date
                if not parsed_dates.isna().any() and isinstance(parsed_dates.iloc[split_idx], date):
                    has_valid_dates = True
                    parsed_split_date = parsed_dates.iloc[split_idx]
            if has_valid_dates and parsed_split_date is not None:
                fwd_cutoffs[sym] = parsed_split_date
            else:
                synth_dates = [d.date() for d in pd.bdate_range("2020-01-02", periods=len(work))]
                work["date"] = synth_dates
                fwd_cutoffs[sym] = synth_dates[split_idx]
            train_data[sym] = work.iloc[:split_idx].copy().reset_index(drop=True)
            fwd_start = max(0, split_idx - warmup_bars)
            fwd_data[sym] = work.iloc[fwd_start:].copy().reset_index(drop=True)
    return train_data, fwd_data, fwd_cutoffs


class MechanismEvaluator:
    """Runs frozen equal-budget paired AB/BA comparisons across L3 HarnessVariants."""

    def evaluate_ab(
        self,
        seed_strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        *,
        baseline_variant: HarnessVariant,
        candidate_variant: HarnessVariant,
        max_rounds: int = 3,
        train_ratio: float = 0.65,
        cutoff_ratios: Sequence[float] | None = None,
        seed: int = 42,
    ) -> MechanismExperimentResult:
        """Compare two harness variants strictly on unseen forward calibration."""
        ratios = list(cutoff_ratios) if cutoff_ratios else [train_ratio]
        fold_comparisons: list[MechanismFoldComparison] = []
        base_runs: list[PairedMechanismRun] = []
        cand_runs: list[PairedMechanismRun] = []

        for idx, ratio in enumerate(ratios):
            train_data, fwd_data, fwd_cutoffs = _temporal_split_with_warmup(
                market_data,
                train_ratio=ratio,
            )
            if not train_data or not fwd_data:
                raise ValueError(
                    f"Market dataset too short to form chronological train/forward split at ratio={ratio}"
                )

            base_run = self._run_single_variant(
                seed_strategy=seed_strategy,
                train_data=train_data,
                fwd_data=fwd_data,
                fwd_cutoffs=fwd_cutoffs,
                variant=baseline_variant,
                max_rounds=max_rounds,
                seed=seed + idx,
            )
            cand_run = self._run_single_variant(
                seed_strategy=seed_strategy,
                train_data=train_data,
                fwd_data=fwd_data,
                fwd_cutoffs=fwd_cutoffs,
                variant=candidate_variant,
                max_rounds=max_rounds,
                seed=seed + idx,
            )
            fold_winner = (
                cand_run.variant_id
                if cand_run.calibration_score > base_run.calibration_score + 1e-6
                else base_run.variant_id
            )
            fold_comparisons.append(
                MechanismFoldComparison(
                    cutoff_ratio=ratio,
                    baseline_run=base_run,
                    candidate_run=cand_run,
                    winner_variant_id=fold_winner,
                )
            )
            base_runs.append(base_run)
            cand_runs.append(cand_run)

        agg_base = self._aggregate_runs(base_runs, baseline_variant)
        agg_cand = self._aggregate_runs(cand_runs, candidate_variant)
        base_fp_rate = round(sum(1 for r in base_runs if r.false_promotion) / len(base_runs), 4)
        cand_fp_rate = round(sum(1 for r in cand_runs if r.false_promotion) / len(cand_runs), 4)
        cal_delta = round(agg_cand.calibration_score - agg_base.calibration_score, 6)

        if agg_cand.calibration_score > agg_base.calibration_score + 1e-6:
            winner = agg_cand.variant_id
            rationale = (
                f"Candidate harness '{agg_cand.variant_name}' achieved higher forward calibration "
                f"({agg_cand.calibration_score:.4f} vs {agg_base.calibration_score:.4f}) "
                f"across {len(fold_comparisons)} cutoff fold(s)"
            )
        else:
            winner = agg_base.variant_id
            rationale = (
                f"Baseline harness '{agg_base.variant_name}' held equal or better forward calibration "
                f"({agg_base.calibration_score:.4f} vs {agg_cand.calibration_score:.4f}) "
                f"across {len(fold_comparisons)} cutoff fold(s)"
            )

        return MechanismExperimentResult(
            split_cutoff_ratio=ratios[0],
            baseline_run=agg_base,
            candidate_run=agg_cand,
            winner_variant_id=winner,
            rationale=rationale,
            fold_comparisons=fold_comparisons,
            baseline_false_promotion_rate=base_fp_rate,
            candidate_false_promotion_rate=cand_fp_rate,
            mean_calibration_delta=cal_delta,
        )

    @staticmethod
    def _aggregate_runs(
        runs: list[PairedMechanismRun],
        variant: HarnessVariant,
    ) -> PairedMechanismRun:
        if len(runs) == 1:
            return runs[0]
        decay_vals = [r.decay_ratio for r in runs if r.decay_ratio is not None]
        return PairedMechanismRun(
            variant_id=variant.variant_id,
            variant_name=variant.name,
            champion_id=runs[-1].champion_id,
            train_signals=round(mean(r.train_signals for r in runs)),
            train_avg_return=round(mean(r.train_avg_return for r in runs), 6),
            forward_signals=round(mean(r.forward_signals for r in runs)),
            forward_avg_return=round(mean(r.forward_avg_return for r in runs), 6),
            forward_win_rate=round(mean(r.forward_win_rate for r in runs), 4),
            decay_ratio=round(mean(decay_vals), 4) if decay_vals else None,
            false_promotion=any(r.false_promotion for r in runs),
            calibration_score=round(mean(r.calibration_score for r in runs), 6),
            promotions_count=sum(r.promotions_count for r in runs),
            trials_recorded=sum(r.trials_recorded for r in runs),
            forward_sharpe=round(mean(r.forward_sharpe for r in runs), 4),
            forward_max_drawdown=round(mean(r.forward_max_drawdown for r in runs), 4),
            forward_confidence_score=round(mean(r.forward_confidence_score for r in runs), 4),
            portfolio_gate_passed=all(r.portfolio_gate_passed for r in runs),
        )

    def _run_single_variant(
        self,
        *,
        seed_strategy: Strategy,
        train_data: dict[str, pd.DataFrame],
        fwd_data: dict[str, pd.DataFrame],
        fwd_cutoffs: dict[str, date],
        variant: HarnessVariant,
        max_rounds: int,
        seed: int,
    ) -> PairedMechanismRun:
        from alphaevo.rsi.engine import StockRSIEngine

        charter = RSICharter(
            name=f"mech_{variant.variant_id}",
            seed_strategy_id=seed_strategy.meta.id,
            allowed_levels=list(variant.allowed_levels),
            max_rounds=max_rounds,
            max_candidates_per_round=2,
            max_trials_budget=max_rounds * 10 + 5,
            min_signals=variant.min_signal_gate,
            require_null_control=variant.require_null_control,
            require_strategy_rerun_null=variant.require_strategy_rerun_null,
            strategy_rerun_null_worlds=variant.strategy_rerun_null_worlds,
            require_portfolio_gate=variant.require_portfolio_gate,
            null_permutations=30,
        )
        rsi_engine = StockRSIEngine(
            db_path=":memory:",
            complexity_limit=variant.complexity_limit,
            enable_pinned_scars=variant.enable_pinned_scars,
        )
        try:
            res = rsi_engine.run(
                seed_strategy,
                train_data,
                charter=charter,
                seed=seed,
            )
        finally:
            rsi_engine.close()

        champ = res.champion_strategy
        kernel = EvaluationKernel(min_data_days=15)
        portfolio_cfg = (
            PortfolioGateConfig(enabled=True) if variant.require_portfolio_gate else None
        )
        groups_by_cutoff: dict[date, dict[str, pd.DataFrame]] = {}
        for sym, df in fwd_data.items():
            cut = fwd_cutoffs.get(sym)
            if cut is not None:
                groups_by_cutoff.setdefault(cut, {})[sym] = df

        if not groups_by_cutoff:
            today = date.today()
            fwd_bt = BacktestResult(
                strategy_id=champ.meta.id,
                batch_id="mech_fwd_empty",
                signals=[],
                total_signals=0,
                executed_signals=0,
                skipped_signals=0,
                date_range=(today, today),
            )
            fwd_eval = Evaluator().evaluate_fast(fwd_bt, champ, market_data=fwd_data)
            portfolio_ok = True
        elif len(groups_by_cutoff) == 1:
            sole_cutoff, sole_data = next(iter(groups_by_cutoff.items()))
            outcome = kernel.evaluate(
                champ,
                sole_data,
                mode="fast",
                record_trial=False,
                run_null_control=False,
                portfolio_gate=portfolio_cfg,
                min_entry_date=sole_cutoff,
            )
            raw_bt = outcome.backtest_result
            oos_signals = [
                sig
                for sig in raw_bt.signals
                if sig.symbol in fwd_cutoffs and sig.signal_date >= fwd_cutoffs[sig.symbol]
            ]
            if len(oos_signals) != len(raw_bt.signals) or len(sole_data) != len(fwd_data):
                fwd_bt = BacktestResult(
                    strategy_id=raw_bt.strategy_id,
                    batch_id=raw_bt.batch_id,
                    signals=oos_signals,
                    total_signals=len(oos_signals),
                    executed_signals=len([s for s in oos_signals if s.exit_price is not None]),
                    skipped_signals=len([s for s in oos_signals if s.exit_price is None]),
                    date_range=raw_bt.date_range,
                )
                fwd_eval = Evaluator().evaluate_fast(fwd_bt, champ, market_data=fwd_data)
                portfolio_ok = (
                    EvaluationKernel._evaluate_portfolio_gate(
                        fwd_bt, fwd_data, portfolio_cfg
                    ).passed
                    if portfolio_cfg is not None
                    else True
                )
            else:
                fwd_bt = raw_bt
                fwd_eval = outcome.evaluation
                portfolio_ok = (
                    outcome.portfolio_verdict.passed
                    if outcome.portfolio_verdict is not None
                    else True
                )
        else:
            oos_signals_multi: list[TradeSignal] = []
            group_ranges: list[tuple[date, date]] = []
            for cutoff, group_data in sorted(groups_by_cutoff.items()):
                group_outcome = kernel.evaluate(
                    champ,
                    group_data,
                    mode="fast",
                    record_trial=False,
                    run_null_control=False,
                    portfolio_gate=None,
                    min_entry_date=cutoff,
                )
                group_bt = group_outcome.backtest_result
                group_ranges.append(group_bt.date_range)
                for sig in group_bt.signals:
                    if sig.symbol in fwd_cutoffs and sig.signal_date >= fwd_cutoffs[sig.symbol]:
                        oos_signals_multi.append(sig)
            oos_signals_multi.sort(key=lambda s: (s.signal_date, s.symbol))
            combined_range = (
                min(r[0] for r in group_ranges),
                max(r[1] for r in group_ranges),
            )
            fwd_bt = BacktestResult(
                strategy_id=champ.meta.id,
                batch_id="mech_fwd_multi_cutoff",
                signals=oos_signals_multi,
                total_signals=len(oos_signals_multi),
                executed_signals=len([s for s in oos_signals_multi if s.exit_price is not None]),
                skipped_signals=len([s for s in oos_signals_multi if s.exit_price is None]),
                date_range=combined_range,
            )
            fwd_eval = Evaluator().evaluate_fast(fwd_bt, champ, market_data=fwd_data)
            portfolio_ok = (
                EvaluationKernel._evaluate_portfolio_gate(fwd_bt, fwd_data, portfolio_cfg).passed
                if portfolio_cfg is not None
                else True
            )

        train_avg = res.champion_evaluation.overall.avg_return
        fwd_avg = fwd_eval.overall.avg_return
        fwd_sigs = fwd_eval.overall.signal_count

        decay: float | None = None
        if abs(train_avg) > 1e-6:
            decay = round(fwd_avg / train_avg, 4)

        promoted_beyond_seed = res.promotions > 0
        fwd_null_ok = False
        executed_fwd = [s for s in fwd_bt.signals if s.exit_price is not None]
        if len(executed_fwd) >= 5:
            avg_hold_fwd = max(
                1,
                round(sum(max(1, s.holding_days) for s in executed_fwd) / len(executed_fwd)),
            )
            fwd_oos_only_market: dict[str, pd.DataFrame] = {}
            for sym, df in fwd_data.items():
                cut = fwd_cutoffs.get(sym)
                if cut is None or df.empty or "date" not in df.columns:
                    continue
                sym_dates = pd.to_datetime(df["date"], errors="coerce").dt.date
                oos_slice = df.loc[sym_dates >= cut].copy().reset_index(drop=True)
                if not oos_slice.empty:
                    fwd_oos_only_market[sym] = oos_slice
            has_market_null_pool = any(
                "close" in df.columns and len(df) >= avg_hold_fwd + 3
                for df in fwd_oos_only_market.values()
            )
            if has_market_null_pool:
                null_rep = NullSignalController().evaluate_candidate_against_nulls(
                    fwd_bt.signals,
                    fwd_oos_only_market,
                    n_permutations=25,
                    seed=seed,
                )
                fwd_null_ok = null_rep.passed
        false_promo = promoted_beyond_seed and (fwd_avg <= 0.0 or not fwd_null_ok)

        decay_bonus = 0.02 * min(1.5, max(0.0, decay)) if decay is not None else 0.0
        false_pen = 0.10 if false_promo else 0.0
        calibration = round(fwd_avg - false_pen + decay_bonus, 6)

        return PairedMechanismRun(
            variant_id=variant.variant_id,
            variant_name=variant.name,
            champion_id=champ.meta.id,
            train_signals=res.champion_evaluation.overall.signal_count,
            train_avg_return=round(train_avg, 6),
            forward_signals=fwd_sigs,
            forward_avg_return=round(fwd_avg, 6),
            forward_win_rate=fwd_eval.overall.win_rate,
            decay_ratio=decay,
            false_promotion=false_promo,
            calibration_score=calibration,
            promotions_count=res.promotions,
            trials_recorded=res.total_trials_recorded,
            forward_sharpe=fwd_eval.overall.sharpe_ratio,
            forward_max_drawdown=fwd_eval.overall.max_drawdown,
            forward_confidence_score=fwd_eval.confidence_score,
            portfolio_gate_passed=portfolio_ok,
        )

    @staticmethod
    def render_markdown(result: MechanismExperimentResult) -> str:
        """Render paired L3 mechanism AB/multi-cutoff calibration results as Markdown."""
        lines = [
            "# L3 Meta-RSI Mechanism Calibration Report",
            "",
            f"- Winner Harness: **{result.winner_variant_id}**",
            f"- Rationale: {result.rationale}",
            f"- Calibration Delta (Candidate - Baseline): {result.mean_calibration_delta:+.4f}",
            f"- False Promotion Rate (Baseline vs Candidate): {result.baseline_false_promotion_rate:.1%} vs {result.candidate_false_promotion_rate:.1%}",
            "",
            "| Harness Variant | Champion | Promotions | Trials | Train AvgRet | Fwd AvgRet | Fwd Sharpe | Fwd MaxDD | Decay Ratio | False Promo | Calibration |",
            "|-----------------|----------|------------|--------|--------------|------------|------------|-----------|-------------|-------------|-------------|",
        ]
        for run in (result.baseline_run, result.candidate_run):
            decay_str = f"{run.decay_ratio:.2f}x" if run.decay_ratio is not None else "n/a"
            lines.append(
                f"| {run.variant_name} | {run.champion_id} | {run.promotions_count} | {run.trials_recorded} "
                f"| {run.train_avg_return:.2%} | {run.forward_avg_return:.2%} | {run.forward_sharpe:.2f} "
                f"| {run.forward_max_drawdown:.1%} | {decay_str} | {'YES' if run.false_promotion else 'NO'} "
                f"| {run.calibration_score:.4f} |"
            )

        if result.fold_comparisons:
            lines += [
                "",
                "## Cutoff Fold Breakdown",
                "",
                "| Cutoff Ratio | Baseline FwdCal | Candidate FwdCal | Winner |",
                "|--------------|-----------------|------------------|--------|",
            ]
            for fc in result.fold_comparisons:
                lines.append(
                    f"| {fc.cutoff_ratio:.0%} | {fc.baseline_run.calibration_score:.4f} "
                    f"| {fc.candidate_run.calibration_score:.4f} | {fc.winner_variant_id} |"
                )

        lines += [
            "",
            "---",
            "*Research & educational software only. Mechanism AB calibration requires forward shadow validation.*",
            "",
        ]
        return "\n".join(lines)


__all__ = [
    "HarnessVariant",
    "MechanismEvaluator",
    "MechanismExperimentResult",
    "MechanismFoldComparison",
    "PairedMechanismRun",
]
