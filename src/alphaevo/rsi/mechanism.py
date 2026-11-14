"""L3 Meta-RSI Mechanism Experiment: evaluates research harnesses on forward calibration."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.models.enums import RSILevel, SamplingMethod
from alphaevo.models.execution import SampleBatch
from alphaevo.models.strategy import Strategy
from alphaevo.rsi.program import RSICharter
from alphaevo.truth.null_control import NullSignalController


class HarnessVariant(BaseModel):
    """Configuration of the research/verification harness tested in L3 Meta-RSI."""

    variant_id: str
    name: str
    require_null_control: bool = True
    enable_pinned_scars: bool = True
    complexity_limit: int = Field(default=6, ge=2, le=12)
    min_signal_gate: int = Field(default=15, ge=5)


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


class MechanismExperimentResult(BaseModel):
    """Paired AB/BA L3 harness comparison outcome."""

    split_cutoff_ratio: float
    baseline_run: PairedMechanismRun
    candidate_run: PairedMechanismRun
    winner_variant_id: str
    rationale: str


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
        seed: int = 42,
    ) -> MechanismExperimentResult:
        """Compare two harness variants strictly on unseen forward calibration."""
        train_data, fwd_data = _temporal_split_market(market_data, train_ratio=train_ratio)
        if not train_data or not fwd_data:
            raise ValueError("Market dataset too short to form chronological train/forward split")

        base_run = self._run_single_variant(
            seed_strategy=seed_strategy,
            train_data=train_data,
            fwd_data=fwd_data,
            variant=baseline_variant,
            max_rounds=max_rounds,
            seed=seed,
        )
        cand_run = self._run_single_variant(
            seed_strategy=seed_strategy,
            train_data=train_data,
            fwd_data=fwd_data,
            variant=candidate_variant,
            max_rounds=max_rounds,
            seed=seed,
        )

        if cand_run.calibration_score > base_run.calibration_score + 1e-6:
            winner = cand_run.variant_id
            rationale = (
                f"Candidate harness '{cand_run.variant_name}' achieved higher forward calibration "
                f"({cand_run.calibration_score:.4f} vs {base_run.calibration_score:.4f})"
            )
        else:
            winner = base_run.variant_id
            rationale = (
                f"Baseline harness '{base_run.variant_name}' held equal or better forward calibration "
                f"({base_run.calibration_score:.4f} vs {cand_run.calibration_score:.4f})"
            )

        return MechanismExperimentResult(
            split_cutoff_ratio=train_ratio,
            baseline_run=base_run,
            candidate_run=cand_run,
            winner_variant_id=winner,
            rationale=rationale,
        )

    def _run_single_variant(
        self,
        *,
        seed_strategy: Strategy,
        train_data: dict[str, pd.DataFrame],
        fwd_data: dict[str, pd.DataFrame],
        variant: HarnessVariant,
        max_rounds: int,
        seed: int,
    ) -> PairedMechanismRun:
        from alphaevo.rsi.engine import StockRSIEngine

        charter = RSICharter(
            name=f"mech_{variant.variant_id}",
            seed_strategy_id=seed_strategy.meta.id,
            allowed_levels=[RSILevel.L1_STRATEGY],
            max_rounds=max_rounds,
            max_candidates_per_round=2,
            max_trials_budget=max_rounds * 10 + 5,
            min_signals=variant.min_signal_gate,
            require_null_control=variant.require_null_control,
            null_permutations=30,
        )
        rsi_engine = StockRSIEngine(
            db_path=":memory:",
            complexity_limit=variant.complexity_limit,
            enable_pinned_scars=variant.enable_pinned_scars,
        )
        res = rsi_engine.run(
            seed_strategy,
            train_data,
            charter=charter,
            seed=seed,
        )

        champ = res.champion_strategy
        bt_engine = BacktestEngine(min_data_days=15)
        evaluator = Evaluator()
        fwd_symbols = sorted(fwd_data.keys())
        dates = [
            pd.to_datetime(df["date"]).dt.date.min()
            for df in fwd_data.values()
            if "date" in df and not df.empty
        ]
        end_dates = [
            pd.to_datetime(df["date"]).dt.date.max()
            for df in fwd_data.values()
            if "date" in df and not df.empty
        ]
        batch = SampleBatch(
            batch_id="mech_fwd",
            strategy_id=champ.meta.id,
            symbols=fwd_symbols,
            date_range=(min(dates), max(end_dates)),
            sampling_method=SamplingMethod.REPRESENTATIVE,
        )
        fwd_bt = bt_engine.run(champ, fwd_data, batch)
        fwd_eval = evaluator.evaluate_fast(fwd_bt, champ)

        train_avg = res.champion_evaluation.overall.avg_return
        fwd_avg = fwd_eval.overall.avg_return
        fwd_sigs = fwd_eval.overall.signal_count

        decay: float | None = None
        if abs(train_avg) > 1e-6:
            decay = round(fwd_avg / train_avg, 4)

        promoted_beyond_seed = res.promotions > 0
        fwd_null_ok = False
        if fwd_sigs >= 5:
            null_rep = NullSignalController().evaluate_candidate_against_nulls(
                fwd_bt.signals,
                fwd_data,
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
        )


__all__ = [
    "HarnessVariant",
    "MechanismEvaluator",
    "MechanismExperimentResult",
    "PairedMechanismRun",
]
