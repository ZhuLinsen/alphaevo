"""Shared Zero-Bypass EvaluationKernel, Locked Holdout Vault, and Portfolio Veto Gate."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.backtest.portfolio import PortfolioBacktester, PortfolioConfig
from alphaevo.core.config import BacktestConfig
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.models.enums import RSILevel, SamplingMethod
from alphaevo.models.execution import (
    BacktestResult,
    EvaluationReport,
    SampleBatch,
    TradeSignal,
)
from alphaevo.models.market import IndicatorContext
from alphaevo.models.strategy import Strategy
from alphaevo.truth.ledger import (
    DeflatedSharpeResult,
    TrialLedger,
    TrialRecord,
    compute_data_fingerprint,
)
from alphaevo.truth.null_control import (
    NullControlReport,
    NullDistributionStats,
    NullSignalController,
)


class LockedHoldoutSplit(BaseModel):
    """Metadata describing a chronological locked out-of-sample holdout split."""

    cutoff_date: date
    holdout_start_date: date
    search_dates: tuple[date, date]
    holdout_dates: tuple[date, date]
    holdout_pct: float
    warmup_bars: int


class PortfolioGateConfig(BaseModel):
    """Configuration for the Layer-2 portfolio-realizability hard veto gate."""

    enabled: bool = False
    initial_capital: float = Field(default=100_000.0, gt=0)
    max_positions: int = Field(default=5, ge=1)
    position_size_pct: float = Field(default=0.20, gt=0, le=1.0)
    max_portfolio_drawdown: float = Field(default=0.30, gt=0, le=1.0)
    min_portfolio_total_return: float = -0.05
    min_portfolio_trades: int = Field(default=1, ge=0)


class PortfolioGateVerdict(BaseModel):
    """Result of evaluating a strategy under Layer-2 portfolio constraints."""

    passed: bool
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    capital_utilization: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class KernelEvaluationOutcome(BaseModel):
    """Unified output emitted by :class:`EvaluationKernel`."""

    strategy_id: str
    family_id: str
    data_fingerprint: str
    backtest_result: BacktestResult
    evaluation: EvaluationReport
    dsr_result: DeflatedSharpeResult | None = None
    null_control: NullControlReport | None = None
    strategy_rerun_null: NullControlReport | None = None
    portfolio_verdict: PortfolioGateVerdict | None = None
    trial_record: TrialRecord | None = None
    veto_reasons: list[str] = Field(default_factory=list)
    passed_all_gates: bool = True


def _coerce_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            return None
        return cast("date", parsed.date())
    except Exception:
        return None


def _date_series(df: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(df["date"], errors="coerce")
    return parsed.dt.date


class LockedHoldoutVault:
    """Enforces physical isolation between offline search data and locked holdout bars."""

    def __init__(
        self,
        *,
        search_data: dict[str, pd.DataFrame],
        holdout_data: dict[str, pd.DataFrame],
        split: LockedHoldoutSplit,
    ) -> None:
        self._search_data = {
            sym: df.copy(deep=False).reset_index(drop=True)
            for sym, df in search_data.items()
            if not df.empty
        }
        self._holdout_data = {
            sym: df.copy(deep=False).reset_index(drop=True)
            for sym, df in holdout_data.items()
            if not df.empty
        }
        self.split = split

    @property
    def search_data(self) -> dict[str, pd.DataFrame]:
        """Return shallow copies of search-window data strictly ``<= cutoff_date``."""
        return {sym: df.copy(deep=False) for sym, df in self._search_data.items()}

    def evaluate_locked_holdout(
        self,
        strategy: Strategy,
        kernel: EvaluationKernel,
        *,
        contexts: dict[str, IndicatorContext] | None = None,
        mode: Literal["fast", "full"] = "full",
        record_trial: bool = True,
        portfolio_gate: PortfolioGateConfig | None = None,
    ) -> KernelEvaluationOutcome:
        """Evaluate *strategy* strictly out-of-sample on locked holdout dates."""
        return kernel.evaluate(
            strategy,
            self._holdout_data,
            contexts=contexts,
            mode=mode,
            record_trial=record_trial,
            portfolio_gate=portfolio_gate,
            min_entry_date=self.split.holdout_start_date,
            notes="locked_holdout",
        )


def split_market_data_with_holdout(
    market_data: dict[str, pd.DataFrame],
    *,
    holdout_pct: float = 0.20,
    warmup_bars: int = 60,
    min_search_bars: int = 30,
    min_holdout_bars: int = 10,
) -> tuple[LockedHoldoutVault, LockedHoldoutSplit]:
    """Chronologically partition market data into isolated search and locked holdout vaults."""
    if not (0.0 < holdout_pct < 0.5):
        raise ValueError(f"holdout_pct must be in (0, 0.5), got {holdout_pct}")

    unique_dates: set[date] = set()
    for df in market_data.values():
        if df.empty or "date" not in df.columns:
            continue
        for item in _date_series(df).dropna().tolist():
            if isinstance(item, date):
                unique_dates.add(item)

    all_dates = sorted(unique_dates)
    required = max(2, min_search_bars + min_holdout_bars)
    if len(all_dates) < required:
        raise ValueError(
            f"Need at least {required} unique trading dates to carve locked holdout, got {len(all_dates)}"
        )

    raw_cutoff = int(len(all_dates) * (1.0 - holdout_pct)) - 1
    cutoff_idx = max(
        min_search_bars - 1,
        min(len(all_dates) - min_holdout_bars - 1, raw_cutoff),
    )
    cutoff_date = all_dates[cutoff_idx]
    holdout_start_date = all_dates[cutoff_idx + 1]
    warmup_idx = max(0, cutoff_idx - max(0, warmup_bars) + 1)
    warmup_start_date = all_dates[warmup_idx]

    search_data: dict[str, pd.DataFrame] = {}
    holdout_data: dict[str, pd.DataFrame] = {}
    for symbol, df in market_data.items():
        if df.empty or "date" not in df.columns:
            continue
        dates = _date_series(df)
        search_slice = df.loc[dates <= cutoff_date]
        if not search_slice.empty:
            search_data[symbol] = cast("pd.DataFrame", search_slice.reset_index(drop=True))
        holdout_slice = df.loc[dates >= warmup_start_date]
        if not holdout_slice.empty:
            holdout_data[symbol] = cast("pd.DataFrame", holdout_slice.reset_index(drop=True))

    split = LockedHoldoutSplit(
        cutoff_date=cutoff_date,
        holdout_start_date=holdout_start_date,
        search_dates=(all_dates[0], cutoff_date),
        holdout_dates=(holdout_start_date, all_dates[-1]),
        holdout_pct=holdout_pct,
        warmup_bars=warmup_bars,
    )
    vault = LockedHoldoutVault(
        search_data=search_data,
        holdout_data=holdout_data,
        split=split,
    )
    return vault, split


class EvaluationKernel:
    """Single canonical entry point coupling backtest, daily MTM, DSR, null gates & Layer-2 veto."""

    def __init__(
        self,
        *,
        backtest_config: BacktestConfig | None = None,
        db_path: str | Path | None = None,
        trial_ledger: TrialLedger | None = None,
        null_controller: NullSignalController | None = None,
        min_data_days: int | None = None,
    ) -> None:
        self._backtest_config = backtest_config or BacktestConfig()
        self._min_data_days = (
            min_data_days if min_data_days is not None else self._backtest_config.min_data_days
        )
        self._trial_ledger = trial_ledger or (TrialLedger(db_path) if db_path is not None else None)
        self._null_controller = null_controller or NullSignalController()
        self._evaluator = Evaluator()
        self._factors_loaded = False
        if db_path is not None:
            self._ensure_custom_factors(db_path)

    def _ensure_custom_factors(self, db_path: str | Path) -> None:
        if self._factors_loaded or str(db_path) == ":memory:":
            return
        resolved = Path(db_path).expanduser()
        if not resolved.is_file():
            return
        try:
            from alphaevo.alpha_factory.factory import load_registered_factors

            load_registered_factors(resolved, status="active")
        except Exception:
            pass
        self._factors_loaded = True

    def _build_engine(self) -> BacktestEngine:
        return BacktestEngine(
            slippage=self._backtest_config.slippage,
            commission=self._backtest_config.commission,
            min_data_days=self._min_data_days,
            fill_policy=self._backtest_config.fill_policy,
        )

    @staticmethod
    def _default_batch(
        strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        min_entry_date: date | None,
        max_entry_date: date | None,
    ) -> SampleBatch:
        dates: list[date] = []
        for df in market_data.values():
            if df.empty or "date" not in df.columns:
                continue
            for item in _date_series(df).dropna().tolist():
                if isinstance(item, date):
                    dates.append(item)
        start_d = min(dates) if dates else date.today()
        end_d = max(dates) if dates else date.today()
        if min_entry_date is not None and min_entry_date > start_d:
            start_d = min_entry_date
        if max_entry_date is not None and max_entry_date < end_d:
            end_d = max_entry_date
        if start_d > end_d:
            start_d = end_d
        return SampleBatch(
            batch_id="kernel_eval",
            strategy_id=strategy.meta.id,
            symbols=sorted(market_data.keys()),
            date_range=(start_d, end_d),
            sampling_method=SamplingMethod.REPRESENTATIVE,
        )

    @staticmethod
    def _empty_null_report(method: str, reason: str) -> NullControlReport:
        empty = NullDistributionStats(method=method, n_permutations=0)
        return NullControlReport(
            observed_avg_return=0.0,
            observed_signal_count=0,
            block_shuffle=empty,
            time_shift=empty,
            random_entry=empty,
            worst_null_threshold=0.0,
            worst_p_value=1.0,
            passed=False,
            reasons=[reason],
        )

    @staticmethod
    def _evaluate_portfolio_gate(
        bt_result: BacktestResult,
        market_data: dict[str, pd.DataFrame],
        cfg: PortfolioGateConfig,
    ) -> PortfolioGateVerdict:
        pf = PortfolioBacktester(
            PortfolioConfig(
                initial_capital=cfg.initial_capital,
                max_positions=cfg.max_positions,
                position_size_pct=cfg.position_size_pct,
            )
        )
        pf_res = pf.simulate(bt_result, market_data=market_data)
        reasons: list[str] = []
        if pf_res.total_trades < cfg.min_portfolio_trades:
            reasons.append(f"trades<{cfg.min_portfolio_trades} ({pf_res.total_trades})")
        if pf_res.max_drawdown > cfg.max_portfolio_drawdown:
            reasons.append(f"drawdown>{cfg.max_portfolio_drawdown:.1%} ({pf_res.max_drawdown:.1%})")
        if pf_res.total_return < cfg.min_portfolio_total_return:
            reasons.append(
                f"return<{cfg.min_portfolio_total_return:.1%} ({pf_res.total_return:.1%})"
            )
        return PortfolioGateVerdict(
            passed=len(reasons) == 0,
            total_return=round(float(pf_res.total_return), 6),
            max_drawdown=round(float(pf_res.max_drawdown), 6),
            sharpe_ratio=round(float(pf_res.sharpe_ratio), 4),
            total_trades=int(pf_res.total_trades),
            capital_utilization=round(float(pf_res.capital_utilization), 4),
            reasons=reasons,
        )

    def evaluate(
        self,
        strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        *,
        batch: SampleBatch | None = None,
        contexts: dict[str, IndicatorContext] | None = None,
        mode: Literal["fast", "full"] = "fast",
        family_id: str | None = None,
        record_trial: bool = True,
        run_null_control: bool = False,
        run_strategy_rerun_null: bool = False,
        null_permutations: int = 60,
        null_worlds: int = 5,
        min_signals: int = 20,
        min_dsr_prob: float = 0.0,
        max_drawdown: float | None = None,
        portfolio_gate: PortfolioGateConfig | None = None,
        min_entry_date: date | None = None,
        max_entry_date: date | None = None,
        rsi_level: RSILevel = RSILevel.L1_STRATEGY,
        round_num: int = 0,
        accepted: bool = False,
        notes: str = "",
        seed: int = 42,
    ) -> KernelEvaluationOutcome:
        """Evaluate *strategy* cleanly through the unified truth kernel."""
        bt_engine = self._build_engine()
        active_batch = batch or self._default_batch(
            strategy, market_data, min_entry_date, max_entry_date
        )
        bt_result = bt_engine.run(
            strategy,
            market_data,
            active_batch,
            contexts=contexts,
            min_entry_date=min_entry_date,
            max_entry_date=max_entry_date,
        )

        if mode == "full":
            evaluation = self._evaluator.evaluate(
                bt_result,
                strategy,
                market_data=market_data,
                contexts=contexts,
                backtest_config=self._backtest_config,
            )
        else:
            evaluation = self._evaluator.evaluate_fast(
                bt_result,
                strategy,
                market_data=market_data,
                contexts=contexts,
            )

        executed: list[TradeSignal] = [s for s in bt_result.signals if s.exit_price is not None]
        resolved_family = (
            self._trial_ledger.resolve_trial_family_id(strategy, family_id)
            if self._trial_ledger is not None
            else (family_id or strategy.meta.family_id)
        )
        data_fp = compute_data_fingerprint(market_data)

        dsr_returns = (
            self._evaluator.compute_daily_mtm_returns(executed, market_data)[0]
            if evaluation.overall.metrics_basis == "daily_mtm" and market_data
            else []
        )
        dsr_input: list[float] = dsr_returns if dsr_returns else [s.return_pct for s in executed]
        dsr_obs = len(dsr_input) if dsr_returns else evaluation.overall.signal_count

        dsr_result: DeflatedSharpeResult | None = None
        if self._trial_ledger is not None:
            prior = self._trial_ledger.count_family_trials(resolved_family)
            trial_n = max(1, prior + (1 if record_trial else 0))
            dsr_result = self._trial_ledger.compute_dsr_for_trials(
                family_id=resolved_family,
                n_trials=trial_n,
                returns=dsr_input,
                observed_sharpe=evaluation.overall.sharpe_ratio,
                signal_count=dsr_obs,
            )
            evaluation.family_trial_count = trial_n
            evaluation.deflated_sharpe_prob = dsr_result.dsr_probability

        null_control: NullControlReport | None = None
        if run_null_control:
            if len(executed) < min_signals or evaluation.overall.avg_return <= 0.0:
                null_control = self._empty_null_report(
                    "block_shuffle",
                    f"signals={len(executed)}<{min_signals} or avg_return={evaluation.overall.avg_return:.4f}<=0",
                )
            else:
                null_control = self._null_controller.evaluate_candidate_against_nulls(
                    bt_result.signals,
                    market_data=market_data,
                    n_permutations=null_permutations,
                    min_signals=min_signals,
                    seed=seed,
                )

        strategy_rerun_null: NullControlReport | None = None
        if run_strategy_rerun_null:
            if len(executed) < min_signals or evaluation.overall.avg_return <= 0.0:
                strategy_rerun_null = self._empty_null_report(
                    "strategy_rerun",
                    f"signals={len(executed)}<{min_signals} or avg_return={evaluation.overall.avg_return:.4f}<=0",
                )
            else:
                strategy_rerun_null = self._null_controller.evaluate_strategy_against_null_worlds(
                    strategy,
                    market_data,
                    evaluation,
                    n_worlds=null_worlds,
                    seed=seed,
                    backtest_engine=bt_engine,
                )

        if null_control is not None or strategy_rerun_null is not None:
            passed_null = (True if null_control is None else null_control.passed) and (
                True if strategy_rerun_null is None else strategy_rerun_null.passed
            )
            p_vals = [
                rep.worst_p_value for rep in (null_control, strategy_rerun_null) if rep is not None
            ]
            evaluation.null_control_passed = passed_null
            evaluation.null_p_value = round(max(p_vals), 4) if p_vals else None

        portfolio_verdict: PortfolioGateVerdict | None = None
        if portfolio_gate is not None and portfolio_gate.enabled:
            portfolio_verdict = self._evaluate_portfolio_gate(
                bt_result,
                market_data,
                portfolio_gate,
            )

        trial_record: TrialRecord | None = None
        if record_trial and self._trial_ledger is not None:
            trial_record = self._trial_ledger.record_trial(
                strategy=strategy,
                trial_family_id=resolved_family,
                data_fingerprint=data_fp,
                rsi_level=rsi_level,
                round_num=round_num,
                signal_count=evaluation.overall.signal_count,
                sharpe_ratio=evaluation.overall.sharpe_ratio,
                returns=dsr_input,
                confidence_score=evaluation.confidence_score,
                passed_null_control=evaluation.null_control_passed,
                accepted=accepted,
                notes=notes,
            )

        veto_reasons: list[str] = []
        if evaluation.overall.signal_count < min_signals:
            veto_reasons.append(f"signals<{min_signals} ({evaluation.overall.signal_count})")
        if evaluation.overall.avg_return <= 0.0:
            veto_reasons.append(f"avg_return<=0 ({evaluation.overall.avg_return:.4%})")
        if max_drawdown is not None and evaluation.overall.max_drawdown > max_drawdown:
            veto_reasons.append(
                f"max_drawdown>{max_drawdown:.1%} ({evaluation.overall.max_drawdown:.1%})"
            )
        if evaluation.anti_overfit.is_overfit:
            veto_reasons.append("anti_overfit_flagged")
        dsr_prob = float(evaluation.deflated_sharpe_prob or 0.0)
        if min_dsr_prob > 0.0 and dsr_prob < min_dsr_prob:
            veto_reasons.append(f"dsr<{min_dsr_prob:.1%} ({dsr_prob:.1%})")
        if run_null_control and evaluation.null_control_passed is not True:
            veto_reasons.append("null_control_failed")
        if run_strategy_rerun_null and (
            strategy_rerun_null is None or not strategy_rerun_null.passed
        ):
            veto_reasons.append("strategy_rerun_null_failed")
        if portfolio_verdict is not None and not portfolio_verdict.passed:
            veto_reasons.extend(f"portfolio:{r}" for r in portfolio_verdict.reasons)

        return KernelEvaluationOutcome(
            strategy_id=strategy.meta.id,
            family_id=resolved_family,
            data_fingerprint=data_fp,
            backtest_result=bt_result,
            evaluation=evaluation,
            dsr_result=dsr_result,
            null_control=null_control,
            strategy_rerun_null=strategy_rerun_null,
            portfolio_verdict=portfolio_verdict,
            trial_record=trial_record,
            veto_reasons=veto_reasons,
            passed_all_gates=len(veto_reasons) == 0,
        )


__all__ = [
    "EvaluationKernel",
    "KernelEvaluationOutcome",
    "LockedHoldoutSplit",
    "LockedHoldoutVault",
    "PortfolioGateConfig",
    "PortfolioGateVerdict",
    "split_market_data_with_holdout",
]
