"""Statistical validation of factor candidates.

Checks IC (Information Coefficient), IR, monthly win-rate, turnover,
and cross-factor correlation to filter out weak or redundant factors.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Outcome of validating one factor."""

    factor_name: str
    passed: bool
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0  # IC mean / IC std
    monthly_win_rate: float = 0.0  # % of months with IC > 0
    turnover: float = 0.0  # avg rank change between periods
    reasons: list[str] = Field(default_factory=list)


class ValidationThresholds(BaseModel):
    """Configurable thresholds for factor validation."""

    min_ic_abs: float = 0.02
    min_ir: float = 0.3
    min_monthly_win_rate: float = 0.5
    max_turnover: float = 0.8
    max_cross_correlation: float = 0.7
    min_cross_section_size: int = 4


class FactorValidator:
    """Validate factor quality using statistical tests.

    Example::

        validator = FactorValidator()
        result = validator.validate(
            factor_values=series_of_factor_scores,
            forward_returns=series_of_next_day_returns,
        )
        if result.passed:
            print("Factor is valid!")
    """

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self,
        factor_name: str,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        *,
        expected_direction: Literal["positive", "negative"] = "positive",
        dates: pd.Series | None = None,
        symbols: pd.Series | None = None,
        existing_factors: dict[str, pd.Series] | None = None,
    ) -> ValidationResult:
        """Run all validation checks on a factor.

        Args:
            factor_name: Identifier for the factor.
            factor_values: Factor scores (aligned with forward_returns).
            forward_returns: Next-period returns for each observation.
            dates: Optional date series for monthly IC bucketing.
            existing_factors: Other factors to check cross-correlation against.
        """
        reasons: list[str] = []

        # 1. Compute rank IC (Spearman correlation)
        ic_mean, ic_std = self._compute_ic(
            factor_values,
            forward_returns,
            dates=dates,
            symbols=symbols,
        )
        ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

        expected_sign = 1 if expected_direction == "positive" else -1
        directed_ic = ic_mean * expected_sign
        directed_ir = ir * expected_sign

        if directed_ic < self.thresholds.min_ic_abs:
            if ic_mean == 0.0 or np.sign(ic_mean) != expected_sign:
                reasons.append(
                    f"IC mean {ic_mean:.4f} contradicts expected {expected_direction} direction"
                )
            else:
                reasons.append(
                    f"Directional IC mean {directed_ic:.4f} < {self.thresholds.min_ic_abs}"
                )

        if directed_ir < self.thresholds.min_ir:
            if ir == 0.0 or np.sign(ir) != expected_sign:
                reasons.append(f"IR {ir:.4f} contradicts expected {expected_direction} direction")
            else:
                reasons.append(f"Directional IR {directed_ir:.4f} < {self.thresholds.min_ir}")

        # 2. Monthly win rate
        monthly_wr = self._monthly_win_rate(
            factor_values,
            forward_returns,
            dates=dates,
            symbols=symbols,
            expected_direction=expected_direction,
        )
        if monthly_wr < self.thresholds.min_monthly_win_rate:
            reasons.append(
                f"Monthly win rate {monthly_wr:.2%} < {self.thresholds.min_monthly_win_rate:.2%}"
            )

        # 3. Turnover
        turnover = self._compute_turnover(
            factor_values,
            dates=dates,
            symbols=symbols,
        )
        if turnover > self.thresholds.max_turnover:
            reasons.append(f"Turnover {turnover:.4f} > {self.thresholds.max_turnover}")

        # 4. Cross-factor correlation
        if existing_factors:
            for other_name, other_vals in existing_factors.items():
                corr = self._rank_correlation(factor_values, other_vals)
                if abs(corr) > self.thresholds.max_cross_correlation:
                    reasons.append(
                        f"Correlated with {other_name}: {corr:.4f} > "
                        f"{self.thresholds.max_cross_correlation}"
                    )

        return ValidationResult(
            factor_name=factor_name,
            passed=len(reasons) == 0,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            monthly_win_rate=monthly_wr,
            turnover=turnover,
            reasons=reasons,
        )

    def _compute_ic(
        self,
        factor: pd.Series,
        returns: pd.Series,
        *,
        dates: pd.Series | None = None,
        symbols: pd.Series | None = None,
    ) -> tuple[float, float]:
        """Compute rank IC (Spearman correlation) between factor and returns."""
        combined = self._build_frame(
            factor,
            returns=returns,
            dates=dates,
            symbols=symbols,
        ).dropna(subset=["f", "r"])
        if len(combined) < 10:
            return 0.0, 1.0

        if dates is not None and symbols is not None:
            cross_sectional_ics = self._cross_sectional_ic_series(combined)
            if len(cross_sectional_ics) >= 3:
                ic_mean = float(cross_sectional_ics.mean())
                ic_std = float(cross_sectional_ics.std(ddof=0))
                return ic_mean, ic_std if ic_std > 1e-10 else 1.0

        # Rank-based correlation
        rank_f = combined["f"].rank()
        rank_r = combined["r"].rank()
        corr = rank_f.corr(rank_r)
        if np.isnan(corr):
            return 0.0, 1.0

        # For rolling IC, split into chunks for std estimation
        chunk_size = max(len(combined) // 5, 10)
        ics = []
        for start in range(0, len(combined) - chunk_size + 1, chunk_size):
            chunk = combined.iloc[start : start + chunk_size]
            rf = chunk["f"].rank()
            rr = chunk["r"].rank()
            c = rf.corr(rr)
            if not np.isnan(c):
                ics.append(c)

        ic_std = float(np.std(ics)) if len(ics) > 1 else 1.0
        return float(corr), ic_std

    def _monthly_win_rate(
        self,
        factor: pd.Series,
        returns: pd.Series,
        dates: pd.Series | None,
        symbols: pd.Series | None = None,
        expected_direction: Literal["positive", "negative"] = "positive",
    ) -> float:
        """Fraction of months where IC matches the expected direction."""
        combined = self._build_frame(
            factor,
            returns=returns,
            dates=dates,
            symbols=symbols,
        ).dropna(subset=["f", "r"])
        if len(combined) < 20:
            return 0.0

        expected_sign = 1 if expected_direction == "positive" else -1

        if dates is not None and symbols is not None:
            daily_ics = self._cross_sectional_ic_series(combined)
            if len(daily_ics) == 0:
                return 0.0
            month_scores = daily_ics.groupby(
                pd.DatetimeIndex(daily_ics.index).to_period("M")
            ).mean()
            return (
                float((month_scores * expected_sign > 0).mean()) if len(month_scores) > 0 else 0.0
            )

        if dates is not None:
            combined["month"] = pd.to_datetime(combined["date"]).dt.to_period("M")
        else:
            # Use positional bucketing
            n = len(combined)
            bucket_size = max(n // 12, 5)
            combined["month"] = [i // bucket_size for i in range(n)]

        wins = 0
        total = 0
        for _, group in combined.groupby("month"):
            if len(group) < 5:
                continue
            rf = group["f"].rank()
            rr = group["r"].rank()
            ic = rf.corr(rr)
            if not np.isnan(ic):
                total += 1
                if ic * expected_sign > 0:
                    wins += 1

        return wins / total if total > 0 else 0.0

    def _compute_turnover(
        self,
        factor: pd.Series,
        *,
        dates: pd.Series | None = None,
        symbols: pd.Series | None = None,
    ) -> float:
        """Average rank change between consecutive observations.

        Returns value in [0, 1] where 0 = no rank change, 1 = complete shuffle.
        """
        if dates is not None and symbols is not None:
            combined = self._build_frame(
                factor,
                dates=dates,
                symbols=symbols,
            ).dropna(subset=["f"])
            if len(combined) < 10:
                return 0.0

            pivot = combined.pivot_table(
                index="date",
                columns="symbol",
                values="f",
                aggfunc="last",
            ).sort_index()
            if len(pivot) < 2:
                return 0.0

            turnovers: list[float] = []
            prev_ranks: pd.Series | None = None
            for _, row in pivot.iterrows():
                current = row.dropna()
                if len(current) < self.thresholds.min_cross_section_size:
                    continue
                current_ranks = current.rank()
                if prev_ranks is not None:
                    common = prev_ranks.index.intersection(current_ranks.index)
                    if len(common) >= self.thresholds.min_cross_section_size:
                        diff = prev_ranks.loc[common].to_numpy(dtype=float) - current_ranks.loc[
                            common
                        ].to_numpy(dtype=float)
                        turnovers.append(float(np.mean(np.abs(diff)) / len(common)))
                prev_ranks = current_ranks

            return float(np.mean(turnovers)) if turnovers else 0.0

        vals = factor.dropna()
        if len(vals) < 10:
            return 0.0

        # Split into two halves and compare ranks
        mid = len(vals) // 2
        rank1 = vals.iloc[:mid].rank()
        rank2 = vals.iloc[mid : mid + len(rank1)].rank()

        if len(rank1) != len(rank2):
            rank2 = rank2.iloc[: len(rank1)]

        if len(rank1) == 0:
            return 0.0

        # Normalize rank diff
        max_diff = len(rank1)
        diff = rank1.to_numpy(dtype=float) - rank2.to_numpy(dtype=float)
        return float(np.mean(np.abs(diff)) / max_diff)

    @staticmethod
    def _rank_correlation(s1: pd.Series, s2: pd.Series) -> float:
        """Spearman rank correlation between two series."""
        combined = pd.DataFrame({"a": s1, "b": s2}).dropna()
        if len(combined) < 10:
            return 0.0
        corr = combined["a"].rank().corr(combined["b"].rank())
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def _build_frame(
        factor: pd.Series,
        *,
        returns: pd.Series | None = None,
        dates: pd.Series | None = None,
        symbols: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Align validation inputs into a single dataframe."""
        payload: dict[str, pd.Series] = {"f": factor}
        if returns is not None:
            payload["r"] = returns
        if dates is not None:
            payload["date"] = pd.Series(pd.to_datetime(dates), index=factor.index)
        if symbols is not None:
            payload["symbol"] = pd.Series(symbols, index=factor.index)
        return cast("pd.DataFrame", pd.DataFrame(payload))

    def _cross_sectional_ic_series(self, combined: pd.DataFrame) -> pd.Series:
        """Compute one rank IC per date from cross-sectional slices."""
        if "date" not in combined.columns or "symbol" not in combined.columns:
            return pd.Series(dtype=float)

        ic_by_date: dict[pd.Timestamp, float] = {}
        for dt, group in combined.groupby("date"):
            clean = group.dropna(subset=["f", "r"])
            if clean["symbol"].nunique() < self.thresholds.min_cross_section_size:
                continue
            ic = clean["f"].rank().corr(clean["r"].rank())
            if not np.isnan(ic):
                timestamp = self._coerce_timestamp(dt)
                if timestamp is not None:
                    ic_by_date[timestamp] = float(ic)

        if not ic_by_date:
            return pd.Series(dtype=float)
        return cast("pd.Series", pd.Series(ic_by_date).sort_index())

    @staticmethod
    def _coerce_timestamp(value: object) -> pd.Timestamp | None:
        """Best-effort scalar timestamp coercion for grouped date keys."""
        scalar = cast(
            "str | bytes | int | float | date | datetime | np.datetime64 | pd.Timestamp",
            value,
        )
        try:
            parsed = pd.to_datetime(scalar, errors="coerce")
        except (TypeError, ValueError):
            return None
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            return parsed
        return None
