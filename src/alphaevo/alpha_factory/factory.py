"""Alpha Factory — end-to-end pipeline for factor discovery.

Orchestrates: Synthesizer → Sandbox → Validator → Store → Registry.

Usage::

    factory = AlphaFactory(llm_client, db_path="~/.alphaevo/alphaevo.db")
    results = await factory.discover(
        context="Low win rate due to false breakouts",
        ohlcv_data=df,
        forward_returns=returns_series,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
from alphaevo.alpha_factory.sandbox import FactorSandbox, build_restricted_namespace
from alphaevo.alpha_factory.synthesizer import FactorHypothesis, FactorSynthesizer
from alphaevo.alpha_factory.validator import FactorValidator, ValidationThresholds
from alphaevo.backtest.indicators import IndicatorFn, IndicatorRegistry

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from alphaevo.core.llm import LLMClient
    from alphaevo.models.market import IndicatorContext

logger = logging.getLogger(__name__)


def _build_indicator_fn(
    factor_code: str,
    sandbox: FactorSandbox,
) -> Callable[[pd.DataFrame, int, IndicatorContext | None], float | bool]:
    """Create an indicator function from validated factor code."""

    def _zero_indicator(
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> float:
        del df, idx, ctx
        return 0.0

    is_safe, _ = sandbox.validate_code(factor_code)
    if not is_safe:
        return _zero_indicator

    namespace = build_restricted_namespace()
    try:
        exec(factor_code, namespace)  # noqa: S102
    except Exception:
        return _zero_indicator

    compute_fn = namespace.get("compute")
    if compute_fn is None:
        return _zero_indicator
    compute = cast("Callable[[pd.DataFrame, int], Any]", compute_fn)

    def indicator_fn(
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> float | bool:
        del ctx
        try:
            pit_df = df.iloc[: idx + 1]
            val = float(compute(pit_df, idx))
            import numpy as np

            return val if np.isfinite(val) else 0.0
        except Exception:
            return 0.0

    return indicator_fn


def _execute_factor_series(
    code: str,
    ohlcv_data: pd.DataFrame,
    sandbox: FactorSandbox,
    *,
    verify_pit: bool = True,
) -> tuple[pd.Series | None, str | None]:
    """Execute factor code while respecting symbol boundaries."""
    if "_symbol" not in ohlcv_data.columns:
        sandbox_result = sandbox.execute(code, ohlcv_data, verify_pit=verify_pit)
        if not sandbox_result.success or sandbox_result.values is None:
            return None, sandbox_result.error or "Sandbox execution failed"
        return pd.Series(sandbox_result.values, index=ohlcv_data.index, dtype=float), None

    factor_values = pd.Series(index=ohlcv_data.index, dtype=float)
    for symbol, group in ohlcv_data.groupby("_symbol", sort=False):
        sandbox_result = sandbox.execute(code, group, verify_pit=verify_pit)
        if not sandbox_result.success or sandbox_result.values is None:
            err = sandbox_result.error or "Sandbox execution failed"
            return None, f"{symbol}: {err}"
        if len(sandbox_result.values) != len(group):
            return None, f"{symbol}: execution length mismatch"
        factor_values.loc[group.index] = pd.Series(
            sandbox_result.values,
            index=group.index,
            dtype=float,
        )

    return factor_values.reindex(ohlcv_data.index), None


def register_factor_record(
    record: FactorRecord,
    *,
    sandbox: FactorSandbox | None = None,
) -> None:
    """Register a persisted factor record into the dynamic indicator registry."""
    effective_sandbox = sandbox or FactorSandbox()
    IndicatorRegistry.register_dynamic(
        record.name,
        cast("IndicatorFn", _build_indicator_fn(record.code, effective_sandbox)),
    )


def load_registered_factors(
    db_path: str | Path,
    *,
    status: str = "active",
) -> list[str]:
    """Load persisted factors from storage into the dynamic registry."""
    store = FactorStore(db_path)
    try:
        records = store.list_all(status=None if status == "all" else status)
        loaded: list[str] = []
        sandbox = FactorSandbox()
        for record in records:
            if status != "all" and record.status != status:
                continue
            register_factor_record(record, sandbox=sandbox)
            loaded.append(record.name)
        return loaded
    finally:
        store.close()


class DiscoveryResult:
    """Summary of one discovery run."""

    def __init__(self) -> None:
        self.proposed: list[FactorHypothesis] = []
        self.sandbox_passed: list[str] = []
        self.sandbox_failed: list[tuple[str, str]] = []  # (name, error)
        self.validation_passed: list[str] = []
        self.validation_failed: list[tuple[str, list[str]]] = []  # (name, reasons)
        self.registered: list[str] = []

    @property
    def success_count(self) -> int:
        return len(self.registered)

    def summary(self) -> str:
        lines = [
            f"Proposed: {len(self.proposed)}",
            f"Sandbox passed: {len(self.sandbox_passed)}",
            f"Sandbox failed: {len(self.sandbox_failed)}",
            f"Validation passed: {len(self.validation_passed)}",
            f"Validation failed: {len(self.validation_failed)}",
            f"Registered: {len(self.registered)}",
        ]
        return " | ".join(lines)


class AlphaFactory:
    """End-to-end factor discovery pipeline."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        db_path: str | Path = ":memory:",
        sandbox_timeout: int = 30,
        validation_thresholds: ValidationThresholds | None = None,
        max_refine_attempts: int = 2,
        verify_pit: bool = True,
    ) -> None:
        self._synthesizer = FactorSynthesizer(llm_client)
        self._sandbox = FactorSandbox(timeout_seconds=sandbox_timeout)
        self._validator = FactorValidator(validation_thresholds)
        self._store = FactorStore(db_path)
        self._max_refine = max_refine_attempts
        self._verify_pit = verify_pit

    async def discover(
        self,
        context: str,
        ohlcv_data: pd.DataFrame,
        forward_returns: pd.Series,
        *,
        max_candidates: int = 3,
        dates: pd.Series | None = None,
        register: bool = True,
    ) -> DiscoveryResult:
        """Run the full discovery pipeline.

        1. Ask LLM to synthesize factor candidates
        2. Validate each through the sandbox
        3. Statistically validate surviving factors
        4. Store and register successful factors
        """
        result = DiscoveryResult()

        # Existing factors for dedup
        existing_names = IndicatorRegistry.available()
        existing_factor_values: dict[str, pd.Series] = {}

        # Phase 1: Synthesize
        hypotheses = await self._synthesizer.generate(
            context=context,
            existing_factors=existing_names,
            max_candidates=max_candidates,
        )
        result.proposed = hypotheses
        if not hypotheses:
            logger.warning("No factor hypotheses generated")
            return result

        # Phase 2-4: For each candidate
        for hyp in hypotheses:
            await self._process_candidate(
                hyp,
                ohlcv_data,
                forward_returns,
                dates,
                existing_factor_values,
                result,
                register,
            )

        logger.info("Alpha Factory discovery: %s", result.summary())
        return result

    async def _process_candidate(
        self,
        hyp: FactorHypothesis,
        ohlcv_data: pd.DataFrame,
        forward_returns: pd.Series,
        dates: pd.Series | None,
        existing_factor_values: dict[str, pd.Series],
        result: DiscoveryResult,
        register: bool,
    ) -> None:
        """Process a single factor candidate through sandbox → validation → store."""
        # Phase 2: Sandbox execution (with retry)
        factor_series, sandbox_error = _execute_factor_series(
            hyp.code, ohlcv_data, self._sandbox, verify_pit=self._verify_pit
        )

        if factor_series is None:
            # Try refining
            for _attempt in range(self._max_refine):
                refined = await self._synthesizer.refine(hyp, sandbox_error or "Unknown error")
                if refined is None:
                    break
                hyp = refined
                factor_series, sandbox_error = _execute_factor_series(
                    hyp.code,
                    ohlcv_data,
                    self._sandbox,
                    verify_pit=self._verify_pit,
                )
                if factor_series is not None:
                    break

        if factor_series is None:
            result.sandbox_failed.append((hyp.name, sandbox_error or ""))
            return

        result.sandbox_passed.append(hyp.name)

        # Phase 3: Statistical validation
        vr = self._validator.validate(
            factor_name=hyp.name,
            factor_values=factor_series,
            forward_returns=forward_returns,
            expected_direction=hyp.expected_direction,
            dates=dates,
            symbols=ohlcv_data["_symbol"] if "_symbol" in ohlcv_data.columns else None,
            existing_factors=existing_factor_values if existing_factor_values else None,
        )

        if not vr.passed:
            result.validation_failed.append((hyp.name, vr.reasons))
            return

        result.validation_passed.append(hyp.name)

        # Phase 4: Store and register
        record = FactorRecord(
            name=hyp.name,
            description=hyp.description,
            rationale=hyp.rationale,
            code=hyp.code,
            expected_direction=hyp.expected_direction,
            ic_mean=vr.ic_mean,
            ic_std=vr.ic_std,
            ir=vr.ir,
            monthly_win_rate=vr.monthly_win_rate,
            turnover=vr.turnover,
        )
        self._store.save(record)

        if register:
            register_factor_record(record, sandbox=self._sandbox)
            result.registered.append(hyp.name)

        # Track for cross-correlation checks
        existing_factor_values[hyp.name] = factor_series

    def _register_factor(self, hyp: FactorHypothesis) -> None:
        """Register factor into IndicatorRegistry for use in strategies."""
        IndicatorRegistry.register_dynamic(
            hyp.name,
            cast("IndicatorFn", _build_indicator_fn(hyp.code, self._sandbox)),
        )

    @property
    def store(self) -> FactorStore:
        """Access the factor store directly."""
        return self._store

    def close(self) -> None:
        self._store.close()
