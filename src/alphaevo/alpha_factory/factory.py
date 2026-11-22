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

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
from alphaevo.alpha_factory.sandbox import FactorSandbox
from alphaevo.alpha_factory.synthesizer import FactorHypothesis, FactorSynthesizer
from alphaevo.alpha_factory.validator import FactorValidator, ValidationThresholds
from alphaevo.backtest.indicators import IndicatorFn, IndicatorRegistry

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from alphaevo.core.llm import LLMClient
    from alphaevo.models.market import IndicatorContext

logger = logging.getLogger(__name__)

_INDICATOR_CACHE_MAX_ENTRIES = 8


def _object_type_digest(values: list[Any]) -> bytes:
    """Hash per-element Python type identity for object-dtype columns/indexes."""
    hasher = hashlib.blake2b(digest_size=16)
    for val in values:
        cls = type(val)
        hasher.update(f"{cls.__module__}.{cls.__qualname__}\x00".encode())
    return hasher.digest()


def _df_cache_key(df: pd.DataFrame) -> tuple[Any, ...] | None:
    """Build a dtype-, category-, object-type-, column-, index-, and order-sensitive cache key."""
    try:
        col_dtypes = tuple(str(dt) for dt in df.dtypes)
        idx_dtype = str(df.index.dtype)
        col_cats = tuple(
            (
                col,
                str(df[col].cat.categories.dtype),
                tuple(df[col].cat.categories.tolist()),
                bool(df[col].cat.ordered),
            )
            for col in df.columns
            if isinstance(df[col].dtype, pd.CategoricalDtype)
        )
        idx_cats = (
            (
                str(cast("Any", df.index).categories.dtype),
                tuple(cast("Any", df.index).categories.tolist()),
                bool(cast("Any", df.index).ordered),
            )
            if isinstance(df.index.dtype, pd.CategoricalDtype)
            else ()
        )
        obj_col_sigs = tuple(
            (col, _object_type_digest(df[col].tolist()))
            for col in df.columns
            if df[col].dtype == object
        )
        if isinstance(df.index, pd.MultiIndex):
            obj_idx_sig: tuple[Any, ...] = tuple(
                (
                    str(df.index.get_level_values(i).dtype),
                    (
                        _object_type_digest(df.index.get_level_values(i).tolist())
                        if df.index.get_level_values(i).dtype == object
                        else b""
                    ),
                )
                for i in range(df.index.nlevels)
            )
        elif df.index.dtype == object:
            obj_idx_sig = (_object_type_digest(df.index.tolist()),)
        else:
            obj_idx_sig = ()
        if df.empty:
            return (
                tuple(df.columns),
                col_dtypes,
                idx_dtype,
                col_cats,
                idx_cats,
                obj_col_sigs,
                obj_idx_sig,
                (0, 0),
                b"",
            )
        hashed = pd.util.hash_pandas_object(df, index=True)
        digest = hashlib.blake2b(
            hashed.to_numpy(dtype=np.uint64).tobytes(), digest_size=16
        ).digest()
        return (
            tuple(df.columns),
            col_dtypes,
            idx_dtype,
            col_cats,
            idx_cats,
            obj_col_sigs,
            obj_idx_sig,
            df.shape,
            digest,
        )
    except Exception:
        return None


def _build_indicator_fn(
    factor_code: str,
    sandbox: FactorSandbox,
) -> Callable[[pd.DataFrame, int, IndicatorContext | None], float | bool]:
    """Create an isolated worker-backed indicator function from validated factor code."""

    def _zero_indicator(
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> float:
        del df, idx, ctx
        return 0.0

    is_safe, _ = sandbox.preflight_code(factor_code)
    if not is_safe:
        return _zero_indicator

    series_cache: OrderedDict[tuple[Any, ...], list[float]] = OrderedDict()
    cache_lock = threading.Lock()

    def indicator_fn(
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> float | bool:
        del ctx
        if idx < 0 or idx >= len(df):
            return 0.0

        cache_key = _df_cache_key(df)
        if cache_key is not None:
            with cache_lock:
                cached_values = series_cache.get(cache_key)
                if cached_values is not None:
                    series_cache.move_to_end(cache_key)
                    if 0 <= idx < len(cached_values):
                        val = float(cached_values[idx])
                        return val if np.isfinite(val) else 0.0

        result = sandbox.execute(
            factor_code,
            df,
            verify_pit=False,
            enforce_pit_slice=True,
        )
        if not result.success or result.values is None or len(result.values) != len(df):
            return 0.0

        computed = [float(v) if np.isfinite(float(v)) else 0.0 for v in result.values]
        if cache_key is not None:
            with cache_lock:
                series_cache[cache_key] = computed
                series_cache.move_to_end(cache_key)
                while len(series_cache) > _INDICATOR_CACHE_MAX_ENTRIES:
                    series_cache.popitem(last=False)

        return computed[idx]

    cast("Any", indicator_fn)._alphaevo_factor_code_hash = hashlib.sha256(
        factor_code.encode("utf-8")
    ).hexdigest()
    return indicator_fn


def _execute_factor_series(
    code: str,
    ohlcv_data: pd.DataFrame,
    sandbox: FactorSandbox,
    *,
    verify_pit: bool = True,
    enforce_pit_slice: bool = False,
) -> tuple[pd.Series | None, str | None]:
    """Execute factor code while respecting symbol boundaries."""
    if "_symbol" not in ohlcv_data.columns:
        sandbox_result = sandbox.execute(
            code,
            ohlcv_data,
            verify_pit=verify_pit,
            enforce_pit_slice=enforce_pit_slice,
        )
        if not sandbox_result.success or sandbox_result.values is None:
            return None, sandbox_result.error or "Sandbox execution failed"
        return pd.Series(sandbox_result.values, index=ohlcv_data.index, dtype=float), None

    factor_values = pd.Series(index=ohlcv_data.index, dtype=float)
    for symbol, group in ohlcv_data.groupby("_symbol", sort=False):
        sandbox_result = sandbox.execute(
            code,
            group,
            verify_pit=verify_pit,
            enforce_pit_slice=enforce_pit_slice,
        )
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
        """Close backing store and sandbox worker process."""
        self._store.close()
        self._sandbox.close()

    def __enter__(self) -> AlphaFactory:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
