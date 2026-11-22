"""L2 Factor Co-STEER blind-spot routing and L3 Executable Pinned Scars."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
from alphaevo.alpha_factory.factory import _build_indicator_fn, register_factor_record
from alphaevo.alpha_factory.sandbox import FactorSandbox
from alphaevo.backtest.indicators import IndicatorFn, IndicatorRegistry
from alphaevo.models.enums import ChangeType, EvidenceLevel, ExitReason, RSILevel
from alphaevo.models.execution import EvaluationReport, StrategyChange
from alphaevo.models.strategy import Strategy

_DYNAMIC_LEASE_LOCK = threading.RLock()
_DYNAMIC_LEASES: dict[str, int] = {}
_TRANSIENT_FACTORS: set[str] = set()


def _uses_indicator(strategy: Strategy, indicator_name: str) -> bool:
    conds = (
        strategy.entry.triggers
        + strategy.entry.conditions
        + strategy.entry.guards
        + strategy.entry.filters
        + strategy.exit.triggers
    )
    stop_conds = getattr(strategy.exit.stop_loss, "conditions", None) or []
    return any(getattr(cond, "indicator", None) == indicator_name for cond in [*conds, *stop_conds])


def acquire_dynamic_lease(
    factor_name: str | None,
    *,
    in_pre_registered: bool,
    factor_record: Any | None,
    factor_store: FactorStore,
) -> str | None:
    """Acquire a refcounted lease for a transient L2 dynamic factor across concurrent runs."""
    if not factor_name:
        return None
    with _DYNAMIC_LEASE_LOCK:
        is_transient = factor_name in _TRANSIENT_FACTORS
        if not in_pre_registered and not is_transient and factor_store.get(factor_name) is None:
            _TRANSIENT_FACTORS.add(factor_name)
            is_transient = True
        if not is_transient:
            return None
        if not IndicatorRegistry.is_registered(factor_name) and factor_record is not None:
            register_factor_record(factor_record)
        _DYNAMIC_LEASES[factor_name] = _DYNAMIC_LEASES.get(factor_name, 0) + 1
        return factor_name


def release_dynamic_lease(
    factor_name: str | None,
    *,
    persisted: bool,
    protected_strategies: tuple[Strategy, ...],
) -> None:
    """Release one dynamic factor lease and unregister only when zero transient leases remain."""
    if not factor_name:
        return
    with _DYNAMIC_LEASE_LOCK:
        if persisted:
            _TRANSIENT_FACTORS.discard(factor_name)
        rem = _DYNAMIC_LEASES.get(factor_name, 1) - 1
        if rem > 0:
            _DYNAMIC_LEASES[factor_name] = rem
            return
        _DYNAMIC_LEASES.pop(factor_name, None)
        if persisted or factor_name not in _TRANSIENT_FACTORS:
            return
        _TRANSIENT_FACTORS.discard(factor_name)
        if any(_uses_indicator(s, factor_name) for s in protected_strategies):
            return
        IndicatorRegistry.unregister_dynamic(factor_name)


FailureClusterId = Literal[
    "zero_signal_lockout",
    "whipsaw_stopout_in_chop",
    "weak_payoff_tail",
    "missing_discriminatory_factor",
    "healthy_refinement",
]


class BlindSpotCluster(BaseModel):
    """Diagnosed failure cluster motivating L1 mutation or L2 factor discovery."""

    cluster_id: FailureClusterId
    severity: Literal["high", "medium", "low"]
    evidence: str
    suggested_action: str
    bottleneck_hint: str = ""


class CoSteerDecision(BaseModel):
    """Routing decision between L1 Strategy Evolution and L2 Factor Discovery."""

    chosen_level: RSILevel
    cluster: BlindSpotCluster
    synthesized_factor_name: str | None = None
    synthesized_via: Literal["llm", "offline_template"] | None = None
    pit_verified: bool = False
    persisted_to_store: bool = False
    lease_acquired: bool = False
    synthesized_factor_record: FactorRecord | None = None
    proposed_changes: list[StrategyChange] = Field(default_factory=list)


class OfflineFactorTemplate(BaseModel):
    """Point-in-time pure-pandas/numpy factor template for offline L2 Co-STEER."""

    name: str
    description: str
    op: Literal["<", "<=", ">", ">="]
    threshold: float
    expected_direction: Literal["positive", "negative"] = "positive"
    code: str


_OFFLINE_FACTOR_TEMPLATES: tuple[OfflineFactorTemplate, ...] = (
    OfflineFactorTemplate(
        name="intraday_range_compression_10d",
        description="10-day relative range compression ratio (smaller = calmer base before rebound)",
        op="<=",
        threshold=1.15,
        expected_direction="negative",
        code="""
import numpy as np

def compute(df, idx):
    if idx < 10:
        return 1.0
    sub = df.iloc[idx - 9 : idx + 1]
    spreads = (sub["high"].astype(float) - sub["low"].astype(float)) / np.maximum(1e-6, sub["close"].astype(float))
    avg_spread = float(spreads.mean())
    curr_spread = float(spreads.iloc[-1])
    if avg_spread <= 1e-8:
        return 1.0
    return curr_spread / avg_spread
""".strip(),
    ),
    OfflineFactorTemplate(
        name="pullback_efficiency_5d",
        description="5-day directional efficiency ratio isolating orderly pullbacks vs panic cascades",
        op=">=",
        threshold=-0.45,
        code="""
import numpy as np

def compute(df, idx):
    if idx < 6:
        return 0.0
    closes = df["close"].iloc[idx - 5 : idx + 1].astype(float).tolist()
    net = closes[-1] - closes[0]
    path = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    if path <= 1e-8:
        return 0.0
    return net / path
""".strip(),
    ),
    OfflineFactorTemplate(
        name="volume_impulse_z5",
        description="Short-horizon standardized volume impulse relative to recent 10-day window",
        op=">=",
        threshold=-0.5,
        code="""
import numpy as np

def compute(df, idx):
    if idx < 10:
        return 0.0
    vols = df["volume"].iloc[idx - 9 : idx + 1].astype(float)
    std = float(vols.std(ddof=0))
    if std <= 1e-8:
        return 0.0
    return (float(vols.iloc[-1]) - float(vols.mean())) / std
""".strip(),
    ),
    OfflineFactorTemplate(
        name="trend_alignment_slope_20d",
        description="20-day normalized moving-average slope filtering counter-trend knife catches",
        op=">=",
        threshold=-0.01,
        code="""
import numpy as np

def compute(df, idx):
    if idx < 19:
        return 0.0
    closes = df["close"].iloc[idx - 19 : idx + 1].astype(float)
    ma_now = float(closes.mean())
    half_prev = float(closes.iloc[:10].mean())
    half_curr = float(closes.iloc[10:].mean())
    if ma_now <= 1e-8:
        return 0.0
    return (half_curr - half_prev) / ma_now
""".strip(),
    ),
    OfflineFactorTemplate(
        name="downside_vol_ratio_10d",
        description="10-day downside vs total return volatility ratio filtering asymmetric panic bars",
        op="<=",
        threshold=0.78,
        expected_direction="negative",
        code="""
import numpy as np

def compute(df, idx):
    if idx < 10:
        return 0.5
    closes = df["close"].iloc[idx - 10 : idx + 1].astype(float).to_numpy()
    rets = np.diff(closes) / np.maximum(1e-6, closes[:-1])
    total_energy = float(np.sum(np.square(rets)))
    if total_energy <= 1e-12:
        return 0.5
    down_energy = float(np.sum(np.square(np.minimum(0.0, rets))))
    return float(np.sqrt(down_energy / total_energy))
""".strip(),
    ),
    OfflineFactorTemplate(
        name="close_location_value_5d",
        description="5-day average close location value measuring close-near-high accumulation strength",
        op=">=",
        threshold=-0.15,
        code="""
import numpy as np

def compute(df, idx):
    if idx < 5:
        return 0.0
    sub = df.iloc[idx - 4 : idx + 1]
    high = sub["high"].astype(float).to_numpy()
    low = sub["low"].astype(float).to_numpy()
    close = sub["close"].astype(float).to_numpy()
    span = np.maximum(1e-6, high - low)
    clv = (2.0 * close - high - low) / span
    return float(np.mean(clv))
""".strip(),
    ),
    OfflineFactorTemplate(
        name="atr_expansion_ratio_14d",
        description="5-day vs 14-day true-range expansion ratio filtering parabolic exhaustion spikes",
        op="<=",
        threshold=1.35,
        expected_direction="negative",
        code="""
import numpy as np

def compute(df, idx):
    if idx < 14:
        return 1.0
    sub = df.iloc[idx - 14 : idx + 1]
    high = sub["high"].astype(float).to_numpy()
    low = sub["low"].astype(float).to_numpy()
    close = sub["close"].astype(float).to_numpy()
    prev_c = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_c), np.abs(low[1:] - prev_c)),
    )
    atr14 = float(np.mean(tr[-14:]))
    if atr14 <= 1e-8:
        return 1.0
    atr5 = float(np.mean(tr[-5:]))
    return atr5 / atr14
""".strip(),
    ),
)


def _select_ordered_templates(cluster_id: FailureClusterId) -> list[OfflineFactorTemplate]:
    """Order PIT-safe L2 factor templates to target the diagnosed failure cluster."""
    by_name = {t.name: t for t in _OFFLINE_FACTOR_TEMPLATES}
    priority_map: dict[FailureClusterId, tuple[str, ...]] = {
        "whipsaw_stopout_in_chop": (
            "intraday_range_compression_10d",
            "downside_vol_ratio_10d",
            "atr_expansion_ratio_14d",
            "pullback_efficiency_5d",
        ),
        "weak_payoff_tail": (
            "trend_alignment_slope_20d",
            "close_location_value_5d",
            "volume_impulse_z5",
            "intraday_range_compression_10d",
        ),
        "missing_discriminatory_factor": (
            "pullback_efficiency_5d",
            "close_location_value_5d",
            "trend_alignment_slope_20d",
            "atr_expansion_ratio_14d",
        ),
        "zero_signal_lockout": ("pullback_efficiency_5d",),
        "healthy_refinement": ("close_location_value_5d", "trend_alignment_slope_20d"),
    }
    ordered: list[OfflineFactorTemplate] = []
    seen: set[str] = set()
    for name in priority_map.get(cluster_id, ()):
        tmpl = by_name.get(name)
        if tmpl is not None and name not in seen:
            ordered.append(tmpl)
            seen.add(name)
    for tmpl in _OFFLINE_FACTOR_TEMPLATES:
        if tmpl.name not in seen:
            ordered.append(tmpl)
            seen.add(tmpl.name)
    return ordered


class ExecutableScar(BaseModel):
    """Machine-checkable guardrail distilled from repeated or severe RSI failures."""

    scar_id: str
    strategy_family: str
    failure_mechanism: str
    forbidden_change_type: ChangeType | None = None
    target_pattern: str = ""
    op: Literal["forbid_target", "min_value", "max_value"] = "forbid_target"
    threshold: float | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.BACKTEST
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def matches(self, strategy_family: str, change: StrategyChange) -> bool:
        """Return True when ``change`` triggers this scar's deterministic veto."""
        if self.strategy_family not in {"*", strategy_family}:
            return False
        if (
            self.forbidden_change_type is not None
            and change.change_type != self.forbidden_change_type
        ):
            return False
        if self.target_pattern and self.target_pattern not in change.target:
            return False
        if self.op == "forbid_target":
            return True
        if self.threshold is None or not isinstance(change.to_value, (int, float)):
            return False
        val = float(change.to_value)
        if self.op == "min_value":
            return val < self.threshold
        if self.op == "max_value":
            return val > self.threshold
        return False


class ScarRegistry:
    """Persists and matches ExecutableScars across Meta-RSI runs."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._scars: list[ExecutableScar] = []
        if self._path is not None and self._path.is_file():
            self._load()

    def add(self, scar: ExecutableScar) -> ExecutableScar:
        for existing in self._scars:
            if (
                existing.strategy_family == scar.strategy_family
                and existing.forbidden_change_type == scar.forbidden_change_type
                and existing.target_pattern == scar.target_pattern
                and existing.op == scar.op
            ):
                return existing
        self._scars.append(scar)
        self._save()
        return scar

    def list_all(self, strategy_family: str | None = None) -> list[ExecutableScar]:
        if strategy_family is None:
            return list(self._scars)
        return [s for s in self._scars if s.strategy_family in {"*", strategy_family}]

    def synthesize_from_failure(
        self,
        *,
        strategy: Strategy,
        changes: list[StrategyChange],
        signal_count: int,
        score_delta: float,
        evidence_level: EvidenceLevel = EvidenceLevel.BACKTEST,
    ) -> list[ExecutableScar]:
        """Distill an ExecutableScar when a candidate triggers zero-signal lockout or crash."""
        created: list[ExecutableScar] = []
        if signal_count == 0 and changes:
            for ch in changes:
                scar = ExecutableScar(
                    scar_id=f"scar_{uuid.uuid4().hex[:8]}",
                    strategy_family=strategy.meta.family_id,
                    failure_mechanism=(
                        f"Zero-signal lockout caused by {ch.change_type.value} on {ch.target} -> {ch.to_value}"
                    ),
                    forbidden_change_type=ch.change_type,
                    target_pattern=ch.target,
                    op="forbid_target",
                    evidence_level=evidence_level,
                )
                created.append(self.add(scar))
        elif score_delta < -0.08 and changes:
            ch = changes[0]
            scar = ExecutableScar(
                scar_id=f"scar_{uuid.uuid4().hex[:8]}",
                strategy_family=strategy.meta.family_id,
                failure_mechanism=(
                    f"Severe score collapse ({score_delta:+.1%}) after {ch.change_type.value} on {ch.target}"
                ),
                forbidden_change_type=ch.change_type,
                target_pattern=ch.target,
                op="forbid_target",
                evidence_level=evidence_level,
            )
            created.append(self.add(scar))
        return created

    def _load(self) -> None:
        if self._path is None or not self._path.is_file():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            self._scars = [ExecutableScar.model_validate(item) for item in raw]

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, Any]] = [s.model_dump(mode="json") for s in self._scars]
        self._path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class BlindSpotMiner:
    """Clusters failure trades and steers L1 DSL evolution vs hybrid L2 factor discovery."""

    def __init__(
        self,
        *,
        sandbox: FactorSandbox | None = None,
        factor_store: FactorStore | None = None,
        llm_model: str | None = None,
        api_key: str | None = None,
        synthesizer: Any | None = None,
    ) -> None:
        self._sandbox = sandbox or FactorSandbox(timeout_seconds=10)
        self._owns_sandbox = sandbox is None
        self._factor_store = factor_store
        # Transient LLM synthesizer settings only; never persisted or logged.
        self._llm_model = (llm_model or "").strip() or None
        self._api_key = (api_key or "").strip() or None
        self._synthesizer = synthesizer
        self._pit_cache: dict[str, bool] = {}
        self._failed_llm_factors: set[str] = set()

    def close(self) -> None:
        """Close owned sandbox worker resources if created by this miner."""
        if self._owns_sandbox:
            self._sandbox.close()

    def __enter__(self) -> BlindSpotMiner:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _calibrate_causal_threshold(
        self,
        code: str,
        sample_df: pd.DataFrame,
        expected_direction: Literal["positive", "negative"],
    ) -> tuple[Literal[">=", "<="], float] | None:
        """Calibrate causal factor gate threshold purely from sample distribution."""
        res = self._sandbox.execute(code, sample_df, verify_pit=False)
        if not res.success or not res.values:
            return None
        start_idx = min(20, len(res.values) // 3)
        vals = [float(v) for v in res.values[start_idx:] if math.isfinite(float(v))]
        if len(vals) < 10:
            return None
        arr = np.asarray(vals, dtype=float)
        if float(np.std(arr)) <= 1e-8:
            return None
        if expected_direction == "negative":
            return "<=", round(float(np.quantile(arr, 0.65)), 4)
        return ">=", round(float(np.quantile(arr, 0.35)), 4)

    @staticmethod
    def _run_maybe_async(maybe_coro: Any) -> Any:
        if not inspect.isawaitable(maybe_coro):
            return maybe_coro

        async def _consume(awaitable_obj: Any) -> Any:
            return await awaitable_obj

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_consume(maybe_coro))
        # Running inside an active event loop; execute awaitable in a worker thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _consume(maybe_coro)).result(timeout=45)

    def _try_llm_synthesize(
        self,
        cluster: BlindSpotCluster,
        used_indicators: set[str],
        sample_df: pd.DataFrame,
    ) -> (
        tuple[str, str, str, Literal[">=", "<="], float, Literal["positive", "negative"], str]
        | None
    ):
        """Attempt PIT-verified LLM factor synthesis, returning None on any failure."""
        synth = self._synthesizer
        if synth is None:
            if not self._llm_model or self._llm_model.lower().startswith("heuristic"):
                return None
            try:
                from alphaevo.alpha_factory.synthesizer import FactorSynthesizer
                from alphaevo.core.config import ConfigManager
                from alphaevo.core.llm import LLMClient

                cfg = ConfigManager().load()
                cfg.llm.model = self._llm_model
                eff_key = self._api_key or ConfigManager().get_llm_api_key()
                if not eff_key:
                    return None
                synth = FactorSynthesizer(LLMClient(cfg.llm, api_key=eff_key))
            except Exception:
                return None

        context = (
            f"Failure cluster={cluster.cluster_id}: {cluster.evidence}. "
            f"Bottleneck: {cluster.bottleneck_hint or 'none'}. "
            f"Goal: {cluster.suggested_action}"
        )
        try:
            hypotheses = self._run_maybe_async(
                synth.generate(
                    context=context,
                    existing_factors=sorted(used_indicators),
                    failed_attempts=sorted(self._failed_llm_factors),
                    max_candidates=2,
                )
            )
        except Exception:
            return None

        if not hypotheses:
            return None

        for hyp in hypotheses:
            fname = str(getattr(hyp, "name", "")).strip()
            code = str(getattr(hyp, "code", "")).strip()
            desc = str(getattr(hyp, "description", "")).strip() or fname
            direction_raw = str(getattr(hyp, "expected_direction", "positive")).strip().lower()
            direction: Literal["positive", "negative"] = (
                "negative" if direction_raw == "negative" else "positive"
            )
            if (
                not fname
                or len(fname) > 64
                or not fname.isidentifier()
                or fname in used_indicators
                or fname in self._failed_llm_factors
                or IndicatorRegistry.is_registered(fname)
                or (self._factor_store is not None and self._factor_store.get(fname) is not None)
                or not code
            ):
                if fname:
                    self._failed_llm_factors.add(fname)
                continue

            ok_ast, _ast_err = self._sandbox.validate_code(code)
            if not ok_ast:
                self._failed_llm_factors.add(fname)
                continue

            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if code_hash not in self._pit_cache:
                self._pit_cache[code_hash] = self._sandbox.probe_pit(code, sample_df).passed
            if not self._pit_cache[code_hash]:
                self._failed_llm_factors.add(fname)
                continue

            calibrated = self._calibrate_causal_threshold(code, sample_df, direction)
            if calibrated is None:
                self._failed_llm_factors.add(fname)
                continue

            op_val, thresh_val = calibrated
            return fname, desc, code, op_val, thresh_val, direction, "llm"

        return None

    def diagnose_blind_spot(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        stagnant_l1_rounds: int = 0,
    ) -> BlindSpotCluster:
        """Identify dominant failure mode blocking higher out-of-sample expectancy."""
        ov = evaluation.overall
        if ov.signal_count < 5:
            return BlindSpotCluster(
                cluster_id="zero_signal_lockout",
                severity="high",
                evidence=f"signal_count={ov.signal_count} < 5 with logic='{strategy.entry.logic}'",
                suggested_action="Relax entry.logic to 'or' or loosen restrictive thresholds",
            )

        stop_losses = sum(
            1 for fc in evaluation.failure_cases if fc.exit_reason == ExitReason.STOP_LOSS
        )
        if evaluation.failure_cases and stop_losses >= max(2, len(evaluation.failure_cases) // 2):
            return BlindSpotCluster(
                cluster_id="whipsaw_stopout_in_chop",
                severity="high",
                evidence=(
                    f"{stop_losses}/{len(evaluation.failure_cases)} worst failures hit stop-loss early"
                ),
                suggested_action="Widen stop tolerance or filter noisy entries via L2 range compression factor",
            )

        if ov.profit_loss_ratio < 1.1 and ov.signal_count >= 15:
            return BlindSpotCluster(
                cluster_id="weak_payoff_tail",
                severity="medium",
                evidence=f"P/L ratio={ov.profit_loss_ratio:.2f}, max_hold={strategy.exit.max_holding_days}d",
                suggested_action="Extend holding window or raise take-profit R:R multiple",
            )

        if stagnant_l1_rounds >= 2:
            return BlindSpotCluster(
                cluster_id="missing_discriminatory_factor",
                severity="medium",
                evidence=f"L1 DSL mutations plateaued for {stagnant_l1_rounds} rounds",
                suggested_action="Synthesize and PIT-probe a new L2 orthogonal factor",
            )

        return BlindSpotCluster(
            cluster_id="healthy_refinement",
            severity="low",
            evidence=f"win_rate={ov.win_rate:.1%}, avg_return={ov.avg_return:.2%}",
            suggested_action="Refine entry thresholds and exit profile around incumbent",
        )

    def steer(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        sample_df: pd.DataFrame | None,
        *,
        allowed_levels: list[RSILevel],
        stagnant_l1_rounds: int = 0,
        persist: bool = True,
        acquire_lease: bool = False,
    ) -> CoSteerDecision:
        """Decide L1 vs L2 step and synthesize/PIT-verify an L2 factor when escalating."""
        cluster = self.diagnose_blind_spot(
            strategy,
            evaluation,
            stagnant_l1_rounds=stagnant_l1_rounds,
        )

        used_indicators = {
            c.indicator
            for c in (
                strategy.entry.triggers
                + strategy.entry.conditions
                + strategy.entry.guards
                + strategy.entry.filters
            )
        }

        should_try_l2 = (
            RSILevel.L2_FACTOR in allowed_levels
            and evaluation.overall.signal_count >= 10
            and sample_df is not None
            and len(sample_df) >= 15
            and (
                stagnant_l1_rounds >= 2
                or cluster.cluster_id
                in {"missing_discriminatory_factor", "whipsaw_stopout_in_chop"}
            )
        )

        if should_try_l2 and sample_df is not None:
            existing_dynamic_map = IndicatorRegistry._dynamic_registry
            existing_dynamic = set(existing_dynamic_map.keys())
            chosen: (
                tuple[
                    str,
                    str,
                    str,
                    Literal["<", "<=", ">", ">="],
                    float,
                    Literal["positive", "negative"],
                    Literal["llm", "offline_template"],
                ]
                | None
            ) = None
            reused_stored_record: FactorRecord | None = None
            llm_candidate = self._try_llm_synthesize(cluster, used_indicators, sample_df)
            if llm_candidate is not None:
                chosen = (
                    llm_candidate[0],
                    llm_candidate[1],
                    llm_candidate[2],
                    llm_candidate[3],
                    llm_candidate[4],
                    llm_candidate[5],
                    "llm",
                )
            else:
                for template in _select_ordered_templates(cluster.cluster_id):
                    code_hash = hashlib.sha256(template.code.encode("utf-8")).hexdigest()
                    if template.name in used_indicators:
                        continue
                    if template.name in existing_dynamic:
                        existing_fn = existing_dynamic_map.get(template.name)
                        if getattr(existing_fn, "_alphaevo_factor_code_hash", None) != code_hash:
                            continue
                    elif IndicatorRegistry.is_registered(template.name):
                        continue
                    stored: FactorRecord | None = None
                    if self._factor_store is not None:
                        stored = self._factor_store.get(template.name)
                        if stored is not None and (
                            stored.status != "active" or stored.code != template.code
                        ):
                            continue
                    if code_hash not in self._pit_cache:
                        self._pit_cache[code_hash] = self._sandbox.probe_pit(
                            template.code, sample_df
                        ).passed
                    if not self._pit_cache[code_hash]:
                        continue
                    reused_stored_record = stored
                    chosen = (
                        template.name,
                        template.description,
                        template.code,
                        template.op,
                        template.threshold,
                        template.expected_direction,
                        "offline_template",
                    )
                    break

            if chosen is not None:
                f_name, f_desc, f_code, f_op, f_thresh, f_dir, via = chosen
                f_code_hash = hashlib.sha256(f_code.encode("utf-8")).hexdigest()
                lease_acquired = False
                with _DYNAMIC_LEASE_LOCK:
                    live_dynamic = IndicatorRegistry._dynamic_registry
                    if f_name in live_dynamic:
                        if (
                            getattr(live_dynamic.get(f_name), "_alphaevo_factor_code_hash", None)
                            != f_code_hash
                        ):
                            return CoSteerDecision(
                                chosen_level=RSILevel.L1_STRATEGY,
                                cluster=cluster,
                                proposed_changes=[],
                            )
                    else:
                        if (
                            not persist
                            and reused_stored_record is None
                            and (
                                self._factor_store is None or self._factor_store.get(f_name) is None
                            )
                        ):
                            _TRANSIENT_FACTORS.add(f_name)
                        IndicatorRegistry.register_dynamic(
                            f_name,
                            cast("IndicatorFn", _build_indicator_fn(f_code, self._sandbox)),
                        )
                    if acquire_lease and not persist and f_name in _TRANSIENT_FACTORS:
                        _DYNAMIC_LEASES[f_name] = _DYNAMIC_LEASES.get(f_name, 0) + 1
                        lease_acquired = True
                factor_record = (
                    reused_stored_record
                    if reused_stored_record is not None
                    else FactorRecord(
                        name=f_name,
                        description=f_desc,
                        rationale=f"Co-STEER L2 ({cluster.cluster_id}, via={via}): {cluster.evidence}",
                        code=f_code,
                        expected_direction=f_dir,
                        status="active",
                    )
                )
                persisted = reused_stored_record is not None
                if persist and self._factor_store is not None and reused_stored_record is None:
                    stored_rec, inserted = self._factor_store.save_if_absent(factor_record)
                    if not inserted and (
                        stored_rec.status != "active" or stored_rec.code != factor_record.code
                    ):
                        with _DYNAMIC_LEASE_LOCK:
                            if (
                                f_name not in existing_dynamic
                                and _DYNAMIC_LEASES.get(f_name, 0) == 0
                            ):
                                _TRANSIENT_FACTORS.discard(f_name)
                                IndicatorRegistry.unregister_dynamic(f_name)
                        return CoSteerDecision(
                            chosen_level=RSILevel.L1_STRATEGY,
                            cluster=cluster,
                            proposed_changes=[],
                        )
                    factor_record = stored_rec
                    persisted = True
                if strategy.entry.triggers:
                    target_field = "entry.guards"
                elif strategy.entry.logic == "or":
                    # Hard-filter factors belong in entry.filters so OR triggers don't become looser.
                    target_field = "entry.filters"
                else:
                    target_field = "entry.conditions"
                change = StrategyChange(
                    change_type=ChangeType.ADD_CONDITION,
                    target=target_field,
                    from_value=None,
                    to_value={
                        "indicator": f_name,
                        "op": f_op,
                        "value": f_thresh,
                    },
                    reason=f"Co-STEER L2 factor ({cluster.cluster_id}): {f_desc}",
                )
                return CoSteerDecision(
                    chosen_level=RSILevel.L2_FACTOR,
                    cluster=cluster,
                    synthesized_factor_name=f_name,
                    synthesized_via=via,
                    pit_verified=True,
                    persisted_to_store=persisted,
                    lease_acquired=lease_acquired,
                    synthesized_factor_record=factor_record,
                    proposed_changes=[change],
                )

        return CoSteerDecision(
            chosen_level=RSILevel.L1_STRATEGY,
            cluster=cluster,
            proposed_changes=[],
        )


__all__ = [
    "BlindSpotCluster",
    "BlindSpotMiner",
    "CoSteerDecision",
    "ExecutableScar",
    "FailureClusterId",
    "OfflineFactorTemplate",
    "ScarRegistry",
]
