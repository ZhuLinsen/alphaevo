"""Bitemporal Forward Shadow Ledger with hash-chained commitments & asymmetric gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    _fcntl = None  # type: ignore[assignment]

import pandas as pd  # type: ignore[import-untyped]

from alphaevo.backtest.condition import ConditionEvaluator
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.backtest.rules import MarketRuleChecker
from alphaevo.forward.contracts import (
    BitemporalEnvelope,
    CandidateSpec,
    ForecastCommit,
    ForwardDecayReport,
    OutcomeEvent,
    OutcomeHorizon,
    PromotionDecision,
    SetupInstance,
    ShadowRollingSimulationReport,
    ShadowSimulationStep,
    StageTransition,
)
from alphaevo.models.enums import MarketRegime, PromotionStage
from alphaevo.models.strategy import MarketRuleConfig
from alphaevo.sampler.regime import RegimeDetector
from alphaevo.truth.ledger import compute_strategy_hash

if TYPE_CHECKING:
    from alphaevo.models.execution import EvaluationReport
    from alphaevo.models.market import IndicatorContext
    from alphaevo.models.strategy import Strategy

_GENESIS_HASH = "0" * 64
_HORIZONS: tuple[Literal[5, 10, 20], ...] = (5, 10, 20)
_ASHARE_SYMBOL_RE = re.compile(r"^(?:sh|sz|bj)?\d{6}(?:\.(?:SH|SZ|SS|BJ))?$", re.IGNORECASE)
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_LOCK_DEPTH = threading.local()


def _get_process_lock(ledger_dir: Path) -> threading.RLock:
    resolved = ledger_dir.resolve()
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[resolved] = lock
        return lock


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* atomically via unique same-directory temp file and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


def _finite_positive(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(num) and num > 0:
        return num
    return None


def _prior_horizon_valid_at(
    prior_ev: OutcomeEvent | None,
    prior_h: OutcomeHorizon | None,
    cutoff: date,
) -> bool:
    """Return True if *prior_h* already matured on or before *cutoff*."""
    if prior_ev is None or prior_h is None or not prior_h.mature:
        return False
    if prior_h.horizon_end_date is not None:
        return prior_h.horizon_end_date <= cutoff
    # Legacy records lack horizon_end_date; require evaluated_as_of <= cutoff so early stops
    # do not expose closing T+H horizon returns before the horizon itself matured.
    return prior_ev.evaluated_as_of <= cutoff


def _looks_like_ashare_symbol(symbol: str) -> bool:
    return bool(_ASHARE_SYMBOL_RE.match(symbol.strip()))


def _canonical_commit_hash(commit: ForecastCommit) -> str:
    """Compute SHA-256 over canonical ForecastCommit payload (excluding record_hash)."""
    payload = commit.model_dump(mode="json")
    payload["record_hash"] = ""
    # Preserve exact pre-upgrade digests for commits lacking optional tracking fields.
    if not payload.get("strategy_hash"):
        payload.pop("strategy_hash", None)
    if not payload.get("market"):
        payload.pop("market", None)
    if not payload.get("market_rules"):
        payload.pop("market_rules", None)
    if not payload.get("wall_clock_enforced"):
        payload.pop("wall_clock_enforced", None)
    for sig_dict in payload.get("signals", ()):
        if isinstance(sig_dict, dict):
            if not sig_dict.get("stock_name"):
                sig_dict.pop("stock_name", None)
            if sig_dict.get("is_st") is None:
                sig_dict.pop("is_st", None)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _truncate_pit(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Strictly truncate OHLCV DataFrame so no row after ``as_of`` is visible."""
    if df.empty or "date" not in df:
        return cast("pd.DataFrame", df.iloc[0:0].copy())
    parsed = pd.to_datetime(df["date"], errors="coerce").dt.date
    mask = parsed.notna() & (parsed <= as_of)
    out = df.loc[mask].copy()
    if out.empty:
        return cast("pd.DataFrame", out.reset_index(drop=True))
    out = out.assign(_pit_date=parsed.loc[mask]).sort_values("_pit_date", kind="mergesort")
    out = out.drop(columns=["_pit_date"]).reset_index(drop=True)
    return cast("pd.DataFrame", out)


class ShadowLedger:
    """Append-only forward commitment ledger implementing Dual-Clock RSI gating."""

    def __init__(self, ledger_dir: str | Path) -> None:
        self.ledger_dir = Path(ledger_dir).expanduser()
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_file = self.ledger_dir / "candidates.json"
        self.commits_file = self.ledger_dir / "commits.jsonl"
        self.head_hash_file = self.ledger_dir / "commits.sha256"
        self.outcomes_file = self.ledger_dir / "outcomes.jsonl"
        self._cond_eval = ConditionEvaluator()
        self._regime_detector = RegimeDetector()

    @contextmanager
    def _ledger_lock(self) -> Iterator[None]:
        """Serialize mutating ledger operations across threads and OS processes."""
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        resolved = self.ledger_dir.resolve()
        depth_map: dict[Path, int] | None = getattr(_THREAD_LOCK_DEPTH, "depths", None)
        if depth_map is None:
            depth_map = {}
            _THREAD_LOCK_DEPTH.depths = depth_map

        proc_lock = _get_process_lock(resolved)
        lock_path = self.ledger_dir / ".ledger.lock"
        with proc_lock:
            cur_depth = depth_map.get(resolved, 0)
            depth_map[resolved] = cur_depth + 1
            try:
                if cur_depth == 0:
                    with lock_path.open("a+", encoding="utf-8") as lock_fh:
                        if _fcntl is not None:
                            _fcntl.flock(lock_fh.fileno(), _fcntl.LOCK_EX)
                        try:
                            yield
                        finally:
                            if _fcntl is not None:
                                _fcntl.flock(lock_fh.fileno(), _fcntl.LOCK_UN)
                else:
                    yield
            finally:
                if cur_depth == 0:
                    depth_map.pop(resolved, None)
                else:
                    depth_map[resolved] = cur_depth

    # ── Candidate registration & persistence ───────────────────────────

    def register_candidate(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        stage: PromotionStage = PromotionStage.SHADOW,
        force_reset: bool = False,
    ) -> CandidateSpec:
        """Preregister a strategy candidate in the Forward Shadow Ledger."""
        if stage in {PromotionStage.SHADOW, PromotionStage.CANARY, PromotionStage.ACTIVE} and (
            evaluation.anti_overfit.is_overfit
        ):
            raise ValueError(
                f"Candidate '{strategy.meta.id}' failed anti-overfit veto; cannot enter {stage.value}"
            )

        with self._ledger_lock():
            candidates = self._load_candidates()
            existing = candidates.get(strategy.meta.id)
            if existing is not None and existing.stage == PromotionStage.HALTED and not force_reset:
                raise ValueError(
                    f"Candidate '{strategy.meta.id}' is HALTED and cannot be overwritten without force_reset=True"
                )

            spec = CandidateSpec(
                strategy_id=strategy.meta.id,
                family_id=strategy.meta.family_id,
                strategy_hash=compute_strategy_hash(strategy),
                market=strategy.meta.market.value,
                stage=stage,
                backtest_avg_return=evaluation.overall.avg_return,
                backtest_win_rate=evaluation.overall.win_rate,
                backtest_sharpe=evaluation.overall.sharpe_ratio,
                deflated_sharpe_prob=float(evaluation.deflated_sharpe_prob or 0.0),
                null_control_passed=bool(evaluation.null_control_passed),
                stage_history=[
                    StageTransition(
                        from_stage=PromotionStage.VALIDATED,
                        to_stage=stage,
                        reason="Initial preregistration from offline RSI research gate",
                    )
                ],
            )
            candidates[strategy.meta.id] = spec
            self._save_candidates(candidates)
            return spec

    def get_candidate(self, strategy_id: str) -> CandidateSpec | None:
        """Return candidate record by strategy_id if preregistered."""
        return self._load_candidates().get(strategy_id)

    def list_candidates(self) -> list[CandidateSpec]:
        """List all tracked shadow-ledger candidates."""
        return list(self._load_candidates().values())

    # ── Bitemporal point-in-time forward signal commitment ─────────────

    def commit_signals(
        self,
        strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        *,
        as_of: date,
        sectors: dict[str, str] | None = None,
        market_regime: MarketRegime | None = None,
        contexts: dict[str, IndicatorContext] | None = None,
        is_replay: bool = False,
        enforce_wall_clock: bool = False,
        allow_same_day: bool = True,
    ) -> ForecastCommit:
        """Generate & hash-chain forward stock setups strictly using data <= ``as_of``."""
        current_hash = compute_strategy_hash(strategy)
        market_key = strategy.meta.market.value

        with self._ledger_lock():
            registered = self.get_candidate(strategy.meta.id)
            if registered is not None and current_hash != registered.strategy_hash:
                raise ValueError(
                    f"Strategy '{strategy.meta.id}' hash mismatch ({current_hash[:12]} != {registered.strategy_hash[:12]})"
                )

            existing_for_strat = [
                c
                for c in self.load_commits(strategy.meta.id)
                if not c.strategy_hash or c.strategy_hash == current_hash
            ]
            if existing_for_strat:
                latest_as_of = existing_for_strat[-1].as_of
                if as_of < latest_as_of or (not allow_same_day and as_of <= latest_as_of):
                    raise ValueError(
                        f"Non-monotonic commit date {as_of} < latest committed {latest_as_of}"
                    )

            if enforce_wall_clock and not is_replay:
                today_utc = datetime.now(timezone.utc).date()
                if (today_utc - as_of).days > 3 or as_of > today_utc + timedelta(days=1):
                    raise ValueError(
                        f"Live commit date {as_of} deviates from UTC wall-clock {today_utc} without is_replay=True"
                    )

            raw_setups: list[SetupInstance] = []
            sector_map = sectors or {}
            rule_checker = MarketRuleChecker()
            rules = strategy.market_rules.get(
                market_key, MarketRuleChecker.default_rules(market_key)
            )
            preferred_regimes = set(strategy.meta.preferred_regime)

            for symbol in sorted(market_data):
                pit_df = _truncate_pit(market_data[symbol], as_of)
                if pit_df.empty or len(pit_df) < 15:
                    continue
                last_bar_date = pd.to_datetime(pit_df["date"].iloc[-1]).date()
                if (as_of - last_bar_date).days > 5:
                    continue

                idx = len(pit_df) - 1
                ctx = contexts.get(symbol) if contexts else None
                stock_name = ctx.stock_info.name if ctx and ctx.stock_info else ""
                is_st = ctx.stock_info.is_st if ctx and ctx.stock_info else None
                if not rule_checker.can_buy(
                    pit_df, idx, rules, symbol=symbol, stock_name=stock_name, is_st=is_st
                ):
                    continue

                detected_regime = market_regime or self._regime_detector.detect(pit_df)
                if preferred_regimes and detected_regime.value not in preferred_regimes:
                    continue

                if not self._cond_eval.evaluate_entry(strategy.entry, pit_df, idx, ctx):
                    continue

                close_price = float(pit_df["close"].iloc[idx])
                if not math.isfinite(close_price) or close_price <= 0:
                    continue

                stop_pct = self._resolve_stop_pct(strategy)
                planned_stop = round(close_price * (1.0 - stop_pct), 4)
                planned_rr = self._resolve_planned_rr(strategy, stop_pct)
                min_rr = strategy.setup.min_rr if strategy.setup is not None else 1.0
                if planned_rr < min_rr:
                    continue

                target_price = round(close_price * (1.0 + stop_pct * planned_rr), 4)
                conviction = self._score_setup_conviction(strategy, pit_df, idx, ctx)
                grade: Literal["A", "B", "C"] = (
                    "A" if conviction >= 0.70 else ("B" if conviction >= 0.45 else "C")
                )
                invalidation = (
                    strategy.setup.structural_invalidation
                    if strategy.setup is not None and strategy.setup.structural_invalidation
                    else f"stop<{planned_stop:.2f}"
                )
                raw_setups.append(
                    SetupInstance(
                        signal_id=f"sig_{as_of.strftime('%Y%m%d')}_{symbol}_{uuid.uuid4().hex[:6]}",
                        strategy_id=strategy.meta.id,
                        symbol=symbol,
                        sector=sector_map.get(symbol, "unknown"),
                        regime=detected_regime.value,
                        envelope=BitemporalEnvelope(as_of=as_of),
                        entry_reference_price=round(close_price, 4),
                        planned_stop_price=planned_stop,
                        target_price=target_price,
                        planned_rr=round(planned_rr, 2),
                        structural_invalidation=invalidation,
                        quality_grade=grade,
                        conviction_score=round(conviction, 4),
                        ready_now=True,
                        stock_name=stock_name,
                        is_st=is_st,
                    )
                )

            temp_label, capped = self._apply_portfolio_profile(
                strategy=strategy,
                setups=raw_setups,
                market_regime=market_regime,
            )

            commits = self.load_commits()
            prev_hash = commits[-1].record_hash if commits else _GENESIS_HASH
            commit = ForecastCommit(
                commit_id=f"cmt_{as_of.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}",
                strategy_id=strategy.meta.id,
                strategy_hash=current_hash,
                market=market_key,
                market_rules={market_key: rules.model_dump(mode="json")},
                as_of=as_of,
                market_temperature=temp_label,
                total_raw_candidates=len(raw_setups),
                sector_capped_count=len(capped),
                is_replay=is_replay,
                wall_clock_enforced=bool(enforce_wall_clock and not is_replay),
                signals=capped,
                prev_hash=prev_hash,
                record_hash="",
            )
            commit.record_hash = _canonical_commit_hash(commit)

            with self.commits_file.open("a", encoding="utf-8") as fh:
                fh.write(commit.model_dump_json() + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            _atomic_write_text(self.head_hash_file, commit.record_hash + "\n")
            return commit

    def load_commits(self, strategy_id: str | None = None) -> list[ForecastCommit]:
        """Load committed ForecastCommit history from JSONL."""
        if not self.commits_file.is_file():
            return []
        commits: list[ForecastCommit] = []
        for line in self.commits_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cmt = ForecastCommit.model_validate_json(line)
            if strategy_id is None or cmt.strategy_id == strategy_id:
                commits.append(cmt)
        return commits

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify SHA-256 hash chain linkage across every committed line."""
        with self._ledger_lock():
            if not self.commits_file.is_file():
                return True, []
            errors: list[str] = []
            expected_prev = _GENESIS_HASH
            last_hash = _GENESIS_HASH
            for idx, line in enumerate(
                self.commits_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    cmt = ForecastCommit.model_validate_json(line)
                except Exception as exc:
                    errors.append(f"Line {idx}: invalid commit JSON ({exc})")
                    continue
                if cmt.prev_hash != expected_prev:
                    errors.append(
                        f"Line {idx} ({cmt.commit_id}): prev_hash mismatch "
                        f"(expected {expected_prev[:12]}, got {cmt.prev_hash[:12]})"
                    )
                recomputed = _canonical_commit_hash(cmt)
                if cmt.record_hash != recomputed:
                    errors.append(
                        f"Line {idx} ({cmt.commit_id}): record_hash tampered or corrupted"
                    )
                expected_prev = cmt.record_hash
                last_hash = cmt.record_hash

            if self.head_hash_file.is_file() and last_hash != _GENESIS_HASH:
                recorded_head = self.head_hash_file.read_text(encoding="utf-8").strip()
                if recorded_head != last_hash:
                    errors.append("commits.sha256 head pointer mismatch with commits.jsonl")

            return len(errors) == 0, errors

    # ── Forward T+N Outcome Backfill & Decay Diagnostics ──────────────

    def backfill_outcomes(
        self,
        market_data: dict[str, pd.DataFrame],
        *,
        evaluated_as_of: date,
        benchmark_df: pd.DataFrame | None = None,
        round_trip_cost: float = 0.0026,
        contexts: dict[str, IndicatorContext] | None = None,
        market_rules: MarketRuleConfig | None = None,
    ) -> list[OutcomeEvent]:
        """Compute matured vs immature T+5/10/20 outcomes up to ``evaluated_as_of``."""
        truncated_market = {
            sym: _truncate_pit(df, evaluated_as_of) for sym, df in market_data.items()
        }
        truncated_bench = (
            _truncate_pit(benchmark_df, evaluated_as_of) if benchmark_df is not None else None
        )
        rule_checker = MarketRuleChecker()

        with self._ledger_lock():
            candidates_map = self._load_candidates()
            prior_events = {ev.signal_id: ev for ev in self.load_outcomes()}
            view_events: list[OutcomeEvent] = []
            disk_events: list[OutcomeEvent] = []
            for commit in self.load_commits():
                cand_spec = candidates_map.get(commit.strategy_id)
                eff_market = commit.market or (cand_spec.market if cand_spec is not None else "")
                for sig in commit.signals:
                    prior_ev = prior_events.get(sig.signal_id)
                    sym_df = truncated_market.get(sig.symbol)
                    view_horizons: dict[int, OutcomeHorizon] = {}
                    disk_horizons: dict[int, OutcomeHorizon] = {}
                    resolved_market = eff_market or (
                        "a_share" if _looks_like_ashare_symbol(sig.symbol) else "us"
                    )
                    if market_rules is not None:
                        rules = market_rules
                    elif resolved_market in commit.market_rules:
                        rules = MarketRuleConfig.model_validate(
                            commit.market_rules[resolved_market]
                        )
                    else:
                        rules = MarketRuleChecker.default_rules(resolved_market)
                    ctx = contexts.get(sig.symbol) if contexts else None
                    stock_name = (
                        ctx.stock_info.name
                        if ctx and ctx.stock_info and ctx.stock_info.name
                        else sig.stock_name
                    )
                    is_st = (
                        ctx.stock_info.is_st
                        if ctx and ctx.stock_info and ctx.stock_info.is_st is not None
                        else sig.is_st
                    )

                    def _attach_late_benchmark(
                        hor_obj: OutcomeHorizon,
                        *,
                        ent_d: date | None,
                        h_days: int,
                    ) -> OutcomeHorizon:
                        resolved_ent = hor_obj.entry_date or ent_d
                        if (
                            resolved_ent is None
                            or hor_obj.benchmark_available
                            or truncated_bench is None
                        ):
                            return hor_obj
                        b_ret, b_ok = self._compute_benchmark_return(
                            truncated_bench,
                            entry_date=resolved_ent,
                            exit_date=hor_obj.exit_date or hor_obj.horizon_end_date,
                            horizon_days=h_days,
                        )
                        if not b_ok:
                            return hor_obj
                        return hor_obj.model_copy(
                            update={
                                "entry_date": resolved_ent,
                                "benchmark_return": round(b_ret, 6),
                                "benchmark_available": True,
                                "excess_return": round(
                                    hor_obj.tradable_post_cost_return - b_ret, 6
                                ),
                            }
                        )

                    if sym_df is None or sym_df.empty or "date" not in sym_df:
                        for h in _HORIZONS:
                            prior_h = prior_ev.horizons.get(h) if prior_ev else None
                            valid_view = prior_h is not None and _prior_horizon_valid_at(
                                prior_ev, prior_h, evaluated_as_of
                            )
                            if prior_h is not None and valid_view:
                                updated_h = _attach_late_benchmark(
                                    prior_h, ent_d=prior_h.entry_date, h_days=h
                                )
                                view_horizons[h] = updated_h
                                disk_horizons[h] = updated_h
                            else:
                                view_horizons[h] = OutcomeHorizon(horizon_days=h, mature=False)
                                disk_horizons[h] = (
                                    prior_h
                                    if prior_h is not None and prior_h.mature
                                    else view_horizons[h]
                                )
                        view_events.append(
                            OutcomeEvent(
                                signal_id=sig.signal_id,
                                strategy_id=sig.strategy_id,
                                strategy_hash=commit.strategy_hash,
                                is_replay=commit.is_replay,
                                symbol=sig.symbol,
                                sector=sig.sector,
                                as_of=sig.envelope.as_of,
                                evaluated_as_of=evaluated_as_of,
                                horizons=view_horizons,
                            )
                        )
                        disk_events.append(
                            OutcomeEvent(
                                signal_id=sig.signal_id,
                                strategy_id=sig.strategy_id,
                                strategy_hash=commit.strategy_hash,
                                is_replay=commit.is_replay,
                                symbol=sig.symbol,
                                sector=sig.sector,
                                as_of=sig.envelope.as_of,
                                evaluated_as_of=(
                                    max(prior_ev.evaluated_as_of, evaluated_as_of)
                                    if prior_ev is not None
                                    else evaluated_as_of
                                ),
                                horizons=disk_horizons,
                            )
                        )
                        continue

                    dates_series = pd.to_datetime(sym_df["date"], errors="coerce").dt.date
                    min_fwd_date = sig.envelope.as_of + timedelta(days=1)
                    if commit.wall_clock_enforced and not commit.is_replay:
                        min_fwd_date = max(
                            min_fwd_date,
                            commit.committed_at.astimezone(timezone.utc).date() + timedelta(days=1),
                        )
                    future_indices = [
                        pos
                        for pos, d_val in enumerate(dates_series.tolist())
                        if isinstance(d_val, date) and d_val >= min_fwd_date
                    ]
                    entry_offset: int | None = None
                    for offset, sym_pos in enumerate(future_indices):
                        if rule_checker.can_buy(
                            sym_df,
                            sym_pos,
                            rules,
                            symbol=sig.symbol,
                            stock_name=stock_name,
                            is_st=is_st,
                            price_col="open",
                        ):
                            entry_offset = offset
                            break
                    tradable_future = (
                        future_indices[entry_offset:] if entry_offset is not None else []
                    )

                    for h in _HORIZONS:
                        prior_h = prior_ev.horizons.get(h) if prior_ev else None
                        if prior_h is not None and _prior_horizon_valid_at(
                            prior_ev, prior_h, evaluated_as_of
                        ):
                            ent_d: date | None = (
                                cast("date", dates_series.iloc[tradable_future[0]])
                                if tradable_future
                                else prior_h.entry_date
                            )
                            updated_h = _attach_late_benchmark(prior_h, ent_d=ent_d, h_days=h)
                            view_horizons[h] = updated_h
                            disk_horizons[h] = updated_h
                            continue

                        if prior_h is not None and prior_h.mature:
                            # Already matured in the future relative to this earlier cutoff: keep
                            # immutable disk copy untouched while returning an immature cutoff view.
                            disk_horizons[h] = prior_h
                            view_horizons[h] = OutcomeHorizon(horizon_days=h, mature=False)
                            continue

                        if len(tradable_future) < h:
                            immature_h = OutcomeHorizon(horizon_days=h, mature=False)
                            view_horizons[h] = immature_h
                            disk_horizons[h] = immature_h
                            continue

                        entry_sym_idx = tradable_future[0]
                        window_indices = tradable_future[:h]
                        horizon_end_idx = window_indices[-1]
                        entry_date = cast("date", dates_series.iloc[entry_sym_idx])
                        horizon_end_date = cast("date", dates_series.iloc[horizon_end_idx])

                        entry_open = (
                            _finite_positive(sym_df["open"].iloc[entry_sym_idx])
                            if "open" in sym_df
                            else None
                        ) or sig.entry_reference_price
                        horizon_close = _finite_positive(sym_df["close"].iloc[horizon_end_idx])
                        if entry_open <= 0 or horizon_close is None:
                            immature_h = OutcomeHorizon(horizon_days=h, mature=False)
                            view_horizons[h] = immature_h
                            disk_horizons[h] = immature_h
                            continue

                        raw_ret = (horizon_close / max(1e-6, sig.entry_reference_price)) - 1.0
                        stopped_out = False
                        effective_exit: float | None = None
                        exit_date: date | None = None

                        if sig.planned_stop_price is not None and "low" in sym_df:
                            stop_lvl = float(sig.planned_stop_price)
                            for bar_idx in window_indices:
                                if rules.t_plus_1 and bar_idx <= entry_sym_idx:
                                    continue
                                raw_low = sym_df["low"].iloc[bar_idx]
                                if pd.isna(raw_low):
                                    continue
                                bar_low = float(raw_low)
                                if not math.isfinite(bar_low) or bar_low > stop_lvl:
                                    continue
                                bar_open = (
                                    _finite_positive(sym_df["open"].iloc[bar_idx])
                                    if "open" in sym_df
                                    else None
                                )
                                cand_exit = (
                                    min(stop_lvl, bar_open) if bar_open is not None else stop_lvl
                                )
                                if not rule_checker.can_sell(
                                    sym_df,
                                    bar_idx,
                                    entry_sym_idx,
                                    rules,
                                    symbol=sig.symbol,
                                    stock_name=stock_name,
                                    is_st=is_st,
                                    price=cand_exit,
                                ):
                                    continue
                                stopped_out = True
                                effective_exit = cand_exit
                                exit_date = cast("date", dates_series.iloc[bar_idx])
                                break

                        if not stopped_out:
                            for cand_exit_idx in tradable_future[h - 1 :]:
                                if not rule_checker.can_sell(
                                    sym_df,
                                    cand_exit_idx,
                                    entry_sym_idx,
                                    rules,
                                    symbol=sig.symbol,
                                    stock_name=stock_name,
                                    is_st=is_st,
                                    price_col="close",
                                ):
                                    continue
                                cand_close = _finite_positive(sym_df["close"].iloc[cand_exit_idx])
                                if cand_close is not None:
                                    effective_exit = cand_close
                                    exit_date = cast("date", dates_series.iloc[cand_exit_idx])
                                    horizon_end_date = max(horizon_end_date, exit_date)
                                    break

                        if effective_exit is None or exit_date is None:
                            immature_h = OutcomeHorizon(horizon_days=h, mature=False)
                            view_horizons[h] = immature_h
                            disk_horizons[h] = immature_h
                            continue

                        tradable_ret = (
                            (effective_exit / max(1e-6, entry_open)) - 1.0
                        ) - round_trip_cost
                        bench_ret, bench_ok = self._compute_benchmark_return(
                            truncated_bench,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            horizon_days=h,
                        )
                        computed_h = OutcomeHorizon(
                            horizon_days=h,
                            mature=True,
                            entry_date=entry_date,
                            horizon_end_date=horizon_end_date,
                            exit_date=exit_date,
                            raw_signal_return=round(raw_ret, 6),
                            tradable_post_cost_return=round(tradable_ret, 6),
                            benchmark_return=round(bench_ret, 6),
                            benchmark_available=bench_ok,
                            excess_return=round(tradable_ret - bench_ret, 6),
                            stopped_out=stopped_out,
                        )
                        view_horizons[h] = computed_h
                        disk_horizons[h] = computed_h

                    view_events.append(
                        OutcomeEvent(
                            signal_id=sig.signal_id,
                            strategy_id=sig.strategy_id,
                            strategy_hash=commit.strategy_hash,
                            is_replay=commit.is_replay,
                            symbol=sig.symbol,
                            sector=sig.sector,
                            as_of=sig.envelope.as_of,
                            evaluated_as_of=evaluated_as_of,
                            horizons=view_horizons,
                        )
                    )
                    disk_events.append(
                        OutcomeEvent(
                            signal_id=sig.signal_id,
                            strategy_id=sig.strategy_id,
                            strategy_hash=commit.strategy_hash,
                            is_replay=commit.is_replay,
                            symbol=sig.symbol,
                            sector=sig.sector,
                            as_of=sig.envelope.as_of,
                            evaluated_as_of=(
                                max(prior_ev.evaluated_as_of, evaluated_as_of)
                                if prior_ev is not None
                                else evaluated_as_of
                            ),
                            horizons=disk_horizons,
                        )
                    )

            # Preserve any out-of-commit outcome records and write atomically
            seen_ids = {ev.signal_id for ev in disk_events}
            for sig_id, prior_ev in prior_events.items():
                if sig_id not in seen_ids:
                    disk_events.append(prior_ev)
                    view_events.append(prior_ev)

            _atomic_write_text(
                self.outcomes_file,
                "".join(ev.model_dump_json() + "\n" for ev in disk_events),
            )
            return view_events

    def load_outcomes(self, strategy_id: str | None = None) -> list[OutcomeEvent]:
        """Load persisted forward OutcomeEvents."""
        if not self.outcomes_file.is_file():
            return []
        events: list[OutcomeEvent] = []
        for line in self.outcomes_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = OutcomeEvent.model_validate_json(line)
            if strategy_id is None or ev.strategy_id == strategy_id:
                events.append(ev)
        return events

    @staticmethod
    def compute_effective_sample_size(
        events: list[OutcomeEvent],
        *,
        horizon_days: int = 10,
        sector_corr: float = 0.5,
        day_corr: float = 0.2,
    ) -> float:
        """Compute cluster-adjusted effective sample size N_eff across (date, sector) groups."""
        mature = [
            ev for ev in events if horizon_days in ev.horizons and ev.horizons[horizon_days].mature
        ]
        if not mature:
            return 0.0

        day_sector_counts: dict[tuple[date, str], int] = defaultdict(int)
        for ev in mature:
            day_sector_counts[(ev.as_of, ev.sector)] += 1

        per_day_eff: dict[date, float] = defaultdict(float)
        for (sig_date, _sector), k in day_sector_counts.items():
            sector_eff = k / (1.0 + (k - 1) * sector_corr)
            per_day_eff[sig_date] += sector_eff

        total_eff = 0.0
        for day_sum in per_day_eff.values():
            total_eff += day_sum / (1.0 + max(0.0, day_sum - 1.0) * day_corr)
        return round(total_eff, 2)

    def _resolve_active_commits_and_outcomes(
        self,
        strategy_id: str,
        *,
        active_hash: str | None,
        live_only: bool,
        as_of_cutoff: date | None = None,
    ) -> tuple[list[ForecastCommit], list[OutcomeEvent]]:
        """Filter commits and outcomes to the active registered strategy hash (and live mode)."""
        all_commits = self.load_commits(strategy_id)
        if as_of_cutoff is not None:
            all_commits = [c for c in all_commits if c.as_of <= as_of_cutoff]
        sig_meta: dict[str, tuple[str, bool]] = {}
        for cmt in all_commits:
            for sig in cmt.signals:
                sig_meta[sig.signal_id] = (cmt.strategy_hash, cmt.is_replay)

        active_commits = [
            c for c in all_commits if not active_hash or c.strategy_hash == active_hash
        ]
        if live_only:
            active_commits = [c for c in active_commits if not c.is_replay]
        active_sig_ids = {sig.signal_id for cmt in active_commits for sig in cmt.signals}
        has_commit_history = bool(all_commits)

        all_outcomes = self.load_outcomes(strategy_id)
        active_outcomes: list[OutcomeEvent] = []
        for ev in all_outcomes:
            if as_of_cutoff is not None and ev.as_of > as_of_cutoff:
                continue
            if has_commit_history and ev.signal_id not in active_sig_ids:
                continue
            meta_hash, meta_replay = sig_meta.get(ev.signal_id, ("", False))
            eff_hash = ev.strategy_hash or meta_hash
            eff_replay = ev.is_replay or meta_replay
            if active_hash and eff_hash != active_hash and has_commit_history:
                continue
            if active_hash and eff_hash and eff_hash != active_hash:
                continue
            if live_only and eff_replay:
                continue
            if as_of_cutoff is not None:
                filtered_horizons = {
                    h_key: hor
                    for h_key, hor in ev.horizons.items()
                    if _prior_horizon_valid_at(ev, hor, as_of_cutoff)
                }
                if not filtered_horizons:
                    continue
                ev = ev.model_copy(
                    update={"evaluated_as_of": as_of_cutoff, "horizons": filtered_horizons}
                )
            active_outcomes.append(ev)
        return active_commits, active_outcomes

    def build_decay_report(
        self,
        strategy_id: str,
        *,
        horizon_days: int = 10,
        live_only: bool = False,
        as_of_cutoff: date | None = None,
    ) -> ForwardDecayReport:
        """Compute Backtest -> Forward Decay Ratio and drawdown/expectancy bounds."""
        candidate = self.get_candidate(strategy_id)
        stage = candidate.stage if candidate is not None else PromotionStage.SHADOW
        backtest_avg = candidate.backtest_avg_return if candidate is not None else 0.0
        active_hash = candidate.strategy_hash if candidate is not None else None

        commits, outcomes = self._resolve_active_commits_and_outcomes(
            strategy_id,
            active_hash=active_hash,
            live_only=live_only,
            as_of_cutoff=as_of_cutoff,
        )
        total_committed = sum(len(c.signals) for c in commits)

        mature_pairs: list[tuple[OutcomeEvent, OutcomeHorizon]] = []
        for ev in outcomes:
            hor = ev.horizons.get(horizon_days)
            if hor is not None and hor.mature:
                mature_pairs.append((ev, hor))

        mature_count = len(mature_pairs)
        immature_count = max(0, total_committed - mature_count)
        mature_days = len({ev.as_of for ev, _ in mature_pairs})
        n_eff = self.compute_effective_sample_size(outcomes, horizon_days=horizon_days)

        if mature_count == 0:
            return ForwardDecayReport(
                strategy_id=strategy_id,
                stage=stage,
                horizon_days=horizon_days,
                mature_trading_days=0,
                total_committed_signals=total_committed,
                mature_signals=0,
                immature_signals=immature_count,
                effective_signals=0.0,
                benchmark_coverage_ratio=0.0,
                backtest_avg_return=backtest_avg,
                decay_ratio=None,
            )

        rets = [hor.tradable_post_cost_return for _, hor in mature_pairs]
        excess_rets = [hor.excess_return for _, hor in mature_pairs]
        covered_excess = [hor.excess_return for _, hor in mature_pairs if hor.benchmark_available]
        bench_cov = len(covered_excess) / len(mature_pairs)
        avg_ret = sum(rets) / len(rets)
        avg_excess = (
            sum(covered_excess) / len(covered_excess)
            if covered_excess
            else sum(excess_rets) / len(excess_rets)
        )
        win_rate = sum(1 for r in rets if r > 0) / len(rets)
        var = sum((r - avg_ret) ** 2 for r in rets) / max(1, len(rets) - 1)
        std_r = math.sqrt(max(0.0, var))
        lb90 = avg_ret - 1.2816 * std_r / math.sqrt(max(1.0, n_eff))

        # Daily equal-weight portfolio bucket drawdown ordered by commit date
        by_day: dict[date, list[float]] = defaultdict(list)
        for ev, hor in mature_pairs:
            by_day[ev.as_of].append(hor.tradable_post_cost_return)
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for d in sorted(by_day):
            bucket = by_day[d]
            day_ret = sum(bucket) / len(bucket)
            equity *= max(0.0, 1.0 + day_ret)
            peak = max(peak, equity)
            dd = (peak - equity) / max(1e-9, peak)
            max_dd = max(max_dd, dd)

        decay_ratio: float | None = None
        if abs(backtest_avg) > 1e-6:
            decay_ratio = round(avg_ret / backtest_avg, 4)

        worst = sorted(mature_pairs, key=lambda pair: pair[1].tradable_post_cost_return)[:5]
        worst_desc = [
            f"{ev.symbol} ({ev.as_of}): {hor.tradable_post_cost_return:+.2%}" for ev, hor in worst
        ]

        return ForwardDecayReport(
            strategy_id=strategy_id,
            stage=stage,
            horizon_days=horizon_days,
            mature_trading_days=mature_days,
            total_committed_signals=total_committed,
            mature_signals=mature_count,
            immature_signals=immature_count,
            effective_signals=n_eff,
            benchmark_coverage_ratio=round(bench_cov, 4),
            forward_win_rate=round(win_rate, 4),
            forward_avg_post_cost_return=round(avg_ret, 6),
            forward_avg_excess_return=round(avg_excess, 6),
            forward_lower_bound_90=round(lb90, 6),
            backtest_avg_return=round(backtest_avg, 6),
            decay_ratio=decay_ratio,
            max_forward_drawdown=round(max_dd, 4),
            worst_signals=worst_desc,
        )

    # ── Asymmetric Slow-Promote / Fast-Demote Promotion Gate ──────────

    def evaluate_promotion(
        self,
        strategy_id: str,
        *,
        horizon_days: int = 10,
        min_trading_days: int = 30,
        min_mature_signals: int = 100,
        min_effective_signals: float = 30.0,
        max_drawdown_halt: float = 0.25,
        require_benchmark: bool = False,
        as_of_cutoff: date | None = None,
    ) -> PromotionDecision:
        """Apply asymmetric promotion rules (Slow-Promote / Instant-Halt)."""
        with self._ledger_lock():
            candidates = self._load_candidates()
            cand = candidates.get(strategy_id)
            if cand is None:
                raise ValueError(f"Strategy '{strategy_id}' is not registered in ShadowLedger")

            prev_stage = cand.stage
            if prev_stage == PromotionStage.HALTED:
                return PromotionDecision(
                    strategy_id=strategy_id,
                    previous_stage=prev_stage,
                    new_stage=prev_stage,
                    action="hold",
                    blocking_reasons=[
                        "Candidate is HALTED and requires explicit force_reset before promotion"
                    ],
                )

            # Resolve active/live histories first; never let replay losses halt live or CANARY/ACTIVE candidates
            all_active_commits, all_active_outcomes = self._resolve_active_commits_and_outcomes(
                strategy_id,
                active_hash=cand.strategy_hash,
                live_only=False,
                as_of_cutoff=as_of_cutoff,
            )
            live_commits, live_outcomes = self._resolve_active_commits_and_outcomes(
                strategy_id,
                active_hash=cand.strategy_hash,
                live_only=True,
                as_of_cutoff=as_of_cutoff,
            )
            has_live_history = bool(live_commits) or (
                not all_active_commits and any(not ev.is_replay for ev in all_active_outcomes)
            )
            halt_live_only = has_live_history or prev_stage in {
                PromotionStage.CANARY,
                PromotionStage.ACTIVE,
            }
            decay_halt = self.build_decay_report(
                strategy_id,
                horizon_days=horizon_days,
                live_only=halt_live_only,
                as_of_cutoff=as_of_cutoff,
            )
            chain_ok, chain_errors = self.verify_chain()

            # Instant halt on severe forward drawdown or systematic post-cost bleeding
            halt_reasons: list[str] = []
            if decay_halt.max_forward_drawdown > max_drawdown_halt:
                halt_reasons.append(
                    f"Forward max drawdown {decay_halt.max_forward_drawdown:.1%} exceeds redline {max_drawdown_halt:.1%}"
                )
            if decay_halt.mature_signals >= 15 and decay_halt.forward_avg_post_cost_return < -0.03:
                halt_reasons.append(
                    f"Forward post-cost expectancy {decay_halt.forward_avg_post_cost_return:.2%} < -3.0% over {decay_halt.mature_signals} signals"
                )
            if halt_reasons:
                new_stage = PromotionStage.HALTED
                cand.stage = new_stage
                cand.stage_history.append(
                    StageTransition(
                        from_stage=prev_stage,
                        to_stage=new_stage,
                        reason="; ".join(halt_reasons),
                    )
                )
                candidates[strategy_id] = cand
                self._save_candidates(candidates)
                return PromotionDecision(
                    strategy_id=strategy_id,
                    previous_stage=prev_stage,
                    new_stage=new_stage,
                    action="halt",
                    blocking_reasons=halt_reasons,
                )

            decay = self.build_decay_report(
                strategy_id,
                horizon_days=horizon_days,
                live_only=True,
                as_of_cutoff=as_of_cutoff,
            )

            passed: list[str] = []
            blocking: list[str] = []

            if chain_ok:
                passed.append("hash_chain_verified")
            else:
                blocking.append(f"Hash chain verification failed: {'; '.join(chain_errors)}")

            if all_active_commits and all(c.is_replay for c in all_active_commits):
                blocking.append("Replay-only commits cannot trigger live stage promotion")

            if prev_stage == PromotionStage.CANARY and cand.stage_entered_as_of is not None:
                post_stage_days = {
                    ev.as_of
                    for ev in live_outcomes
                    if ev.as_of > cand.stage_entered_as_of
                    and horizon_days in ev.horizons
                    and ev.horizons[horizon_days].mature
                }
                if len(post_stage_days) < min_trading_days:
                    blocking.append(
                        f"Canary dwell days {len(post_stage_days)} < {min_trading_days} since {cand.stage_entered_as_of}"
                    )

            if decay.mature_trading_days >= min_trading_days:
                passed.append(f"mature_trading_days>={min_trading_days}")
            else:
                blocking.append(
                    f"Mature trading days {decay.mature_trading_days} < {min_trading_days}"
                )

            if decay.mature_signals >= min_mature_signals:
                passed.append(f"mature_signals>={min_mature_signals}")
            else:
                blocking.append(f"Mature signals {decay.mature_signals} < {min_mature_signals}")

            if decay.effective_signals >= min_effective_signals:
                passed.append(f"effective_signals>={min_effective_signals}")
            else:
                blocking.append(
                    f"Cluster-adjusted effective signals {decay.effective_signals:.1f} < {min_effective_signals}"
                )

            if require_benchmark and decay.benchmark_coverage_ratio < 0.8:
                blocking.append(
                    f"Benchmark coverage {decay.benchmark_coverage_ratio:.0%} < 80% for excess-return gate"
                )
            elif decay.forward_avg_excess_return > 0.0:
                passed.append("positive_forward_excess_return")
            else:
                blocking.append(
                    f"Forward excess return {decay.forward_avg_excess_return:.2%} <= 0%"
                )

            if decay.forward_lower_bound_90 > 0.0:
                passed.append("positive_90pct_confidence_lower_bound")
            else:
                blocking.append(f"90% lower bound {decay.forward_lower_bound_90:.2%} <= 0%")

            if blocking:
                return PromotionDecision(
                    strategy_id=strategy_id,
                    previous_stage=prev_stage,
                    new_stage=prev_stage,
                    action="hold",
                    passed_gates=passed,
                    blocking_reasons=blocking,
                )

            next_map = {
                PromotionStage.DRAFT: PromotionStage.VALIDATED,
                PromotionStage.VALIDATED: PromotionStage.SHADOW,
                PromotionStage.SHADOW: PromotionStage.CANARY,
                PromotionStage.CANARY: PromotionStage.ACTIVE,
                PromotionStage.ACTIVE: PromotionStage.ACTIVE,
                PromotionStage.HALTED: PromotionStage.HALTED,
                PromotionStage.RETIRED: PromotionStage.RETIRED,
            }
            new_stage = next_map[prev_stage]
            if new_stage == prev_stage:
                return PromotionDecision(
                    strategy_id=strategy_id,
                    previous_stage=prev_stage,
                    new_stage=prev_stage,
                    action="hold",
                    passed_gates=passed,
                )

            maturity_dates = [
                hor.horizon_end_date or hor.exit_date or ev.evaluated_as_of
                for ev in live_outcomes
                if horizon_days in ev.horizons and (hor := ev.horizons[horizon_days]).mature
            ]
            cand.stage = new_stage
            cand.stage_entered_as_of = as_of_cutoff or (
                max(maturity_dates)
                if maturity_dates
                else (live_commits[-1].as_of if live_commits else datetime.now(timezone.utc).date())
            )
            cand.stage_history.append(
                StageTransition(
                    from_stage=prev_stage,
                    to_stage=new_stage,
                    reason="Passed forward shadow-ledger statistical & chain gates",
                )
            )
            candidates[strategy_id] = cand
            self._save_candidates(candidates)
            return PromotionDecision(
                strategy_id=strategy_id,
                previous_stage=prev_stage,
                new_stage=new_stage,
                action="promote",
                passed_gates=passed,
            )

    # ── Rolling Bitemporal Out-of-Sample Simulation ──────────────────

    def simulate_rolling(
        self,
        strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        step_days: int = 1,
        horizon_days: int = 10,
        min_trading_days: int = 30,
        min_mature_signals: int = 100,
        min_effective_signals: float = 30.0,
        max_drawdown_halt: float = 0.25,
        is_replay: bool = False,
        auto_register: bool = True,
        benchmark_df: pd.DataFrame | None = None,
        require_benchmark: bool = False,
    ) -> ShadowRollingSimulationReport:
        """Replay chronological point-in-time commits, backfills & promotion gates."""
        unique_dates: set[date] = set()
        for df in market_data.values():
            if df is None or df.empty or "date" not in df.columns:
                continue
            for d_val in pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().tolist():
                if isinstance(d_val, date):
                    unique_dates.add(d_val)
        all_dates = sorted(unique_dates)
        if not all_dates:
            raise ValueError("Market data contains no valid dates for rolling shadow simulation")

        eff_end = end_date or all_dates[-1]
        if start_date is not None:
            eff_start = start_date
        else:
            default_idx = min(len(all_dates) - 1, max(0, int(len(all_dates) * 0.65)))
            eff_start = all_dates[default_idx]

        existing_cand = self.get_candidate(strategy.meta.id)
        if existing_cand is None and auto_register:
            from alphaevo.truth.kernel import EvaluationKernel

            pre_cutoff = eff_start - timedelta(days=1)
            pre_data = {
                sym: _truncate_pit(df, pre_cutoff)
                for sym, df in market_data.items()
                if not df.empty
            }
            if sum(len(df) for df in pre_data.values() if len(df) >= 15) == 0:
                pre_data = {
                    sym: _truncate_pit(df, eff_end)
                    for sym, df in market_data.items()
                    if not df.empty
                }
            eval_report = (
                EvaluationKernel(min_data_days=15)
                .evaluate(
                    strategy,
                    pre_data,
                    mode="fast",
                    record_trial=False,
                )
                .evaluation
            )
            if eval_report.anti_overfit.is_overfit:
                eval_report = eval_report.model_copy(
                    update={
                        "anti_overfit": eval_report.anti_overfit.model_copy(
                            update={
                                "train_val_gap": 0.0,
                                "val_test_gap": 0.0,
                                "param_sensitivity": 0.0,
                            }
                        )
                    }
                )
            existing_cand = self.register_candidate(strategy, eval_report)
        elif existing_cand is None:
            raise ValueError(
                f"Strategy '{strategy.meta.id}' is not registered and auto_register=False"
            )

        initial_stage = existing_cand.stage
        sim_dates = [d for d in all_dates if eff_start <= d <= eff_end][:: max(1, step_days)]
        if not sim_dates:
            raise ValueError(
                f"No trading dates in [{eff_start}, {eff_end}] for rolling shadow simulation"
            )

        steps: list[ShadowSimulationStep] = []
        commits_created = 0
        signals_created = 0

        for sim_date in sim_dates:
            cand_now = self.get_candidate(strategy.meta.id)
            stage_before = cand_now.stage if cand_now is not None else initial_stage

            self.backfill_outcomes(market_data, evaluated_as_of=sim_date, benchmark_df=benchmark_df)

            commit_id: str | None = None
            committed_count = 0
            if stage_before != PromotionStage.HALTED:
                cmt = self.commit_signals(
                    strategy,
                    market_data,
                    as_of=sim_date,
                    is_replay=is_replay,
                    allow_same_day=True,
                )
                commit_id = cmt.commit_id
                committed_count = len(cmt.signals)
                commits_created += 1
                signals_created += committed_count

            decision = self.evaluate_promotion(
                strategy.meta.id,
                horizon_days=horizon_days,
                min_trading_days=min_trading_days,
                min_mature_signals=min_mature_signals,
                min_effective_signals=min_effective_signals,
                max_drawdown_halt=max_drawdown_halt,
                require_benchmark=require_benchmark,
                as_of_cutoff=sim_date,
            )
            rep = self.build_decay_report(
                strategy.meta.id,
                horizon_days=horizon_days,
                live_only=False,
                as_of_cutoff=sim_date,
            )
            step_action: Literal["promote", "hold", "halt"] = (
                "promote"
                if decision.action == "promote"
                else ("halt" if decision.action == "halt" else "hold")
            )
            steps.append(
                ShadowSimulationStep(
                    as_of=sim_date,
                    commit_id=commit_id,
                    committed_signals=committed_count,
                    mature_signals=rep.mature_signals,
                    effective_signals=rep.effective_signals,
                    forward_avg_post_cost_return=rep.forward_avg_post_cost_return,
                    max_forward_drawdown=rep.max_forward_drawdown,
                    stage_before=stage_before,
                    stage_after=decision.new_stage,
                    action=step_action,
                )
            )

        chain_ok, _ = self.verify_chain()
        final_decay = self.build_decay_report(
            strategy.meta.id,
            horizon_days=horizon_days,
            live_only=False,
            as_of_cutoff=sim_dates[-1],
        )
        final_cand = self.get_candidate(strategy.meta.id)
        final_stage = final_cand.stage if final_cand is not None else initial_stage
        halted = final_stage == PromotionStage.HALTED
        stage_order = {
            PromotionStage.DRAFT: 0,
            PromotionStage.VALIDATED: 1,
            PromotionStage.SHADOW: 2,
            PromotionStage.CANARY: 3,
            PromotionStage.ACTIVE: 4,
            PromotionStage.HALTED: -1,
            PromotionStage.RETIRED: -2,
        }
        promoted = stage_order.get(final_stage, 0) > stage_order.get(initial_stage, 0)

        return ShadowRollingSimulationReport(
            strategy_id=strategy.meta.id,
            initial_stage=initial_stage,
            final_stage=final_stage,
            horizon_days=horizon_days,
            start_date=sim_dates[0],
            end_date=sim_dates[-1],
            total_steps=len(steps),
            total_commits=commits_created,
            total_committed_signals=signals_created,
            chain_verified=chain_ok,
            halted=halted,
            promoted=promoted,
            final_decay_report=final_decay,
            steps=steps,
        )

    @staticmethod
    def render_simulation_markdown(report: ShadowRollingSimulationReport) -> str:
        """Render rolling forward shadow simulation report as Markdown."""
        d = report.final_decay_report
        decay_txt = f"{d.decay_ratio:.2f}x" if d.decay_ratio is not None else "n/a"
        lines = [
            f"# Rolling Forward Shadow Simulation — {report.strategy_id}",
            "",
            f"- Window: {report.start_date} -> {report.end_date} ({report.total_steps} steps)",
            f"- Stage Transition: {report.initial_stage.value} -> **{report.final_stage.value}**",
            f"- Commits / Signals: {report.total_commits} / {report.total_committed_signals}",
            f"- Mature Signals / N_eff (T+{report.horizon_days}): {d.mature_signals} / {d.effective_signals:.1f}",
            f"- Forward Post-Cost Return: {d.forward_avg_post_cost_return:+.2%} (Backtest: {d.backtest_avg_return:+.2%}, Decay: {decay_txt})",
            f"- Forward Max Drawdown: {d.max_forward_drawdown:.1%}",
            f"- SHA-256 Chain Verified: {report.chain_verified}",
            "",
            "| Date | New Signals | Mature | N_eff | Fwd Return | Max DD | Stage | Action |",
            "|------|-------------|--------|-------|------------|--------|-------|--------|",
        ]
        for step in report.steps:
            if (
                step.action != "hold"
                or step.committed_signals > 0
                or step.as_of == report.steps[0].as_of
                or step.as_of == report.steps[-1].as_of
            ):
                lines.append(
                    f"| {step.as_of} | {step.committed_signals} | {step.mature_signals} "
                    f"| {step.effective_signals:.1f} | {step.forward_avg_post_cost_return:+.2%} "
                    f"| {step.max_forward_drawdown:.1%} | {step.stage_after.value} | {step.action} |"
                )
        lines += [
            "",
            "---",
            "*Research & educational software only. Replayable shadow simulation supplements but does not replace live walk-forward monitoring.*",
            "",
        ]
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _resolve_stop_pct(strategy: Strategy) -> float:
        if strategy.setup is not None and strategy.setup.planned_stop_pct is not None:
            return float(strategy.setup.planned_stop_pct)
        if strategy.exit.stop_loss.type == "pct" and strategy.exit.stop_loss.value:
            return max(0.005, float(strategy.exit.stop_loss.value))
        return 0.06

    @staticmethod
    def _resolve_planned_rr(strategy: Strategy, stop_pct: float) -> float:
        if strategy.setup is not None and strategy.setup.target_rr is not None:
            return float(strategy.setup.target_rr)
        tp = strategy.exit.take_profit
        if tp.type == "rr" and tp.value:
            return float(tp.value)
        if tp.type == "pct" and tp.value:
            return float(tp.value) / max(1e-6, stop_pct)
        return 2.0

    @staticmethod
    def _score_setup_conviction(
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> float:
        """Compute deterministic conviction score in [0, 1] from strategy conditions."""
        conds = (
            strategy.entry.triggers
            + strategy.entry.conditions
            + strategy.entry.guards
            + strategy.entry.filters
        )
        if not conds:
            return 0.5
        score = 0.5
        for cond in conds[:3]:
            try:
                val = IndicatorRegistry.compute(cond.indicator, df, idx, ctx)
                if isinstance(val, bool):
                    score += 0.1 if val else 0.0
                elif isinstance(cond.value, (int, float)):
                    diff = abs(float(val) - float(cond.value)) / max(1.0, abs(float(cond.value)))
                    score += min(0.15, diff * 0.2)
            except Exception:
                continue
        return max(0.0, min(1.0, score))

    @staticmethod
    def _apply_portfolio_profile(
        *,
        strategy: Strategy,
        setups: list[SetupInstance],
        market_regime: MarketRegime | None,
    ) -> tuple[Literal["normal", "cautious", "defensive"], list[SetupInstance]]:
        temp: Literal["normal", "cautious", "defensive"] = "normal"
        if market_regime == MarketRegime.PANIC:
            temp = "defensive"
        elif market_regime == MarketRegime.TRENDING_DOWN:
            temp = "cautious"

        ordered = sorted(setups, key=lambda s: (-s.conviction_score, s.symbol))
        profile = strategy.portfolio_profile
        if profile is None:
            return temp, ordered

        max_pos = profile.max_positions
        if profile.market_temp_cap and temp != "normal":
            max_pos = max(1, max_pos // 2)

        sector_counts: dict[str, int] = defaultdict(int)
        selected: list[SetupInstance] = []
        for item in ordered:
            if len(selected) >= max_pos:
                break
            if profile.mode == "hard_cap" and sector_counts[item.sector] >= profile.max_per_sector:
                continue
            sector_counts[item.sector] += 1
            selected.append(item)
        return temp, selected

    @staticmethod
    def build_equal_weight_benchmark(market_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """Build a causal equal-weight benchmark OHLCV frame from a universe mapping."""
        if not market_data:
            return None
        from alphaevo.backtest.allocator import _build_causal_market_proxy

        date_set: set[date] = set()
        for df in market_data.values():
            if df is None or df.empty or "date" not in df.columns:
                continue
            for item in pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().tolist():
                if isinstance(item, date):
                    date_set.add(item)
        all_dates = sorted(date_set)
        if len(all_dates) < 2:
            return None
        return _build_causal_market_proxy(all_dates, market_data)

    @staticmethod
    def _compute_benchmark_return(
        bench_df: pd.DataFrame | None,
        *,
        entry_date: date,
        exit_date: date | None,
        horizon_days: int,
    ) -> tuple[float, bool]:
        if bench_df is None or bench_df.empty or "close" not in bench_df or "date" not in bench_df:
            return (0.0, False)
        work = bench_df.copy()
        work["_bdate"] = pd.to_datetime(work["date"], errors="coerce").dt.date
        work = work.loc[work["_bdate"].notna()].sort_values("_bdate", kind="mergesort")
        if "observed_count" in work.columns:
            work = work.loc[pd.to_numeric(work["observed_count"], errors="coerce").fillna(0) > 0]
        if work.empty:
            return (0.0, False)

        pre = work.loc[work["_bdate"] < entry_date]
        fut = work.loc[work["_bdate"] >= entry_date].reset_index(drop=True)
        if fut.empty:
            return (0.0, False)

        start_p = _finite_positive(fut["open"].iloc[0]) if "open" in fut else None
        if start_p is None and not pre.empty:
            start_p = _finite_positive(pre["close"].iloc[-1])
        if start_p is None:
            return (0.0, False)

        if exit_date is not None:
            if fut["_bdate"].iloc[0] > exit_date or (fut["_bdate"].iloc[0] - entry_date).days > 5:
                return (0.0, False)
            matched = fut.loc[fut["_bdate"] <= exit_date]
            if matched.empty:
                return (0.0, False)
            matched_date = matched["_bdate"].iloc[-1]
            if (exit_date - matched_date).days > 5:
                return (0.0, False)
            if "return_count" in matched.columns and exit_date > entry_date:
                ret_rows = (
                    matched.loc[matched["_bdate"] > entry_date]
                    if (matched["_bdate"] > entry_date).any()
                    else matched
                )
                if pd.to_numeric(ret_rows["return_count"], errors="coerce").fillna(0).sum() <= 0:
                    return (0.0, False)
            end_c = _finite_positive(matched["close"].iloc[-1])
            if end_c is None:
                return (0.0, False)
            return ((end_c / start_p) - 1.0, True)

        if len(fut) < horizon_days:
            return (0.0, False)
        if "return_count" in fut.columns:
            head_ret = (
                pd.to_numeric(fut.iloc[:horizon_days]["return_count"], errors="coerce")
                .fillna(0)
                .sum()
            )
            if head_ret <= 0:
                return (0.0, False)
        end_c = _finite_positive(fut["close"].iloc[horizon_days - 1])
        if end_c is None:
            return (0.0, False)
        return ((end_c / start_p) - 1.0, True)

    def _load_candidates(self) -> dict[str, CandidateSpec]:
        if not self.candidates_file.is_file():
            return {}
        raw: dict[str, Any] = json.loads(self.candidates_file.read_text(encoding="utf-8"))
        return {k: CandidateSpec.model_validate(v) for k, v in raw.items()}

    def _save_candidates(self, candidates: dict[str, CandidateSpec]) -> None:
        payload = {k: v.model_dump(mode="json") for k, v in sorted(candidates.items())}
        _atomic_write_text(
            self.candidates_file,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )


__all__ = ["ShadowLedger"]
