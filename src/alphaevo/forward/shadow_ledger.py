"""Bitemporal Forward Shadow Ledger with hash-chained commitments & asymmetric gates."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]

from alphaevo.backtest.condition import ConditionEvaluator
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.forward.contracts import (
    BitemporalEnvelope,
    CandidateSpec,
    ForecastCommit,
    ForwardDecayReport,
    OutcomeEvent,
    OutcomeHorizon,
    PromotionDecision,
    SetupInstance,
    StageTransition,
)
from alphaevo.models.enums import MarketRegime, PromotionStage
from alphaevo.sampler.regime import RegimeDetector
from alphaevo.truth.ledger import compute_strategy_hash

if TYPE_CHECKING:
    from alphaevo.models.execution import EvaluationReport
    from alphaevo.models.market import IndicatorContext
    from alphaevo.models.strategy import Strategy

_GENESIS_HASH = "0" * 64
_HORIZONS: tuple[Literal[5, 10, 20], ...] = (5, 10, 20)


def _canonical_commit_hash(commit: ForecastCommit) -> str:
    """Compute SHA-256 over canonical ForecastCommit payload (excluding record_hash)."""
    payload = commit.model_dump(mode="json")
    payload["record_hash"] = ""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _truncate_pit(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Strictly truncate OHLCV DataFrame so no row after ``as_of`` is visible."""
    if df.empty or "date" not in df:
        return cast("pd.DataFrame", df.iloc[0:0].copy())
    dates = pd.to_datetime(df["date"]).dt.date
    mask = dates <= as_of
    out = df.loc[mask].copy().reset_index(drop=True)
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

    # ── Candidate registration & persistence ───────────────────────────

    def register_candidate(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        stage: PromotionStage = PromotionStage.SHADOW,
    ) -> CandidateSpec:
        """Preregister a strategy candidate in the Forward Shadow Ledger."""
        if stage in {PromotionStage.SHADOW, PromotionStage.CANARY, PromotionStage.ACTIVE} and (
            evaluation.anti_overfit.is_overfit
        ):
            raise ValueError(
                f"Candidate '{strategy.meta.id}' failed anti-overfit veto; cannot enter {stage.value}"
            )

        candidates = self._load_candidates()
        spec = CandidateSpec(
            strategy_id=strategy.meta.id,
            family_id=strategy.meta.family_id,
            strategy_hash=compute_strategy_hash(strategy),
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
    ) -> ForecastCommit:
        """Generate & hash-chain forward stock setups strictly using data <= ``as_of``."""
        raw_setups: list[SetupInstance] = []
        sector_map = sectors or {}

        for symbol in sorted(market_data):
            pit_df = _truncate_pit(market_data[symbol], as_of)
            if pit_df.empty or len(pit_df) < 15:
                continue
            last_bar_date = pd.to_datetime(pit_df["date"].iloc[-1]).date()
            if (as_of - last_bar_date).days > 5:
                continue

            idx = len(pit_df) - 1
            ctx = contexts.get(symbol) if contexts else None
            if not self._cond_eval.evaluate_entry(strategy.entry, pit_df, idx, ctx):
                continue

            close_price = float(pit_df["close"].iloc[idx])
            if close_price <= 0:
                continue

            stop_pct = self._resolve_stop_pct(strategy)
            planned_stop = round(close_price * (1.0 - stop_pct), 4)
            planned_rr = self._resolve_planned_rr(strategy, stop_pct)
            min_rr = strategy.setup.min_rr if strategy.setup is not None else 1.0
            if planned_rr < min_rr:
                continue

            target_price = round(close_price * (1.0 + stop_pct * planned_rr), 4)
            detected_regime = market_regime or self._regime_detector.detect(pit_df)
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
            as_of=as_of,
            market_temperature=temp_label,
            total_raw_candidates=len(raw_setups),
            sector_capped_count=len(capped),
            signals=capped,
            prev_hash=prev_hash,
            record_hash="",
        )
        commit.record_hash = _canonical_commit_hash(commit)

        with self.commits_file.open("a", encoding="utf-8") as fh:
            fh.write(commit.model_dump_json() + "\n")
        self.head_hash_file.write_text(commit.record_hash + "\n", encoding="utf-8")
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
                errors.append(f"Line {idx} ({cmt.commit_id}): record_hash tampered or corrupted")
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
    ) -> list[OutcomeEvent]:
        """Compute matured vs immature T+5/10/20 outcomes up to ``evaluated_as_of``."""
        truncated_market = {
            sym: _truncate_pit(df, evaluated_as_of) for sym, df in market_data.items()
        }
        truncated_bench = (
            _truncate_pit(benchmark_df, evaluated_as_of) if benchmark_df is not None else None
        )

        events: list[OutcomeEvent] = []
        for commit in self.load_commits():
            for sig in commit.signals:
                sym_df = truncated_market.get(sig.symbol)
                horizons_out: dict[int, OutcomeHorizon] = {}
                if sym_df is None or sym_df.empty or "date" not in sym_df:
                    for h in _HORIZONS:
                        horizons_out[h] = OutcomeHorizon(horizon_days=h, mature=False)
                    events.append(
                        OutcomeEvent(
                            signal_id=sig.signal_id,
                            strategy_id=sig.strategy_id,
                            symbol=sig.symbol,
                            sector=sig.sector,
                            as_of=sig.envelope.as_of,
                            evaluated_as_of=evaluated_as_of,
                            horizons=horizons_out,
                        )
                    )
                    continue

                dates = pd.to_datetime(sym_df["date"]).dt.date
                future_bars = sym_df.loc[dates > sig.envelope.as_of].reset_index(drop=True)
                for h in _HORIZONS:
                    if len(future_bars) < h:
                        horizons_out[h] = OutcomeHorizon(horizon_days=h, mature=False)
                        continue
                    window = future_bars.iloc[:h]
                    entry_open = (
                        float(window["open"].iloc[0])
                        if "open" in window
                        else sig.entry_reference_price
                    )
                    entry_open = entry_open if entry_open > 0 else sig.entry_reference_price
                    exit_close = float(window["close"].iloc[-1])
                    exit_date = pd.to_datetime(window["date"].iloc[-1]).date()

                    raw_ret = (exit_close / max(1e-6, sig.entry_reference_price)) - 1.0
                    stopped_out = False
                    effective_exit = exit_close
                    if sig.planned_stop_price is not None and "low" in window:
                        hit_mask = (
                            pd.to_numeric(window["low"], errors="coerce") <= sig.planned_stop_price
                        )
                        if bool(hit_mask.any()):
                            stopped_out = True
                            effective_exit = float(sig.planned_stop_price)
                            hit_pos = int(hit_mask.to_numpy().argmax())
                            exit_date = pd.to_datetime(window["date"].iloc[hit_pos]).date()

                    tradable_ret = (
                        (effective_exit / max(1e-6, entry_open)) - 1.0
                    ) - round_trip_cost
                    bench_ret = self._compute_benchmark_return(
                        truncated_bench,
                        start_after=sig.envelope.as_of,
                        horizon_days=h,
                    )
                    horizons_out[h] = OutcomeHorizon(
                        horizon_days=h,
                        mature=True,
                        exit_date=exit_date,
                        raw_signal_return=round(raw_ret, 6),
                        tradable_post_cost_return=round(tradable_ret, 6),
                        benchmark_return=round(bench_ret, 6),
                        excess_return=round(tradable_ret - bench_ret, 6),
                        stopped_out=stopped_out,
                    )

                events.append(
                    OutcomeEvent(
                        signal_id=sig.signal_id,
                        strategy_id=sig.strategy_id,
                        symbol=sig.symbol,
                        sector=sig.sector,
                        as_of=sig.envelope.as_of,
                        evaluated_as_of=evaluated_as_of,
                        horizons=horizons_out,
                    )
                )

        # Deduplicate outcomes by (signal_id, evaluated_as_of) rewrite for clean latest state
        with self.outcomes_file.open("w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(ev.model_dump_json() + "\n")
        return events

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

    def build_decay_report(
        self,
        strategy_id: str,
        *,
        horizon_days: int = 10,
    ) -> ForwardDecayReport:
        """Compute Backtest -> Forward Decay Ratio and drawdown/expectancy bounds."""
        candidate = self.get_candidate(strategy_id)
        stage = candidate.stage if candidate is not None else PromotionStage.SHADOW
        backtest_avg = candidate.backtest_avg_return if candidate is not None else 0.0

        commits = self.load_commits(strategy_id)
        total_committed = sum(len(c.signals) for c in commits)
        outcomes = self.load_outcomes(strategy_id)

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
                backtest_avg_return=backtest_avg,
                decay_ratio=None,
            )

        rets = [hor.tradable_post_cost_return for _, hor in mature_pairs]
        excess_rets = [hor.excess_return for _, hor in mature_pairs]
        avg_ret = sum(rets) / len(rets)
        avg_excess = sum(excess_rets) / len(excess_rets)
        win_rate = sum(1 for r in rets if r > 0) / len(rets)
        var = sum((r - avg_ret) ** 2 for r in rets) / max(1, len(rets) - 1)
        std_r = math.sqrt(max(0.0, var))
        lb90 = avg_ret - 1.2816 * std_r / math.sqrt(max(1.0, n_eff))

        # Sequential drawdown on realized forward trades ordered by commit date
        ordered = sorted(mature_pairs, key=lambda pair: pair[0].as_of)
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for _, hor in ordered:
            equity *= 1.0 + hor.tradable_post_cost_return
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
    ) -> PromotionDecision:
        """Apply asymmetric promotion rules (Slow-Promote / Instant-Halt)."""
        candidates = self._load_candidates()
        cand = candidates.get(strategy_id)
        if cand is None:
            raise ValueError(f"Strategy '{strategy_id}' is not registered in ShadowLedger")

        prev_stage = cand.stage
        decay = self.build_decay_report(strategy_id, horizon_days=horizon_days)
        chain_ok, chain_errors = self.verify_chain()

        # Instant halt on severe forward drawdown or systematic post-cost bleeding
        halt_reasons: list[str] = []
        if decay.max_forward_drawdown > max_drawdown_halt:
            halt_reasons.append(
                f"Forward max drawdown {decay.max_forward_drawdown:.1%} exceeds redline {max_drawdown_halt:.1%}"
            )
        if decay.mature_signals >= 15 and decay.forward_avg_post_cost_return < -0.03:
            halt_reasons.append(
                f"Forward post-cost expectancy {decay.forward_avg_post_cost_return:.2%} < -3.0% over {decay.mature_signals} signals"
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

        passed: list[str] = []
        blocking: list[str] = []

        if chain_ok:
            passed.append("hash_chain_verified")
        else:
            blocking.append(f"Hash chain verification failed: {'; '.join(chain_errors)}")

        if decay.mature_trading_days >= min_trading_days:
            passed.append(f"mature_trading_days>={min_trading_days}")
        else:
            blocking.append(f"Mature trading days {decay.mature_trading_days} < {min_trading_days}")

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

        if decay.forward_avg_excess_return > 0.0:
            passed.append("positive_forward_excess_return")
        else:
            blocking.append(f"Forward excess return {decay.forward_avg_excess_return:.2%} <= 0%")

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

        cand.stage = new_stage
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
    def _compute_benchmark_return(
        bench_df: pd.DataFrame | None,
        *,
        start_after: date,
        horizon_days: int,
    ) -> float:
        if bench_df is None or bench_df.empty or "close" not in bench_df or "date" not in bench_df:
            return 0.0
        dates = pd.to_datetime(bench_df["date"]).dt.date
        pre = bench_df.loc[dates <= start_after]
        fut = bench_df.loc[dates > start_after]
        if pre.empty or len(fut) < horizon_days:
            return 0.0
        start_c = float(pre["close"].iloc[-1])
        end_c = float(fut["close"].iloc[horizon_days - 1])
        if start_c <= 0:
            return 0.0
        return (end_c / start_c) - 1.0

    def _load_candidates(self) -> dict[str, CandidateSpec]:
        if not self.candidates_file.is_file():
            return {}
        raw: dict[str, Any] = json.loads(self.candidates_file.read_text(encoding="utf-8"))
        return {k: CandidateSpec.model_validate(v) for k, v in raw.items()}

    def _save_candidates(self, candidates: dict[str, CandidateSpec]) -> None:
        payload = {k: v.model_dump(mode="json") for k, v in sorted(candidates.items())}
        self.candidates_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


__all__ = ["ShadowLedger"]
