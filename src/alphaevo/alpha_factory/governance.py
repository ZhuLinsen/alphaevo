"""Dynamic Alpha Factor Governance, OOS Decay Auditing, and Lineage Tracking."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
from alphaevo.alpha_factory.factory import _execute_factor_series
from alphaevo.alpha_factory.sandbox import FactorSandbox
from alphaevo.alpha_factory.validator import FactorValidator, ValidationThresholds
from alphaevo.models.strategy import Strategy
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.store import StrategyStore
from alphaevo.truth.pit_probe import probe_point_in_time_invariance

logger = logging.getLogger(__name__)

FactorHealthStatus = Literal[
    "healthy",
    "decayed",
    "redundant",
    "pit_unsafe",
    "execution_failed",
]


class FactorGovernanceConfig(BaseModel):
    """Thresholds controlling factor out-of-sample auditing and retirement."""

    train_ratio: float = Field(default=0.6, gt=0.2, lt=0.9)
    forward_horizon: int = Field(default=5, ge=1, le=60)
    min_oos_ic: float = Field(default=0.015, ge=0.0)
    min_oos_ir: float = Field(default=0.20, ge=0.0)
    min_decay_ratio: float = Field(default=0.25, ge=0.0)
    max_redundancy_corr: float = Field(default=0.75, gt=0.0, le=1.0)
    require_pit_invariance: bool = True


class FactorAuditEntry(BaseModel):
    """Out-of-sample governance scorecard for one persisted factor."""

    name: str
    status_before: Literal["active", "retired", "failed"] = "active"
    expected_direction: Literal["positive", "negative"] = "positive"
    health: FactorHealthStatus
    pit_passed: bool = True
    train_ic: float = 0.0
    train_ir: float = 0.0
    oos_ic: float = 0.0
    oos_ir: float = 0.0
    ic_decay_ratio: float | None = None
    turnover: float = 0.0
    max_peer_corr: float = 0.0
    most_correlated_peer: str | None = None
    referenced_strategies: list[str] = Field(default_factory=list)
    referenced_families: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    action_taken: Literal["kept", "retired", "would_retire"] = "kept"


class FactorGovernanceReport(BaseModel):
    """Full lifecycle governance audit summary across dynamic factors."""

    audited_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_audited: int = 0
    healthy_count: int = 0
    decayed_count: int = 0
    redundant_count: int = 0
    pit_unsafe_count: int = 0
    execution_failed_count: int = 0
    retired_count: int = 0
    auto_retire_applied: bool = False
    entries: list[FactorAuditEntry] = Field(default_factory=list)


def _split_panel(
    market_data: dict[str, pd.DataFrame],
    *,
    train_ratio: float,
    forward_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Chronologically split symbol OHLCVs into stacked train and OOS evaluation panels."""
    train_frames: list[pd.DataFrame] = []
    oos_frames: list[pd.DataFrame] = []
    sample_df: pd.DataFrame | None = None

    for sym in sorted(market_data):
        df = market_data[sym]
        if df is None or df.empty or "close" not in df.columns or len(df) < 25:
            continue
        ordered = (
            df.sort_values("date").reset_index(drop=True)
            if "date" in df.columns
            else df.reset_index(drop=True)
        )
        if sample_df is None or len(ordered) > len(sample_df):
            sample_df = ordered
        close = pd.to_numeric(ordered["close"], errors="coerce").astype(float)
        ordered = ordered.copy()
        ordered["_symbol"] = sym
        ordered["_fwd_ret"] = close.shift(-forward_horizon) / close - 1.0

        split_idx = max(12, int(len(ordered) * train_ratio))
        split_idx = min(split_idx, len(ordered) - 10)
        if split_idx < 10 or len(ordered) - split_idx < 10:
            continue
        train_frames.append(ordered.iloc[:split_idx].copy())
        oos_frames.append(ordered.iloc[split_idx:].copy())

    train_panel = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    oos_panel = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    return (
        cast("pd.DataFrame", train_panel),
        cast("pd.DataFrame", oos_panel),
        sample_df,
    )


def _collect_factor_lineage(
    strategies: Sequence[Strategy],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Map dynamic indicator names -> referencing strategy IDs & family IDs."""
    strat_map: dict[str, set[str]] = defaultdict(set)
    fam_map: dict[str, set[str]] = defaultdict(set)
    for strat in strategies:
        conds = (
            strat.entry.triggers
            + strat.entry.conditions
            + strat.entry.guards
            + strat.entry.filters
            + strat.exit.triggers
        )
        stop_conds = getattr(strat.exit.stop_loss, "conditions", None) or []
        for cond in [*conds, *stop_conds]:
            ind = getattr(cond, "indicator", "")
            if not ind:
                continue
            strat_map[ind].add(strat.meta.id)
            fam_map[ind].add(strat.meta.family_id)
    return (
        {k: sorted(v) for k, v in strat_map.items()},
        {k: sorted(v) for k, v in fam_map.items()},
    )


class FactorGovernanceEngine:
    """Audits persisted Alpha Factory factors on OOS IC/IR, PIT invariance, and redundancy."""

    def __init__(
        self,
        *,
        db_path: str | Path = ":memory:",
        config: FactorGovernanceConfig | None = None,
        sandbox_timeout: int = 15,
    ) -> None:
        self.db_path = db_path
        self.config = config or FactorGovernanceConfig()
        self._sandbox = FactorSandbox(timeout_seconds=sandbox_timeout)
        self._validator = FactorValidator(
            ValidationThresholds(
                min_ic_abs=self.config.min_oos_ic,
                min_ir=self.config.min_oos_ir,
            )
        )

    def audit_factors(
        self,
        market_data: dict[str, pd.DataFrame],
        *,
        records: Sequence[FactorRecord] | None = None,
        strategies: Sequence[Strategy] | None = None,
        auto_retire: bool = False,
    ) -> FactorGovernanceReport:
        """Audit factors across PIT safety, OOS predictive decay, and pairwise redundancy."""
        store = FactorStore(self.db_path)
        try:
            target_records = (
                list(records) if records is not None else store.list_all(status="active")
            )
            if not target_records:
                return FactorGovernanceReport(auto_retire_applied=auto_retire)

            loaded_strategies = self._resolve_strategies(strategies)
            strat_lineage, fam_lineage = _collect_factor_lineage(loaded_strategies)
            train_panel, oos_panel, sample_df = _split_panel(
                market_data,
                train_ratio=self.config.train_ratio,
                forward_horizon=self.config.forward_horizon,
            )

            oos_series_by_name: dict[str, pd.Series] = {}
            prelim: list[dict[str, object]] = []

            for rec in target_records:
                reasons: list[str] = []
                pit_ok = True
                if self.config.require_pit_invariance and sample_df is not None:
                    pit_res = probe_point_in_time_invariance(rec.code, sample_df)
                    pit_ok = pit_res.passed
                    if not pit_ok:
                        if pit_res.error:
                            reasons.append(f"PIT check error: {pit_res.error}")
                        elif pit_res.violations:
                            v = pit_res.violations[0]
                            reasons.append(
                                f"PIT violation ({v.probe_type}) at idx={v.probe_index}: "
                                f"diff={v.abs_diff:.4g}"
                            )
                        else:
                            reasons.append("Failed point-in-time invariance probe")

                train_ic, train_ir = 0.0, 0.0
                oos_ic, oos_ir, turnover = 0.0, 0.0, 0.0
                exec_ok = True

                if train_panel.empty or oos_panel.empty:
                    exec_ok = False
                    reasons.append("Market panel too short to split train/OOS slices")
                else:
                    tr_vals, tr_err = _execute_factor_series(
                        rec.code, train_panel, self._sandbox, verify_pit=False
                    )
                    te_vals, te_err = _execute_factor_series(
                        rec.code, oos_panel, self._sandbox, verify_pit=False
                    )
                    if tr_vals is None or te_vals is None:
                        exec_ok = False
                        reasons.append(f"Sandbox execution failed: {te_err or tr_err}")
                    else:
                        train_ic, train_ir = self._validator._compute_ic(
                            tr_vals,
                            train_panel["_fwd_ret"],
                            dates=train_panel["date"] if "date" in train_panel.columns else None,
                            symbols=train_panel["_symbol"],
                        )
                        oos_ic, oos_ir = self._validator._compute_ic(
                            te_vals,
                            oos_panel["_fwd_ret"],
                            dates=oos_panel["date"] if "date" in oos_panel.columns else None,
                            symbols=oos_panel["_symbol"],
                        )
                        turnover = self._validator._compute_turnover(
                            te_vals,
                            dates=oos_panel["date"] if "date" in oos_panel.columns else None,
                            symbols=oos_panel["_symbol"],
                        )
                        oos_series_by_name[rec.name] = te_vals

                sign = 1.0 if rec.expected_direction == "positive" else -1.0
                dir_train_ic = train_ic * sign
                dir_oos_ic = oos_ic * sign
                dir_oos_ir = oos_ir * sign
                decay_ratio: float | None = None
                if dir_train_ic > 1e-6:
                    decay_ratio = round(dir_oos_ic / dir_train_ic, 4)

                prelim.append(
                    {
                        "record": rec,
                        "pit_ok": pit_ok,
                        "exec_ok": exec_ok,
                        "train_ic": round(train_ic, 4),
                        "train_ir": round(train_ir, 4),
                        "oos_ic": round(oos_ic, 4),
                        "oos_ir": round(oos_ir, 4),
                        "dir_oos_ic": dir_oos_ic,
                        "dir_oos_ir": dir_oos_ir,
                        "decay_ratio": decay_ratio,
                        "turnover": round(turnover, 4),
                        "reasons": reasons,
                    }
                )

            # Evaluate redundancy against higher-ranked healthy peers (greedy by descending OOS IR)
            ranked = sorted(
                prelim,
                key=lambda item: float(cast("float", item["dir_oos_ir"])),
                reverse=True,
            )
            kept_healthy_names: list[str] = []
            entries: list[FactorAuditEntry] = []

            for item in ranked:
                rec = cast("FactorRecord", item["record"])
                pit_ok = bool(item["pit_ok"])
                exec_ok = bool(item["exec_ok"])
                reasons = list(cast("list[str]", item["reasons"]))
                dir_oos_ic = float(cast("float", item["dir_oos_ic"]))
                dir_oos_ir = float(cast("float", item["dir_oos_ir"]))
                decay_val = cast("float | None", item["decay_ratio"])

                max_corr = 0.0
                best_peer: str | None = None
                cur_series = oos_series_by_name.get(rec.name)
                if cur_series is not None:
                    for peer_name in kept_healthy_names:
                        peer_series = oos_series_by_name.get(peer_name)
                        if peer_series is None:
                            continue
                        corr = self._validator._rank_correlation(cur_series, peer_series)
                        if np.isfinite(corr) and abs(corr) > abs(max_corr):
                            max_corr = round(abs(float(corr)), 4)
                            best_peer = peer_name

                health: FactorHealthStatus
                if not pit_ok:
                    health = "pit_unsafe"
                elif not exec_ok:
                    health = "execution_failed"
                else:
                    decay_failed = False
                    if dir_oos_ic < self.config.min_oos_ic:
                        decay_failed = True
                        reasons.append(
                            f"Directional OOS IC {dir_oos_ic:.4f} < {self.config.min_oos_ic:.4f}"
                        )
                    if dir_oos_ir < self.config.min_oos_ir:
                        decay_failed = True
                        reasons.append(
                            f"Directional OOS IR {dir_oos_ir:.4f} < {self.config.min_oos_ir:.4f}"
                        )
                    if decay_val is not None and decay_val < self.config.min_decay_ratio:
                        decay_failed = True
                        reasons.append(
                            f"IC decay ratio {decay_val:.2f}x < {self.config.min_decay_ratio:.2f}x"
                        )
                    if decay_failed:
                        health = "decayed"
                    elif max_corr > self.config.max_redundancy_corr and best_peer is not None:
                        health = "redundant"
                        reasons.append(
                            f"High OOS rank correlation {max_corr:.2f} with stronger factor '{best_peer}'"
                        )
                    else:
                        health = "healthy"
                        kept_healthy_names.append(rec.name)

                action: Literal["kept", "retired", "would_retire"] = "kept"
                if health != "healthy":
                    if auto_retire:
                        if store.get(rec.name) is not None:
                            store.retire(rec.name)
                        action = "retired"
                    else:
                        action = "would_retire"

                entries.append(
                    FactorAuditEntry(
                        name=rec.name,
                        status_before=rec.status,
                        expected_direction=rec.expected_direction,
                        health=health,
                        pit_passed=pit_ok,
                        train_ic=float(cast("float", item["train_ic"])),
                        train_ir=float(cast("float", item["train_ir"])),
                        oos_ic=float(cast("float", item["oos_ic"])),
                        oos_ir=float(cast("float", item["oos_ir"])),
                        ic_decay_ratio=decay_val,
                        turnover=float(cast("float", item["turnover"])),
                        max_peer_corr=max_corr,
                        most_correlated_peer=best_peer,
                        referenced_strategies=strat_lineage.get(rec.name, []),
                        referenced_families=fam_lineage.get(rec.name, []),
                        reasons=reasons,
                        action_taken=action,
                    )
                )

            return FactorGovernanceReport(
                total_audited=len(entries),
                healthy_count=sum(1 for e in entries if e.health == "healthy"),
                decayed_count=sum(1 for e in entries if e.health == "decayed"),
                redundant_count=sum(1 for e in entries if e.health == "redundant"),
                pit_unsafe_count=sum(1 for e in entries if e.health == "pit_unsafe"),
                execution_failed_count=sum(1 for e in entries if e.health == "execution_failed"),
                retired_count=sum(1 for e in entries if e.action_taken == "retired"),
                auto_retire_applied=auto_retire,
                entries=entries,
            )
        finally:
            store.close()

    def _resolve_strategies(self, strategies: Sequence[Strategy] | None) -> list[Strategy]:
        if strategies is not None:
            return list(strategies)
        resolved: dict[str, Strategy] = {}
        try:
            strat_store = StrategyStore(self.db_path)
            for s in strat_store.list_all():
                resolved[s.meta.id] = s
        except Exception:
            pass
        try:
            from alphaevo.cli.demo import _find_builtin_strategy_dir

            bdir = _find_builtin_strategy_dir()
            if bdir is not None:
                parser = StrategyParser()
                for s in parser.parse_directory(bdir):
                    resolved.setdefault(s.meta.id, s)
        except Exception:
            pass
        return list(resolved.values())

    @staticmethod
    def render_markdown(report: FactorGovernanceReport) -> str:
        """Render factor governance audit summary into Markdown."""
        lines = [
            "# Alpha Factory Governance Audit Report",
            "",
            f"- Audited Factors: {report.total_audited}",
            f"- Healthy: {report.healthy_count}",
            f"- Decayed: {report.decayed_count}",
            f"- Redundant: {report.redundant_count}",
            f"- PIT Unsafe: {report.pit_unsafe_count}",
            f"- Execution Failed: {report.execution_failed_count}",
            f"- Retired: {report.retired_count} (auto_retire={report.auto_retire_applied})",
            "",
            "| Factor | Health | PIT | Train IC | OOS IC | OOS IR | Decay | MaxCorr | Action | Referenced By |",
            "|--------|--------|-----|----------|--------|--------|-------|---------|--------|---------------|",
        ]
        for e in report.entries:
            decay_str = f"{e.ic_decay_ratio:.2f}x" if e.ic_decay_ratio is not None else "n/a"
            refs = ", ".join(e.referenced_strategies[:3]) if e.referenced_strategies else "—"
            lines.append(
                f"| `{e.name}` | {e.health} | {'PASS' if e.pit_passed else 'FAIL'} "
                f"| {e.train_ic:+.4f} | {e.oos_ic:+.4f} | {e.oos_ir:+.2f} "
                f"| {decay_str} | {e.max_peer_corr:.2f} | {e.action_taken} | {refs} |"
            )
            for reason in e.reasons:
                lines.append(f"  - `{e.name}` finding: {reason}")
        lines += [
            "",
            "---",
            "*Research & educational software only. Factor governance audit does not constitute investment advice.*",
            "",
        ]
        return "\n".join(lines)


__all__ = [
    "FactorAuditEntry",
    "FactorGovernanceConfig",
    "FactorGovernanceEngine",
    "FactorGovernanceReport",
    "FactorHealthStatus",
]
