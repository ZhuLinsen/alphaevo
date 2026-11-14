"""Core Dual-Clock Stock RSI Engine (`autoresearch` keep/revert ratchet + L1/L2 + L0/QD)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.models.enums import EvidenceLevel, RSILevel, SamplingMethod
from alphaevo.models.execution import EvaluationReport, SampleBatch, StrategyChange
from alphaevo.models.strategy import Strategy
from alphaevo.reflection.critic import SelfCritic
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore
from alphaevo.reflection.mutator import StrategyMutator
from alphaevo.rsi.co_steer import BlindSpotMiner, ScarRegistry
from alphaevo.rsi.council import HeterogeneousCouncil
from alphaevo.rsi.program import RSICharter
from alphaevo.rsi.qd_archive import ArchiveSummary, MAPElitesArchive
from alphaevo.strategy.dsl.serializer import StrategySerializer
from alphaevo.truth.ledger import (
    TrialLedger,
    compute_data_fingerprint,
    compute_kernel_fingerprint,
)
from alphaevo.truth.null_control import NullControlReport, NullSignalController


class RSIRoundRow(BaseModel):
    """One row of the Karpathy-style ``results.tsv`` RSI experiment log."""

    round_num: int
    strategy_id: str
    level: RSILevel
    signals: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    dsr_probability: float
    null_threshold: float
    null_passed: bool
    fitness_score: float
    status: Literal["baseline", "keep", "stepping_stone", "revert", "vetoed"]
    description: str


class RSIRunResult(BaseModel):
    """Complete verifiable output of one Stock RSI evolution session."""

    charter_name: str
    trial_family_id: str
    seed_strategy_id: str
    champion_id: str
    champion_strategy: Strategy
    champion_evaluation: EvaluationReport
    total_trials_recorded: int
    kept_rounds: int
    promotions: int
    kernel_hash_before: str
    kernel_hash_after: str
    kernel_intact: bool
    rows: list[RSIRoundRow] = Field(default_factory=list)
    archive_summary: ArchiveSummary
    scars_created: int = 0
    tsv_path: Path | None = None
    report_path: Path | None = None
    champion_yaml_path: Path | None = None


def _compute_rsi_fitness(
    evaluation: EvaluationReport,
    *,
    objective: Literal["robust_expectancy", "confidence", "deflated_sharpe"],
    min_signals: int,
    max_drawdown: float,
) -> float:
    """Compute bounded fitness score penalizing sparsity, overfit, and tail risk."""
    ov = evaluation.overall
    if ov.signal_count == 0:
        return 0.01

    dsr = float(evaluation.deflated_sharpe_prob or 0.5)
    if objective == "confidence":
        raw = evaluation.confidence_score
    elif objective == "deflated_sharpe":
        raw = 0.5 * dsr + 0.3 * min(1.0, max(0.0, ov.sharpe_ratio / 2.5)) + 0.2 * ov.win_rate
    else:
        wr = min(1.0, ov.win_rate / 0.65)
        ret = min(1.0, max(0.0, (ov.avg_return + 0.01) / 0.06))
        pl = min(1.0, ov.profit_loss_ratio / 2.5)
        dd = max(0.0, 1.0 - ov.max_drawdown / max(0.05, max_drawdown))
        raw = 0.25 * wr + 0.30 * ret + 0.20 * dsr + 0.15 * pl + 0.10 * dd

    if ov.signal_count < min_signals:
        raw *= ov.signal_count / max(1, min_signals)
    if ov.max_drawdown > max_drawdown:
        raw *= 0.65
    if evaluation.anti_overfit.is_overfit:
        raw *= 0.70
    if evaluation.null_control_passed is False:
        raw *= 0.80

    return round(max(0.0, min(1.0, raw)), 6)


def _structural_strategy_hash(strategy: Strategy) -> str:
    """Hash strategy structure excluding mutable identity metadata (id/version/parent_id)."""
    payload = strategy.model_dump(mode="json", exclude={"meta", "complexity_score"})
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _date_bounds(market_data: dict[str, pd.DataFrame]) -> tuple[date, date]:
    starts: list[date] = []
    ends: list[date] = []
    for df in market_data.values():
        if df.empty or "date" not in df:
            continue
        ds = pd.to_datetime(df["date"]).dt.date
        starts.append(ds.min())
        ends.append(ds.max())
    if not starts or not ends:
        today = date.today()
        return today, today
    return min(starts), max(ends)


class StockRSIEngine:
    """Self-improving stock strategy/factor loop guarded by L0 Truth & MAP-Elites."""

    def __init__(
        self,
        *,
        db_path: str | Path = "~/.alphaevo/alphaevo.db",
        complexity_limit: int = 8,
        enable_pinned_scars: bool = True,
        council: HeterogeneousCouncil | None = None,
    ) -> None:
        self._trial_ledger = TrialLedger(db_path)
        self._experience_store = ExperienceStore(db_path)
        self._scar_registry = ScarRegistry()
        self._enable_pinned_scars = enable_pinned_scars
        self._critic = SelfCritic(
            experience_store=self._experience_store,
            complexity_limit=complexity_limit,
        )
        self._council = council or HeterogeneousCouncil(critic=self._critic)
        self._miner = BlindSpotMiner()
        self._null_ctrl = NullSignalController()
        self._bt_engine = BacktestEngine(min_data_days=15)
        self._evaluator = Evaluator()

    def run(
        self,
        seed_strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        *,
        charter: RSICharter | None = None,
        output_dir: Path | None = None,
        seed: int = 42,
    ) -> RSIRunResult:
        """Execute an bounded L1/L2 RSI evolution session and persist traceable outputs."""
        cfg = charter or RSICharter(seed_strategy_id=seed_strategy.meta.id)
        kernel_before = compute_kernel_fingerprint()
        data_fp = compute_data_fingerprint(market_data)
        family_id = self._trial_ledger.resolve_trial_family_id(seed_strategy)
        symbols = sorted(market_data.keys())
        date_range = _date_bounds(market_data)
        sample_df = next(
            (df for df in market_data.values() if not df.empty and len(df) >= 15),
            None,
        )

        mutator = StrategyMutator(max_changes=3, complexity_limit=8)
        archive = MAPElitesArchive()

        # Baseline evaluation (Round 0)
        base_eval, base_null, base_bt_signals = self._eval_with_l0(
            strategy=seed_strategy,
            market_data=market_data,
            symbols=symbols,
            date_range=date_range,
            family_id=family_id,
            charter=cfg,
            seed=seed,
        )
        self._trial_ledger.record_trial(
            strategy=seed_strategy,
            trial_family_id=family_id,
            data_fingerprint=data_fp,
            rsi_level=RSILevel.L1_STRATEGY,
            round_num=0,
            signal_count=base_eval.overall.signal_count,
            sharpe_ratio=base_eval.overall.sharpe_ratio,
            returns=[s.return_pct for s in base_bt_signals],
            confidence_score=base_eval.confidence_score,
            passed_null_control=base_eval.null_control_passed,
            accepted=True,
            notes="baseline",
        )
        # Re-read updated family count onto report
        base_eval.family_trial_count = self._trial_ledger.count_family_trials(family_id)
        incumbent_strategy = seed_strategy.model_copy(deep=True)
        incumbent_eval = base_eval
        incumbent_fitness = _compute_rsi_fitness(
            base_eval,
            objective=cfg.objective,
            min_signals=cfg.min_signals,
            max_drawdown=cfg.max_drawdown,
        )
        archive.insert(incumbent_strategy, incumbent_eval, incumbent_fitness)

        rows: list[RSIRoundRow] = [
            RSIRoundRow(
                round_num=0,
                strategy_id=seed_strategy.meta.id,
                level=RSILevel.L1_STRATEGY,
                signals=base_eval.overall.signal_count,
                win_rate=base_eval.overall.win_rate,
                avg_return=base_eval.overall.avg_return,
                sharpe_ratio=base_eval.overall.sharpe_ratio,
                dsr_probability=float(base_eval.deflated_sharpe_prob or 0.0),
                null_threshold=base_null.worst_null_threshold if base_null else 0.0,
                null_passed=bool(base_eval.null_control_passed),
                fitness_score=incumbent_fitness,
                status="baseline",
                description="Initial seed strategy",
            )
        ]

        stagnant_l1 = 0
        kept_rounds = 0
        promotions = 0
        scars_created = 0
        next_version = seed_strategy.meta.version + 1
        seen_hashes: set[str] = {_structural_strategy_hash(seed_strategy)}

        for rnd in range(1, cfg.max_rounds + 1):
            if self._trial_ledger.count_family_trials(family_id) >= cfg.max_trials_budget:
                break

            if self._enable_pinned_scars:
                self._council.set_scars(self._scar_registry.list_all(family_id))

            steer = self._miner.steer(
                incumbent_strategy,
                incumbent_eval,
                sample_df,
                allowed_levels=cfg.allowed_levels,
                stagnant_l1_rounds=stagnant_l1,
            )

            deliberations = self._council.generate_candidates(
                incumbent_strategy,
                incumbent_eval,
                extra_changes=[steer.proposed_changes] if steer.proposed_changes else None,
                mission_notes=cfg.mission_notes,
                max_candidates=cfg.max_candidates_per_round,
            )
            approved = [d for d in deliberations if d.audit.approved]
            if not approved:
                rows.append(
                    RSIRoundRow(
                        round_num=rnd,
                        strategy_id=incumbent_strategy.meta.id,
                        level=steer.chosen_level,
                        signals=incumbent_eval.overall.signal_count,
                        win_rate=incumbent_eval.overall.win_rate,
                        avg_return=incumbent_eval.overall.avg_return,
                        sharpe_ratio=incumbent_eval.overall.sharpe_ratio,
                        dsr_probability=float(incumbent_eval.deflated_sharpe_prob or 0.0),
                        null_threshold=0.0,
                        null_passed=bool(incumbent_eval.null_control_passed),
                        fitness_score=incumbent_fitness,
                        status="vetoed",
                        description="All proposals blocked by Adversarial Prosecutor / Pinned Scars",
                    )
                )
                stagnant_l1 += 1
                continue

            best_cand_bundle: (
                tuple[
                    Strategy,
                    EvaluationReport,
                    NullControlReport | None,
                    float,
                    list[StrategyChange],
                    RSILevel,
                ]
                | None
            ) = None

            for delib in approved:
                if self._trial_ledger.count_family_trials(family_id) >= cfg.max_trials_budget:
                    break
                changes = delib.proposal.changes
                level_used = (
                    RSILevel.L2_FACTOR
                    if any(c.change_type.value == "discover_factor" for c in changes)
                    or (
                        steer.chosen_level == RSILevel.L2_FACTOR
                        and changes == steer.proposed_changes
                    )
                    else RSILevel.L1_STRATEGY
                )
                try:
                    cand_strategy = mutator.mutate(incumbent_strategy, changes, atomic=True)
                except Exception:
                    continue

                cand_id = f"{family_id}_rsi_v{next_version}"
                cand_strategy.meta.id = cand_id
                cand_strategy.meta.version = next_version
                cand_strategy.meta.parent_id = incumbent_strategy.meta.id

                shash = _structural_strategy_hash(cand_strategy)
                if shash in seen_hashes:
                    continue
                seen_hashes.add(shash)

                cand_eval, cand_null, cand_bt_sigs = self._eval_with_l0(
                    strategy=cand_strategy,
                    market_data=market_data,
                    symbols=symbols,
                    date_range=date_range,
                    family_id=family_id,
                    charter=cfg,
                    seed=seed + rnd,
                )
                self._trial_ledger.record_trial(
                    strategy=cand_strategy,
                    trial_family_id=family_id,
                    data_fingerprint=data_fp,
                    rsi_level=level_used,
                    round_num=rnd,
                    signal_count=cand_eval.overall.signal_count,
                    sharpe_ratio=cand_eval.overall.sharpe_ratio,
                    returns=[s.return_pct for s in cand_bt_sigs],
                    confidence_score=cand_eval.confidence_score,
                    passed_null_control=cand_eval.null_control_passed,
                    accepted=False,
                    notes=changes[0].reason[:120],
                )
                cand_fit = _compute_rsi_fitness(
                    cand_eval,
                    objective=cfg.objective,
                    min_signals=cfg.min_signals,
                    max_drawdown=cfg.max_drawdown,
                )

                archive.insert(cand_strategy, cand_eval, cand_fit)

                if self._enable_pinned_scars and (
                    (
                        incumbent_eval.overall.signal_count > 0
                        and cand_eval.overall.signal_count == 0
                    )
                    or (cand_fit - incumbent_fitness) < -0.08
                ):
                    new_scars = self._scar_registry.synthesize_from_failure(
                        strategy=cand_strategy,
                        changes=changes,
                        signal_count=cand_eval.overall.signal_count,
                        score_delta=cand_fit - incumbent_fitness,
                    )
                    scars_created += len(new_scars)

                ev_level = (
                    EvidenceLevel.NULL_PASSED
                    if cand_eval.null_control_passed is True
                    else EvidenceLevel.BACKTEST
                )
                self._experience_store.record(
                    ExperienceRecord(
                        strategy_family=family_id,
                        strategy_id=cand_id,
                        round_num=rnd,
                        change_type=changes[0].change_type,
                        target=changes[0].target,
                        from_value=changes[0].from_value,
                        to_value=changes[0].to_value,
                        reason=changes[0].reason,
                        score_before=incumbent_fitness,
                        score_after=cand_fit,
                        score_delta=round(cand_fit - incumbent_fitness, 4),
                        worked=cand_fit > incumbent_fitness,
                        lesson=f"{level_used.value}: {changes[0].reason}",
                        action_type="rsi_engine",
                        evidence_level=ev_level,
                    )
                )

                if best_cand_bundle is None or cand_fit > best_cand_bundle[3]:
                    best_cand_bundle = (
                        cand_strategy,
                        cand_eval,
                        cand_null,
                        cand_fit,
                        changes,
                        level_used,
                    )
                next_version += 1

            if best_cand_bundle is None:
                stagnant_l1 += 1
                continue

            cand_strat, cand_eval, cand_null, cand_fit, chs, lvl = best_cand_bundle
            passes_gate = (
                cand_fit > incumbent_fitness + 1e-4
                and cand_eval.overall.avg_return > 0.0
                and cand_eval.overall.max_drawdown <= cfg.max_drawdown
                and not cand_eval.anti_overfit.is_overfit
                and cand_eval.overall.signal_count >= min(5, cfg.min_signals)
                and (not cfg.require_null_control or cand_eval.null_control_passed is True)
            )
            is_stepping_stone = (
                not passes_gate
                and promotions == 0
                and cand_eval.overall.signal_count >= min(5, cfg.min_signals)
                and cand_fit > incumbent_fitness + 1e-4
                and (
                    incumbent_eval.overall.signal_count < 5
                    or (
                        cand_eval.overall.max_drawdown
                        <= max(cfg.max_drawdown, incumbent_eval.overall.max_drawdown)
                        and cand_eval.overall.avg_return >= incumbent_eval.overall.avg_return
                    )
                )
            )
            desc = "; ".join(f"{c.target}->{c.to_value}" for c in chs)

            if passes_gate:
                incumbent_strategy = cand_strat
                incumbent_eval = cand_eval
                incumbent_fitness = cand_fit
                kept_rounds += 1
                promotions += 1
                stagnant_l1 = 0
                status: Literal["keep", "stepping_stone", "revert"] = "keep"
            elif is_stepping_stone:
                # Escape a 0-signal dead zone without counting as an L0 promotion.
                incumbent_strategy = cand_strat
                incumbent_eval = cand_eval
                incumbent_fitness = cand_fit
                kept_rounds += 1
                stagnant_l1 = 0
                status = "stepping_stone"
            else:
                stagnant_l1 += 1
                status = "revert"

            rows.append(
                RSIRoundRow(
                    round_num=rnd,
                    strategy_id=cand_strat.meta.id,
                    level=lvl,
                    signals=cand_eval.overall.signal_count,
                    win_rate=cand_eval.overall.win_rate,
                    avg_return=cand_eval.overall.avg_return,
                    sharpe_ratio=cand_eval.overall.sharpe_ratio,
                    dsr_probability=float(cand_eval.deflated_sharpe_prob or 0.0),
                    null_threshold=cand_null.worst_null_threshold if cand_null else 0.0,
                    null_passed=bool(cand_eval.null_control_passed),
                    fitness_score=cand_fit,
                    status=status,
                    description=desc,
                )
            )

        kernel_after = compute_kernel_fingerprint()
        result = RSIRunResult(
            charter_name=cfg.name,
            trial_family_id=family_id,
            seed_strategy_id=seed_strategy.meta.id,
            champion_id=incumbent_strategy.meta.id,
            champion_strategy=incumbent_strategy,
            champion_evaluation=incumbent_eval,
            total_trials_recorded=self._trial_ledger.count_family_trials(family_id),
            kept_rounds=kept_rounds,
            promotions=promotions,
            kernel_hash_before=kernel_before,
            kernel_hash_after=kernel_after,
            kernel_intact=(kernel_before == kernel_after),
            rows=rows,
            archive_summary=archive.coverage_summary(),
            scars_created=scars_created,
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            tsv_path = output_dir / "results.tsv"
            report_path = output_dir / "rsi_run_report.md"
            yaml_path = output_dir / f"{incumbent_strategy.meta.id}.yaml"
            tsv_path.write_text(self.render_tsv(result), encoding="utf-8")
            report_path.write_text(self.render_markdown(result), encoding="utf-8")
            StrategySerializer().to_file(incumbent_strategy, yaml_path)
            result.tsv_path = tsv_path
            result.report_path = report_path
            result.champion_yaml_path = yaml_path

        return result

    def _eval_with_l0(
        self,
        *,
        strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        symbols: list[str],
        date_range: tuple[date, date],
        family_id: str,
        charter: RSICharter,
        seed: int,
    ) -> tuple[EvaluationReport, NullControlReport | None, list[Any]]:
        batch = SampleBatch(
            batch_id="rsi_eval",
            strategy_id=strategy.meta.id,
            symbols=symbols,
            date_range=date_range,
            sampling_method=SamplingMethod.REPRESENTATIVE,
        )
        bt = self._bt_engine.run(strategy, market_data, batch)
        ev = self._evaluator.evaluate_fast(strategy=strategy, result=bt)
        executed = [s for s in bt.signals if s.exit_price is not None]

        # Compute DSR using accumulated trial count + current trial
        trial_n = max(1, self._trial_ledger.count_family_trials(family_id) + 1)
        dsr_res = self._trial_ledger.compute_dsr_for_trials(
            family_id=family_id,
            n_trials=trial_n,
            returns=[s.return_pct for s in executed],
            observed_sharpe=ev.overall.sharpe_ratio,
            signal_count=ev.overall.signal_count,
        )
        ev.family_trial_count = trial_n
        ev.deflated_sharpe_prob = dsr_res.dsr_probability

        null_rep: NullControlReport | None = None
        if charter.require_null_control:
            if len(executed) < min(5, charter.min_signals) or ev.overall.avg_return <= 0.0:
                ev.null_control_passed = False
                ev.null_p_value = 1.0
            else:
                null_rep = self._null_ctrl.evaluate_candidate_against_nulls(
                    bt.signals,
                    market_data=market_data,
                    n_permutations=charter.null_permutations,
                    min_signals=min(5, charter.min_signals),
                    seed=seed,
                )
                ev.null_control_passed = null_rep.passed
                ev.null_p_value = null_rep.worst_p_value
        else:
            ev.null_control_passed = None
        return ev, null_rep, executed

    @staticmethod
    def render_tsv(result: RSIRunResult) -> str:
        """Render Karpathy autoresearch-compatible ``results.tsv`` ledger."""
        header = (
            "round\tstrategy_id\tlevel\tsignals\twin_rate\tavg_return\t"
            "sharpe\tdsr_prob\tnull_p95\tnull_passed\tfitness\tstatus\tdescription"
        )
        lines = [header]
        for r in result.rows:
            lines.append(
                f"{r.round_num}\t{r.strategy_id}\t{r.level.value}\t{r.signals}\t"
                f"{r.win_rate:.4f}\t{r.avg_return:.6f}\t{r.sharpe_ratio:.4f}\t"
                f"{r.dsr_probability:.4f}\t{r.null_threshold:.6f}\t{r.null_passed}\t"
                f"{r.fitness_score:.6f}\t{r.status}\t{r.description}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def render_markdown(result: RSIRunResult) -> str:
        """Render full verifiable Stock RSI Markdown summary."""
        lines = [
            f"# AlphaEvo Stock RSI Report: {result.charter_name}",
            "",
            f"- Seed Strategy: `{result.seed_strategy_id}`",
            f"- Champion Strategy: `{result.champion_id}`",
            f"- Trial Family ID: `{result.trial_family_id}` (Total Trials Counted: {result.total_trials_recorded})",
            f"- L0 Truth Kernel Intact: **{result.kernel_intact}** (`{result.kernel_hash_before}`)",
            f"- Kept Rounds: {result.kept_rounds} | Promotions (L0-Verified): {result.promotions} | Distilled Pinned Scars: {result.scars_created}",
            f"- MAP-Elites Niches Occupied: {result.archive_summary.total_cells_occupied}",
            "",
            "## Evolution Ratchet Ledger",
            "",
            "| Round | Strategy | Level | Signals | Win Rate | Avg Return | DSR | Null Gate | Fitness | Status | Change |",
            "|-------|----------|-------|---------|----------|------------|-----|-----------|---------|--------|--------|",
        ]
        for r in result.rows:
            lines.append(
                f"| {r.round_num} | `{r.strategy_id}` | {r.level.value} | {r.signals} | "
                f"{r.win_rate:.1%} | {r.avg_return:.2%} | {r.dsr_probability:.1%} | "
                f"{'PASS' if r.null_passed else 'FAIL'} | {r.fitness_score:.1%} | "
                f"**{r.status.upper()}** | {r.description} |"
            )
        lines += [
            "",
            "> Research & educational use only. Past simulated performance does not guarantee future returns.",
        ]
        return "\n".join(lines) + "\n"


__all__ = ["RSIRoundRow", "RSIRunResult", "StockRSIEngine"]
