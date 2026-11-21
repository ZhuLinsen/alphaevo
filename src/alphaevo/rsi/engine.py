"""Core Dual-Clock Stock RSI Engine (`autoresearch` keep/revert ratchet + L1/L2 + L0/QD)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from alphaevo.alpha_factory.factor_store import FactorStore
from alphaevo.alpha_factory.factory import (
    load_registered_factors,
    register_factor_record,
)
from alphaevo.core.config import BacktestConfig
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.models.enums import EvidenceLevel, RSILevel
from alphaevo.models.execution import EvaluationReport, StrategyChange
from alphaevo.models.strategy import Strategy
from alphaevo.reflection.critic import SelfCritic
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore
from alphaevo.reflection.mutator import StrategyMutator
from alphaevo.rsi.co_steer import BlindSpotMiner, ScarRegistry
from alphaevo.rsi.council import HeterogeneousCouncil
from alphaevo.rsi.program import RSICharter
from alphaevo.rsi.qd_archive import ArchiveSummary, MAPElitesArchive
from alphaevo.strategy.dsl.serializer import StrategySerializer
from alphaevo.strategy.store import StrategyStore
from alphaevo.truth.kernel import (
    EvaluationKernel,
    KernelEvaluationOutcome,
    LockedHoldoutVault,
    PortfolioGateConfig,
    PortfolioGateVerdict,
    split_market_data_with_holdout,
)
from alphaevo.truth.ledger import (
    TrialLedger,
    compute_kernel_fingerprint,
)
from alphaevo.truth.null_control import NullSignalController

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.forward.shadow_ledger import ShadowLedger
    from alphaevo.provenance.store import RunArtifactStore


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
    champion_promoted: bool = False
    champion_strategy: Strategy
    champion_evaluation: EvaluationReport
    working_strategy: Strategy | None = None
    working_evaluation: EvaluationReport | None = None
    total_trials_recorded: int
    kept_rounds: int
    promotions: int
    kernel_hash_before: str
    kernel_hash_after: str
    kernel_intact: bool
    rows: list[RSIRoundRow] = Field(default_factory=list)
    archive_summary: ArchiveSummary
    scars_created: int = 0
    holdout_evaluation: EvaluationReport | None = None
    portfolio_verdict: PortfolioGateVerdict | None = None
    enrolled_in_shadow: bool = False
    persisted_factors: list[str] = Field(default_factory=list)
    run_id: str | None = None
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

    dsr = (
        float(evaluation.deflated_sharpe_prob)
        if evaluation.deflated_sharpe_prob is not None
        else 0.0
    )
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


class StockRSIEngine:
    """Self-improving stock strategy/factor loop guarded by L0 Truth & MAP-Elites."""

    def __init__(
        self,
        *,
        db_path: str | Path = "~/.alphaevo/alphaevo.db",
        complexity_limit: int = 8,
        enable_pinned_scars: bool = True,
        council: HeterogeneousCouncil | None = None,
        backtest_config: BacktestConfig | None = None,
    ) -> None:
        self._db_path = db_path
        self._backtest_config = backtest_config or BacktestConfig(min_data_days=15)
        self._trial_ledger = TrialLedger(db_path)
        self._experience_store = ExperienceStore(db_path)
        self._factor_store = FactorStore(db_path)
        if str(db_path) != ":memory:":
            with contextlib.suppress(Exception):
                load_registered_factors(db_path, status="active")
        else:
            for rec in self._factor_store.list_all(status="active"):
                register_factor_record(rec)
        scar_path = (
            Path(db_path).expanduser().with_suffix(".scars.jsonl")
            if str(db_path) != ":memory:"
            else None
        )
        self._scar_registry = ScarRegistry(path=scar_path)
        self._enable_pinned_scars = enable_pinned_scars
        self._critic = SelfCritic(
            experience_store=self._experience_store,
            complexity_limit=complexity_limit,
        )
        self._council = council or HeterogeneousCouncil(critic=self._critic)
        self._miner = BlindSpotMiner(factor_store=self._factor_store)
        self._null_ctrl = NullSignalController()
        self._evaluator = Evaluator()
        self._kernel = EvaluationKernel(
            backtest_config=self._backtest_config,
            db_path=None,
            trial_ledger=self._trial_ledger,
            null_controller=self._null_ctrl,
            min_data_days=15,
        )

    def run(
        self,
        seed_strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        *,
        charter: RSICharter | None = None,
        output_dir: Path | None = None,
        seed: int = 42,
        persist_champion: bool = True,
        shadow_ledger: ShadowLedger | None = None,
        auto_enroll_shadow: bool = False,
        artifact_store: RunArtifactStore | None = None,
    ) -> RSIRunResult:
        """Execute a bounded L1/L2 RSI evolution session and persist traceable outputs."""
        cfg = charter or RSICharter(seed_strategy_id=seed_strategy.meta.id)
        kernel_before = compute_kernel_fingerprint()

        vault: LockedHoldoutVault | None = None
        search_market = market_data
        if cfg.locked_holdout_pct > 0.0:
            try:
                vault, _split = split_market_data_with_holdout(
                    market_data,
                    holdout_pct=cfg.locked_holdout_pct,
                )
                search_market = vault.search_data
            except ValueError:
                vault = None
                search_market = market_data

        family_id = self._trial_ledger.resolve_trial_family_id(seed_strategy)
        sample_df = next(
            (df for df in search_market.values() if not df.empty and len(df) >= 15),
            None,
        )

        mutator = StrategyMutator(max_changes=3, complexity_limit=cfg.max_complexity)
        archive = MAPElitesArchive()
        rng = random.Random(seed)
        portfolio_gate_cfg = (
            PortfolioGateConfig(
                enabled=True,
                max_portfolio_drawdown=cfg.max_portfolio_drawdown,
            )
            if cfg.require_portfolio_gate
            else None
        )

        # Baseline evaluation (Round 0) via canonical EvaluationKernel
        base_outcome = self._eval_with_kernel(
            strategy=seed_strategy,
            market_data=search_market,
            family_id=family_id,
            charter=cfg,
            seed=seed,
            portfolio_gate=portfolio_gate_cfg,
            rsi_level=RSILevel.L1_STRATEGY,
            round_num=0,
            accepted=True,
            notes="baseline",
        )
        base_eval = base_outcome.evaluation
        base_null = base_outcome.null_control
        working_strategy = seed_strategy.model_copy(deep=True)
        working_eval = base_eval
        working_fitness = _compute_rsi_fitness(
            base_eval,
            objective=cfg.objective,
            min_signals=cfg.min_signals,
            max_drawdown=cfg.max_drawdown,
        )
        verified_champion_strategy = seed_strategy.model_copy(deep=True)
        verified_champion_eval = base_eval
        verified_champion_fitness = working_fitness
        verified_champion_outcome = base_outcome
        has_verified_promotion = False
        archive.insert(working_strategy, working_eval, working_fitness)

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
                fitness_score=working_fitness,
                status="baseline",
                description="Initial seed strategy",
            )
        ]

        stagnant_l1 = 0
        kept_rounds = 0
        promotions = 0
        scars_created = 0
        persisted_factors: list[str] = []
        next_version = seed_strategy.meta.version + 1
        seen_hashes: set[str] = {_structural_strategy_hash(seed_strategy)}
        tried_by_parent: dict[str, set[tuple[tuple[str, str, str], ...]]] = {}

        for rnd in range(1, cfg.max_rounds + 1):
            if self._trial_ledger.count_family_trials(family_id) >= cfg.max_trials_budget:
                break

            if self._enable_pinned_scars:
                self._council.set_scars(self._scar_registry.list_all(family_id))

            parent_strategy = working_strategy
            parent_eval = working_eval
            if stagnant_l1 >= 1:
                elite = archive.sample_elite(exclude_id=working_strategy.meta.id, rng=rng)
                if elite is not None:
                    parent_strategy = elite.strategy
                    parent_eval = elite.evaluation

            steer = self._miner.steer(
                parent_strategy,
                parent_eval,
                sample_df,
                allowed_levels=cfg.allowed_levels,
                stagnant_l1_rounds=stagnant_l1,
            )
            if (
                steer.persisted_to_store
                and steer.synthesized_factor_name
                and steer.synthesized_factor_name not in persisted_factors
            ):
                persisted_factors.append(steer.synthesized_factor_name)

            parent_tried = tried_by_parent.setdefault(parent_strategy.meta.id, set())
            deliberations = self._council.generate_candidates(
                parent_strategy,
                parent_eval,
                extra_changes=[steer.proposed_changes] if steer.proposed_changes else None,
                mission_notes=cfg.mission_notes,
                max_candidates=cfg.max_candidates_per_round,
                exclude_sigs=parent_tried,
            )
            approved = [d for d in deliberations if d.audit.approved]
            if not approved:
                rows.append(
                    RSIRoundRow(
                        round_num=rnd,
                        strategy_id=parent_strategy.meta.id,
                        level=steer.chosen_level,
                        signals=parent_eval.overall.signal_count,
                        win_rate=parent_eval.overall.win_rate,
                        avg_return=parent_eval.overall.avg_return,
                        sharpe_ratio=parent_eval.overall.sharpe_ratio,
                        dsr_probability=float(parent_eval.deflated_sharpe_prob or 0.0),
                        null_threshold=0.0,
                        null_passed=bool(parent_eval.null_control_passed),
                        fitness_score=working_fitness,
                        status="vetoed",
                        description="All proposals blocked by Adversarial Prosecutor / Pinned Scars",
                    )
                )
                stagnant_l1 += 1
                continue

            best_cand_bundle: (
                tuple[
                    Strategy,
                    KernelEvaluationOutcome,
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
                parent_tried.add(
                    tuple(sorted((c.change_type.value, c.target, str(c.to_value)) for c in changes))
                )
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
                    cand_strategy = mutator.mutate(parent_strategy, changes, atomic=True)
                except Exception:
                    continue

                cand_id = f"{family_id}_v{next_version}"
                cand_strategy.meta.id = cand_id
                cand_strategy.meta.version = next_version
                cand_strategy.meta.parent_id = parent_strategy.meta.id

                shash = _structural_strategy_hash(cand_strategy)
                if shash in seen_hashes:
                    continue
                seen_hashes.add(shash)

                cand_outcome = self._eval_with_kernel(
                    strategy=cand_strategy,
                    market_data=search_market,
                    family_id=family_id,
                    charter=cfg,
                    seed=seed + rnd,
                    portfolio_gate=portfolio_gate_cfg,
                    rsi_level=level_used,
                    round_num=rnd,
                    accepted=False,
                    notes=changes[0].reason[:120],
                )
                cand_eval = cand_outcome.evaluation
                cand_fit = _compute_rsi_fitness(
                    cand_eval,
                    objective=cfg.objective,
                    min_signals=cfg.min_signals,
                    max_drawdown=cfg.max_drawdown,
                )

                archive.insert(cand_strategy, cand_eval, cand_fit)

                if self._enable_pinned_scars and (
                    (working_eval.overall.signal_count > 0 and cand_eval.overall.signal_count == 0)
                    or (cand_fit - working_fitness) < -0.08
                ):
                    new_scars = self._scar_registry.synthesize_from_failure(
                        strategy=cand_strategy,
                        changes=changes,
                        signal_count=cand_eval.overall.signal_count,
                        score_delta=cand_fit - working_fitness,
                    )
                    scars_created += len(new_scars)

                ev_level = (
                    EvidenceLevel.NULL_PASSED
                    if cand_eval.null_control_passed is True
                    else EvidenceLevel.BACKTEST
                )
                conf_delta = round(cand_eval.confidence_score - working_eval.confidence_score, 4)
                self._experience_store.record(
                    ExperienceRecord(
                        strategy_family=cand_strategy.meta.family_id,
                        strategy_id=cand_id,
                        round_num=rnd,
                        change_type=changes[0].change_type,
                        target=changes[0].target,
                        from_value=changes[0].from_value,
                        to_value=changes[0].to_value,
                        reason=changes[0].reason,
                        score_before=working_eval.confidence_score,
                        score_after=cand_eval.confidence_score,
                        score_delta=conf_delta,
                        worked=cand_fit > working_fitness,
                        lesson=f"{level_used.value} (fitness={cand_fit:.3f}): {changes[0].reason}",
                        action_type="rsi_engine",
                        evidence_level=ev_level,
                        score_metric="confidence_score",
                    )
                )

                if best_cand_bundle is None or cand_fit > best_cand_bundle[2]:
                    best_cand_bundle = (
                        cand_strategy,
                        cand_outcome,
                        cand_fit,
                        changes,
                        level_used,
                    )
                next_version += 1

            if best_cand_bundle is None:
                stagnant_l1 += 1
                continue

            cand_strat, cand_outcome, cand_fit, chs, lvl = best_cand_bundle
            cand_eval = cand_outcome.evaluation
            cand_null = cand_outcome.null_control
            best_trial = cand_outcome.trial_record
            dsr_prob = (
                float(cand_eval.deflated_sharpe_prob)
                if cand_eval.deflated_sharpe_prob is not None
                else 0.0
            )
            passes_base_gates = (
                cand_fit > verified_champion_fitness + 1e-4
                and cand_eval.overall.avg_return > 0.0
                and cand_eval.overall.max_drawdown <= cfg.max_drawdown
                and not cand_eval.anti_overfit.is_overfit
                and cand_eval.overall.signal_count >= cfg.min_signals
                and dsr_prob >= cfg.min_dsr_prob
                and (not cfg.require_null_control or cand_eval.null_control_passed is True)
                and (
                    not cfg.require_portfolio_gate
                    or (
                        cand_outcome.portfolio_verdict is not None
                        and cand_outcome.portfolio_verdict.passed
                    )
                )
            )
            if passes_base_gates and cfg.require_strategy_rerun_null:
                rerun_null = self._null_ctrl.evaluate_strategy_against_null_worlds(
                    cand_strat,
                    search_market,
                    cand_eval,
                    n_worlds=cfg.strategy_rerun_null_worlds,
                    seed=seed + rnd,
                    backtest_engine=self._kernel._build_engine(),
                )
                cand_outcome.strategy_rerun_null = rerun_null
                if not rerun_null.passed:
                    cand_eval.null_control_passed = False
                    cand_eval.null_p_value = round(
                        max(float(cand_eval.null_p_value or 0.0), rerun_null.worst_p_value),
                        4,
                    )
                passes_gate = rerun_null.passed
            else:
                passes_gate = passes_base_gates

            is_stepping_stone = (
                not passes_gate
                and promotions == 0
                and cand_eval.overall.signal_count >= min(5, cfg.min_signals)
                and cand_fit > working_fitness + 1e-4
                and (
                    working_eval.overall.signal_count < 5
                    or (
                        cand_eval.overall.max_drawdown
                        <= max(cfg.max_drawdown, working_eval.overall.max_drawdown)
                        and cand_eval.overall.avg_return >= working_eval.overall.avg_return
                    )
                )
            )
            desc = "; ".join(f"{c.target}->{c.to_value}" for c in chs)

            if passes_gate:
                if best_trial is not None:
                    self._trial_ledger.mark_trial_accepted(best_trial.trial_id, True)
                working_strategy = cand_strat
                working_eval = cand_eval
                working_fitness = cand_fit
                verified_champion_strategy = cand_strat
                verified_champion_eval = cand_eval
                verified_champion_fitness = cand_fit
                verified_champion_outcome = cand_outcome
                has_verified_promotion = True
                kept_rounds += 1
                promotions += 1
                stagnant_l1 = 0
                status: Literal["keep", "stepping_stone", "revert"] = "keep"
            elif is_stepping_stone:
                # Escape a 0-signal dead zone for search continuity without counting as L0 promotion.
                working_strategy = cand_strat
                working_eval = cand_eval
                working_fitness = cand_fit
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

        holdout_eval: EvaluationReport | None = None
        if vault is not None:
            holdout_outcome = vault.evaluate_locked_holdout(
                verified_champion_strategy,
                self._kernel,
                mode="full",
                record_trial=False,
                portfolio_gate=portfolio_gate_cfg,
            )
            holdout_eval = holdout_outcome.evaluation

        if persist_champion and has_verified_promotion and str(self._db_path) != ":memory:":
            try:
                strat_store = StrategyStore(self._db_path)
                strat_store.save(verified_champion_strategy)
                strat_store.save_evaluation(verified_champion_eval)
            except Exception:
                pass

        enrolled_in_shadow = False
        if auto_enroll_shadow and has_verified_promotion and shadow_ledger is not None:
            try:
                shadow_ledger.register_candidate(verified_champion_strategy, verified_champion_eval)
                enrolled_in_shadow = True
            except Exception:
                enrolled_in_shadow = False

        kernel_after = compute_kernel_fingerprint()
        result = RSIRunResult(
            charter_name=cfg.name,
            trial_family_id=family_id,
            seed_strategy_id=seed_strategy.meta.id,
            champion_id=verified_champion_strategy.meta.id,
            champion_promoted=has_verified_promotion,
            champion_strategy=verified_champion_strategy,
            champion_evaluation=verified_champion_eval,
            working_strategy=working_strategy,
            working_evaluation=working_eval,
            total_trials_recorded=self._trial_ledger.count_family_trials(family_id),
            kept_rounds=kept_rounds,
            promotions=promotions,
            kernel_hash_before=kernel_before,
            kernel_hash_after=kernel_after,
            kernel_intact=(kernel_before == kernel_after),
            rows=rows,
            archive_summary=archive.coverage_summary(),
            scars_created=scars_created,
            holdout_evaluation=holdout_eval,
            portfolio_verdict=verified_champion_outcome.portfolio_verdict,
            enrolled_in_shadow=enrolled_in_shadow,
            persisted_factors=persisted_factors,
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            tsv_path = output_dir / "results.tsv"
            report_path = output_dir / "rsi_run_report.md"
            yaml_path = output_dir / f"{verified_champion_strategy.meta.id}.yaml"
            tsv_path.write_text(self.render_tsv(result), encoding="utf-8")
            md_text = self.render_markdown(result)
            report_path.write_text(md_text, encoding="utf-8")
            StrategySerializer().to_file(verified_champion_strategy, yaml_path)
            result.tsv_path = tsv_path
            result.report_path = report_path
            result.champion_yaml_path = yaml_path
        else:
            md_text = self.render_markdown(result)

        effective_artifact_store = artifact_store
        if effective_artifact_store is None and output_dir is not None:
            from alphaevo.provenance.store import RunArtifactStore

            effective_artifact_store = RunArtifactStore(
                artifact_root=output_dir,
                db_path=self._db_path,
            )
        if effective_artifact_store is not None:
            try:
                manifest = effective_artifact_store.save_run(
                    kind="rsi",
                    strategy=verified_champion_strategy,
                    evaluation=verified_champion_eval,
                    market_data=search_market,
                    trades=verified_champion_outcome.backtest_result.signals,
                    initial_strategy=seed_strategy,
                    adapter_name="snapshot_or_live",
                    requested_protocol="full_research_v1",
                    data_reproducibility="replayable",
                    data_mutability="replayable_snapshot",
                    random_seed=seed,
                    report_markdown=md_text,
                )
                result.run_id = manifest.run_id
            except Exception:
                pass

        return result

    def _eval_with_kernel(
        self,
        *,
        strategy: Strategy,
        market_data: dict[str, pd.DataFrame],
        family_id: str,
        charter: RSICharter,
        seed: int,
        portfolio_gate: PortfolioGateConfig | None,
        rsi_level: RSILevel,
        round_num: int,
        accepted: bool,
        notes: str,
    ) -> KernelEvaluationOutcome:
        return self._kernel.evaluate(
            strategy,
            market_data,
            mode="fast",
            family_id=family_id,
            record_trial=True,
            run_null_control=charter.require_null_control,
            run_strategy_rerun_null=False,
            null_permutations=charter.null_permutations,
            min_signals=charter.min_signals,
            min_dsr_prob=charter.min_dsr_prob,
            max_drawdown=charter.max_drawdown,
            portfolio_gate=portfolio_gate,
            rsi_level=rsi_level,
            round_num=round_num,
            accepted=accepted,
            notes=notes,
            seed=seed,
        )

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
        if result.holdout_evaluation is not None:
            ho = result.holdout_evaluation.overall
            lines += [
                "",
                "## Locked Out-of-Sample Holdout",
                "",
                f"- Signals: {ho.signal_count} | Win Rate: {ho.win_rate:.1%} | Avg Return: {ho.avg_return:.2%} | Max DD: {ho.max_drawdown:.1%} | Sharpe: {ho.sharpe_ratio:.2f}",
            ]
        if result.portfolio_verdict is not None:
            pv = result.portfolio_verdict
            lines += [
                "",
                "## Layer-2 Portfolio Gate",
                "",
                f"- Passed: **{pv.passed}** | Total Return: {pv.total_return:.2%} | Max DD: {pv.max_drawdown:.1%} | Trades: {pv.total_trades}",
            ]
        if result.persisted_factors or result.enrolled_in_shadow:
            lines += ["", "## Convergence & Provenance", ""]
            if result.persisted_factors:
                lines.append(f"- Persisted L2 Factors: {', '.join(result.persisted_factors)}")
            if result.enrolled_in_shadow:
                lines.append(f"- Shadow Ledger Enrolled: `{result.champion_id}`")
            if result.run_id:
                lines.append(f"- Run Manifest ID: `{result.run_id}`")
        lines += [
            "",
            "> Research & educational use only. Past simulated performance does not guarantee future returns.",
        ]
        return "\n".join(lines) + "\n"


__all__ = ["RSIRoundRow", "RSIRunResult", "StockRSIEngine"]
