"""Adversarial Heterogeneous Multi-Model Council for Stock RSI."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import (
    EvaluationReport,
    ReflectionResult,
    StrategyChange,
)
from alphaevo.models.strategy import Strategy, StrategyCondition
from alphaevo.reflection.critic import SelfCritic
from alphaevo.research_committee import ResearchCommittee
from alphaevo.rsi.co_steer import BlindSpotCluster, ExecutableScar
from alphaevo.rsi.diagnostics import ConditionPassRate, StrategyBottleneckReport
from alphaevo.strategy.tunable import is_period_tunable_target, resolve_tunable_target

logger = logging.getLogger(__name__)


_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("claude", "anthropic")),
    ("openai", ("gpt", "openai", "codex", "o1", "o3", "o4")),
    ("google", ("gemini", "google", "agy", "vertex_ai")),
    ("deepseek", ("deepseek",)),
    ("qwen", ("qwen",)),
    ("moonshot", ("kimi", "moonshot")),
    ("zhipu", ("glm", "zhipu")),
    ("minimax", ("minimax",)),
)


def infer_model_family(model_name: str) -> str:
    """Map model identifiers to provider/architecture families for independence checks."""
    lower = model_name.strip().lower()
    if not lower or lower.startswith(("heuristic", "deterministic", "offline")):
        return "deterministic"
    for family, tokens in _FAMILY_KEYWORDS:
        if any(tok in lower for tok in tokens):
            return family
    if "/" in lower:
        return lower.split("/", 1)[0]
    return lower


class ModelSpec(BaseModel):
    """Role assignment inside the Heterogeneous Research Council."""

    model: str = "heuristic"
    backend: Literal["deterministic", "litellm", "cli"] = "deterministic"
    cli_tool: Literal["agy", "codex"] | None = None
    timeout_seconds: int = Field(default=60, ge=5, le=300)

    @property
    def family(self) -> str:
        return infer_model_family(self.model)


class HeterogeneousCouncilConfig(BaseModel):
    """Requires distinct LLM model families for Proposer vs Prosecutor roles."""

    proposer: ModelSpec = Field(default_factory=ModelSpec)
    builder: ModelSpec = Field(default_factory=ModelSpec)
    prosecutor: ModelSpec = Field(default_factory=ModelSpec)

    @model_validator(mode="after")
    def _enforce_adversarial_independence(self) -> HeterogeneousCouncilConfig:
        p_fam = self.proposer.family
        q_fam = self.prosecutor.family
        if p_fam != "deterministic" and q_fam != "deterministic" and p_fam == q_fam:
            raise ValueError(
                f"HeterogeneousCouncil requires proposer_family != prosecutor_family "
                f"(got both '{p_fam}')"
            )
        return self


class CouncilProposal(BaseModel):
    """Candidate change bundle put forward by Proposer/Builder."""

    proposer_model: str
    proposer_family: str
    rationale: str
    changes: list[StrategyChange] = Field(default_factory=list)


class ProsecutorAudit(BaseModel):
    """Adversarial risk/leakage/scar verdict issued by Prosecutor."""

    prosecutor_model: str
    prosecutor_family: str
    approved: bool
    objections: list[str] = Field(default_factory=list)
    committee_status: str = "pass"


class CouncilDeliberation(BaseModel):
    """Combined proposal and adversarial audit trail for one candidate."""

    proposal: CouncilProposal
    audit: ProsecutorAudit


def _format_diagnostic_context(
    strategy: Strategy,
    evaluation: EvaluationReport,
    *,
    blind_spot: BlindSpotCluster | None = None,
    bottleneck: StrategyBottleneckReport | None = None,
    mission_notes: str = "",
    scars: list[Any] | None = None,
) -> str:
    """Format compact bottleneck, failure-cluster, and pinned-scar telemetry for prompts."""
    ov = evaluation.overall
    sl = strategy.exit.stop_loss
    tp = strategy.exit.take_profit
    lines = [
        (
            f"Strategy={strategy.meta.id} (logic={strategy.entry.logic}, "
            f"hold={strategy.exit.max_holding_days}d, SL={sl.type}:{sl.value or sl.multiplier}, "
            f"TP={tp.type}:{tp.value or tp.trigger_pct})"
        ),
        (
            f"Metrics: signals={ov.signal_count}, win_rate={ov.win_rate:.1%}, "
            f"avg_ret={ov.avg_return:.2%}, max_dd={ov.max_drawdown:.1%}, "
            f"sharpe={ov.sharpe_ratio:.2f}, overfit={evaluation.anti_overfit.is_overfit}"
        ),
    ]
    if bottleneck is not None:
        weakest = sorted(bottleneck.conditions, key=lambda c: c.pass_rate)[:3]
        weak_txt = (
            ", ".join(f"{c.group}.{c.indicator}({c.pass_rate:.1%})" for c in weakest) or "none"
        )
        tightest = (
            f"{bottleneck.tightest_bottleneck.group}.{bottleneck.tightest_bottleneck.indicator}"
            if bottleneck.tightest_bottleneck is not None
            else "none"
        )
        lines.append(
            f"Bottleneck X-Ray: tightest={tightest}, "
            f"combined_pass={bottleneck.combined_pass_rate:.2%}, weakest=[{weak_txt}] | "
            f"{bottleneck.diagnosis_summary[:180]}"
        )
    if blind_spot is not None:
        lines.append(
            f"Blind Spot Cluster: {blind_spot.cluster_id} ({blind_spot.severity}) — "
            f"{blind_spot.evidence[:160]} | Action: {blind_spot.suggested_action[:140]}"
        )
    if scars:
        scar_txt = "; ".join(
            f"[{getattr(s, 'forbidden_change_type', '*')}:{getattr(s, 'target_pattern', '')}] "
            f"{getattr(s, 'failure_mechanism', '')[:70]}"
            for s in scars[:5]
        )
        lines.append(f"Active Pinned Scars (MUST NOT VIOLATE): {scar_txt}")
    if mission_notes.strip():
        lines.append(f"Charter Mission: {mission_notes.strip()[:220]}")
    return "\n".join(lines)


class HeterogeneousCouncil:
    """Proposes, audits, and filters mutations with adversarial model separation."""

    def __init__(
        self,
        config: HeterogeneousCouncilConfig | None = None,
        *,
        critic: SelfCritic | None = None,
        scars: list[ExecutableScar] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or HeterogeneousCouncilConfig()
        self._critic = critic or SelfCritic(scars=scars or [])
        if scars is not None:
            self._critic.set_scars(scars)
        self._committee = ResearchCommittee()
        # Transient session credential only; never stored in config or serialized.
        self._api_key = api_key

    def set_scars(self, scars: list[ExecutableScar]) -> None:
        self._critic.set_scars(scars)

    def generate_candidates(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        extra_changes: list[list[StrategyChange]] | None = None,
        mission_notes: str = "",
        max_candidates: int = 8,
        exclude_sigs: set[tuple[tuple[str, str, str], ...]] | None = None,
        blind_spot: BlindSpotCluster | None = None,
        bottleneck: StrategyBottleneckReport | None = None,
    ) -> list[CouncilDeliberation]:
        """Generate candidate mutations and filter every bundle through the Prosecutor."""
        proposals: list[CouncilProposal] = []
        p_spec = self.config.proposer

        if extra_changes:
            for ch_list in extra_changes:
                if ch_list:
                    proposals.append(
                        CouncilProposal(
                            proposer_model=p_spec.model,
                            proposer_family=p_spec.family,
                            rationale=mission_notes[:120] or "Co-STEER candidate",
                            changes=ch_list,
                        )
                    )

        # Optional CLI / LLM proposal enrichment (gracefully falls back to heuristics)
        if p_spec.backend == "cli":
            cli_prop = self._try_cli_proposer(
                strategy,
                evaluation,
                p_spec,
                mission_notes,
                blind_spot=blind_spot,
                bottleneck=bottleneck,
            )
            if cli_prop is not None:
                proposals.append(cli_prop)
        elif p_spec.backend == "litellm":
            llm_prop = self._try_litellm_proposer(
                strategy,
                evaluation,
                p_spec,
                blind_spot=blind_spot,
                bottleneck=bottleneck,
                mission_notes=mission_notes,
            )
            if llm_prop is not None:
                proposals.append(llm_prop)

        # Deterministic failure-driven diagnostic proposals
        proposals.extend(
            self._build_heuristic_proposals(
                strategy,
                evaluation,
                p_spec,
                blind_spot=blind_spot,
                bottleneck=bottleneck,
            )
        )

        # Deduplicate while preserving proposal priority and excluding already-tried signatures
        deliberations: list[CouncilDeliberation] = []
        seen: set[tuple[tuple[str, str, str], ...]] = set(exclude_sigs or set())

        for prop in proposals:
            if not prop.changes:
                continue
            sig = tuple(
                sorted((c.change_type.value, c.target, str(c.to_value)) for c in prop.changes)
            )
            if sig in seen:
                continue
            seen.add(sig)

            audit = self._run_prosecutor(
                strategy,
                evaluation,
                prop.changes,
                blind_spot=blind_spot,
                bottleneck=bottleneck,
                mission_notes=mission_notes,
            )
            deliberations.append(CouncilDeliberation(proposal=prop, audit=audit))
            if len([d for d in deliberations if d.audit.approved]) >= max_candidates:
                break

        return deliberations

    def audit_changes(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        changes: list[StrategyChange],
        *,
        blind_spot: BlindSpotCluster | None = None,
        bottleneck: StrategyBottleneckReport | None = None,
        mission_notes: str = "",
    ) -> ProsecutorAudit:
        """Audit an externally constructed candidate (e.g. genetic crossover) via Prosecutor."""
        return self._run_prosecutor(
            strategy,
            evaluation,
            changes,
            blind_spot=blind_spot,
            bottleneck=bottleneck,
            mission_notes=mission_notes,
        )

    def _run_prosecutor(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        changes: list[StrategyChange],
        *,
        blind_spot: BlindSpotCluster | None = None,
        bottleneck: StrategyBottleneckReport | None = None,
        mission_notes: str = "",
    ) -> ProsecutorAudit:
        """Audit candidate changes against SelfCritic, Pinned Scars, and Committee."""
        q_spec = self.config.prosecutor
        reflection = ReflectionResult(
            strategy_id=strategy.meta.id,
            evaluation_id=evaluation.evaluation_id,
            proposed_changes=changes,
        )
        verdict = self._critic.critique(strategy, evaluation, reflection)
        committee = self._committee.review(strategy, evaluation, mutation_plan=changes)

        objections: list[str] = [f"{c.target}: {reason}" for c, reason in verdict.rejected]
        approved = len(verdict.approved) == len(changes)

        diag_ctx = _format_diagnostic_context(
            strategy,
            evaluation,
            blind_spot=blind_spot,
            bottleneck=bottleneck,
            mission_notes=mission_notes,
            scars=self._critic._scars,
        )

        # Optional adversarial LLM / CLI prosecutor veto check
        if approved and q_spec.backend == "cli":
            cli_veto = self._try_cli_prosecutor(
                strategy, evaluation, changes, q_spec, diag_ctx=diag_ctx
            )
            if cli_veto:
                objections.extend(cli_veto)
                approved = False
        elif approved and q_spec.backend == "litellm":
            llm_veto = self._try_litellm_prosecutor(
                strategy, evaluation, changes, q_spec, diag_ctx=diag_ctx
            )
            if llm_veto:
                objections.extend(llm_veto)
                approved = False

        return ProsecutorAudit(
            prosecutor_model=q_spec.model,
            prosecutor_family=q_spec.family,
            approved=approved,
            objections=objections,
            committee_status=committee.overall_status,
        )

    @staticmethod
    def _build_heuristic_proposals(
        strategy: Strategy,
        evaluation: EvaluationReport,
        p_spec: ModelSpec,
        *,
        blind_spot: BlindSpotCluster | None = None,
        bottleneck: StrategyBottleneckReport | None = None,
    ) -> list[CouncilProposal]:
        """Build failure-driven, bottleneck-aware mutation proposals (no API key needed)."""
        ov = evaluation.overall

        def _make_prop(changes: list[StrategyChange], rationale: str) -> CouncilProposal:
            return CouncilProposal(
                proposer_model=p_spec.model,
                proposer_family=p_spec.family,
                rationale=rationale,
                changes=changes,
            )

        def _interleave(*groups: list[CouncilProposal]) -> list[CouncilProposal]:
            merged: list[CouncilProposal] = []
            max_len = max((len(g) for g in groups), default=0)
            for idx in range(max_len):
                for grp in groups:
                    if idx < len(grp):
                        merged.append(grp[idx])
            return merged

        def _find_cond(group: str, indicator: str) -> StrategyCondition | None:
            buckets = {
                "triggers": strategy.entry.triggers,
                "conditions": strategy.entry.conditions,
                "guards": strategy.entry.guards,
                "filters": strategy.entry.filters,
            }
            for c in buckets.get(group, ()):
                if c.indicator == indicator:
                    return c
            for c in (
                strategy.entry.triggers
                + strategy.entry.conditions
                + strategy.entry.guards
                + strategy.entry.filters
            ):
                if c.indicator == indicator:
                    return c
            return None

        def _tunable_values(
            param_target: str,
            lo: float,
            hi: float,
            step: float,
            current: float | int,
            *,
            prefer_loosen: bool = False,
            cond_op: str | None = None,
        ) -> list[int | float]:
            is_period = is_period_tunable_target(param_target)
            if prefer_loosen and cond_op in {">", ">="}:
                raw = [current - step, current - step * 2, lo, current + step]
            elif prefer_loosen and cond_op in {"<", "<="}:
                raw = [current + step, current + step * 2, hi, current - step]
            else:
                raw = [current + step, current - step]
                if is_period or step * 2 <= (hi - lo):
                    raw.extend([current + step * 2, current - step * 2, hi, lo])

            vals: list[int | float] = []
            seen: set[float] = set()
            for r in raw:
                cand: int | float = int(round(r)) if is_period else round(float(r), 4)
                if cand == current or cand < lo or cand > hi or float(cand) in seen:
                    continue
                seen.add(float(cand))
                vals.append(cand)
            return vals

        tunable_by_target = {p.target: p for p in strategy.params.tunable}
        bottleneck_props: list[CouncilProposal] = []
        single_tunables: list[StrategyChange] = []
        period_props: list[CouncilProposal] = []
        entry_props: list[CouncilProposal] = []
        exit_tunable_props: list[CouncilProposal] = []

        # 1. Bottleneck-driven unlock proposals for sparse / zero-signal lockouts
        is_sparse = ov.signal_count < 15 or (
            blind_spot is not None and blind_spot.cluster_id == "zero_signal_lockout"
        )
        if is_sparse:
            ordered_conds: list[ConditionPassRate] = []
            guard_product_choke = (
                bottleneck is not None
                and bottleneck.guard_group_pass_rate < 0.25
                and (len(strategy.entry.guards) + len(strategy.entry.filters)) >= 2
            )
            if bottleneck is not None and bottleneck.conditions:
                ordered_conds = sorted(
                    bottleneck.conditions,
                    key=lambda c: (
                        0
                        if (
                            (strategy.entry.logic == "or" or guard_product_choke)
                            and c.group in {"guards", "filters"}
                        )
                        else 1,
                        c.pass_rate,
                    ),
                )

            pass_cutoff = 0.60 if guard_product_choke else 0.30
            single_loosen_props: list[CouncilProposal] = []
            loosened_changes: list[StrategyChange] = []
            guard_remove_candidates: list[StrategyChange] = []
            for cond_rep in ordered_conds:
                if cond_rep.pass_rate >= pass_cutoff:
                    continue
                cond_obj = _find_cond(cond_rep.group, cond_rep.indicator)
                cond_op = cond_obj.op if cond_obj is not None else None
                for tgt in cond_rep.matching_tunable_targets:
                    param = tunable_by_target.get(tgt)
                    if param is None:
                        continue
                    curr = resolve_tunable_target(strategy, tgt)
                    if not isinstance(curr, (int, float)):
                        continue
                    lo, hi = param.range
                    step = param.step or (hi - lo) / 6.0
                    cands = _tunable_values(
                        tgt,
                        lo,
                        hi,
                        step,
                        curr,
                        prefer_loosen=True,
                        cond_op=cond_op,
                    )
                    for cand_val in cands[:2]:
                        ch = StrategyChange(
                            change_type=ChangeType.LOOSEN_FILTER,
                            target=tgt,
                            from_value=curr,
                            to_value=cand_val,
                            reason=(
                                f"Unlock bottleneck {cond_rep.group}.{cond_rep.indicator} "
                                f"(pass_rate={cond_rep.pass_rate:.1%}): {curr} -> {cand_val}"
                            ),
                        )
                        loosened_changes.append(ch)
                        single_loosen_props.append(
                            _make_prop(
                                [ch],
                                f"Bottleneck X-Ray unlock on {cond_rep.indicator} ({cond_rep.pass_rate:.1%})",
                            )
                        )

                # Guard/filter relaxation & pruning when AND guards choke signal flow
                is_guard_like = cond_rep.group in {"guards", "filters"}
                if (is_guard_like or not cond_rep.matching_tunable_targets) and (
                    cond_rep.pass_rate < 0.15 or (guard_product_choke and is_guard_like)
                ):
                    if (
                        not cond_rep.matching_tunable_targets
                        and cond_obj is not None
                        and isinstance(cond_obj.value, (int, float))
                        and not isinstance(cond_obj.value, bool)
                    ):
                        v_curr = float(cond_obj.value)
                        if cond_obj.op in {"<", "<="} and v_curr > 0:
                            v_nxt = round(v_curr * 1.45, 4)
                        elif cond_obj.op in {">", ">="}:
                            v_nxt = round(v_curr * 0.65 if v_curr > 0 else v_curr - 0.1, 4)
                        else:
                            v_nxt = v_curr
                        if v_nxt != v_curr:
                            tgt_str = (
                                f"entry.{cond_rep.group}[indicator={cond_rep.indicator}].value"
                            )
                            ch_relax = StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=tgt_str,
                                from_value=v_curr,
                                to_value=v_nxt,
                                reason=f"Relax tight {cond_rep.group} threshold ({cond_rep.pass_rate:.1%} pass)",
                            )
                            loosened_changes.append(ch_relax)
                            single_loosen_props.append(
                                _make_prop([ch_relax], f"Relax bottleneck {cond_rep.indicator}")
                            )

                    n_guards_total = len(strategy.entry.guards) + len(strategy.entry.filters)
                    if (
                        bottleneck is not None
                        and ov.signal_count < 5
                        and is_guard_like
                        and n_guards_total >= 2
                    ):
                        rm_ch = StrategyChange(
                            change_type=ChangeType.REMOVE_CONDITION,
                            target=f"entry.{cond_rep.group}[indicator={cond_rep.indicator}]",
                            from_value=cond_rep.value,
                            to_value=None,
                            reason=(
                                f"Prune sparse-lockout guard {cond_rep.indicator} "
                                f"(pass_rate={cond_rep.pass_rate:.1%})"
                            ),
                        )
                        guard_remove_candidates.append(rm_ch)
                        single_loosen_props.append(
                            _make_prop(
                                [rm_ch],
                                f"Prune bottleneck {cond_rep.group}.{cond_rep.indicator} (pass={cond_rep.pass_rate:.1%})",
                            )
                        )

            trigger_group_len = len(strategy.entry.triggers or strategy.entry.conditions)
            logic_props: list[CouncilProposal] = []
            if strategy.entry.logic == "and" and trigger_group_len >= 2:
                unlock_change = StrategyChange(
                    change_type=ChangeType.CHANGE_LOGIC,
                    target="entry.logic",
                    from_value="and",
                    to_value="or",
                    reason="Unlock signal flow from over-constrained AND trigger stack",
                )
                unlock_bundle: list[StrategyChange] = [unlock_change]
                seen_bundle_targets: set[str] = {"entry.logic"}
                for lch in loosened_changes:
                    if len(unlock_bundle) >= 3:
                        break
                    if lch.target not in seen_bundle_targets:
                        unlock_bundle.append(lch)
                        seen_bundle_targets.add(lch.target)
                if len(unlock_bundle) == 1:
                    for param in strategy.params.tunable:
                        if len(unlock_bundle) >= 2 or not param.target.startswith("entry."):
                            continue
                        if param.target in seen_bundle_targets:
                            continue
                        curr = resolve_tunable_target(strategy, param.target)
                        if not isinstance(curr, (int, float)):
                            continue
                        lo, hi = param.range
                        step = param.step or (hi - lo) / 4.0
                        bundle_val: int | float = (
                            int(round(min(hi, curr + step * 2)))
                            if is_period_tunable_target(param.target)
                            else round(float(min(hi, curr + step)), 4)
                        )
                        if bundle_val != curr and lo <= bundle_val <= hi:
                            unlock_bundle.append(
                                StrategyChange(
                                    change_type=ChangeType.TIGHTEN_FILTER
                                    if bundle_val > curr
                                    else ChangeType.LOOSEN_FILTER,
                                    target=param.target,
                                    from_value=curr,
                                    to_value=bundle_val,
                                    reason=f"Calibrate {param.label or param.target} alongside OR unlock",
                                )
                            )
                            seen_bundle_targets.add(param.target)
                if len(unlock_bundle) > 1:
                    logic_props.append(
                        _make_prop(unlock_bundle, "Coordinated OR unlock + bottleneck calibration")
                    )
                if guard_remove_candidates:
                    logic_props.append(
                        _make_prop(
                            [unlock_change, guard_remove_candidates[0]],
                            "Coordinated OR unlock + prune choking guard",
                        )
                    )
                logic_props.append(_make_prop([unlock_change], "Unlock sparse signal flow"))

            # When AND triggers bottleneck signal flow, prioritize structural OR unlock first.
            if logic_props:
                bottleneck_props.extend(logic_props)
                bottleneck_props.extend(single_loosen_props)
            else:
                bottleneck_props.extend(single_loosen_props)

        # 2. Exit & risk calibration across pct/atr stop-loss and rr/trailing take-profit
        exit_levers: list[StrategyChange] = []
        exit_props: list[CouncilProposal] = []
        if strategy.exit.stop_loss.type == "pct" and strategy.exit.stop_loss.value is not None:
            curr_sl = float(strategy.exit.stop_loss.value)
            for delta in (0.02, -0.015, 0.03):
                nxt = round(min(0.18, max(0.02, curr_sl + delta)), 3)
                if abs(nxt - curr_sl) >= 0.005:
                    ch_sl = StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.stop_loss.value",
                        from_value=curr_sl,
                        to_value=nxt,
                        reason="Calibrate pct stop-loss tolerance against chop whipsaw",
                    )
                    if not exit_levers:
                        exit_levers.append(ch_sl)
                    exit_props.append(_make_prop([ch_sl], "Stop-loss risk calibration"))
        elif (
            strategy.exit.stop_loss.type == "atr" and strategy.exit.stop_loss.multiplier is not None
        ):
            curr_mult = float(strategy.exit.stop_loss.multiplier)
            for delta in (0.5, 1.0, -0.3):
                nxt_mult = round(min(5.0, max(0.8, curr_mult + delta)), 2)
                if abs(nxt_mult - curr_mult) >= 0.1:
                    ch_atr = StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.stop_loss.multiplier",
                        from_value=curr_mult,
                        to_value=nxt_mult,
                        reason="Calibrate ATR stop-loss multiplier for volatility regime",
                    )
                    if not exit_levers:
                        exit_levers.append(ch_atr)
                    exit_props.append(_make_prop([ch_atr], "ATR stop-loss calibration"))

        curr_hold = strategy.exit.max_holding_days
        for new_hold in (curr_hold + 5, curr_hold + 3, max(3, curr_hold - 3)):
            if new_hold != curr_hold and 2 <= new_hold <= 35:
                ch_hold = StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.max_holding_days",
                    from_value=curr_hold,
                    to_value=new_hold,
                    reason=f"Adjust max holding horizon {curr_hold} -> {new_hold} days",
                )
                if len(exit_levers) < 2 and new_hold > curr_hold:
                    exit_levers.append(ch_hold)
                exit_props.append(_make_prop([ch_hold], "Holding window tuning"))

        if strategy.exit.take_profit.type == "rr" and strategy.exit.take_profit.value is not None:
            curr_tp = float(strategy.exit.take_profit.value)
            for delta in (0.5, 2.0, -0.4):
                nxt_tp = round(min(6.0, max(1.0, curr_tp + delta)), 2)
                if abs(nxt_tp - curr_tp) >= 0.1:
                    ch_tp = StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.take_profit.value",
                        from_value=curr_tp,
                        to_value=nxt_tp,
                        reason="Calibrate R:R take-profit multiple",
                    )
                    if len(exit_levers) < 3 and delta > 0:
                        exit_levers.append(ch_tp)
                    exit_props.append(_make_prop([ch_tp], "Take-profit payoff adjustment"))
        elif strategy.exit.take_profit.type == "trailing":
            curr_trig = float(strategy.exit.take_profit.trigger_pct or 0.08)
            for delta in (0.02, -0.02):
                nxt_trig = round(min(0.25, max(0.03, curr_trig + delta)), 3)
                if abs(nxt_trig - curr_trig) >= 0.005:
                    ch_tr = StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.take_profit.trigger_pct",
                        from_value=curr_trig,
                        to_value=nxt_trig,
                        reason="Calibrate trailing take-profit activation threshold",
                    )
                    exit_props.append(_make_prop([ch_tr], "Trailing trigger calibration"))

        if len(exit_levers) >= 2 and ov.signal_count >= 10:
            exit_props.insert(
                0, _make_prop(exit_levers[:3], "Coordinated exit risk & payoff calibration")
            )

        # 3. Partition parameter neighborhood into period / entry / exit groups
        for param in strategy.params.tunable:
            current = resolve_tunable_target(strategy, param.target)
            if not isinstance(current, (int, float)):
                continue
            lo, hi = param.range
            step = param.step or (hi - lo) / 8.0
            cand_vals = _tunable_values(param.target, lo, hi, step, current)
            for idx_c, cand in enumerate(cand_vals):
                if param.target.startswith("exit."):
                    ctype = ChangeType.ADJUST_EXIT
                elif cand > current:
                    ctype = ChangeType.LOOSEN_FILTER
                else:
                    ctype = ChangeType.TIGHTEN_FILTER
                ch = StrategyChange(
                    change_type=ctype,
                    target=param.target,
                    from_value=current,
                    to_value=cand,
                    reason=f"Tune {param.label or param.target}: {current} -> {cand}",
                )
                if idx_c < 2:
                    single_tunables.append(ch)
                prop = _make_prop([ch], f"Tune {param.target}")
                if is_period_tunable_target(param.target):
                    period_props.append(prop)
                elif param.target.startswith("exit."):
                    exit_tunable_props.append(prop)
                else:
                    entry_props.append(prop)

        # 4. Failure-driven prioritization across groups
        cluster_id = blind_spot.cluster_id if blind_spot is not None else "healthy_refinement"
        out: list[CouncilProposal] = []
        if is_sparse or cluster_id == "zero_signal_lockout":
            out.extend(bottleneck_props)
            out.extend(_interleave(entry_props, period_props))
            out.extend(_interleave(exit_props, exit_tunable_props))
        elif cluster_id == "whipsaw_stopout_in_chop" or ov.max_drawdown > 0.25:
            out.extend(_interleave(exit_props, exit_tunable_props))
            out.extend(_interleave(period_props, entry_props))
        elif cluster_id == "weak_payoff_tail":
            out.extend(_interleave(exit_props, exit_tunable_props))
            out.extend(_interleave(entry_props, period_props))
        elif evaluation.anti_overfit.is_overfit:
            out.extend(_interleave(period_props, exit_props))
            out.extend(_interleave(entry_props, exit_tunable_props))
        else:
            out.extend(_interleave(exit_props, entry_props, period_props, exit_tunable_props))

        if single_tunables and exit_levers:
            combo = [single_tunables[0], exit_levers[0]]
            if single_tunables[0].target != exit_levers[0].target:
                out.append(_make_prop(combo, "Joint entry-tunable and exit-risk calibration"))

        return out

    def _try_litellm_proposer(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        p_spec: ModelSpec,
        *,
        blind_spot: BlindSpotCluster | None = None,
        bottleneck: StrategyBottleneckReport | None = None,
        mission_notes: str = "",
    ) -> CouncilProposal | None:
        """Ask bottleneck-aware LiteLLM Proposer for structured changes; fallback cleanly."""
        try:
            from alphaevo.backtest.indicators import IndicatorRegistry
            from alphaevo.core.config import ConfigManager
            from alphaevo.core.llm import LLMClient
            from alphaevo.reflection.analyzer import ReflectionAnalyzer
            from alphaevo.strategy.dsl.serializer import StrategySerializer

            cfg = ConfigManager().load()
            cfg.llm.model = p_spec.model
            api_key = self._api_key or ConfigManager().get_llm_api_key()
            if not api_key:
                return None
            llm = LLMClient(cfg.llm, api_key=api_key)

            diag_ctx = _format_diagnostic_context(
                strategy,
                evaluation,
                blind_spot=blind_spot,
                bottleneck=bottleneck,
                mission_notes=mission_notes,
                scars=self._critic._scars,
            )
            tunables = ", ".join(p.target for p in strategy.params.tunable[:12]) or "none"
            avail_inds = ", ".join(IndicatorRegistry.available()[:40])
            yaml_excerpt = StrategySerializer().to_yaml(strategy)[:1600]
            prompt = (
                "You are an expert quantitative strategy Proposer inside an adversarial L0-grounded RSI loop.\n"
                f"{diag_ctx}\n"
                f"Tunable targets: {tunables}\n"
                f"Sample available indicators: {avail_inds}\n"
                f"Strategy YAML:\n```yaml\n{yaml_excerpt}\n```\n"
                "Propose at most 3 high-signal changes attacking the diagnosed bottleneck or failure cluster.\n"
                'Respond ONLY with JSON: {"rationale": "...", "changes": [{"change_type": '
                '"tighten_filter|loosen_filter|add_condition|remove_condition|adjust_exit|change_logic", '
                '"target": "...", "from_value": null, "to_value": ..., "reason": "..."}]}'
            )
            try:
                parsed = llm.chat_json(
                    [{"role": "user", "content": prompt}],
                    model=p_spec.model,
                    timeout=p_spec.timeout_seconds,
                )
                raw_changes = parsed.get("changes") if isinstance(parsed, dict) else None
                parsed_changes: list[StrategyChange] = []
                if isinstance(raw_changes, list):
                    for item in raw_changes[:3]:
                        if not isinstance(item, dict) or not item.get("target"):
                            continue
                        try:
                            ctype = ChangeType(str(item["change_type"]).strip())
                            parsed_changes.append(
                                StrategyChange(
                                    change_type=ctype,
                                    target=str(item["target"]).strip(),
                                    from_value=item.get("from_value"),
                                    to_value=item.get("to_value"),
                                    reason=str(item.get("reason", f"LiteLLM {p_spec.model}")),
                                )
                            )
                        except Exception:
                            continue
                if parsed_changes:
                    return CouncilProposal(
                        proposer_model=p_spec.model,
                        proposer_family=p_spec.family,
                        rationale=str(parsed.get("rationale", "Diagnostic LiteLLM proposal")),
                        changes=parsed_changes,
                    )
            except Exception as inner_exc:
                logger.debug(
                    "Diagnostic LiteLLM proposer fell back to ReflectionAnalyzer: %s", inner_exc
                )

            ref = ReflectionAnalyzer(llm=llm).reflect(strategy, evaluation)
            if ref.proposed_changes:
                return CouncilProposal(
                    proposer_model=p_spec.model,
                    proposer_family=p_spec.family,
                    rationale=ref.reflection_summary or "LiteLLM proposal",
                    changes=ref.proposed_changes,
                )
        except Exception as exc:
            logger.debug("LiteLLM proposer fallback to heuristic: %s", exc)
        return None

    def _try_cli_proposer(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        p_spec: ModelSpec,
        mission_notes: str,
        *,
        blind_spot: BlindSpotCluster | None = None,
        bottleneck: StrategyBottleneckReport | None = None,
    ) -> CouncilProposal | None:
        """Invoke optional local CLI agent (agy / codex) via safe list args."""
        tool = p_spec.cli_tool
        if tool not in {"agy", "codex"}:
            return None
        diag_ctx = _format_diagnostic_context(
            strategy,
            evaluation,
            blind_spot=blind_spot,
            bottleneck=bottleneck,
            mission_notes=mission_notes,
            scars=self._critic._scars,
        )
        prompt = (
            f"{diag_ctx}\n"
            "Respond ONLY with JSON object containing 'target', 'to_value', 'change_type', 'reason'."
        )
        cmd = (
            ["agy", "chat", "--model", p_spec.model, prompt]
            if tool == "agy"
            else ["codex", "exec", "--model", p_spec.model, prompt]
        )
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=p_spec.timeout_seconds,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return None
            raw = proc.stdout.strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end <= start:
                return None
            parsed: dict[str, Any] = json.loads(raw[start : end + 1])
            ctype = ChangeType(str(parsed["change_type"]))
            change = StrategyChange(
                change_type=ctype,
                target=str(parsed["target"]),
                from_value=None,
                to_value=parsed.get("to_value"),
                reason=str(parsed.get("reason", f"{tool}:{p_spec.model}")),
            )
            return CouncilProposal(
                proposer_model=p_spec.model,
                proposer_family=p_spec.family,
                rationale=change.reason,
                changes=[change],
            )
        except Exception as exc:
            logger.debug("CLI proposer %s fallback to heuristic: %s", tool, exc)
            return None

    @staticmethod
    def _try_cli_prosecutor(
        strategy: Strategy,
        evaluation: EvaluationReport,
        changes: list[StrategyChange],
        q_spec: ModelSpec,
        *,
        diag_ctx: str = "",
    ) -> list[str]:
        """Optionally invoke adversarial CLI prosecutor; falls back cleanly on unavailability."""
        tool = q_spec.cli_tool
        if tool not in {"agy", "codex"}:
            return []
        change_desc = "; ".join(f"{c.change_type.value}:{c.target}->{c.to_value}" for c in changes)
        ctx_line = f"{diag_ctx}\n" if diag_ctx else ""
        prompt = (
            f"{ctx_line}"
            f"Audit proposed changes [{change_desc}] for strategy {strategy.meta.id} "
            f"(signals={evaluation.overall.signal_count}, overfit={evaluation.anti_overfit.is_overfit}). "
            'Respond JSON {"veto": bool, "reason": "..."}'
        )
        cmd = (
            ["agy", "chat", "--model", q_spec.model, prompt]
            if tool == "agy"
            else ["codex", "exec", "--model", q_spec.model, prompt]
        )
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=q_spec.timeout_seconds,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return []
            raw = proc.stdout.strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                parsed: dict[str, Any] = json.loads(raw[start : end + 1])
                if parsed.get("veto") is True:
                    return [str(parsed.get("reason", "Vetoed by CLI Prosecutor"))]
        except Exception as exc:
            logger.debug(
                "CLI prosecutor %s unavailable (%s); using deterministic critic", tool, exc
            )
        return []

    def _try_litellm_prosecutor(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        changes: list[StrategyChange],
        q_spec: ModelSpec,
        *,
        diag_ctx: str = "",
    ) -> list[str]:
        """Optionally invoke LiteLLM prosecutor; falls back cleanly when unconfigured."""
        try:
            from alphaevo.core.config import ConfigManager
            from alphaevo.core.llm import LLMClient

            cfg = ConfigManager().load()
            cfg.llm.model = q_spec.model
            api_key = self._api_key or ConfigManager().get_llm_api_key()
            if not api_key:
                return []
            llm = LLMClient(cfg.llm, api_key=api_key)
            change_desc = "; ".join(
                f"{c.change_type.value}:{c.target}->{c.to_value}" for c in changes
            )
            ctx_line = f"{diag_ctx}\n" if diag_ctx else ""
            prompt = (
                f"{ctx_line}"
                f"You are an adversarial quantitative risk prosecutor. Audit changes [{change_desc}] "
                f"for strategy {strategy.meta.id} (signals={evaluation.overall.signal_count}, "
                f"avg_return={evaluation.overall.avg_return:.4f}, overfit={evaluation.anti_overfit.is_overfit}). "
                'Respond ONLY with JSON {"veto": bool, "reason": "..."}.'
            )
            parsed = llm.chat_json(
                [{"role": "user", "content": prompt}],
                model=q_spec.model,
                timeout=q_spec.timeout_seconds,
            )
            if parsed.get("veto") is True:
                return [str(parsed.get("reason", "Vetoed by LiteLLM Prosecutor"))]
        except Exception as exc:
            logger.debug("LiteLLM prosecutor fallback to deterministic critic: %s", exc)
        return []


__all__ = [
    "CouncilDeliberation",
    "CouncilProposal",
    "HeterogeneousCouncil",
    "HeterogeneousCouncilConfig",
    "ModelSpec",
    "ProsecutorAudit",
    "infer_model_family",
]
