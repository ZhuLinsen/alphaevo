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
from alphaevo.models.strategy import Strategy
from alphaevo.reflection.critic import SelfCritic
from alphaevo.research_committee import ResearchCommittee
from alphaevo.rsi.co_steer import ExecutableScar
from alphaevo.strategy.tunable import is_period_tunable_target, resolve_tunable_target

logger = logging.getLogger(__name__)


def infer_model_family(model_name: str) -> str:
    """Map model identifiers to provider/architecture families for independence checks."""
    lower = model_name.strip().lower()
    if not lower or lower.startswith(("heuristic", "deterministic", "offline")):
        return "deterministic"
    if any(tok in lower for tok in ("claude", "anthropic")):
        return "anthropic"
    if any(tok in lower for tok in ("gpt", "openai", "codex", "o1", "o3", "o4")):
        return "openai"
    if any(tok in lower for tok in ("gemini", "google", "agy")):
        return "google"
    if "deepseek" in lower:
        return "deepseek"
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


class HeterogeneousCouncil:
    """Proposes, audits, and filters mutations with adversarial model separation."""

    def __init__(
        self,
        config: HeterogeneousCouncilConfig | None = None,
        *,
        critic: SelfCritic | None = None,
        scars: list[ExecutableScar] | None = None,
    ) -> None:
        self.config = config or HeterogeneousCouncilConfig()
        self._critic = critic or SelfCritic(scars=scars or [])
        if scars is not None:
            self._critic.set_scars(scars)
        self._committee = ResearchCommittee()

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
            cli_prop = self._try_cli_proposer(strategy, evaluation, p_spec, mission_notes)
            if cli_prop is not None:
                proposals.append(cli_prop)
        elif p_spec.backend == "litellm":
            llm_prop = self._try_litellm_proposer(strategy, evaluation, p_spec)
            if llm_prop is not None:
                proposals.append(llm_prop)

        # Deterministic stock RSI heuristic proposals + tunable neighborhood
        proposals.extend(self._build_heuristic_proposals(strategy, evaluation, p_spec))

        # Deduplicate while preserving proposal priority
        deliberations: list[CouncilDeliberation] = []
        seen: set[tuple[tuple[str, str, str], ...]] = set()

        for prop in proposals:
            if not prop.changes:
                continue
            sig = tuple(
                sorted((c.change_type.value, c.target, str(c.to_value)) for c in prop.changes)
            )
            if sig in seen:
                continue
            seen.add(sig)

            audit = self._run_prosecutor(strategy, evaluation, prop.changes)
            deliberations.append(CouncilDeliberation(proposal=prop, audit=audit))
            if len([d for d in deliberations if d.audit.approved]) >= max_candidates:
                break

        return deliberations

    def _run_prosecutor(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        changes: list[StrategyChange],
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

        # Optional adversarial LLM / CLI prosecutor veto check
        if approved and q_spec.backend == "cli":
            cli_veto = self._try_cli_prosecutor(strategy, evaluation, changes, q_spec)
            if cli_veto:
                objections.extend(cli_veto)
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
    ) -> list[CouncilProposal]:
        """Build domain-grounded stock mutation proposals (no API key needed)."""
        out: list[CouncilProposal] = []
        ov = evaluation.overall

        def _add(changes: list[StrategyChange], rationale: str) -> None:
            out.append(
                CouncilProposal(
                    proposer_model=p_spec.model,
                    proposer_family=p_spec.family,
                    rationale=rationale,
                    changes=changes,
                )
            )

        # Escape over-constrained zero-signal / sparse lockouts first
        if ov.signal_count < 15 and strategy.entry.logic == "and":
            unlock_bundle: list[StrategyChange] = [
                StrategyChange(
                    change_type=ChangeType.CHANGE_LOGIC,
                    target="entry.logic",
                    from_value="and",
                    to_value="or",
                    reason="Unlock signal flow from over-confirmed AND condition stack",
                )
            ]
            for cond in strategy.entry.conditions:
                if len(unlock_bundle) >= 3:
                    break
                if cond.indicator == "close_to_ma20_pct":
                    unlock_bundle.append(
                        StrategyChange(
                            change_type=ChangeType.LOOSEN_FILTER,
                            target="entry.conditions[indicator=close_to_ma20_pct].indicator",
                            from_value="close_to_ma20_pct",
                            to_value="close_to_ma60_pct",
                            reason=(
                                "Anchor reversal support on MA60 to prevent whipsaws during OR unlock"
                            ),
                        )
                    )
                elif cond.indicator == "volume_ratio_1d_5d":
                    try:
                        curr_vol = float(cond.value)
                    except (TypeError, ValueError):
                        curr_vol = 2.0
                    if curr_vol < 2.0:
                        unlock_bundle.append(
                            StrategyChange(
                                change_type=ChangeType.TIGHTEN_FILTER,
                                target="entry.conditions[indicator=volume_ratio_1d_5d].value",
                                from_value=curr_vol,
                                to_value=2.0,
                                reason="Require 2.0x volume confirmation alongside OR unlock",
                            )
                        )
            if len(unlock_bundle) > 1:
                _add(
                    unlock_bundle,
                    "Coordinated OR unlock with MA60 anchor and 2.0x volume confirmation",
                )
            _add([unlock_bundle[0]], "Unlock sparse signal flow")

        # Cut false rebounds after unlocking OR flow by anchoring support & tightening volume
        if strategy.entry.logic == "or" and ov.signal_count > 35:
            for cond in strategy.entry.conditions:
                if cond.indicator == "close_to_ma20_pct":
                    _add(
                        [
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target="entry.conditions[indicator=close_to_ma20_pct].indicator",
                                from_value="close_to_ma20_pct",
                                to_value="close_to_ma60_pct",
                                reason=(
                                    "Anchor reversal support on MA60 to reduce choppy rebound whipsaws"
                                ),
                            )
                        ],
                        "Switch support anchor to MA60",
                    )
                elif cond.indicator == "volume_ratio_1d_5d":
                    try:
                        curr_vol = float(cond.value)
                    except (TypeError, ValueError):
                        curr_vol = 2.0
                    if curr_vol < 2.0:
                        _add(
                            [
                                StrategyChange(
                                    change_type=ChangeType.TIGHTEN_FILTER,
                                    target="entry.conditions[indicator=volume_ratio_1d_5d].value",
                                    from_value=curr_vol,
                                    to_value=2.0,
                                    reason=(
                                        "Require stronger volume expansion (2.0x) after OR unlock"
                                    ),
                                )
                            ],
                            "Tighten volume confirmation to 2.0x",
                        )

        # Coordinated exit payoff & volatility calibration bundle
        if ov.signal_count >= 15 and (ov.profit_loss_ratio < 1.8 or ov.avg_return < 0.02):
            exit_bundle: list[StrategyChange] = []
            if (
                strategy.exit.stop_loss.type == "pct"
                and strategy.exit.stop_loss.value is not None
                and float(strategy.exit.stop_loss.value) < 0.08
            ):
                exit_bundle.append(
                    StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.stop_loss.value",
                        from_value=float(strategy.exit.stop_loss.value),
                        to_value=0.08,
                        reason="Widen stop-loss tolerance to 8% against reversal chop",
                    )
                )
            if strategy.exit.max_holding_days < 14:
                exit_bundle.append(
                    StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.max_holding_days",
                        from_value=strategy.exit.max_holding_days,
                        to_value=14,
                        reason="Extend holding horizon to 14d for mean-reversion capture",
                    )
                )
            if (
                strategy.exit.take_profit.type == "rr"
                and strategy.exit.take_profit.value is not None
                and float(strategy.exit.take_profit.value) < 6.0
            ):
                exit_bundle.append(
                    StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.take_profit.value",
                        from_value=float(strategy.exit.take_profit.value),
                        to_value=6.0,
                        reason="Raise take-profit target to 6.0R to capture fat-tail rebounds",
                    )
                )
            if len(exit_bundle) >= 2:
                _add(
                    exit_bundle[:3],
                    "Coordinated stock exit payoff and volatility tolerance calibration",
                )

        # Tunable numeric/indicator neighbors
        for param in strategy.params.tunable:
            current = resolve_tunable_target(strategy, param.target)
            if not isinstance(current, (int, float)):
                continue
            lo, hi = param.range
            step = param.step or (hi - lo) / 8.0
            for raw_cand in (current + step, current - step):
                cand: int | float = (
                    int(round(raw_cand))
                    if is_period_tunable_target(param.target)
                    else round(float(raw_cand), 4)
                )
                if cand == current or cand < lo or cand > hi:
                    continue
                if param.target.startswith("exit."):
                    ctype = ChangeType.ADJUST_EXIT
                elif cand > current:
                    ctype = ChangeType.LOOSEN_FILTER
                else:
                    ctype = ChangeType.TIGHTEN_FILTER
                _add(
                    [
                        StrategyChange(
                            change_type=ctype,
                            target=param.target,
                            from_value=current,
                            to_value=cand,
                            reason=f"Step {param.label or param.target}: {current} -> {cand}",
                        )
                    ],
                    f"Tune {param.target}",
                )

        # Exit risk / holding / payoff adjustments tailored for stock whipsaw & tail payoff
        if strategy.exit.stop_loss.type == "pct" and strategy.exit.stop_loss.value is not None:
            curr_sl = float(strategy.exit.stop_loss.value)
            for delta in (0.02, -0.015, 0.03):
                nxt = round(min(0.18, max(0.02, curr_sl + delta)), 3)
                if abs(nxt - curr_sl) >= 0.005:
                    _add(
                        [
                            StrategyChange(
                                change_type=ChangeType.ADJUST_EXIT,
                                target="exit.stop_loss.value",
                                from_value=curr_sl,
                                to_value=nxt,
                                reason="Calibrate stop-loss tolerance against chop whipsaw",
                            )
                        ],
                        "Stop-loss risk calibration",
                    )

        curr_hold = strategy.exit.max_holding_days
        for new_hold in (curr_hold + 3, curr_hold + 5, max(3, curr_hold - 3)):
            if new_hold != curr_hold and 2 <= new_hold <= 30:
                _add(
                    [
                        StrategyChange(
                            change_type=ChangeType.ADJUST_EXIT,
                            target="exit.max_holding_days",
                            from_value=curr_hold,
                            to_value=new_hold,
                            reason=f"Adjust max holding horizon {curr_hold} -> {new_hold} days",
                        )
                    ],
                    "Holding window tuning",
                )

        if strategy.exit.take_profit.type == "rr" and strategy.exit.take_profit.value is not None:
            curr_tp = float(strategy.exit.take_profit.value)
            for delta in (0.5, 2.0, -0.4):
                nxt_tp = round(min(6.0, max(1.0, curr_tp + delta)), 2)
                if abs(nxt_tp - curr_tp) >= 0.1:
                    _add(
                        [
                            StrategyChange(
                                change_type=ChangeType.ADJUST_EXIT,
                                target="exit.take_profit.value",
                                from_value=curr_tp,
                                to_value=nxt_tp,
                                reason="Calibrate R:R take-profit multiple",
                            )
                        ],
                        "Take-profit payoff adjustment",
                    )

        return out

    def _try_litellm_proposer(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        p_spec: ModelSpec,
    ) -> CouncilProposal | None:
        """Optionally ask ReflectionAnalyzer/LiteLLM for one structured proposal."""
        try:
            from alphaevo.core.config import ConfigManager
            from alphaevo.core.llm import LLMClient
            from alphaevo.reflection.analyzer import ReflectionAnalyzer

            cfg = ConfigManager().load()
            cfg.llm.model = p_spec.model
            api_key = ConfigManager().get_llm_api_key()
            if not api_key:
                return None
            llm = LLMClient(cfg.llm, api_key=api_key)
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

    @staticmethod
    def _try_cli_proposer(
        strategy: Strategy,
        evaluation: EvaluationReport,
        p_spec: ModelSpec,
        mission_notes: str,
    ) -> CouncilProposal | None:
        """Invoke optional local CLI agent (agy / codex) via safe list args."""
        tool = p_spec.cli_tool
        if tool not in {"agy", "codex"}:
            return None
        prompt = (
            f"Strategy {strategy.meta.id} signals={evaluation.overall.signal_count}, "
            f"win_rate={evaluation.overall.win_rate:.2%}, avg_return={evaluation.overall.avg_return:.2%}. "
            f"Mission: {mission_notes[:200]}. "
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
    ) -> list[str]:
        """Optionally invoke adversarial CLI prosecutor; falls back cleanly on unavailability."""
        tool = q_spec.cli_tool
        if tool not in {"agy", "codex"}:
            return []
        change_desc = "; ".join(f"{c.change_type.value}:{c.target}->{c.to_value}" for c in changes)
        prompt = (
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


__all__ = [
    "CouncilDeliberation",
    "CouncilProposal",
    "HeterogeneousCouncil",
    "HeterogeneousCouncilConfig",
    "ModelSpec",
    "ProsecutorAudit",
    "infer_model_family",
]
