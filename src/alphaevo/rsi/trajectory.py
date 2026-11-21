"""JSONL trajectory and DPO preference pair exporter for Stock RSI evolution runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from alphaevo.rsi.engine import DeliberationRecord, RSIRunResult


class RSITrajectoryStep(BaseModel):
    """Single deliberative transition step recorded during Stock RSI evolution."""

    round_num: int
    parent_strategy_id: str
    secondary_parent_id: str | None = None
    candidate_strategy_id: str | None = None
    proposer_model: str
    proposer_family: str
    prosecutor_model: str
    prosecutor_family: str
    rationale: str = ""
    proposed_changes: list[str] = Field(default_factory=list)
    prosecutor_approved: bool
    prosecutor_objections: list[str] = Field(default_factory=list)
    l0_outcome: str
    veto_reasons: list[str] = Field(default_factory=list)
    fitness_before: float = 0.0
    fitness_after: float | None = None
    dsr_probability: float | None = None
    null_passed: bool | None = None
    signals: int | None = None


class DPOPreferencePair(BaseModel):
    """Direct Preference Optimization pair contrasting accepted vs vetoed/reverted mutations."""

    pair_id: str
    strategy_family: str
    round_num: int
    prompt: str
    chosen: str
    rejected: str
    chosen_outcome: str
    rejected_outcome: str
    fitness_margin: float = 0.0
    rejection_reasons: list[str] = Field(default_factory=list)


class RSITrajectoryExport(BaseModel):
    """Structured evolution trajectory + DPO training flywheel extracted from one RSI session."""

    charter_name: str
    seed_strategy_id: str
    champion_id: str
    steps: list[RSITrajectoryStep] = Field(default_factory=list)
    dpo_pairs: list[DPOPreferencePair] = Field(default_factory=list)


def _format_candidate_action(rec: DeliberationRecord) -> str:
    changes = "; ".join(rec.proposed_changes) or "no-op"
    return f"Rationale: {rec.rationale}\nChanges: {changes}"


def build_rsi_trajectory(result: RSIRunResult) -> RSITrajectoryExport:
    """Extract chronological trajectory steps and contrasted DPO pairs from an RSI run."""
    steps = [
        RSITrajectoryStep(
            round_num=d.round_num,
            parent_strategy_id=d.parent_strategy_id,
            secondary_parent_id=d.secondary_parent_id,
            candidate_strategy_id=d.candidate_strategy_id,
            proposer_model=d.proposer_model,
            proposer_family=d.proposer_family,
            prosecutor_model=d.prosecutor_model,
            prosecutor_family=d.prosecutor_family,
            rationale=d.rationale,
            proposed_changes=list(d.proposed_changes),
            prosecutor_approved=d.prosecutor_approved,
            prosecutor_objections=list(d.prosecutor_objections),
            l0_outcome=d.l0_outcome,
            veto_reasons=list(d.veto_reasons),
            fitness_before=d.fitness_before,
            fitness_after=d.fitness_after,
            dsr_probability=d.dsr_probability,
            null_passed=d.null_passed,
            signals=d.signals,
        )
        for d in result.deliberations
    ]

    winning_pool = [d for d in result.deliberations if d.l0_outcome in {"keep", "stepping_stone"}]
    if not winning_pool:
        evaluated = [
            d
            for d in result.deliberations
            if d.fitness_after is not None and d.fitness_after > d.fitness_before + 1e-6
        ]
        if evaluated:
            winning_pool = [max(evaluated, key=lambda d: float(d.fitness_after or 0.0))]

    rejected_pool = [
        d
        for d in result.deliberations
        if d.l0_outcome in {"prosecutor_vetoed", "revert", "evaluated_non_best"}
        and (bool(d.veto_reasons) or d.l0_outcome == "prosecutor_vetoed")
    ]

    pairs: list[DPOPreferencePair] = []
    seen_pairs: set[tuple[str, str]] = set()
    family = result.trial_family_id

    for w in winning_pool:
        chosen_txt = _format_candidate_action(w)
        w_fit = float(w.fitness_after if w.fitness_after is not None else w.fitness_before)
        same_round_rejected = [r for r in rejected_pool if r.round_num == w.round_num]
        candidates_to_pair = same_round_rejected or rejected_pool
        for r in candidates_to_pair:
            if r is w:
                continue
            r_fit = float(r.fitness_after if r.fitness_after is not None else r.fitness_before)
            is_vetoed = r.l0_outcome == "prosecutor_vetoed" or bool(r.veto_reasons)
            if not is_vetoed and w_fit <= r_fit + 1e-6:
                continue
            rejected_txt = _format_candidate_action(r)
            if chosen_txt == rejected_txt or (chosen_txt, rejected_txt) in seen_pairs:
                continue
            seen_pairs.add((chosen_txt, rejected_txt))
            margin = round(max(0.0, w_fit - r_fit), 6)
            prompt = (
                f"Charter={result.charter_name} | Parent={w.parent_strategy_id} | "
                f"Round={w.round_num} | PriorFitness={w.fitness_before:.4f}. "
                "Propose a verified YAML DSL mutation that improves risk-adjusted edge "
                "without triggering L0 overfit/null/scar vetoes."
            )
            pairs.append(
                DPOPreferencePair(
                    pair_id=f"{family}_r{w.round_num}_{len(pairs) + 1}",
                    strategy_family=family,
                    round_num=w.round_num,
                    prompt=prompt,
                    chosen=chosen_txt,
                    rejected=rejected_txt,
                    chosen_outcome=w.l0_outcome,
                    rejected_outcome=r.l0_outcome,
                    fitness_margin=margin,
                    rejection_reasons=list(r.veto_reasons),
                )
            )
            if len(pairs) >= 50:
                break
        if len(pairs) >= 50:
            break

    return RSITrajectoryExport(
        charter_name=result.charter_name,
        seed_strategy_id=result.seed_strategy_id,
        champion_id=result.champion_id,
        steps=steps,
        dpo_pairs=pairs,
    )


def export_rsi_trajectory(result: RSIRunResult, output_dir: Path) -> dict[str, Path]:
    """Write ``rsi_trajectory.jsonl`` and ``rsi_dpo_pairs.jsonl`` inside *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    export = build_rsi_trajectory(result)

    traj_path = output_dir / "rsi_trajectory.jsonl"
    dpo_path = output_dir / "rsi_dpo_pairs.jsonl"

    traj_lines = [json.dumps(s.model_dump(mode="json"), ensure_ascii=False) for s in export.steps]
    dpo_lines = [
        json.dumps(p.model_dump(mode="json"), ensure_ascii=False) for p in export.dpo_pairs
    ]

    traj_path.write_text("\n".join(traj_lines) + ("\n" if traj_lines else ""), encoding="utf-8")
    dpo_path.write_text("\n".join(dpo_lines) + ("\n" if dpo_lines else ""), encoding="utf-8")
    return {"trajectory": traj_path, "dpo_pairs": dpo_path}


__all__ = [
    "DPOPreferencePair",
    "RSITrajectoryExport",
    "RSITrajectoryStep",
    "build_rsi_trajectory",
    "export_rsi_trajectory",
]
