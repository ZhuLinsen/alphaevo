"""Karpathy-style program.md charter parser for Stock RSI loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from alphaevo.models.enums import RSILevel


class RSICharter(BaseModel):
    """Structured constitution controlling what an autonomous Stock RSI loop may mutate."""

    name: str = "stock_rsi_default"
    seed_strategy_id: str = "rsi_reversion_v1"
    objective: Literal["robust_expectancy", "confidence", "deflated_sharpe"] = "robust_expectancy"
    allowed_levels: list[RSILevel] = Field(
        default_factory=lambda: [RSILevel.L1_STRATEGY, RSILevel.L2_FACTOR]
    )
    max_rounds: int = Field(default=6, ge=1, le=50)
    max_candidates_per_round: int = Field(default=3, ge=1, le=10)
    max_trials_budget: int = Field(default=50, ge=1)
    min_signals: int = Field(default=20, ge=1)
    max_drawdown: float = Field(default=0.30, gt=0.0, le=0.80)
    min_dsr_prob: float = Field(default=0.20, ge=0.0, le=1.0)
    max_complexity: int = Field(default=8, ge=1, le=20)
    require_null_control: bool = True
    null_permutations: int = Field(default=60, ge=10)
    mission_notes: str = ""


def parse_rsi_program(source: str | Path) -> RSICharter:
    """Parse a ``program.md`` file (YAML frontmatter + mission Markdown) or YAML string."""
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"RSI program file not found: {source}")
        text = source.read_text(encoding="utf-8")
    else:
        maybe_path = Path(source)
        text = maybe_path.read_text(encoding="utf-8") if maybe_path.is_file() else source

    stripped = text.strip()
    meta: dict[str, Any] = {}
    mission = stripped

    if stripped.startswith("---"):
        parts = stripped.split("---", 2)
        if len(parts) >= 3:
            loaded = yaml.safe_load(parts[1])
            if isinstance(loaded, dict):
                meta = loaded
            mission = parts[2].strip()
    else:
        loaded = yaml.safe_load(stripped)
        if isinstance(loaded, dict):
            meta = loaded
            mission = str(loaded.get("mission_notes", ""))

    if "mission_notes" not in meta or not meta["mission_notes"]:
        meta["mission_notes"] = mission
    return RSICharter.model_validate(meta)


__all__ = ["RSICharter", "parse_rsi_program"]
