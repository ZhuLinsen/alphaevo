"""Canonical Pydantic contracts for immutable research run provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

RunKind = Literal["run", "evolve", "optimize", "rsi"]
ProtocolName = Literal["fast_research_v1", "full_research_v1", "official_v1"]
DataReproducibility = Literal["replayable", "refetch_dependent", "non_reproducible"]
DataMutability = Literal[
    "immutable_bundle",
    "replayable_snapshot",
    "mutable_provider",
    "unknown",
]


class StrategyProvenanceInfo(BaseModel):
    """Identity and content digest of the evaluated strategy snapshot."""

    id: str
    version: int
    family_id: str = ""
    snapshot_hash: str


class CodeVersionInfo(BaseModel):
    """Git/package version state recorded at run time."""

    source: str = "git"
    commit: str = "unknown"
    dirty: bool = True
    diff_hash: str | None = None


class DataProvenanceInfo(BaseModel):
    """Data source, window, and reproducibility contract for one run."""

    adapter: str = "unknown"
    date_range: tuple[str, str] | None = None
    fingerprint_hash: str
    reproducibility: DataReproducibility = "refetch_dependent"
    data_mutability: DataMutability = "mutable_provider"
    data_bundle_id: str | None = None
    symbol_count: int = 0


class ConfigProvenanceInfo(BaseModel):
    """Digest of the allowlisted non-secret configuration snapshot."""

    snapshot_hash: str


class DeterminismInfo(BaseModel):
    """Random seed and reproducibility policy markers."""

    random_seed: int = 42
    policy: list[str] = Field(default_factory=list)


class ArtifactPathsInfo(BaseModel):
    """Relative file paths stored inside the immutable artifact directory."""

    artifact_dir: str
    report: str = "report.md"
    evaluation: str = "evaluation.json"
    trades: str = "trades.jsonl"
    strategy_snapshot: str = "strategy_snapshot.yaml"
    initial_strategy_snapshot: str | None = None
    config_snapshot: str = "config_snapshot.json"
    data_fingerprint: str = "data_fingerprint.json"


class RunSummaryInfo(BaseModel):
    """Headline metrics mirrored for fast manifest & index inspection."""

    score: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    profit_loss_ratio: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    signal_count: int = 0
    metrics_basis: str = "trade_sequence"


class RunManifest(BaseModel):
    """Immutable run manifest written to ``manifest.json`` per evaluation run."""

    run_id: str
    kind: RunKind
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    requested_protocol: ProtocolName
    effective_protocol: ProtocolName
    protocol_downgrade_reason: str | None = None
    strategy: StrategyProvenanceInfo
    code_version: CodeVersionInfo
    data: DataProvenanceInfo
    config: ConfigProvenanceInfo
    determinism: DeterminismInfo = Field(default_factory=DeterminismInfo)
    artifacts: ArtifactPathsInfo
    summary: RunSummaryInfo
    status: str = "completed"


class RunIndexEntry(BaseModel):
    """Compact SQLite index row pointing to the authoritative ``manifest.json``."""

    run_id: str
    kind: RunKind
    strategy_id: str
    strategy_version: int = 1
    requested_protocol: ProtocolName
    effective_protocol: ProtocolName
    created_at: datetime
    artifact_dir: str
    score: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    profit_loss_ratio: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    signal_count: int = 0
    code_dirty: bool = True
    data_reproducibility: DataReproducibility = "refetch_dependent"
    status: str = "completed"


class RunComparisonResult(BaseModel):
    """High-signal comparison between two immutable run manifests."""

    run_id_1: str
    run_id_2: str
    comparable: bool
    incomparability_reasons: list[str] = Field(default_factory=list)
    protocol_1: ProtocolName
    protocol_2: ProtocolName
    strategy_hash_changed: bool = False
    config_hash_changed: bool = False
    data_fingerprint_changed: bool = False
    score_delta: float = 0.0
    win_rate_delta: float = 0.0
    avg_return_delta: float = 0.0
    max_drawdown_delta: float = 0.0
    sharpe_ratio_delta: float = 0.0
    signal_count_delta: int = 0
    changed_strategy_paths: list[str] = Field(default_factory=list)


__all__ = [
    "ArtifactPathsInfo",
    "CodeVersionInfo",
    "ConfigProvenanceInfo",
    "DataMutability",
    "DataProvenanceInfo",
    "DataReproducibility",
    "DeterminismInfo",
    "ProtocolName",
    "RunComparisonResult",
    "RunIndexEntry",
    "RunKind",
    "RunManifest",
    "RunSummaryInfo",
    "StrategyProvenanceInfo",
]
