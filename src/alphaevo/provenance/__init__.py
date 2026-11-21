"""Immutable research run provenance contracts, artifact storage & comparison."""

from alphaevo.provenance.contracts import (
    ArtifactPathsInfo,
    CodeVersionInfo,
    ConfigProvenanceInfo,
    DataMutability,
    DataProvenanceInfo,
    DataReproducibility,
    DeterminismInfo,
    ProtocolName,
    RunComparisonResult,
    RunIndexEntry,
    RunKind,
    RunManifest,
    RunSummaryInfo,
    StrategyProvenanceInfo,
)
from alphaevo.provenance.store import (
    RunArtifactStore,
    build_data_fingerprint_payload,
    build_safe_config_snapshot,
    detect_code_version,
    evaluate_protocol_gate,
)

__all__ = [
    "ArtifactPathsInfo",
    "CodeVersionInfo",
    "ConfigProvenanceInfo",
    "DataMutability",
    "DataProvenanceInfo",
    "DataReproducibility",
    "DeterminismInfo",
    "ProtocolName",
    "RunArtifactStore",
    "RunComparisonResult",
    "RunIndexEntry",
    "RunKind",
    "RunManifest",
    "RunSummaryInfo",
    "StrategyProvenanceInfo",
    "build_data_fingerprint_payload",
    "build_safe_config_snapshot",
    "detect_code_version",
    "evaluate_protocol_gate",
]
