"""L1 + L2 + L3 Recursive Self-Improvement (RSI) Engine for Stock Research."""

from alphaevo.rsi.co_steer import (
    BlindSpotCluster,
    BlindSpotMiner,
    CoSteerDecision,
    ExecutableScar,
    FailureClusterId,
    OfflineFactorTemplate,
    ScarRegistry,
)
from alphaevo.rsi.council import (
    CouncilDeliberation,
    CouncilProposal,
    HeterogeneousCouncil,
    HeterogeneousCouncilConfig,
    ModelSpec,
    ProsecutorAudit,
    infer_model_family,
)
from alphaevo.rsi.engine import RSIRoundRow, RSIRunResult, StockRSIEngine
from alphaevo.rsi.mechanism import (
    HarnessVariant,
    MechanismEvaluator,
    MechanismExperimentResult,
    PairedMechanismRun,
)
from alphaevo.rsi.program import RSICharter, parse_rsi_program
from alphaevo.rsi.qd_archive import (
    ArchiveCell,
    ArchiveSummary,
    MAPElitesArchive,
    NicheCoord,
    classify_niche,
)

__all__ = [
    "ArchiveCell",
    "ArchiveSummary",
    "BlindSpotCluster",
    "BlindSpotMiner",
    "CoSteerDecision",
    "CouncilDeliberation",
    "CouncilProposal",
    "ExecutableScar",
    "FailureClusterId",
    "HarnessVariant",
    "HeterogeneousCouncil",
    "HeterogeneousCouncilConfig",
    "MAPElitesArchive",
    "MechanismEvaluator",
    "MechanismExperimentResult",
    "ModelSpec",
    "NicheCoord",
    "OfflineFactorTemplate",
    "PairedMechanismRun",
    "ProsecutorAudit",
    "RSICharter",
    "RSIRoundRow",
    "RSIRunResult",
    "ScarRegistry",
    "StockRSIEngine",
    "classify_niche",
    "infer_model_family",
    "parse_rsi_program",
]
