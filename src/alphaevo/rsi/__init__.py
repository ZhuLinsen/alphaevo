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
from alphaevo.rsi.diagnostics import (
    ConditionPassRate,
    StrategyBottleneckReport,
    analyze_entry_bottlenecks,
)
from alphaevo.rsi.engine import (
    DeliberationRecord,
    RSIRoundRow,
    RSIRunResult,
    StockRSIEngine,
)
from alphaevo.rsi.mechanism import (
    HarnessVariant,
    MechanismEvaluator,
    MechanismExperimentResult,
    MechanismFoldComparison,
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
from alphaevo.rsi.trajectory import (
    DPOPreferencePair,
    RSITrajectoryExport,
    RSITrajectoryStep,
    build_rsi_trajectory,
    export_rsi_trajectory,
)

__all__ = [
    "ArchiveCell",
    "ArchiveSummary",
    "BlindSpotCluster",
    "BlindSpotMiner",
    "CoSteerDecision",
    "ConditionPassRate",
    "CouncilDeliberation",
    "CouncilProposal",
    "DPOPreferencePair",
    "DeliberationRecord",
    "ExecutableScar",
    "FailureClusterId",
    "HarnessVariant",
    "HeterogeneousCouncil",
    "HeterogeneousCouncilConfig",
    "MAPElitesArchive",
    "MechanismEvaluator",
    "MechanismExperimentResult",
    "MechanismFoldComparison",
    "ModelSpec",
    "NicheCoord",
    "OfflineFactorTemplate",
    "PairedMechanismRun",
    "ProsecutorAudit",
    "RSICharter",
    "RSIRoundRow",
    "RSIRunResult",
    "RSITrajectoryExport",
    "RSITrajectoryStep",
    "ScarRegistry",
    "StockRSIEngine",
    "StrategyBottleneckReport",
    "analyze_entry_bottlenecks",
    "build_rsi_trajectory",
    "classify_niche",
    "export_rsi_trajectory",
    "infer_model_family",
    "parse_rsi_program",
]
