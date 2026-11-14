"""L0 Immutable Anti-Delusion Truth Kernel package."""

from alphaevo.truth.ledger import (
    DeflatedSharpeResult,
    TrialLedger,
    TrialRecord,
    compute_data_fingerprint,
    compute_deflated_sharpe,
    compute_kernel_fingerprint,
    compute_strategy_hash,
)
from alphaevo.truth.null_control import (
    NullControlReport,
    NullDistributionStats,
    NullSignalController,
    SystemNullAuditReport,
    SystemNullWorldResult,
    generate_block_shuffled_market,
)
from alphaevo.truth.pit_probe import (
    PITProbeResult,
    PITProbeViolation,
    probe_point_in_time_invariance,
)

__all__ = [
    "DeflatedSharpeResult",
    "NullControlReport",
    "NullDistributionStats",
    "NullSignalController",
    "PITProbeResult",
    "PITProbeViolation",
    "SystemNullAuditReport",
    "SystemNullWorldResult",
    "TrialLedger",
    "TrialRecord",
    "compute_data_fingerprint",
    "compute_deflated_sharpe",
    "compute_kernel_fingerprint",
    "compute_strategy_hash",
    "generate_block_shuffled_market",
    "probe_point_in_time_invariance",
]
