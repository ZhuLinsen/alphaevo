"""Alpha Factory — LLM-driven factor synthesis, validation, and registration.

This module enables AlphaEvo to *invent* new technical indicators from OHLCV
data, validate them via IC/IR analysis, and dynamically register them into
the IndicatorRegistry for use in strategy evolution.
"""

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
from alphaevo.alpha_factory.factory import (
    AlphaFactory,
    DiscoveryResult,
    load_registered_factors,
    register_factor_record,
)
from alphaevo.alpha_factory.governance import (
    FactorAuditEntry,
    FactorGovernanceConfig,
    FactorGovernanceEngine,
    FactorGovernanceReport,
    FactorHealthStatus,
)
from alphaevo.alpha_factory.sandbox import FactorSandbox, SandboxResult
from alphaevo.alpha_factory.synthesizer import FactorHypothesis, FactorSynthesizer
from alphaevo.alpha_factory.validator import (
    FactorValidator,
    ValidationResult,
    ValidationThresholds,
)

__all__ = [
    "AlphaFactory",
    "DiscoveryResult",
    "FactorAuditEntry",
    "FactorGovernanceConfig",
    "FactorGovernanceEngine",
    "FactorGovernanceReport",
    "FactorHealthStatus",
    "FactorHypothesis",
    "FactorRecord",
    "FactorSandbox",
    "FactorStore",
    "FactorSynthesizer",
    "FactorValidator",
    "SandboxResult",
    "ValidationResult",
    "ValidationThresholds",
    "load_registered_factors",
    "register_factor_record",
]
