"""Data layer — adapters, data management, and universe providers."""

from alphaevo.data.adapter import DataAdapter, DataManager
from alphaevo.data.quality import (
    DataQualityFinding,
    DataQualityReport,
    build_data_quality_report,
)
from alphaevo.data.universe import (
    AdapterUniverseProvider,
    CuratedUniverseProvider,
    CustomUniverseProvider,
    UniverseProvider,
)

__all__ = [
    "AdapterUniverseProvider",
    "CuratedUniverseProvider",
    "CustomUniverseProvider",
    "DataAdapter",
    "DataManager",
    "DataQualityFinding",
    "DataQualityReport",
    "UniverseProvider",
    "build_data_quality_report",
]
