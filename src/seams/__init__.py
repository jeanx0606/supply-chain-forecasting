"""Seam package: protocols, catalogue and the single resolution factory."""
from .catalogue import BY_CODE, BY_REP, ENTRIES, SeamEntry, get, markers
from .factory import SeamBundle, build_seams
from .protocols import (
    AltTaskSpec,
    BaselineDefinition,
    DemoSelector,
    GrainAdapter,
    InitialStockStrategy,
    MetricSet,
    ModelBackend,
    ModelSelection,
    NegativeSalesPolicy,
    OutputWriter,
    PresetSplitPolicy,
    ProvisionalRef,
    RestockRetentionPolicy,
    SafetyStockPolicy,
    Seam,
    SeamDescription,
    SplitPolicy,
    TrendChannelPolicy,
    WeatherJoinMode,
    WeekConvention,
)

__all__ = [
    "AltTaskSpec", "BY_CODE", "BY_REP", "BaselineDefinition", "DemoSelector",
    "ENTRIES", "GrainAdapter", "InitialStockStrategy", "MetricSet",
    "ModelBackend", "ModelSelection", "OutputWriter", "PresetSplitPolicy",
    "ProvisionalRef", "RestockRetentionPolicy", "SafetyStockPolicy", "Seam",
    "SeamBundle", "SeamDescription", "SeamEntry", "SplitPolicy",
    "TrendChannelPolicy", "WeatherJoinMode", "WeekConvention", "build_seams",
    "get", "markers",
]