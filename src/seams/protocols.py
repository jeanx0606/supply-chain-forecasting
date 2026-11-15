"""Seam declarations. One marker per decision, one decision per configuration key."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable


@dataclass(frozen=True)
class SeamDescription:
    code: str
    rep_id: str
    implementation: str
    provisional: bool = True

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "rep_id": self.rep_id,
            "implementation": self.implementation,
            "provisional": self.provisional,
        }


@runtime_checkable
class Seam(Protocol):
    SEAM_ID: ClassVar[str]

    def describe(self) -> SeamDescription: ...


@dataclass(frozen=True)
class ProvisionalRef:
    rep_id: str
    confirmer_role: str
    decision: str


@runtime_checkable
class SplitPolicy(Seam, Protocol):
    def resolve(self, release_dates): ...


@runtime_checkable
class PresetSplitPolicy(Seam, Protocol):
    def available(self) -> bool: ...


@runtime_checkable
class GrainAdapter(Seam, Protocol):
    def build(self, frame): ...


@runtime_checkable
class InitialStockStrategy(Seam, Protocol):
    def estimate(self, sales_row, events): ...


@runtime_checkable
class SafetyStockPolicy(Seam, Protocol):
    def compute(self, ctx): ...


@runtime_checkable
class RestockRetentionPolicy(Seam, Protocol):
    def retain(self, events, release_map, n_weeks): ...


@runtime_checkable
class NegativeSalesPolicy(Seam, Protocol):
    def apply(self, values): ...


@runtime_checkable
class WeatherJoinMode(Seam, Protocol):
    def join(self, sales, weekly): ...


@runtime_checkable
class TrendChannelPolicy(Seam, Protocol):
    def status(self, file_present: bool, malformed: bool): ...


@runtime_checkable
class DemoSelector(Seam, Protocol):
    def select(self, candidates): ...


@runtime_checkable
class OutputWriter(Seam, Protocol):
    def write(self, bundle, run_id: str): ...


@runtime_checkable
class ModelBackend(Seam, Protocol):
    def make_ensemble(self, horizon: int): ...


@runtime_checkable
class WeekConvention(Seam, Protocol):
    def to_date(self, year: int, week: int): ...


@runtime_checkable
class BaselineDefinition(Seam, Protocol):
    def make(self, name: str): ...


@runtime_checkable
class AltTaskSpec(Seam, Protocol):
    def build(self, frame, meta): ...


@runtime_checkable
class MetricSet(Seam, Protocol):
    def evaluate(self, y_true, y_pred, meta, model_name): ...


@runtime_checkable
class FeatureSetSpec(Seam, Protocol):
    def expected_columns(self, frame, input_weeks, origin_week): ...

    def assert_closed(self, features, plan): ...


@runtime_checkable
class ModelSelectionPolicy(Seam, Protocol):
    def score_validation(self, predictions, y_val, meta_val, metric_seam, model): ...

    def choose(self, figures, candidates, default_model): ...


@runtime_checkable
class ModelSelection(Seam, Protocol):
    def available(self) -> tuple[str, ...]: ...

    def main(self) -> str: ...