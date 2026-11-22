"""Typed result bundle, completeness validation and the decision-week invariant."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import StageError
from ..evaluate.aggregate import MetricRecord
from ..preprocess.inventory import InventoryTrajectory, WeeklyInventoryEntry
from ..replenish.policy import BaselineComparison, ReplenishmentRecommendation
from ..replenish.risk import RiskAssessment
from ..seams.protocols import ProvisionalRef, SeamDescription


@dataclass(frozen=True)
class UnitProvenance:
    """O9: reported unit, applied factor, sign policy and the register revision."""

    unit: str = "raw_sales_units"
    applied_scale_factor: float | None = None
    scale_factor_source: str = "none"
    sign_policy: str = "clip_at_zero"
    negative_cells: int = 0
    register_version: int = 2

    def as_dict(self) -> dict:
        return {
            "unit": self.unit,
            "applied_scale_factor": self.applied_scale_factor,
            "scale_factor_source": self.scale_factor_source,
            "sign_policy": self.sign_policy,
            "negative_cells": int(self.negative_cells),
            "register_version": int(self.register_version),
        }


@dataclass(frozen=True)
class QualityReport:
    n_rows: int
    n_items: int
    n_stores: int
    date_min: str
    date_max: str
    duplicate_count: int
    pair_coverage: float
    excluded: dict
    monday_violations: int = 0
    weekly_only_marker: str = "weekly"
    restock_policy: str = "lifecycle_membership"
    restock_date_max_state: dict = field(default_factory=dict)
    negative_sales_policy: str = "clip_at_zero"
    negative_cells: int = 0
    negative_cells_by_week: dict = field(default_factory=dict)
    unit: str = "raw_sales_units"
    applied_scale_factor: float | None = None

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_items": self.n_items,
            "n_stores": self.n_stores, "date_min": self.date_min,
            "date_max": self.date_max, "duplicate_count": self.duplicate_count,
            "pair_coverage": self.pair_coverage, "excluded": self.excluded,
            "monday_violations": self.monday_violations,
            "weekly_only_marker": self.weekly_only_marker,
            "restock_policy": self.restock_policy,
            "restock_date_max_state": self.restock_date_max_state,
            "negative_sales_policy": self.negative_sales_policy,
            "negative_cells": self.negative_cells,
            "negative_cells_by_week": self.negative_cells_by_week,
            "unit": self.unit,
            "applied_scale_factor": self.applied_scale_factor,
        }


@dataclass(frozen=True)
class SplitProvenance:
    cut_train_end: str
    cut_validation_end: str
    counts: dict
    method: str
    preset: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"cut_train_end": self.cut_train_end,
                "cut_validation_end": self.cut_validation_end,
                "counts": self.counts, "method": self.method,
                "preset": self.preset}


MANDATORY_MEMBERS = (
    "quality", "metrics", "main_model", "split", "demonstration",
    "replenishment", "replenishment_baseline", "risk", "channels", "seams",
    "provisional", "available_models", "self_test", "unit_provenance",
    "register_version", "feature_set", "selection", "conditional",
    "scenarios", "error_scale", "error_slices", "reconstruction_audit",
)


@dataclass
class ResultBundle:
    effective_config: dict
    quality: QualityReport
    metrics: dict
    main_model: str
    split: SplitProvenance
    demonstration: InventoryTrajectory
    replenishment: ReplenishmentRecommendation
    replenishment_baseline: BaselineComparison
    risk: RiskAssessment
    channels: tuple
    seams: tuple
    provisional: tuple
    available_models: tuple
    self_test: bool
    run_id: str = "run"
    seed: int = 42
    alt_task: list = field(default_factory=list)
    artefact_dir: str | None = None
    n_evaluated_series: int = 0
    demo_rule: str = "max_horizon_demand"
    demo_forecast: dict = field(default_factory=dict)
    unit_provenance: UnitProvenance | None = None
    register_version: int | None = None
    secondary_comparison: dict | None = None
    feature_set: dict | None = None
    error_scale: dict | None = None
    error_scale_by_model: dict | None = None
    assumptions: list | None = None
    scenarios: dict | None = None
    error_slices: dict | None = None
    reconstruction_audit: dict | None = None
    selection: dict | None = None
    conditional: dict | None = None
    main_model_status: str = "configured"
    main_model_note: str = ""

    def validate_complete(self) -> None:
        missing = [name for name in MANDATORY_MEMBERS if getattr(self, name, None) is None]
        if missing:
            raise StageError("publish", f"missing bundle member: {missing[0]}",
                             next_command="python -m codes.run_pipeline")
        if not self.metrics:
            raise StageError("publish", "missing bundle member: metrics",
                             next_command="python -m codes.run_pipeline")
        if not isinstance(self.metrics, dict):
            raise StageError("publish", "metrics must be a non-empty mapping")
        if not self.provisional:
            raise StageError("publish", "provisional reference list is empty")
        if not self.split.preset.get("statement"):
            raise StageError("publish", "preset partition statement is missing")
        if int(self.selection.get("test_reads", 0)) != len(self.metrics):
            raise StageError(
                "publish",
                f"selection.test_reads ({self.selection.get('test_reads')}) "
                f"!= reported metrics ({len(self.metrics)})")
        if not self.feature_set.get("n_columns"):
            raise StageError("publish", "feature_set carries no columns")
        if not self.conditional:
            raise StageError("publish", "conditional gate statement is missing")

        recommendation = self.replenishment
        if int(recommendation.decision_week) != int(recommendation.forecast_origin_week):
            raise StageError(
                "publish",
                f"decision_week ({recommendation.decision_week}) != "
                f"forecast_origin_week ({recommendation.forecast_origin_week})")
        snapshot = self.demonstration.at(int(recommendation.decision_week))
        if abs(float(snapshot.reconstructed_stock)
               - float(recommendation.inventory_snapshot)) > 1e-6:
            raise StageError(
                "publish",
                f"inventory snapshot week {snapshot.week_no} does not match the "
                f"decision week {recommendation.decision_week}")

    def to_dict(self) -> dict:
        return {
            "effective_config": self.effective_config,
            "quality": self.quality.as_dict(),
            "metrics": {name: record.as_dict() for name, record in self.metrics.items()},
            "main_model": self.main_model,
            "split": self.split.as_dict(),
            "demonstration": {
                "key": list(self.demonstration.key) if self.demonstration.key else None,
                "entries": self.demonstration.as_rows(),
                "initial_stock": {
                    "value": self.demonstration.initial_stock.value,
                    "strategy_name": self.demonstration.initial_stock.strategy_name,
                    "assumption_note": self.demonstration.initial_stock.assumption_note,
                    "rep_id": self.demonstration.initial_stock.rep_id,
                    "base_value": self.demonstration.initial_stock.base_value,
                    "correction": self.demonstration.initial_stock.correction,
                },
                "sellout_week": self.demonstration.sellout_week,
                "remaining_weeks_after_sellout":
                    self.demonstration.remaining_weeks_after_sellout,
                "reconstruction_marker": self.demonstration.reconstruction_marker,
                "reconstruction": {
                    "policy": self.demonstration.policy,
                    "opening_stock_correction":
                        self.demonstration.opening_stock_correction,
                    "min_raw_stock": self.demonstration.min_raw_stock,
                    "negative_weeks": self.demonstration.negative_weeks,
                    "would_go_negative": self.demonstration.would_go_negative,
                },
            },
            "replenishment": self.replenishment.as_dict(),
            "replenishment_baseline": self.replenishment_baseline.as_dict(),
            "risk": self.risk.as_dict(),
            "channels": [status.as_dict() if hasattr(status, "as_dict") else status
                         for status in self.channels],
            "seams": [desc.as_dict() if hasattr(desc, "as_dict") else desc
                      for desc in self.seams],
            "provisional": [
                {"rep_id": ref.rep_id, "confirmer_role": ref.confirmer_role,
                 "decision": ref.decision}
                for ref in self.provisional
            ],
            "available_models": list(self.available_models),
            "self_test": bool(self.self_test),
            "run_id": self.run_id,
            "seed": int(self.seed),
            "alt_task": self.alt_task,
            "artefact_dir": self.artefact_dir,
            "n_evaluated_series": int(self.n_evaluated_series),
            "demo_rule": self.demo_rule,
            "demo_forecast": self.demo_forecast,
            "unit_provenance": self.unit_provenance.as_dict()
            if self.unit_provenance else None,
            "register_version": self.register_version,
            "secondary_comparison": self.secondary_comparison,
            "feature_set": self.feature_set,
            "error_scale": self.error_scale,
            "error_scale_by_model": self.error_scale_by_model,
            "assumptions": self.assumptions,
            "scenarios": self.scenarios,
            "error_slices": self.error_slices,
            "reconstruction_audit": self.reconstruction_audit,
            "selection": self.selection,
            "conditional": self.conditional,
            "main_model_status": self.main_model_status,
            "main_model_note": self.main_model_note,
        }

    def metric(self, model: str) -> MetricRecord:
        return self.metrics[model]