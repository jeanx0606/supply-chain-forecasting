"""Risk grading - stock-out is a PROXY; no stock-out column exists (C4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ConfigError

GRADES = ("high", "medium", "low", "unknown")


@dataclass(frozen=True)
class RiskAssessment:
    stockout_proxy: str
    overstock: str
    demand_slice_total: float
    inventory_snapshot: float
    thresholds: dict
    proxy_note: str

    def as_dict(self) -> dict:
        return {
            "stockout_proxy": self.stockout_proxy,
            "overstock": self.overstock,
            "demand_slice_total": self.demand_slice_total,
            "inventory_snapshot": self.inventory_snapshot,
            "thresholds": self.thresholds,
            "proxy_note": self.proxy_note,
        }


def grade_stockout(inventory: float, demand, high_weeks: int = 1,
                   medium_weeks: int = 2) -> str:
    values = np.asarray(demand, dtype=float).ravel()
    if values.size == 0:
        return "unknown"
    high_boundary = float(values[:max(1, int(high_weeks))].sum())
    medium_boundary = float(values[:max(1, int(medium_weeks))].sum())
    if float(inventory) < high_boundary:
        return "high"
    if float(inventory) < medium_boundary:
        return "medium"
    return "low"


def grade_overstock(inventory: float, demand, high_ratio: float = 2.0,
                    medium_ratio: float = 1.2) -> str:
    values = np.asarray(demand, dtype=float).ravel()
    total = float(values.sum())
    if total <= 0:
        return "unknown"
    ratio = float(inventory) / total
    if ratio > float(high_ratio):
        return "high"
    if ratio > float(medium_ratio):
        return "medium"
    return "low"


def validate_ratios(high_ratio: float, medium_ratio: float) -> None:
    if float(medium_ratio) >= float(high_ratio):
        raise ConfigError(
            "replenish.overstock_medium_ratio",
            "overstock_medium_ratio must be lower than overstock_high_ratio",
            observed=(medium_ratio, high_ratio), expected="medium < high")


def assess(inventory: float, demand, high_weeks: int = 1, medium_weeks: int = 2,
           high_ratio: float = 2.0, medium_ratio: float = 1.2) -> RiskAssessment:
    validate_ratios(high_ratio, medium_ratio)
    values = np.asarray(demand, dtype=float).ravel()
    return RiskAssessment(
        stockout_proxy=grade_stockout(inventory, values, high_weeks, medium_weeks),
        overstock=grade_overstock(inventory, values, high_ratio, medium_ratio),
        demand_slice_total=round(float(values.sum()), 6),
        inventory_snapshot=float(inventory),
        thresholds={
            "high_weeks": int(high_weeks),
            "medium_weeks": int(medium_weeks),
            "overstock_high_ratio": float(high_ratio),
            "overstock_medium_ratio": float(medium_ratio),
        },
        proxy_note=("stock-out risk is a PROXY inferred from reconstructed inventory - "
                    "no stock-out column exists in the data (C4)"),
    )