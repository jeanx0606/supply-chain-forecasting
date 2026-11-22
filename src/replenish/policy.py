"""The (s, S) recommendation on one decision week, plus the baseline comparison."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .safety import DemandContext, SafetyStockPolicyBase


@dataclass(frozen=True)
class ReplenishmentContext:
    key: tuple | None
    decision_week: int
    forecast_origin_week: int
    inventory_at_decision_week: float
    demand_slice: tuple
    lead_time_weeks: int
    order_cycle_weeks: int


@dataclass(frozen=True)
class AssumptionBlock:
    lead_time_weeks: int
    order_cycle_weeks: int
    service_level: float
    service_level_applied: bool
    scope_note: str
    assumed_statement: str
    protection_weeks: int = 1
    z_value: float | None = None
    simulation_label: str = "simulation under stated assumptions"


@dataclass(frozen=True)
class ReplenishmentRecommendation:
    reorder_point: float
    target_level: float
    recommended_qty: int
    trigger: bool
    decision_week: int
    forecast_origin_week: int
    assumptions: AssumptionBlock
    forecast_unavailable: bool
    quantity_vs_timing_caption: str
    inventory_snapshot: float
    provisional: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "reorder_point": self.reorder_point,
            "target_level": self.target_level,
            "recommended_qty": self.recommended_qty,
            "trigger": self.trigger,
            "decision_week": self.decision_week,
            "forecast_origin_week": self.forecast_origin_week,
            "assumptions": {
                "lead_time_weeks": self.assumptions.lead_time_weeks,
                "order_cycle_weeks": self.assumptions.order_cycle_weeks,
                "service_level": self.assumptions.service_level,
                "service_level_applied": self.assumptions.service_level_applied,
                "scope_note": self.assumptions.scope_note,
                "assumed_statement": self.assumptions.assumed_statement,
                "protection_weeks": self.assumptions.protection_weeks,
                "z_value": self.assumptions.z_value,
                "simulation_label": self.assumptions.simulation_label,
            },
            "forecast_unavailable": self.forecast_unavailable,
            "quantity_vs_timing_caption": self.quantity_vs_timing_caption,
            "inventory_snapshot": self.inventory_snapshot,
        }


@dataclass(frozen=True)
class BaselineComparison:
    target_level: float
    recommended_qty: int
    target_weeks: int
    rule_name: str = "fixed_cycle_fixed_level"

    def as_dict(self) -> dict:
        return {"target_level": self.target_level,
                "recommended_qty": self.recommended_qty,
                "target_weeks": self.target_weeks,
                "rule_name": self.rule_name}


def recommend(ctx: ReplenishmentContext, safety: SafetyStockPolicyBase,
              variability: float | None = None) -> ReplenishmentRecommendation:
    demand = np.asarray(ctx.demand_slice, dtype=float).ravel()
    unavailable = demand.size == 0
    if unavailable:
        demand = np.zeros(ctx.lead_time_weeks + ctx.order_cycle_weeks)

    lead = max(int(ctx.lead_time_weeks), 1)
    cycle = max(int(ctx.order_cycle_weeks), 1)
    lead_demand = float(demand[:lead].sum())
    cycle_demand = float(demand[lead:lead + cycle].sum())

    buffer = safety.compute(DemandContext(
        demand_slice=tuple(demand.tolist()), lead_time_weeks=lead,
        order_cycle_weeks=cycle, service_level=safety.service_level,
        variability=variability))

    reorder_point = lead_demand + buffer.quantity
    target_level = max(reorder_point, reorder_point + cycle_demand)
    quantity = max(0, int(round(target_level - float(ctx.inventory_at_decision_week))))

    assumptions = AssumptionBlock(
        lead_time_weeks=lead,
        order_cycle_weeks=cycle,
        service_level=safety.service_level,
        service_level_applied=buffer.service_level_applied,
        scope_note=buffer.scope_note,
        assumed_statement=("lead time, service level and order cycle are ASSUMED - "
                           "no order dates, lead times or costs exist in the data (C5)"),
        protection_weeks=buffer.protection_weeks,
        z_value=buffer.z_value,
    )
    return ReplenishmentRecommendation(
        reorder_point=round(reorder_point, 4),
        target_level=round(target_level, 4),
        recommended_qty=quantity,
        trigger=bool(float(ctx.inventory_at_decision_week) <= reorder_point),
        decision_week=int(ctx.decision_week),
        forecast_origin_week=int(ctx.forecast_origin_week),
        assumptions=assumptions,
        forecast_unavailable=unavailable,
        quantity_vs_timing_caption=("quantity is comparable to observed restock shipments; "
                                    "timing is assumption-driven"),
        inventory_snapshot=float(ctx.inventory_at_decision_week),
    )


def recommend_baseline(ctx: ReplenishmentContext, target_weeks: int = 2) -> BaselineComparison:
    demand = np.asarray(ctx.demand_slice, dtype=float).ravel()
    target = float(demand[:max(1, int(target_weeks))].sum()) if demand.size else 0.0
    quantity = max(0, int(round(target - float(ctx.inventory_at_decision_week))))
    return BaselineComparison(target_level=round(target, 4), recommended_qty=quantity,
                              target_weeks=int(target_weeks))


def sS_policy(reconstructed_stock: float, forecast_demand,
              lead_time_weeks: int = 1, order_cycle_weeks: int = 1,
              safety=None, service_level: float = 0.95) -> dict:
    """Functional facade kept for direct unit use."""
    from .safety import make_safety_stock

    policy = safety or make_safety_stock("normal_approx", service_level)
    ctx = ReplenishmentContext(None, lead_time_weeks, lead_time_weeks,
                               reconstructed_stock, tuple(np.asarray(
                                   forecast_demand, dtype=float).ravel().tolist()),
                               lead_time_weeks, order_cycle_weeks)
    return recommend(ctx, policy).as_dict()