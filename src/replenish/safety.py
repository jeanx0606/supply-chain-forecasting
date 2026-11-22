"""Safety stock: implemented from the service level, or declared out of scope (REP-10)."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ..errors import ConfigError
from ..seams.protocols import SeamDescription


@dataclass(frozen=True)
class DemandContext:
    demand_slice: tuple
    lead_time_weeks: int
    order_cycle_weeks: int
    service_level: float
    variability: float | None


@dataclass(frozen=True)
class SafetyStockResult:
    quantity: float
    method: str
    service_level_applied: bool
    scope_note: str
    protection_weeks: int = 1


class SafetyStockPolicyBase:
    SEAM_ID = "SEAM-SAFETY-STOCK"

    def __init__(self, service_level: float = 0.95):
        value = float(service_level)
        if not 0.0 < value < 1.0:
            raise ConfigError("replenish.service_level", "outside (0, 1)",
                              observed=value, expected="(0, 1)")
        self.service_level = value

    def describe(self) -> SeamDescription:
        return SeamDescription("safety_stock", "REP-10", self.method)

    def compute(self, ctx: DemandContext) -> SafetyStockResult:
        raise NotImplementedError


class NormalApproxSafetyStock(SafetyStockPolicyBase):
    method = "normal_approx"

    def compute(self, ctx: DemandContext) -> SafetyStockResult:
        variability = ctx.variability
        if variability is None:
            variability = _residual_std(ctx.demand_slice)
        z_value = NormalDist().inv_cdf(self.service_level)
        protection = max(int(ctx.lead_time_weeks) + int(ctx.order_cycle_weeks), 1)
        buffer = max(0.0, z_value * float(variability) * np.sqrt(protection))
        return SafetyStockResult(
            quantity=float(buffer), method=self.method,
            service_level_applied=True,
            scope_note=(f"service level {self.service_level:.2f} applied over "
                        f"protection period P={protection} week(s) (REP-10)"),
            protection_weeks=protection,
        )


class ZeroSafetyStock(SafetyStockPolicyBase):
    method = "none"

    def compute(self, ctx: DemandContext) -> SafetyStockResult:
        _ = ctx.variability
        _ = self.service_level
        return SafetyStockResult(
            quantity=0.0, method=self.method,
            service_level_applied=False,
            scope_note="service level OUT OF SCOPE (REP-10)",
            protection_weeks=max(int(ctx.lead_time_weeks)
                                 + int(ctx.order_cycle_weeks), 1),
        )


def _residual_std(values) -> float:
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 2:
        return 0.0
    return float(np.std(array, ddof=1))


def make_safety_stock(method, service_level: float = 0.95) -> SafetyStockPolicyBase:
    name = getattr(method, "value", method)
    if name == "normal_approx":
        return NormalApproxSafetyStock(service_level)
    if name == "none":
        return ZeroSafetyStock(service_level)
    raise ConfigError("replenish.safety_stock_method", "unregistered method",
                      observed=name, expected=["normal_approx", "none"])