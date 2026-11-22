"""Scenario metrics with the Chapter 3 / Chapter 7 formulas."""
from __future__ import annotations

from dataclasses import dataclass


def safe_ratio(numerator, denominator) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator = float(denominator)
    if denominator <= 0:
        return None
    return float(numerator) / denominator


def fill_rate(sold, demanded) -> float | None:
    """Units sold / units demanded."""
    return safe_ratio(sold, demanded)


def stockout_rate(stockout_weeks, store_weeks) -> float | None:
    """Share of pair-weeks where stock reached zero while demand remained."""
    return safe_ratio(stockout_weeks, store_weeks)


def overstock_ratio(leftover, supplied) -> float | None:
    """Leftover units at the horizon end / total units supplied."""
    return safe_ratio(leftover, supplied)


def turnover(sold, average_on_hand) -> float | None:
    """Units sold / average on-hand stock."""
    return safe_ratio(sold, average_on_hand)


def weeks_of_cover(sold, average_on_hand, n_weeks: int = 12) -> float | None:
    """Average on-hand / average weekly sales; None when nothing sold."""
    rate = safe_ratio(sold, n_weeks)
    if not rate:
        return None
    return safe_ratio(average_on_hand, rate)


def cost_index(lost_units, leftover_units, cu, co) -> float | None:
    """cu x lost + co x leftover; None until the §7.7 cost ratio is locked."""
    if cu is None or co is None:
        return None
    return float(cu) * float(lost_units) + float(co) * float(leftover_units)


@dataclass(frozen=True)
class ScenarioMetrics:
    fill_rate: float | None
    stockout_rate: float | None
    overstock_ratio: float | None
    turnover: float | None
    weeks_of_cover: float | None
    cost_index: float | None

    def as_dict(self) -> dict:
        return {"fill_rate": self.fill_rate,
                "stockout_rate": self.stockout_rate,
                "overstock_ratio": self.overstock_ratio,
                "turnover": self.turnover,
                "weeks_of_cover": self.weeks_of_cover,
                "cost_index": self.cost_index}