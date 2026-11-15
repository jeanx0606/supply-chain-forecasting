"""Inventory trajectory reconstruction; I0 is assumed, the trajectory is not (C3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..errors import ConfigError
from ..seams.protocols import SeamDescription


@dataclass(frozen=True)
class InitialStockEstimate:
    value: float
    strategy_name: str
    assumption_note: str
    rep_id: str = "REP-07"


@dataclass(frozen=True)
class WeeklyInventoryEntry:
    week_no: int
    restock: float
    sales: float
    reconstructed_stock: float


@dataclass(frozen=True)
class InventoryTrajectory:
    entries: tuple
    initial_stock: InitialStockEstimate
    sellout_week: int | None
    remaining_weeks_after_sellout: int | None
    key: tuple | None = None
    reconstruction_marker: str = field(default="reconstructed", init=False)

    def at(self, week_no: int) -> WeeklyInventoryEntry:
        for entry in self.entries:
            if entry.week_no == int(week_no):
                return entry
        raise KeyError(f"week {week_no} outside the reconstructed horizon")

    def as_rows(self) -> list:
        return [
            {"week_no": e.week_no, "restock": e.restock, "sales": e.sales,
             "reconstructed_stock": e.reconstructed_stock}
            for e in self.entries
        ]


def _q(row, n: int) -> float:
    try:
        value = row.get(f"q{n}", 0.0)
    except AttributeError:
        value = row[f"q{n}"] if f"q{n}" in row else 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value != value:
        return 0.0
    return value


class InitialStockStrategy:
    SEAM_ID = "SEAM-INITIAL-STOCK"
    name = "base"

    def describe(self) -> SeamDescription:
        return SeamDescription("initial_stock", "REP-07", self.name)

    def estimate(self, sales_row, events) -> InitialStockEstimate:
        raise NotImplementedError


class LeadTimeDemandStrategy(InitialStockStrategy):
    name = "lead_time_demand"

    def __init__(self, lead_time_weeks: int = 1, safety_factor: float = 1.0):
        self.lead_time_weeks = max(1, int(lead_time_weeks))
        self.safety_factor = float(safety_factor)

    def estimate(self, sales_row, events) -> InitialStockEstimate:
        base = sum(_q(sales_row, n) for n in range(1, self.lead_time_weeks + 1))
        value = round(base * self.safety_factor, 6)
        return InitialStockEstimate(
            value=value, strategy_name=self.name,
            assumption_note=(f"I0 assumed as the first {self.lead_time_weeks} week(s) of demand "
                             f"times safety factor {self.safety_factor}; absolute level uncertain"))

    def __call__(self, sales_row, events):
        estimate = self.estimate(sales_row, events)
        return estimate.value


class FirstRestockStrategy(InitialStockStrategy):
    name = "first_restock"

    def estimate(self, sales_row, events) -> InitialStockEstimate:
        if not events:
            return InitialStockEstimate(
                0.0, self.name,
                "initial stock assumed equal to the first restock; no restock event exists")
        first_week = min(events)
        return InitialStockEstimate(
            float(events[first_week]), self.name,
            "initial stock assumed equal to the first restock shipment")

    def __call__(self, sales_row, events):
        return self.estimate(sales_row, events).value


class RollingMinimumStrategy(InitialStockStrategy):
    name = "rolling_minimum"

    def estimate(self, sales_row, events) -> InitialStockEstimate:
        values = [_q(sales_row, n) for n in range(1, 13)]
        nonzero = [value for value in values if value > 0]
        base = (min(nonzero) / 2.0) if nonzero else 0.0
        return InitialStockEstimate(
            float(base), self.name,
            "initial stock assumed as half the smallest non-zero weekly sale")

    def __call__(self, sales_row, events):
        return self.estimate(sales_row, events).value


VALID_METHODS = ("lead_time_demand", "first_restock", "rolling_minimum")


def make_initial_stock_strategy(method, lead_time_weeks: int = 1,
                                safety_factor: float = 1.0) -> InitialStockStrategy:
    name = getattr(method, "value", method)
    if name == "lead_time_demand":
        return LeadTimeDemandStrategy(lead_time_weeks, safety_factor)
    if name == "first_restock":
        return FirstRestockStrategy()
    if name == "rolling_minimum":
        return RollingMinimumStrategy()
    raise ConfigError("inventory.initial_stock_method", "unknown strategy",
                      observed=name, expected=list(VALID_METHODS))


def build_trajectory(sales_row, events: dict, strategy: InitialStockStrategy,
                     n_weeks: int = 12, key=None) -> InventoryTrajectory:
    estimate = strategy.estimate(sales_row, events)
    stock = float(estimate.value)
    entries = []
    sellout_week = None
    for n in range(1, n_weeks + 1):
        restock = float(events.get(n, 0.0))
        sales = _q(sales_row, n)
        stock = stock + restock - sales
        entries.append(WeeklyInventoryEntry(n, restock, sales, round(stock, 6)))
        if sellout_week is None and stock <= 0:
            sellout_week = n
    remaining = (n_weeks - sellout_week) if sellout_week is not None else None
    return InventoryTrajectory(tuple(entries), estimate, sellout_week, remaining, key=key)


def initial_stock_value(strategy: InitialStockStrategy, sales_row, events) -> float:
    return float(strategy.estimate(sales_row, events).value)