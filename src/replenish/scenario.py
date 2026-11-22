"""R0-R4 multi-week scenario ladder on the §7.6.3 base-stock rule.

Main setting only: L = 1, R = 1, z = 95%, alpha = 1.0, A3 on. The §7.7 grid is
deliberately not swept here. Between R1 and R4 the rule, lead time, review cycle and
service level are identical; only the forecast (and its validation sigma_e) changes.
R0 replays the observed restocks; R0b applies the opening stock and never reorders.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ..errors import ConfigError
from .scenario_metrics import (ScenarioMetrics, cost_index, fill_rate,
                               overstock_ratio, stockout_rate, turnover,
                               weeks_of_cover)

SCENARIO_LABELS = {
    "R0": "actual restocks replayed (real-world baseline)",
    "R0b": "opening stock only, no reorder",
    "R1": "naive baseline + base-stock rule",
    "R2": "category-curve baseline + base-stock rule",
    "R3": "best model + base-stock rule (proposed)",
    "R4": "perfect foresight + base-stock rule (theoretical upper bound)",
}
FORECAST_SCENARIOS = ("R1", "R2", "R3", "R4")
SIMULATION_LABEL = "simulation under stated assumptions"
COST_INDEX_REASON = "cost ratio cu:co is a §7.7 grid value, not yet locked"


@dataclass(frozen=True)
class ScenarioSettings:
    lead_time_weeks: int = 1
    order_cycle_weeks: int = 1
    service_level: float = 0.95
    n_weeks: int = 12
    forecast_origin_week: int = 2

    def __post_init__(self):
        if int(self.lead_time_weeks) < 0:
            raise ConfigError("scenario.lead_time_weeks", "must be >= 0",
                              observed=self.lead_time_weeks)
        if int(self.order_cycle_weeks) < 1:
            raise ConfigError("scenario.order_cycle_weeks", "must be >= 1",
                              observed=self.order_cycle_weeks)
        if not 0 < float(self.service_level) < 1:
            raise ConfigError("scenario.service_level", "must be in (0, 1)",
                              observed=self.service_level)
        if int(self.n_weeks) <= int(self.forecast_origin_week):
            raise ConfigError("scenario.n_weeks", "must exceed the forecast origin",
                              observed=self.n_weeks)
        if int(self.forecast_origin_week) != 2:
            raise ConfigError(
                "scenario.forecast_origin_week",
                "the ladder is defined for the 2-in/10-out task",
                observed=self.forecast_origin_week, expected=2)

    @property
    def protection_weeks(self) -> int:
        return int(self.lead_time_weeks) + int(self.order_cycle_weeks)

    @property
    def horizon(self) -> int:
        return int(self.n_weeks) - int(self.forecast_origin_week)

    def as_dict(self) -> dict:
        return {"lead_time_weeks": int(self.lead_time_weeks),
                "order_cycle_weeks": int(self.order_cycle_weeks),
                "service_level": float(self.service_level),
                "protection_weeks": self.protection_weeks,
                "n_weeks": int(self.n_weeks),
                "forecast_origin_week": int(self.forecast_origin_week),
                "horizon": self.horizon,
                "simulation_label": SIMULATION_LABEL,
                "cost_index": None,
                "cost_index_reason": COST_INDEX_REASON}


@dataclass(frozen=True)
class ScenarioPair:
    key: tuple
    sales: tuple
    restocks: dict
    opening: float


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario: str
    label: str
    forecast_source: str
    sigma_e: float
    settings: dict
    n_pairs: int
    orders: int
    sold_units: float
    demanded_units: float
    lost_units: float
    supplied_units: float
    leftover_units: float
    stockout_weeks: int
    store_weeks: int
    average_on_hand: float
    metrics: ScenarioMetrics

    @property
    def fill_rate(self):
        return self.metrics.fill_rate

    @property
    def stockout_rate(self):
        return self.metrics.stockout_rate

    @property
    def overstock_ratio(self):
        return self.metrics.overstock_ratio

    @property
    def turnover(self):
        return self.metrics.turnover

    @property
    def weeks_of_cover(self):
        return self.metrics.weeks_of_cover

    def as_dict(self) -> dict:
        return {"scenario": self.scenario, "label": self.label,
                "forecast_source": self.forecast_source,
                "sigma_e": float(self.sigma_e),
                "n_pairs": int(self.n_pairs), "orders": int(self.orders),
                "sold_units": float(self.sold_units),
                "demanded_units": float(self.demanded_units),
                "lost_units": float(self.lost_units),
                "supplied_units": float(self.supplied_units),
                "leftover_units": float(self.leftover_units),
                "stockout_weeks": int(self.stockout_weeks),
                "store_weeks": int(self.store_weeks),
                "average_on_hand": float(self.average_on_hand),
                **self.metrics.as_dict()}


def _target(forecast, review_week: int, settings: ScenarioSettings,
            z_value: float, sigma_e: float) -> float:
    start = int(review_week) - int(settings.forecast_origin_week)
    end = start + settings.protection_weeks
    window = [float(forecast[i]) for i in range(max(0, start),
                                                min(len(forecast), end))]
    return (sum(window)
            + z_value * float(sigma_e) * float(np.sqrt(settings.protection_weeks)))


def simulate_pair(pair: ScenarioPair, forecast, settings: ScenarioSettings,
                  sigma_e: float = 0.0, mode: str = "forecast") -> dict:
    """Run one pair over weeks 1..n; returns totals and the end-of-week on-hand path."""
    if mode not in ("replay", "no_reorder", "forecast"):
        raise ConfigError("scenario.mode", "unsupported simulation mode",
                          observed=mode, expected=["replay", "no_reorder", "forecast"])
    forecast = list(forecast) if forecast is not None else []
    z_value = NormalDist().inv_cdf(float(settings.service_level))
    on_hand = float(pair.opening)
    outstanding = {}
    supplied = float(pair.opening)
    sold = demanded = lost = 0.0
    stockout_weeks = 0
    orders = 0
    on_hand_path = []

    for week in range(1, int(settings.n_weeks) + 1):
        received = outstanding.pop(week, 0.0)
        on_hand += received
        supplied += received

        if mode == "replay":
            restock = float(pair.restocks.get(week, 0.0))
            on_hand += restock
            supplied += restock

        demand = float(pair.sales[week - 1]) if week <= len(pair.sales) else 0.0
        demanded += demand
        taken = min(demand, on_hand)
        sold += taken
        on_hand -= taken
        unmet = demand - taken
        if unmet > 1e-9:
            lost += unmet
            stockout_weeks += 1
        on_hand_path.append(on_hand)

        if mode == "forecast" and (settings.forecast_origin_week
                                   <= week < int(settings.n_weeks)):
            position = on_hand + sum(outstanding.values())
            target = _target(forecast, week, settings, z_value, sigma_e)
            quantity = max(0.0, target - position)
            arrival = week + int(settings.lead_time_weeks) + 1
            if quantity > 1e-9 and arrival <= int(settings.n_weeks):
                outstanding[arrival] = outstanding.get(arrival, 0.0) + quantity
                orders += 1

    average = (float(np.mean(on_hand_path)) if on_hand_path else 0.0)
    return {"sold": sold, "demanded": demanded, "lost": lost,
            "supplied": supplied, "leftover": on_hand_path[-1] if on_hand_path else 0.0,
            "stockout_weeks": stockout_weeks, "average_on_hand": average,
            "orders": orders}


def run_scenario(name: str, pairs, settings: ScenarioSettings,
                 forecasts=None, forecast_source: str = "", sigma_e: float = 0.0,
                 cost_ratio=None) -> ScenarioOutcome:
    if name not in SCENARIO_LABELS:
        raise ConfigError("scenario.name", "unknown scenario", observed=name,
                          expected=list(SCENARIO_LABELS))
    mode = {"R0": "replay", "R0b": "no_reorder"}.get(name, "forecast")
    pairs = list(pairs)
    if mode == "forecast" and forecasts is not None and len(forecasts) != len(pairs):
        raise ConfigError("scenario.forecasts", "one forecast per pair is required",
                          observed=len(forecasts), expected=len(pairs))
    if mode == "forecast" and forecasts is not None:
        horizon = int(settings.horizon)
        short = [index for index, item in enumerate(forecasts)
                 if item is None or len(item) != horizon]
        if short:
            raise ConfigError(
                "scenario.forecasts",
                "every forecast must cover the full horizon",
                observed=f"{len(forecasts[short[0]])} values",
                expected=f"{horizon} values")

    sold = demanded = lost = supplied = leftover = 0.0
    stockout_weeks = 0
    orders = 0
    average_total = 0.0
    cu, co = (cost_ratio if cost_ratio else (None, None))
    for index, pair in enumerate(pairs):
        forecast = None if mode != "forecast" else (
            forecasts[index] if forecasts is not None else None)
        if mode == "forecast" and (forecast is None or len(forecast) == 0):
            raise ConfigError("scenario.forecasts", "forecast arm needs a forecast",
                              observed=None, expected=f"10 values for {name}")
        result = simulate_pair(pair, forecast, settings, sigma_e, mode)
        sold += result["sold"]
        demanded += result["demanded"]
        lost += result["lost"]
        supplied += result["supplied"]
        leftover += result["leftover"]
        stockout_weeks += result["stockout_weeks"]
        orders += result["orders"]
        average_total += result["average_on_hand"]

    store_weeks = len(pairs) * int(settings.n_weeks)
    metrics = ScenarioMetrics(
        fill_rate=fill_rate(sold, demanded),
        stockout_rate=stockout_rate(stockout_weeks, store_weeks),
        overstock_ratio=overstock_ratio(leftover, supplied),
        turnover=turnover(sold, average_total),
        weeks_of_cover=weeks_of_cover(sold, average_total, int(settings.n_weeks)),
        cost_index=cost_index(lost, leftover, cu, co),
    )
    return ScenarioOutcome(
        scenario=name, label=SCENARIO_LABELS[name],
        forecast_source=forecast_source, sigma_e=float(sigma_e),
        settings=settings.as_dict(), n_pairs=len(pairs), orders=int(orders),
        sold_units=sold, demanded_units=demanded, lost_units=lost,
        supplied_units=supplied, leftover_units=leftover,
        stockout_weeks=int(stockout_weeks), store_weeks=int(store_weeks),
        average_on_hand=average_total, metrics=metrics)


def run_ladder(pairs, settings: ScenarioSettings, forecast_sources: dict,
               sigma_by_scenario: dict, cost_ratio=None) -> list:
    """R0/R0b always run; R1-R4 run only when a forecast source was supplied."""
    outcomes = [run_scenario("R0", pairs, settings, cost_ratio=cost_ratio),
                run_scenario("R0b", pairs, settings, cost_ratio=cost_ratio)]
    for name in FORECAST_SCENARIOS:
        source = forecast_sources.get(name)
        if not source:
            continue
        label, forecasts = source
        outcomes.append(run_scenario(
            name, pairs, settings, forecasts=forecasts, forecast_source=label,
            sigma_e=float(sigma_by_scenario.get(name, 0.0)), cost_ratio=cost_ratio))
    return outcomes


def ladder_payload(outcomes, settings: ScenarioSettings,
                   forecast_sources: dict | None = None,
                   cost_ratio=None) -> dict:
    provided = ({outcome.scenario for outcome in outcomes}
                if forecast_sources is None else set(forecast_sources))
    unavailable = [
        {"scenario": name, "reason": "no forecast source supplied"}
        for name in FORECAST_SCENARIOS if name not in provided]
    return {"settings": settings.as_dict(),
            "cost_ratio": list(cost_ratio) if cost_ratio else None,
            "cost_index_reason": COST_INDEX_REASON,
            "unavailable": unavailable,
            "rows": [outcome.as_dict() for outcome in outcomes]}