"""Panel registry: exactly six panels, in the declared order (DFX-C5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import forecast, inventory, metrics, overview, replenishment, risk


@dataclass(frozen=True)
class Panel:
    key: str
    label: str
    order: int
    build_view: Callable
    render: Callable


PANEL_REGISTRY: tuple[Panel, ...] = (
    Panel("overview", "P1 Overview", 1, overview.build_view, overview.render),
    Panel("forecast", "P2 Forecast", 2, forecast.build_view, forecast.render),
    Panel("metrics", "P3 Metrics", 3, metrics.build_view, metrics.render),
    Panel("inventory", "P4 Inventory", 4, inventory.build_view, inventory.render),
    Panel("replenishment", "P5 Replenishment", 5,
          replenishment.build_view, replenishment.render),
    Panel("risk", "P6 Risk", 6, risk.build_view, risk.render),
)


def labels() -> list:
    return [panel.label for panel in PANEL_REGISTRY]


def views(bundle) -> dict:
    return {panel.key: panel.build_view(bundle) for panel in PANEL_REGISTRY}