"""Deterministic demonstration selection - never row order (D5, REP-11)."""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import StageError
from ..seams.protocols import SeamDescription


@dataclass(frozen=True)
class DemoCandidate:
    key: tuple
    horizon_demand: float
    row_index: int


class MaxHorizonDemandSelector:
    SEAM_ID = "SEAM-DEMO-SELECT"
    rule = "max_horizon_demand"

    def describe(self) -> SeamDescription:
        return SeamDescription("demo_select", "REP-11", self.rule)

    def select(self, candidates) -> DemoCandidate:
        items = list(candidates)
        if not items:
            raise StageError("demo", "no test-partition candidate available",
                             next_command="python -m codes.sample_data --overwrite")
        ordered = sorted(items, key=lambda c: (int(c.key[0]), int(c.key[1])))
        return max(ordered, key=lambda c: float(c.horizon_demand))


class PinnedSelector(MaxHorizonDemandSelector):
    rule = "pinned"

    def __init__(self, item: int, store: int):
        self.item = int(item)
        self.store = int(store)

    def select(self, candidates) -> DemoCandidate:
        items = list(candidates)
        for candidate in items:
            if candidate.key == (self.item, self.store):
                return candidate
        raise StageError("demo", f"pinned key ({self.item}, {self.store}) not in the "
                                f"test partition")


def make_demo_selector(rule, pin_item=None, pin_store=None) -> MaxHorizonDemandSelector:
    name = getattr(rule, "value", rule)
    if name == "pinned":
        if pin_item is None or pin_store is None:
            from ..errors import ConfigError
            raise ConfigError("demo.pin_item", "pinned selection needs both keys",
                              observed=(pin_item, pin_store), expected="item and store")
        return PinnedSelector(pin_item, pin_store)
    return MaxHorizonDemandSelector()