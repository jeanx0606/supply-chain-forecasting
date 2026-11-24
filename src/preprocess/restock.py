"""Restock retention and lifecycle-week attribution (D2, D11, REP-04)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..seams.protocols import SeamDescription
from .calendar import IsoWeekConvention


@dataclass(frozen=True)
class RetentionOutcome:
    events_by_key: dict
    excluded: dict
    policy_name: str
    restock_date_max_state: dict
    opening_by_key: dict = field(default_factory=dict)
    first_offset_by_key: dict = field(default_factory=dict)


class RestockRetentionPolicy:
    SEAM_ID = "SEAM-RESTOCK-TRUNC"

    def __init__(self, n_weeks: int = 12, cutoff: str = "2019-12-31"):
        self.n_weeks = int(n_weeks)
        self.cutoff = pd.Timestamp(cutoff).date()
        self.weeks = IsoWeekConvention()
        self.policy_name = "lifecycle_membership"

    def describe(self) -> SeamDescription:
        return SeamDescription("restock_trunc", "REP-04", self.policy_name)

    def _excluded(self) -> dict:
        return {"opening_stock": 0, "after_lifecycle": 0,
                "calendar_cutoff": 0, "no_matching_record": 0}

    def retain(self, events: pd.DataFrame, release_map: dict,
               n_weeks: int | None = None) -> RetentionOutcome:
        horizon = int(n_weeks or self.n_weeks)
        excluded = self._excluded()
        grouped: dict = {}
        opening: dict = {}
        first_offset: dict = {}
        for row in events.itertuples(index=False):
            key = (int(row.external_code), int(row.retail))
            release = release_map.get(key)
            if release is None:
                excluded["no_matching_record"] += 1
                continue
            event_date = self.weeks.to_date(int(row.year), int(row.week))
            offset = (event_date - release).days // 7
            first_offset[key] = min(first_offset.get(key, offset), offset)
            lifecycle_week = offset + 1
            if self.enabled_cutoff() and event_date > self.cutoff:
                excluded["calendar_cutoff"] += 1
                continue
            quantity = float(row.qty)
            if quantity < 0:
                quantity = 0.0
            if lifecycle_week <= 1:
                opening[key] = opening.get(key, 0.0) + quantity
                excluded["opening_stock"] += 1
                continue
            if lifecycle_week > horizon:
                excluded["after_lifecycle"] += 1
                continue
            grouped.setdefault(key, {})[int(lifecycle_week)] = (
                grouped.setdefault(key, {}).get(int(lifecycle_week), 0.0) + quantity)

        state = {"parameter": "truncation.restock_date_max",
                 "value": str(self.cutoff), "state": "inactive_parameter"}
        if self.enabled_cutoff():
            state["state"] = "applied"
        return RetentionOutcome(grouped, excluded, self.policy_name, state,
                                opening, first_offset)

    def enabled_cutoff(self) -> bool:
        return False


class CalendarCutoffRetention(RestockRetentionPolicy):
    def __init__(self, n_weeks: int = 12, cutoff: str = "2019-12-31"):
        super().__init__(n_weeks, cutoff)
        self.policy_name = "calendar_cutoff"

    def enabled_cutoff(self) -> bool:
        return True


def make_retention_policy(policy, cutoff: str, n_weeks: int = 12) -> RestockRetentionPolicy:
    name = getattr(policy, "value", policy)
    if name == "calendar_cutoff":
        return CalendarCutoffRetention(n_weeks, cutoff)
    if name == "lifecycle_membership":
        return RestockRetentionPolicy(n_weeks, cutoff)
    from ..errors import ConfigError
    raise ConfigError("truncation.restock_policy", "unsupported retention policy",
                      observed=name,
                      expected=["lifecycle_membership", "calendar_cutoff"])