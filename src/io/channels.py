"""Channel status registry: no declared channel may be silently absent (D6, REP-09)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..seams.protocols import SeamDescription


class ChannelName(str, Enum):
    SALES = "sales"
    RESTOCKS = "restocks"
    PRICE_DISCOUNT = "price_discount"
    GTRENDS = "gtrends"
    WEATHER = "weather"


class ChannelState(str, Enum):
    CONSUMED = "consumed"
    DEFERRED = "deferred"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ChannelStatus:
    name: str
    value: str
    reason: str
    file: str | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value,
                "reason": self.reason, "file": self.file}


class ChannelRegistry:
    def __init__(self, statuses):
        self._statuses = {status.name: status for status in statuses}

    def status(self, name) -> ChannelStatus:
        key = getattr(name, "value", name)
        return self._statuses[key]

    def all(self) -> tuple:
        order = [name.value for name in ChannelName]
        return tuple(self._statuses[k] for k in order if k in self._statuses)

    def as_list(self) -> list:
        return [status.as_dict() for status in self.all()]


class TrendsPolicy:
    SEAM_ID = "SEAM-TRENDS"

    def __init__(self, policy="deferred"):
        self.policy = getattr(policy, "value", policy)

    def describe(self) -> SeamDescription:
        return SeamDescription("trends", "REP-09", self.policy)

    def status(self, file_present: bool, malformed: bool = False) -> ChannelStatus:
        if self.policy == "deferred":
            return ChannelStatus("gtrends", ChannelState.DEFERRED.value,
                                 "out_of_scope", "vis2_gtrends_data.csv")
        if not file_present:
            return ChannelStatus("gtrends", ChannelState.SKIPPED.value,
                                 "file_absent", "vis2_gtrends_data.csv")
        if malformed:
            return ChannelStatus("gtrends", ChannelState.SKIPPED.value,
                                 "malformed", "vis2_gtrends_data.csv")
        if self.policy == "skip":
            return ChannelStatus("gtrends", ChannelState.SKIPPED.value,
                                 "out_of_scope", "vis2_gtrends_data.csv")
        return ChannelStatus("gtrends", ChannelState.CONSUMED.value,
                             "consumed", "vis2_gtrends_data.csv")


def _file_state(path: Path) -> tuple[bool, bool]:
    if not path.exists():
        return False, False
    try:
        header = path.open("r", encoding="utf-8").readline()
    except OSError:
        return True, True
    return True, not bool(header.strip())


def probe_channels(cfg) -> ChannelRegistry:
    raw = cfg.data.raw_dir
    files = cfg.data.files
    statuses = []
    for name, fname in ((ChannelName.SALES, files.sales),
                        (ChannelName.RESTOCKS, files.restocks),
                        (ChannelName.PRICE_DISCOUNT, files.price_discount)):
        present, malformed = _file_state(raw / fname)
        if not present:
            statuses.append(ChannelStatus(name.value, ChannelState.SKIPPED.value,
                                          "file_absent", fname))
        elif malformed:
            statuses.append(ChannelStatus(name.value, ChannelState.SKIPPED.value,
                                          "malformed", fname))
        else:
            statuses.append(ChannelStatus(name.value, ChannelState.CONSUMED.value,
                                          "consumed", fname))

    present, malformed = _file_state(raw / files.weather)
    if getattr(cfg.channels.weather, "value", cfg.channels.weather) == "skip":
        statuses.append(ChannelStatus("weather", ChannelState.SKIPPED.value,
                                      "out_of_scope", files.weather))
    elif not present:
        statuses.append(ChannelStatus("weather", ChannelState.SKIPPED.value,
                                      "file_absent", files.weather))
    elif malformed:
        statuses.append(ChannelStatus("weather", ChannelState.SKIPPED.value,
                                      "malformed", files.weather))
    else:
        statuses.append(ChannelStatus("weather", ChannelState.CONSUMED.value,
                                      "consumed", files.weather))

    present, malformed = _file_state(raw / files.gtrends)
    trends = TrendsPolicy(cfg.channels.gtrends)
    statuses.append(trends.status(present, malformed))
    return ChannelRegistry(statuses)