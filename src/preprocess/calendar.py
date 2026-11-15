"""One ISO-8601 week convention for restock, weather and lifecycle arithmetic (D20)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..seams.protocols import SeamDescription


@dataclass(frozen=True, order=True)
class IsoWeek:
    """A declared ISO year-week pair; the week starts on Monday."""

    year: int
    week: int

    def to_date(self) -> date:
        return date.fromisocalendar(int(self.year), int(self.week), 1)

    @classmethod
    def from_date(cls, value) -> "IsoWeek":
        if not isinstance(value, date):
            from datetime import datetime
            value = datetime.fromisoformat(str(value)).date()
        iso = value.isocalendar()
        return cls(int(iso.year), int(iso.week))

    def label(self) -> str:
        return f"{self.year:04d}-W{self.week:02d}"


class IsoWeekConvention:
    SEAM_ID = "SEAM-WEEK-CONVENTION"
    implementation = "iso8601"

    def describe(self) -> SeamDescription:
        return SeamDescription("week_convention", "REP-13", self.implementation)

    def to_date(self, year: int, week: int) -> date:
        return IsoWeek(int(year), int(week)).to_date()

    def week_of(self, value) -> IsoWeek:
        return IsoWeek.from_date(value)

    def week_index(self, value) -> tuple[int, int]:
        iso = self.week_of(value)
        return iso.year, iso.week


def make_week_convention(name: str) -> IsoWeekConvention:
    if name != "iso8601":
        from ..errors import ConfigError
        raise ConfigError("time.week_convention", "unsupported convention",
                          observed=name, expected="iso8601")
    return IsoWeekConvention()


def to_monday(value) -> date:
    """Monday of the ISO week containing ``value``."""
    return IsoWeek.from_date(value).to_date()