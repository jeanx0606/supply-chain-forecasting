"""One field contract per channel (D10)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelContract:
    name: str
    required: tuple
    week_columns: int | None = None
    optional: tuple = ()


SALES_CONTRACT = ChannelContract(
    name="sales",
    required=("external_code", "retail", "release_date"),
    week_columns=12,
    optional=("season", "category", "color", "fabric", "restock"),
)

RESTOCK_CONTRACT = ChannelContract(
    name="restocks",
    required=("external_code", "retail", "week", "year", "qty"),
)

PRICE_CONTRACT = ChannelContract(
    name="price_discount",
    required=("external_code", "retail", "price"),
    week_columns=12,
)

TRENDS_CONTRACT = ChannelContract(name="gtrends", required=("date",))
WEATHER_CONTRACT = ChannelContract(name="weather", required=("locality", "date"))

CATEGORICAL = ("season", "category", "color", "fabric")
KEY_COLUMNS = ("external_code", "retail")
WEATHER_FIELDS = (
    "avg_temp", "min_temp", "max_temp", "dew_point", "humidity", "visibility",
    "avg_wind", "max_wind", "gust", "sea_pressure", "avg_pressure", "rain",
)


@dataclass(frozen=True)
class WeekColumnMap:
    prefix: str
    n_weeks: int

    def label(self, index: int) -> str:
        return f"{self.prefix}{index + 1}"