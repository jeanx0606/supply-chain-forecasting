"""Daily weather to ISO-week aggregation and a fan-out-free join (D3, REP-08)."""
from __future__ import annotations

import pandas as pd

from ..errors import ConfigError, StageError
from ..io.schema import WEATHER_FIELDS
from ..seams.protocols import SeamDescription


def interpolate_localities(weather: pd.DataFrame, max_days: int = 30):
    """Linear interpolation inside each locality before weekly aggregation."""
    frames = []
    gaps = []
    for locality, group in weather.groupby("locality", observed=True):
        group = group.sort_values("date").set_index("date")
        if group.empty or pd.isna(group.index.min()):
            continue
        full = pd.date_range(group.index.min(), group.index.max(), freq="D")
        before = group.reindex(full)
        numeric = [c for c in WEATHER_FIELDS if c in before.columns]
        missing_before = int(before[numeric].isna().any(axis=1).sum())
        filled = before.copy()
        filled[numeric] = filled[numeric].interpolate(
            method="linear", limit=max_days, limit_direction="both")
        missing_after = int(filled[numeric].isna().any(axis=1).sum())
        if missing_after:
            gaps.append({"locality": int(locality), "missing_days": missing_after})
        filled["locality"] = locality
        frames.append(filled.reset_index().rename(columns={"index": "date"}))
        del missing_before
    if not frames:
        return weather.copy(), []
    return pd.concat(frames, ignore_index=True), gaps


def to_weekly(weather: pd.DataFrame, agg_map: dict) -> pd.DataFrame:
    """One aggregate per (locality, ISO week); wx_ prefixed columns."""
    frame = weather.copy()
    iso = pd.DatetimeIndex(frame["date"]).isocalendar()
    frame["iso_year"] = iso.year.to_numpy()
    frame["iso_week"] = iso.week.to_numpy()

    reducers = {field: reducer for field, reducer in agg_map.items() if field in frame.columns}
    if not reducers:
        return pd.DataFrame(columns=["locality", "iso_year", "iso_week"])
    weekly = (frame.groupby(["locality", "iso_year", "iso_week"], observed=True)
              .agg(reducers)
              .reset_index()
              .rename(columns={field: f"wx_{field}" for field in reducers}))
    return weekly


class WeatherJoinModeBase:
    SEAM_ID = "SEAM-WEATHER-JOIN"
    implementation = "pooled_representative"

    def describe(self) -> SeamDescription:
        return SeamDescription("weather_join", "REP-08", self.implementation)

    def _assert_cardinality(self, sales: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
        if len(joined) != len(sales):
            raise StageError(
                "weather", f"weather join changed record count "
                           f"({len(sales)} -> {len(joined)}) - fan-out detected")
        return joined

    def join(self, sales: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PooledRepresentativeJoin(WeatherJoinModeBase):
    implementation = "pooled_representative"

    def join(self, sales: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
        if "week_1_date" not in sales.columns or weekly.empty:
            return sales.copy()
        reduced = (weekly.groupby(["iso_year", "iso_week"], observed=True)
                   .mean(numeric_only=True)
                   .reset_index())
        iso = pd.DatetimeIndex(sales["week_1_date"]).isocalendar()
        work = sales.copy()
        work["iso_year"] = iso.year.to_numpy()
        work["iso_week"] = iso.week.to_numpy()
        joined = work.merge(reduced, on=["iso_year", "iso_week"], how="left")
        return self._assert_cardinality(sales, joined)


class OmittedJoin(WeatherJoinModeBase):
    implementation = "omitted"

    def join(self, sales: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
        return sales.copy()


class MappedJoin(WeatherJoinModeBase):
    implementation = "mapped"

    def __init__(self, locality_map: dict | None = None):
        self.locality_map = locality_map

    def join(self, sales: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
        if not self.locality_map:
            raise ConfigError("weather.join_mode",
                              "weather.join_mode=mapped requires a store-to-locality map",
                              observed="mapped", expected="a locality map or pooled_representative")
        return sales.copy()


def make_weather_join(mode, locality_map: dict | None = None) -> WeatherJoinModeBase:
    name = getattr(mode, "value", mode)
    if name == "pooled_representative":
        return PooledRepresentativeJoin()
    if name == "omitted":
        return OmittedJoin()
    if name == "mapped":
        return MappedJoin(locality_map)
    raise ConfigError("weather.join_mode", "unregistered join mode", observed=name,
                      expected=["pooled_representative", "omitted", "mapped"])


def weather_feature_columns(frame: pd.DataFrame) -> list:
    return [c for c in frame.columns if str(c).startswith("wx_")]