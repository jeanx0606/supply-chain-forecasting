"""Weather: reducers, gap handling and fan-out-free join (D3, REP-08)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.errors import ConfigError
from src.preprocess.weather import (
    MappedJoin,
    OmittedJoin,
    PooledRepresentativeJoin,
    interpolate_localities,
    to_weekly,
)

ROOT = Path(__file__).resolve().parents[2]
AGG = {"rain": "sum", "max_temp": "max", "humidity": "mean", "avg_temp": "mean"}


def weather(locality=1, days=7, start="2019-01-07", rain=1.0, max_temp=20.0,
            humidity=50.0):
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({
        "locality": locality,
        "date": dates,
        "rain": rain,
        "max_temp": max_temp,
        "humidity": humidity,
        "avg_temp": 10.0,
    })


def test_reducers_apply_per_iso_week():
    weekly = to_weekly(weather(days=7), AGG)
    assert len(weekly) == 1
    assert weekly["wx_rain"].iloc[0] == 7.0
    assert weekly["wx_max_temp"].iloc[0] == 20.0
    assert weekly["wx_humidity"].iloc[0] == 50.0


def test_small_gap_is_filled_before_aggregation():
    frame = weather(days=15)
    frame.loc[3:7, ["rain", "max_temp", "humidity", "avg_temp"]] = float("nan")
    filled, gaps = interpolate_localities(frame, max_days=30)
    assert gaps == []
    weekly = to_weekly(filled, AGG)
    assert len(weekly) == 3
    assert weekly["wx_rain"].notna().all()


def test_gap_beyond_limit_is_reported_and_run_continues():
    frame = weather(days=60)
    frame.loc[10:50, ["rain", "max_temp", "humidity", "avg_temp"]] = float("nan")
    filled, gaps = interpolate_localities(frame, max_days=5)
    assert gaps and gaps[0]["locality"] == 1
    assert gaps[0]["missing_days"] > 0


def test_join_preserves_record_count_with_twenty_localities():
    days = 60
    frames = [weather(locality=loc, days=days, rain=float(loc)) for loc in range(1, 21)]
    weekly = to_weekly(pd.concat(frames, ignore_index=True), AGG)
    sales = pd.DataFrame({
        "external_code": range(1000),
        "retail": 1,
        "week_1_date": pd.to_datetime(
            (pd.Timestamp("2019-01-07") + pd.to_timedelta(
                pd.Series(range(1000)) % days, unit="D"))),
    })
    joined = PooledRepresentativeJoin().join(sales, weekly)
    assert len(joined) == 1000
    assert sum(c.startswith("wx_") for c in joined.columns) == len(AGG)


def test_omitted_skips_and_mapped_requires_a_map():
    sales = pd.DataFrame({"week_1_date": pd.to_datetime(["2019-01-07"])})
    weekly = to_weekly(weather(), AGG)
    assert list(OmittedJoin().join(sales, weekly).columns) == ["week_1_date"]
    with pytest.raises(ConfigError):
        MappedJoin().join(sales, weekly)


def test_weather_file_has_exactly_one_join_site():
    text = (ROOT / "src" / "preprocess" / "weather.py").read_text(encoding="utf-8")
    assert text.count(".merge(") == 1