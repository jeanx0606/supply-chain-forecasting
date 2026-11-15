"""Weekly time index: q1 is the release week, q12 is release + 11 weeks (C1, C8)."""
from __future__ import annotations

import pandas as pd

from ..errors import ConfigError


def build_week_index(release_date, n_weeks: int = 12) -> dict:
    base = pd.Timestamp(release_date)
    return {n: base + pd.Timedelta(weeks=n - 1) for n in range(1, n_weeks + 1)}


def attach_week_index(sales: pd.DataFrame, n_weeks: int = 12) -> pd.DataFrame:
    frame = sales.copy()
    release = pd.to_datetime(frame["release_date"], errors="coerce")
    for n in range(1, n_weeks + 1):
        frame[f"week_{n}_date"] = release + pd.Timedelta(weeks=n - 1)
    return frame


def monday_violations(series) -> int:
    values = pd.to_datetime(pd.Series(series), errors="coerce").dropna()
    if values.empty:
        return 0
    return int((values.dt.weekday != 0).sum())


def validate_monday(series: pd.Series) -> bool:
    return monday_violations(series) == 0


def window_summary(input_weeks: int, output_weeks: int, n_weeks: int = 12) -> str:
    return f"{input_weeks} in / {output_weeks} out = {input_weeks + output_weeks} of {n_weeks} lifecycle weeks"


def check_window(input_weeks: int, output_weeks: int, n_weeks: int = 12) -> None:
    if input_weeks < 1 or output_weeks < 1:
        raise ConfigError("spine.input_weeks", "windows must be positive",
                          observed=(input_weeks, output_weeks), expected=">= 1")
    if input_weeks + output_weeks > n_weeks:
        raise ConfigError(
            "spine.input_weeks",
            f"input({input_weeks}) + output({output_weeks}) > n_weeks({n_weeks})",
            observed=input_weeks + output_weeks, expected=f"<= {n_weeks}")