"""Weekly index: q1 is the release week, all steps are exactly 7 days (C1, C8)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.errors import ConfigError
from src.preprocess.time_index import (
    attach_week_index,
    build_week_index,
    check_window,
    monday_violations,
    validate_monday,
    window_summary,
)


def test_week_index_length_and_offset():
    index = build_week_index(pd.Timestamp("2019-01-07"), 12)
    assert len(index) == 12
    assert index[1] == pd.Timestamp("2019-01-07")
    assert index[12] == pd.Timestamp("2019-01-07") + pd.Timedelta(weeks=11)


def test_attach_week_index_all_seven_days_apart():
    frame = pd.DataFrame({"release_date": pd.to_datetime(["2019-01-07", "2019-03-04"])})
    out = attach_week_index(frame, 12)
    for n in range(1, 12):
        delta = out[f"week_{n + 1}_date"] - out[f"week_{n}_date"]
        assert (delta == pd.Timedelta(days=7)).all()


def test_monday_violations_are_counted_not_raised():
    series = pd.to_datetime(["2019-01-07", "2019-01-08", "2019-01-14"])
    assert monday_violations(series) == 1
    assert validate_monday(series) is False


def test_window_summary_states_the_combination():
    check_window(2, 10, 12)
    assert window_summary(2, 10, 12) == "2 in / 10 out = 12 of 12 lifecycle weeks"
    with pytest.raises(ConfigError):
        check_window(4, 10, 12)