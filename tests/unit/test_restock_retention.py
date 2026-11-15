"""Restock retention: lifecycle attribution from dates, exclusions by reason (D2, D11)."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.preprocess.calendar import IsoWeek
from src.preprocess.restock import (
    CalendarCutoffRetention,
    RestockRetentionPolicy,
)


def events(rows):
    return pd.DataFrame(rows, columns=["external_code", "retail", "week", "year", "qty"])


def test_lifecycle_week_from_date_difference_not_calendar_week():
    release = date(2019, 3, 4)
    event = events([(1, 2, 11, 2019, 8)])
    outcome = RestockRetentionPolicy(n_weeks=12).retain(event, {(1, 2): release})
    assert outcome.events_by_key[(1, 2)] == {2: 8.0}


def test_week_after_lifecycle_excluded_and_counted():
    release = date(2019, 3, 4)
    late = release + timedelta(weeks=12)
    week = IsoWeek.from_date(late)
    event = events([(1, 2, week.week, week.year, 5)])
    outcome = RestockRetentionPolicy(n_weeks=12).retain(event, {(1, 2): release})
    assert (1, 2) not in outcome.events_by_key
    assert outcome.excluded["after_lifecycle"] == 1


def test_before_release_excluded():
    release = date(2019, 3, 4)
    week = IsoWeek.from_date(release - timedelta(weeks=2))
    event = events([(1, 2, week.week, week.year, 5)])
    outcome = RestockRetentionPolicy(n_weeks=12).retain(event, {(1, 2): release})
    assert outcome.excluded["before_release"] == 1


def test_no_matching_record_counted():
    event = events([(9, 9, 3, 2019, 5)])
    outcome = RestockRetentionPolicy(n_weeks=12).retain(event, {})
    assert outcome.excluded["no_matching_record"] == 1


def test_late_2019_release_retains_2020_shipments():
    release = date(2019, 12, 2)
    rows = []
    for week in range(1, 5):
        point = IsoWeek(2020, week).to_date()
        lifecycle = (point - release).days // 7 + 1
        rows.append((1, 2, week, 2020, float(lifecycle)))
    outcome = RestockRetentionPolicy(n_weeks=12).retain(events(rows), {(1, 2): release})
    assert outcome.events_by_key[(1, 2)] == {5: 5.0, 6: 6.0, 7: 7.0, 8: 8.0}
    assert outcome.policy_name == "lifecycle_membership"


def test_calendar_cutoff_excludes_and_counts():
    release = date(2019, 12, 2)
    rows = [(1, 2, week, 2020, 5.0) for week in range(2, 6)]
    outcome = CalendarCutoffRetention(n_weeks=12, cutoff="2019-12-31").retain(
        events(rows), {(1, 2): release})
    assert (1, 2) not in outcome.events_by_key
    assert outcome.excluded["calendar_cutoff"] == 4


def test_calendar_cutoff_keeps_2019_12_30_shipment():
    release = date(2019, 12, 2)
    outcome = CalendarCutoffRetention(n_weeks=12, cutoff="2019-12-31").retain(
        events([(1, 2, 1, 2020, 4.0)]), {(1, 2): release})
    assert outcome.events_by_key[(1, 2)] == {5: 4.0}


def test_cutoff_parameter_state_is_reported():
    release = date(2019, 6, 3)
    outcome = RestockRetentionPolicy(n_weeks=12).retain(
        events([(1, 2, 20, 2019, 3)]), {(1, 2): release})
    assert outcome.restock_date_max_state["parameter"] == "truncation.restock_date_max"
    assert outcome.restock_date_max_state["state"] == "inactive_parameter"