"""ISO week convention: year-boundary correctness and no %W arithmetic (D20)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from src.preprocess.calendar import IsoWeek, IsoWeekConvention, to_monday

ROOT = Path(__file__).resolve().parents[2]


def test_iso_round_trip_at_year_boundary():
    week = IsoWeek(2020, 1)
    assert week.to_date() == date(2019, 12, 30)
    assert IsoWeek.from_date(date(2019, 12, 30)) == IsoWeek(2020, 1)


def test_2019_12_30_maps_to_2020_w01_everywhere():
    convention = IsoWeekConvention()
    assert convention.week_of(date(2019, 12, 30)) == IsoWeek(2020, 1)
    assert convention.to_date(2020, 1) == date(2019, 12, 30)


def test_monday_of_week():
    assert to_monday(date(2019, 12, 31)) == date(2019, 12, 30)


def test_no_percent_w_or_strftime_week_arithmetic():
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in ("%W", "%U", "isoweekday()"):
            if token in text:
                offenders.append((str(path.relative_to(ROOT)), token))
    assert offenders == []