"""Risk grading: monotone, configurable, unknown when demand is zero (C4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.errors import ConfigError
from src.replenish.risk import assess, grade_overstock, grade_stockout

ORDER = {"high": 2, "medium": 1, "low": 0, "unknown": -1}


def test_grade_levels():
    demand = [10.0] * 10
    assert grade_stockout(1.0, demand) == "high"
    assert grade_stockout(15.0, demand) == "medium"
    assert grade_stockout(100.0, demand) == "low"
    assert grade_overstock(1.0, demand) == "low"
    assert grade_overstock(500.0, demand) == "high"


def test_unknown_when_no_demand():
    assert grade_stockout(10.0, []) == "unknown"
    assert grade_overstock(10.0, []) == "unknown"


def test_stockout_risk_monotone_non_increasing():
    demand = [10.0] * 10
    grades = [ORDER[grade_stockout(stock, demand)] for stock in range(0, 300, 10)]
    assert all(a >= b for a, b in zip(grades, grades[1:]))


def test_overstock_risk_monotone_non_decreasing():
    demand = [10.0] * 10
    grades = [ORDER[grade_overstock(stock, demand)] for stock in range(0, 500, 25)]
    assert all(a <= b for a, b in zip(grades, grades[1:]))


def test_threshold_change_changes_grade():
    demand = [10.0] * 10
    assert grade_stockout(15.0, demand, high_weeks=1, medium_weeks=2) == "medium"
    assert grade_stockout(15.0, demand, high_weeks=2, medium_weeks=2) == "high"


def test_ratio_order_enforced():
    with pytest.raises(ConfigError) as info:
        assess(10.0, [1.0], medium_ratio=2.0, high_ratio=2.0)
    assert "overstock_medium_ratio" in str(info.value)


def test_assessment_carries_proxy_note_and_thresholds():
    result = assess(5.0, [10.0] * 10)
    assert "PROXY" in result.proxy_note
    assert result.thresholds["high_weeks"] == 1
    assert result.thresholds["overstock_high_ratio"] == 2.0