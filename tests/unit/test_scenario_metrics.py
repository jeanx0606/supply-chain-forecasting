"""Scenario metrics formulas (Chapter 3 / 7 definitions)."""
from __future__ import annotations

import pytest

from src.replenish.scenario_metrics import (cost_index, fill_rate, overstock_ratio,
                                           stockout_rate, turnover, weeks_of_cover)


def test_fill_rate_is_units_sold_over_units_demanded():
    assert fill_rate(30, 40) == pytest.approx(0.75)
    assert fill_rate(0, 0) is None
    assert fill_rate(5, 0) is None


def test_stockout_rate_is_the_share_of_store_weeks():
    assert stockout_rate(2, 10) == pytest.approx(0.2)
    assert stockout_rate(0, 10) == 0.0
    assert stockout_rate(0, 0) is None


def test_overstock_ratio_is_leftover_over_supplied():
    assert overstock_ratio(4, 40) == pytest.approx(0.1)
    assert overstock_ratio(4, 0) is None


def test_turnover_and_weeks_of_cover_are_reciprocal_in_units():
    assert turnover(120, 30) == pytest.approx(4.0)
    assert weeks_of_cover(120, 30, 12) == pytest.approx(3.0)
    assert weeks_of_cover(0, 30, 12) is None
    assert turnover(120, 0) is None


def test_cost_index_stays_none_until_the_ratio_is_locked():
    assert cost_index(5, 3, None, None) is None
    assert cost_index(5, 3, 1.0, 2.0) == pytest.approx(11.0)