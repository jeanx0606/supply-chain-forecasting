"""Inventory: three strategies, identity, sell-out onset, no clamping (D21, D14)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.errors import ConfigError
from src.preprocess.inventory import (
    build_trajectory,
    make_initial_stock_strategy,
)


def row(value=5.0):
    return pd.Series({f"q{n}": value for n in range(1, 13)})


def test_initial_stock_strategies():
    assert make_initial_stock_strategy("lead_time_demand", 1, 1.0).estimate(row(), {}).value == 5.0
    assert make_initial_stock_strategy("first_restock").estimate(row(), {1: 7}).value == 7.0
    assert make_initial_stock_strategy("rolling_minimum").estimate(row(), {}).value == 2.5


def test_unknown_strategy_aborts_listing_valid_names():
    with pytest.raises(ConfigError) as info:
        make_initial_stock_strategy("magic")
    assert "lead_time_demand" in str(info.value)
    assert "first_restock" in str(info.value)
    assert "rolling_minimum" in str(info.value)


def test_reconstruction_identity():
    trajectory = build_trajectory(row(), {1: 10.0}, make_initial_stock_strategy(
        "lead_time_demand", 1, 1.0), n_weeks=12)
    previous = 5.0
    for entry in trajectory.entries:
        previous = previous + entry.restock - entry.sales
        assert abs(previous - entry.reconstructed_stock) < 1e-9


def test_first_week_after_restock():
    trajectory = build_trajectory(row(), {1: 10.0}, make_initial_stock_strategy(
        "lead_time_demand", 1, 1.0), n_weeks=12)
    assert trajectory.at(1).reconstructed_stock == 10.0


def test_sellout_onset_is_the_first_crossing():
    trajectory = build_trajectory(row(), {}, make_initial_stock_strategy(
        "first_restock"), n_weeks=12)
    # first_restock with no event gives I0=0 -> sell-out at week 1
    assert trajectory.sellout_week == 1

    from src.preprocess.inventory import InitialStockEstimate, InitialStockStrategy

    class FixedOpening(InitialStockStrategy):
        name = "lead_time_demand"

        def estimate(self, sales_row, events, opening=0.0):
            return InitialStockEstimate(10.0, self.name, "assumed opening stock")

    sales = pd.Series({f"q{n}": 5.0 for n in range(1, 13)})
    trajectory = build_trajectory(sales, {}, FixedOpening(), n_weeks=12)
    assert trajectory.sellout_week == 2
    assert trajectory.remaining_weeks_after_sellout == 10
    assert all(entry.reconstructed_stock > 0 for entry in trajectory.entries[:1])


def test_negative_stock_not_clamped_and_marker_present():
    sales = pd.Series({f"q{n}": 50.0 for n in range(1, 13)})
    trajectory = build_trajectory(sales, {}, make_initial_stock_strategy(
        "first_restock"), n_weeks=10)
    assert len(trajectory.entries) == 10
    assert trajectory.entries[-1].reconstructed_stock < 0
    assert trajectory.reconstruction_marker == "reconstructed"