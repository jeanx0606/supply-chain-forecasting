"""Chapter 6/7 acceptance criteria (section 6.7 / 8) asserted in code."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate.aggregate import StandardMetricSet
from src.preprocess.inventory import build_trajectory, initial_stock_value, \
    make_initial_stock_strategy
from src.preprocess.split import ChronologicalSplit
from src.replenish.policy import ReplenishmentContext, recommend
from src.replenish.risk import assess
from src.replenish.safety import make_safety_stock

RAW_COLUMNS = [f"q{n}" for n in range(1, 13)]


def sales_row():
    return {column: 4.0 for column in RAW_COLUMNS}


def test_weekly_time_units_are_consecutive_weeks():
    trajectory = build_trajectory(
        sales_row(), {1: 2.0},
        make_initial_stock_strategy("lead_time_demand", 1, 1.0), n_weeks=12)
    weeks = [entry.week_no for entry in trajectory.entries]
    assert weeks == list(range(1, 13))
    assert all(entry.week_no == int(entry.week_no) for entry in trajectory.entries)


def test_chronological_split_never_shuffles_and_cuts_are_ordered():
    dates = pd.Series(pd.date_range("2019-01-07", periods=40, freq="W-MON"))
    assignment = ChronologicalSplit(0.70, 0.15, 0.15).resolve(dates)
    assert assignment.method == "chronological"
    assert assignment.cut_train_end < assignment.cut_validation_end
    assert assignment.train.iloc[:28].all()
    assert assignment.test.iloc[-1]
    assert sum(assignment.counts.values()) == len(dates)


def test_reconstructed_prefix_is_on_estimated_inventory_fields():
    trajectory = build_trajectory(
        sales_row(), {}, make_initial_stock_strategy("first_restock"), n_weeks=12)
    row = trajectory.as_rows()[0]
    assert "reconstructed_stock" in row
    assert "stock" not in row
    assert trajectory.reconstruction_marker == "reconstructed"


def test_proxy_prefix_is_on_inferred_risk_fields():
    assessment = assess(5.0, (4.0,) * 10)
    assert assessment.stockout_proxy in ("high", "medium", "low", "unknown")
    assert "PROXY" in assessment.proxy_note
    assert "stockout_proxy" in assessment.as_dict()


def test_item_level_aggregation_is_the_primary_figure():
    actual = np.array([[5.0], [5.0], [10.0]])
    predicted = np.array([[0.0], [10.0], [10.0]])
    meta = pd.DataFrame({"external_code": [1, 1, 2], "retail": [10, 11, 10]})
    record = StandardMetricSet().evaluate(actual, predicted, meta, "m")
    assert record.wape_pooled == pytest.approx(0.5)
    assert record.wape_item_level == pytest.approx(0.0)
    assert record.n_evaluated_series == 2


def test_simulation_assumptions_are_visible_and_labelled():
    context = ReplenishmentContext(
        key=(1, 1), decision_week=2, forecast_origin_week=2,
        inventory_at_decision_week=0.0, demand_slice=(5.0,) * 10,
        lead_time_weeks=1, order_cycle_weeks=1)
    recommendation = recommend(context, make_safety_stock("normal_approx", 0.95), 2.0)
    payload = recommendation.as_dict()["assumptions"]
    assert payload["simulation_label"] == "simulation under stated assumptions"
    assert "assumed" in payload["assumed_statement"].lower()
    assert payload["protection_weeks"] == 2
    assert isinstance(payload["service_level_applied"], bool)


def test_restock_quantity_and_replenishment_timing_are_separated():
    context = ReplenishmentContext(
        key=(1, 1), decision_week=2, forecast_origin_week=2,
        inventory_at_decision_week=0.0, demand_slice=(5.0,) * 10,
        lead_time_weeks=1, order_cycle_weeks=1)
    recommendation = recommend(context, make_safety_stock("none"), None)
    caption = recommendation.quantity_vs_timing_caption
    assert "quantity" in caption
    assert "timing" in caption and "assumption" in caption


def test_initial_stock_strategy_states_its_assumption():
    strategy = make_initial_stock_strategy("lead_time_demand", 1, 1.0)
    estimate = strategy.estimate(sales_row(), {})
    assert estimate.rep_id == "REP-07"
    assert "assumed" in estimate.assumption_note.lower()
    assert initial_stock_value(strategy, sales_row(), {}) == pytest.approx(4.0)