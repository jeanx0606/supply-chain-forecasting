"""Replenishment policy: s, S, quantity, assumptions and decision week (D4, D5)."""
from __future__ import annotations

import numpy as np
import pytest

from src.errors import ConfigError
from src.replenish.policy import (
    ReplenishmentContext,
    recommend,
    recommend_baseline,
)
from src.replenish.safety import make_safety_stock


def context(stock=0.0, demand=(5.0,) * 10, lead=1, cycle=1, decision=2):
    return ReplenishmentContext(
        key=(1, 2), decision_week=decision, forecast_origin_week=decision,
        inventory_at_decision_week=stock, demand_slice=tuple(demand),
        lead_time_weeks=lead, order_cycle_weeks=cycle)


def zero():
    return make_safety_stock("none", 0.95)


def test_reorder_point_and_target_level():
    recommendation = recommend(context(stock=0.0), zero())
    assert recommendation.reorder_point == 5.0
    assert recommendation.target_level == 10.0


def test_quantity_formula_and_non_negative():
    recommendation = recommend(context(stock=3.0), zero())
    assert recommendation.recommended_qty == 7
    assert recommendation.recommended_qty == max(
        0, round(recommendation.target_level - 3.0))
    for stock in (0.0, 5.0, 50.0, 5000.0):
        assert recommend(context(stock=stock), zero()).recommended_qty >= 0


def test_no_order_when_above_target():
    recommendation = recommend(context(stock=1000.0), zero())
    assert recommendation.recommended_qty == 0
    assert recommendation.trigger is False


def test_decision_week_equals_forecast_origin():
    recommendation = recommend(context(decision=2), zero())
    assert recommendation.decision_week == recommendation.forecast_origin_week == 2


def test_order_cycle_changes_target_but_not_reorder_point():
    one = recommend(context(stock=0.0, cycle=1), zero())
    two = recommend(context(stock=0.0, cycle=2), zero())
    assert one.reorder_point == two.reorder_point
    assert two.target_level > one.target_level


def test_assumptions_block_states_assumed_values():
    recommendation = recommend(context(), zero())
    block = recommendation.assumptions
    assert "assumed" in block.assumed_statement.lower()
    assert block.lead_time_weeks == 1
    assert block.order_cycle_weeks == 1
    assert block.service_level == 0.95
    assert block.service_level_applied is False
    assert "OUT OF SCOPE" in block.scope_note


def test_baseline_comparison_always_present():
    baseline = recommend_baseline(context(stock=3.0), target_weeks=2)
    assert baseline.target_level == 10.0
    assert baseline.recommended_qty == 7
    assert baseline.rule_name


def test_empty_forecast_is_flagged():
    recommendation = recommend(context(demand=(), stock=0.0), zero())
    assert recommendation.forecast_unavailable is True
    assert recommendation.recommended_qty == 0


def test_service_level_outside_range_rejected():
    with pytest.raises(ConfigError):
        make_safety_stock("none", 1.4)