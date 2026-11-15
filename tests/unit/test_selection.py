"""Selection: validation-only tournament, qualification, retained default (D24, REP-18)."""
from __future__ import annotations

import inspect

import pytest

from src.errors import StageError
from src.models.selection import CandidateFigure, SelectionGuard, choose, qualifies


def figure(name, value, records=98, available=True):
    return CandidateFigure(name, value, records, available)


def test_qualification_truth_table():
    assert qualifies(0.10, 0.20, "must_not_be_worse_than_naive") is True
    assert qualifies(0.20, 0.20, "must_not_be_worse_than_naive") is True
    assert qualifies(0.21, 0.20, "must_not_be_worse_than_naive") is False
    assert qualifies(0.20, 0.20, "strictly_better") is False
    assert qualifies(0.19, 0.20, "strictly_better") is True
    assert qualifies(None, 0.20, "strictly_better") is False
    assert qualifies(0.19, 0.20, "strictly_better", tolerance=0.0) is True
    assert qualifies(0.2005, 0.20, "must_not_be_worse_than_naive", tolerance=0.001) is True


def test_choose_picks_the_lowest_validation_figure():
    figures = [figure("seasonal_naive", 0.30), figure("moving_average", 0.12),
               figure("gradient_boosting", 0.40)]
    decision = choose(figures, default_model="gradient_boosting",
                      candidates=["seasonal_naive", "moving_average",
                                  "gradient_boosting"])
    assert decision.chosen_model == "moving_average"
    assert decision.outcome == "qualified"


def test_tie_is_recorded_and_broken_by_declared_order():
    figures = [figure("seasonal_naive", 0.20), figure("moving_average", 0.10),
               figure("gradient_boosting", 0.10)]
    decision = choose(figures, candidates=["gradient_boosting", "moving_average",
                                          "seasonal_naive"],
                      default_model="gradient_boosting")
    assert decision.tie == ("gradient_boosting", "moving_average")
    assert decision.chosen_model == "gradient_boosting"


def test_unavailable_figure_never_qualifies_and_is_not_zero_filled():
    figures = [figure("seasonal_naive", 0.20), figure("moving_average", None, available=False)]
    decision = choose(figures, mode="strictly_better", default_model="moving_average")
    assert decision.chosen_model == "moving_average"
    assert decision.outcome == "no candidate qualified"


def test_no_qualification_retains_the_default_and_records_the_gap():
    figures = [figure("seasonal_naive", 0.10), figure("gradient_boosting", 0.40)]
    decision = choose(figures, mode="strictly_better", default_model="gradient_boosting",
                      on_none="retained_default")
    assert decision.chosen_model == "gradient_boosting"
    assert decision.outcome == "no candidate qualified"
    assert decision.on_none == "retained_default"
    assert decision.note


def test_least_bad_promotes_provisionally():
    figures = [figure("seasonal_naive", 0.10), figure("gradient_boosting", 0.40)]
    decision = choose(figures, mode="strictly_better", default_model="seasonal_naive",
                      on_none="least_bad_provisional")
    assert decision.chosen_model == "gradient_boosting"
    assert decision.on_none == "least_bad_provisional"


def test_fail_run_refuses_to_publish():
    figures = [figure("seasonal_naive", 0.10), figure("gradient_boosting", 0.40)]
    with pytest.raises(StageError):
        choose(figures, mode="strictly_better", default_model="gradient_boosting",
               on_none="fail_run")


def test_choose_takes_no_array_parameter():
    names = {p.name for p in inspect.signature(choose).parameters.values()}
    assert not ({"X", "y", "mask", "ndarray", "predictions"} & names)


def test_guard_refuses_pre_seal_and_double_read():
    guard = SelectionGuard()
    with pytest.raises(StageError):
        guard.evaluate_test("seasonal_naive")
    guard.seal()
    guard.evaluate_test("seasonal_naive")
    with pytest.raises(StageError):
        guard.evaluate_test("seasonal_naive")