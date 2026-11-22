"""R0-R4 ladder: timeline, isolation, replay, no-reorder and perfect foresight."""
from __future__ import annotations

import pytest

from src.errors import ConfigError
from src.replenish.scenario import (ScenarioPair, ScenarioSettings,
                                    ladder_payload, run_ladder, run_scenario,
                                    simulate_pair)


def settings(lead=1, cycle=1, weeks=12):
    return ScenarioSettings(lead_time_weeks=lead, order_cycle_weeks=cycle,
                            service_level=0.95, n_weeks=weeks,
                            forecast_origin_week=2)


def pair(opening=10.0, sales=5.0, restocks=None, weeks=12):
    return ScenarioPair(key=(1, 1), sales=tuple([float(sales)] * weeks),
                        restocks=dict(restocks or {}), opening=float(opening))


def test_order_placed_at_end_of_week_two_arrives_after_one_full_lead_week():
    result = simulate_pair(pair(opening=10.0), [5.0, 5.0], settings(weeks=4),
                           sigma_e=0.0, mode="forecast")
    assert result["orders"] == 1
    assert result["stockout_weeks"] == 1
    assert result["lost"] == pytest.approx(5.0)
    assert result["sold"] == pytest.approx(15.0)
    assert result["demanded"] == pytest.approx(20.0)
    assert result["leftover"] == pytest.approx(5.0)


def test_zero_lead_time_removes_the_week_three_gap():
    result = simulate_pair(pair(opening=10.0), [5.0, 5.0], settings(lead=0, weeks=4),
                           sigma_e=0.0, mode="forecast")
    assert result["stockout_weeks"] == 0
    assert result["sold"] == pytest.approx(20.0)
    assert result["leftover"] == pytest.approx(0.0)


def test_no_review_happens_in_the_final_week():
    result = simulate_pair(pair(opening=100.0), [1.0] * 10, settings(weeks=12),
                           sigma_e=0.0, mode="forecast")
    assert result["orders"] == 0


def test_r0_replays_restocks_and_r0b_never_reorders():
    base = pair(opening=0.0, sales=1.0, restocks={2: 4.0, 5: 6.0})
    replay = run_scenario("R0", [base], settings())
    no_reorder = run_scenario("R0b", [base], settings())
    assert replay.supplied_units == pytest.approx(10.0)
    assert no_reorder.supplied_units == pytest.approx(0.0)
    assert replay.orders == no_reorder.orders == 0
    assert no_reorder.fill_rate == 0.0
    assert replay.fill_rate == pytest.approx(10.0 / 12.0)


def test_only_the_forecast_changes_between_forecast_arms():
    base = pair(opening=0.0, sales=2.0)
    low = run_scenario("R1", [base], settings(), forecasts=[[1.0] * 10],
                       forecast_source="low", sigma_e=1.0)
    high = run_scenario("R1", [base], settings(), forecasts=[[9.0] * 10],
                        forecast_source="high", sigma_e=1.0)
    assert low.settings == high.settings
    assert low.sigma_e == high.sigma_e
    assert low.orders != high.orders or low.leftover_units != high.leftover_units
    assert low.fill_rate is not None and low.overstock_ratio is not None


def test_perfect_foresight_beats_the_naive_arm_on_a_late_spike():
    sales = (0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0)
    demand = (0.0, 0.0) + sales
    base = ScenarioPair(key=(1, 1), sales=demand, restocks={}, opening=0.0)
    naive = run_scenario("R1", [base], settings(), forecasts=[tuple([5.0] * 10)],
                         forecast_source="naive", sigma_e=0.0)
    perfect = run_scenario("R4", [base], settings(), forecasts=[list(sales)],
                           forecast_source="actuals", sigma_e=0.0)
    assert perfect.fill_rate == pytest.approx(1.0)
    assert naive.fill_rate == pytest.approx(10.0 / 30.0)
    assert perfect.stockout_rate < naive.stockout_rate


def test_ladder_runs_r0_r0b_and_every_supplied_forecast_arm():
    base = pair(opening=5.0, sales=2.0)
    outcomes = run_ladder(
        [base], settings(),
        {"R1": ("naive", [[2.0] * 10]), "R2": ("B2", [[2.0] * 10]),
         "R3": ("model", [[2.0] * 10]), "R4": ("actuals", [[2.0] * 10])},
        {"R1": 1.5, "R2": 1.6, "R3": 1.49, "R4": 0.0})
    names = [outcome.scenario for outcome in outcomes]
    assert names == ["R0", "R0b", "R1", "R2", "R3", "R4"]
    assert all(outcome.settings == settings().as_dict() for outcome in outcomes)
    assert [o.sigma_e for o in outcomes[2:]] == [1.5, 1.6, 1.49, 0.0]
    assert all(outcome.metrics.cost_index is None for outcome in outcomes)


def test_metrics_stay_undefined_for_an_empty_pair_set():
    outcome = run_scenario("R0", [], settings())
    assert outcome.metrics.fill_rate is None
    assert outcome.metrics.overstock_ratio is None
    assert outcome.metrics.stockout_rate is None


def test_the_ladder_refuses_a_forecast_origin_other_than_week_two():
    with pytest.raises(ConfigError) as info:
        ScenarioSettings(forecast_origin_week=1)
    assert "2-in/10-out" in str(info.value)


def test_a_forecast_must_cover_the_full_horizon():
    with pytest.raises(ConfigError) as info:
        run_scenario("R1", [pair()], settings(), forecasts=[[1.0] * 5])
    assert "full horizon" in str(info.value)


def test_payload_records_a_scenario_that_has_no_forecast_source():
    outcome = run_scenario("R0", [pair()], settings())
    payload = ladder_payload([outcome], settings(), {"R1": ("naive", None)})
    missing = {row["scenario"] for row in payload["unavailable"]}
    assert missing == {"R2", "R3", "R4"}
    assert payload["rows"][0]["scenario"] == "R0"