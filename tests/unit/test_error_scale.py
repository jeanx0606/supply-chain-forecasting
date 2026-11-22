"""sigma_e contract: measured on validation, ddof=1, aligned shapes (7.4.4, 7.6.3)."""
from __future__ import annotations

import numpy as np
import pytest

from src.errors import StageError
from src.evaluate.error_scale import ErrorScale, measure, sigma_e


def test_constant_error_has_zero_spread():
    actual = np.full((5, 10), 7.0)
    predicted = np.full((5, 10), 4.0)
    assert sigma_e(actual, predicted) == 0.0


def test_alternating_errors_match_sample_standard_deviation():
    actual = np.array([[1.0], [3.0]])
    predicted = np.zeros((2, 1))
    assert sigma_e(actual, predicted) == pytest.approx(np.sqrt(2.0))


def test_single_observation_is_zero_not_undefined():
    assert sigma_e([[5.0]], [[2.0]]) == 0.0


def test_measure_declares_partition_unit_and_horizon():
    errors = np.array([[1.0], [2.0], [3.0], [4.0]])
    actual = np.tile(errors, (1, 10))
    record = measure(actual, np.zeros((4, 10)))
    assert isinstance(record, ErrorScale)
    assert record.partition == "validation"
    assert record.horizon_weeks == 10
    assert record.n_observations == 40
    assert record.unit == "raw_sales_units_per_week"
    as_dict = record.as_dict()
    assert as_dict["partition"] == "validation"
    assert as_dict["sigma_e"] == pytest.approx(np.std(actual, ddof=1))


def test_misaligned_predictions_are_rejected():
    with pytest.raises(StageError) as info:
        sigma_e(np.ones((3, 10)), np.ones((4, 10)))
    assert "sigma_e" in str(info.value)


def test_errors_are_actual_minus_forecast_in_sign_convention():
    actual = np.array([[0.0], [0.0]])
    predicted = np.array([[1.0], [-1.0]])
    assert sigma_e(actual, predicted) == pytest.approx(np.sqrt(2.0))