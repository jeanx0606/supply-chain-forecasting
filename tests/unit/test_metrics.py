"""Metrics and stratified evaluation (B1, B3, B12, B19)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate.aggregate import StandardMetricSet
from src.evaluate.metrics import mae, rmse, wape


def test_wape_perfect_and_zero_denominator():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert wape(values, values) == 0.0
    assert wape(np.zeros((2, 2)), np.ones((2, 2))) is None
    assert mae([], []) is None
    assert rmse([], []) is None


def test_item_level_sums_rows_before_error():
    actual = np.array([[1.0], [2.0], [100.0]])
    predicted = np.array([[0.0], [0.0], [100.0]])
    meta = pd.DataFrame({"external_code": [1, 1, 2], "retail": [10, 11, 10]})
    record = StandardMetricSet().evaluate(actual, predicted, meta, "m")
    # item 1 total 3 vs 0 -> error 3; item 2 exact -> WAPE = 3 / 103
    assert abs(record.wape_item_level - 3.0 / 103.0) < 1e-9
    assert record.n_evaluated_series == 2
    assert record.model == "m"


def test_store_weighted_and_small_store_subset():
    actual = np.arange(1.0, 9.0)[:, None]
    predicted = actual - 1.0
    meta = pd.DataFrame({
        "external_code": [1, 1, 2, 2, 3, 3, 4, 4],
        "retail": [10, 11, 10, 11, 20, 21, 20, 21],
    })
    record = StandardMetricSet().evaluate(actual, predicted, meta, "m")
    assert record.wape_store_weighted is not None
    assert record.wape_small_stores is not None
    assert record.n_small_stores == 4


def test_metadata_misalignment_degrades_to_pooled_only():
    actual = np.ones((4, 2))
    predicted = np.zeros((4, 2))
    meta = pd.DataFrame({"external_code": [1, 2]})
    record = StandardMetricSet().evaluate(actual, predicted, meta, "m")
    assert record.wape_item_level is None
    assert record.wape_pooled == 1.0
    assert record.warnings


def test_record_carries_identity_and_no_mape_field():
    actual = np.ones((3, 2))
    meta = pd.DataFrame({"external_code": [1, 2, 3], "retail": [1, 1, 1]})
    record = StandardMetricSet().evaluate(actual, actual, meta, "seasonal_naive",
                                          "2019-01-01", "2019-06-01")
    assert record.n_evaluated_records == 3
    assert record.cut_train_end == "2019-01-01"
    assert record.cut_validation_end == "2019-06-01"
    assert "mape" not in record.as_dict()


def test_primary_comes_from_the_seam():
    seam = StandardMetricSet(primary="wape_pooled")
    assert seam.describe().rep_id == "REP-03"
    assert seam.primary == "wape_pooled"