"""Error slices: categories, price, discount, volume, min size and rankings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.errors import StageError
from src.evaluate.slices import error_slices


def fixtures():
    n, horizon = 8, 3
    categories = ["alpha"] * 4 + ["beta"] * 4
    features = pd.DataFrame({"lag_1": np.arange(n, dtype=float)})
    for name in ("alpha", "beta"):
        features[f"category_{name}"] = [1.0 if c == name else 0.0 for c in categories]
    raw = pd.DataFrame({
        "price": np.arange(1.0, n + 1),
        "q1": [1.0] * 4 + [5.0] * 4,
        "q2": [1.0] * 4 + [5.0] * 4,
        "disc_w3": [0.0, 0.2, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        "disc_w4": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.6, 0.0],
        "disc_w5": [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.7],
    })
    actual = np.vstack([[1.0] * horizon] * 4 + [[2.0] * horizon] * 4)
    predicted = np.vstack([[2.0] * horizon] * 4 + [[2.0] * horizon] * 4)
    meta = pd.DataFrame({"external_code": np.arange(1, n + 1),
                         "retail": [1] * n})
    return actual, predicted, features, raw, meta


def by_key(payload):
    return {(row["slice"], row["group"]): row
            for row in payload["slices"]}


def test_category_slices_report_n_and_the_standard_metrics():
    payload = error_slices(*fixtures())
    rows = by_key(payload)
    alpha = rows[("category", "alpha")]
    assert alpha["n"] == 12
    assert alpha["wape"] == pytest.approx(1.0)
    assert alpha["mae"] == pytest.approx(1.0)
    assert alpha["bias"] == pytest.approx(1.0)
    beta = rows[("category", "beta")]
    assert beta["wape"] == pytest.approx(0.0)


def test_price_bands_partition_every_row():
    payload = error_slices(*fixtures())
    rows = [row for row in payload["slices"] if row["slice"] == "price_band"]
    assert {row["group"] for row in rows} <= {"Q1", "Q2", "Q3", "Q4"}
    assert sum(row["n"] for row in rows) == 24


def test_discount_slice_is_cell_level():
    payload = error_slices(*fixtures())
    rows = by_key(payload)
    assert rows[("discount_period", "any_discount")]["n"] == 6
    assert rows[("discount_period", "no_discount")]["n"] == 18


def test_volume_slice_uses_the_observed_input_window():
    payload = error_slices(*fixtures())
    rows = by_key(payload)
    assert rows[("volume_band", "low_volume")]["n"] == 12
    assert rows[("volume_band", "rest")]["n"] == 12


def test_groups_below_the_minimum_size_are_flagged_and_not_ranked():
    payload = error_slices(*fixtures(), min_size=100)
    assert payload["reported_count"] == 0
    assert payload["worst_groups"] == []
    assert all(row["status"] == "below_min_size" for row in payload["slices"])


def test_worst_group_and_top_rows_are_published():
    payload = error_slices(*fixtures())
    assert payload["worst_groups"][0]["group"] == "alpha"
    assert payload["top_error_rows"][0]["external_code"] == 1
    assert payload["top_cumulative_errors"][0]["cumulative_error"] == pytest.approx(3.0)


def test_slice_payload_is_not_labelled_as_a_proxy():
    payload = error_slices(*fixtures())
    assert "not proxy" in payload["definition"]
    assert payload["test_partition"] == "test"
    assert payload["min_slice_size"] == 10


def test_missing_category_columns_are_recorded_not_dropped():
    actual, predicted, features, raw, meta = fixtures()
    payload = error_slices(actual, predicted,
                           features.drop(columns=["category_alpha", "category_beta"]),
                           raw, meta)
    assert "category" in {row["slice"] for row in payload["unavailable"]}


def test_misaligned_shapes_are_rejected():
    actual, predicted, features, raw, meta = fixtures()
    with pytest.raises(StageError):
        error_slices(actual, predicted[:-1], features, raw, meta)