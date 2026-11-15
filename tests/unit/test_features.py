"""Features and targets: window contract, identifiers excluded, alt task (D8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import build_task
from src.features.tasks import AltTaskSpec, build_alt_task


def frame(n=12):
    rng = np.random.default_rng(5)
    data = {
        "external_code": np.arange(1, n + 1),
        "retail": np.resize([1, 2], n),
        "release_date": pd.date_range("2019-01-07", periods=n, freq="W-MON"),
        "season": ["SS19"] * n,
        "category": np.where(np.arange(n) % 2 == 0, "tops", "skirts"),
        "color": ["black"] * n,
        "fabric": (["cotton", "knit", "denim", "viscose"] * ((n // 4) + 1))[:n],
        "price": rng.uniform(0.1, 0.9, n),
        "disc_w1": rng.uniform(0, 0.5, n),
    }
    for week in range(1, 13):
        data[f"q{week}"] = rng.integers(0, 10, n).astype(float)
    return pd.DataFrame(data)


def test_main_task_window_and_target_labels():
    task = build_task(frame(), 2, 10, 12)
    assert [c for c in task.features.columns if c.startswith("lag_")] == ["lag_1", "lag_2"]
    assert list(task.targets.columns) == [f"t{i}" for i in range(1, 11)]
    assert task.origin_week == 2


def test_identifiers_absent_from_feature_matrix():
    task = build_task(frame(), 2, 10, 12)
    for column in ("external_code", "retail", "release_date"):
        assert column not in task.features.columns
    assert task.features.select_dtypes(include=[np.number]).shape[1] == task.features.shape[1]


def test_categorical_indicators_are_float_and_complete():
    task = build_task(frame(12), 2, 10, 12)
    fabric_columns = [c for c in task.features.columns if c.startswith("fabric_")]
    assert len(fabric_columns) == 4
    assert all(task.features[c].dtype == float for c in fabric_columns)


def test_store_scale_is_distinct_items():
    task = build_task(frame(12), 2, 10, 12)
    assert "store_size" in task.features.columns
    assert (task.features["store_size"] > 0).all()


def test_missing_target_week_becomes_zero():
    data = frame(6)
    data.loc[0, "q12"] = np.nan
    task = build_task(data, 2, 10, 12)
    assert task.targets.loc[0, "t10"] == 0.0


def test_relative_price_has_no_currency_unit():
    task = build_task(frame(), 2, 10, 12)
    assert task.features["price_rel"].max() < 10.0


def test_alt_task_has_eight_points_and_declared_origins():
    points = build_alt_task(frame(15), 4, 1, 12)
    assert len(points) == 8
    assert points[0].origin_week == 4
    assert points[-1].origin_week == 11
    assert points[0].labels == ("t1",)
    assert [c for c in points[0].features.columns if c.startswith("lag_")] == [
        "lag_1", "lag_2", "lag_3", "lag_4"]


def test_alt_task_disabled_by_seam():
    assert AltTaskSpec(enabled=False).build(frame(), 12) == ()
    assert AltTaskSpec(enabled=True).build(frame(15), 12)[0].origin_week == 4

def test_scaled_frame_is_refused_at_the_feature_boundary():
    import pytest
    from src.errors import StageError
    from src.io.unit_contract import SCALED_UNIT, stamp

    data = frame(12)
    stamp(data, SCALED_UNIT)
    with pytest.raises(StageError) as info:
        build_task(data, 2, 10, 12)
    assert "C10" in str(info.value)
