"""Closed feature set: declared order, input-week scope, closure (D25, REP-19)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.errors import StageError
from src.features.build import build_features
from src.features.spec import FeatureSpec


def frame(n=12):
    rng = np.random.default_rng(4)
    data = {
        "external_code": np.arange(1, n + 1),
        "retail": np.resize([1, 2], n),
        "release_date": pd.date_range("2019-01-07", periods=n, freq="W-MON"),
        "season": ["SS19"] * n,
        "category": np.where(np.arange(n) % 2 == 0, "tops", "skirts"),
        "color": ["black"] * n,
        "fabric": (["cotton", "knit", "denim", "viscose"] * 4)[:n],
        "price": rng.uniform(0.1, 0.9, n),
        "wx_rain": rng.uniform(0, 5, n),
    }
    for week in range(1, 13):
        data[f"q{week}"] = rng.integers(1, 10, n).astype(float)
        data[f"disc_w{week}"] = rng.uniform(0, 0.5, n)
    return pd.DataFrame(data)


def test_declared_column_order():
    spec = FeatureSpec()
    feats, plan = build_features(frame(), 2, 2, spec=spec)
    assert list(feats.columns)[:4] == ["lag_1", "lag_2",
                                       "discount_lag_1", "discount_lag_2"]
    assert "item_mean_input" in feats.columns
    assert "store_mean_input" in feats.columns
    assert "price_x_discount_1" in feats.columns
    assert "wx_rain" in feats.columns
    flattened = [column for family in plan.families for column in family.columns]
    assert flattened == list(feats.columns)


def test_discount_scope_is_the_input_window():
    data = frame()
    feats, _ = build_features(data, 2, 2, spec=FeatureSpec())
    assert "discount_lag_2" in feats.columns
    assert "disc_w3" not in feats.columns
    mutated = data.copy()
    mutated.loc[:, "disc_w3"] = 9.9
    other, _ = build_features(mutated, 2, 2, spec=FeatureSpec())
    assert feats.equals(other)


def test_interaction_is_the_elementwise_product():
    feats, _ = build_features(frame(), 2, 2, spec=FeatureSpec())
    expected = feats["price_rel"].to_numpy() * feats["discount_lag_1"].to_numpy()
    assert np.allclose(feats["price_x_discount_1"].to_numpy(), expected)


def test_interaction_disabled_drops_the_family():
    feats, _ = build_features(frame(), 2, 2,
                              spec=FeatureSpec(interaction=False))
    assert not any(c.startswith("price_x_discount") for c in feats.columns)


def test_aggregates_read_only_the_input_window():
    data = frame(8)
    feats, _ = build_features(data, 2, 2, spec=FeatureSpec())
    mutated = data.copy()
    for week in range(3, 13):
        mutated.loc[:, f"q{week}"] = 999.0
    other, _ = build_features(mutated, 2, 2, spec=FeatureSpec())
    assert feats["item_mean_input"].equals(other["item_mean_input"])
    assert feats["store_mean_input"].equals(other["store_mean_input"])


def test_depth_beyond_the_window_is_refused():
    with pytest.raises(StageError) as info:
        FeatureSpec(lag_depth=3).plan(frame(), 2, 2)
    assert "outside the input window" in str(info.value)


def test_closure_rejects_an_injected_column():
    feats, plan = build_features(frame(), 2, 2, spec=FeatureSpec())
    injected = feats.copy()
    injected["surprise_feature"] = 1.0
    with pytest.raises(StageError) as info:
        FeatureSpec().assert_closed(injected, plan)
    assert "surprise_feature" in str(info.value)


def test_row_count_is_invariant():
    data = frame(9)
    feats, _ = build_features(data, 2, 2, spec=FeatureSpec())
    assert len(feats) == len(data)


def test_weather_off_disables_the_family():
    feats, _ = build_features(frame(), 2, 2, spec=FeatureSpec(weather="off"))
    assert not any(c.startswith("wx_") for c in feats.columns)