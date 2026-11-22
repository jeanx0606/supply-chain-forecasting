"""B2 category-curve and B3 attribute-k-NN: shape, fallback, scaling, no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baseline import AttributeKNN, CategoryCurve
from src.models.registry import available_models, get_model

ROWS = [
    ("dress", "SS19", 10.0),
    ("dress", "SS19", 10.0),
    ("dress", "AW19", 10.0),
    ("tops", "SS19", 3.0),
    ("tops", "SS19", 3.0),
    ("skirts", "AW19", 1.0),
    ("skirts", "AW19", 1.0),
]


def features():
    records = []
    for category, season, level in ROWS:
        records.append({
            "lag_1": level * 0.8, "lag_2": level * 1.2,
            "price_rel": 0.5, "store_size": 2.0,
            f"category_{category}": 1.0,
            f"season_{season}": 1.0,
            "color_black": 1.0,
            "fabric_cotton": 1.0,
        })
    return pd.DataFrame(records)


def targets():
    return np.vstack([[level] * 10 for _, _, level in ROWS])


def query(category, season, level):
    return pd.DataFrame([{
        "lag_1": level * 0.8, "lag_2": level * 1.2,
        "price_rel": 0.5, "store_size": 2.0,
        f"category_{category}": 1.0,
        f"season_{season}": 1.0,
        "color_black": 1.0,
        "fabric_cotton": 1.0,
    }])


def test_both_baselines_are_registered_and_build_ten_outputs():
    assert {"category_curve", "attribute_knn"} <= set(available_models())
    for name in ("category_curve", "attribute_knn"):
        model = get_model(name)
        prediction = model.fit(features(), targets()).predict(features())
        assert prediction.shape == (len(ROWS), 10)
        assert (prediction >= 0).all()
        assert np.isfinite(prediction).all()


def test_category_curve_scales_to_observed_weeks():
    model = CategoryCurve(min_group_size=2).fit(features(), targets())
    prediction = model.predict(query("tops", "SS19", 6.0))
    assert prediction[0, 0] == 6.0


def test_category_curve_falls_back_from_pair_to_category():
    model = CategoryCurve(min_group_size=2).fit(features(), targets())
    prediction = model.predict(query("dress", "ZZ99", 10.0))
    assert prediction[0, 0] == 10.0


def test_category_curve_falls_back_to_global_for_unseen_category():
    model = CategoryCurve(min_group_size=2).fit(features(), targets())
    prediction = model.predict(query("coats", "ZZ99", 10.0))
    assert prediction.shape == (1, 10)
    assert prediction[0, 0] > 0


def test_attribute_knn_uses_the_nearest_signature():
    model = AttributeKNN(k=1).fit(features(), targets())
    prediction = model.predict(query("tops", "SS19", 3.0))
    assert prediction[0, 0] == 3.0


def test_attribute_knn_survives_a_matrix_without_attributes():
    frame = features().loc[:, ["lag_1", "lag_2"]]
    model = AttributeKNN(k=2).fit(frame, targets())
    prediction = model.predict(frame.loc[:1])
    assert prediction.shape == (2, 10)
    assert (prediction >= 0).all()


def test_negative_training_curves_are_clipped_at_zero():
    model = AttributeKNN(k=2).fit(features(), -targets())
    assert (model.predict(features()) >= 0).all()


def test_predict_never_requires_a_target_column():
    X = features()
    model = CategoryCurve().fit(X, targets())
    with_targets = X.assign(t1=np.arange(len(X), dtype=float))
    assert np.array_equal(model.predict(X), model.predict(with_targets))


def test_both_baselines_ignore_targets_at_predict_time():
    X = features()
    for cls in (CategoryCurve, AttributeKNN):
        first = cls().fit(X, targets()).predict(X)
        second = cls().fit(X, np.full_like(targets(), 999.0)).predict(X)
        assert not np.array_equal(first, second)
        assert np.array_equal(cls().fit(X, targets()).predict(X), first)