"""LightGBM horizon model: one estimator, horizon as feature, train-only cap."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from src.errors import StageError
from src.models.lightgbm_model import HORIZON_FEATURE, LightGBMHorizon
from src.models.registry import available_models, get_model

lightgbm = pytest.importorskip("lightgbm")


def features(n=40, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lag_1": rng.integers(0, 30, n).astype(float),
        "lag_2": rng.integers(0, 30, n).astype(float),
        "price_rel": rng.uniform(0.1, 0.9, n),
        "store_size": rng.integers(1, 5, n).astype(float),
    })


def targets(n=40, horizon=10):
    base = np.arange(1.0, horizon + 1)
    return np.vstack([base * (i % 5 + 1) for i in range(n)])


def test_registry_exposes_the_model_and_builds_it_without_a_backend():
    assert "lightgbm_horizon" in available_models()
    model = get_model("lightgbm_horizon", seed=7)
    assert isinstance(model, LightGBMHorizon)
    assert model.seed == 7


def test_one_estimator_covers_every_horizon_week():
    model = LightGBMHorizon({"n_estimators": 30}, seed=7).fit(features(), targets())
    assert hasattr(model.model_, "predict")
    assert not hasattr(model.model_, "models")
    assert model.training_rows_ == 40 * 10
    assert model.horizon_ == 10


def test_horizon_is_a_feature_and_horizon_curves_differ():
    model = LightGBMHorizon({"n_estimators": 60}, seed=7).fit(features(), targets())
    prediction = model.predict(features(5))
    assert prediction.shape == (5, 10)
    assert (prediction >= 0).all()
    assert np.isfinite(prediction).all()
    assert not np.allclose(prediction[:, 0], prediction[:, -1])
    assert HORIZON_FEATURE not in model.columns_


def test_extreme_weekly_values_are_capped_for_training_only():
    y = targets(2000)
    y[0, 0] = 1_000_000.0
    model = LightGBMHorizon({"n_estimators": 10}, seed=7).fit(features(2000), y)
    assert model.cap_.shape == (10,)
    assert model.cap_[0] < 1000.0
    prediction = model.predict(features(3))
    assert prediction.max() < 1_000_000.0


def test_negative_targets_are_clipped_before_poisson_fit():
    y = targets()
    y[:5] = -3.0
    model = LightGBMHorizon({"n_estimators": 20}, seed=7).fit(features(), y)
    assert model.cap_.min() >= 0.0
    assert (model.predict(features(4)) >= 0).all()


def test_missing_wheel_is_lazy_and_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    model = LightGBMHorizon()
    with pytest.raises(StageError) as info:
        model.fit(features(4), targets(4))
    message = str(info.value)
    assert "lightgbm" in message
    assert "pip install lightgbm" in message


def test_fit_is_required_before_predict():
    model = LightGBMHorizon()
    with pytest.raises(StageError):
        model.predict(features(2))