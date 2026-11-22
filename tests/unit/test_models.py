"""Models: input-only baselines, horizon ensemble, registry, seed (D1, D12, D17)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from src.errors import RegistryError, StageError
from src.models.backends import make_backend
from src.models.baseline import MovingAverage, SeasonalNaive
from src.models.registry import available_models, get_model


def features(n=6):
    return pd.DataFrame({
        "lag_1": np.arange(1.0, n + 1),
        "lag_2": np.arange(11.0, 11.0 + n),
        "fabric_cotton": np.zeros(n),
    })


def targets(n=6, horizon=10):
    return np.tile(np.arange(1.0, horizon + 1), (n, 1))


def test_seasonal_naive_repeats_last_input_lag():
    X = features()
    model = SeasonalNaive("last_input").fit(X, targets())
    prediction = model.predict(X)
    assert prediction.shape == (6, 10)
    assert (prediction[:, 0] == X["lag_2"].to_numpy()).all()


def test_mean_reference_averages_inputs():
    X = features()
    model = SeasonalNaive("mean_input").fit(X, targets())
    prediction = model.predict(X)
    expected = X[["lag_1", "lag_2"]].mean(axis=1).to_numpy()
    assert np.allclose(prediction[:, 0], expected)


def test_moving_average_uses_inputs():
    X = features()
    prediction = MovingAverage().fit(X, targets()).predict(X)
    assert np.allclose(prediction[:, 0], X[["lag_1", "lag_2"]].mean(axis=1))


def test_registry_lists_models_and_rejects_unknown():
    names = available_models()
    for name in ("seasonal_naive", "moving_average", "lag_regression", "gradient_boosting"):
        assert name in names
    assert list(names) == sorted(names)
    with pytest.raises(RegistryError) as info:
        get_model("nope")
    assert "available" in str(info.value)


def test_boosting_has_one_component_per_horizon():
    backend = make_backend("sklearn_hgb", {"max_iter": 20}, 42)
    model = get_model("gradient_boosting", backend=backend, seed=42)
    model.fit(features(20), targets(20))
    assert len(model.ensemble_.models) == 10
    prediction = model.predict(features(4))
    assert prediction.shape == (4, 10)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()


def test_predictions_clipped_at_zero():
    X = features(4)
    y = -np.ones((4, 10))
    for model in (SeasonalNaive(), MovingAverage()):
        prediction = model.fit(X, y).predict(X)
        assert (prediction >= 0).all()


def test_same_seed_reproduces_identical_arrays():
    X, y = features(30), targets(30)
    backend = make_backend("sklearn_hgb", {}, 7)
    first = get_model("gradient_boosting", backend=backend, seed=7).fit(X, y).predict(X)
    backend = make_backend("sklearn_hgb", {}, 7)
    second = get_model("gradient_boosting", backend=backend, seed=7).fit(X, y).predict(X)
    assert np.array_equal(first, second)


def test_lightgbm_backend_builds_now_that_the_wheel_is_present():
    backend = make_backend("lightgbm", {"max_iter": 10}, 42)
    ensemble = backend.make_ensemble(3)
    assert ensemble.horizon == 3


def test_lightgbm_absence_is_still_reported_with_an_action(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    backend = make_backend("lightgbm", {}, 42)
    with pytest.raises(StageError) as info:
        backend.make_ensemble(10)
    message = str(info.value)
    assert "model.backend" in message
    assert "sklearn_hgb" in message


def test_main_model_never_falls_back_silently():
    from src.errors import ConfigError
    from src.models.registry import ModelSelection

    selection = ModelSelection("gradient_boosting", "seasonal_naive")
    with pytest.raises(ConfigError) as info:
        selection.require_main({"seasonal_naive": np.zeros((2, 2))})
    assert "gradient_boosting" in str(info.value)