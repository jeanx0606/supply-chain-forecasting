"""D1 evidence: no baseline may read the target window."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baseline import MovingAverage, SeasonalNaive, LagRegression


def features(n=25):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "lag_1": rng.integers(1, 20, n).astype(float),
        "lag_2": rng.integers(1, 20, n).astype(float),
        "discount": rng.uniform(0, 1, n),
    })


def fit_predict(model, X, y):
    return model.fit(X, y).predict(X)


def test_perturbing_targets_leaves_naive_baselines_byte_identical():
    """D1 concerns the naive family: their prediction must not touch the targets.

    LagRegression is a *supervised* baseline: it legitimately learns from the
    training targets, so the invariant here is only that its features never
    contain target columns (checked by construction in test_features).
    """
    X = features()
    original = np.arange(25.0 * 10).reshape(25, 10)
    perturbed = np.full((25, 10), 999.0)
    for model in (SeasonalNaive(), SeasonalNaive("mean_input"), MovingAverage()):
        first = fit_predict(model, X, original)
        second = fit_predict(model, X, perturbed)
        assert np.array_equal(first, second), type(model).__name__


def test_lag_regression_never_sees_a_target_column():
    X = features()
    y = np.arange(25.0 * 10).reshape(25, 10)
    model = LagRegression().fit(X, y)
    assert model.columns_ == list(X.columns)
    assert not any(str(column).startswith("t") for column in model.columns_)


def test_naive_reference_is_configurable_without_call_site_edits():
    from src.models.baseline import BaselineDefinition

    X = features()
    y = np.ones((25, 10))
    last = BaselineDefinition("last_input").make("seasonal_naive")
    mean = BaselineDefinition("mean_input").make("seasonal_naive")
    assert not np.array_equal(fit_predict(last, X, y), fit_predict(mean, X, y))
    assert BaselineDefinition("last_input").describe().rep_id == "REP-15"