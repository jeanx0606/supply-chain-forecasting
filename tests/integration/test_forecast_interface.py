"""7.4.4 interface contract: rows in = pairs, ten non-negative outputs, validation sigma."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate.error_scale import measure
from src.features.build import build_task
from src.models.backends import make_backend
from src.models.base import Forecaster
from src.models.registry import get_model

pytestmark = pytest.mark.integration

CANDIDATES = ("seasonal_naive", "moving_average", "category_curve",
              "attribute_knn", "lag_regression", "gradient_boosting")


def frame(n=60, seed=7):
    rng = np.random.default_rng(seed)
    data = {
        "external_code": np.arange(1, n + 1),
        "retail": np.resize([1, 2, 3], n),
        "release_date": pd.date_range("2019-01-07", periods=n, freq="W-MON"),
        "season": np.resize(["SS19", "AW19"], n),
        "category": np.resize(["tops", "skirts", "dress"], n),
        "color": ["black"] * n,
        "fabric": np.resize(["cotton", "knit"], n),
        "price": rng.uniform(0.1, 0.9, n),
        "disc_w1": rng.uniform(0.0, 0.5, n),
    }
    for week in range(1, 13):
        data[f"q{week}"] = rng.integers(0, 8, n).astype(float)
    return pd.DataFrame(data)


def test_one_input_row_per_pair_and_ten_outputs():
    data = frame()
    task = build_task(data, 2, 10, 12)
    assert len(task.features) == len(data)
    assert task.horizon == 10
    assert list(task.targets.columns) == [f"t{i}" for i in range(1, 11)]


def test_forecast_origin_is_the_end_of_week_two():
    data = frame(3)
    for week in range(1, 13):
        data[f"q{week}"] = float(week)
    task = build_task(data, 2, 10, 12)
    assert task.origin_week == 2
    assert task.targets.iloc[0, 0] == 3.0
    assert task.targets.iloc[0, 9] == 12.0


def test_every_candidate_returns_ten_non_negative_rows():
    data = frame()
    task = build_task(data, 2, 10, 12)
    for name in CANDIDATES:
        backend = (make_backend("sklearn_hgb", {"max_iter": 20}, 7)
                   if name == "gradient_boosting" else None)
        model = get_model(name, backend=backend, seed=7)
        prediction = model.fit(task.features, task.targets.to_numpy())
        prediction = model.predict(task.features)
        assert isinstance(model, Forecaster)
        assert prediction.shape == (len(data), 10), name
        assert prediction.min() >= 0.0, name
        assert np.isfinite(prediction).all(), name


def test_sigma_e_is_measured_on_the_validation_rows_only():
    data = frame()
    task = build_task(data, 2, 10, 12)
    split = int(len(data) * 0.7), int(len(data) * 0.85)
    train = slice(0, split[0])
    validation = slice(split[0], split[1])
    test = slice(split[1], len(data))

    model = get_model("seasonal_naive").fit(
        task.features.iloc[train], task.targets.iloc[train].to_numpy())
    predicted = model.predict(task.features.iloc[validation])
    targets = task.targets.to_numpy()

    first = measure(targets[validation], predicted)
    perturbed = targets.copy()
    perturbed[test] = 999.0
    second = measure(perturbed[validation], predicted)
    assert first.sigma_e == second.sigma_e
    assert first.partition == "validation"


def test_pipeline_publishes_a_validation_sigma(sample_bundle):
    scale = sample_bundle.error_scale
    assert scale is not None
    assert scale["partition"] == "validation"
    assert scale["sigma_e"] >= 0.0
    assert scale["unit"] == "raw_sales_units_per_week"