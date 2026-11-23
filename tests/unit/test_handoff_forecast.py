"""Handoff adapter: feature contract, leakage guard, clipping, manifest validation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.errors import ConfigError, StageError
from src.handoff.forecast import (CATEGORICAL_COLUMNS, DERIVED_COLUMNS,
                                  HORIZON_WEEKS, LindaLightGBM, build_features,
                                  load_manifest)

FEATURES = (
    "h", "obs_0", "obs_1", "obs_2", "obs_3", "obs_sum", "obs_last",
    "category_id", "color_id", "fabric_id", "season_type", "release_year",
    "release_month", "release_week", "init_stock", "price", "disc_h", "disc_obs",
    "gt_cat_wm4", "gt_cat_wm1", "gt_cat_w0", "gt_cat_w1",
    "gt_col_wm4", "gt_col_wm1", "gt_col_w0", "gt_col_w1",
    "gt_fab_wm4", "gt_fab_wm1", "gt_fab_w0", "gt_fab_w1",
)
SIGMA = {"B0": 2.0123, "B1": 1.7461, "B2": 1.5853, "B3": 1.6225, "LightGBM": 1.4881}


def write_manifest(directory, **overrides):
    payload = {
        "setup": {"observed_weeks": 2, "forecast_weeks": "3-12 (files: 2..11)"},
        "model_file": "lightgbm_final_model.txt",
        "objective": "poisson",
        "feature_order": list(FEATURES),
        "categorical_features": list(CATEGORICAL_COLUMNS),
        "sigma_e_validation_SS19": dict(SIGMA),
        "replenishment_main": {"lead_time_weeks": 1, "review_weeks": 1,
                               "service_level": 0.95, "z": 1.6449},
        "z_values": {"0.95": 1.6449},
    }
    for key, value in overrides.items():
        payload[key] = value
    path = directory / "prototype_params.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_frame(rows: int = 4) -> pd.DataFrame:
    data = {
        "external_code": np.arange(rows),
        "retail": np.arange(rows) + 10,
        "0": [0.0, 3.0, -2.0, 1.0][:rows],
        "1": [2.0, 0.0, 4.0, 1.0][:rows],
        "price": [0.05] * rows,
        "init_stock": [5.0] * rows,
        "season_type": ["SS", "AW", "SS", "AW"][:rows],
        "release_year": [2016] * rows,
        "release_month": [11] * rows,
        "release_week": [48] * rows,
        "category_id": [0, 1, 2, 0][:rows],
        "color_id": [0, 1, 2, 1][:rows],
        "fabric_id": [0, 1, 2, 3][:rows],
    }
    for week in range(2, 12):
        data[str(week)] = [2.0] * rows
    for week in range(12):
        data[f"disc_{week}"] = [0.0] * rows
    for column in ("gt_cat_wm4", "gt_cat_wm1", "gt_cat_w0", "gt_cat_w1",
                   "gt_col_wm4", "gt_col_wm1", "gt_col_w0", "gt_col_w1",
                   "gt_fab_wm4", "gt_fab_wm1", "gt_fab_w0", "gt_fab_w1"):
        data[column] = [60.0] * rows
    return pd.DataFrame(data)


class FakeBooster:
    """Deterministic stand-in: reads only the two observed weeks, the horizon and disc_h."""

    def __init__(self, features=FEATURES, categories=None, offset=-1.0):
        self._features = tuple(features)
        self.pandas_categorical = categories
        self.offset = float(offset)
        self.last_matrix = None

    def feature_name(self):
        return list(self._features)

    def predict(self, matrix):
        self.last_matrix = matrix
        values = (matrix["obs_0"].to_numpy(dtype=float)
                  + matrix["obs_1"].to_numpy(dtype=float)
                  + matrix["h"].to_numpy(dtype=float) * 0.0
                  + matrix["disc_h"].to_numpy(dtype=float))
        return values + self.offset


def make_model(tmp_path, booster=None, **overrides):
    write_manifest(tmp_path, **overrides)
    manifest = load_manifest(tmp_path)
    return LindaLightGBM(manifest, booster=booster or FakeBooster())


def test_manifest_reads_the_declared_contract(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path).parent)
    assert manifest.feature_order == FEATURES
    assert manifest.categorical_columns == CATEGORICAL_COLUMNS
    assert manifest.sigma("LightGBM") == pytest.approx(1.4881)
    assert manifest.summary()["n_features"] == len(FEATURES)


def test_manifest_rejects_a_missing_sigma(tmp_path):
    write_manifest(tmp_path, sigma_e_validation_SS19={"B0": 2.0})
    with pytest.raises(ConfigError):
        load_manifest(tmp_path)


def test_manifest_rejects_a_different_categorical_set(tmp_path):
    write_manifest(tmp_path, categorical_features=["category_id"])
    with pytest.raises(ConfigError):
        load_manifest(tmp_path)


def test_manifest_rejects_a_non_two_week_setup(tmp_path):
    write_manifest(tmp_path, setup={"observed_weeks": 1})
    with pytest.raises(ConfigError):
        load_manifest(tmp_path)


def test_features_follow_the_declared_order_and_values(tmp_path):
    model = make_model(tmp_path)
    matrix = build_features(make_frame(), model.feature_order)
    assert list(matrix.columns) == list(FEATURES)
    assert sorted(matrix["h"].unique()) == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    assert matrix["obs_0"].tolist()[:4] == [0.0, 3.0, 0.0, 1.0]
    assert matrix["obs_sum"].tolist()[:4] == [2.0, 3.0, 4.0, 2.0]
    assert matrix["obs_2"].isna().all() and matrix["obs_3"].isna().all()
    assert matrix["disc_obs"].tolist()[:4] == [0.0, 0.0, 0.0, 0.0]
    assert matrix["season_type"].tolist()[:4] == [0, 1, 0, 1]
    for column in CATEGORICAL_COLUMNS:
        assert isinstance(matrix[column].dtype, pd.CategoricalDtype)


def test_features_raise_on_a_missing_source_column(tmp_path):
    model = make_model(tmp_path)
    frame = make_frame().drop(columns=["price"])
    with pytest.raises(StageError):
        model.build_features(frame)


def test_prediction_shape_and_clipping(tmp_path):
    model = make_model(tmp_path)
    forecasts = model.predict_pairs(make_frame())
    assert forecasts.shape == (4, HORIZON_WEEKS)
    assert forecasts.min() >= 0.0


def test_future_sales_are_never_read(tmp_path):
    model = make_model(tmp_path)
    frame = make_frame()
    baseline = model.predict_pairs(frame)
    mutated = frame.copy()
    for week in range(2, 12):
        mutated[str(week)] = mutated[str(week)].to_numpy(dtype=float) * 7.0 + 123.0
    assert np.array_equal(baseline, model.predict_pairs(mutated))


def test_observed_sales_and_discounted_weeks_are_read(tmp_path):
    model = make_model(tmp_path)
    frame = make_frame()
    baseline = model.predict_pairs(frame)
    observed = frame.copy()
    observed["0"] = observed["0"].to_numpy(dtype=float) + 5.0
    assert not np.array_equal(baseline, model.predict_pairs(observed))
    discounts = frame.copy()
    for week in range(2, 12):
        discounts[f"disc_{week}"] = 0.5
    assert not np.array_equal(baseline, model.predict_pairs(discounts))


def test_feature_contract_mismatch_is_rejected(tmp_path):
    write_manifest(tmp_path)
    manifest = load_manifest(tmp_path)
    with pytest.raises(ConfigError):
        LindaLightGBM(manifest, booster=FakeBooster(features=FEATURES[:-1]))


def test_empty_input_returns_an_empty_matrix(tmp_path):
    model = make_model(tmp_path)
    assert model.predict_pairs(make_frame().iloc[:0]).shape == (0, HORIZON_WEEKS)


def test_describe_states_the_non_claims(tmp_path):
    model = make_model(tmp_path)
    described = model.describe()
    assert described["reads_future_sales"] is False
    assert described["weather_features"] is False
    assert described["image_features"] is False
    assert described["simulation_label"] == "simulation under stated assumptions"