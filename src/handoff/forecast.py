"""Call the trained handoff model: one LightGBM, 2 observed weeks, 10 forecasts.

The handoff contract is fixed: one model trained on SS17-SS19 predicts weeks 3-12
(file columns 2..11) for every item-store pair from the two observed weeks only.
Future *sales* are never read; future *discounts* are read because promotions are
planned and known at the forecast origin. This module reimplements the feature
contract in repository code so the prototype does not import a teammate's script;
``src.handoff.verify`` asserts parity against the reference predictor whenever it
is supplied.

The safety-stock constant ``sigma_e`` ships with the manifest as a fixed number per
model. It is the standard deviation of the validation-season forecast error of the
*validation-trained* model, so it must not be recomputed from the final model,
which is retrained on train+validation (see the verification report).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import ConfigError, StageError

OBSERVED_WEEKS = 2
N_WEEKS = 12
HORIZON_WEEKS = N_WEEKS - OBSERVED_WEEKS
OBSERVED_SALES_COLUMNS = ("0", "1")
CATEGORICAL_COLUMNS = ("category_id", "color_id", "fabric_id")
MODEL_FILENAME = "lightgbm_final_model.txt"
PARAMS_FILENAME = "prototype_params.json"
SIMULATION_LABEL = "simulation under stated assumptions"
SIGMA_KEYS = ("B0", "B1", "B2", "B3", "LightGBM")

DERIVED_COLUMNS = ("h", "obs_0", "obs_1", "obs_2", "obs_3", "obs_sum",
                   "obs_last", "disc_h", "disc_obs")
DISCOUNT_COLUMNS = tuple(f"disc_{week}" for week in range(N_WEEKS))


@dataclass(frozen=True)
class HandoffManifest:
    """The declarations the handoff ships with its model file."""

    handoff_dir: Path
    model_file: str
    objective: str
    feature_order: tuple
    categorical_columns: tuple
    sigma_e: dict
    main_setting: dict
    z_values: dict

    @property
    def model_path(self) -> Path:
        return Path(self.handoff_dir) / self.model_file

    @property
    def params_path(self) -> Path:
        return Path(self.handoff_dir) / PARAMS_FILENAME

    def sigma(self, model: str) -> float:
        if model not in self.sigma_e:
            raise ConfigError(
                "handoff.sigma_e", "model has no validation sigma_e",
                observed=model, expected=list(SIGMA_KEYS),
                next_command="python -m codes.verify_handoff")
        return float(self.sigma_e[model])

    def summary(self) -> dict:
        return {
            "handoff_dir": str(self.handoff_dir),
            "model_file": self.model_file,
            "objective": self.objective,
            "n_features": len(self.feature_order),
            "categorical_columns": list(self.categorical_columns),
            "sigma_e": dict(self.sigma_e),
            "main_setting": dict(self.main_setting),
            "z_values": dict(self.z_values),
        }


def load_manifest(handoff_dir) -> HandoffManifest:
    """Read and validate ``prototype_params.json``."""
    directory = Path(handoff_dir)
    params_path = directory / PARAMS_FILENAME
    if not params_path.exists():
        raise ConfigError(
            "handoff.params", "parameter manifest is missing",
            observed=str(params_path), expected=PARAMS_FILENAME,
            next_command="python -m codes.verify_handoff")
    with open(params_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    feature_order = raw.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order:
        raise ConfigError("handoff.feature_order", "must be a non-empty list",
                          observed=feature_order)
    categorical = raw.get("categorical_features") or []
    if list(categorical) != list(CATEGORICAL_COLUMNS):
        raise ConfigError(
            "handoff.categorical_features",
            "the adapter encodes exactly the three integer attribute columns",
            observed=categorical, expected=list(CATEGORICAL_COLUMNS))

    setup = raw.get("setup") or {}
    observed_weeks = setup.get("observed_weeks", OBSERVED_WEEKS)
    if int(observed_weeks) != OBSERVED_WEEKS:
        raise ConfigError(
            "handoff.setup.observed_weeks",
            "the adapter implements the 2-in/10-out task only",
            observed=observed_weeks, expected=OBSERVED_WEEKS)

    sigma = raw.get("sigma_e_validation_SS19") or {}
    missing = [key for key in SIGMA_KEYS if key not in sigma]
    if missing:
        raise ConfigError("handoff.sigma_e_validation_SS19",
                          "missing per-model validation sigma_e",
                          observed=missing, expected=list(SIGMA_KEYS))
    for key in SIGMA_KEYS:
        value = float(sigma[key])
        if value < 0:
            raise ConfigError("handoff.sigma_e_validation_SS19",
                              "sigma_e must be non-negative", observed=value)

    return HandoffManifest(
        handoff_dir=directory,
        model_file=str(raw.get("model_file") or MODEL_FILENAME),
        objective=str(raw.get("objective") or "poisson"),
        feature_order=tuple(str(name) for name in feature_order),
        categorical_columns=tuple(str(name) for name in categorical),
        sigma_e={key: float(sigma[key]) for key in SIGMA_KEYS},
        main_setting=dict(raw.get("replenishment_main") or {}),
        z_values={str(level): float(value)
                  for level, value in (raw.get("z_values") or {}).items()},
    )


def _season_type(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=int)
    text = series.astype(str).str.upper()
    return text.str.startswith("A").to_numpy(dtype=int)


def build_features(frame: pd.DataFrame, feature_names=None,
                   categorical_categories=None) -> pd.DataFrame:
    """One row per (pair, forecast week) built from the two observed weeks.

    ``categorical_categories`` is the mapping stored inside the model file. Passing
    it keeps the category codes identical to training even when the caller scores a
    subset that lacks some attribute values.
    """
    frame = pd.DataFrame(frame).reset_index(drop=True)
    names = tuple(str(name) for name in (feature_names or ()))
    if not names:
        raise StageError("handoff", "no feature order supplied",
                         next_command="python -m codes.verify_handoff")

    needed = [column for column in names if column not in DERIVED_COLUMNS]
    needed += list(OBSERVED_SALES_COLUMNS)
    needed += ["disc_0", "disc_1"] + [f"disc_{week}"
                                      for week in range(OBSERVED_WEEKS, N_WEEKS)]
    missing = sorted({column for column in needed
                      if column not in frame.columns})
    if missing:
        raise StageError(
            "handoff",
            f"input is missing columns required by the model: {missing}",
            next_command="python -m codes.verify_handoff")

    static = [column for column in names if column not in DERIVED_COLUMNS]
    base = frame[static].copy()
    base["season_type"] = _season_type(frame["season_type"])
    base["obs_0"] = np.clip(frame["0"].to_numpy(dtype=float), 0, None)
    base["obs_1"] = np.clip(frame["1"].to_numpy(dtype=float), 0, None)
    base["obs_2"] = np.nan
    base["obs_3"] = np.nan
    base["obs_sum"] = base["obs_0"] + base["obs_1"]
    base["obs_last"] = base["obs_1"]
    base["disc_obs"] = frame[["disc_0", "disc_1"]].to_numpy(dtype=float).mean(axis=1)

    blocks = []
    for horizon in range(OBSERVED_WEEKS, N_WEEKS):
        block = base.copy()
        block["h"] = float(horizon)
        block["disc_h"] = frame[f"disc_{horizon}"].to_numpy(dtype=float)
        blocks.append(block)
    matrix = pd.concat(blocks, ignore_index=True)
    matrix = matrix[list(names)].copy()

    for index, column in enumerate(CATEGORICAL_COLUMNS):
        if column not in matrix.columns:
            continue
        categories = None
        if categorical_categories is not None and index < len(categorical_categories):
            categories = list(categorical_categories[index])
        matrix[column] = _as_categorical(matrix[column], categories)
    return matrix


def _as_categorical(values: pd.Series, categories=None):
    """Encode an integer attribute column the way the model file expects.

    When the trained category list covers every observed value, encode against it so
    the codes cannot drift. When a code outside the trained list appears, fall back to
    the reference predictor's mapping; measurement on the shipped data shows the two
    paths produce identical forecasts, including for the one unseen fabric code.
    """
    if categories and values.dropna().isin(categories).all():
        return pd.Categorical(values, categories=list(categories))
    return pd.Categorical(values)


class LindaLightGBM:
    """Pre-trained handoff model as a callable forecaster (weeks 3-12, clipped at 0)."""

    name = "linda_lightgbm"
    requires_backend = False
    pre_trained = True
    observed_weeks = OBSERVED_WEEKS
    horizon_weeks = HORIZON_WEEKS

    def __init__(self, manifest: HandoffManifest, booster=None):
        self.manifest = manifest
        self.booster = booster if booster is not None else self._load()
        self._verify_feature_contract()

    def _load(self):
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - lightgbm is declared
            raise StageError("handoff", "backend 'lightgbm' is not installed",
                             next_command="pip install lightgbm") from exc
        path = self.manifest.model_path
        if not path.exists():
            raise ConfigError("handoff.model", "model file is missing",
                              observed=str(path), expected=MODEL_FILENAME,
                              next_command="python -m codes.verify_handoff")
        # Load from the text body, not from the path: the LightGBM C API opens a
        # narrow path and fails on non-ASCII directories on Windows.
        return lgb.Booster(model_str=path.read_text(encoding="utf-8"))

    def _verify_feature_contract(self):
        produced = tuple(self.booster.feature_name())
        expected = tuple(self.manifest.feature_order)
        if produced != expected:
            raise ConfigError(
                "handoff.feature_order",
                "model feature list differs from the manifest",
                observed=list(produced), expected=list(expected),
                next_command="python -m codes.verify_handoff")

    @property
    def feature_order(self) -> tuple:
        return tuple(self.manifest.feature_order)

    @property
    def categorical_categories(self):
        return getattr(self.booster, "pandas_categorical", None)

    @property
    def sigma_e(self) -> float:
        return self.manifest.sigma("LightGBM")

    def build_features(self, frame) -> pd.DataFrame:
        return build_features(frame, self.feature_order, self.categorical_categories)

    def predict_pairs(self, frame) -> np.ndarray:
        """(n_pairs, 10) forecasts for weeks 3..12, clipped at 0."""
        frame = pd.DataFrame(frame)
        if len(frame) == 0:
            return np.zeros((0, HORIZON_WEEKS), dtype=float)
        matrix = self.build_features(frame)
        raw = np.asarray(self.booster.predict(matrix), dtype=float)
        if raw.size != HORIZON_WEEKS * len(frame):
            raise StageError(
                "handoff",
                f"model returned {raw.size} values for {len(frame)} pairs",
                next_command="python -m codes.verify_handoff")
        return np.clip(raw.reshape(HORIZON_WEEKS, len(frame)).T, 0.0, None)

    def describe(self) -> dict:
        return {
            "model": self.name,
            "objective": self.manifest.objective,
            "observed_weeks": OBSERVED_WEEKS,
            "horizon_weeks": HORIZON_WEEKS,
            "forecast_weeks": "3-12 (file columns 2..11)",
            "reads_future_sales": False,
            "uses_future_discounts": True,
            "weather_features": False,
            "image_features": False,
            "sigma_e_lightgbm": self.sigma_e,
            "sigma_e_source": "validation season SS19, validation-trained model",
            "simulation_label": SIMULATION_LABEL,
        }