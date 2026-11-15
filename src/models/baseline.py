"""Baselines that read the input window only - never a target column (D1, REP-15)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..seams.protocols import SeamDescription
from .base import Forecaster


def lag_columns(X) -> list:
    frame = pd.DataFrame(X)
    columns = [c for c in frame.columns if str(c).startswith("lag_")]
    return sorted(columns, key=lambda c: int(str(c).split("_")[-1]))


def _lag_matrix(X) -> np.ndarray:
    frame = pd.DataFrame(X)
    columns = lag_columns(frame)
    if not columns:
        raise ValueError("feature matrix carries no lag_ columns")
    return frame[columns].to_numpy(dtype=float)


def _as_2d(y) -> np.ndarray:
    array = np.asarray(y, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    return array


class SeasonalNaive(Forecaster):
    """Repeats the selected observed input week; the target window is never read."""

    name = "seasonal_naive"

    def __init__(self, reference: str = "last_input"):
        self.reference = reference
        self.reference_values_ = None
        self.out_dim_ = 1

    def fit(self, X, y, **kwargs):
        self.out_dim_ = _as_2d(y).shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        lags = _lag_matrix(X)
        if self.reference == "mean_input":
            reference = lags.mean(axis=1)
        else:
            reference = lags[:, -1]
        return self._clip(np.repeat(reference[:, None], self.out_dim_, axis=1))


class MovingAverage(Forecaster):
    """Averages the observed input weeks only."""

    name = "moving_average"

    def fit(self, X, y, **kwargs):
        self.out_dim_ = _as_2d(y).shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        reference = _lag_matrix(X).mean(axis=1)
        return self._clip(np.repeat(reference[:, None], self.out_dim_, axis=1))


class LagRegression(Forecaster):
    """Per-horizon ridge regression on input features; interpretable baseline."""

    name = "lag_regression"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.columns_ = None
        self.models_ = []

    def fit(self, X, y, **kwargs):
        from sklearn.linear_model import Ridge

        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        targets = _as_2d(y)
        values = frame.to_numpy(dtype=float)
        self.models_ = []
        for horizon in range(targets.shape[1]):
            model = Ridge(alpha=self.alpha)
            model.fit(values, targets[:, horizon])
            self.models_.append(model)
        return self

    def predict(self, X) -> np.ndarray:
        frame = pd.DataFrame(X).reindex(columns=self.columns_, fill_value=0.0)
        predictions = [model.predict(frame.to_numpy(dtype=float))
                       for model in self.models_]
        return self._clip(np.column_stack(predictions))


BASELINE_CLASSES = {
    SeasonalNaive.name: SeasonalNaive,
    MovingAverage.name: MovingAverage,
    LagRegression.name: LagRegression,
}


class BaselineDefinition:
    """Declared reference week for the naive family (REP-15)."""

    SEAM_ID = "SEAM-BASELINE-DEF"

    def __init__(self, reference="last_input"):
        self.reference = getattr(reference, "value", reference)

    def describe(self) -> SeamDescription:
        return SeamDescription("baseline_def", "REP-15", self.reference)

    def make(self, name: str) -> Forecaster:
        if name == SeasonalNaive.name:
            return SeasonalNaive(self.reference)
        if name == MovingAverage.name:
            return MovingAverage()
        if name == LagRegression.name:
            return LagRegression()
        raise KeyError(name)