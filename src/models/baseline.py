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


ATTRIBUTE_PREFIXES = ("category_", "season_", "color_", "fabric_")
CATEGORY_PREFIXES = ("category_", "season_")


def _attribute_columns(frame, prefixes=ATTRIBUTE_PREFIXES) -> list:
    return [c for c in frame.columns if str(c).startswith(tuple(prefixes))]


def _active_names(values: np.ndarray, columns: list) -> list:
    rows = np.asarray(values, dtype=float) > 0.5
    return [tuple(columns[i] for i in np.flatnonzero(row)) for row in rows]


def _keys_for(frame, columns) -> list:
    names = list(columns)
    if not names:
        return [()] * len(frame)
    values = frame.reindex(columns=names, fill_value=0.0).to_numpy(dtype=float)
    return _active_names(values, names)


def _accumulate(keys, targets: np.ndarray, levels: np.ndarray) -> dict:
    table = {}
    for key, curve, level in zip(keys, targets, levels):
        entry = table.get(key)
        if entry is None:
            table[key] = [np.asarray(curve, dtype=float), 1, float(level)]
        else:
            entry[0] = entry[0] + np.asarray(curve, dtype=float)
            entry[1] += 1
            entry[2] += float(level)
    return {key: {"curve": value[0] / value[1], "count": value[1],
                  "level": value[2] / value[1]}
            for key, value in table.items()}


class CategoryCurve(Forecaster):
    """B2: mean past-product curve in the same category/season, scaled to observed weeks.

    Fit reads training targets only; predict reads the input window and attributes.
    Group hierarchy: (category, season) -> category -> global.
    """

    name = "category_curve"

    def __init__(self, min_group_size: int = 2, scale_to_observed: bool = True):
        self.min_group_size = max(1, int(min_group_size))
        self.scale_to_observed = bool(scale_to_observed)
        self.out_dim_ = 1
        self.full_ = {}
        self.category_ = {}
        self.global_curve_ = None
        self.global_level_ = 0.0
        self.full_columns_ = ()
        self.category_columns_ = ()

    def fit(self, X, y, **kwargs):
        frame = pd.DataFrame(X)
        targets = _as_2d(y)
        self.out_dim_ = targets.shape[1]
        levels = _lag_matrix(frame).mean(axis=1)
        self.full_columns_ = _attribute_columns(frame, CATEGORY_PREFIXES)
        self.category_columns_ = [c for c in self.full_columns_
                                  if str(c).startswith("category_")]
        self.full_ = _accumulate(_keys_for(frame, self.full_columns_), targets, levels)
        self.category_ = _accumulate(_keys_for(frame, self.category_columns_),
                                     targets, levels)
        self.global_curve_ = targets.mean(axis=0)
        self.global_level_ = float(levels.mean()) if levels.size else 0.0
        return self

    def _lookup(self, full_key, category_key):
        entry = self.full_.get(full_key)
        if entry and entry["count"] >= self.min_group_size:
            return entry["curve"], entry["level"]
        entry = self.category_.get(category_key)
        if entry and entry["count"] >= self.min_group_size:
            return entry["curve"], entry["level"]
        return self.global_curve_, self.global_level_

    def predict(self, X) -> np.ndarray:
        frame = pd.DataFrame(X)
        levels = _lag_matrix(frame).mean(axis=1)
        full_keys = _keys_for(frame, self.full_columns_)
        category_keys = _keys_for(frame, self.category_columns_)
        rows = []
        for level, full_key, category_key in zip(levels, full_keys, category_keys):
            curve, group_level = self._lookup(full_key, category_key)
            factor = 1.0
            if self.scale_to_observed and group_level > 0:
                factor = float(level) / float(group_level)
            rows.append(np.asarray(curve, dtype=float) * factor)
        if not rows:
            return np.zeros((0, self.out_dim_))
        return self._clip(np.vstack(rows))


class AttributeKNN(Forecaster):
    """B3: average curves of the k nearest past products in attribute space.

    Similarity is Hamming distance over the one-hot attribute columns; neighbour
    curves are weighted by signature frequency and scaled to observed weeks.
    """

    name = "attribute_knn"

    def __init__(self, k: int = 5, scale_to_observed: bool = True):
        self.k = max(1, int(k))
        self.scale_to_observed = bool(scale_to_observed)
        self.out_dim_ = 1
        self.attribute_columns_ = ()
        self.signatures_ = None
        self.curves_ = None
        self.levels_ = None
        self.counts_ = None
        self.global_curve_ = None
        self.global_level_ = 0.0

    def fit(self, X, y, **kwargs):
        frame = pd.DataFrame(X)
        targets = _as_2d(y)
        self.out_dim_ = targets.shape[1]
        levels = _lag_matrix(frame).mean(axis=1)
        self.attribute_columns_ = _attribute_columns(frame)
        keys = _keys_for(frame, self.attribute_columns_)
        table = _accumulate(keys, targets, levels)
        names = sorted(table)
        columns = list(self.attribute_columns_)
        self.signatures_ = np.array(
            [[1.0 if column in signature else 0.0 for column in columns]
             for signature in names], dtype=float).reshape(len(names), len(columns))
        self.curves_ = np.vstack([table[name]["curve"] for name in names])
        self.levels_ = np.array([table[name]["level"] for name in names], dtype=float)
        self.counts_ = np.array([table[name]["count"] for name in names], dtype=float)
        self.global_curve_ = targets.mean(axis=0)
        self.global_level_ = float(levels.mean()) if levels.size else 0.0
        return self

    def _distance(self, queried: np.ndarray) -> np.ndarray:
        train = self.signatures_
        product = queried @ train.T
        distance = (queried.sum(axis=1)[:, None] + train.sum(axis=1)[None, :]
                    - 2.0 * product)
        return np.clip(distance, 0.0, None)

    def _neighbour_table(self, signatures) -> dict:
        unique = sorted(set(signatures))
        columns = list(self.attribute_columns_)
        queried = np.array(
            [[1.0 if column in signature else 0.0 for column in columns]
             for signature in unique], dtype=float).reshape(len(unique), len(columns))
        distance = self._distance(queried)
        k = min(self.k, len(self.counts_))
        table = {}
        for position, signature in enumerate(unique):
            nearest = np.argsort(distance[position], kind="stable")[:k]
            weights = self.counts_[nearest]
            total = float(weights.sum())
            curve = (self.curves_[nearest] * weights[:, None]).sum(axis=0) / total
            level = float((self.levels_[nearest] * weights).sum() / total)
            table[signature] = (curve, level)
        return table

    def predict(self, X) -> np.ndarray:
        frame = pd.DataFrame(X)
        levels = _lag_matrix(frame).mean(axis=1)
        if self.global_curve_ is None:
            raise ValueError("attribute_knn is not fitted")
        signatures = _keys_for(frame, self.attribute_columns_)
        table = self._neighbour_table(signatures) if self.counts_ is not None else {}
        rows = []
        for signature, level in zip(signatures, levels):
            curve, neighbour_level = table.get(
                signature, (self.global_curve_, self.global_level_))
            factor = 1.0
            if self.scale_to_observed and neighbour_level > 0:
                factor = float(level) / float(neighbour_level)
            rows.append(np.asarray(curve, dtype=float) * factor)
        if not rows:
            return np.zeros((0, self.out_dim_))
        return self._clip(np.vstack(rows))


BASELINE_CLASSES = {
    SeasonalNaive.name: SeasonalNaive,
    MovingAverage.name: MovingAverage,
    CategoryCurve.name: CategoryCurve,
    AttributeKNN.name: AttributeKNN,
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
        if name == CategoryCurve.name:
            return CategoryCurve()
        if name == AttributeKNN.name:
            return AttributeKNN()
        if name == LagRegression.name:
            return LagRegression()
        raise KeyError(name)