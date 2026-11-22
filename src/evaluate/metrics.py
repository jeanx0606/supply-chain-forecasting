"""Error metrics with explicit undefined-denominator handling."""
from __future__ import annotations

import numpy as np


def _flat(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def wape(y_true, y_pred) -> float | None:
    actual = _flat(y_true)
    predicted = _flat(y_pred)
    denominator = float(np.abs(actual).sum())
    if denominator == 0 or actual.size == 0:
        return None
    return float(np.abs(actual - predicted).sum() / denominator)


def mae(y_true, y_pred) -> float | None:
    actual = _flat(y_true)
    predicted = _flat(y_pred)
    if actual.size == 0:
        return None
    return float(np.abs(actual - predicted).mean())


def rmse(y_true, y_pred) -> float | None:
    actual = _flat(y_true)
    predicted = _flat(y_pred)
    if actual.size == 0:
        return None
    return float(np.sqrt(((actual - predicted) ** 2).mean()))


def bias(y_true, y_pred) -> float | None:
    """Mean signed error (forecast - actual), in raw units; positive = over-forecast."""
    actual = _flat(y_true)
    predicted = _flat(y_pred)
    if actual.size == 0:
        return None
    return float((predicted - actual).mean())


def format_metric(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"